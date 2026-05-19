# RAG — Teknik Standartlar Belgesi Sorgulama Sistemi

Tam offline çalışan, kurumsal düzeyde RAG (Retrieval-Augmented Generation) pipeline'ı.
Mühendislik standartları ve teknik PDF belgelerini doğal dille sorgulamak için tasarlanmıştır.

---

## Mimari

```
pdfs/
 └── *.pdf
      │
      ▼ core/ingestion.py
 Section-aware chunking
 (bölüm başlıklarına göre kes, overlap ekle)
      │
      ▼ core/retrieval.py
 BGE-M3 Embedding → FAISS IndexFlatIP
 + BM25Okapi (lexical)
 → Hybrid skor: α·semantic + (1-α)·BM25
      │
      ▼ core/llm.py
 Ollama (Qwen2.5:14b) — öncelikli
 llama.cpp GGUF        — fallback
      │
      ▼
 Cevap + [S#] citation etiketleri + kaynak tablosu
```

---

## Kurulum

### 1. Python bağımlılıkları

```bash
pip install -r requirements.txt
```

### 2. Embedding modeli (offline indirme)

```python
# İNTERNET OLAN makinede bir kez çalıştır:
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-m3").save("models/bge-m3")
```

Alternatif (daha hafif):
```python
SentenceTransformer("intfloat/multilingual-e5-large").save("models/multilingual-e5-large")
```

### 3. LLM — Ollama (önerilen)

```bash
# https://ollama.com/download
ollama pull qwen2.5:14b        # ~9 GB, en iyi kalite
# veya
ollama pull qwen2.5:7b         # ~5 GB, daha hızlı
```

Ortam değişkeniyle model değiştir:
```bash
OLLAMA_MODEL=qwen2.5:7b python cli.py ui
```

### 4. LLM — llama.cpp GGUF (Ollama yoksa)

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
    filename="qwen2.5-14b-instruct-q4_k_m.gguf",
    local_dir="models/llm",
    local_dir_use_symlinks=False,
)
```

---

## Kullanım

### Streamlit UI

```bash
python cli.py ui
# → http://localhost:8501
```

### Komut satırı

```bash
# PDF'leri indeksle
python cli.py index --docs ./pdfs

# Soru sor
python cli.py ask "SAE J3016 sürüş otomasyonu seviyeleri nelerdir?"

# Standart filtresiyle soru
python cli.py ask "Tolerans gereksinimleri nedir?" --filter SAE_J3016 --k 8
```

---

## Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `DOCS_DIR` | `./pdfs` | PDF klasörü |
| `INDEX_DIR` | `./index` | FAISS index dizini |
| `EMB_MODEL` | `./models/bge-m3` | Embedding model klasörü |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama sunucu adresi |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Ollama model adı |
| `LLM_GGUF` | `./models/llm/...gguf` | Fallback GGUF yolu |
| `CHUNK_SIZE` | `1400` | Chunk boyutu (karakter) |
| `CHUNK_OVERLAP` | `200` | Overlap (karakter) |
| `TOP_K` | `6` | Retrieval'dan dönen parça sayısı |
| `SCORE_THRESHOLD` | `0.40` | Minimum hybrid skor eşiği |
| `BM25_ALPHA` | `0.60` | Semantic ağırlığı (1.0=tam semantic) |
| `LLM_TEMPERATURE` | `0.05` | LLM temperature (düşük=az hallüsinasyon) |

---

## Proje Yapısı

```
rag_enterprise/
├── cli.py                  # Komut satırı giriş noktası
├── requirements.txt
├── core/
│   ├── config.py           # Merkezi konfigürasyon
│   ├── ingestion.py        # PDF okuma + section-aware chunking
│   ├── retrieval.py        # Hybrid FAISS + BM25
│   ├── llm.py              # Ollama / llama.cpp LLM katmanı
│   ├── indexer.py          # İndeks oluşturma orkestratörü
│   └── pipeline.py         # ask() — tek giriş noktası
├── ui/
│   └── app.py              # Streamlit arayüzü
├── pdfs/                   # PDF'lerinizi buraya koyun
├── index/                  # Otomatik oluşturulur
└── models/
    ├── bge-m3/             # Embedding modeli
    └── llm/                # GGUF model (Ollama kullanıyorsan gerekmez)
```

---

## Geliştirme Yol Haritası

- [ ] ChromaDB / Milvus ile incremental index güncelleme (yeni PDF eklenince tam yeniden indeksleme yerine)
- [ ] Metadata filtresi UI'ye standart listesi dropdown'u olarak ekle
- [ ] Çok-dönüşlü sohbet geçmişi
- [ ] Qwen3 thinking mode entegrasyonu (karmaşık teknik muhakeme için)
- [ ] Tablo ve şekil extraction (tabula, pdfplumber)
- [ ] Evaluation pipeline (RAGAS ile retrieval + cevap kalitesi ölçümü)
