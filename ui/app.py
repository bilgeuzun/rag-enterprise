# ui/app.py — Streamlit arayüzü
from __future__ import annotations
import sys
import threading
from pathlib import Path

# Proje kökünü sys.path'e ekle (python ui/app.py veya streamlit run ui/app.py)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core.config   import DOCS_DIR, INDEX_DIR, OLLAMA_MODEL, EMB_MODEL
from core.indexer  import build_full_index
from core.pipeline import ask
from core.retrieval import reset_index_cache

# ─── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG | Teknik Standartlar",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
code, .stCodeBlock, .mono {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * {
    color: #c8cdd8 !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280 !important;
}

/* Başlık */
.rag-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}
.rag-title {
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #f0f4ff;
}
.rag-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    background: #1a2035;
    color: #4f8ef7;
    border: 1px solid #2a3a5c;
    border-radius: 4px;
    padding: 2px 8px;
}
.rag-sub {
    color: #6b7280;
    font-size: 0.88rem;
    margin-bottom: 1.5rem;
}

/* Cevap kutusu */
.answer-box {
    background: #0f1117;
    border: 1px solid #1e2130;
    border-left: 3px solid #4f8ef7;
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-top: 0.5rem;
    line-height: 1.7;
    color: #dde3f0;
}

/* Hit kartları */
.hit-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
}
.hit-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #4f8ef7;
    margin-bottom: 0.25rem;
}
.hit-score {
    color: #6b7280;
    font-size: 0.72rem;
}
.hit-snippet {
    color: #9ca3af;
    line-height: 1.55;
}

/* Durum pill */
.status-ok   { color: #34d399; font-size: 0.8rem; }
.status-warn { color: #fbbf24; font-size: 0.8rem; }
.status-err  { color: #f87171; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─── Yardımcılar ─────────────────────────────────────────────────────────────

_INDEX_LOCK = threading.Lock()

def _index_exists() -> bool:
    return (INDEX_DIR / "faiss.index").exists() and (INDEX_DIR / "corpus.json").exists()

def _corpus_size() -> int:
    import json
    path = INDEX_DIR / "corpus.json"
    if not path.exists():
        return 0
    try:
        with path.open() as f:
            return len(json.load(f))
    except Exception:
        return 0

def _pdf_count(docs_dir: Path) -> int:
    return len(list(docs_dir.glob("*.pdf"))) if docs_dir.exists() else 0


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")

    # Dizin seçimi
    docs_dir_str = st.text_input(
        "PDF Klasörü",
        value=str(DOCS_DIR),
        placeholder="/path/to/pdfs",
    )
    docs_dir = Path(docs_dir_str)

    # Retrieval
    st.markdown("---")
    st.markdown("**Retrieval**")
    top_k = st.slider("Top-K (kaç parça)", 1, 12, 6)
    alpha = st.slider("Semantic ağırlığı (α)", 0.0, 1.0, 0.60, 0.05,
                      help="1.0 = tam semantic | 0.0 = tam BM25")
    threshold = st.slider("Skor eşiği", 0.0, 1.0, 0.40, 0.05)

    # Standart filtresi
    st.markdown("**Filtre** _(isteğe bağlı)_")
    standard_filter = st.text_input(
        "Standart adı",
        placeholder="SAE_J3016",
        help="Boş bırakırsan tüm belgeler aranır.",
    )

    # LLM
    st.markdown("---")
    st.markdown("**LLM**")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.05, 0.01)

    # Durum
    st.markdown("---")
    st.markdown("**Sistem Durumu**")
    pdf_n   = _pdf_count(docs_dir)
    chunk_n = _corpus_size()
    idx_ok  = _index_exists()

    st.markdown(
        f"<span class='status-{'ok' if pdf_n else 'warn'}'>📄 {pdf_n} PDF</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span class='status-{'ok' if idx_ok else 'warn'}'>🗄 {chunk_n} chunk indekslendi</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span class='status-ok'>🤖 Model: `{Path(EMB_MODEL).name}`</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span class='status-ok'>💬 LLM: `{OLLAMA_MODEL}` (Ollama)</span>",
        unsafe_allow_html=True,
    )

    # İndeks oluştur
    st.markdown("---")
    if st.button("📚 İndeksi Oluştur / Yenile", use_container_width=True):
        with st.spinner("İndeksleniyor…"):
            result = build_full_index(docs_dir)
        if result["ok"]:
            st.success(result["message"])
            reset_index_cache()
            st.rerun()
        else:
            st.error(result["message"])


# ─── Ana alan ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="rag-header">
  <span class="rag-title">📐 Teknik Standartlar RAG</span>
  <span class="rag-badge">OFFLINE / LOCAL</span>
</div>
<p class="rag-sub">
  Mühendislik standartlarını ve teknik PDF belgelerinizi doğal dille sorgulayın.
  Cevaplar yalnızca belgelerinizden üretilir.
</p>
""", unsafe_allow_html=True)

# Soru girişi
col_q, col_btn = st.columns([6, 1])
with col_q:
    question = st.text_input(
        "Sorunuz",
        placeholder="Örn: SAE J3016'ya göre sürüş otomasyonu seviyeleri nelerdir?",
        label_visibility="collapsed",
    )
with col_btn:
    ask_btn = st.button("🔍 Sor", use_container_width=True, type="primary")

# ─── Sorgu ───────────────────────────────────────────────────────────────────

if ask_btn and question.strip():
    if not _index_exists():
        st.warning("Önce sol panelden **İndeksi Oluştur** butonuna basın.")
    else:
        with st.spinner("İlgili parçalar aranıyor ve yanıt oluşturuluyor…"):
            filters = {"standard": standard_filter} if standard_filter.strip() else None
            result  = ask(
                question=question,
                k=top_k,
                alpha=alpha,
                score_threshold=threshold,
                filters=filters,
            )

        # Cevap
        st.markdown("#### 💬 Yanıt")
        st.markdown(
            f"<div class='answer-box'>{result['answer']}</div>",
            unsafe_allow_html=True,
        )

        # Kaynaklar + hit detayları
        if result["hits"]:
            st.markdown("---")
            tab_src, tab_hits = st.tabs(["📚 Kaynaklar", "🔎 Retrieval Detayı"])

            with tab_src:
                st.markdown(result["sources"])

            with tab_hits:
                st.caption(f"{result['hit_count']} parça bulundu (α={alpha}, eşik={threshold})")
                for i, h in enumerate(result["hits"], 1):
                    meta    = h["meta"]
                    snippet = h["text"].replace("\n", " ")[:300]
                    score   = h.get("score", 0.0)
                    st.markdown(f"""
<div class="hit-card">
  <div class="hit-meta">[S{i}] {meta['file']} — s.{meta['page']}
    {' — ' + meta['section'][:60] if meta.get('section') else ''}
  </div>
  <div class="hit-score">skor: {score:.4f}</div>
  <div class="hit-snippet">{snippet}…</div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("Eşik üzerinde ilgili parça bulunamadı. Eşiği düşürmeyi veya soruyu değiştirmeyi deneyin.")

elif ask_btn:
    st.warning("Lütfen bir soru yazın.")

# ─── Hoşgeldin ekranı ────────────────────────────────────────────────────────
if not question:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🗂 Nasıl kullanılır?**")
        st.caption("1. Sol panelden PDF klasörünü seçin.\n2. 'İndeksi Oluştur' butonuna basın.\n3. Sorunuzu yazıp 🔍 Sor'a basın.")
    with c2:
        st.markdown("**⚡ Retrieval**")
        st.caption("Hybrid arama: Semantic (FAISS) + Lexical (BM25). α kaydırıcısı ikisi arasındaki dengeyi ayarlar.")
    with c3:
        st.markdown("**🔒 Tam offline**")
        st.caption("Modeller lokalde çalışır. İnternet bağlantısı gerekmez. Kurumsal ağda güvenle kullanılabilir.")
