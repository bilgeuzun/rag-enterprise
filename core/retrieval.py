# core/retrieval.py — Hybrid retrieval: Semantic (FAISS) + Lexical (BM25)
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import faiss

from core.config import (
    INDEX_DIR, EMB_MODEL,
    TOP_K, SCORE_THRESHOLD, BM25_ALPHA,
)

# Lazy imports (büyük kütüphaneler; sadece gerektiğinde yükle)
_EMB_MODEL = None
_BM25      = None
_CORPUS: list[dict[str, Any]] = []
_FAISS_INDEX: faiss.Index | None = None


# ── Embedding modeli ──────────────────────────────────────────────────────────

def get_embedding_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        os.environ.setdefault("HF_HUB_OFFLINE",      "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _EMB_MODEL = SentenceTransformer(EMB_MODEL, device="cpu")
    return _EMB_MODEL


def embed(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


# ── İndeks kaydet / yükle ─────────────────────────────────────────────────────

def save_index(corpus: list[dict[str, Any]], embeddings: np.ndarray) -> None:
    dim = embeddings.shape[1]
    
    # FAISS: cosine ~ IP üzerinde normalize edilmiş vektörler
    # Büyük corpus için IVF_PQ kullan; < 50k chunk için Flat yeterli
    if len(corpus) > 50_000:
        nlist = min(256, len(corpus) // 40)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, 16, 8)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with (INDEX_DIR / "corpus.json").open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"[OK] {len(corpus)} chunk indekslendi → {INDEX_DIR}")


def load_index() -> tuple[faiss.Index, list[dict[str, Any]]]:
    global _FAISS_INDEX, _CORPUS
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX, _CORPUS

    idx_path  = INDEX_DIR / "faiss.index"
    corp_path = INDEX_DIR / "corpus.json"

    if not (idx_path.exists() and corp_path.exists()):
        raise FileNotFoundError(
            "İndeks bulunamadı. Önce 'İndeksi Oluştur' butonuna basın."
        )

    _FAISS_INDEX = faiss.read_index(str(idx_path))
    with corp_path.open("r", encoding="utf-8") as f:
        _CORPUS = json.load(f)

    return _FAISS_INDEX, _CORPUS


def reset_index_cache() -> None:
    """Yeni indeks oluşturulunca bellek içi cache'i temizle."""
    global _FAISS_INDEX, _CORPUS, _BM25
    _FAISS_INDEX = None
    _CORPUS      = []
    _BM25        = None


# ── BM25 ─────────────────────────────────────────────────────────────────────

def get_bm25(corpus: list[dict[str, Any]]):
    global _BM25
    if _BM25 is None:
        from rank_bm25 import BM25Okapi
        tokenized = [item["text"].lower().split() for item in corpus]
        _BM25 = BM25Okapi(tokenized)
    return _BM25


# ── Hybrid retrieve ───────────────────────────────────────────────────────────

def retrieve(
    query: str,
    k: int = TOP_K,
    alpha: float = BM25_ALPHA,
    score_threshold: float = SCORE_THRESHOLD,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: alpha * semantic + (1-alpha) * BM25

    filters: ör. {"standard": "SAE_J3016"} — metadata üzerinde AND filtresi
    """
    index, corpus = load_index()

    # ── Metadata filtresi ──────────────────────────────────────────────────────
    if filters:
        allowed_idx = {
            i for i, item in enumerate(corpus)
            if all(item["meta"].get(k) == v for k, v in filters.items())
        }
        if not allowed_idx:
            return []
    else:
        allowed_idx = None

    fetch_k = min(k * 5, len(corpus))  # daha geniş havuz, sonra birleştir

    # ── Semantic skor ──────────────────────────────────────────────────────────
    qvec = embed([query])
    D, I = index.search(qvec, fetch_k)
    sem: dict[int, float] = {}
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        if allowed_idx is not None and idx not in allowed_idx:
            continue
        sem[int(idx)] = float(score)

    # ── BM25 skor ─────────────────────────────────────────────────────────────
    bm25  = get_bm25(corpus)
    raw   = bm25.get_scores(query.lower().split())
    mn, mx = raw.min(), raw.max()
    rng   = mx - mn if mx != mn else 1.0
    bm25_norm: dict[int, float] = {
        i: float((raw[i] - mn) / rng)
        for i in range(len(corpus))
        if (allowed_idx is None or i in allowed_idx)
    }

    # ── Skor birleştirme ───────────────────────────────────────────────────────
    combined: dict[int, float] = {}
    all_idx = set(sem) | set(bm25_norm)
    for i in all_idx:
        s = alpha * sem.get(i, 0.0) + (1 - alpha) * bm25_norm.get(i, 0.0)
        combined[i] = s

    # Sıralama + eşik
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # Aynı (dosya, sayfa) tekrarlarını eleme — farklı chunk varsa en iyi skoru al
    seen: set[tuple[str, int]] = set()
    hits: list[dict[str, Any]] = []

    for idx, score in ranked:
        if score < score_threshold:
            break
        item = corpus[idx]
        key  = (item["meta"]["file"], item["meta"]["page"])
        if key in seen:
            continue
        seen.add(key)
        hits.append({**item, "score": score})
        if len(hits) >= k:
            break

    return hits
