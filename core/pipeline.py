# core/pipeline.py — Tek giriş noktası: soru → cevap
from __future__ import annotations
from typing import Any

from core.retrieval import retrieve
from core.llm       import generate_answer
from core.config    import TOP_K, SCORE_THRESHOLD, BM25_ALPHA


def ask(
    question: str,
    k: int               = TOP_K,
    alpha: float         = BM25_ALPHA,
    score_threshold: float = SCORE_THRESHOLD,
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Tek fonksiyon: soru al, (hits → llm) çalıştır, sonuç döndür.

    Döndürür:
    {
      "answer":   str,          # LLM cevabı [S#] etiketleriyle
      "sources":  str,          # markdown kaynak listesi
      "hits":     list[dict],   # ham retrieval sonuçları
      "hit_count": int,
    }
    """
    question = question.strip()
    if not question:
        return {
            "answer": "Lütfen bir soru yazın.",
            "sources": "",
            "hits": [],
            "hit_count": 0,
        }

    hits = retrieve(
        query=question,
        k=k,
        alpha=alpha,
        score_threshold=score_threshold,
        filters=filters,
    )

    answer, sources_md = generate_answer(question, hits)

    return {
        "answer":    answer,
        "sources":   sources_md,
        "hits":      hits,
        "hit_count": len(hits),
    }
