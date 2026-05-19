# core/indexer.py — İndeks oluşturma orkestratörü
from __future__ import annotations
from pathlib import Path
from typing import Any

from core.ingestion  import build_corpus
from core.retrieval  import embed, save_index, reset_index_cache
from core.config     import DOCS_DIR


def build_full_index(
    docs_dir: Path = DOCS_DIR,
    progress_cb=None,          # isteğe bağlı ilerleme callback'i: fn(float, str)
) -> dict[str, Any]:
    """
    Tam pipeline: PDF → corpus → embedding → FAISS + BM25 hazır.
    Döndürür: {"ok": bool, "chunks": int, "pdf_count": int, "message": str}
    """
    def _progress(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(frac, msg)
        else:
            print(f"[{frac*100:.0f}%] {msg}")

    try:
        pdfs = list(docs_dir.glob("*.pdf"))
        if not pdfs:
            return {"ok": False, "chunks": 0, "pdf_count": 0,
                    "message": f"PDF bulunamadı: {docs_dir}"}

        _progress(0.05, f"{len(pdfs)} PDF okunuyor ve parçalanıyor…")
        corpus = build_corpus(docs_dir)

        if not corpus:
            return {"ok": False, "chunks": 0, "pdf_count": len(pdfs),
                    "message": "Metinler çıkarılamadı (taralı PDF mi?)."}

        _progress(0.35, f"{len(corpus)} parça embed ediliyor…")
        embs = embed([item["text"] for item in corpus])

        _progress(0.90, "FAISS index yazılıyor…")
        save_index(corpus, embs)

        reset_index_cache()          # bellekteki eski index'i temizle

        _progress(1.00, "Tamam!")
        return {
            "ok":        True,
            "chunks":    len(corpus),
            "pdf_count": len(pdfs),
            "message":   f"{len(corpus)} parça, {len(pdfs)} PDF başarıyla indekslendi.",
        }

    except Exception as exc:
        return {"ok": False, "chunks": 0, "pdf_count": 0, "message": str(exc)}
