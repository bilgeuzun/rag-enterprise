# core/ingestion.py — PDF okuma, temizleme, section-aware chunking
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from core.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Metin temizleme ────────────────────────────────────────────────────────────

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ").replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    lines = [ln.strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


# ── Sayfa metni (OCR fallback) ────────────────────────────────────────────────

def extract_page_text(doc: fitz.Document, page_no: int, ocr_lang: str = "tur+eng") -> str:
    page = doc.load_page(page_no)
    txt = clean_text(page.get_text("text"))

    # Yeterli metin varsa OCR'a gerek yok
    if len(txt) >= 60:
        return txt

    if not OCR_AVAILABLE:
        return txt

    try:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_txt = pytesseract.image_to_string(img, lang=ocr_lang)
        return clean_text(ocr_txt) or txt
    except Exception:
        return txt


# ── Section-aware chunking ────────────────────────────────────────────────────
#
# Teknik standartlarda (SAE, ISO, IEC, DIN…) genellikle
# numaralı bölüm başlıkları olur: "3.2 Requirements", "4.1.1 …"
# Önce semantik sınırlarda kes; büyük bölümleri overlap ile parçala.

_SECTION_RE = re.compile(
    r"(?m)(?=^\s*"
    r"(?:\d+(?:\.\d+)*\s+[A-ZÇĞİÖŞÜa-zçğışöüA-Z]"   # "3.2 Requirements"
    r"|\b(?:SCOPE|PURPOSE|DEFINITIONS|REQUIREMENTS"
    r"|APPENDIX|INTRODUCTION|REFERENCES|ABSTRACT"
    r"|KAPSAM|TANIMLAR|GEREKSINIMLER|EKLER)\b"         # Türkçe başlıklar
    r"))"
)


def _section_title(text: str) -> str:
    """İlk satırı bölüm başlığı olarak döndür (metadata için)."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:120]


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, str]]:
    """
    Her chunk: {"text": ..., "section": ...}
    """
    text = clean_text(text)
    if not text:
        return []

    # Bölüm sınırlarına göre böl
    sections = _SECTION_RE.split(text)
    chunks: list[dict[str, str]] = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        title = _section_title(sec)

        if len(sec) <= size:
            chunks.append({"text": sec, "section": title})
            continue

        # Büyük bölümü overlap ile parçala
        start = 0
        while start < len(sec):
            end = min(start + size, len(sec))
            chunks.append({"text": sec[start:end], "section": title})
            if end == len(sec):
                break
            start = max(0, end - overlap)

    return chunks


# ── Corpus oluşturma ──────────────────────────────────────────────────────────

def build_corpus(docs_dir: Path) -> list[dict[str, Any]]:
    """
    Dönen her kayıt:
    {
      "text": str,
      "meta": {
        "file": str,
        "pdf_path": str,
        "page": int,
        "chunk_idx": int,
        "section": str,
        "standard": str,   # dosya adından çıkarılan ön ek, ör. "SAE_J3016"
      }
    }
    """
    corpus: list[dict[str, Any]] = []
    pdfs = sorted(docs_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"PDF bulunamadı: {docs_dir}")

    for pdf in pdfs:
        # Standart adını dosya adından çıkar (uzantısız)
        standard = pdf.stem.replace(" ", "_")

        try:
            doc = fitz.open(str(pdf))
        except Exception as exc:
            print(f"[UYARI] Açılamadı: {pdf.name} — {exc}")
            continue

        for pno in range(len(doc)):
            page_txt = extract_page_text(doc, pno)
            if not page_txt:
                continue

            for cidx, chunk in enumerate(chunk_text(page_txt)):
                corpus.append({
                    "text": chunk["text"],
                    "meta": {
                        "file":      pdf.name,
                        "pdf_path":  str(pdf.resolve()),
                        "page":      pno + 1,
                        "chunk_idx": cidx,
                        "section":   chunk["section"],
                        "standard":  standard,
                    },
                })
        doc.close()

    return corpus
