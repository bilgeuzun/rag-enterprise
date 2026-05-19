# core/config.py — Merkezi konfigürasyon
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Dizinler ──────────────────────────────────────────────────────────────────
DOCS_DIR  = Path(os.environ.get("DOCS_DIR",  str(BASE_DIR / "pdfs")))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", str(BASE_DIR / "index")))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Model yolları ─────────────────────────────────────────────────────────────
EMB_MODEL = os.environ.get(
    "EMB_MODEL",
    str(BASE_DIR / "models" / "bge-m3")           # önerilen: BAAI/bge-m3
    # alternatif: "models/multilingual-e5-large"
)

# Ollama tercih edilir; yoksa llama.cpp GGUF fallback
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL",    "qwen2.5:14b")

LLM_GGUF = os.environ.get(
    "LLM_GGUF",
    str(BASE_DIR / "models" / "llm" / "qwen2.5-14b-instruct-q4_k_m.gguf")
)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE",    "1400"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K          = int(os.environ.get("TOP_K", "6"))
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.40"))
BM25_ALPHA     = float(os.environ.get("BM25_ALPHA", "0.60"))  # semantic ağırlığı

# ── LLM inference ─────────────────────────────────────────────────────────────
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.05"))
LLM_MAX_TOKENS  = int(os.environ.get("LLM_MAX_TOKENS",  "768"))
LLM_N_CTX       = int(os.environ.get("LLM_N_CTX",      "4096"))
