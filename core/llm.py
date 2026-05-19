# core/llm.py — LLM katmanı: Ollama öncelikli, llama.cpp fallback
from __future__ import annotations
import re
import subprocess
from typing import Any

from core.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_GGUF,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_N_CTX,
)


# ── Ollama ────────────────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _ollama_chat(messages: list[dict], temperature: float, max_tokens: int) -> str:
    import json, urllib.request
    payload = json.dumps({
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["message"]["content"].strip()


# ── llama.cpp fallback ────────────────────────────────────────────────────────

_LLM_CPP = None

def _auto_gpu_layers() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        vram = float(out[0]) / 1024.0
        if vram >= 7.5:  return 50
        if vram >= 5.5:  return 35
        if vram >= 3.5:  return 20
    except Exception:
        pass
    return 0


def _get_llm_cpp():
    global _LLM_CPP
    if _LLM_CPP is None:
        from llama_cpp import Llama
        import os
        if not os.path.exists(LLM_GGUF):
            raise FileNotFoundError(
                f"GGUF model bulunamadı: {LLM_GGUF}\n"
                "Ollama kurun veya LLM_GGUF ortam değişkenini ayarlayın."
            )
        _LLM_CPP = Llama(
            model_path=LLM_GGUF,
            n_ctx=LLM_N_CTX,
            n_threads=__import__("os").cpu_count() or 4,
            n_gpu_layers=_auto_gpu_layers(),
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
    return _LLM_CPP


def _llama_cpp_chat(messages: list[dict], temperature: float, max_tokens: int) -> str:
    llm = _get_llm_cpp()
    out = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return out["choices"][0]["message"]["content"].strip()


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Sen mühendislik standartları ve teknik PDF belgelerinde uzmanlaşmış bir asistansın.

KURALLAR:
1. Yanıtını YALNIZCA verilen kaynak pasajlarına dayandır.
2. Her bilgi cümlesinin sonuna ilgili kaynağı köşeli parantezle ekle: [S1], [S2] …
3. Birden fazla kaynaktan gelen bilgiyi birleştiriyorsan tüm etiketleri yaz: [S1][S3]
4. Teknik sayısal değerleri (tolerans, gerilim, basınç, ölçü) ASLA değiştirme — olduğu gibi aktar.
5. Bağlamda yeterli bilgi yoksa: "Bu konuda belgelerde yeterli bilgi bulunamadı." de.
6. Varsayım yapma; emin değilsen "belirtilmemiş" yaz.
7. Yanıtı maddeli ve net yaz; teknik jargonu koru.\
"""


def build_messages(question: str, hits: list[dict[str, Any]]) -> list[dict]:
    blocks = []
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        tag  = f"[S{i}]"
        ref  = f"{meta['file']} — s.{meta['page']}"
        if meta.get("section"):
            ref += f" — {meta['section'][:80]}"
        blocks.append(f"{tag} {ref}\n{h['text'].strip()}")

    context = "\n\n---\n\n".join(blocks)

    user_msg = (
        f"Soru: {question}\n\n"
        f"Kaynak pasajlar:\n{context}\n\n"
        "Yanıt (her cümle sonunda [S#] etiketi olsun):"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


# ── Citation doğrulama ────────────────────────────────────────────────────────

def validate_citations(answer: str, hit_count: int) -> str:
    """Modelin halüsine ettiği [S#] referanslarını [?] ile işaretle."""
    valid = {f"[S{i}]" for i in range(1, hit_count + 1)}
    found = set(re.findall(r"\[S\d+\]", answer))
    for tag in found - valid:
        answer = answer.replace(tag, "[?]")
    return answer


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    hits: list[dict[str, Any]],
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int    = LLM_MAX_TOKENS,
) -> tuple[str, str]:
    """
    Returns (answer_text, sources_markdown)
    """
    if not hits:
        return "Bu konuda belgelerinizde ilgili bilgi bulunamadı.", ""

    messages = build_messages(question, hits)

    # Ollama tercih edilir
    if _ollama_available():
        raw = _ollama_chat(messages, temperature, max_tokens)
        backend = f"Ollama ({OLLAMA_MODEL})"
    else:
        raw = _llama_cpp_chat(messages, temperature, max_tokens)
        backend = "llama.cpp"

    answer = validate_citations(raw, len(hits))

    # Kaynak tablosu
    source_lines = []
    for i, h in enumerate(hits, 1):
        meta    = h["meta"]
        snippet = h["text"].replace("\n", " ")[:160]
        score   = h.get("score", 0.0)
        source_lines.append(
            f"**[S{i}]** `{meta['file']}` — s.{meta['page']}"
            + (f" — *{meta['section'][:60]}*" if meta.get("section") else "")
            + f"  _(skor: {score:.3f})_\n> {snippet}…"
        )

    sources_md = (
        f"---\n### 📚 Kaynaklar _(backend: {backend})_\n\n"
        + "\n\n".join(source_lines)
    )

    return answer, sources_md
