#!/usr/bin/env python
# cli.py — Komut satırı arayüzü
"""
Kullanım:
  python cli.py index --docs ./pdfs
  python cli.py ask "SAE J3016 seviyeleri nelerdir?" --k 6
  python cli.py ui
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_index(args):
    from core.indexer import build_full_index
    docs_dir = Path(args.docs).resolve()
    print(f"[INDEX] {docs_dir}")
    result = build_full_index(docs_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_ask(args):
    from core.pipeline import ask
    filters = {"standard": args.filter} if args.filter else None
    result  = ask(
        question=args.question,
        k=args.k,
        alpha=args.alpha,
        score_threshold=args.threshold,
        filters=filters,
    )
    print(result["answer"])
    print()
    print(result["sources"])


def cmd_ui(_args):
    app = ROOT / "ui" / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.port=8501",
        "--server.address=127.0.0.1",
    ])


def main():
    ap = argparse.ArgumentParser(description="RAG | Teknik Standartlar")
    sub = ap.add_subparsers(dest="cmd")

    # index
    p_idx = sub.add_parser("index", help="PDF'leri indeksle")
    p_idx.add_argument("--docs", default="./pdfs", help="PDF klasörü")

    # ask
    p_ask = sub.add_parser("ask", help="Soru sor")
    p_ask.add_argument("question")
    p_ask.add_argument("--k",         type=int,   default=6)
    p_ask.add_argument("--alpha",     type=float, default=0.60)
    p_ask.add_argument("--threshold", type=float, default=0.40)
    p_ask.add_argument("--filter",    type=str,   default=None,
                       help="Standart filtresi, ör: SAE_J3016")

    # ui
    sub.add_parser("ui", help="Streamlit arayüzünü başlat")

    args = ap.parse_args()
    if   args.cmd == "index": cmd_index(args)
    elif args.cmd == "ask":   cmd_ask(args)
    elif args.cmd == "ui":    cmd_ui(args)
    else:                     ap.print_help()


if __name__ == "__main__":
    main()
