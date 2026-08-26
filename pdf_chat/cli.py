"""The command-line interface: `python -m pdf_chat ...`.

Two sub-commands mirror the two verbs of the pipeline:

    python -m pdf_chat ingest data/*.pdf --index .index
    python -m pdf_chat ask "how many vacation days?" --index .index

`ingest` reads and indexes your documents once and saves the index to a folder.
`ask` loads that index and answers a question. Add --real to use a live LLM
(needs an API key in your environment); leave it off to run fully offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generate import Answer
from .pipeline import Rag


def _print_answer(answer: Answer) -> None:
    """Pretty-print an Answer: the text, then its sources, then the confidence."""
    print("\n" + answer.text + "\n")
    if answer.refused:
        print(f"(confidence {answer.top_score:.3f} -- below the guardrail, so I refused)")
        return
    print("Sources:")
    for tag in answer.citations():
        print(f"  - {tag}")
    print(f"\n(top match score: {answer.top_score:.3f})")


def cmd_ingest(args: argparse.Namespace) -> int:
    files = [Path(p) for p in args.files]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(f"These files do not exist: {', '.join(missing)}", file=sys.stderr)
        return 1
    rag = Rag(chunk_size=args.chunk_size, overlap=args.overlap)
    n = rag.ingest(files)
    rag.save(args.index)
    print(f"Indexed {n} chunks from {len(files)} file(s) into {args.index!r}.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    index = Path(args.index)
    if not index.exists():
        print(f"No index at {args.index!r}. Run `ingest` first.", file=sys.stderr)
        return 1
    rag = Rag.load(index)
    answer = rag.ask(args.question, k=args.k, offline=not args.real, model=args.model)
    _print_answer(answer)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_chat",
        description="Chat with your PDFs using retrieval-augmented generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="read + index PDF/TXT files into a saved index")
    p_ingest.add_argument("files", nargs="+", help="the .pdf/.txt files to index")
    p_ingest.add_argument("--index", default=".index", help="folder to save the index in")
    p_ingest.add_argument("--chunk-size", type=int, default=500, dest="chunk_size")
    p_ingest.add_argument("--overlap", type=int, default=100)
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a question against a saved index")
    p_ask.add_argument("question", help="your question, in quotes")
    p_ask.add_argument("--index", default=".index", help="folder the index was saved in")
    p_ask.add_argument("-k", type=int, default=4, help="how many chunks to retrieve")
    p_ask.add_argument("--real", action="store_true", help="use a real LLM (needs an API key)")
    p_ask.add_argument("--model", default=None, help="override the LiteLLM model string")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
