#!/usr/bin/env python
"""
main.py  —  Command-line interface for the RAG Semantic Search module.

Usage examples
--------------
# Index a directory of documents, then search:
    python main.py index --source data/documents/ --save data/index.pkl

# Search against a previously built index:
    python main.py search --index data/index.pkl --query "Quelles sont les recommandations?"

# Index then search (one step):
    python main.py run --source data/documents/ --query "Quels sont les cas d'usage ?"
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

from rag_module.pipeline import RAGPipeline


# ──────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ──────────────────────────────────────────────────────────────────────────────

def cmd_index(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    print(f"\n[INFO] Indexing documents from: {args.source}")
    n = pipeline.index_directory(args.source)
    print(f"[INFO] {n} fragments indexed successfully.")

    if args.save:
        pipeline.save_index(args.save)
        print(f"[INFO] Index saved to: {args.save}")

    if args.export_meta:
        pipeline.export_metadata(args.export_meta)
        print(f"[INFO] Metadata exported to: {args.export_meta}")


def cmd_search(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(top_k=args.top_k)
    if not args.index:
        print("[ERROR] --index is required for the 'search' command.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading index from: {args.index}")
    pipeline.load_index(args.index)
    print(f"[INFO] {pipeline.fragment_count} fragments loaded.")

    query = args.query or input("\nEntrez votre question : ").strip()
    results = pipeline.search(query, top_k=args.top_k)
    pipeline.display(query, results)


def cmd_run(args: argparse.Namespace) -> None:
    """Index then search in one step (useful for demos)."""
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )

    if args.source:
        print(f"\n[INFO] Indexing documents from: {args.source}")
        n = pipeline.index_directory(args.source)
        print(f"[INFO] {n} fragments indexed.")
    elif args.index:
        pipeline.load_index(args.index)
        print(f"[INFO] {pipeline.fragment_count} fragments loaded.")
    else:
        print("[ERROR] Provide --source or --index.", file=sys.stderr)
        sys.exit(1)

    if args.save and args.source:
        pipeline.save_index(args.save)

    # Interactive loop
    print("\n[INFO] Semantic search ready. Type 'quit' to exit.\n")
    while True:
        query = args.query or input("Question : ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        results = pipeline.search(query)
        pipeline.display(query, results)
        if args.query:          # non-interactive: run once and exit
            break


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-search",
        description="RAG Semantic Search Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top-k",       type=int, default=3,  help="Number of results (default: 3)")
    common.add_argument("--chunk-size",  type=int, default=512, help="Chunk size in characters (default: 512)")
    common.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap (default: 64)")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- index ---
    p_idx = sub.add_parser("index", parents=[common], help="Index documents into the vector store.")
    p_idx.add_argument("--source",      required=True, help="Directory of documents to index.")
    p_idx.add_argument("--save",        default="data/index.pkl", help="Path to save the index.")
    p_idx.add_argument("--export-meta", default=None,  help="Optional path to export metadata as JSON.")
    p_idx.set_defaults(func=cmd_index)

    # --- search ---
    p_srch = sub.add_parser("search", parents=[common], help="Search a pre-built index.")
    p_srch.add_argument("--index",      required=True, help="Path to the saved index (.pkl).")
    p_srch.add_argument("--query",      default=None,  help="Query string (interactive if omitted).")
    p_srch.set_defaults(func=cmd_search)

    # --- run (index + search) ---
    p_run = sub.add_parser("run", parents=[common], help="Index documents then run interactive search.")
    p_run.add_argument("--source",      default=None,  help="Directory of documents to index.")
    p_run.add_argument("--index",       default=None,  help="Load an existing index instead.")
    p_run.add_argument("--save",        default=None,  help="Save the index after building it.")
    p_run.add_argument("--query",       default=None,  help="One-shot query (non-interactive).")
    p_run.set_defaults(func=cmd_run)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
