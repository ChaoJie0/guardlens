"""W2 offline pipeline CLI: DuckDB -> corpus -> BGE-M3 embeddings -> Chroma.

Usage:
    uv run python -m guardlens.build_index \
        --db data/derived/guardlens.duckdb \
        --reports "/path/to/turn-reports"
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from .corpus import build_corpus
from .vectorstore import Embedder, index_documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GuardLens corpus + vector index")
    parser.add_argument("--db", default="data/derived/guardlens.duckdb")
    parser.add_argument(
        "--reports",
        default="/path/to/turn-reports",
        help="turn-*.md reports dir (optional)",
    )
    parser.add_argument("--out", default="data/derived/corpus")
    parser.add_argument("--chroma", default="data/derived/chroma")
    args = parser.parse_args(argv)

    reports = args.reports if Path(args.reports).exists() else None
    t0 = time.perf_counter()
    docs = build_corpus(args.db, reports, args.out)
    by_type = Counter(d.doc_type for d in docs)
    print(f"corpus: {len(docs)} docs in {time.perf_counter() - t0:.1f}s")
    for t, n in by_type.most_common():
        print(f"  {t:<16} {n}")

    embedder = Embedder()
    # force model load now so the chosen device is reported before encoding
    _ = embedder.model
    print(
        f"embedder: {embedder.model_name} device={embedder.device} "
        f"seq_len={embedder.max_seq_len} batch={embedder.batch_size}"
    )
    t1 = time.perf_counter()
    n = index_documents(docs, args.chroma, embedder=embedder)
    print(f"indexed {n} docs into Chroma in {time.perf_counter() - t1:.1f}s -> {args.chroma}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
