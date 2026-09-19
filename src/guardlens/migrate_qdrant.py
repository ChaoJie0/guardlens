"""W3 migration CLI: corpus.jsonl -> BGE-M3 (MPS) embeddings -> Qdrant.

Assumes Qdrant is running via docker compose (localhost:6333). Embeddings use
the same MPS-aware Embedder as W2; documents are re-encoded for the new store.

Usage:
    docker compose up -d
    uv run python -m guardlens.migrate_qdrant
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .corpus import Document
from .qdrant_store import COLLECTION, DEFAULT_URL, count_documents, index_documents
from .vectorstore import Embedder


def load_corpus(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs.append(
                Document(
                    evidence_id=d["evidence_id"],
                    doc_type=d["doc_type"],
                    text=d["text"],
                    ts=d.get("ts"),
                    tool=d.get("tool"),
                    category=d.get("category"),
                    decision=d.get("decision"),
                    metadata=d.get("metadata", {}),
                )
            )
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate corpus to Qdrant")
    parser.add_argument("--corpus", default="data/derived/corpus/corpus.jsonl")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    docs = load_corpus(args.corpus)
    print(f"loaded {len(docs)} corpus docs from {args.corpus}")

    embedder = Embedder()
    _ = embedder.model  # force device report
    print(
        f"embedder device={embedder.device} seq={embedder.max_seq_len} batch={embedder.batch_size}"
    )

    t0 = time.perf_counter()
    n = index_documents(
        docs,
        embedder=embedder,
        url=args.url,
        name=args.collection,
        batch_size=args.batch_size,
    )
    dt = time.perf_counter() - t0
    total = count_documents(args.url, args.collection)
    print(f"indexed {n} -> Qdrant '{args.collection}' in {dt:.1f}s; count={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
