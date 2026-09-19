"""Qdrant vector backend (W3 production store).

Replaces Chroma as the primary store. Runs via ``docker compose`` with a
local persistent volume; payload fields (ts/tool/category/decision/doc_type/
evidence_id) enable server-side pre-filtering.

Chroma remains on disk as the W2 fallback / teaching baseline.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .corpus import Document
from .vectorstore import Embedder

COLLECTION = "guardlens_corpus"
DEFAULT_URL = "http://localhost:6333"
VECTOR_DIM = 1024  # bge-m3 (regardless of input seq_len, output dim fixed)

_PAYLOAD_META_KEY = "metadata_json"


def connect(url: str = DEFAULT_URL) -> QdrantClient:
    # Local pinned server (1.12.x) may lag the client; the vectors/payload/
    # query_points API we use is stable across these versions.
    return QdrantClient(url=url, timeout=30, check_compatibility=False)


def reset_collection(client: QdrantClient, name: str = COLLECTION) -> None:
    client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=VECTOR_DIM, distance=qm.Distance.COSINE),
    )


def _payload(doc: Document) -> dict[str, Any]:
    import json

    return {
        "evidence_id": doc.evidence_id,
        "doc_type": doc.doc_type or "",
        "ts": doc.ts or "",
        "tool": doc.tool or "",
        "category": doc.category or "",
        "decision": doc.decision or "",
        _PAYLOAD_META_KEY: json.dumps(doc.metadata, ensure_ascii=False),
    }


def index_documents(
    docs: list[Document],
    *,
    embedder: Embedder | None = None,
    url: str = DEFAULT_URL,
    name: str = COLLECTION,
    batch_size: int = 128,
) -> int:
    """Embed with BGE-M3 (MPS when available) and upsert to Qdrant."""
    if not docs:
        return 0
    embedder = embedder or Embedder()
    client = connect(url)
    reset_collection(client, name)

    n = 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        vectors = embedder.encode([d.text for d in batch])
        points = [
            qm.PointStruct(
                id=idx,
                vector=vec,
                payload={**_payload(doc), "text": doc.text},
            )
            for idx, (doc, vec) in enumerate(zip(batch, vectors, strict=True), start=start)
        ]
        client.upsert(collection_name=name, points=points, wait=True)
        n += len(batch)
    client.close()
    return n


def count_documents(url: str = DEFAULT_URL, name: str = COLLECTION) -> int:
    client = connect(url)
    try:
        return client.count(collection_name=name).count
    finally:
        client.close()
