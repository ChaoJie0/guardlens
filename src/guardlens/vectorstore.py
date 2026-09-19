"""Local embeddings (BGE-M3) + Chroma persistence.

W2 deliberately uses the simple stack: local SentenceTransformer embeddings
and Chroma. W3 will add hybrid retrieval / rerank and migrate to Qdrant — the
corpus contract (Document/evidence_id/payload) is defined in corpus.py and
kept stable across that migration.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Prefer the domestic HF mirror before importing the heavy stack; harmless if
# unreachable (the normal hub is used as fallback).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from .corpus import Document  # noqa: E402

EMBED_MODEL = "BAAI/bge-m3"
COLLECTION = "guardlens_corpus"
# Semantic retrieval over audit cards/reports does not need BGE-M3's full
# 8192-token window; 1024 keeps the semantic signal while making CPU encoding
# ~an order of magnitude faster. Texts are truncated before tokenisation.
MAX_SEQ_LEN = 1024
MAX_CHARS = 3600


class Embedder:
    """Thin lazy wrapper around a local SentenceTransformer (BGE-M3)."""

    def __init__(self, model_name: str = EMBED_MODEL, max_seq_len: int = MAX_SEQ_LEN) -> None:
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            m = SentenceTransformer(self.model_name)
            m.max_seq_length = self.max_seq_len
            self._model = m
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        clipped = [t[:MAX_CHARS] for t in texts]
        vectors = self.model.encode(
            clipped,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(clipped) > 64,
        )
        return [v.tolist() for v in vectors]


def _get_client(persist_dir: str | Path) -> chromadb.api.ClientAPI:
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def reset_collection(persist_dir: str | Path, name: str = COLLECTION):
    client = _get_client(persist_dir)
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001
        pass
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def index_documents(
    docs: Iterable[Document],
    persist_dir: str | Path,
    *,
    embedder: Embedder | None = None,
    name: str = COLLECTION,
    batch_size: int = 64,
) -> int:
    """Embed and upsert documents into Chroma. Returns number indexed."""
    docs = list(docs)
    if not docs:
        return 0
    embedder = embedder or Embedder()
    collection = reset_collection(persist_dir, name)

    n = 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        texts = [d.text for d in batch]
        embeddings = embedder.encode(texts)
        metadatas: list[dict[str, Any]] = []
        for d in batch:
            md = {
                "doc_type": d.doc_type or "",
                "ts": d.ts or "",
                "tool": d.tool or "",
                "category": d.category or "",
                "decision": d.decision or "",
                "metadata_json": json.dumps(d.metadata, ensure_ascii=False),
            }
            metadatas.append(md)
        collection.upsert(
            ids=[d.evidence_id for d in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        n += len(batch)
    return n


def open_collection(persist_dir: str | Path, name: str = COLLECTION):
    client = _get_client(persist_dir)
    return client.get_collection(name=name)
