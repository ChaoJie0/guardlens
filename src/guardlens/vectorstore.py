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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _ensure_model_cached(model_name: str) -> None:
    """Avoid the multi-minute stall where SentenceTransformer issues a HEAD
    request to huggingface.co on every load (which is refused offline and
    retried with backoff). If the model is already in the local HF cache, run
    in offline mode. Otherwise let it download via the configured mirror.
    """
    if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
        return
    try:
        from huggingface_hub import try_to_load_from_cache

        if try_to_load_from_cache(model_name, "config.json") is not None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
    except Exception:  # noqa: BLE001
        pass


import chromadb  # noqa: E402
import torch  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from .corpus import Document  # noqa: E402

EMBED_MODEL = "BAAI/bge-m3"
COLLECTION = "guardlens_corpus"
# BGE-M3 supports up to 8192 tokens, but audit cards lead with
# decision/tool/reason/command, so 384 tokens keeps the retrieval signal while
# the Apple Silicon GPU (MPS) encodes ~13 docs/s (~7 min for 5.5k docs) vs
# ~22 min on CPU at seq512. Full text is still stored in Chroma for the LLM —
# only the embedding input is truncated.
MAX_SEQ_LEN = 384
# ~384 mixed CJK/ASCII tokens is roughly 1300 chars; clip before encode.
MAX_CHARS = 1300
# Measured on M5/MPS: batch 64 is faster than 128 for this model size; CPU
# uses a smaller batch to cap peak memory.
BATCH_SIZE_GPU = 64
BATCH_SIZE_CPU = 32


def pick_device(prefer: str = "mps") -> str:
    """Return the best available torch device.

    Order on Apple Silicon: ``mps`` (Metal) -> ``cpu``. A probe tensor guards
    against a build that reports MPS available but fails at kernel launch.
    """
    if prefer == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize() if hasattr(torch.mps, "synchronize") else None
            torch.tensor([1.0], device="mps") * 2
            return "mps"
        except Exception as exc:  # noqa: BLE001
            print(f"[embedder] MPS unavailable at runtime ({exc}); using CPU")
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    """Thin lazy wrapper around a local SentenceTransformer (BGE-M3).

    Uses the Apple Silicon GPU (Metal/MPS) automatically and falls back to CPU
    if MPS is unavailable or errors at load time.
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        max_seq_len: int = MAX_SEQ_LEN,
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.requested_device = device
        self.device = device or "cpu"
        self.batch_size = BATCH_SIZE_CPU
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> SentenceTransformer:
        _ensure_model_cached(self.model_name)
        target = self.requested_device or pick_device()
        # Load weights on CPU (avoids a network/device-init race inside the
        # constructor), then move to the accelerator and probe a real encode.
        m = SentenceTransformer(self.model_name, device="cpu")
        m.max_seq_length = self.max_seq_len
        device = "cpu"
        if target in ("mps", "cuda"):
            try:
                m.to(target)
                probe = m.encode(["mps probe"], batch_size=1, normalize_embeddings=True)
                if target == "mps":
                    torch.mps.synchronize()
                assert len(probe[0]) > 0
                device = target
            except Exception as exc:  # noqa: BLE001
                print(f"[embedder] {target} encode probe failed ({exc}); using CPU")
                m.to("cpu")
                device = "cpu"
        self.device = device
        self.batch_size = BATCH_SIZE_GPU if device in ("mps", "cuda") else BATCH_SIZE_CPU
        print(f"[embedder] {self.model_name} on {device} (seq={self.max_seq_len})")
        return m

    def encode(self, texts: list[str]) -> list[list[float]]:
        clipped = [t[:MAX_CHARS] for t in texts]
        model = self.model  # triggers device selection once
        vectors = model.encode(
            clipped,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(clipped) > 128,
        )
        if self.device == "mps":
            # ensure any async Metal work is finished before returning
            torch.mps.synchronize()
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
