"""FastAPI backend: the heavy models (BGE-M3, reranker, Qdrant client) load
ONCE when this service starts. The Streamlit frontend talks to it over HTTP+SSE,
so a browser refresh never pays the model-load cold cost again.

Run:
    uvicorn guardlens.server:app --port 8000

Paths default to the standard local layout and can be overridden with
``GUARDLENS_DB`` / ``GUARDLENS_CORPUS``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .orchestrator import Orchestrator

CORPUS_PATH = os.environ.get("GUARDLENS_CORPUS", "data/derived/corpus/corpus.jsonl")
DB_PATH = os.environ.get("GUARDLENS_DB", "data/derived/guardlens.duckdb")

_engine: Orchestrator | None = None


def get_engine() -> Orchestrator:
    global _engine
    if _engine is None:
        _engine = Orchestrator(corpus_path=CORPUS_PATH, db_path=DB_PATH, k=5)
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all local models once at server start (the ~80s cold cost), so the
    # first real question is warm.
    get_engine()
    yield


app = FastAPI(title="GuardLens API", lifespan=lifespan)


class AskReq(BaseModel):
    question: str


def _ser(obj):
    """Best-effort JSON-safe conversion for dataclasses (Hit / Answer)."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj


@app.get("/api/health")
def health():
    return {"ok": _engine is not None}


@app.post("/api/ask")
def ask(req: AskReq):
    eng = get_engine()

    def gen():
        for ev in eng.stream(req.question):
            safe = {k: _ser(v) for k, v in ev.items()}
            yield f"data: {json.dumps(safe, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
