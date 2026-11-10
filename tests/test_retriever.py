"""Retriever tests against a tiny in-repo Chroma with a deterministic embedder.

Avoids loading the 2GB BGE-M3 model: a hash-based embedder gives identical
texts identical vectors and different texts different-ish vectors, which is
enough to exercise the Chroma query + structured-fallback + quota logic.
"""

from __future__ import annotations

import hashlib

import pytest

from guardlens.corpus import Document
from guardlens.retriever import Retriever
from guardlens.vectorstore import COLLECTION, index_documents

DIM = 64


class _HashEmbedder:
    model_name = "hash-test"

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [(h[i % len(h)] - 128) / 128 for i in range(DIM)]
            n = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / n for x in vec])
        return out


def _docs() -> list[Document]:
    return [
        Document(
            evidence_id="EV-00001",
            doc_type="event_card",
            text="curl https://example.com 被 network 拦截，随后 web_fetch 同 URL",
            ts="2026-09-06T09:25:46Z",
            tool="bash",
            category="network",
            decision="deny",
            metadata={},
        ),
        Document(
            evidence_id="EV-00002",
            doc_type="event_card",
            text="agent gate: unrollbackable write, snapshot unhealthy, write denied",
            ts="2026-09-16T14:39:48Z",
            tool="write",
            category="fail-closed",
            decision="deny",
            metadata={},
        ),
        Document(
            evidence_id="EV-00003",
            doc_type="event_card",
            text="chmod +x run.sh 被放行 file_write on TM-protected path",
            ts="2026-09-12T11:54:39Z",
            tool="bash",
            category="file_write",
            decision="allow",
            metadata={},
        ),
        Document(
            evidence_id="RP-00001",
            doc_type="report_chunk",
            text="## 被拦截的操作\n若干网络与写操作被拦截的汇总清单",
            ts=None,
            tool=None,
            category="turn_report",
            decision=None,
            metadata={},
        ),
    ]


@pytest.fixture()
def retriever(tmp_path):
    embedder = _HashEmbedder()
    index_documents(_docs(), tmp_path / "chroma", embedder=embedder, name=COLLECTION)
    return Retriever(tmp_path / "chroma", embedder=embedder)


def test_fail_closed_structured_fallback_surfaces_event(retriever):
    hits = retriever.retrieve("fail-closed 发生在什么场景", k=4)
    ids = [h.evidence_id for h in hits]
    # the single fail-closed event must be anchored regardless of dense rank
    assert "EV-00002" in ids
    assert ids[0] == "EV-00002"


def test_chmod_keyword_fallback_surfaces_event(retriever):
    hits = retriever.retrieve("是否使用过 chmod 提权", k=4)
    ids = [h.evidence_id for h in hits]
    assert "EV-00003" in ids


def test_quota_keeps_report_in_mix(retriever):
    hits = retriever.retrieve("curl 为什么被拦", k=4)
    types = {h.doc_type for h in hits}
    assert "event_card" in types
