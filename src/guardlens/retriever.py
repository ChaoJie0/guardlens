"""Naive top-k semantic retrieval (W2).

W3 will add BM25 hybrid fusion + rerank + filters; here we keep a single,
easy-to-test dense top-k over Chroma and return typed hits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vectorstore import COLLECTION, Embedder, open_collection

# Intent -> a focused card-vocabulary query (kept short to avoid dilution).
_INTENT_QUERIES: list[tuple[str, str]] = [
    ("绕过", "curl 拦截 web_fetch 同一 URL 换工具"),
    ("网络", "curl wget 外联 network BLOCKED"),
    ("提权", "sudo chmod 提权 BLOCKED"),
    ("写文件", "file_write 工作区外 路径 BLOCKED"),
]

# Rare-but-real events lose to dense similarity. A triggered intent gets an
# exact structured fallback (metadata filter and/or document keyword) so the
# 5 fail-closed rows / chmod attempts are never silently missed. W2 only;
# W3 replaces this with BM25 hybrid fusion.
_STRUCTURED_FALLBACKS: list[tuple[str, dict[str, Any], list[str]]] = [
    ("fail", {"category": "fail-closed"}, []),
    ("fail-closed", {"category": "fail-closed"}, []),
    ("提权", {}, ["sudo", "chmod", "chown"]),
    ("chmod", {}, ["chmod"]),
    ("sudo", {}, ["sudo"]),
]


@dataclass(slots=True)
class Hit:
    evidence_id: str
    score: float
    doc_type: str
    text: str
    ts: str | None
    tool: str | None
    category: str | None
    decision: str | None
    metadata: dict[str, Any]

    def brief(self, max_chars: int = 600) -> str:
        return self.text[:max_chars]


class Retriever:
    def __init__(
        self,
        persist_dir: str | Path,
        *,
        collection_name: str = COLLECTION,
        embedder: Embedder | None = None,
    ) -> None:
        self.collection = open_collection(persist_dir, collection_name)
        self.embedder = embedder or Embedder()

    def _query(
        self,
        question: str,
        k: int,
        where: dict[str, Any] | None,
        *,
        where_document: dict[str, Any] | None = None,
    ) -> list[Hit]:
        [query_vec] = self.embedder.encode([question])
        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where or None,
            where_document=where_document or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        hits: list[Hit] = []
        for evidence_id, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            try:
                extra = json.loads(meta.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                extra = {}
            hits.append(
                Hit(
                    evidence_id=evidence_id,
                    score=round(score, 4),
                    doc_type=meta.get("doc_type", ""),
                    text=text,
                    ts=meta.get("ts") or None,
                    tool=meta.get("tool") or None,
                    category=meta.get("category") or None,
                    decision=meta.get("decision") or None,
                    metadata=extra,
                )
            )
        return hits

    def _structured_hits(self, question: str, budget: int) -> list[Hit]:
        """Exact fallback for triggered intents (rare real events)."""
        out: list[Hit] = []
        for term, where, keywords in _STRUCTURED_FALLBACKS:
            if term not in question:
                continue
            if where:
                out += self._query(question, budget, where)
            for kw in keywords:
                out += self._query(
                    question,
                    budget,
                    {"doc_type": "event_card"},
                    where_document={"$contains": kw},
                )
        # de-dup, preserve the order added (exact matches are authoritative)
        seen: set[str] = set()
        unique: list[Hit] = []
        for h in out:
            if h.evidence_id not in seen:
                seen.add(h.evidence_id)
                unique.append(h)
        return unique

    def retrieve(
        self,
        question: str,
        k: int = 6,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Dense top-k with light query expansion + per-type quota (W2).

        No BM25/rerank yet (W3). Two pragmatic fixes:

        * expand a few Chinese audit intents into the vocabulary the cards
          actually use (deny/block/tool names), so an abstract question like
          “绕过网络限制” still surfaces the curl/web_fetch event cards;
        * the ~4.2k near-duplicate ``## 被拦截的操作`` report chunks otherwise
          crowd event cards out — over-fetch and reserve slots for events.
        """
        over = max(k * 4, 24)
        queries = [question]
        for term, focused in _INTENT_QUERIES:
            if term in question:
                queries.append(focused)
                break

        # Structured exact matches go first: they prove a rare event exists
        # (5 fail-closed rows, chmod attempts) even when dense rank buries them.
        anchored = [] if where else self._structured_hits(question, k)

        merged: dict[str, Hit] = {h.evidence_id: h for h in anchored}
        for q in queries:
            for h in self._query(q, over, where):
                cur = merged.get(h.evidence_id)
                if cur is None or h.score > cur.score:
                    merged[h.evidence_id] = h
        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)

        if where:  # explicit filter -> pure rank
            return hits[:k]

        anchor_ids = {h.evidence_id for h in anchored}
        events = [
            h
            for h in hits
            if h.doc_type in ("event_card", "event_cluster") or h.evidence_id in anchor_ids
        ]
        reports = [h for h in hits if h.doc_type == "report_chunk"]
        # Reserve the first slots for anchored exact evidence, then type quota.
        picked = list(anchored)[:k]
        n_event = max(1, k - k // 3)
        picked += [h for h in events if h.evidence_id not in anchor_ids][
            : max(0, n_event - len(picked))
        ]
        have = {h.evidence_id for h in picked}
        picked += [h for h in reports if h.evidence_id not in have][: k - len(picked)]
        if len(picked) < k:
            picked += [h for h in hits if h.evidence_id not in have][: k - len(picked)]
        return picked[:k]
