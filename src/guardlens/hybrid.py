"""W3 hybrid retrieval: BM25 + dense (Qdrant) fused by RRF, then cross-encoder rerank.

Pipeline per question:
  1. dense top-N from Qdrant (BGE-M3) — semantic similarity;
  2. sparse top-N from an in-memory BM25 (exact command/tool/path/CJK terms);
  3. Reciprocal Rank Fusion (k=60) into a fused top-20;
  4. bge-reranker-v2-m3 cross-encodes the 20 -> top-5 for the LLM.

This fixes the W2 failure mode where abstract phrasings or exact command
tokens ("chmod", "fail-closed", "web_fetch") were missed by pure vectors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qdrant_client import models as qm

from .bm25 import BM25Index, build_index
from .qdrant_store import COLLECTION, DEFAULT_URL, connect
from .reranker import Reranker
from .retriever import Hit
from .vectorstore import Embedder

RRF_K = 60
DENSE_N = 60
SPARSE_N = 60
FUSE_N = 30
RERANK_N = 45
DEFAULT_TOP_K = 5

# W2-style intent anchors preserved: rare real events must surface.
_STRUCTURED_FILTERS: list[tuple[str, dict[str, Any], list[str]]] = [
    ("fail", {"category": "fail-closed"}, []),
    ("fail-closed", {"category": "fail-closed"}, []),
    ("提权", {}, ["sudo", "chmod", "chown"]),
    ("chmod", {}, ["chmod"]),
    ("sudo", {}, ["sudo"]),
    ("curl", {}, ["curl"]),
    ("web_fetch", {}, ["web_fetch"]),
    ("urllib", {}, ["urllib", "urlopen"]),
    ("Mozilla", {}, ["Mozilla"]),
    ("npm", {}, ["npm i", "npm install"]),
    ("node", {}, ["node -e", "node --eval"]),
]

# Abstract Chinese phrasings -> the exact vocabulary the cards use. Each
# expansion is run as an extra dense+sparse query and merged by RRF; this is
# what lets “绕过网络限制” reach the curl/web_fetch event cards.
_INTENT_EXPANSIONS: list[tuple[str, list[str]]] = [
    ("绕过", ["curl 拦截 web_fetch 同一 URL 换工具", "curl denied then web_fetch"]),
    ("网络限制", ["curl web_fetch network BLOCKED 外联", "wget curl 网络访问"]),
    ("提权", ["chmod +x sudo 提权 权限"]),
    ("走私", ["curl 外发 上传 exfiltration 数据外联"]),
]


def _expand(question: str) -> list[str]:
    extras: list[str] = []
    for term, queries in _INTENT_EXPANSIONS:
        if term in question:
            extras.extend(queries)
    return extras


# Anchor body (before surrounding context) and its “命令:” line.
_ANCHOR_HEAD = re.compile(r"^(.*?)(?:\n(?:前文事件|后文事件|相关回合|上下文)|$)", re.S)
_ANCHOR_CMD = re.compile(r"命令[:：]\s*(.*)")


def rrf_fuse(
    ranked_lists: Iterable[list[str]],
    *,
    k: int = RRF_K,
    top_n: int = FUSE_N,
    weights: Iterable[float] | None = None,
) -> list[tuple[str, float]]:
    weights = list(weights) if weights is not None else []
    scores: dict[str, float] = {}
    for li, ranked in enumerate(ranked_lists):
        w = weights[li] if li < len(weights) else 1.0
        for rank, evidence_id in enumerate(ranked):
            scores[evidence_id] = scores.get(evidence_id, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


class HybridRetriever:
    """BM25 + Qdrant dense + RRF + rerank, returning W2-compatible Hits."""

    def __init__(
        self,
        corpus_path: str | Path,
        *,
        qdrant_url: str = DEFAULT_URL,
        collection: str = COLLECTION,
        embedder: Embedder | None = None,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        client: Any | None = None,
        dense_fn: Any | None = None,
    ) -> None:
        # client/dense_fn injection lets tests run without a Qdrant server or
        # the 2GB model; production connects lazily on first dense query.
        self._client = client
        self._client_pending = client is None and dense_fn is None
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.dense_fn = dense_fn
        self.embedder = embedder  # created lazily in production
        self.bm25 = bm25 or build_index(corpus_path)
        self.reranker = reranker  # lazy: only created when rerank=True
        self._by_id: dict[str, dict[str, Any]] = {d["evidence_id"]: d for d in self.bm25.docs}

    @property
    def client(self):
        if self._client is None:
            self._client = connect(self.qdrant_url)
        return self._client

    # -- dense -----------------------------------------------------------
    def dense_search(
        self, question: str, n: int = DENSE_N, *, where: dict | None = None
    ) -> list[Hit]:
        if self.dense_fn is not None:
            return self.dense_fn(question, n, where)
        if self.embedder is None:
            self.embedder = Embedder()
        [vec] = self.embedder.encode([question])
        res = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=n,
            query_filter=_to_qdrant_filter(where),
            with_payload=True,
        )
        return [self._hit_from_point(p, score_attr="dense") for p in res.points]

    # -- sparse ----------------------------------------------------------
    def sparse_search(self, question: str, n: int = SPARSE_N) -> list[Hit]:
        out: list[Hit] = []
        for evidence_id, score in self.bm25.search(question, k=n):
            h = self._hit_from_doc(evidence_id)
            if h:
                h.score = round(score, 4)
                out.append(h)
        return out

    def structured_anchor(self, question: str, budget: int = 15) -> list[Hit]:
        """Exact metadata/keyword anchor for rare events (W2 fallback kept).

        For keyword intents (chmod/sudo) prefer cards whose *anchor body*
        (leading section, before the before/after context) contains the term,
        so a command appearing only in a surrounding-event line does not win.
        """
        hits: list[Hit] = []
        for term, where, keywords in _STRUCTURED_FILTERS:
            if term not in question:
                continue
            if where:
                for h in self._dense_with_filter(question, where, budget):
                    h.metadata = {**(h.metadata or {}), "anchor_kind": "payload"}
                    hits.append(h)
            for kw in keywords:
                # Wide BM25 candidate set, then prioritise by where the term
                # appears: (1) the anchor “命令:” line starts with the term
                # (a real `chmod +x` event) > (2) anywhere in the anchor body
                # > (3) only in surrounding context. Length-normalised BM25
                # alone buries the long real-command cards, so this explicit
                # positional signal is required.
                cand = [eid for eid, _ in self.bm25.search(kw, k=300)]
                rank_of = {eid: i for i, eid in enumerate(cand)}
                cmd_line, primary, secondary = [], [], []
                for eid in cand:
                    doc = self._by_id.get(eid)
                    if not doc or doc.get("doc_type") != "event_card":
                        continue
                    text = doc.get("text", "")
                    head_m = _ANCHOR_HEAD.search(text)
                    head = head_m.group(1) if head_m else text[:420]
                    m = _ANCHOR_CMD.search(head)
                    cmd = (m.group(1) if m else "").lower()
                    if re.search(rf"(^|[\s;|&]){re.escape(kw)}\b", cmd):
                        cmd_line.append(eid)
                    elif kw in head.lower():
                        primary.append(eid)
                    else:
                        secondary.append(eid)
                for tier in (cmd_line, primary, secondary):
                    tier.sort(key=rank_of.get)
                ordered = (cmd_line + primary + secondary)[:budget]
                hits.extend(filter(None, (self._hit_from_doc(i) for i in ordered)))
        seen: set[str] = set()
        uniq = []
        for h in hits:
            if h.evidence_id not in seen and h.doc_type in ("event_card", "event_cluster"):
                seen.add(h.evidence_id)
                uniq.append(h)
        return uniq

    def _dense_with_filter(self, question: str, where: dict, budget: int) -> list[Hit]:
        try:
            return self.dense_search(question, budget, where=where)
        except Exception:  # noqa: BLE001
            return []

    # -- fused public API ------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: int = DEFAULT_TOP_K,
        *,
        rerank: bool = True,
        anchors: bool = True,
    ) -> list[Hit]:
        def _is_ev(h: Hit) -> bool:
            return h.doc_type in ("event_card", "event_cluster")

        def _balanced(hits: list[Hit], per_type: int) -> list[Hit]:
            """Three-way type balance: event cards, event clusters, and the
            rest. Clusters are scarce (123 vs 2038 cards) but dense/BM25 rank
            them late because their text is long, so they get a reserved quota
            instead of competing with cards for the same event slots."""
            cards = [h for h in hits if h.doc_type == "event_card"][:per_type]
            clusters = [h for h in hits if h.doc_type == "event_cluster"][: max(5, per_type // 5)]
            rp = [h for h in hits if h.doc_type not in ("event_card", "event_cluster")][:per_type]
            return cards + clusters + rp

        anchor = self.structured_anchor(question) if anchors else []

        # Over-fetch each channel, then type-balance BEFORE fusion so the ~3.4x
        # more numerous report chunks cannot monopolise RRF while cross-type
        # reciprocal signals (an event card and a report chunk reinforcing each
        # other) are still preserved in a single global fusion.
        all_ranked: list[list[str]] = []
        weights: list[float] = []
        all_hits: list[Hit] = []
        # Strong single-channel matches must survive RRF dilution: the top-2 of
        # every balanced channel enters the pool directly (capped at 12 after
        # dedupe) so a #1-ranked event is not out-scored by docs present in
        # many channels at mediocre ranks — without letting boosts monopolise
        # the rerank input and starve the RRF-fused candidates. Clusters are a
        # separate concern: the cross-encoder re-scores them well against the
        # *expanded* query, but RRF naturally ranks them too low to enter the
        # fused top-30, so each channel's best cluster is boosted too (capped 6).
        boost_ids: list[str] = []
        cluster_boost: list[str] = []

        def _absorb(hits: list[Hit], w: float):
            bal = _balanced(hits, FUSE_N)
            boost_ids.extend(h.evidence_id for h in bal[:2])
            clusters = [h for h in bal if h.doc_type == "event_cluster"]
            if clusters:
                cluster_boost.append(clusters[0].evidence_id)
            all_ranked.append([h.evidence_id for h in bal])
            weights.append(w)
            all_hits.extend(bal)

        _absorb(self.dense_search(question, DENSE_N * 2), 1.0)
        _absorb(self.sparse_search(question, SPARSE_N * 2), 1.0)
        for eq in _expand(question):
            _absorb(self.dense_search(eq, DENSE_N * 2), 0.6)
            _absorb(self.sparse_search(eq, SPARSE_N * 2), 0.6)

        # Anchors are capped before the pool so a keyword-heavy question
        # (npm+node, chmod+sudo) cannot monopolise the rerank input.
        anchor = (anchor[:25] if anchors else []) if anchor else []
        fused = rrf_fuse(all_ranked, weights=weights, top_n=FUSE_N)
        anchor_ids = {h.evidence_id for h in anchor}

        pool: list[Hit] = []
        seen: set[str] = set()
        for h in anchor:  # exact anchors enter the pool first
            if h.evidence_id not in seen:
                pool.append(h)
                seen.add(h.evidence_id)
        hit_by_id = {h.evidence_id: h for h in all_hits + anchor}
        boost_seen: set[str] = set()
        boost_final: list[str] = []
        for evidence_id in boost_ids:
            if evidence_id in boost_seen or evidence_id in seen:
                continue
            boost_seen.add(evidence_id)
            boost_final.append(evidence_id)
            if len(boost_final) >= 12:
                break
        for evidence_id in boost_final:  # strong single-channel matches
            h = hit_by_id.get(evidence_id) or self._hit_from_doc(evidence_id)
            if h:
                pool.append(h)
                seen.add(evidence_id)
        # Per-channel best cluster (deduped, capped 6) enters ahead of the RRF
        # remainder: sequence-evidence clusters rank top in BM25 across several
        # expansion channels yet fall out of the fused top-30, and the
        # cross-encoder only scores them correctly against the expanded query.
        cl_boost_seen: set[str] = set()
        cl_boost_final: list[str] = []
        for evidence_id in cluster_boost:
            if evidence_id in cl_boost_seen or evidence_id in seen:
                continue
            cl_boost_seen.add(evidence_id)
            cl_boost_final.append(evidence_id)
            if len(cl_boost_final) >= 6:
                break
        for evidence_id in cl_boost_final:
            h = hit_by_id.get(evidence_id) or self._hit_from_doc(evidence_id)
            if h:
                pool.append(h)
                seen.add(evidence_id)
        for evidence_id, _rrf in fused:  # then the RRF-fused remainder
            if evidence_id in seen:
                continue
            h = hit_by_id.get(evidence_id) or self._hit_from_doc(evidence_id)
            if h:
                pool.append(h)
                seen.add(evidence_id)

        # Anchor policy:
        #  • payload anchors (exact server-side metadata filter, e.g. all 5
        #    fail-closed events) are ground truth for that intent and may fill
        #    all k slots, ordered by rerank among themselves;
        #  • keyword anchors (chmod/sudo primary-command hits) are strong but
        #    noisier, so they cap at k-1 to leave room for semantic context.
        payload_ids = {
            h.evidence_id for h in anchor if (h.metadata or {}).get("anchor_kind") == "payload"
        }

        def _is_event(h: Hit) -> bool:
            return h.doc_type in ("event_card", "event_cluster")

        # The pool is already type-balanced per channel; anchors enter first.
        non_anchor = [h for h in pool if h.evidence_id not in anchor_ids][: RERANK_N - len(anchor)]
        rerank_input = (list(anchor) + non_anchor)[:RERANK_N]
        if rerank:
            if self.reranker is None:
                self.reranker = Reranker()
            # Rerank against the question PLUS its intent expansions. Abstract
            # phrasings ("有没有绕过网络限制的尝试？") score long cluster texts
            # poorly, while the concrete expansion ("curl denied then
            # web_fetch") lifts the same clusters to the top; the expanded
            # query is what the fusion pool was built from, so rerank with it.
            rerank_query = question
            if anchors:
                exps = _expand(question)
                if exps:
                    seen_q: set[str] = set()
                    parts = [question]
                    for e in exps:
                        if e not in seen_q:
                            seen_q.add(e)
                            parts.append(e)
                            if len(parts) >= 4:
                                break
                    rerank_query = " ".join(parts)
            ranked = self.reranker.rerank(rerank_query, rerank_input, top_k=FUSE_N)
            ranked_anchor = [h for h in ranked if h.evidence_id in anchor_ids]
            payload_hits = [h for h in ranked_anchor if h.evidence_id in payload_ids]
            keyword_hits = [h for h in ranked_anchor if h.evidence_id not in payload_ids]
            kept = payload_hits[: min(len(payload_hits), k)]
            # Keyword anchors promote recall of exact events, but a low-scoring
            # anchor (e.g. `sudo` on the command line of an unrelated card)
            # must not displace a genuinely relevant rerank hit. Only anchors
            # scoring at/above the rerank median get pinned; the rest fall
            # through to the normal rerank ordering.
            if ranked:
                _med = sorted(float(h.score) for h in ranked)[len(ranked) // 2]
            else:
                _med = 0.0
            if keyword_hits and ranked:
                keyword_hits = [h for h in keyword_hits if float(h.score) >= _med]
            kept += keyword_hits[: max(0, k - 1 - len(kept))]
            have = {h.evidence_id for h in kept}

            rest = [h for h in ranked if h.evidence_id not in have]
            chosen = self._dedupe(kept + rest)[:k]
            if payload_hits:
                # exact metadata anchor already establishes the concrete events
                return chosen
            # The cross-encoder scores long report/cluster text inconsistently
            # and can demote the top RRF evidence (e.g. the event cluster that
            # summarises an evasion sequence). Anchor the result with the single
            # best RRF candidate, then fill the rest by rerank relevance — but
            # only if the cross-encoder itself scored it at/above the median,
            # so a boosted-but-irrelevant candidate can never be pinned on top.
            if non_anchor:
                top_rrf = non_anchor[0]
                top_rrf_ok = top_rrf.evidence_id not in {h.evidence_id for h in chosen[: k - 1]}
                if top_rrf_ok and float(getattr(top_rrf, "score", 0.0) or 0.0) >= _med:
                    chosen = self._dedupe([top_rrf] + chosen)[:k]
            # Sequence-intent questions (intent expansion active) are best
            # answered with a cluster: it summarises the whole evasion session,
            # but the cross-encoder ranks report chunks slightly higher. When
            # the query was expanded AND a cluster scored at/above the rerank
            # median, guarantee one cluster slot so the LLM sees session-level
            # evidence, not only per-event cards/report rows.
            if rerank_query != question and not any(h.doc_type == "event_cluster" for h in chosen):
                best_cl = next(
                    (h for h in ranked if h.doc_type == "event_cluster" and float(h.score) >= _med),
                    None,
                )
                if best_cl and best_cl.evidence_id not in {h.evidence_id for h in chosen[: k - 1]}:
                    chosen = self._dedupe(chosen[: k - 1] + [best_cl])[:k]
            # Safety: never return a top-k with no event card when one exists.
            if not any(_is_event(h) for h in chosen):
                best_event = next((h for h in rest if _is_event(h)), None)
                if best_event:
                    chosen[-1] = best_event
            return self._dedupe(chosen)[:k]
        return rerank_input[:k]

    @staticmethod
    def _dedupe(hits: list[Hit]) -> list[Hit]:
        seen: set[str] = set()
        out = []
        for h in hits:
            if h.evidence_id not in seen:
                seen.add(h.evidence_id)
                out.append(h)
        return out

    # -- conversions -----------------------------------------------------
    def _hit_from_point(self, point, score_attr: str) -> Hit:
        p = point.payload or {}
        try:
            extra = json.loads(p.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            extra = {}
        return Hit(
            evidence_id=p.get("evidence_id", ""),
            score=round(float(point.score), 4),
            doc_type=p.get("doc_type", ""),
            text=p.get("text", ""),
            ts=p.get("ts") or None,
            tool=p.get("tool") or None,
            category=p.get("category") or None,
            decision=p.get("decision") or None,
            metadata=extra,
        )

    def _hit_from_doc(self, evidence_id: str) -> Hit | None:
        d = self._by_id.get(evidence_id)
        if not d:
            return None
        try:
            extra = d.get("metadata") or {}
        except Exception:  # noqa: BLE001
            extra = {}
        return Hit(
            evidence_id=evidence_id,
            score=0.0,
            doc_type=d.get("doc_type", ""),
            text=d.get("text", ""),
            ts=d.get("ts"),
            tool=d.get("tool"),
            category=d.get("category"),
            decision=d.get("decision"),
            metadata=extra,
        )


def _to_qdrant_filter(where: dict[str, Any] | None):
    if not where:
        return None
    must = [
        qm.FieldCondition(key=key, match=qm.MatchValue(value=value)) for key, value in where.items()
    ]
    return qm.Filter(must=must)
