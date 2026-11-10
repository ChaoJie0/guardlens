"""Stage-by-stage trace: where does each gold id drop out of retrieval?

Traces gold ids through: dense raw top-N -> type-balanced -> sparse raw top-N
-> type-balanced -> intent-expansion queries -> RRF fused pool -> rerank top-k.
Prints a D/B/R/5 marker per gold id so the exact break point is visible.

Usage:
    uv run python -m eval.trace_recall
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "data/derived/corpus/corpus.jsonl"

from eval.run_eval import load_golden  # noqa: E402
from guardlens.hybrid import (  # noqa: E402
    DENSE_N,
    FUSE_N,
    SPARSE_N,
    HybridRetriever,
    _expand,
    rrf_fuse,
)

EVENT_TYPES = ("event_card", "event_cluster")


def _balanced(hits, per_type: int):
    """Same per-type quota the retriever applies before fusion."""
    events = [h for h in hits if h.doc_type in EVENT_TYPES][:per_type]
    others = [h for h in hits if h.doc_type not in EVENT_TYPES][:per_type]
    return events + others


def main() -> None:
    r = HybridRetriever(str(CORPUS))
    cases = load_golden()
    total, hit5 = 0, 0

    for c in cases:
        if not c.gold:
            continue
        golds = set(c.gold)
        total += len(golds)
        q = c.q

        dense_raw = r.dense_search(q, DENSE_N * 2)
        dense_bal = _balanced(dense_raw, FUSE_N)
        sparse_raw = r.sparse_search(q, SPARSE_N * 2)
        sparse_bal = _balanced(sparse_raw, FUSE_N)

        ranked_lists: list[list[str]] = []
        weights: list[float] = []
        all_hits: list = []
        channels: list[tuple[list, float]] = [(dense_raw, 1.0), (sparse_raw, 1.0)]
        for eq in _expand(q):
            channels.append((r.dense_search(eq, DENSE_N * 2), 0.6))
            channels.append((r.sparse_search(eq, SPARSE_N * 2), 0.6))
        for hits, weight in channels:
            bal = _balanced(hits, FUSE_N)
            ranked_lists.append([h.evidence_id for h in bal])
            weights.append(weight)
            all_hits.extend(bal)

        fused = rrf_fuse(ranked_lists, weights=weights, top_n=FUSE_N)
        fused_ids = {eid for eid, _ in fused}

        anchor = r.structured_anchor(q)
        anchor_ids = {h.evidence_id for h in anchor}

        pool_ids = set(anchor_ids) | {eid for eid, _ in fused}
        in_pool = [h for h in all_hits + anchor if h.evidence_id in pool_ids]
        non_anchor = [h for h in in_pool if h.evidence_id not in anchor_ids]
        non_anchor = non_anchor[: FUSE_N - len(anchor)]
        rerank_input = list(anchor) + non_anchor
        ranked = r.reranker.rerank(q, rerank_input, top_k=FUSE_N) if r.reranker else rerank_input
        top5 = [h.evidence_id for h in ranked[:5]]

        dense_ids = {h.evidence_id for h in dense_raw}
        sparse_ids = {h.evidence_id for h in sparse_raw}

        def _mark(gid: str, d=dense_ids, s=sparse_ids, f=fused_ids, t=top5) -> str:
            return "".join(
                [
                    "D" if gid in d else "-",
                    "B" if gid in s else "-",
                    "R" if gid in f else "-",
                    "5" if gid in t else "-",
                ]
            )

        hit5 += len([g for g in golds if g in top5])
        print(f"\n=== {c.id} [{c.type}] {q}")
        print(
            f"    dense_raw={len(dense_raw)} dense_bal={len(dense_bal)} "
            f"sparse_raw={len(sparse_raw)} sparse_bal={len(sparse_bal)} "
            f"fused={len(fused)} anchor={len(anchor)}"
        )
        for gid in sorted(golds):
            print(f"    {gid}: dense/sparse/fused/top5 = {_mark(gid)}")
        print(f"    top5={top5[:5]}")
        for gid in sorted(golds):
            if gid in top5:
                continue
            for rank, h in enumerate(dense_raw):
                if h.evidence_id == gid:
                    print(f"      [{gid}] dense_raw rank={rank} score={h.score:.3f}")
                    break

    print(f"\n===== top5 recall {hit5}/{total} = {hit5 / total:.1%} =====")


if __name__ == "__main__":
    main()
