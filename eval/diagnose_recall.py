"""Diagnose golden-set Recall@5 against the current corpus.

For each golden case: confirm the gold ids still exist in the corpus, run the
full hybrid+rerank retrieval, and show where the gold documents land (top-5 /
top-10 / missed) plus their doc types — so recall problems can be separated
from ranking problems.

Usage:
    uv run python -m eval.diagnose_recall
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "data/derived/corpus/corpus.jsonl"

from eval.run_eval import load_golden  # noqa: E402
from guardlens.hybrid import HybridRetriever  # noqa: E402


def main() -> None:
    docs_by_id: dict[str, dict] = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                doc = json.loads(line)
                docs_by_id[doc["evidence_id"]] = doc

    print(f"corpus documents: {len(docs_by_id)}")
    print("doc types:", dict(Counter(d["doc_type"] for d in docs_by_id.values())))

    r = HybridRetriever(str(CORPUS))
    cases = load_golden()

    n_hit5 = 0
    n_gold = 0
    for c in cases:
        if not c.gold:
            continue
        golds = c.gold
        n_gold += len(golds)
        gtypes = Counter(docs_by_id[g]["doc_type"] for g in golds if g in docs_by_id)
        missing = [g for g in golds if g not in docs_by_id]
        hits = r.retrieve(c.q, k=10, rerank=True)
        top = [h.evidence_id for h in hits]
        in5 = [g for g in golds if g in top[:5]]
        in10 = [g for g in golds if g in top]
        n_hit5 += len(in5)
        print(f"\n=== {c.id} [{c.type}] gold={golds}")
        print(f"    gold types={dict(gtypes)} missing={missing}")
        print(f"    hit@5={in5} hit@10={in10} miss={[g for g in golds if g not in top]}")
        for h in hits[:6]:
            text = h.text.replace("\n", " ")[:90]
            print(f"      [{h.doc_type}] {h.evidence_id} s={h.score:.3f} {text}")

    if n_gold:
        print(f"\n===== gold hit {n_hit5}/{n_gold} = {n_hit5 / n_gold:.1%} (top-5) =====")


if __name__ == "__main__":
    main()
