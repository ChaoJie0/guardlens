"""W2 (Chroma dense top-k + quota) vs W3 (BM25+dense RRF + rerank) recall.

Runs the SAME question set against both retrievers and reports, per question:
the top-5 ids under each pipeline and which ground-truth evidence ids are hit
(Recall@5). No LLM calls — purely retrieval, so it is reproducible and fast
(the W3 retriever loads BGE-M3 on MPS and the reranker; W2 reuses Chroma).

Ground truth here is the curated ids established in W2 plus the new exact-term
questions that motivated hybrid retrieval.
"""

from __future__ import annotations

import argparse

from guardlens.hybrid import HybridRetriever
from guardlens.reranker import Reranker
from guardlens.retriever import Retriever

# question -> evidence ids that a correct retrieval should surface
CASES = [
    ("curl 被拦是为什么", ["EV-01202", "EV-00110", "EV-00112", "EV-00111"]),
    # gold here means "evidence that actually records a network deny + tool
    # swap"; several event clusters (CL-*) legitimately summarise those seq.
    (
        "有没有绕过网络限制的尝试",
        ["EV-00887", "CL-0000", "EV-00001", "EV-00890", "CL-0025", "CL-0012", "CL-0057"],
    ),
    ("fail-closed 发生在什么场景", ["EV-00897", "EV-00898", "EV-00900", "EV-00901", "EV-00902"]),
    ("写文件被拦截的原因", ["EV-01128", "EV-00267", "EV-00155"]),
    ("是否使用过 chmod 或 sudo 提权", ["EV-00603", "EV-00584", "EV-00773", "EV-00442"]),
    # new questions that pure vectors handled poorly in W2
    ("node -e fetch 绕过 npm 网络拦截", ["EV-00890", "EV-00887", "CL-0047"]),
    (
        "把 fetch 写到临时 js 文件再 node 执行绕过",
        ["EV-00887", "EV-00888", "EV-00889", "EV-00890", "CL-0047", "CL-0048"],
    ),
    ("curl 被拦后改用 web_fetch 访问同一个网址", ["EV-00001", "EV-00000", "CL-0000"]),
]


def recall(retrieved: list[str], gold: list[str], k: int) -> float:
    got = {g for g in gold if g in set(retrieved[:k])}
    return len(got) / len(gold) if gold else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma", default="data/derived/chroma")
    ap.add_argument("--corpus", default="data/derived/corpus/corpus.jsonl")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    w2 = Retriever(args.chroma)
    w3 = HybridRetriever(args.corpus, reranker=Reranker())

    print(f"{'question':34} {'W2 R@5':>7} {'W3 R@5':>7}  win")
    print("-" * 72)
    sum_w2 = sum_w3 = 0.0
    for question, gold in CASES:
        w2_ids = [h.evidence_id for h in w2.retrieve(question, k=args.k)]
        w3_hits = w3.retrieve(question, k=args.k, rerank=True)
        w3_ids = [h.evidence_id for h in w3_hits]
        r2, r3 = recall(w2_ids, gold, args.k), recall(w3_ids, gold, args.k)
        sum_w2 += r2
        sum_w3 += r3
        win = "W3" if r3 > r2 else ("tie" if r3 == r2 else "W2")
        print(f"{question[:32]:34} {r2:7.2f} {r3:7.2f}  {win}")
        w2_hit = [g for g in gold if g in set(w2_ids)]
        w3_hit = [g for g in gold if g in set(w3_ids)]
        only3 = [g for g in w3_hit if g not in w2_hit]
        print(f"{'  gold hit W2':34} {w2_hit}")
        print(f"{'  gold hit W3':34} {w3_hit}")
        if only3:
            print(f"{'  newly recovered by W3':34} {only3}")
    n = len(CASES)
    print("-" * 72)
    print(f"mean Recall@5  W2={sum_w2 / n:.3f}   W3={sum_w3 / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
