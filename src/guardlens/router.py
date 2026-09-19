"""Query router: classify a question into STATS / EVIDENCE / MIXED (W3).

Deterministic rule-based routing (no extra LLM call). STATS questions are
answered from DuckDB via the guarded SQL/known-stats tool; EVIDENCE questions
go to hybrid retrieval + rerank; MIXED does both and merges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STATS = "STATS"
EVIDENCE = "EVIDENCE"
MIXED = "MIXED"

# Explicit numeric/aggregation intent.
_STATS_CUES = [
    "多少",
    "几起",
    "几次",
    "几个",
    "数量",
    "总数",
    "占比",
    "比例",
    "率",
    "统计",
    "分布",
    "排名",
    "top",
    "次数",
    "百分比",
    "合计",
    "平均",
]
# Asking for a concrete event / reason / sequence -> evidence.
_EVIDENCE_CUES = [
    "为什么",
    "原因",
    "怎么回事",
    "哪些事件",
    "具体",
    "举例",
    "证据",
    "引用",
    "绕过",
    "尝试",
    "场景",
    "命令是什么",
    "发生了什么",
    "序列",
    "如何被",
    "是不是",
    "有没有",
    "是否",
    "举例说明",
]
# Questions that clearly want a number AND an explanation.
_MIXED_PAIRS = [
    ("拦截", "为什么"),
    ("拦截率", "什么"),
    ("几起", "为什么"),
    ("多少", "原因"),
]


@dataclass(slots=True)
class Route:
    kind: str
    wants_number: bool
    wants_evidence: bool


def classify(question: str) -> Route:
    q = question.strip()
    low = q.lower()

    has_number = any(c in q for c in _STATS_CUES) or bool(re.search(r"\d+\s*(起|次|个|%)", q))
    has_evidence = any(c in q for c in _EVIDENCE_CUES)
    # A bare "有没有/是否" with no numeric cue is evidence lookup.
    if re.search(r"(fail-closed|chmod|sudo|curl|web_fetch|绕过)", low) and not has_number:
        has_evidence = True

    mixed = has_number and has_evidence
    if not mixed:
        for a, b in _MIXED_PAIRS:
            if a in q and b in q:
                mixed = True
                break

    if mixed:
        return Route(MIXED, wants_number=True, wants_evidence=True)
    if has_number:
        return Route(STATS, wants_number=True, wants_evidence=False)
    # default: evidence (the dominant RAG use case)
    return Route(EVIDENCE, wants_number=False, wants_evidence=True)
