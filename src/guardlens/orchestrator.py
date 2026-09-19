"""W3 thin orchestration: route -> stats(SQL) / evidence(hybrid) / mixed.

Deterministic router first. STATS questions answer from DuckDB through the
guarded read-only SQL tool and the numbers are passed to the LLM verbatim
(the model only phrases them). EVIDENCE questions go through hybrid retrieval
+ cross-encoder rerank. MIXED does both.

Citation hardening (W3): every cited evidence id must (a) exist in the corpus
and (b) be in the retrieved set for this question; otherwise the answer is
flagged and the fabricated id is stripped from the valid list.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openai import OpenAI

from .hybrid import HybridRetriever
from .llm_config import LLMConfig, resolve_config
from .qa import (
    ABSTAIN,
    SYSTEM_PROMPT,
    Answer,
    _extra_body,
    build_user_prompt,
    verify_citations,
)
from .router import EVIDENCE, MIXED, STATS, classify
from .sql_tool import known_stats

CORPUS_DEFAULT = "data/derived/corpus/corpus.jsonl"
DB_DEFAULT = "data/derived/guardlens.duckdb"


def load_existing_ids(corpus_path: str | Path = CORPUS_DEFAULT) -> set[str]:
    ids: set[str] = set()
    with Path(corpus_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["evidence_id"])
    return ids


class Orchestrator:
    def __init__(
        self,
        *,
        corpus_path: str | Path = CORPUS_DEFAULT,
        db_path: str | Path = DB_DEFAULT,
        retriever: HybridRetriever | None = None,
        config: LLMConfig | None = None,
        k: int = 5,
    ) -> None:
        self.corpus_path = corpus_path
        self.db_path = db_path
        self.retriever = retriever or HybridRetriever(corpus_path)
        self.cfg = config or resolve_config()
        self.k = k
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url, timeout=90)
        self.existing_ids = load_existing_ids(corpus_path)

    # -- routing ----------------------------------------------------------
    def route(self, question: str):
        return classify(question)

    def _stats_text(self, question: str) -> str | None:
        return known_stats(question, self.db_path)

    def _evidence_hits(self, question: str):
        return self.retriever.retrieve(question, k=self.k, rerank=True)

    # -- non-streaming ----------------------------------------------------
    def answer(self, question: str) -> Answer:
        route = self.route(question)
        stats = None
        hits = []
        if route.kind in (STATS, MIXED):
            stats = self._stats_text(question)
        if route.kind in (EVIDENCE, MIXED):
            hits = self._evidence_hits(question)

        # Pure STATS with a deterministic number and no evidence need: short
        # LLM phrasing over the exact figure, or return the number directly.
        user = build_user_prompt(question, hits, stats=stats)
        if route.kind == STATS and stats and not hits:
            user += (
                "\n\n<instruction>这是 STATS 类问题：必须原样使用上面 <stats> 的数字，"
                "用一句话组织语言，不要编造或补充 <stats> 之外的数值。</instruction>"
            )
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            extra_body=_extra_body(self.cfg),
        )
        text = resp.choices[0].message.content or ""
        return self._finalize(text, hits, route.kind, stats)

    # -- streaming --------------------------------------------------------
    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        route = self.route(question)
        yield {"type": "route", "route": route.kind}

        stats = None
        hits: list = []
        if route.kind in (STATS, MIXED):
            stats = self._stats_text(question)
            yield {"type": "stats", "stats": stats}
        if route.kind in (EVIDENCE, MIXED):
            hits = self._evidence_hits(question)
            for h in hits:
                yield {"type": "hit", "hit": h}

        user = build_user_prompt(question, hits, stats=stats)
        if route.kind == STATS and stats and not hits:
            user += (
                "\n\n<instruction>STATS 类：原样使用 <stats> 数字组织一句话，"
                "不要编造其它数值。</instruction>"
            )
        stream = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            stream=True,
            extra_body=_extra_body(self.cfg),
        )
        buf: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                buf.append(delta)
                yield {"type": "token", "delta": delta}
        yield {"type": "final", "answer": self._finalize("".join(buf), hits, route.kind, stats)}

    # -- finalize with hardened citation checks --------------------------
    def _finalize(self, text: str, hits, route_kind: str, stats: str | None) -> Answer:
        retrieved = [h.evidence_id for h in hits]
        valid, invalid = verify_citations(text, retrieved)
        # second gate: id must globally exist in the corpus (paranoia vs a
        # well-formed-but-fabricated id); retrieved membership already implies
        # this, but keep both checks explicit and observable.
        fabricated = [c for c in valid if c not in self.existing_ids]
        if fabricated:
            invalid.extend(fabricated)
            valid = [c for c in valid if c not in set(fabricated)]

        head = text.lstrip()[:40]
        # STATS answers are number-grounded, not citation-grounded; only treat
        # as abstention for evidence-seeking routes with no usable evidence.
        abstained = (
            route_kind != STATS
            and ABSTAIN in head
            and not valid
            and not (route_kind == MIXED and stats)
        )
        return Answer(
            text=text,
            citations=sorted(set(valid + invalid)),
            valid_citations=valid,
            invalid_citations=sorted(set(invalid)),
            abstained=abstained,
            retrieved_ids=retrieved,
        )
