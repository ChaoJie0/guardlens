"""Evidence-grounded Q&A over the audit corpus.

Contract enforced in the prompt and verified after generation:

* every factual claim cites an evidence id present in the retrieved set;
* statistics must come from provided <stats> (caller computes via DuckDB),
  the model must not invent numbers;
* insufficient evidence -> a fixed abstention phrase.

W2 answer format is Markdown with ``[^EV-00001]``-style citations; a
post-check verifies every citation id is real and retrieved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from .llm_config import LLMConfig, resolve_config
from .retriever import Hit, Retriever

ABSTAIN = "日志中未发现"
CITATION_RE = re.compile(r"\[\^(EV|CL|RP)-\d{4,5}\]")

SYSTEM_PROMPT = """你是一名 AI Agent 安全运营分析师，只基于提供的审计证据回答问题。
严格遵守：
1. 每个事实结论后必须用 [^证据ID] 引用，ID 只能来自 <context> 中实际给出的证据。
2. 数值类问题以 <stats> 为准，禁止估算或编造数字；<stats> 没有就不要给数字。
3. 证据不足以回答时，只能回答“日志中未发现”，并说明缺什么证据。
4. 引用原始命令/理由时可摘录，但要保留证据 ID；不要臆造命令、URL、时间。
5. 用简洁中文，先给结论再给依据。最后另起一行列出“引用：”并去重排列使用到的 ID。
"""


def build_user_prompt(
    question: str,
    hits: list[Hit],
    *,
    stats: str | None = None,
) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"<evidence id={h.evidence_id} type={h.doc_type} "
            f"ts={h.ts or ''} tool={h.tool or ''} decision={h.decision or ''}>\n"
            f"{h.text}\n</evidence>"
        )
    context = "\n\n".join(blocks)
    stats_block = f"<stats>\n{stats}\n</stats>\n\n" if stats else ""
    return f"{stats_block}<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>"


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    abstained: bool
    retrieved_ids: list[str]


def verify_citations(text: str, retrieved_ids: list[str]) -> tuple[list[str], list[str]]:
    raw = [m.group(0) for m in CITATION_RE.finditer(text)]
    # normalise [^EV-00001] -> EV-00001
    cited_ids = sorted({c.strip("[^]") for c in raw})
    allowed = set(retrieved_ids)
    valid = [c for c in cited_ids if c in allowed]
    invalid = [c for c in cited_ids if c not in allowed]
    return valid, invalid


def _extra_body(cfg: LLMConfig) -> dict[str, Any] | None:
    if cfg.disable_thinking:
        return {"thinking": {"type": "disabled"}}
    return None


class QAEngine:
    def __init__(
        self,
        retriever: Retriever,
        *,
        config: LLMConfig | None = None,
        k: int = 6,
    ) -> None:
        self.retriever = retriever
        self.cfg = config or resolve_config()
        self.k = k
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url, timeout=60)

    def _retrieve(self, question: str) -> list[Hit]:
        return self.retriever.retrieve(question, k=self.k)

    def answer(
        self,
        question: str,
        *,
        stats: str | None = None,
        hits: list[Hit] | None = None,
    ) -> Answer:
        hits = hits or self._retrieve(question)
        user = build_user_prompt(question, hits, stats=stats)
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
        return self._finalize(text, hits)

    def stream(
        self,
        question: str,
        *,
        stats: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield {type} events: 'hit' (retrieved evidence) then token deltas."""
        hits = self._retrieve(question)
        for h in hits:
            yield {"type": "hit", "hit": h}
        user = build_user_prompt(question, hits, stats=stats)
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
        buf = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                buf.append(delta)
                yield {"type": "token", "delta": delta}
        text = "".join(buf)
        final = self._finalize(text, hits)
        yield {"type": "final", "answer": final}

    def _finalize(self, text: str, hits: list[Hit]) -> Answer:
        ids = [h.evidence_id for h in hits]
        valid, invalid = verify_citations(text, ids)
        # A "日志中未发现" clause is normal when negating one sub-question
        # (e.g. "有 chmod，但未发现 sudo"). Count abstention only when the
        # answer leads with the phrase AND cites no valid evidence.
        head = text.lstrip()[:40]
        abstained = ABSTAIN in head and not valid
        return Answer(
            text=text,
            citations=sorted(set(valid + invalid)),
            valid_citations=valid,
            invalid_citations=invalid,
            abstained=abstained,
            retrieved_ids=ids,
        )


def hits_to_evidence_jsonl(hits: list[Hit]) -> str:
    return json.dumps(
        [
            {
                "evidence_id": h.evidence_id,
                "ts": h.ts,
                "tool": h.tool,
                "decision": h.decision,
                "category": h.category,
                "text": h.text,
            }
            for h in hits
        ],
        ensure_ascii=False,
        indent=2,
    )


def make_engine(
    persist_dir: str | Path,
    *,
    k: int = 6,
    config: LLMConfig | None = None,
) -> QAEngine:
    return QAEngine(Retriever(persist_dir), k=k, config=config)
