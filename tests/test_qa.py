"""Tests for citation verification and prompt assembly (no network/model)."""

from __future__ import annotations

from guardlens.qa import ABSTAIN, build_user_prompt, verify_citations
from guardlens.retriever import Hit


def _hit(eid: str, text: str = "evidence body") -> Hit:
    return Hit(
        evidence_id=eid,
        score=0.9,
        doc_type="event_card",
        text=text,
        ts="2026-09-06T09:25:46Z",
        tool="bash",
        category="network",
        decision="deny",
        metadata={},
    )


def test_citation_verify_valid_and_invalid():
    text = "curl 被拦截 [^EV-00001]，另有伪造引用 [^EV-09999]。"
    valid, invalid = verify_citations(text, ["EV-00001", "EV-00002"])
    assert valid == ["EV-00001"]
    assert invalid == ["EV-09999"]


def test_build_user_prompt_contains_evidence_and_stats():
    hits = [_hit("EV-00001", "curl -s https://example.com 被网络策略拦截")]
    prompt = build_user_prompt("curl 为什么被拦", hits, stats="deny=1139")
    assert "<evidence id=EV-00001" in prompt
    assert "curl -s https://example.com" in prompt
    assert "<stats>" in prompt and "deny=1139" in prompt
    assert "curl 为什么被拦" in prompt


def test_abstain_constant_consistent():
    assert ABSTAIN in "证据不足时回答：日志中未发现"
