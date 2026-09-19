"""W4 eval-harness tests with a tiny corpus, a fake retriever and a fake LLM.

No BGE-M3, no Qdrant, no reranker, no network: the LLM runner is injected and
returns canned answers, so the metric logic (recall, citation accuracy,
abstention, number accuracy, ablation grid, report rendering) is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eval.run_eval import (
    GoldenCase,
    _normalize_lead,
    _retrieval_recall,
    evaluate_case,
    load_golden,
    render_report,
    retrieve,
    run_ablation,
    summarize,
    validate_golden,
)

from guardlens.retriever import Hit

# --------------------------------------------------------------------------
# tiny corpus-backed retriever with injected dense fn (mirrors W3 test style)
# --------------------------------------------------------------------------


def _hit(eid: str, doc_type: str = "event_card", score: float = 0.5) -> Hit:
    return Hit(
        evidence_id=eid,
        score=score,
        doc_type=doc_type,
        text=f"[证据 {eid}] 测试事件卡\n命令: curl https://example.test 被拦截",
        ts="2026-09-01T00:00:00+00:00",
        tool="bash",
        category="network",
        decision="deny",
        metadata={},
    )


class _FakeRetriever:
    """dense returns a fixed order; retrieve mirrors that order (BM25/rrf/rerank
    are exercised in W3 tests — here we only need a controllable ranking)."""

    def __init__(self, order: list[str]):
        self._order = order

    def dense_search(self, question, n, *, where=None):
        return [_hit(e) for e in self._order[:n]]

    def retrieve(self, question, k=5, *, rerank=True, anchors=True):
        return [_hit(e, score=0.9 - 0.01 * i) for i, e in enumerate(self._order[:k])]


def _runner(text: str, prompt_tokens: float = 100.0, completion_tokens: float = 20.0):
    def _run(system: str, user: str):
        return (
            text,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            0.5,
        )

    return _run


def _case(**kw) -> GoldenCase:
    base = dict(id="t1", q="curl 被拦是为什么？", type="EVIDENCE", gold=["EV-0001"], expect=[])
    base.update(kw)
    return GoldenCase(**base)


# --------------------------------------------------------------------------
# metric primitives
# --------------------------------------------------------------------------


def test_recall_fraction_and_empty_gold():
    r, hit = _retrieval_recall(["a", "b"], ["x", "b", "z", "a"], k=5)
    assert r == 1.0 and set(hit) == {"a", "b"}
    r, hit = _retrieval_recall(["a", "b", "c", "d"], ["b"], k=5)
    assert r == 0.25
    r, _ = _retrieval_recall([], ["whatever"], k=5)
    assert r == 1.0  # STATS question is not penalised


def test_normalize_lead_strips_fillers():
    assert _normalize_lead("结论：日志中未发现 X").startswith("日志中未发现")
    assert _normalize_lead("  根据证据，curl 被拦").startswith("curl")
    assert not _normalize_lead("存在绕过").startswith("日志中未发现")


# --------------------------------------------------------------------------
# end-to-end case evaluation with fake LLM
# --------------------------------------------------------------------------


def test_answerable_case_valid_citation_counts():
    retr = _FakeRetriever(["EV-0001", "EV-0002"])
    ans = "curl 因外联被拦 [^EV-0001]。"
    r = evaluate_case(
        _case(gold=["EV-0001"]),
        retr,
        _runner(ans),
        db_path=":memory:",
        existing_ids={"EV-0001", "EV-0002"},
    )
    assert r.recall_at_5 == 1.0
    assert r.citation_ok and r.valid_citations == ["EV-0001"]
    assert r.abstention_ok and not r.abstained


def test_fabricated_citation_is_invalid():
    retr = _FakeRetriever(["EV-0001"])
    ans = "据记录发生了某事 [^EV-0999]。"  # 0999 not retrieved nor existing
    r = evaluate_case(
        _case(gold=["EV-0001"]),
        retr,
        _runner(ans),
        db_path=":memory:",
        existing_ids={"EV-0001"},
    )
    assert not r.citation_ok and "EV-0999" in r.invalid_citations


def test_unanswerable_case_must_abstain():
    retr = _FakeRetriever(["EV-0001"])
    good = evaluate_case(
        _case(gold=[], unanswerable=True),
        retr,
        _runner("结论：日志中未发现反弹 shell。"),
        db_path=":memory:",
        existing_ids={"EV-0001"},
    )
    assert good.abstention_ok and good.abstained
    # model hallucinates an event -> abstention fails
    bad = evaluate_case(
        _case(gold=[], unanswerable=True),
        retr,
        _runner("发现反弹 shell [^EV-0001]。"),
        db_path=":memory:",
        existing_ids={"EV-0001"},
    )
    assert not bad.abstention_ok and not bad.abstained


def test_answerable_case_must_not_falsely_abstain():
    retr = _FakeRetriever(["EV-0001"])
    r = evaluate_case(
        _case(gold=["EV-0001"]),
        retr,
        _runner("日志中未发现相关事件。"),
        db_path=":memory:",
        existing_ids={"EV-0001"},
    )
    assert not r.abstention_ok  # false abstention on an answerable question


def test_nuanced_answer_with_citation_is_not_false_abstention():
    # answerable yes/no question: negates one aspect but still cites real
    # evidence -> not a pure abstention, must not be marked wrong.
    retr = _FakeRetriever(["EV-0001", "EV-0002"])
    ans = "结论：日志中未发现成功绕过的记录；但尝试均被拦截 [^EV-0001]。"
    r = evaluate_case(
        _case(gold=["EV-0001"]),
        retr,
        _runner(ans),
        db_path=":memory:",
        existing_ids={"EV-0001", "EV-0002"},
    )
    assert r.abstained is False
    assert r.abstention_ok is True
    assert r.citation_ok is True


def test_stats_number_accuracy(tmp_path):
    # STATS route: known_stats needs a real DB; use the shipped DB if present,
    # otherwise just assert the number-match logic via a number-bearing MIXED
    # case is unnecessary — exercise the pure string gate:
    retr = _FakeRetriever([])
    r = evaluate_case(
        _case(type="EVIDENCE", gold=[], number="42"),
        retr,
        _runner("答案里没有那个数字。"),
        db_path=":memory:",
        existing_ids=set(),
    )
    # EVIDENCE with no gold -> recall 1.0, number missing -> False
    assert r.number_ok is False and r.recall_at_5 == 1.0
    r2 = evaluate_case(
        _case(type="EVIDENCE", gold=[], number="42"),
        retr,
        _runner("数值是 42 次。"),
        db_path=":memory:",
        existing_ids=set(),
    )
    assert r2.number_ok is True


# --------------------------------------------------------------------------
# ablation grid uses the retrieve() dispatcher
# --------------------------------------------------------------------------


def test_retrieve_modes_dispatch():
    retr = _FakeRetriever(["EV-0001", "EV-0002", "EV-0003"])
    assert [h.evidence_id for h in retrieve(retr, "q", "dense", 2)] == ["EV-0001", "EV-0002"]
    assert [h.evidence_id for h in retrieve(retr, "q", "hybrid", 2)] == ["EV-0001", "EV-0002"]
    assert [h.evidence_id for h in retrieve(retr, "q", "hybrid_rerank", 2)] == [
        "EV-0001",
        "EV-0002",
    ]
    with pytest.raises(ValueError):
        retrieve(retr, "q", "bogus", 2)


def test_ablation_grid_and_summary(tmp_path):
    retr = _FakeRetriever(["EV-0001", "EV-0002"])
    cases = [_case(gold=["EV-0001"]), _case(gold=["EV-0099"])]
    out = run_ablation(cases, retr, modes=("dense", "hybrid_rerank"), ks=(3, 5))
    assert len(out["grid"]) == 4
    for g in out["grid"]:
        assert g["n"] == 2  # STATS/empty-gold cases are skipped
        assert 0.0 <= g["recall_at5_mean"] <= 1.0

    # summarize over canned CaseResults
    results = [
        evaluate_case(
            _case(gold=["EV-0001"]),
            retr,
            _runner("ok [^EV-0001]"),
            db_path=":memory:",
            existing_ids={"EV-0001", "EV-0002"},
        ),
        evaluate_case(
            _case(id="t2", gold=[], unanswerable=True),
            retr,
            _runner("日志中未发现。"),
            db_path=":memory:",
            existing_ids={"EV-0001", "EV-0002"},
        ),
    ]
    s = summarize(results)
    assert s["citation_accuracy_pct"] == 100.0
    assert s["abstention_accuracy_pct"] == 100.0
    assert s["latency_p50_s"] == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------------------
# golden YAML + validation + report render
# --------------------------------------------------------------------------


def test_validate_golden_flags_problems():
    cases = [
        _case(id="ok", q="拦截率是多少？", type="STATS", gold=[]),
        _case(id="bad", q="拦截率是多少？", type="EVIDENCE", gold=["EV-404"]),
        _case(id="u", q="有没有挖矿？", type="EVIDENCE", gold=["EV-1"], unanswerable=True),
    ]
    problems = validate_golden(cases, {"EV-1"})
    assert any("bad" in p and "router" in p for p in problems)
    assert any("bad" in p and "EV-404" in p for p in problems)
    assert any("unanswerable" in p for p in problems)  # gold must be empty
    assert any("exactly 2 unanswerable" in p for p in problems)


def test_shipped_golden_loads_and_is_well_formed():
    path = Path(__file__).resolve().parents[1] / "eval" / "golden20.yaml"
    if not path.exists():
        pytest.skip("golden20.yaml not present")
    cases = load_golden(path)
    assert len(cases) == 20
    assert sum(c.unanswerable for c in cases) == 2
    kinds = {c.type for c in cases}
    assert kinds <= {"STATS", "EVIDENCE", "MIXED"}
    # ids unique, every non-empty gold has at least one id
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_render_report_contains_sections_and_no_pii_placeholder():
    cases = [_case()]
    retr = _FakeRetriever(["EV-0001"])
    results = [
        evaluate_case(
            _case(), retr, _runner("ok [^EV-0001]"), db_path=":memory:", existing_ids={"EV-0001"}
        ),
    ]
    s = summarize(results)
    abl = run_ablation(cases, retr, modes=("dense", "hybrid_rerank"), ks=(5,))
    md = render_report(cases, results, s, abl, model="m")
    assert "Recall@5" in md and "引用正确率" in md and "拒答正确率" in md
    assert "dense" in md and "hybrid_rerank" in md
    # report must not leak a real home path
    assert "/Users/" not in md
