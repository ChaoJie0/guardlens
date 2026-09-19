"""W4 offline evaluation for GuardLens.

Runs the golden set (eval/golden20.yaml) through the same router → guarded
SQL / hybrid retrieval → LLM pipeline used in production and reports:

  * Recall@5            gold evidence ids retrieved in the top-5
  * citation accuracy   every cited id exists AND is in the retrieved set
  * abstention accuracy unanswerable questions answered with “日志中未发现”
                         (answerable questions must NOT abstain)
  * number accuracy     deterministic STATS/MIXED figures present verbatim
  * latency             p50/p95 end-to-end wall time
  * token cost          prompt/completion tokens per question

It also runs a retrieval ABLATION matrix (dense vs hybrid vs hybrid+rerank,
K in 3/5/8, rerank on/off) at the retrieval layer (no LLM), so the report can
show how much hybrid retrieval + rerank improve Recall over W2 pure vectors.

Everything is local: the LLM is whatever endpoint llm_config.resolve_config()
returns; no data leaves the machine except the prompt to your own configured
model. Usage:

    uv run python -m eval.run_eval                 # full E2E + ablation
    uv run python -m eval.run_eval --ablation-only # retrieval only, no LLM
    uv run python -m eval.run_eval --report eval/eval_report.md

The script is import-safe: build_retriever / evaluate_case / run_* are
functions with injectable retriever/runner so tests use a fake embedder and a
fake LLM with no Qdrant server and no 2GB model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from guardlens.hybrid import DEFAULT_TOP_K, HybridRetriever
from guardlens.llm_config import LLMConfig, resolve_config
from guardlens.qa import SYSTEM_PROMPT, _extra_body, build_user_prompt, verify_citations
from guardlens.router import EVIDENCE, MIXED, STATS, classify
from guardlens.sql_tool import known_stats

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DEFAULT = ROOT / "eval" / "golden20.yaml"
RESULTS_DEFAULT = ROOT / "eval" / "results.json"
CORPUS_DEFAULT = str(ROOT / "data/derived/corpus/corpus.jsonl")
DB_DEFAULT = str(ROOT / "data/derived/guardlens.duckdb")
ABSTAIN = "日志中未发现"

RETRIEVAL_MODES = ("dense", "hybrid", "hybrid_rerank")
KS = (3, 5, 8)


# ---------------------------------------------------------------------------
# golden set loading / validation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GoldenCase:
    id: str
    q: str
    type: str
    gold: list[str]
    expect: list[str]
    number: str | None = None
    unanswerable: bool = False


def load_golden(path: str | Path = GOLDEN_DEFAULT) -> list[GoldenCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cases: list[GoldenCase] = []
    for item in raw:
        cases.append(
            GoldenCase(
                id=item["id"],
                q=item["q"],
                type=item["type"],
                gold=list(item.get("gold") or []),
                expect=list(item.get("expect") or []),
                number=item.get("number"),
                unanswerable=bool(item.get("unanswerable", False)),
            )
        )
    return cases


def validate_golden(cases: list[GoldenCase], existing_ids: set[str]) -> list[str]:
    """Return human-readable problems (unknown gold id / type mismatch)."""
    problems: list[str] = []
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.type] = counts.get(c.type, 0) + 1
        routed = classify(c.q).kind
        if routed != c.type:
            problems.append(f"{c.id}: golden type {c.type} but router says {routed}")
        if c.unanswerable and c.gold:
            problems.append(f"{c.id}: unanswerable cases must have empty gold")
        for gid in c.gold:
            if gid not in existing_ids:
                problems.append(f"{c.id}: gold id {gid} not in corpus")
    n_un = sum(c.unanswerable for c in cases)
    if n_un != 2:
        problems.append(f"expected exactly 2 unanswerable cases, found {n_un}")
    return problems


# ---------------------------------------------------------------------------
# retrieval policies (the ablation axis)
# ---------------------------------------------------------------------------


def retrieve(retriever: HybridRetriever, question: str, mode: str, k: int):
    """Return top-k hits for one retrieval mode."""
    if mode == "dense":
        # W2-style pure dense vector retrieval (no BM25, no rerank, no anchors).
        return retriever.dense_search(question, k)[:k]
    if mode == "hybrid":
        return retriever.retrieve(question, k=k, rerank=False)
    if mode == "hybrid_rerank":
        return retriever.retrieve(question, k=k, rerank=True)
    raise ValueError(f"unknown retrieval mode {mode!r}")


def build_retriever(corpus_path: str = CORPUS_DEFAULT, **kw: Any) -> HybridRetriever:
    return HybridRetriever(corpus_path, **kw)


# ---------------------------------------------------------------------------
# LLM runner (injectable for tests)
# ---------------------------------------------------------------------------

# A runner takes (system, user, config) and returns (text, usage_dict, seconds).
LLMRunner = Callable[[str, str], tuple[str, dict[str, float], float]]


def make_live_runner(cfg: LLMConfig | None = None) -> LLMRunner:
    from openai import OpenAI

    config = cfg or resolve_config()
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=90)

    def _run(system: str, user: str) -> tuple[str, dict[str, float], float]:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            extra_body=_extra_body(config),
        )
        dt = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = {
            "prompt_tokens": float(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": float(getattr(u, "completion_tokens", 0) or 0),
            "total_tokens": float(getattr(u, "total_tokens", 0) or 0),
        }
        return text, usage, dt

    return _run


# ---------------------------------------------------------------------------
# per-case E2E evaluation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CaseResult:
    id: str
    type: str
    route: str
    recall_at_5: float
    gold_hit: list[str]
    cited: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    citation_ok: bool
    abstained: bool
    abstention_ok: bool
    number_ok: bool | None
    latency_s: float
    prompt_tokens: float
    completion_tokens: float
    retrieved_ids: list[str] = field(default_factory=list)


_LEAD_FILLERS = (
    "结论：",
    "结论:",
    "答：",
    "答:",
    "回答：",
    "回答:",
    "根据证据，",
    "根据日志，",
    "根据提供的证据，",
    "经查，",
    "经核查，",
)


def _normalize_lead(text: str, window: int = 80) -> str:
    """Return the answer start with short connective fillers stripped."""
    lead = text.lstrip()[:window].lstrip("　 \t>*-#")
    changed = True
    while changed:
        changed = False
        for f in _LEAD_FILLERS:
            if lead.startswith(f):
                lead = lead[len(f) :].lstrip("　 \t")
                changed = True
    return lead


def _retrieval_recall(gold: list[str], retrieved: list[str], k: int = 5) -> tuple[float, list[str]]:
    if not gold:
        return 1.0, []  # non-evidence question: not penalised
    topk = retrieved[:k]
    hit = [g for g in gold if g in topk]
    return len(hit) / len(gold), hit


def evaluate_case(
    case: GoldenCase,
    retriever: HybridRetriever,
    runner: LLMRunner,
    *,
    db_path: str = DB_DEFAULT,
    mode: str = "hybrid_rerank",
    k: int = DEFAULT_TOP_K,
    existing_ids: set[str] | None = None,
) -> CaseResult:
    route = classify(case.q)
    stats = None
    hits: list = []
    if route.kind in (STATS, MIXED):
        stats = known_stats(case.q, db_path)
    if route.kind in (EVIDENCE, MIXED):
        hits = retrieve(retriever, case.q, mode, k)
    retrieved_ids = [h.evidence_id for h in hits]

    user = build_user_prompt(case.q, hits, stats=stats)
    if route.kind == STATS:
        user += (
            "\n\n<instruction>STATS 类：原样使用 <stats> 数字组织一句话，"
            "不要编造其它数值。</instruction>"
        )

    text, usage, latency = runner(SYSTEM_PROMPT, user)

    valid, invalid = verify_citations(text, retrieved_ids)
    if existing_ids is not None:
        fabricated = [c for c in valid if c not in existing_ids]
        if fabricated:
            invalid = sorted(set(invalid + fabricated))
            valid = [c for c in valid if c not in set(fabricated)]
    cited = sorted(set(valid + invalid))
    # citation accuracy: no invalid citations (STATS with zero citations is fine)
    citation_ok = not invalid

    # Abstention = the answer leads with a not-found statement AND offers no
    # supporting evidence (no valid citations). A nuanced answer that negates
    # only one aspect ("未发现成功绕过…，但尝试均被拦截 [^EV]") still cites real
    # evidence and is therefore NOT a pure abstention. Tolerate a short
    # connective prefix ("结论：") via _normalize_lead.
    lead = _normalize_lead(text)
    states_not_found = lead.startswith(ABSTAIN)
    pure_abstain = states_not_found and not valid
    abstained = route.kind != STATS and pure_abstain and not (route.kind == MIXED and stats)
    if case.unanswerable:
        # correct iff the model explicitly states the evidence is absent in its
        # lead sentence. It may still cite retrieved cards while explaining why
        # they do not match; those citations are real/retrieved.
        abstention_ok = states_not_found
    else:
        # an answerable evidence question must not perform a PURE abstention;
        # nuanced not-found-for-one-aspect answers that still cite evidence are
        # allowed.
        abstention_ok = not abstained if route.kind in (EVIDENCE, MIXED) else True

    number_ok: bool | None = None
    if case.number is not None:
        number_ok = case.number in text

    recall, gold_hit = _retrieval_recall(case.gold, retrieved_ids, k=5)
    return CaseResult(
        id=case.id,
        type=case.type,
        route=route.kind,
        recall_at_5=recall,
        gold_hit=gold_hit,
        cited=cited,
        valid_citations=valid,
        invalid_citations=invalid,
        citation_ok=citation_ok,
        abstained=abstained,
        abstention_ok=abstention_ok,
        number_ok=number_ok,
        latency_s=latency,
        prompt_tokens=usage.get("prompt_tokens", 0.0),
        completion_tokens=usage.get("completion_tokens", 0.0),
        retrieved_ids=retrieved_ids,
    )


# ---------------------------------------------------------------------------
# aggregate + ablation
# ---------------------------------------------------------------------------


def _pct(x: float) -> float:
    return round(100.0 * x, 1)


def run_e2e(
    cases: list[GoldenCase],
    retriever: HybridRetriever,
    runner: LLMRunner,
    *,
    db_path: str = DB_DEFAULT,
    mode: str = "hybrid_rerank",
    k: int = DEFAULT_TOP_K,
    existing_ids: set[str] | None = None,
    retries: int = 3,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                results.append(
                    evaluate_case(
                        case,
                        retriever,
                        runner,
                        db_path=db_path,
                        mode=mode,
                        k=k,
                        existing_ids=existing_ids,
                    )
                )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - network/API errors: wait+retry
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1))
        if last_exc is not None:
            raise RuntimeError(
                f"{case.id} failed after {retries} retries: {last_exc}"
            ) from last_exc
    return results


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results)
    lat = sorted(r.latency_s for r in results)

    def pct(x: float) -> float:
        return round(100.0 * x, 1)

    def quantile(data: list[float], q: float) -> float:
        if not data:
            return 0.0
        idx = min(len(data) - 1, int(round(q * (len(data) - 1))))
        return round(data[idx], 2)

    num_cases = [r for r in results if r.number_ok is not None]
    evidence_results = [r for r in results if r.type != STATS]
    ev_recall = (
        statistics.mean(r.recall_at_5 for r in evidence_results) if evidence_results else 0.0
    )
    return {
        "n": n,
        "recall_at5_mean": round(statistics.mean(r.recall_at_5 for r in results), 4),
        "recall_at5_pct": pct(statistics.mean(r.recall_at_5 for r in results)),
        "recall_at5_evidence_pct": pct(ev_recall),
        "citation_accuracy_pct": pct(sum(r.citation_ok for r in results) / n),
        "abstention_accuracy_pct": pct(sum(r.abstention_ok for r in results) / n),
        "number_accuracy_pct": (
            pct(sum(r.number_ok for r in num_cases) / len(num_cases)) if num_cases else None
        ),
        "latency_p50_s": quantile(lat, 0.50),
        "latency_p95_s": quantile(lat, 0.95),
        "latency_mean_s": round(statistics.mean(lat), 2),
        "prompt_tokens_total": int(sum(r.prompt_tokens for r in results)),
        "completion_tokens_total": int(sum(r.completion_tokens for r in results)),
        "prompt_tokens_mean": round(statistics.mean(r.prompt_tokens for r in results), 1),
        "completion_tokens_mean": round(statistics.mean(r.completion_tokens for r in results), 1),
    }


def run_ablation(
    cases: list[GoldenCase],
    retriever: HybridRetriever,
    *,
    modes: tuple[str, ...] = RETRIEVAL_MODES,
    ks: tuple[int, ...] = KS,
) -> dict[str, Any]:
    """Retrieval-only Recall@5 grid (no LLM). dense ignores k-rerank by design."""
    grid: list[dict[str, Any]] = []
    for mode in modes:
        for k in ks:
            rows = []
            for case in cases:
                if case.type == STATS or not case.gold:
                    continue  # retrieval recall only meaningful for gold ids
                hits = retrieve(retriever, case.q, mode, k)
                ids = [h.evidence_id for h in hits]
                recall, _ = _retrieval_recall(case.gold, ids, k=5)
                rows.append({"id": case.id, "recall_at5": recall, "topk": ids})
            mean = round(statistics.mean(r["recall_at5"] for r in rows), 4) if rows else 0.0
            grid.append(
                {
                    "mode": mode,
                    "k": k,
                    "n": len(rows),
                    "recall_at5_mean": mean,
                    "recall_at5_pct": _pct(mean),
                    "rows": rows,
                }
            )
    return {"grid": grid}


# ---------------------------------------------------------------------------
# report rendering (markdown) — only aggregate / redacted content
# ---------------------------------------------------------------------------


def render_report(
    cases: list[GoldenCase],
    results: list[CaseResult] | None,
    summary: dict[str, Any] | None,
    ablation: dict[str, Any],
    *,
    model: str = "local-configured-LLM",
) -> str:
    lines: list[str] = []
    lines.append("# GuardLens W4 离线评测报告\n")
    lines.append(
        "> 本报告全部在本地生成：语料为已脱敏的审计事件，数字来自真实评测运行，"
        "未手工编造；不含真实用户名、主机路径、内网域名/IP 或邮箱。\n"
    )
    lines.append(f"- 黄金集：{len(cases)} 题（STATS / EVIDENCE / MIXED，含 2 道无答案拒答题）")
    lines.append(f"- 生成模型：`{model}`（OpenAI 兼容端点，凭证取自环境变量，报告不含密钥）")
    lines.append("- 检索：BGE-M3 稠密 + BM25 稀疏 RRF + bge-reranker-v2-m3（Qdrant）\n")

    if summary:
        lines.append("## 1. 端到端总览（混合检索 + rerank，K=5）\n")
        lines.append(
            "> 口径：本节 Recall@5 对全部 20 题取均值；STATS 题无 gold 文档，记为 1"
            "（确定性取数）。纯检索层、仅证据/混合题的 Recall@5 见第 2 节对照实验。\n"
        )
        lines.append("| 指标 | 结果 |")
        lines.append("|---|---|")
        lines.append(f"| 题目数 | {summary['n']} |")
        lines.append(
            f"| **Recall@5（20 题均值，STATS 记满分）** | **{summary['recall_at5_pct']}%** |"
        )
        lines.append(
            f"| Recall@5（仅证据/混合题，14 题） | {summary['recall_at5_evidence_pct']}% |"
        )
        cite = summary["citation_accuracy_pct"]
        abst = summary["abstention_accuracy_pct"]
        lines.append(f"| **引用正确率**（真实存在且在召回集内） | **{cite}%** |")
        lines.append(f"| **拒答正确率**（该拒答则拒答、否则不拒答） | **{abst}%** |")
        if summary["number_accuracy_pct"] is not None:
            lines.append(
                f"| 数值正确率（STATS/MIXED 数字原样命中） | {summary['number_accuracy_pct']}% |"
            )
        lines.append(
            f"| 延迟 p50 / p95 | {summary['latency_p50_s']}s / {summary['latency_p95_s']}s |"
        )
        lines.append(f"| 延迟均值 | {summary['latency_mean_s']}s |")
        lines.append(
            f"| token（prompt / completion 总量） | "
            f"{summary['prompt_tokens_total']} / {summary['completion_tokens_total']} |"
        )
        lines.append(
            f"| 每问 token（prompt / completion 均值） | "
            f"{summary['prompt_tokens_mean']} / {summary['completion_tokens_mean']} |\n"
        )

    if ablation:
        lines.append("## 2. 检索对照实验（Recall@5，仅检索层、无 LLM）\n")
        lines.append("对比 W2 纯稠密向量 vs BM25 混合 vs 混合+rerank，以及 K=3/5/8、rerank 开关。")
        lines.append("Recall@5 = 应命中证据进入前 5 的比例（只对有 gold 的证据/混合题统计）。\n")
        lines.append("| 检索模式 | K | rerank | 题数 | Recall@5 |")
        lines.append("|---|---|---|---|---|")
        for g in ablation["grid"]:
            rr = "✅" if g["mode"] == "hybrid_rerank" else "—"
            lines.append(f"| {g['mode']} | {g['k']} | {rr} | {g['n']} | {g['recall_at5_pct']}% |")
        dense = next(
            (g["recall_at5_pct"] for g in ablation["grid"] if g["mode"] == "dense" and g["k"] == 5),
            None,
        )
        best = max(
            (g["recall_at5_pct"] for g in ablation["grid"] if g["mode"] == "hybrid_rerank"),
            default=None,
        )
        if dense is not None and best is not None:
            lines.append(
                f"\n**结论：W2 纯稠密 Recall@5={dense}% → 混合+rerank 最优 {best}%，"
                f"净提升 {round(best - dense, 1)} 个百分点。**\n"
            )

    if results:
        lines.append("## 3. 逐题结果（脱敏，仅证据 id / 指标）\n")
        lines.append("| 题号 | 类型 | 路由 | Recall@5 | 引用合法 | 拒答正确 | 数值正确 | 延迟s |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in results:
            num = "—" if r.number_ok is None else ("✅" if r.number_ok else "❌")
            lines.append(
                f"| {r.id} | {r.type} | {r.route} | {round(100 * r.recall_at_5)}% | "
                f"{'✅' if r.citation_ok else '❌ ' + ','.join(r.invalid_citations)} | "
                f"{'✅' if r.abstention_ok else '❌'} | {num} | {round(r.latency_s, 2)} |"
            )
        lines.append("")

    lines.append("## 4. 结论与说明\n")
    lines.append(
        "- **能确定的交给确定工具**：STATS/MIXED 的数字由只读 DuckDB 给出并原样进入 prompt，"
        "LLM 只组织语言，因此数值稳定、可复现。\n"
        "- **混合检索 + rerank 的价值集中在精确命令词与绕行序列**：纯稠密对 curl/chmod/同 URL "
        "换工具等词项不敏感，BM25 精确命中 + RRF + cross-encoder 重排把这些题的召回拉起来。\n"
        "- **反幻觉**：每条引用必须真实存在且在本题召回集内；两道无答案题（反弹 shell、挖矿）"
        "应答“日志中未发现”，以拒答正确率量化。\n"
        "- 延迟主要由 LLM 生成主导；本地 embedding/rerank 在 Apple Silicon MPS 上运行。\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="GuardLens W4 offline evaluation")
    ap.add_argument("--golden", default=str(GOLDEN_DEFAULT))
    ap.add_argument("--corpus", default=CORPUS_DEFAULT)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--report", default=str(ROOT / "eval" / "eval_report.md"))
    ap.add_argument("--results", default=str(RESULTS_DEFAULT))
    ap.add_argument("--ablation-only", action="store_true", help="retrieval grid only, no LLM E2E")
    ap.add_argument("--mode", default="hybrid_rerank", choices=RETRIEVAL_MODES)
    ap.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    cases = load_golden(args.golden)
    retriever = build_retriever(args.corpus)
    from guardlens.orchestrator import load_existing_ids

    existing_ids = load_existing_ids(args.corpus)
    problems = validate_golden(cases, existing_ids)
    if problems:
        print("Golden set validation problems:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print(f"golden set OK: {len(cases)} cases")

    print("running retrieval ablation ...")
    ablation = run_ablation(cases, retriever)
    for g in ablation["grid"]:
        print(f"  {g['mode']:15} k={g['k']}  Recall@5={g['recall_at5_pct']}%")

    results = None
    summary = None
    model = "local-configured-LLM"
    if not args.ablation_only:
        cfg = resolve_config()
        model = cfg.model
        print(f"running E2E with {model} ...")
        runner = make_live_runner(cfg)
        results = run_e2e(
            cases,
            retriever,
            runner,
            db_path=args.db,
            mode=args.mode,
            k=args.k,
            existing_ids=existing_ids,
        )
        summary = summarize(results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    report = render_report(cases, results, summary, ablation, model=model)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"wrote {args.report}")

    payload = {
        "summary": summary,
        "ablation": ablation,
        "cases": [
            {
                "id": r.id,
                "type": r.type,
                "route": r.route,
                "recall_at5": r.recall_at_5,
                "gold_hit": r.gold_hit,
                "citation_ok": r.citation_ok,
                "invalid_citations": r.invalid_citations,
                "abstained": r.abstained,
                "abstention_ok": r.abstention_ok,
                "number_ok": r.number_ok,
                "latency_s": round(r.latency_s, 3),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "retrieved_ids": r.retrieved_ids,
            }
            for r in (results or [])
        ],
    }
    Path(args.results).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.results}")


if __name__ == "__main__":
    main()
