"""GuardLens dashboard.

Tabs:
  • 看板 (W1): KPI / trend / category / tool table from DuckDB
  • 问答 (W2/W3): cited RAG Q&A with streaming + evidence drawer (via the API)
  • 安全信号 (W3): risk families / bypass sequences / false-positive proxies

Run:
    uv run streamlit run app.py

Paths are configurable so the bundled sample can be opened without touching
the default database:
    GUARDLENS_DB=data/derived/sample.duckdb uv run streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path(os.environ.get("GUARDLENS_DB", "data/derived/guardlens.duckdb"))
CHROMA_PATH = Path(os.environ.get("GUARDLENS_CHROMA", "data/derived/chroma"))  # W2 fallback
CORPUS_PATH = Path(os.environ.get("GUARDLENS_CORPUS", "data/derived/corpus/corpus.jsonl"))
API_BASE = os.environ.get("GUARDLENS_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="GuardLens · Agent 审计分析台", layout="wide")


# ---------------------------------------------------------------------------
# W1 dashboard data
# ---------------------------------------------------------------------------


@st.cache_data
def load(db_path: str) -> dict[str, pd.DataFrame]:
    if not Path(db_path).exists():
        st.error(f"数据库不存在：{db_path}，请先运行 `uv run python -m guardlens.ingest`")
        st.stop()
    con = duckdb.connect(db_path, read_only=True)
    try:
        events = con.execute("SELECT * FROM events").df()
        kpi = con.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN decision='allow' THEN 1 ELSE 0 END) AS allow_n,
                SUM(CASE WHEN decision='deny'  THEN 1 ELSE 0 END) AS deny_n,
                SUM(CASE WHEN category='fail-closed' THEN 1 ELSE 0 END) AS fail_closed
            FROM events
            """
        ).df()
        by_tool = con.execute(
            "SELECT tool, COUNT(*) AS n FROM events GROUP BY 1 ORDER BY n DESC"
        ).df()
        by_category = con.execute(
            "SELECT category, COUNT(*) AS n FROM events GROUP BY 1 ORDER BY n DESC"
        ).df()
        trend = con.execute(
            """
            SELECT CAST(ts AS DATE) AS day,
                   SUM(CASE WHEN decision='deny' THEN 1 ELSE 0 END) AS denies,
                   COUNT(*) AS total
            FROM events
            GROUP BY 1 ORDER BY 1
            """
        ).df()
    finally:
        con.close()
    return {
        "events": events,
        "kpi": kpi,
        "by_tool": by_tool,
        "by_category": by_category,
        "trend": trend,
    }


def render_dashboard() -> None:
    data = load(str(DB_PATH))
    kpi = data["kpi"].iloc[0]
    total = int(kpi["total"])
    deny_n = int(kpi["deny_n"])
    intercept_rate = deny_n / total if total else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总事件数", f"{total:,}")
    c2.metric("放行 allow", f"{int(kpi['allow_n']):,}")
    c3.metric("拦截 deny", f"{deny_n:,}")
    c4.metric("拦截率", f"{intercept_rate:.2%}")
    c5.metric("fail-closed", int(kpi["fail_closed"]))

    st.divider()
    st.subheader("每日拦截趋势")
    fig_trend = px.bar(
        data["trend"],
        x="day",
        y="denies",
        labels={"day": "日期", "denies": "拦截数"},
        color_discrete_sequence=["#d62728"],
    )
    fig_trend.add_scatter(
        x=data["trend"]["day"],
        y=data["trend"]["total"],
        name="总调用",
        mode="lines+markers",
    )
    st.plotly_chart(fig_trend, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("category 分布")
        fig_cat = px.bar(
            data["by_category"],
            x="n",
            y="category",
            orientation="h",
            labels={"n": "事件数", "category": ""},
        ).update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cat, width="stretch")
    with right:
        st.subheader("tool top 分布表")
        st.dataframe(
            data["by_tool"].style.format({"n": "{:,}"}),
            width="stretch",
            hide_index=True,
        )

    with st.expander("原始事件（前 200 行，衍生字段可查）"):
        st.dataframe(data["events"].head(200), width="stretch")


# ---------------------------------------------------------------------------
# W2 RAG Q&A
# ---------------------------------------------------------------------------


# Backend (FastAPI) host: heavy models load once there; this frontend just
# streams events over HTTP, so browser refreshes never reload the models.
# Configured at the top of the file via GUARDLENS_API_BASE.


def api_health() -> bool:
    import httpx

    try:
        r = httpx.get(f"{API_BASE}/api/health", timeout=5)
        return bool(r.json().get("ok", False))
    except Exception:  # noqa: BLE001
        return False


def stream_question(question: str):
    """POST /api/ask and yield decoded SSE events (same shape as Orchestrator.stream)."""
    import json

    import httpx

    with httpx.stream(
        "POST",
        f"{API_BASE}/api/ask",
        json={"question": question},
        timeout=httpx.Timeout(connect=30, read=120, write=30, pool=30),
    ) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


@st.cache_data
def stats_context() -> str:
    """Authoritative DuckDB numbers injected into the prompt for numeric Qs."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        total, allow_n, deny_n, fc = con.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN decision='allow' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN decision='deny' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN category='fail-closed' THEN 1 ELSE 0 END)
            FROM events
            """
        ).fetchone()
        tool_rows = con.execute(
            "SELECT tool, COUNT(*) n FROM events GROUP BY 1 ORDER BY n DESC LIMIT 12"
        ).fetchall()
        cat_deny = con.execute(
            "SELECT category, COUNT(*) n FROM events WHERE decision='deny' "
            "GROUP BY 1 ORDER BY n DESC"
        ).fetchall()
        span = con.execute("SELECT MIN(ts), MAX(ts) FROM events").fetchone()
    finally:
        con.close()
    rate = deny_n / total
    lines = [
        f"事件总数={total}；allow={allow_n}；deny={deny_n}；拦截率={rate:.4f}；fail-closed={fc}",
        f"时间范围={span[0]} ~ {span[1]}",
        "工具调用量 top: " + ", ".join(f"{t}={n}" for t, n in tool_rows),
        "拦截按类别: " + ", ".join(f"{c}={n}" for c, n in cat_deny),
    ]
    return "\n".join(lines)


def render_qa() -> None:
    if not CORPUS_PATH.exists():
        st.error(f"语料不存在：{CORPUS_PATH}，请先运行 `uv run python -m guardlens.build_index`")
        st.stop()

    try:
        from guardlens.llm_config import resolve_config

        cfg = resolve_config()
        st.caption(
            f"W3：BM25+稠密 RRF · bge-reranker · Qdrant payload 过滤 ｜ LLM `{cfg.model}`"
            f" @ {cfg.base_url}（来源 {cfg.source}）"
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"未找到可用 LLM 凭证：{exc}")
        st.stop()

    if not api_health():
        st.error(
            "后端模型服务未启动。请先运行：`uv run uvicorn guardlens.server:app --port 8000`"
            "（首次启动会加载本地模型，约 1 分钟）"
        )
        st.stop()

    question = st.text_input(
        "提问（证据 / 统计 / 混合类均可，自动路由）",
        placeholder="例如：拦截率多少？curl 被拦后有没有换 web_fetch？fail-closed 几起、什么场景？",
    )
    go = st.button("提问", type="primary")
    if not (go and question.strip()):
        st.info(
            "STATS 类自动走 DuckDB 只读 SQL（数值不经过模型编造），EVIDENCE 类走混合检索+rerank，"
            "MIXED 类先取数再检索。回答流式输出并附带原始证据。"
        )
        return

    answer_box = st.empty()
    route_box = st.empty()
    progress_box = st.caption("⏳ 正在路由…")
    buf: list[str] = []
    hits: list = []
    route = stats = None

    for ev in stream_question(question):
        t = ev["type"]
        if t == "route":
            route = ev["route"]
            route_box.info(f"查询路由：**{route}**")
            progress_box.caption("🔎 取数 / 混合检索 + Rerank 中…")
        elif t == "stats":
            stats = ev["stats"]
            if stats:
                st.caption(f"📐 确定性取数（原样透传）：{stats}")
        elif t == "hit":
            hits.append(ev["hit"])
        elif t == "token":
            if buf:
                progress_box.empty()  # 第一个 token 出现即清掉检索进度，进入打字机
            buf.append(ev["delta"])
            answer_box.markdown("".join(buf) + " ▌")
        elif t == "final":
            answer = ev["answer"]
            answer_box.markdown(answer["text"])
            meta_cols = st.columns(4)
            meta_cols[0].metric("有效引用", len(answer["valid_citations"]))
            meta_cols[1].metric("非法引用", len(answer["invalid_citations"]))
            meta_cols[2].metric("是否拒答", "是" if answer["abstained"] else "否")
            meta_cols[3].metric("召回证据", len(answer["retrieved_ids"]))
            if answer["invalid_citations"]:
                st.warning(f"模型引用了检索集之外/不存在的证据：{answer['invalid_citations']}")

    st.divider()
    st.subheader(f"证据（{len(hits)}）")
    for h in hits:
        score = f"rerank {h['score']:.3f}"
        label = f"{h['evidence_id']} · {h['doc_type']} · {score} · {h['ts'] or ''}"
        with st.expander(label):
            st.caption(
                f"tool={h['tool'] or '-'} category={h['category'] or '-'} "
                f"decision={h['decision'] or '-'}"
            )
            st.text(h["text"])
            _feedback_row(question, h)


def _feedback_row(question: str, h) -> None:
    """👍/👎 human confirmation loop; feedback goes to a separate DuckDB."""
    from guardlens.feedback import FEEDBACK_DB_DEFAULT, Feedback, add_feedback, verdict_counts

    key = f"fb_{h['evidence_id']}_{abs(hash(question)) % 100000}"
    c1, c2, note_box = st.columns([1, 1, 6])
    if c1.button("👍 准确", key=f"{key}_up"):
        add_feedback(
            Feedback(
                question=question,
                evidence_id=h["evidence_id"],
                verdict="up",
                answer_excerpt=h["text"][:200],
            ),
            FEEDBACK_DB_DEFAULT,
        )
        st.toast("已记录为准确（策略可保持）")
    if c2.button("👎 误拦/无关", key=f"{key}_down"):
        add_feedback(
            Feedback(
                question=question,
                evidence_id=h["evidence_id"],
                verdict="down",
                answer_excerpt=h["text"][:200],
            ),
            FEEDBACK_DB_DEFAULT,
        )
        st.toast("已记录为疑似误拦/无关，进入人工确认闭环")
    counts = verdict_counts(FEEDBACK_DB_DEFAULT)
    note_box.caption(
        f"人工反馈累计：👍 {counts['up']} ｜ 👎 {counts['down']}（写入独立反馈库，不改审计数据）"
    )


# ---------------------------------------------------------------------------
# W3 security signals: risk families / evasion / false-positive proxies
# ---------------------------------------------------------------------------


@st.cache_data
def _risk_families():
    from guardlens.analytics import risk_families

    return risk_families(str(DB_PATH))


@st.cache_data
def _evasion_signals():
    from guardlens.analytics import detect_evasion

    return detect_evasion(str(DB_PATH))


@st.cache_data
def _fp_proxies():
    from guardlens.analytics import detect_fp_proxies

    return detect_fp_proxies(str(DB_PATH))


def render_signals() -> None:
    st.subheader("🧭 风险族归因（deny 事件归并）")
    fams = _risk_families()
    total = sum(f.count for f in fams)
    st.caption(
        f"共 {total} 条 deny 归并为 {len(fams)} 个风险族；unknown 占比高=策略保守/可观测盲区。"
    )
    fdf = pd.DataFrame(
        [{"风险族": f.label, "key": f.key, "数量": f.count, "占比%": f.pct} for f in fams]
    )
    st.dataframe(fdf, width="stretch", hide_index=True)
    st.plotly_chart(
        px.pie(fdf, names="风险族", values="数量", title="deny 风险族分布"),
        use_container_width=True,
    )
    for f in fams:
        with st.expander(f"{f.label} · {f.count} 起（{f.pct}%）— 代表事件"):
            for e in f.examples:
                st.code(
                    f"{e['ts']}\n{e['tool']} / {e['category']}\n"
                    f"{(e['command'] or e['reason'])[:240]}",
                    language="bash",
                )

    st.divider()
    st.subheader("🕵️ 绕过序列检测（deny → 短时同义/换工具调用）")
    ev = _evasion_signals()
    st.caption(f"命中 {len(ev)} 条疑似绕行信号（规则：双侧网络意图 + 换工具或命令改写）。")
    edf = pd.DataFrame(
        [
            {
                "信号": s.evidence_id,
                "类型": s.kind,
                "间隔s": s.delta_sec,
                "被拦工具": s.denied_tool,
                "随后工具": s.follow_tool,
                "被拦命令": s.denied_command,
                "随后调用": s.follow_command,
            }
            for s in ev
        ]
    )
    st.dataframe(edf, width="stretch", hide_index=True)

    st.divider()
    st.subheader("🩺 误拦代理分析（同意图 deny→allow，疑似可策略优化）")
    fps = _fp_proxies()
    st.caption(
        f"命中 {len(fps)} 条代理信号（同一实质动作被拦后换法成功）。"
        "这是代理指标而非真实误报率——需用 👎/👍 人工确认，闭环在问答页证据卡上。"
    )
    fpdf = pd.DataFrame(
        [
            {
                "信号": p.evidence_id,
                "工具": p.tool,
                "间隔s": p.delta_sec,
                "意图签名": p.pattern,
                "被拦": p.denied_summary,
                "随后放行": p.allow_summary,
            }
            for p in fps
        ]
    )
    st.dataframe(fpdf, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🛡️ GuardLens — Agent 审计日志分析")
tab_dash, tab_qa, tab_sig = st.tabs(
    ["📊 看板（W1）", "💬 问答（W3 混合检索+Rerank）", "🚨 安全信号（W3）"]
)
with tab_dash:
    render_dashboard()
with tab_qa:
    render_qa()
with tab_sig:
    render_signals()
