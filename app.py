"""GuardLens dashboard.

Tabs:
  • 看板 (W1): KPI / trend / category / tool table from DuckDB
  • 问答 (W2): evidence-grounded RAG Q&A with streaming + evidence drawer

Run:
    uv run streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path("data/derived/guardlens.duckdb")
CHROMA_PATH = Path("data/derived/chroma")

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


@st.cache_resource
def get_engine():
    from guardlens.qa import make_engine

    return make_engine(str(CHROMA_PATH), k=6)


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
    if not CHROMA_PATH.exists():
        st.error(f"向量库不存在：{CHROMA_PATH}，请先运行 `uv run python -m guardlens.build_index`")
        st.stop()

    try:
        from guardlens.llm_config import resolve_config

        cfg = resolve_config()
        st.caption(f"LLM: `{cfg.model}` @ {cfg.base_url}（来源 {cfg.source}）")
    except Exception as exc:  # noqa: BLE001
        st.error(f"未找到可用 LLM 凭证：{exc}")
        st.stop()

    question = st.text_input(
        "提问（证据类问题）",
        placeholder="例如：curl 被拦是为什么？有没有绕过尝试？fail-closed 发生在什么场景？",
    )
    go = st.button("提问", type="primary")
    if not (go and question.strip()):
        st.info("输入问题后点击「提问」，回答将流式输出并附带可点开的原始证据。")
        return

    engine = get_engine()
    answer_box = st.empty()
    buf: list[str] = []
    hits: list = []

    with st.spinner("检索证据并生成中…"):
        for ev in engine.stream(question, stats=stats_context()):
            if ev["type"] == "hit":
                hits.append(ev["hit"])
            elif ev["type"] == "token":
                buf.append(ev["delta"])
                answer_box.markdown("".join(buf) + " ▌")
            elif ev["type"] == "final":
                answer = ev["answer"]
                answer_box.markdown(answer.text)
                meta_cols = st.columns(4)
                meta_cols[0].metric("引用数", len(answer.valid_citations))
                meta_cols[1].metric("非法引用", len(answer.invalid_citations))
                meta_cols[2].metric("是否拒答", "是" if answer.abstained else "否")
                meta_cols[3].metric("召回证据", len(answer.retrieved_ids))
                if answer.invalid_citations:
                    st.warning(f"模型引用了检索集之外的证据：{answer.invalid_citations}")

    st.divider()
    st.subheader(f"证据（{len(hits)}）")
    for h in hits:
        sim = f"相似度 {h.score:.3f}"
        label = f"{h.evidence_id} · {h.doc_type} · {sim} · {h.ts or ''}"
        with st.expander(label):
            st.caption(
                f"tool={h.tool or '-'} category={h.category or '-'} decision={h.decision or '-'}"
            )
            st.text(h.text)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🛡️ GuardLens — agent-gate 审计日志分析")
tab_dash, tab_qa = st.tabs(["📊 看板（W1）", "💬 问答（W2 RAG）"])
with tab_dash:
    render_dashboard()
with tab_qa:
    render_qa()
