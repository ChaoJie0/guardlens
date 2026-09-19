"""GuardLens W1 static dashboard.

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

st.set_page_config(page_title="GuardLens · Agent 审计分析台", layout="wide")
st.title("🛡️ GuardLens — agent-gate 审计日志分析（W1 数据底座）")


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


data = load(str(DB_PATH))
kpi = data["kpi"].iloc[0]
total = int(kpi["total"])
deny_n = int(kpi["deny_n"])
intercept_rate = deny_n / total if total else 0.0

# ---------------------------------------------------------------- KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("总事件数", f"{total:,}")
c2.metric("放行 allow", f"{int(kpi['allow_n']):,}")
c3.metric("拦截 deny", f"{deny_n:,}")
c4.metric("拦截率", f"{intercept_rate:.2%}")
c5.metric("fail-closed", int(kpi["fail_closed"]))

st.divider()

# ---------------------------------------------------------------- trend
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
    yaxis="y",
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
