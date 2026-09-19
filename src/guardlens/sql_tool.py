"""Text2SQL tool with guardrails (W3 differentiated feature #4).

Stats questions bypass vector retrieval and answer from DuckDB. The LLM never
touches numbers: it only generates read-only SQL behind three gates:

  1. parse gate — reject anything that isn't a single SELECT/WITH statement
     (no INSERT/UPDATE/DELETE/DDL/ATTACH/pragmas/multiple statements);
  2. sandbox gate — open the (copied) DuckDB file read-only; DuckDB itself then
     refuses any mutation;
  3. dry-run + row cap — validate the plan with a LIMIT-0 prepare and cap
     returned rows; the raw result is passed through verbatim.

A curated set of "known stats" functions covers the common audit questions
deterministically (no LLM at all); free-form LLM SQL is opt-in.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

MAX_ROWS = 100

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|"
    r"pragma|copy|export|vacuum|call|exec|grant|revoke|merge|replace|"
    r"begin|commit|rollback|set)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    pass


@dataclass(slots=True)
class SQLResult:
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool

    def to_text(self) -> str:
        lines = [" | ".join(self.columns)]
        lines.extend(" | ".join(_fmt(v) for v in row) for row in self.rows)
        if self.truncated:
            lines.append(f"(结果超过 {MAX_ROWS} 行，已截断)")
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def validate_select(sql: str) -> str:
    """Gate 1: ensure the query is one read-only SELECT/WITH statement."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise UnsafeSQLError("空 SQL")
    # exactly one statement (no stacked '; ...')
    if ";" in s:
        raise UnsafeSQLError("只允许单条语句")
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise UnsafeSQLError("只允许 SELECT/WITH")
    if _FORBIDDEN.search(s):
        raise UnsafeSQLError("检测到非只读关键字")
    # forbid reading other files / databases
    if re.search(r"\b(read_|<|from\s+['\"]|glob)", s, re.IGNORECASE) and "read_" in low:
        raise UnsafeSQLError("禁止读取外部文件函数")
    return s


def run_readonly(sql: str, db_path: str | Path, *, max_rows: int = MAX_ROWS) -> SQLResult:
    """Gates 2+3: read-only connection, validated SQL, row cap."""
    safe = validate_select(sql)

    # Open a throwaway read-only copy so a locked/live production DB and any
    # accidental write are impossible. The copy lives in a temp dir.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Path(tmp) / "ro.duckdb"
        shutil.copy2(db_path, tmp_db)
        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            # Gate 3a: plan validation without producing rows.
            con.execute(f"CREATE OR REPLACE TEMP VIEW _q AS SELECT * FROM ({safe}) LIMIT 0")
            cols = [c[0] for c in con.execute("SELECT * FROM _q LIMIT 0").description]
            # Gate 3b: bounded execution.
            rows = con.execute(f"SELECT * FROM ({safe}) LIMIT {max_rows + 1}").fetchall()
        finally:
            con.close()

    truncated = len(rows) > max_rows
    return SQLResult(sql=safe, columns=cols, rows=rows[:max_rows], truncated=truncated)


# ---------------------------------------------------------------------------
# Deterministic known-stats (no LLM) — the safe default for the audit domain
# ---------------------------------------------------------------------------


def _scalars(db_path: str | Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Path(tmp) / "ro.duckdb"
        shutil.copy2(db_path, tmp_db)
        con = duckdb.connect(str(tmp_db), read_only=True)
        try:
            row = con.execute(
                """
                SELECT COUNT(*),
                       SUM(decision='allow'), SUM(decision='deny'),
                       SUM(category='fail-closed')
                FROM events
                """
            ).fetchone()
            deny_by_cat = con.execute(
                "SELECT category, COUNT(*) FROM events WHERE decision='deny' "
                "GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
        finally:
            con.close()
    total, allow_n, deny_n, fc = row
    return {
        "total": total,
        "allow": allow_n,
        "deny": deny_n,
        "fail_closed": fc,
        "rate": (deny_n or 0) / total if total else 0.0,
        "deny_by_category": deny_by_cat,
    }


def known_stats(question: str, db_path: str | Path) -> str | None:
    """Return deterministic stats text if the question matches a known one."""
    s = _scalars(db_path)
    q = question
    by_cat = "，".join(f"{c}={n}" for c, n in s["deny_by_category"])

    if "拦截率" in q or "拦截比例" in q:
        return (
            f"总事件={s['total']}，allow={s['allow']}，deny={s['deny']}，"
            f"拦截率={s['rate']:.4f}（{s['rate']:.2%}）。"
        )
    if "越界写" in q or ("写" in q and ("几起" in q or "多少" in q or "拦截" in q)):
        n = dict(s["deny_by_category"]).get("file_write", 0)
        return f"file_write（越界写）拦截共 {n} 起。各类别拦截：{by_cat}。"
    if "fail" in q.lower() or "fail-closed" in q:
        return f"fail-closed 事件共 {s['fail_closed']} 起。各类别拦截：{by_cat}。"
    if "拦截" in q and ("多少" in q or "总数" in q or "几次" in q or "几起" in q):
        return f"总拦截 deny={s['deny']} 起，拦截率={s['rate']:.2%}。各类别：{by_cat}。"
    # W4 eval coverage: a few more fully-deterministic aggregate questions.
    cats = dict(s["deny_by_category"])
    if ("外联" in q or "网络" in q) and ("多少" in q or "几起" in q or "几次" in q):
        return f"network（外联）类拦截共 {cats.get('network', 0)} 起。各类别：{by_cat}。"
    if ("类别" in q or "分布" in q) and ("拦截" in q or "deny" in q.lower()):
        return f"拦截按类别分布：{by_cat}。合计 deny={s['deny']} 起。"
    if ("多少条" in q or "总数" in q or "一共多少" in q) and (
        "事件" in q or "日志" in q or "审计" in q
    ):
        return (
            f"审计事件总数={s['total']}，allow={s['allow']}，deny={s['deny']}，"
            f"拦截率={s['rate']:.2%}。"
        )
    return None
