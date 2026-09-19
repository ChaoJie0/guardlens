"""Human 👍/👎 feedback on evidence answers, persisted to DuckDB (W3).

Feedback is written to a SEPARATE database (never the read-only audit copy),
so the curated audit data is never mutated. The table supports the FP-proxy
workflow: a 👎 on a signal/answer feeds the human-confirmation loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

FEEDBACK_DB_DEFAULT = "data/derived/feedback.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_feedback (
    id BIGINT PRIMARY KEY,
    created_ts TIMESTAMP,
    question VARCHAR,
    evidence_id VARCHAR,
    verdict VARCHAR,          -- 'up' | 'down'
    note VARCHAR,
    answer_excerpt VARCHAR
);
"""


def _connect(path: str | Path) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA)
    return con


@dataclass(slots=True)
class Feedback:
    question: str
    evidence_id: str
    verdict: str
    note: str = ""
    answer_excerpt: str = ""


def add_feedback(fb: Feedback, path: str | Path = FEEDBACK_DB_DEFAULT) -> int:
    con = _connect(path)
    try:
        next_id = con.execute("SELECT COALESCE(MAX(id),0)+1 FROM answer_feedback").fetchone()[0]
        con.execute(
            "INSERT INTO answer_feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                next_id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                fb.question,
                fb.evidence_id,
                fb.verdict,
                fb.note,
                fb.answer_excerpt[:500],
            ],
        )
        return int(next_id)
    finally:
        con.close()


def list_feedback(path: str | Path = FEEDBACK_DB_DEFAULT, limit: int = 100):
    if not Path(path).exists():
        return []
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(
            "SELECT id, created_ts, question, evidence_id, verdict, note "
            "FROM answer_feedback ORDER BY id DESC LIMIT ?",
            [limit],
        ).fetchall()
    finally:
        con.close()


def verdict_counts(path: str | Path = FEEDBACK_DB_DEFAULT) -> dict[str, int]:
    if not Path(path).exists():
        return {"up": 0, "down": 0}
    con = duckdb.connect(str(path), read_only=True)
    try:
        rows = dict(
            con.execute("SELECT verdict, COUNT(*) FROM answer_feedback GROUP BY 1").fetchall()
        )
        return {"up": int(rows.get("up", 0)), "down": int(rows.get("down", 0))}
    finally:
        con.close()
