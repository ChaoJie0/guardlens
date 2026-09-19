"""DuckDB persistence for parsed audit events."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb

from .schema import Event

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    ts              TIMESTAMP NOT NULL,
    ts_raw          VARCHAR,
    tool            VARCHAR NOT NULL,
    category        VARCHAR NOT NULL,
    decision        VARCHAR NOT NULL,
    reason          VARCHAR,
    target_paths    JSON,
    args_json       JSON,
    snapshot_id     VARCHAR,
    -- derived shell features (NULL for non-bash / missing command)
    segment_count   INTEGER,
    first_binary    VARCHAR,
    binaries        JSON,
    redirect_targets JSON,
    risk_tokens     JSON
);
"""

DUPLICATE_VIEW_SQL = """
CREATE OR REPLACE TABLE events AS
SELECT DISTINCT * FROM events;
"""


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA_SQL)
    return con


def reset(con: duckdb.DuckDBPyConnection) -> None:
    """Drop & recreate the events table (full reload semantics)."""
    con.execute("DROP TABLE IF EXISTS events")
    con.execute(SCHEMA_SQL)


def _row_from_event(event: Event, features: dict[str, Any] | None) -> tuple[Any, ...]:
    args = event.args
    command = args.get("command") if isinstance(args, dict) else None
    feat = features or {}
    return (
        event.ts,  # DuckDB auto-casts ISO-8601 to TIMESTAMP
        event.ts,
        event.tool,
        event.category,
        event.decision,
        event.reason,
        json.dumps(event.targetPaths, ensure_ascii=False),
        json.dumps(event.args, ensure_ascii=False),
        event.snapshotId,
        feat.get("segment_count") if command is not None else None,
        feat.get("first_binary") if command is not None else None,
        json.dumps(feat.get("binaries", []), ensure_ascii=False) if command is not None else None,
        json.dumps(feat.get("redirect_targets", []), ensure_ascii=False)
        if command is not None
        else None,
        json.dumps(feat.get("risk_tokens", []), ensure_ascii=False)
        if command is not None
        else None,
    )


INSERT_SQL = """
INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def insert_events(
    con: duckdb.DuckDBPyConnection,
    rows: Iterable[tuple[Event, dict[str, Any] | None]],
) -> int:
    payload = [_row_from_event(ev, feat) for ev, feat in rows]
    if not payload:
        return 0
    con.executemany(INSERT_SQL, payload)
    return len(payload)
