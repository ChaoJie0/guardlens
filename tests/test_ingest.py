"""Tests for streaming ingestion semantics."""

from __future__ import annotations

import io
import json

from guardlens.ingest import iter_audit_jsonl


def _line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_double_encoded_args_are_repaired() -> None:
    # Real-world dirty row: args written as a JSON *string*.
    payload = {
        "ts": "2026-09-11T06:21:58.854Z",
        "tool": "send_message",
        "category": "unknown",
        "decision": "deny",
        "reason": "BLOCKED (unknown): Unknown tool: send_message",
        "targetPaths": [],
        "args": json.dumps({"agent_id": "session-x", "message": "hi"}, ensure_ascii=False),
    }
    good, quarantined = iter_audit_jsonl(io.StringIO(_line(payload)))
    assert len(good) == 1
    assert quarantined == []
    event, _features = good[0]
    assert event.args["agent_id"] == "session-x"


def test_bad_lines_are_quarantined_with_lineno() -> None:
    valid = _line(
        {
            "ts": "2026-09-06T09:25:22.766Z",
            "tool": "bash",
            "category": "read",
            "decision": "allow",
            "reason": "ok",
            "targetPaths": [],
            "args": {},
        }
    )
    stream = io.StringIO(valid + "\n" + "{not json}\n" + "\n")
    good, quarantined = iter_audit_jsonl(stream)
    assert len(good) == 1
    assert len(quarantined) == 1
    assert quarantined[0]["lineno"] == 2
    assert "JSONDecodeError" in quarantined[0]["error"]
