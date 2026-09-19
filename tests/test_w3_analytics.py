"""W3 analytics tests: risk families, evasion sequences, FP proxies.

Builds a tiny DuckDB with the same `events` schema; no model or Qdrant needed.
"""

from __future__ import annotations

import json

import duckdb

from guardlens.analytics import (
    detect_evasion,
    detect_fp_proxies,
    risk_families,
)


def _make_db(tmp_path, rows):
    db = tmp_path / "mini.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE events (
            ts TIMESTAMP, tool VARCHAR, category VARCHAR, decision VARCHAR,
            reason VARCHAR, args_json VARCHAR, first_binary VARCHAR
        )
        """
    )
    for r in rows:
        con.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                r["ts"],
                r["tool"],
                r["category"],
                r["decision"],
                r.get("reason", ""),
                json.dumps({"command": r.get("command", ""), "url": r.get("url")}),
                (r.get("command") or "").split()[0] if r.get("command") else None,
            ],
        )
    con.close()
    return str(db)


def _ev(ts, tool, cat, dec, cmd, reason=""):
    return {
        "ts": ts,
        "tool": tool,
        "category": cat,
        "decision": dec,
        "command": cmd,
        "reason": reason,
    }


def test_risk_families_group_and_pct(tmp_path):
    rows = [
        _ev(
            "2026-09-10 10:00:00",
            "bash",
            "network",
            "deny",
            "curl https://evil.example/x",
            reason="BLOCKED (network)",
        ),
        _ev(
            "2026-09-10 10:01:00",
            "bash",
            "network",
            "deny",
            "wget https://evil.example/y",
            reason="BLOCKED (network)",
        ),
        _ev(
            "2026-09-10 10:02:00",
            "bash",
            "file_write",
            "deny",
            "echo x >> /etc/hosts",
            reason="path not TM-protected 越界",
        ),
        _ev(
            "2026-09-10 10:03:00", "bash", "unknown", "deny", "customtool", reason="Unknown command"
        ),
    ]
    db = _make_db(tmp_path, rows)
    fams = risk_families(db)
    total = sum(f.count for f in fams)
    assert total == 4
    by = {f.key: f for f in fams}
    assert by["egress"].count == 2
    assert by["out_of_sandbox_write"].count == 1
    # every family carries a representative example
    assert by["egress"].examples
    assert abs(sum(f.pct for f in fams) - 100.0) < 0.01


def test_evasion_detects_curl_to_webfetch_swap(tmp_path):
    rows = [
        _ev(
            "2026-09-10 10:00:00", "bash", "network", "deny", 'curl -sL "https://example.com/a.pdf"'
        ),
        _ev(
            "2026-09-10 10:00:07",
            "web_fetch",
            "network",
            "allow",
            "",
        ),
    ]
    rows[1]["url"] = "https://example.com/a.pdf"
    db = _make_db(tmp_path, rows)
    sigs = detect_evasion(db)
    assert len(sigs) == 1
    assert sigs[0].kind == "tool_swap"
    assert sigs[0].denied_tool == "bash"
    assert sigs[0].follow_tool == "web_fetch"


def test_evasion_ignores_unrelated_local_command(tmp_path):
    # a local mkdir deny followed by an unrelated web_search is not evasion
    rows = [
        _ev(
            "2026-09-10 10:00:00",
            "bash",
            "unknown",
            "deny",
            "mkdir -p /tmp/x && soffice --convert-to pdf",
        ),
        _ev("2026-09-10 10:00:05", "web_search", "auto-approve", "allow", ""),
    ]
    db = _make_db(tmp_path, rows)
    assert detect_evasion(db) == []


def test_evasion_flags_rewritten_network_command_then_allowed(tmp_path):
    rows = [
        _ev(
            "2026-09-10 10:00:00",
            "bash",
            "network",
            "deny",
            "curl -X POST http://127.0.0.1:3104/api/login",
        ),
        _ev(
            "2026-09-10 10:00:30",
            "bash",
            "local_network",
            "allow",
            "curl -s -o /dev/null http://127.0.0.1:3104/login",
        ),
    ]
    db = _make_db(tmp_path, rows)
    sigs = detect_evasion(db)
    assert any(s.kind == "blocked_then_allowed" for s in sigs)


def test_fp_proxy_requires_same_substantive_intent(tmp_path):
    rows = [
        # same convert target, denied then allowed via different approach
        _ev("2026-09-10 10:00:00", "bash", "process", "deny", "pdftotext /tmp/a.pdf /tmp/a.txt"),
        _ev("2026-09-10 10:00:40", "bash", "process", "allow", "python3 convert /tmp/a.pdf"),
        # generic cd/echo prefixes must NOT count as the same intent
        _ev("2026-09-10 11:00:00", "bash", "unknown", "deny", "cd /proj && echo hi"),
        _ev("2026-09-10 11:00:10", "bash", "unknown", "allow", "cd /proj && ls -la"),
    ]
    db = _make_db(tmp_path, rows)
    fps = detect_fp_proxies(db)
    assert len(fps) == 1
    assert "convert" in fps[0].pattern
    assert fps[0].delta_sec == 40.0
