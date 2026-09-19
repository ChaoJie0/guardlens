"""Streaming ingestion: agent-gate audit.jsonl -> validated rows -> DuckDB.

Usage:
    uv run python -m guardlens.ingest \
        --input "/path/to/audit.jsonl" \
        --db data/derived/guardlens.duckdb

W1 keeps real paths by default. Add --redact to build a shareable dataset
(/Users/<name> -> /Users/user, plus email/IP/token scrubbing).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from . import db as dbmod
from .parsing import extract_command_features
from .redaction import redact_path, redact_text
from .schema import Event

# Baseline measured on the author's real corpus (21,754 lines, 0 bad).
# Used only for a post-load diff report; never to fabricate counts.
BASELINE: dict[str, Any] = {
    "total": 21754,
    "decision": {"allow": 20629, "deny": 1125},
    "fail_closed": 5,
    "tools": {
        "bash": 7280,
        "read": 3736,
        "write": 3198,
        "read_image": 2496,
        "edit": 1546,
    },
    "unknown_category": 846,
}

# deny reason buckets, keyed by a substring of the reason header.
_REASON_BUCKETS = [
    ("unknown, cannot verify rollback-ability", "unknown_unrollbackable"),
    ("path NOT protected by Time Machine", "write_unprotected_path"),
    ("network, not rollback-able", "network"),
    ("process, not rollback-able", "process"),
    ("vcs_remote", "vcs_remote"),
    ("package", "package"),
    ("network, forced deny", "network_forced_deny"),
    ("read, forced deny", "read_forced_deny"),
    ("system, forced deny", "system_forced_deny"),
]


def _bucket_reason(reason: str) -> str:
    for needle, bucket in _REASON_BUCKETS:
        if needle in reason:
            return bucket
    return "other"


def _coerce_double_encoded_args(obj: dict[str, Any], *, lax: bool = False) -> dict[str, Any]:
    """Repair records whose ``args`` value was serialized twice.

    Some upstream plugin versions wrote ``args`` as a JSON *string* instead
    of an object (observed on ~15/21k rows). Two shapes occur:

    * well-formed JSON strings -> ``json.loads``;
    * truncated / unquoted-enum strings (e.g. ``"sandbox_permissions":
      danger-full-access`` or a LaTeX-heavy ``send_message`` cut mid-string)
      -> under ``lax`` mode, best-effort key recovery, and the raw payload is
      always preserved under ``args["_raw_args"]`` so no information is lost.

    A row only lands in quarantine when it cannot even meet the core
    schema (missing ts/tool/category/decision).
    """
    args = obj.get("args")
    if not isinstance(args, str):
        return obj
    s = args.strip()
    # Require an object-ish start; tolerate missing trailing brace because
    # the upstream plugin truncates some very long payloads mid-string.
    if not s.startswith("{"):
        return obj

    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            obj["args"] = parsed
            return obj
    except json.JSONDecodeError:
        pass

    if not lax:
        return obj

    recovered: dict[str, Any] = {}
    # Recover safe top-level string/enum fields: "key": value-without-quotes
    # or "key": "quoted..." up to the next comma. Best effort, auditable.
    for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"([^"}]{0,500})', s):
        k, v = m.group(1), m.group(2)
        if k not in recovered:
            recovered[k] = v
    # Bare enum values, e.g. "sandbox_permissions": danger-full-access
    for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*([A-Za-z0-9_-]+)\s*[,}]', s):
        recovered.setdefault(m.group(1), m.group(2))

    # Last-resort case (e.g. a 16KB LaTeX message truncated mid-string with
    # no parseable tail): keep the raw payload under a documented key only.
    if not recovered:
        recovered["_raw_args"] = args
    else:
        recovered["_raw_args"] = args
    obj["args"] = recovered
    return obj


def _redact_event_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Apply redaction to the raw dict before validation."""
    if isinstance(obj.get("reason"), str):
        obj["reason"] = redact_text(obj["reason"])
    if isinstance(obj.get("targetPaths"), list):
        obj["targetPaths"] = [
            redact_path(p) if isinstance(p, str) else p for p in obj["targetPaths"]
        ]
    args = obj.get("args")
    if isinstance(args, dict):
        # Recursively redact every string value — command, justification,
        # file content, edit old/new_string, etc. all embed absolute paths.
        obj["args"] = _redact_value(args)
    return obj


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    return value


def iter_audit_jsonl(
    fh: TextIO,
    *,
    redact: bool = False,
    lax_args: bool = True,
) -> tuple[list[tuple[Event, dict[str, Any] | None]], list[dict[str, Any]]]:
    """Stream-parse a JSONL file.

    Returns (good, quarantined). A quarantined row keeps line number + raw
    payload + error so ingestion can report (and later persist) it.

    ``lax_args`` recovers rows whose ``args`` JSON string is malformed by
    the upstream plugin (best-effort key extraction + raw preservation).
    """
    good: list[tuple[Event, dict[str, Any] | None]] = []
    quarantined: list[dict[str, Any]] = []
    repaired = 0

    for lineno, raw in enumerate(fh, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("top-level JSON value is not an object")
            before = obj.get("args")
            obj = _coerce_double_encoded_args(obj, lax=lax_args)
            if isinstance(before, str) and isinstance(obj.get("args"), dict):
                repaired += 1
            if redact:
                obj = _redact_event_dict(obj)
            event = Event.model_validate(obj)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            quarantined.append(
                {
                    "lineno": lineno,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw": line[:1000],
                }
            )
            continue

        command = event.args.get("command") if isinstance(event.args, dict) else None
        features = extract_command_features(command).to_dict() if isinstance(command, str) else None
        good.append((event, features))

    logging.info("repaired double-encoded/malformed args rows: %d", repaired)
    return good, quarantined


def write_quarantine(path: Path, quarantined: list[dict[str, Any]]) -> None:
    if not quarantined:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in quarantined:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(events: list[Event]) -> dict[str, Any]:
    decisions = Counter(e.decision for e in events)
    tools = Counter(e.tool for e in events)
    categories = Counter(e.category for e in events)
    deny_buckets = Counter(_bucket_reason(e.reason) for e in events if e.decision == "deny")
    fail_closed = sum(1 for e in events if e.category == "fail-closed")
    return {
        "total": len(events),
        "decision": dict(decisions),
        "fail_closed": fail_closed,
        "tools": dict(tools),
        "categories": dict(categories),
        "deny_buckets": dict(deny_buckets),
        "intercept_rate": round(decisions["deny"] / len(events), 4) if events else 0,
    }


def _print_diff(name: str, actual: Any, expected: Any, *, monotonic: bool = False) -> bool:
    """Compare one metric against the frozen 2026-09-19 baseline snapshot.

    The upstream audit log is append-only and keeps growing (the autonomous
    agent is still running), so volume metrics are *monotonic*: actual >=
    baseline is expected and healthy; a smaller number signals data loss.
    Structural invariants (e.g. fail-closed count) still must match exactly.
    """
    if monotonic:
        ok = actual >= expected
        flag = "OK+ " if actual > expected else ("OK  " if actual == expected else "LOSS")
    else:
        ok = actual == expected
        flag = "OK  " if ok else "DIFF"
    print(f"  [{flag}] {name}: actual={actual} baseline={expected}")
    return ok


def report_and_diff(metrics: dict[str, Any], quarantined_count: int) -> bool:
    """Print metrics and diff against baseline. Returns True if healthy."""
    print("=" * 72)
    print("GuardLens ingestion report")
    print("=" * 72)
    print(f"total events ........... {metrics['total']}")
    print(f"allow .................. {metrics['decision'].get('allow', 0)}")
    print(f"deny ................... {metrics['decision'].get('deny', 0)}")
    print(f"intercept rate ......... {metrics['intercept_rate']:.2%}")
    print(f"fail-closed ............ {metrics['fail_closed']}")
    print(f"quarantined (bad lines)  {quarantined_count}")

    print("\ntools (top 10):")
    for tool, n in Counter(metrics["tools"]).most_common(10):
        print(f"  {tool:<16} {n}")

    print("\ncategories:")
    for cat, n in Counter(metrics["categories"]).most_common():
        print(f"  {cat:<18} {n}")

    print("\ndeny reason buckets:")
    for bucket, n in Counter(metrics["deny_buckets"]).most_common():
        print(f"  {bucket:<28} {n}")

    print("\n--- baseline diff (frozen 2026-09-19 snapshot; OK+ = grew) ---")
    results = [
        _print_diff("total", metrics["total"], BASELINE["total"], monotonic=True),
        _print_diff(
            "allow",
            metrics["decision"].get("allow", 0),
            BASELINE["decision"]["allow"],
            monotonic=True,
        ),
        _print_diff(
            "deny",
            metrics["decision"].get("deny", 0),
            BASELINE["decision"]["deny"],
            monotonic=True,
        ),
        # Structural invariants: the 5 fail-closed events are historical and
        # must neither disappear nor be fabricated.
        _print_diff("fail_closed", metrics["fail_closed"], BASELINE["fail_closed"]),
        _print_diff(
            "unknown_category",
            metrics["categories"].get("unknown", 0),
            BASELINE["unknown_category"],
            monotonic=True,
        ),
        _print_diff("quarantined", quarantined_count, 0),
    ]
    for tool, expected in BASELINE["tools"].items():
        results.append(
            _print_diff(f"tool:{tool}", metrics["tools"].get(tool, 0), expected, monotonic=True)
        )

    all_ok = all(results)
    print(
        "\nBASELINE HEALTH:",
        "OK (no data loss vs frozen snapshot)"
        if all_ok
        else "UNHEALTHY — investigate (no hardcoding!)",
    )
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_ingestion(
    input_path: Path,
    db_path: Path,
    *,
    redact: bool = False,
    quarantine_path: Path | None = None,
    reset_table: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    with input_path.open("r", encoding="utf-8") as fh:
        good, quarantined = iter_audit_jsonl(fh, redact=redact)

    if quarantine_path is not None:
        write_quarantine(quarantine_path, quarantined)

    con = dbmod.connect(db_path)
    try:
        if reset_table:
            dbmod.reset(con)
        inserted = dbmod.insert_events(con, good)
    finally:
        con.close()

    metrics = compute_metrics([e for e, _ in good])
    metrics["inserted"] = inserted
    metrics["quarantined"] = len(quarantined)
    metrics["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest agent-gate audit.jsonl into DuckDB")
    parser.add_argument(
        "--input",
        default="/path/to/audit.jsonl",
        help="Path to audit.jsonl",
    )
    parser.add_argument("--db", default="data/derived/guardlens.duckdb")
    parser.add_argument(
        "--quarantine",
        default="data/quarantine/quarantine.jsonl",
        help="Where to write bad rows (only if any)",
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Redact /Users/<name>, emails, IPs, tokens (for public datasets)",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2

    metrics = run_ingestion(
        input_path,
        Path(args.db),
        redact=args.redact,
        quarantine_path=Path(args.quarantine),
    )
    print(f"ingested in {metrics['elapsed_seconds']}s -> {args.db}")
    ok = report_and_diff(metrics, metrics["quarantined"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
