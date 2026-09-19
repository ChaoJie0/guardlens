"""W3 security-domain analytics: risk families, evasion sequences, FP proxies.

Everything here is rule-based over DuckDB (deterministic, auditable); the LLM
is only used downstream to *explain* a signal, never to invent one.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

# ---------------------------------------------------------------------------
# Risk-family taxonomy (maps the plugin's category + reason bucket to a family)
# ---------------------------------------------------------------------------

RISK_FAMILIES = [
    ("egress", "外联访问", ("network", "local_network", "vcs_remote")),
    ("data_exfil", "数据走私", ("exfil",)),
    ("out_of_sandbox_write", "越界写", ("file_write",)),
    ("process", "进程/命令执行", ("process", "package", "system")),
    ("unknown_unrollbackable", "不可回滚/未知", ("unknown",)),
    ("fail_closed", "安全边界 fail-closed", ("fail-closed",)),
]

# reason-bucket overrides (see ingest._bucket_reason) used to split egress vs
# package/vcs when category is coarse.
_REASON_FAMILY = [
    (("exfil", "data"), "data_exfil"),
    (("vcs_remote",), "egress"),
    (("package",), "process"),
    (("network",), "egress"),
    (("write_unprotected_path", "越界", "not protected"), "out_of_sandbox_write"),
    (("unknown", "cannot verify"), "unknown_unrollbackable"),
    (("process",), "process"),
]


@dataclass(slots=True)
class FamilyStat:
    key: str
    label: str
    count: int
    pct: float
    examples: list[dict[str, Any]] = field(default_factory=list)


def _family_for(category: str, reason: str) -> str:
    r = (reason or "").lower()
    for needles, fam in _REASON_FAMILY:
        if any(n in r for n in needles):
            return fam
    for key, _label, cats in RISK_FAMILIES:
        if category in cats:
            return key
    return "unknown_unrollbackable"


def _read_only_con(db_path: str | Path):
    """Copy DB to a temp file and open read-only (analytics never mutates)."""
    tmp = Path(tempfile.mkdtemp()) / "ro.duckdb"
    shutil.copy2(db_path, tmp)
    return duckdb.connect(str(tmp), read_only=True), tmp


def risk_families(db_path: str | Path, *, top_examples: int = 5) -> list[FamilyStat]:
    """Aggregate all deny events into risk families with representative rows."""
    con, tmp = _read_only_con(db_path)
    try:
        rows = con.execute(
            """
            SELECT ts, tool, category, decision, reason,
                   json_extract_string(args_json, '$.command') AS command,
                   json_extract_string(args_json, '$.url') AS url
            FROM events WHERE decision='deny'
            """
        ).fetchall()
    finally:
        con.close()

    fam_rows: dict[str, list[tuple]] = {}
    for row in rows:
        ts, tool, category, decision, reason, command, url = row
        fam = _family_for(category or "", reason or "")
        fam_rows.setdefault(fam, []).append(row)

    total = len(rows)
    label_of = {key: label for key, label, _ in RISK_FAMILIES}
    stats: list[FamilyStat] = []
    for key, grouped in fam_rows.items():
        examples = []
        for ts, tool, category, _decision, reason, command, url in grouped[:top_examples]:
            examples.append(
                {
                    "ts": str(ts),
                    "tool": tool,
                    "category": category,
                    "reason": (reason or "")[:200],
                    "command": (command or "")[:200],
                    "url": url,
                }
            )
        stats.append(
            FamilyStat(
                key=key,
                label=label_of.get(key, key),
                count=len(grouped),
                pct=round(100 * len(grouped) / total, 1) if total else 0.0,
                examples=examples,
            )
        )
    return sorted(stats, key=lambda s: s.count, reverse=True)


# ---------------------------------------------------------------------------
# Evasion sequence detection: deny(A) -> near-term synonymous call(B)
# ---------------------------------------------------------------------------

# Tools that perform network access. A swap is only counted as network evasion
# when BOTH sides express an external-access intent, to avoid flagging an
# unrelated local bash command followed by a web_search.
_NETWORK_TOOLS = {"web_fetch", "web_search", "curl", "wget"}
# In bash, tokens that indicate an outbound fetch / download.
_NET_TOKENS = (
    "curl",
    "wget",
    "http://",
    "https://",
    "fetch(",
    "urllib",
    "requests.get",
    "nc ",
    "/dev/tcp",
    "invokE-WebRequest".lower(),
    "iwr ",
)
# Within this window after a deny, a synonymous/rewritten call is suspicious.
EVASION_WINDOW_SEC = 120


@dataclass(slots=True)
class EvasionSignal:
    evidence_id: str
    t_deny: str
    denied_tool: str
    denied_command: str
    t_follow: str
    follow_tool: str
    follow_command: str
    follow_decision: str
    delta_sec: float
    kind: str  # tool_swap | command_rewrite | blocked_then_allowed


def _networkish(command: str) -> bool:
    c = (command or "").lower()
    return any(tok in c for tok in _NET_TOKENS)


def _network_intent(tool: str, command: str, url: str | None) -> bool:
    """True if this event clearly expresses an outbound-access intent."""
    if tool in _NETWORK_TOOLS:
        return True
    return _networkish(command or "") or bool(url and url.startswith("http"))


def _normalized_cmd(command: str | None) -> str:
    """Collapse whitespace/lowercase for an approximate command equality."""
    return " ".join((command or "").lower().split())


# Intent signature for the false-positive proxy. Generic shell prefixes
# (cd/ls/echo/env) are ignored; the signature is built from meaningful verbs
# (network fetch, package manager, permission/chmod, process control) plus the
# first distinctive path/host token, so a deny/allow pair counts only when the
# user retries the SAME substantive action.
_INTENT_VERBS = [
    ("net", ("curl", "wget", "urllib", "requests.", "fetch(", "http://", "https://")),
    ("pkg", ("npm ", "pip", "brew ", "uv ", "pnpm", "yarn", "cargo", "go build")),
    ("perm", ("chmod", "chown", "xattr")),
    ("proc", ("kill", "pkill", "lsof", "launchctl", "osascript")),
    ("archive", ("tar ", "zip ", "unzip", "7z")),
    ("convert", ("soffice", "ffmpeg", "convert ", "magick", "pdftotext")),
    # NB: generic echo / bare redirection are intentionally excluded — they
    # appear in nearly every exploratory shell line and produced FP noise.
]


def _target_token(command: str) -> str:
    """Distinctive target for same-intent matching: host, sensitive path, file."""
    c = command.lower()
    m = re.search(r"https?://([\w.\-]+)", c)
    if m:
        return m.group(1)
    m = re.search(r"(/etc/\S+|/usr/\S+|~/[\w./\-]+)", c)
    if m:
        return m.group(1)
    m = re.search(r"[\w./\-]+\.(sh|py|js|ts|heic|jpg|png|pdf|pptx|docx|txt)\b", c)
    if m:
        return m.group(0)
    return ""


def _intent_signature(command: str | None) -> str | None:
    c = (command or "").lower()
    verbs = [tag for tag, toks in _INTENT_VERBS if any(t in c for t in toks)]
    if not verbs:
        return None
    target = _target_token(c)
    return ",".join(verbs) + (f"@{target}" if target else "")


def detect_evasion(db_path: str | Path, *, window_sec: int = EVASION_WINDOW_SEC):
    """Find deny -> synonymous/rewritten near-term calls in chronological order."""
    con, _ = _read_only_con(db_path)
    try:
        rows = con.execute(
            """
            SELECT ts, tool, category, decision,
                   json_extract_string(args_json, '$.command') AS command,
                   json_extract_string(args_json, '$.url') AS url
            FROM events ORDER BY ts
            """
        ).fetchall()
    finally:
        con.close()

    signals: list[EvasionSignal] = []
    sig_no = 0
    denied_recent: list[tuple] = []
    for row in rows:
        ts, tool, category, decision, command, url = row
        # expire old denies
        denied_recent = [d for d in denied_recent if (ts - d[0]).total_seconds() <= window_sec]

        if decision == "deny":
            denied_recent.append(row)
            continue

        if decision != "allow":
            continue

        for d_ts, d_tool, _d_cat, _d_dec, d_cmd, d_url in denied_recent:
            delta = (ts - d_ts).total_seconds()
            denied_net = _network_intent(d_tool, d_cmd or "", d_url)
            follow_net = _network_intent(tool, command or "", url)
            kind = None
            # Both sides must be about outbound access for a swap to count.
            cross_tool = (
                d_tool != tool
                and (d_tool in _NETWORK_TOOLS or tool in _NETWORK_TOOLS)
                and denied_net
                and follow_net
            )
            if cross_tool:
                kind = "tool_swap"
            elif (
                d_tool == tool == "bash"
                and denied_net
                and follow_net
                and category in ("network", "local_network", "explicit-allow", "auto-approve")
                and _normalized_cmd(d_cmd) != _normalized_cmd(command)
            ):
                # same tool, network intent, rewritten command now allowed
                kind = "blocked_then_allowed"
            if kind:
                sig_no += 1
                signals.append(
                    EvasionSignal(
                        evidence_id=f"EVASION-{sig_no:04d}",
                        t_deny=str(d_ts),
                        denied_tool=d_tool,
                        denied_command=(d_cmd or d_url or "")[:200],
                        t_follow=str(ts),
                        follow_tool=tool,
                        follow_command=(command or url or "")[:200],
                        follow_decision=decision,
                        delta_sec=round(delta, 1),
                        kind=kind,
                    )
                )
                denied_recent = [d for d in denied_recent if d[0] != d_ts]
                break
    return signals


# ---------------------------------------------------------------------------
# False-positive proxy: deny -> same-intent allow within N minutes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FPProxy:
    evidence_id: str
    t_deny: str
    tool: str
    denied_summary: str
    t_allow: str
    allow_summary: str
    delta_sec: float
    pattern: str


FP_WINDOW_SEC = 300


def detect_fp_proxies(db_path: str | Path, *, window_sec: int = FP_WINDOW_SEC):
    """A deny followed by a same-substantive-intent allow = policy-tuning hint.

    Uses :func:`_intent_signature` (meaningful verb + target), not the first
    shell token, so a generic ``cd``/``ls`` prefix never counts as the same
    action. Only pairs where both sides share a non-empty signature qualify.
    """
    con, _ = _read_only_con(db_path)
    try:
        rows = con.execute(
            """
            SELECT ts, tool, category, decision, reason,
                   json_extract_string(args_json, '$.command') AS command
            FROM events ORDER BY ts
            """
        ).fetchall()
    finally:
        con.close()

    proxies: list[FPProxy] = []
    pno = 0
    pending: list[tuple] = []
    for row in rows:
        ts, tool, category, decision, reason, command = row
        pending = [d for d in pending if (ts - d[0]).total_seconds() <= window_sec]
        if decision == "deny":
            sig = _intent_signature(command)
            if sig:  # only remember denials with a classifiable intent
                pending.append((ts, tool, reason, command, sig))
            continue
        if decision != "allow":
            continue
        allow_sig = _intent_signature(command)
        if not allow_sig:
            continue
        for d in pending:
            d_ts, d_tool, d_reason, d_cmd, d_sig = d
            if d_tool == tool and d_sig == allow_sig:
                pno += 1
                proxies.append(
                    FPProxy(
                        evidence_id=f"FP-{pno:04d}",
                        t_deny=str(d_ts),
                        tool=tool,
                        denied_summary=(d_cmd or d_reason or "")[:160],
                        t_allow=str(ts),
                        allow_summary=(command or "")[:160],
                        delta_sec=round((ts - d_ts).total_seconds(), 1),
                        pattern=allow_sig,
                    )
                )
                pending = [x for x in pending if x is not d]
                break
    return proxies
