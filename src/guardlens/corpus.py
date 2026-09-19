"""Offline corpus builder: events -> semantic cards / clusters, reports -> chunks.

W2 scope: build *textual, evidence-linked* documents from the DuckDB events
table and the plugin's ``turn-*.md`` reports, then persist as parquet/json so
the embedding step is decoupled and re-runnable.

Why cards, not raw rows: a deny decision only makes sense relative to the
events around it (``curl`` denied -> ``web_fetch`` to the same URL seconds
later = a possible tool-substitution evasion). Each card therefore embeds a
small before/after event window.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# High-risk allow categories worth card-ifying in addition to deny events.
HIGH_RISK_ALLOW_CATEGORIES = {
    "network",
    "local_network",
    "process",
    "package",
    "vcs_remote",
    "fail-closed",
    "local_exec",
}

WINDOW = 5  # events before/after each anchor
GAP_MINUTES = 10  # session/cluster split on idle gap

_REPORT_H2 = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_TURN_TS = re.compile(r"turn-(\d{4}-\d{2}-\d{2}T[\d-]+)\.md$")


@dataclass(slots=True)
class Document:
    evidence_id: str  # stable, citable id
    doc_type: str  # 'event_card' | 'event_cluster' | 'report_chunk'
    text: str
    ts: str | None
    tool: str | None
    category: str | None
    decision: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_events_df(db_path: str | Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            SELECT ts, ts_raw, tool, category, decision, reason,
                   target_paths, args_json, snapshot_id,
                   segment_count, first_binary, binaries,
                   redirect_targets, risk_tokens
            FROM events
            ORDER BY ts
            """
        ).df()
    finally:
        con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _event_brief(row: pd.Series) -> str:
    """One-line human/embedding-friendly summary of an event."""
    args: dict[str, Any] = {}
    try:
        args = json.loads(row.get("args_json") or "{}")
    except json.JSONDecodeError:
        args = {}
    command = args.get("command")
    url = args.get("url")
    bits = [f"{row['ts'].isoformat()}", f"tool={row['tool']}", f"decision={row['decision']}"]
    if isinstance(command, str) and command:
        bits.append(f"cmd={command.strip()[:300]}")
    if isinstance(url, str) and url:
        bits.append(f"url={url[:200]}")
    reason = (row.get("reason") or "").strip()
    if reason:
        bits.append(f"reason={reason[:300]}")
    risk = row.get("risk_tokens")
    if isinstance(risk, str) and risk not in ("[]", "null"):
        bits.append(f"risk={risk}")
    return " | ".join(bits)


# ---------------------------------------------------------------------------
# Event cards
# ---------------------------------------------------------------------------


def _is_anchor(row: pd.Series) -> bool:
    if row["decision"] == "deny":
        return True
    if row["category"] == "fail-closed":
        return True
    if row["decision"] == "allow" and row["category"] in HIGH_RISK_ALLOW_CATEGORIES:
        # keep high-risk allows but they are numerous for local_exec; always
        # include network/process/package/vcs_remote/fail-closed, sample local_exec
        if row["category"] == "local_exec":
            risk = str(row.get("risk_tokens") or "")
            return any(t in risk for t in ("network", "sudo", "eval", "base64"))
        return True
    return False


def build_event_cards(df: pd.DataFrame, *, window: int = WINDOW) -> list[Document]:
    docs: list[Document] = []
    n = len(df)
    anchor_indices = [i for i in range(n) if _is_anchor(df.iloc[i])]
    for card_no, i in enumerate(anchor_indices):
        row = df.iloc[i]
        lo, hi = max(0, i - window), min(n, i + window + 1)
        before = [_event_brief(df.iloc[j]) for j in range(lo, i)]
        after = [_event_brief(df.iloc[j]) for j in range(i + 1, hi)]

        args: dict[str, Any] = {}
        try:
            args = json.loads(row.get("args_json") or "{}")
        except json.JSONDecodeError:
            args = {}

        lines = [
            f"[证据 {card_no:05d}] 审计事件语义卡",
            f"时间: {row['ts'].isoformat()}",
            f"工具: {row['tool']}  类别: {row['category']}  决策: {row['decision']}",
        ]
        snap = row.get("snapshot_id")
        if isinstance(snap, str) and snap and snap.lower() != "nan":
            lines.append(f"快照: {snap}")
        if isinstance(args.get("command"), str):
            lines.append(f"命令: {args['command'][:1000]}")
        if isinstance(args.get("url"), str):
            lines.append(f"URL: {args['url'][:500]}")
        if row.get("reason"):
            lines.append(f"拦截/放行理由(原文): {row['reason']}")
        if before:
            lines.append("前文事件:")
            lines.extend(f"  - {b}" for b in before)
        if after:
            lines.append("后文事件:")
            lines.extend(f"  - {a}" for a in after)

        docs.append(
            Document(
                evidence_id=f"EV-{card_no:05d}",
                doc_type="event_card",
                text="\n".join(lines),
                ts=row["ts"].isoformat(),
                tool=str(row["tool"]),
                category=str(row["category"]),
                decision=str(row["decision"]),
                metadata={
                    "row_index": int(i),
                    "first_binary": row.get("first_binary"),
                    "risk_tokens": _safe_json(row.get("risk_tokens")),
                    "binaries": _safe_json(row.get("binaries")),
                    "reason": row.get("reason"),
                },
            )
        )
    return docs


def _safe_json(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


# ---------------------------------------------------------------------------
# Event clusters (session-ish windows split by idle gap)
# ---------------------------------------------------------------------------


def build_event_clusters(df: pd.DataFrame, *, gap_minutes: int = GAP_MINUTES) -> list[Document]:
    docs: list[Document] = []
    if df.empty:
        return docs
    gap = timedelta(minutes=gap_minutes)
    start = 0
    cluster_no = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or (df.iloc[i]["ts"] - df.iloc[i - 1]["ts"]) > gap:
            chunk = df.iloc[start:i]
            denies = int((chunk["decision"] == "deny").sum())
            # Only persist clusters that contain something worth asking about.
            if denies > 0 or (chunk["category"] == "fail-closed").any():
                docs.append(_cluster_doc(chunk, cluster_no, gap_minutes))
                cluster_no += 1
            start = i
    return docs


def _cluster_doc(chunk: pd.DataFrame, cluster_no: int, gap_minutes: int) -> Document:
    t0, t1 = chunk.iloc[0]["ts"], chunk.iloc[-1]["ts"]
    total = len(chunk)
    denies = int((chunk["decision"] == "deny").sum())
    top_tools = chunk["tool"].value_counts().head(6).to_dict()
    deny_lines = [
        _event_brief(row)
        for _, row in chunk.iterrows()
        if row["decision"] == "deny" or row["category"] == "fail-closed"
    ][:25]
    lines = [
        f"[簇 {cluster_no:04d}] 事件会话簇（{gap_minutes} 分钟空闲切窗）",
        f"时间范围: {t0.isoformat()} ~ {t1.isoformat()}",
        f"事件数: {total}，拦截: {denies}",
        f"高频工具: {json.dumps(top_tools, ensure_ascii=False)}",
        "拦截/fail-closed 事件:",
        *[f"  - {d}" for d in deny_lines],
    ]
    return Document(
        evidence_id=f"CL-{cluster_no:04d}",
        doc_type="event_cluster",
        text="\n".join(lines),
        ts=t0.isoformat(),
        tool=None,
        category=None,
        decision="deny" if denies else "fail-closed",
        metadata={"event_count": total, "deny_count": denies, "end_ts": t1.isoformat()},
    )


# ---------------------------------------------------------------------------
# Markdown turn reports -> H2 chunks
# ---------------------------------------------------------------------------


def _chunk_markdown(md: str, *, max_chars: int = 1400, overlap: int = 200) -> list[str]:
    """Split a report by H2 sections, further splitting oversized sections."""
    matches = list(_REPORT_H2.finditer(md))
    if not matches:
        sections = [("报告", md)]
    else:
        sections = []
        # preamble before first H2
        if matches[0].start() > 0:
            pre = md[: matches[0].start()].strip()
            if pre:
                sections.append(("概览", pre))
        for j, m in enumerate(matches):
            end = matches[j + 1].start() if j + 1 < len(matches) else len(md)
            sections.append((m.group(1).strip(), md[m.start() : end].strip()))

    chunks: list[str] = []
    for title, body in sections:
        if len(body) <= max_chars:
            chunks.append(body)
            continue
        step = max_chars - overlap
        for k in range(0, len(body), step):
            piece = body[k : k + max_chars]
            chunks.append(f"## {title}（续）\n{piece}" if k else piece)
    return [c for c in chunks if c.strip()]


def build_report_chunks(reports_dir: str | Path) -> list[Document]:
    reports_dir = Path(reports_dir)
    docs: list[Document] = []
    if not reports_dir.exists():
        return docs
    chunk_no = 0
    for md_path in sorted(reports_dir.glob("turn-*.md")):
        try:
            md = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        m = _TURN_TS.search(md_path.name)
        ts = None
        if m:
            ts = m.group(1).replace("-", ":", 2).replace("-", "T", 1)
        for piece in _chunk_markdown(md):
            docs.append(
                Document(
                    evidence_id=f"RP-{chunk_no:05d}",
                    doc_type="report_chunk",
                    text=f"来源回合报告 {md_path.name}\n{piece}",
                    ts=ts,
                    tool=None,
                    category="turn_report",
                    decision=None,
                    metadata={"source_file": md_path.name},
                )
            )
            chunk_no += 1
    return docs


# ---------------------------------------------------------------------------
# Orchestration + persistence
# ---------------------------------------------------------------------------


def build_corpus(
    db_path: str | Path,
    reports_dir: str | Path | None,
    out_dir: str | Path,
) -> list[Document]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_events_df(db_path)
    cards = build_event_cards(df)
    clusters = build_event_clusters(df)
    reports = build_report_chunks(reports_dir) if reports_dir else []
    docs = cards + clusters + reports

    records = [d.to_dict() for d in docs]
    # JSONL keeps nested metadata intact and is trivial to inspect/stream.
    jsonl_path = out_dir / "corpus.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Flat parquet view (metadata as JSON string) for quick analytics.
    flat = pd.DataFrame(
        [
            {
                "evidence_id": d.evidence_id,
                "doc_type": d.doc_type,
                "ts": d.ts,
                "tool": d.tool,
                "category": d.category,
                "decision": d.decision,
                "text": d.text,
                "metadata_json": json.dumps(d.metadata, ensure_ascii=False),
            }
            for d in docs
        ]
    )
    flat.to_parquet(out_dir / "corpus.parquet", index=False)
    return docs


def iter_corpus(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
