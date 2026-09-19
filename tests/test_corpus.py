"""Tests for corpus construction (cards, clusters, markdown chunking)."""

from __future__ import annotations

import pandas as pd

from guardlens.corpus import (
    _chunk_markdown,
    build_event_cards,
    build_event_clusters,
)


def _row(ts, tool="bash", category="read", decision="allow", reason="", command=None):
    return pd.Series(
        {
            "ts": pd.Timestamp(ts, tz="UTC"),
            "tool": tool,
            "category": category,
            "decision": decision,
            "reason": reason,
            "args_json": '{"command": "%s"}' % (command or "") if command else "{}",
            "snapshot_id": None,
            "first_binary": command.split()[0] if command else None,
            "binaries": None,
            "risk_tokens": None,
            "target_paths": "[]",
        }
    )


def test_deny_event_gets_card_with_context_window():
    df = pd.DataFrame(
        [
            _row("2026-09-06T09:00:00Z", command="ls"),
            _row(
                "2026-09-06T09:00:05Z",
                tool="bash",
                category="network",
                decision="deny",
                reason="BLOCKED (network): curl",
                command="curl https://x",
            ),
            _row("2026-09-06T09:00:09Z", tool="web_fetch", category="network", decision="deny"),
        ]
    )
    cards = build_event_cards(df, window=5)
    assert len(cards) == 2  # the two deny events
    card = cards[0]
    assert card.doc_type == "event_card"
    assert card.evidence_id.startswith("EV-")
    assert "curl https://x" in card.text
    assert "前文事件" in card.text and "后文事件" in card.text
    assert "web_fetch" in card.text  # evasion-relevant successor captured


def test_clusters_split_on_idle_gap_and_keep_deny_only():
    df = pd.DataFrame(
        [
            _row("2026-09-06T09:00:00Z"),
            _row("2026-09-06T09:01:00Z", category="network", decision="deny"),
            # >10 min gap -> new cluster
            _row("2026-09-06T09:30:00Z"),
        ]
    )
    clusters = build_event_clusters(df, gap_minutes=10)
    assert len(clusters) == 1  # only the cluster containing a deny survives
    assert clusters[0].evidence_id == "CL-0000"
    assert "拦截: 1" in clusters[0].text


def test_markdown_chunking_respects_h2_and_size():
    md = "# pre\n\n## A\n" + ("x" * 100) + "\n\n## B\n" + ("y" * 3000)
    chunks = _chunk_markdown(md, max_chars=1000, overlap=150)
    assert len(chunks) >= 2  # section B splits
    assert any("x" * 10 in c for c in chunks)
    assert all(len(c) <= 1300 for c in chunks)
