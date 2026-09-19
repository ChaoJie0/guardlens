"""Tests for shell command feature extraction."""

from __future__ import annotations

from guardlens.parsing import extract_command_features
from guardlens.redaction import redact_path, redact_text


def test_none_and_empty() -> None:
    f = extract_command_features(None)
    assert f.segment_count == 0
    assert f.first_binary is None
    assert f.risk_tokens == []


def test_simple_command() -> None:
    f = extract_command_features("ls -la")
    assert f.segment_count == 1
    assert f.first_binary == "ls"
    assert f.binaries == ["ls"]
    assert f.segments == ["ls -la"]


def test_pipe_segments_and_risk_tokens() -> None:
    # The canonical evasion-shaped line from the real audit corpus:
    # curl piped to head — network + pipe + redirect-capable first segment.
    f = extract_command_features("curl -s https://example.com | head -n 10")
    assert f.segment_count == 2
    assert f.first_binary == "curl"
    assert f.binaries == ["curl", "head"]
    assert {"curl", "network", "pipe"}.issubset(set(f.risk_tokens))


def test_redirect_target_captured() -> None:
    f = extract_command_features("echo hi > /tmp/out.txt && cat /tmp/out.txt")
    assert f.segment_count == 2
    assert "/tmp/out.txt" in f.redirect_targets
    assert "redirect" in f.risk_tokens


def test_quoted_pipe_is_not_segment_split() -> None:
    f = extract_command_features("echo 'a|b'")
    assert f.segment_count == 1
    assert f.first_binary == "echo"


def test_unterminated_quote_is_tolerated() -> None:
    f = extract_command_features('echo "oops')
    assert f.segment_count == 1
    assert f.first_binary == "echo"


def test_redaction() -> None:
    assert redact_path("/Users/user/secret") == "/Users/user/secret"
    out = redact_text("contact me at bob@example.com from /Users/user ok?")
    assert "bob@example.com" not in out
    assert "/Users/user" in out
    assert redact_text(None) is None
