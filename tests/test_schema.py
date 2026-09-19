"""Tests for audit record schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guardlens.schema import Event


def _good_kwargs() -> dict:
    return {
        "ts": "2026-09-06T09:25:46.241Z",
        "tool": "bash",
        "category": "network",
        "decision": "deny",
        "reason": "BLOCKED (network, not rollback-able): curl -s",
        "targetPaths": [],
        "args": {"command": "curl -s https://example.com | head -n 10"},
    }


def test_valid_full_record() -> None:
    event = Event.model_validate(_good_kwargs())
    assert event.tool == "bash"
    assert event.decision == "deny"
    assert event.snapshotId is None


def test_optional_fields_default() -> None:
    event = Event.model_validate(
        {
            "ts": "2026-09-06T09:25:22.766Z",
            "tool": "bash",
            "category": "read",
            "decision": "allow",
        }
    )
    assert event.reason == ""
    assert event.targetPaths == []
    assert event.args == {}


def test_unknown_fields_ignored_forward_compat() -> None:
    payload = _good_kwargs() | {"sandbox_permissions": "danger-full-access", "sessionId": "xyz"}
    event = Event.model_validate(payload)
    assert event.decision == "deny"
    assert (
        not hasattr(event, "sandbox_permissions") or "sandbox_permissions" not in event.model_fields
    )


def test_invalid_decision_rejected() -> None:
    payload = _good_kwargs() | {"decision": "maybe"}
    with pytest.raises(ValidationError):
        Event.model_validate(payload)


def test_missing_required_rejected() -> None:
    with pytest.raises(ValidationError):
        Event.model_validate({"tool": "bash", "decision": "allow"})
