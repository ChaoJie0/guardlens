"""Pydantic schema for agent-gate audit records (JSONL).

Mirrors the upstream AuditRecord contract:
ts, tool, category, decision, reason, targetPaths, args, snapshotId?
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["allow", "deny"]


class Event(BaseModel):
    """One validated agent-gate audit event.

    ``model_config`` tolerates unknown fields (forward compatibility with
    newer plugin versions such as ``sandbox_permissions`` / ``justification``)
    while still enforcing the core contract.
    """

    model_config = ConfigDict(extra="ignore")

    ts: str = Field(..., min_length=1)
    tool: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    decision: Decision
    reason: str = ""
    targetPaths: list[str] = Field(default_factory=list)
    args: dict[str, Any] = Field(default_factory=dict)
    snapshotId: str | None = None
