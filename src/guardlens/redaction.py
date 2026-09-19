"""Path / secret redaction for public datasets.

W1 runs on real local data with redaction OFF by default; the pipeline
always *supports* it so the same code produces a shareable sample later.
"""

from __future__ import annotations

import re

# /Users/<name> -> /Users/user (keeps path-shape, removes identity)
_HOME_RE = re.compile(r"/Users/[^/\s\"')]+")

# emails, IPv4 addresses, optional Bearer tokens — cheap high-precision set.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_BEARY_RE = re.compile(r"(?i)(bearer|token|api[_-]?key)\s*[=:]\s*\S+")


def redact_text(text: str | None) -> str | None:
    """Redact identity-ish substrings from a free-text field."""
    if text is None:
        return None
    out = text
    out = _HOME_RE.sub("/Users/user", out)
    out = _BEARY_RE.sub(lambda m: f"{m.group(1)}=***", out)
    out = _EMAIL_RE.sub("redacted@example.com", out)
    out = _IPV4_RE.sub("10.0.0.1", out)
    return out


def redact_path(path: str | None) -> str | None:
    """Redact a single filesystem path."""
    if path is None:
        return None
    return _HOME_RE.sub("/Users/user", path)
