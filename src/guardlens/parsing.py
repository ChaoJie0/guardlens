"""Shell command feature extraction for audit ``args.command`` strings.

We never *execute* anything here; the input is an already-recorded command
line. Parsing is best-effort and defensive: malformed/quoted strings yield
empty/partial structures rather than raising, so ingestion never drops a
record because ``shlex`` could not split it.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

# Shell control operators that split a command line into segments.
# Handles |, ||, &, &&, ;, newline (POSIX-ish; sufficient for audit triage).
_SEGMENT_SPLIT = re.compile(r"\|\|?|&&?|;|\n")

# Redirections: > >> 2> 2>&1 >| etc. Captures operator + optional target.
_REDIRECT = re.compile(r"(?:(?P<fd>\d|&)>?>?\|?(?P<target>\S+)?)")
_REDIRECT_TOKENS = {">", ">>", ">|", "2>", "2>>", "&>", "&>>", "1>", "1>>"}

# Tokens that hint at data-exfiltration / privilege escalation semantics.
# Kept deliberately small and high-precision; the plugin itself owns the
# authoritative 148-word risk vocabulary, this is only a derived triage tag.
_RISK_PATTERNS: dict[str, re.Pattern[str]] = {
    "pipe": re.compile(r"(?<!\\)\|"),
    "redirect": re.compile(r">>?"),
    "curl": re.compile(r"\bcurl\b"),
    "wget": re.compile(r"\bwget\b"),
    "network": re.compile(r"\b(curl|wget|nc|ncat|netcat|ssh|scp|rsync)\b"),
    "subshell": re.compile(r"(\$\(|`|&&|;)"),
    "sudo": re.compile(r"\bsudo\b"),
    "chmod": re.compile(r"\bchmod\b"),
    "rm": re.compile(r"\brm\b"),
    "base64": re.compile(r"\bbase64\b"),
    "eval": re.compile(r"\b(eval|exec)\b"),
}


@dataclass(slots=True)
class CommandFeatures:
    """Derived features for one ``args.command`` value."""

    raw: str = ""
    segments: list[str] = field(default_factory=list)
    segment_count: int = 0
    first_binary: str | None = None
    binaries: list[str] = field(default_factory=list)
    redirect_targets: list[str] = field(default_factory=list)
    risk_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "segments": self.segments,
            "segment_count": self.segment_count,
            "first_binary": self.first_binary,
            "binaries": self.binaries,
            "redirect_targets": self.redirect_targets,
            "risk_tokens": self.risk_tokens,
        }


def _split_segments(command: str) -> list[str]:
    """Split a command line on pipe/control operators, keeping segment text.

    Operators inside single/double quotes are ignored (e.g. ``echo 'a|b'``).
    A single left-to-right scan collects segment boundaries, which is safer
    than splitting a masked copy and mapping offsets back afterwards.
    """
    segments: list[str] = []
    chars: list[str] = []
    quote: str | None = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote is not None:
            chars.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            chars.append(ch)
            i += 1
            continue
        # two-char operators first: ||, &&, >>, >| etc. handled by regex set
        if ch in "|;&" or (ch == "\n"):
            seg = "".join(chars).strip()
            if seg:
                segments.append(seg)
            chars = []
            # consume a duplicated operator char (|| &&)
            if i + 1 < n and command[i + 1] == ch and ch in "|&":
                i += 2
            else:
                i += 1
            continue
        chars.append(ch)
        i += 1
    tail = "".join(chars).strip()
    if tail:
        segments.append(tail)
    return segments


def _first_token(segment: str) -> str | None:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        # Unterminated quote etc. Fall back to whitespace split.
        tokens = segment.split()
    return tokens[0] if tokens else None


def extract_command_features(command: str | None) -> CommandFeatures:
    """Parse a recorded shell command into auditable structural features."""
    if not command or not isinstance(command, str):
        return CommandFeatures()

    segments = _split_segments(command)
    binaries: list[str] = []
    redirect_targets: list[str] = []

    for seg in segments:
        try:
            tokens = shlex.split(seg, posix=True)
        except ValueError:
            tokens = seg.split()
        if not tokens:
            continue
        binaries.append(tokens[0])
        # naive redirection target extraction from the token stream
        for i, tok in enumerate(tokens):
            if tok in _REDIRECT_TOKENS and i + 1 < len(tokens):
                redirect_targets.append(tokens[i + 1])

    risk_tokens = sorted({name for name, pat in _RISK_PATTERNS.items() if pat.search(command)})

    return CommandFeatures(
        raw=command,
        segments=segments,
        segment_count=len(segments),
        first_binary=binaries[0] if binaries else None,
        binaries=binaries,
        redirect_targets=sorted(set(redirect_targets)),
        risk_tokens=risk_tokens,
    )


# Helper retained for tests/debugging of redirection regex surface.
def redirect_matches(text: str) -> list[str]:
    return _REDIRECT.findall(text)  # pragma: no cover - thin wrapper
