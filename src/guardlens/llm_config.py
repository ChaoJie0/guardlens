"""Resolve an OpenAI-compatible LLM endpoint from the environment.

No API key is ever hard-coded, logged or printed. Resolution order:

1. ``GUARDLENS_LLM_API_KEY`` + ``GUARDLENS_LLM_BASE_URL``
   (optionally ``GUARDLENS_LLM_MODEL``) — fully explicit, any provider.
2. ``OPENAI_API_KEY`` (+ optional ``OPENAI_BASE_URL`` / ``GUARDLENS_LLM_MODEL``)
   — the standard variable names used by the OpenAI SDK and most gateways.

Nothing outside the repository is read, so the same code works on any machine:
export the variables in your shell, or put them in a local ``.env`` (untracked).

GuardLens only needs an OpenAI-compatible *chat completions* endpoint; the
model never touches raw logs, it only organises retrieved, cited evidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    source: str
    # Some providers (e.g. thinking-mode models) attach an empty reasoning
    # preamble; disable it for grounded Q&A. Passed as extra_body.
    disable_thinking: bool = False

    def masked(self) -> str:
        k = self.api_key
        return f"{k[:3]}***{k[-2:]}" if len(k) > 6 else "***"


def resolve_config() -> LLMConfig:
    """Find the first usable OpenAI-compatible endpoint."""
    # 1) explicit GuardLens override
    key = os.environ.get("GUARDLENS_LLM_API_KEY")
    base = os.environ.get("GUARDLENS_LLM_BASE_URL")
    if key and base:
        return LLMConfig(
            api_key=key,
            base_url=base,
            model=os.environ.get("GUARDLENS_LLM_MODEL", DEFAULT_MODEL),
            source="env:GUARDLENS_LLM_*",
        )

    # 2) conventional OpenAI SDK variables
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return LLMConfig(
            api_key=key,
            base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("GUARDLENS_LLM_MODEL", DEFAULT_MODEL),
            source="env:OPENAI_API_KEY",
        )

    raise RuntimeError(
        "No OpenAI-compatible API key found. Set GUARDLENS_LLM_API_KEY + "
        "GUARDLENS_LLM_BASE_URL (and optionally GUARDLENS_LLM_MODEL), or set "
        "OPENAI_API_KEY (+ OPENAI_BASE_URL). Retrieval, embedding and analytics "
        "run without a key; only the LLM answer step needs one."
    )
