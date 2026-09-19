"""Resolve an OpenAI-compatible LLM endpoint from the local machine.

No API key is ever hard-coded or printed. Resolution order:

1. Explicit env: ``GUARDLENS_LLM_BASE_URL`` + ``GUARDLENS_LLM_API_KEY``
   (optionally ``GUARDLENS_LLM_MODEL``).
2. ``DEEPSEEK_API_KEY`` env / local credentials file -> DeepSeek official
   OpenAI-compatible endpoint.
3. ``VOLC_ARK_API_KEY`` env / local settings + credentials file -> Volcengine
   Ark coding endpoint (also OpenAI-compatible).

The local credentials file is ``~/.config/llm/credentials.yaml`` with top-level
``refs:`` mapping key-name -> value. We parse only the specific keys we need
and never expose the secret.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LEGACY_CREDENTIALS = Path.home() / ".config" / "llm" / "credentials.yaml"
LEGACY_SETTINGS = Path.home() / ".config" / "llm" / "settings.yaml"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ARK_BASE_URL = "https://api.example.com/v1"
ARK_MODEL = "ark-code-latest"
DEEPSEEK_MODEL = "deepseek-flash"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    source: str
    # DeepSeek flash defaults to thinking mode which can return empty
    # immediate content; disable it for grounded Q&A (passed as extra_body).
    disable_thinking: bool = False

    def masked(self) -> str:
        k = self.api_key
        return f"{k[:3]}***{k[-2:]}" if len(k) > 6 else "***"


def _read_credentials_refs() -> dict[str, Any]:
    """Return the ``refs:`` mapping from local credentials, if present."""
    try:
        data = yaml.safe_load(LEGACY_CREDENTIALS.read_text())
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return {}
    refs = data.get("refs") if isinstance(data, dict) else None
    return refs if isinstance(refs, dict) else {}


def resolve_config() -> LLMConfig:
    """Find the first usable OpenAI-compatible endpoint."""
    refs = _read_credentials_refs()

    def env_or_ref(name: str) -> str | None:
        val = os.environ.get(name)
        if val:
            return val
        val = refs.get(name)
        return val if isinstance(val, str) and val else None

    # 1) fully explicit override
    if os.environ.get("GUARDLENS_LLM_API_KEY") and os.environ.get("GUARDLENS_LLM_BASE_URL"):
        return LLMConfig(
            api_key=os.environ["GUARDLENS_LLM_API_KEY"],
            base_url=os.environ["GUARDLENS_LLM_BASE_URL"],
            model=os.environ.get("GUARDLENS_LLM_MODEL", DEEPSEEK_MODEL),
            source="env:GUARDLENS_LLM_*",
        )

    # 2) DeepSeek official
    dk = env_or_ref("DEEPSEEK_API_KEY")
    if dk:
        return LLMConfig(
            api_key=dk,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
            model=os.environ.get("GUARDLENS_LLM_MODEL", DEEPSEEK_MODEL),
            source="DEEPSEEK_API_KEY (local credentials)",
            disable_thinking=True,
        )

    # 3) Volcengine Ark coding endpoint
    ak = env_or_ref("VOLC_ARK_API_KEY")
    if ak:
        return LLMConfig(
            api_key=ak,
            base_url=ARK_BASE_URL,
            model=os.environ.get("GUARDLENS_LLM_MODEL", ARK_MODEL),
            source="VOLC_ARK_API_KEY (local settings)",
        )

    raise RuntimeError(
        "No OpenAI-compatible API key found. Set DEEPSEEK_API_KEY / VOLC_ARK_API_KEY "
        f"in env or {LEGACY_CREDENTIALS} (refs:), or set GUARDLENS_LLM_API_KEY + "
        "GUARDLENS_LLM_BASE_URL."
    )
