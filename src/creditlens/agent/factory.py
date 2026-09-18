"""Picks the LLM backend: explicit LLM_PROVIDER env var, else whichever
real (non-placeholder) API key is present, preferring Anthropic."""
from __future__ import annotations

import os

from .tools import ToolRuntime

_PLACEHOLDER_VALUES = {"", "your-anthropic-key-here"}


def has_key(env_var: str) -> bool:
    return os.environ.get(env_var, "") not in _PLACEHOLDER_VALUES


def build_agent(runtime: ToolRuntime):
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if not provider:
        if has_key("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif has_key("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            raise RuntimeError(
                "No usable API key found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY in .env."
            )

    if provider == "anthropic":
        from .orchestrator import Agent

        return Agent(runtime)

    if provider == "gemini":
        from .gemini_backend import GeminiAgent

        rate_limit = int(os.environ.get("GEMINI_RATE_LIMIT_PER_MIN", "60"))
        return GeminiAgent(runtime, rate_limit_per_min=rate_limit)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'gemini')")
