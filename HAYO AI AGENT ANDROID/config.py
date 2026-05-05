"""
Central configuration loader for HAYO AI Agent (Android Edition).

Reads .env once, exposes typed constants the rest of the codebase imports.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()

# ── Model Provider ──────────────────────────────────────────────────────────
ProviderName = Literal["anthropic", "google", "openai", "deepseek"]
MODEL_PROVIDER: ProviderName = _get("MODEL_PROVIDER", "google").lower()  # type: ignore
if MODEL_PROVIDER not in ("anthropic", "google", "openai", "deepseek"):
    raise ValueError(
        f"MODEL_PROVIDER must be 'anthropic', 'google', 'openai', or 'deepseek', got '{MODEL_PROVIDER}'"
    )

# ── Google Gemini ───────────────────────────────────────────────────────────
GOOGLE_API_KEY = _get("GOOGLE_API_KEY")
GOOGLE_AGENT_MODEL = _get("GOOGLE_AGENT_MODEL", "gemini-2.5-flash")
GOOGLE_SUMMARIZER_MODEL = _get("GOOGLE_SUMMARIZER_MODEL", "gemini-2.0-flash")

# ── Anthropic Claude ────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")
ANTHROPIC_AGENT_MODEL = _get("ANTHROPIC_AGENT_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_SUMMARIZER_MODEL = _get("ANTHROPIC_SUMMARIZER_MODEL", "claude-haiku-4-5-20251001")

# ── OpenAI ChatGPT ──────────────────────────────────────────────────────────
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_AGENT_MODEL = _get("OPENAI_AGENT_MODEL", "gpt-4o")
OPENAI_SUMMARIZER_MODEL = _get("OPENAI_SUMMARIZER_MODEL", "gpt-4o-mini")

# ── DeepSeek ────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = _get("DEEPSEEK_API_KEY")
DEEPSEEK_AGENT_MODEL = _get("DEEPSEEK_AGENT_MODEL", "deepseek-chat")
DEEPSEEK_SUMMARIZER_MODEL = _get("DEEPSEEK_SUMMARIZER_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = _get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ── Available Providers (for UI display) ────────────────────────────────────
AVAILABLE_PROVIDERS = {
    "google": {
        "label": "Google Gemini",
        "icon": "🟦",
        "key_var": "GOOGLE_API_KEY",
        "models": {"agent": GOOGLE_AGENT_MODEL, "summarizer": GOOGLE_SUMMARIZER_MODEL},
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "icon": "🟠",
        "key_var": "ANTHROPIC_API_KEY",
        "models": {"agent": ANTHROPIC_AGENT_MODEL, "summarizer": ANTHROPIC_SUMMARIZER_MODEL},
    },
    "openai": {
        "label": "OpenAI ChatGPT",
        "icon": "🟢",
        "key_var": "OPENAI_API_KEY",
        "models": {"agent": OPENAI_AGENT_MODEL, "summarizer": OPENAI_SUMMARIZER_MODEL},
    },
    "deepseek": {
        "label": "DeepSeek",
        "icon": "🔵",
        "key_var": "DEEPSEEK_API_KEY",
        "models": {"agent": DEEPSEEK_AGENT_MODEL, "summarizer": DEEPSEEK_SUMMARIZER_MODEL},
    },
}

# ── Agent Behavior ──────────────────────────────────────────────────────────
MAX_ITERATIONS = int(_get("MAX_ITERATIONS", "50"))
MAX_HISTORY = int(_get("MAX_HISTORY", "15"))
SHELL_TIMEOUT = int(_get("SHELL_TIMEOUT", "120"))

# ── Destructive command patterns (Android/Linux) ───────────────────────────
DESTRUCTIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+~", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+\*", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=.*of=/dev/", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bfactory.?reset\b", re.IGNORECASE),
    re.compile(r"\bwipe\s+data\b", re.IGNORECASE),
    re.compile(r"\bpm\s+uninstall\b", re.IGNORECASE),
    re.compile(r"\bpm\s+clear\b", re.IGNORECASE),
    re.compile(r"\bsettings\s+put\s+global\b", re.IGNORECASE),
    re.compile(r"\bsu\s+-c\s+.*rm\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+777\s+/", re.IGNORECASE),
]


def active_provider_key() -> str:
    """Return the env-var name for the currently active provider's API key."""
    return AVAILABLE_PROVIDERS.get(MODEL_PROVIDER, {}).get("key_var", "GOOGLE_API_KEY")


def assert_keys_present() -> None:
    """Raise if the selected provider's API key is empty."""
    key_var = active_provider_key()
    if not os.getenv(key_var):
        raise EnvironmentError(
            f"MODEL_PROVIDER='{MODEL_PROVIDER}' but {key_var} is empty.\n"
            f"Set {key_var} in your .env file."
        )
