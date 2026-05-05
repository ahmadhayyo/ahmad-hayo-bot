"""
Central configuration loader.

Reads .env once, exposes typed constants the rest of the codebase imports.
Every module in agent/, tools/, core/ should import from here — never call
os.getenv directly. This keeps the surface area for misconfiguration tiny.

Supported providers: google, anthropic, openai, deepseek, groq
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# ── Resolve project root and load .env once ──────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent
ENV_PATH: Path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ── Provider selection ──────────────────────────────────────────────────────
ProviderName = Literal["anthropic", "google", "openai", "deepseek", "groq"]
MODEL_PROVIDER: ProviderName = _get("MODEL_PROVIDER", "google").lower()  # type: ignore
if MODEL_PROVIDER not in ("anthropic", "google", "openai", "deepseek", "groq"):
    raise ValueError(
        f"MODEL_PROVIDER must be 'anthropic', 'google', 'openai', 'deepseek', or 'groq', got '{MODEL_PROVIDER}'"
    )

# ── Anthropic ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = _get("ANTHROPIC_API_KEY")
ANTHROPIC_AGENT_MODEL: str = _get("ANTHROPIC_AGENT_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_SUMMARIZER_MODEL: str = _get(
    "ANTHROPIC_SUMMARIZER_MODEL", "claude-haiku-4-5-20251001"
)

# ── Google Gemini ────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = _get("GOOGLE_API_KEY")
GOOGLE_AGENT_MODEL: str = _get("GOOGLE_AGENT_MODEL", "gemini-2.5-flash")
GOOGLE_SUMMARIZER_MODEL: str = _get("GOOGLE_SUMMARIZER_MODEL", "gemini-2.0-flash")

# ── OpenAI (ChatGPT) ────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _get("OPENAI_API_KEY")
OPENAI_AGENT_MODEL: str = _get("OPENAI_AGENT_MODEL", "gpt-4o")
OPENAI_SUMMARIZER_MODEL: str = _get("OPENAI_SUMMARIZER_MODEL", "gpt-4o-mini")

# ── DeepSeek ─────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY: str = _get("DEEPSEEK_API_KEY")
DEEPSEEK_AGENT_MODEL: str = _get("DEEPSEEK_AGENT_MODEL", "deepseek-chat")
DEEPSEEK_SUMMARIZER_MODEL: str = _get("DEEPSEEK_SUMMARIZER_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL: str = _get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ── Groq ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = _get("GROQ_API_KEY")
GROQ_AGENT_MODEL: str = _get("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")
GROQ_SUMMARIZER_MODEL: str = _get("GROQ_SUMMARIZER_MODEL", "llama-3.1-8b-instant")

# ── Agent behaviour ─────────────────────────────────────────────────────────
MAX_ITERATIONS: int = _get_int("MAX_ITERATIONS", 50)
MAX_HISTORY: int = _get_int("MAX_HISTORY", 15)
PS_TIMEOUT: int = _get_int("PS_TIMEOUT", 120)

# ── Workspace / downloads ───────────────────────────────────────────────────
DEFAULT_WORKSPACE: Path = Path(_get("DEFAULT_WORKSPACE", str(ROOT_DIR)))
DESKTOP_DIR: Path = Path(_get("DESKTOP_DIR", str(Path.home() / "Desktop")))
DOWNLOADS_DIR: Path = Path(_get("DOWNLOADS_DIR", str(Path.home() / "Downloads")))

# ── Browser ─────────────────────────────────────────────────────────────────
BROWSER_HEADLESS: bool = _get("BROWSER_HEADLESS", "false").lower() == "true"
BROWSER_USER_DATA_DIR: Path = ROOT_DIR / ".browser_profile"

# ── Safety / HITL ────────────────────────────────────────────────────────────
DESTRUCTIVE_PATTERNS: tuple[str, ...] = (
    "Remove-Item -Recurse",
    "rm -rf",
    "rmdir /s",
    "format ",
    "del /f /s /q",
    "shutdown",
    "Restart-Computer",
    "Stop-Computer",
    "Set-ExecutionPolicy",
    "Invoke-WebRequest -OutFile",
    "Add-LocalGroupMember",
    "New-LocalUser",
    "reg delete",
    "diskpart",
    "cipher /w",
)

# ── All available providers (for UI model selector) ─────────────────────────
AVAILABLE_PROVIDERS: dict[str, dict] = {
    "google": {
        "label": "Google Gemini",
        "icon": "🟦",
        "key_var": "GOOGLE_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "icon": "🟠",
        "key_var": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-haiku-4-5-20251001"],
    },
    "openai": {
        "label": "OpenAI ChatGPT",
        "icon": "🟢",
        "key_var": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    },
    "deepseek": {
        "label": "DeepSeek",
        "icon": "🔵",
        "key_var": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "groq": {
        "label": "Groq",
        "icon": "🟣",
        "key_var": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
}


def active_provider_key() -> str:
    """Return the API key for the currently selected provider."""
    key_map = {
        "anthropic": ANTHROPIC_API_KEY,
        "google": GOOGLE_API_KEY,
        "openai": OPENAI_API_KEY,
        "deepseek": DEEPSEEK_API_KEY,
        "groq": GROQ_API_KEY,
    }
    return key_map.get(MODEL_PROVIDER, "")


def assert_keys_present() -> None:
    """Fail fast at startup if the active provider has no key."""
    if not active_provider_key():
        key_var_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        provider_var = key_var_map.get(MODEL_PROVIDER, "UNKNOWN_KEY")
        raise RuntimeError(
            f"MODEL_PROVIDER='{MODEL_PROVIDER}' but {provider_var} is empty. "
            f"Edit {ENV_PATH} and set it."
        )
