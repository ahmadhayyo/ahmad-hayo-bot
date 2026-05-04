"""
Central configuration loader.

Reads .env once, exposes typed constants the rest of the codebase imports.
Every module in agent/, tools/, core/ should import from here — never call
os.getenv directly. This keeps the surface area for misconfiguration tiny.
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
ProviderName = Literal["anthropic", "google"]
MODEL_PROVIDER: ProviderName = _get("MODEL_PROVIDER", "google").lower()  # type: ignore
if MODEL_PROVIDER not in ("anthropic", "google"):
    raise ValueError(
        f"MODEL_PROVIDER must be 'anthropic' or 'google', got '{MODEL_PROVIDER}'"
    )

# ── Anthropic ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = _get("ANTHROPIC_API_KEY")
ANTHROPIC_AGENT_MODEL: str = _get("ANTHROPIC_AGENT_MODEL", "claude-3-7-sonnet-20250219")
ANTHROPIC_SUMMARIZER_MODEL: str = _get(
    "ANTHROPIC_SUMMARIZER_MODEL", "claude-haiku-4-5-20251001"
)

# ── Google Gemini ────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = _get("GOOGLE_API_KEY")
GOOGLE_AGENT_MODEL: str = _get("GOOGLE_AGENT_MODEL", "gemini-2.0-flash")
GOOGLE_SUMMARIZER_MODEL: str = _get("GOOGLE_SUMMARIZER_MODEL", "gemini-2.0-flash")

# ── Agent behaviour ─────────────────────────────────────────────────────────
MAX_ITERATIONS: int = _get_int("MAX_ITERATIONS", 20)
MAX_HISTORY: int = _get_int("MAX_HISTORY", 15)
PS_TIMEOUT: int = _get_int("PS_TIMEOUT", 120)

# ── Workspace / downloads ───────────────────────────────────────────────────
DEFAULT_WORKSPACE: Path = Path(_get("DEFAULT_WORKSPACE", str(ROOT_DIR)))
DESKTOP_DIR: Path = Path(_get("DESKTOP_DIR", str(Path.home() / "Desktop")))
DOWNLOADS_DIR: Path = Path(_get("DOWNLOADS_DIR", str(Path.home() / "Downloads")))

# ── Browser ─────────────────────────────────────────────────────────────────
# Headful by default — the user wants to SEE the agent driving the browser.
BROWSER_HEADLESS: bool = _get("BROWSER_HEADLESS", "false").lower() == "true"
BROWSER_USER_DATA_DIR: Path = ROOT_DIR / ".browser_profile"

# ── Safety / HITL ────────────────────────────────────────────────────────────
# Patterns that force human approval before execution. Defense-in-depth: even
# if the LLM proposes one of these, the worker pauses and asks the user.
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
    "Invoke-WebRequest -OutFile",  # downloads, not destructive but worth confirming
    "Add-LocalGroupMember",
    "New-LocalUser",
    "reg delete",
    "diskpart",
    "cipher /w",
)


def active_provider_key() -> str:
    """Return the API key for the currently selected provider."""
    if MODEL_PROVIDER == "anthropic":
        return ANTHROPIC_API_KEY
    return GOOGLE_API_KEY


def assert_keys_present() -> None:
    """Fail fast at startup if the active provider has no key."""
    if not active_provider_key():
        provider_var = (
            "ANTHROPIC_API_KEY" if MODEL_PROVIDER == "anthropic" else "GOOGLE_API_KEY"
        )
        raise RuntimeError(
            f"MODEL_PROVIDER='{MODEL_PROVIDER}' but {provider_var} is empty. "
            f"Edit {ENV_PATH} and set it."
        )
