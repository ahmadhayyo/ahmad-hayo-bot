"""
Safety guardrails for HAYO AI Agent (Android Edition).

Every dangerous command flows through needs_human_approval() before execution.
"""

from __future__ import annotations

import re
from config import DESTRUCTIVE_PATTERNS


def needs_human_approval(command: str) -> tuple[bool, str]:
    """Check if a shell command requires user confirmation."""
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            return True, pattern.pattern
    return False, ""


def redact_secrets(text: str) -> str:
    """Scrub API keys / tokens before logging."""
    patterns = [
        r"sk-ant-[A-Za-z0-9_-]{20,}",
        r"AIza[A-Za-z0-9_-]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"ghp_[A-Za-z0-9]{30,}",
        r"\b[A-Za-z0-9]{40,}\b",
    ]
    out = text
    for p in patterns:
        out = re.sub(p, "[REDACTED]", out)
    return out
