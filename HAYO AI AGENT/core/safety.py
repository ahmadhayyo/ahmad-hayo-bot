"""
Safety guardrails — pure functions, no side effects.

These are NOT 'restrictions on the agent' — they exist so that an LLM mistake
doesn't wipe the user's drive. Every dangerous shell command flows through
needs_human_approval() before reaching subprocess.
"""

from __future__ import annotations

from config import DESTRUCTIVE_PATTERNS


def needs_human_approval(command: str) -> tuple[bool, str]:
    """
    Decide whether a shell command must be paused for user confirmation.

    Returns (needs_approval, matched_pattern). If needs_approval is False,
    matched_pattern is an empty string.
    """
    lower = command.lower()
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.lower() in lower:
            return True, pattern
    return False, ""


def redact_secrets(text: str) -> str:
    """
    Scrub things that look like API keys / tokens before logging or sending
    back to the LLM. Cheap heuristics — not a security boundary, just hygiene.
    """
    import re

    # AWS-style, Anthropic sk-ant-, Google AIza, generic 30+ alnum token
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
