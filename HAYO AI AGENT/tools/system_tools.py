"""
System-level tools: shell command execution.

Every shell call goes through `run_powershell` which:
  1. Runs the safety guard (DESTRUCTIVE_PATTERNS).
  2. Returns a structured dict (stdout, stderr, returncode, truncated).
  3. Caps output to MAX_OUTPUT_CHARS so the LLM doesn't drown.

Tools are exposed via @tool so LangGraph can bind them to the model.
"""

from __future__ import annotations

import os
import subprocess
from typing import Annotated

from langchain_core.tools import tool

from config import PS_TIMEOUT
from core.safety import needs_human_approval, redact_secrets

MAX_OUTPUT_CHARS = 8000  # truncation budget per call


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    half = MAX_OUTPUT_CHARS // 2
    head = text[:half]
    tail = text[-half:]
    return f"{head}\n\n...[TRUNCATED {len(text) - MAX_OUTPUT_CHARS} chars]...\n\n{tail}", True


@tool
def run_powershell(
    command: Annotated[str, "The exact PowerShell command to execute on the user's Windows machine."],
    workdir: Annotated[str, "Working directory. Use '.' for current."] = ".",
) -> str:
    """
    Run a PowerShell command and return combined stdout+stderr.

    Use this for: file listings, git operations, package installs, system info,
    creating folders, copying files, running Python scripts, etc.

    For destructive commands (rm -rf, format, shutdown, registry deletes) the
    request will be flagged for human approval before execution.
    """
    needs_approval, pattern = needs_human_approval(command)
    if needs_approval:
        # The worker node intercepts this prefix and asks the user.
        return (
            f"__HITL_REQUIRED__\nMatched destructive pattern: '{pattern}'\n"
            f"Command: {command}\n"
            f"Workdir: {workdir}\n"
            f"Awaiting human approval before execution."
        )

    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=os.path.abspath(workdir),
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {PS_TIMEOUT}s] Command: {command}"
    except FileNotFoundError:
        return "[ERROR] powershell.exe not found. Is this Windows?"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined = stdout
    if stderr:
        combined += f"\n[STDERR]\n{stderr}"

    combined, truncated = _truncate(combined)
    combined = redact_secrets(combined)

    header = f"[exit={completed.returncode}{' truncated' if truncated else ''}]\n"
    return header + combined


@tool
def run_cmd(
    command: Annotated[str, "The exact CMD command to run."],
    workdir: Annotated[str, "Working directory."] = ".",
) -> str:
    """
    Run a classic Windows CMD command. Prefer run_powershell for most things.
    Use this only when a tool truly needs cmd.exe (e.g. legacy .bat files).
    """
    needs_approval, pattern = needs_human_approval(command)
    if needs_approval:
        return (
            f"__HITL_REQUIRED__\nMatched destructive pattern: '{pattern}'\n"
            f"Command: {command}\n"
        )

    try:
        completed = subprocess.run(
            command,
            cwd=os.path.abspath(workdir),
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT,
            shell=True,
        )
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {PS_TIMEOUT}s]"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"

    out = (completed.stdout or "") + (
        f"\n[STDERR]\n{completed.stderr}" if completed.stderr else ""
    )
    out, _ = _truncate(out)
    return f"[exit={completed.returncode}]\n{redact_secrets(out)}"


@tool
def get_env(name: Annotated[str, "Environment variable name."]) -> str:
    """Read a single Windows environment variable. Returns empty string if missing."""
    return os.environ.get(name, "")
