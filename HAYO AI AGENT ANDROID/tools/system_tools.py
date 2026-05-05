"""
Android system tools — shell execution, environment, file operations via bash.
Uses 'su -c' for root commands when needed.
"""

from __future__ import annotations

import os
import subprocess
from langchain_core.tools import tool

_TIMEOUT = int(os.getenv("SHELL_TIMEOUT", "120"))


def _run(cmd: str, timeout: int | None = None, use_root: bool = False) -> str:
    """Run a shell command and return combined stdout+stderr."""
    if use_root:
        cmd = f"su -c '{cmd}'"
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout or _TIMEOUT,
        )
        out = (r.stdout + r.stderr).strip()
        return f"[exit={r.returncode}]\n{out}" if out else f"[exit={r.returncode}] (no output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout or _TIMEOUT}s"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def run_shell(command: str) -> str:
    """تنفيذ أمر shell/bash على الجهاز.
    يستخدم bash العادي. للأوامر التي تحتاج صلاحيات root استخدم run_root.
    
    Args:
        command: الأمر المراد تنفيذه
    """
    from core.safety import needs_human_approval
    flagged, pattern = needs_human_approval(command)
    if flagged:
        return f"__HITL_REQUIRED__\nCommand: {command}\nPattern: {pattern}"
    return _run(command)


@tool
def run_root(command: str) -> str:
    """تنفيذ أمر بصلاحيات root (su -c).
    يستخدم للأوامر التي تحتاج صلاحيات مرتفعة مثل التحكم بالنظام.
    
    Args:
        command: الأمر المراد تنفيذه بصلاحيات root
    """
    from core.safety import needs_human_approval
    flagged, pattern = needs_human_approval(command)
    if flagged:
        return f"__HITL_REQUIRED__\nCommand: su -c '{command}'\nPattern: {pattern}"
    return _run(command, use_root=True)


@tool
def get_env(name: str) -> str:
    """قراءة متغير بيئة.
    
    Args:
        name: اسم المتغير
    """
    return os.getenv(name, f"(not set: {name})")
