"""
Android clipboard tools — uses termux-api or root clipboard access.
Requires: termux-api package (pkg install termux-api) or root.
"""

from __future__ import annotations

import subprocess
from langchain_core.tools import tool


def _shell_cmd(cmd: str, timeout: int = 10) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"[ERROR] {e}"


def _root_cmd(cmd: str, timeout: int = 10) -> str:
    try:
        r = subprocess.run(
            f"su -c '{cmd}'", shell=True,
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def clipboard_get() -> str:
    """قراءة محتوى الحافظة."""
    # Try termux-api first
    result = _shell_cmd("termux-clipboard-get 2>/dev/null")
    if result and "ERROR" not in result:
        return result
    # Fallback: root + service call
    result = _root_cmd("service call clipboard 2 i32 1 i32 0 2>/dev/null")
    return result or "[ERROR] Cannot read clipboard. Install termux-api: pkg install termux-api"


@tool
def clipboard_set(text: str) -> str:
    """نسخ نص إلى الحافظة.
    
    Args:
        text: النص المراد نسخه
    """
    try:
        proc = subprocess.Popen(
            ["termux-clipboard-set"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        proc.communicate(input=text, timeout=10)
        if proc.returncode == 0:
            return f"Copied to clipboard ({len(text)} chars)"
    except FileNotFoundError:
        pass
    except Exception:
        pass
    # Fallback: root am broadcast
    escaped = text.replace("'", "\\'")[:500]
    _root_cmd(f"am broadcast -a clipper.set -e text '{escaped}' 2>/dev/null")
    return f"Copied to clipboard ({len(text)} chars)"


@tool
def clipboard_append(text: str) -> str:
    """إلحاق نص بمحتوى الحافظة الحالي.
    
    Args:
        text: النص المراد إلحاقه
    """
    current = clipboard_get.invoke({})
    if current.startswith("[ERROR]"):
        current = ""
    new_text = current + text
    return clipboard_set.invoke({"text": new_text})
