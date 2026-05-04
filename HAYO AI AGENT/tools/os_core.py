"""
tools/os_core.py — Secure System Control Tools

Provides three LangChain @tool functions for safe Windows OS interaction:
  1. execute_powershell   — Run PowerShell commands with HITL guard for destructive ops.
  2. read_file_content    — Safe file reader with graceful error handling.
  3. manage_files         — shutil-based file/directory operations (no deletion exposed).

SECURITY MODEL
--------------
execute_powershell checks the command against RISKY_KEYWORDS before running it.
If a match is found, the function does NOT execute the command. Instead it returns
the sentinel string  HITL_FLAG + command  so the calling WorkerNode can trigger
the Human-in-the-Loop approval flow via LangGraph's interrupt() mechanism.
"""

import os
import shutil
import subprocess
from langchain_core.tools import tool

# ── Sentinel returned to WorkerNode when approval is needed ─────────────────
HITL_FLAG = "HITL_APPROVAL_REQUIRED:"

# ── Keywords that classify a PowerShell command as destructive ───────────────
RISKY_KEYWORDS: list[str] = [
    "remove-item",
    "rm ",
    "del ",
    "rmdir",
    "rd ",
    "erase ",
    "format-volume",       # ONLY block format-volume (disk formatting), NOT Format-Table/Format-List
    "format c:",           # explicit disk format commands
    "format d:",
    "format e:",
    "clear-disk",
    "initialize-disk",
    "stop-computer",
    "restart-computer",
    "invoke-expression",   # iex — common in malicious payloads
    "iex(",
    "downloadstring",
    "net user",            # account manipulation
    "reg delete",
    "bcdedit",
    "diskpart",
]

# ── PowerShell timeout (seconds) ─────────────────────────────────────────────
PS_TIMEOUT: int = int(os.getenv("PS_TIMEOUT", "30"))

# ── Slow cmdlets that must be blocked (they timeout the connection) ───────────
# Each entry is (blocked_pattern, fast_replacement_hint)
SLOW_CMDLETS: list[tuple[str, str]] = [
    (
        "get-computerinfo",
        "Use instead: (Get-Item 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion').GetValue('ProductName')"
    ),
    (
        "get-counter",
        "Use instead: Get-WmiObject Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory"
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — execute_powershell
# ─────────────────────────────────────────────────────────────────────────────

@tool
def execute_powershell(command: str) -> str:
    """
    Execute a PowerShell command on the local Windows machine and return its output.

    SECURITY: If the command contains any destructive keyword (rm, del, Remove-Item,
    Format, etc.) it will NOT be executed. Instead, a special flag string is returned
    so the agent can pause and ask the user for explicit approval before proceeding.

    Args:
        command: The PowerShell command or script block to run.

    Returns:
        stdout + stderr output as a single string, or a HITL sentinel string
        if the command requires human approval before execution.
    """
    cmd_lower = command.lower()

    # ── Slow-cmdlet guard (blocks before execution, returns fast error) ───────
    for pattern, hint in SLOW_CMDLETS:
        if pattern in cmd_lower:
            return (
                f"❌ BLOCKED: '{pattern}' is prohibited — it takes 30-90 seconds "
                f"and will cause a timeout.\n{hint}\n"
                "Rewrite the command using the suggested alternative and try again."
            )

    # ── Destructive-command guard ────────────────────────────────────────────
    for keyword in RISKY_KEYWORDS:
        if keyword in cmd_lower:
            return f"{HITL_FLAG}{command}"

    # ── Safe execution ───────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT,
            shell=False,   # avoids shell injection; we build argv list ourselves
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if not stdout and not stderr:
            return "✅ Command executed successfully (no output)."

        output = stdout
        if stderr:
            output += f"\n⚠️ STDERR:\n{stderr}"

        return output

    except subprocess.TimeoutExpired:
        return f"❌ ERROR: Command timed out after {PS_TIMEOUT} seconds."
    except FileNotFoundError:
        return "❌ ERROR: PowerShell executable not found. Ensure PowerShell is installed."
    except Exception as exc:
        return f"❌ ERROR: {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — read_file_content
# ─────────────────────────────────────────────────────────────────────────────

@tool
def read_file_content(filepath: str) -> str:
    """
    Read and return the full text content of a file on the local filesystem.

    Useful for inspecting source code, log files, config files, or any text
    document the agent needs to reason about.

    Args:
        filepath: Absolute or relative path to the file.

    Returns:
        The file content as a UTF-8 string, or an error message.
    """
    try:
        abs_path = os.path.abspath(filepath)
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()

        if not content.strip():
            return f"📄 File '{abs_path}' exists but is empty."

        # Limit to 100 KB to avoid flooding the context window
        if len(content) > 102_400:
            content = content[:102_400] + "\n\n[... TRUNCATED — file exceeds 100 KB ...]"

        return content

    except FileNotFoundError:
        return f"❌ ERROR: File not found — '{filepath}'"
    except PermissionError:
        return f"❌ ERROR: Permission denied reading '{filepath}'"
    except IsADirectoryError:
        return f"❌ ERROR: '{filepath}' is a directory, not a file. Use manage_files(action='list') instead."
    except Exception as exc:
        return f"❌ ERROR: {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — manage_files
# ─────────────────────────────────────────────────────────────────────────────

@tool
def manage_files(action: str, path: str, dest: str = "") -> str:
    """
    Perform safe file-system operations using Python's os and shutil modules.

    Supported actions:
      - 'create_dir' : Create directory tree at `path` (mkdir -p equivalent).
      - 'list'       : List contents of directory at `path`.
      - 'exists'     : Check whether `path` exists (returns 'True' or 'False').
      - 'copy'       : Copy file/directory from `path` to `dest`.
      - 'move'       : Move file/directory from `path` to `dest`.
      - 'info'       : Return size and modification time of a path.

    NOTE: Deletion is intentionally excluded. Use execute_powershell for deletions
    (which will trigger the HITL confirmation flow).

    Args:
        action: One of the supported action strings listed above.
        path:   Source path for the operation.
        dest:   Destination path (required for 'copy' and 'move').

    Returns:
        A human-readable result string or an error message.
    """
    try:
        abs_path = os.path.abspath(path)
        abs_dest = os.path.abspath(dest) if dest else ""

        if action == "create_dir":
            os.makedirs(abs_path, exist_ok=True)
            return f"✅ Directory created (or already exists): {abs_path}"

        elif action == "list":
            if not os.path.isdir(abs_path):
                return f"❌ ERROR: '{abs_path}' is not a directory."
            items = os.listdir(abs_path)
            if not items:
                return f"📁 Directory '{abs_path}' is empty."
            lines = []
            for item in sorted(items):
                full = os.path.join(abs_path, item)
                tag = "📁" if os.path.isdir(full) else "📄"
                lines.append(f"  {tag} {item}")
            return f"📂 Contents of '{abs_path}':\n" + "\n".join(lines)

        elif action == "exists":
            return str(os.path.exists(abs_path))

        elif action == "copy":
            if not abs_dest:
                return "❌ ERROR: 'dest' parameter is required for the 'copy' action."
            if os.path.isdir(abs_path):
                shutil.copytree(abs_path, abs_dest, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
                shutil.copy2(abs_path, abs_dest)
            return f"✅ Copied '{abs_path}' → '{abs_dest}'"

        elif action == "move":
            if not abs_dest:
                return "❌ ERROR: 'dest' parameter is required for the 'move' action."
            os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
            shutil.move(abs_path, abs_dest)
            return f"✅ Moved '{abs_path}' → '{abs_dest}'"

        elif action == "info":
            if not os.path.exists(abs_path):
                return f"❌ ERROR: Path does not exist — '{abs_path}'"
            stat = os.stat(abs_path)
            import datetime
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_kb = stat.st_size / 1024
            kind = "Directory" if os.path.isdir(abs_path) else "File"
            return (
                f"ℹ️ {kind}: {abs_path}\n"
                f"   Size  : {size_kb:.2f} KB\n"
                f"   Modified: {mtime}"
            )

        else:
            return (
                f"❌ ERROR: Unknown action '{action}'. "
                "Valid actions: create_dir, list, exists, copy, move, info"
            )

    except PermissionError:
        return f"❌ ERROR: Permission denied on '{path}'"
    except FileNotFoundError:
        return f"❌ ERROR: Path not found — '{path}'"
    except Exception as exc:
        return f"❌ ERROR: {type(exc).__name__}: {exc}"
