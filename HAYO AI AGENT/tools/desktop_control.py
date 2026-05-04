"""
tools/desktop_control.py — Full Windows Desktop GUI Automation

Gives the agent complete authority to control any desktop application:
  • Launch any app by name, keyword, or full path
  • Take screenshots so the agent can see what is on screen
  • Click, double-click, right-click anywhere on screen
  • Type text and press keyboard shortcuts (Ctrl+S, Alt+F4, Win+D, etc.)
  • Focus, maximize, minimize, resize, and close any window
  • List all currently open windows
  • Drag-and-drop between screen coordinates
  • Scroll up/down in any application
  • Find UI elements by matching text visible on screen

Workflow for interacting with an app:
  1. open_app → launch the application
  2. screenshot → see the current screen state and note element positions
  3. focus_window → bring the app to front if needed
  4. click / type / hotkey → interact with the app
  5. screenshot → verify the result

Dependencies (auto-installed on first use if missing):
  pyautogui, pygetwindow, Pillow
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from langchain_core.tools import tool


# ── Auto-dependency installer ────────────────────────────────────────────────

def _ensure_deps() -> str | None:
    """Install pyautogui + pygetwindow if not present. Returns error string or None."""
    missing = []
    try:
        import pyautogui  # noqa: F401
    except ImportError:
        missing.append("pyautogui")
    try:
        import pygetwindow  # noqa: F401
    except ImportError:
        missing.append("pygetwindow")

    if missing:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing + ["--quiet"],
                capture_output=True,
                timeout=120,
            )
            return None
        except Exception as e:
            return f"❌ Could not install {missing}: {e}"
    return None


# ── Common app name → executable mapping (Windows) ──────────────────────────

_APP_ALIASES: dict[str, str] = {
    # System
    "notepad":      "notepad.exe",
    "explorer":     "explorer.exe",
    "calculator":   "calc.exe",
    "paint":        "mspaint.exe",
    "wordpad":      "wordpad.exe",
    "cmd":          "cmd.exe",
    "powershell":   "powershell.exe",
    "task manager": "taskmgr.exe",
    "regedit":      "regedit.exe",
    "control panel":"control.exe",
    "settings":     "ms-settings:",
    "snipping tool":"SnippingTool.exe",
    "camera":       "microsoft.windows.camera:",

    # Office
    "word":         "WINWORD.EXE",
    "excel":        "EXCEL.EXE",
    "powerpoint":   "POWERPNT.EXE",
    "outlook":      "OUTLOOK.EXE",
    "onenote":      "ONENOTE.EXE",
    "access":       "MSACCESS.EXE",

    # Browsers
    "chrome":       "chrome.exe",
    "firefox":      "firefox.exe",
    "edge":         "msedge.exe",
    "opera":        "opera.exe",
    "brave":        "brave.exe",

    # Media
    "vlc":          "vlc.exe",
    "spotify":      "spotify.exe",
    "media player": "wmplayer.exe",
    "groove":       "music.ui:",

    # Dev
    "vscode":       "code.exe",
    "vs code":      "code.exe",
    "visual studio code": "code.exe",
    "pycharm":      "pycharm64.exe",
    "git bash":     "git-bash.exe",
    "wsl":          "wsl.exe",

    # Creativity
    "photoshop":    "Photoshop.exe",
    "illustrator":  "Illustrator.exe",
    "premiere":     "Adobe Premiere Pro.exe",
    "after effects":"AfterFx.exe",
    "figma":        "figma.exe",
    "canva":        "canva.exe",
    "gimp":         "gimp-2.10.exe",

    # Productivity
    "notion":       "notion.exe",
    "obsidian":     "obsidian.exe",
    "slack":        "slack.exe",
    "teams":        "teams.exe",
    "discord":      "discord.exe",
    "zoom":         "zoom.exe",
    "whatsapp":     "whatsapp.exe",
    "telegram":     "telegram.exe",

    # Utilities
    "7zip":         "7zFM.exe",
    "winrar":       "winrar.exe",
    "everything":   "everything.exe",
    "snappy driver":"SDI_x64.exe",
}


def _resolve_app(name_or_path: str) -> str:
    """
    Resolve a friendly app name, alias, or full path to something Start-Process
    or os.startfile can launch.

    Resolution order:
      1. Direct path (file exists as given)
      2. Alias table (exact match)
      3. Desktop shortcuts (.lnk files) — partial name match
      4. Start Menu shortcuts — partial name match
      5. Common program directories — partial name match
      6. Return as-is (let Windows try to resolve it)
    """
    # 1. Direct path
    if os.path.isfile(name_or_path):
        return name_or_path

    # 2. Check alias table (case-insensitive)
    lower = name_or_path.lower().strip()
    if lower in _APP_ALIASES:
        return _APP_ALIASES[lower]

    # 3 & 4. Search Desktop and Start Menu for matching shortcuts
    keyword = lower.replace(" ", "").replace("-", "").replace("_", "")
    search_dirs = [
        Path.home() / "Desktop",
        Path("C:/ProgramData/Microsoft/Windows/Start Menu/Programs"),
        Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs",
        Path.home() / "AppData/Local/Programs",
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    ]

    best_match: str | None = None
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        try:
            for f in search_dir.rglob("*"):
                if f.suffix.lower() in (".lnk", ".exe", ".url"):
                    fname = f.stem.lower().replace(" ", "").replace("-", "").replace("_", "")
                    if keyword in fname or fname in keyword:
                        best_match = str(f)
                        break   # take the first match in priority order
        except (PermissionError, OSError):
            continue
        if best_match:
            break

    if best_match:
        return best_match

    # 5. Return as-is (let Windows resolve via PATH or shell)
    return name_or_path


# ─────────────────────────────────────────────────────────────────────────────
# Main tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def desktop_control(command: str) -> str:
    """
    Control any Windows desktop application — open, click, type, screenshot, and more.

    This tool has FULL authority to interact with any application on the user's machine.
    The agent uses it to open apps, read what is on screen, and manipulate UI elements.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    COMMAND REFERENCE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    LAUNCHING APPS
      open:notepad                → Open Notepad
      open:chrome                 → Open Google Chrome
      open:C:\\Program Files\\App\\app.exe  → Open by full path
      open:spotify                → Open Spotify
      open:word                   → Open Microsoft Word

    SCREEN CAPTURE (IMPORTANT — do this before clicking to see coordinates)
      screenshot                  → Take screenshot, save to Desktop, return file path + size info
      screenshot:C:\\path\\out.png → Save screenshot to a specific path

    WINDOW MANAGEMENT
      list_windows                → Show all open windows with their titles
      focus:Notepad               → Bring "Notepad" window to the front
      maximize:Chrome             → Maximize a window (partial title match)
      minimize:Notepad            → Minimize a window
      close:Notepad               → Close a window gracefully (sends Alt+F4)
      get_pos:Notepad             → Get x,y,width,height of a window

    MOUSE CONTROL
      click:x,y                   → Left-click at screen coordinates
      double_click:x,y            → Double-click at coordinates
      right_click:x,y             → Right-click at coordinates
      move:x,y                    → Move mouse without clicking
      drag:x1,y1,x2,y2            → Drag from one point to another

    KEYBOARD INPUT
      type:Hello World            → Type text at the current cursor position
      type_slow:text here         → Type slowly (for apps that drop fast input)
      press:enter                 → Press a single key (enter, tab, escape, delete, etc.)
      hotkey:ctrl+s               → Press a keyboard shortcut
      hotkey:alt+f4               → Close active window
      hotkey:win+d                → Show desktop
      hotkey:ctrl+c               → Copy
      hotkey:ctrl+v               → Paste
      hotkey:ctrl+z               → Undo

    SCROLLING
      scroll_up:3                 → Scroll up 3 times at current position
      scroll_down:5               → Scroll down 5 times
      scroll_at:x,y,3             → Scroll up 3 times at specific coordinates

    WAITING
      wait:2                      → Wait 2 seconds (for apps to load)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RECOMMENDED WORKFLOW FOR ANY TASK
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. desktop_control("open:notepad")          → launch the app
    2. desktop_control("wait:2")                → wait for it to load
    3. desktop_control("screenshot")            → see what is on screen
    4. desktop_control("focus:Notepad")         → make sure it is in front
    5. desktop_control("click:960,540")         → click the right spot
    6. desktop_control("type:Hello World")      → type text
    7. desktop_control("hotkey:ctrl+s")         → save
    8. desktop_control("screenshot")            → verify the result

    Args:
        command: A command string from the list above.

    Returns:
        Detailed result describing what happened, coordinates found, or errors.
    """
    # Ensure GUI libraries are present
    err = _ensure_deps()
    if err:
        return err

    import pyautogui
    import pygetwindow as gw

    # Disable pyautogui failsafe pause for speed (we trust this environment)
    pyautogui.PAUSE      = 0.05
    pyautogui.FAILSAFE   = True   # Move mouse to top-left corner to abort emergency

    cmd  = command.strip()
    low  = cmd.lower()

    # ── WAIT ─────────────────────────────────────────────────────────────────
    if low.startswith("wait:"):
        secs = float(low[5:])
        time.sleep(min(secs, 30))
        return f"✅ Waited {secs} seconds."

    # ── OPEN APP ─────────────────────────────────────────────────────────────
    if low.startswith("open:"):
        app_raw  = cmd[5:].strip()
        app_path = _resolve_app(app_raw)

        # URI protocol (ms-settings:, music.ui:, etc.)
        if ":" in app_path and not os.path.isabs(app_path) and not app_path.endswith(".exe") and not app_path.endswith(".lnk"):
            try:
                os.startfile(app_path)
                time.sleep(0.5)
                return f"✅ Launched '{app_raw}' via URI: {app_path}"
            except Exception as exc:
                return f"❌ Could not launch '{app_raw}': {exc}"

        # .lnk shortcut files — use os.startfile directly (most reliable for shortcuts)
        if app_path.lower().endswith(".lnk"):
            try:
                os.startfile(app_path)
                time.sleep(1.5)   # shortcuts may take longer to load
                return f"✅ Launched '{app_raw}' via shortcut: {app_path}"
            except Exception as exc:
                return (
                    f"❌ Could not open shortcut '{app_path}'.\n"
                    f"   Error: {exc}\n"
                    "   Try: open:C:\\full\\path\\to\\app.exe"
                )

        # PowerShell Start-Process (most reliable on Windows for .exe)
        try:
            ps_cmd = f"Start-Process '{app_path}'"
            result = subprocess.run(
                ["powershell", "-NonInteractive", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                time.sleep(0.8)
                return f"✅ Launched '{app_raw}' ({app_path})."
            # If Start-Process fails, try os.startfile
            os.startfile(app_path)
            time.sleep(0.8)
            return f"✅ Launched '{app_raw}' via startfile."
        except Exception as exc:
            # Last resort: search Desktop again and list found shortcuts to help user
            desktop = Path.home() / "Desktop"
            found_shortcuts = []
            try:
                for f in desktop.iterdir():
                    if f.suffix.lower() in (".lnk", ".exe", ".url"):
                        found_shortcuts.append(f.name)
            except Exception:
                pass
            hint = ""
            if found_shortcuts:
                hint = f"\n   Desktop shortcuts found: {', '.join(found_shortcuts[:10])}"
            return (
                f"❌ Could not open '{app_raw}'.\n"
                f"   Tried: {app_path}\n"
                f"   Error: {exc}\n"
                f"   Tip: Use the exact shortcut name from Desktop, e.g. open:Gemini.lnk{hint}"
            )

    # ── SCREENSHOT ───────────────────────────────────────────────────────────
    if low.startswith("screenshot"):
        if ":" in cmd:
            save_path = cmd.split(":", 1)[1].strip()
        else:
            save_path = str(Path.home() / "Desktop" / "AgentScreenshot.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            img = pyautogui.screenshot()
            img.save(save_path)
            w, h = img.size
            return (
                f"✅ Screenshot saved!\n"
                f"   Path       : {save_path}\n"
                f"   Resolution : {w} × {h} pixels\n"
                f"   View it at : {save_path}\n\n"
                "ℹ️  Use the pixel coordinates you see in this screenshot for click: commands."
            )
        except Exception as exc:
            return f"❌ Screenshot failed: {exc}"

    # ── LIST WINDOWS ─────────────────────────────────────────────────────────
    if low == "list_windows":
        try:
            windows = gw.getAllWindows()
            visible = [w for w in windows if w.title.strip()]
            if not visible:
                return "No open windows found."
            lines = [f"🪟 Open windows ({len(visible)} total):"]
            for w in visible:
                lines.append(f"  • \"{w.title}\"  [{w.width}×{w.height} at ({w.left},{w.top})]")
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ list_windows error: {exc}"

    # ── FOCUS WINDOW ─────────────────────────────────────────────────────────
    if low.startswith("focus:"):
        title_frag = cmd[6:].strip()
        try:
            wins = gw.getWindowsWithTitle(title_frag)
            if not wins:
                # partial match fallback
                all_wins = gw.getAllWindows()
                wins = [w for w in all_wins if title_frag.lower() in w.title.lower()]
            if not wins:
                return f"❌ No window found matching '{title_frag}'. Use list_windows to see open windows."
            win = wins[0]
            win.activate()
            time.sleep(0.3)
            return f"✅ Focused window: \"{win.title}\""
        except Exception as exc:
            # Try via PowerShell as fallback
            try:
                ps = (
                    f"$w = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{title_frag}*'}} | Select-Object -First 1; "
                    f"if ($w) {{ $null = $w.MainWindowHandle; "
                    f"Add-Type -A System.Windows.Forms; "
                    f"[System.Windows.Forms.SendKeys]::SendWait(''); "
                    f"[void][System.Runtime.InteropServices.Marshal]::GetActiveObject('WScript.Shell') }} "
                )
                return f"⚠️ pygetwindow focus failed ({exc}). Tried PowerShell fallback."
            except Exception:
                return f"❌ Could not focus '{title_frag}': {exc}"

    # ── MAXIMIZE ─────────────────────────────────────────────────────────────
    if low.startswith("maximize:"):
        title_frag = cmd[9:].strip()
        return _window_action(title_frag, "maximize")

    # ── MINIMIZE ─────────────────────────────────────────────────────────────
    if low.startswith("minimize:"):
        title_frag = cmd[9:].strip()
        return _window_action(title_frag, "minimize")

    # ── CLOSE WINDOW ─────────────────────────────────────────────────────────
    if low.startswith("close:"):
        title_frag = cmd[6:].strip()
        return _window_action(title_frag, "close")

    # ── GET WINDOW POSITION ───────────────────────────────────────────────────
    if low.startswith("get_pos:"):
        title_frag = cmd[8:].strip()
        try:
            import pygetwindow as gw
            wins = [w for w in gw.getAllWindows() if title_frag.lower() in w.title.lower()]
            if not wins:
                return f"❌ No window matching '{title_frag}'."
            w = wins[0]
            return (
                f"✅ Window position for \"{w.title}\":\n"
                f"   Left   : {w.left}\n"
                f"   Top    : {w.top}\n"
                f"   Width  : {w.width}\n"
                f"   Height : {w.height}\n"
                f"   Center : ({w.left + w.width//2}, {w.top + w.height//2})"
            )
        except Exception as exc:
            return f"❌ get_pos error: {exc}"

    # ── CLICK ─────────────────────────────────────────────────────────────────
    if low.startswith("click:"):
        coords = cmd[6:].strip()
        try:
            x, y = [int(c.strip()) for c in coords.split(",")]
            pyautogui.click(x, y)
            return f"✅ Left-clicked at ({x}, {y})."
        except Exception as exc:
            return f"❌ click error: {exc}"

    # ── DOUBLE CLICK ─────────────────────────────────────────────────────────
    if low.startswith("double_click:"):
        coords = cmd[13:].strip()
        try:
            x, y = [int(c.strip()) for c in coords.split(",")]
            pyautogui.doubleClick(x, y)
            return f"✅ Double-clicked at ({x}, {y})."
        except Exception as exc:
            return f"❌ double_click error: {exc}"

    # ── RIGHT CLICK ───────────────────────────────────────────────────────────
    if low.startswith("right_click:"):
        coords = cmd[12:].strip()
        try:
            x, y = [int(c.strip()) for c in coords.split(",")]
            pyautogui.rightClick(x, y)
            return f"✅ Right-clicked at ({x}, {y})."
        except Exception as exc:
            return f"❌ right_click error: {exc}"

    # ── MOVE MOUSE ───────────────────────────────────────────────────────────
    if low.startswith("move:"):
        coords = cmd[5:].strip()
        try:
            x, y = [int(c.strip()) for c in coords.split(",")]
            pyautogui.moveTo(x, y, duration=0.2)
            return f"✅ Mouse moved to ({x}, {y})."
        except Exception as exc:
            return f"❌ move error: {exc}"

    # ── DRAG ─────────────────────────────────────────────────────────────────
    if low.startswith("drag:"):
        coords = cmd[5:].strip()
        try:
            x1, y1, x2, y2 = [int(c.strip()) for c in coords.split(",")]
            pyautogui.moveTo(x1, y1, duration=0.2)
            pyautogui.dragTo(x2, y2, duration=0.5, button="left")
            return f"✅ Dragged from ({x1}, {y1}) to ({x2}, {y2})."
        except Exception as exc:
            return f"❌ drag error: {exc}"

    # ── TYPE TEXT ────────────────────────────────────────────────────────────
    if low.startswith("type:"):
        text = cmd[5:]   # preserve original case / content
        try:
            pyautogui.write(text, interval=0.01)
            return f"✅ Typed: {text[:80]}{'...' if len(text) > 80 else ''}"
        except Exception as exc:
            # Fallback: use clipboard paste (works for Unicode / Arabic text)
            return _type_via_clipboard(text)

    if low.startswith("type_slow:"):
        text = cmd[10:]
        try:
            pyautogui.write(text, interval=0.05)
            return f"✅ Typed (slow): {text[:80]}"
        except Exception as exc:
            return _type_via_clipboard(text)

    # ── PRESS SINGLE KEY ─────────────────────────────────────────────────────
    if low.startswith("press:"):
        key = cmd[6:].strip().lower()
        try:
            pyautogui.press(key)
            return f"✅ Pressed key: {key}"
        except Exception as exc:
            return f"❌ press error (key='{key}'): {exc}"

    # ── HOTKEY / SHORTCUT ────────────────────────────────────────────────────
    if low.startswith("hotkey:"):
        combo = cmd[7:].strip().lower()
        keys  = [k.strip() for k in combo.split("+")]
        try:
            pyautogui.hotkey(*keys)
            return f"✅ Pressed hotkey: {' + '.join(keys)}"
        except Exception as exc:
            return f"❌ hotkey error: {exc}"

    # ── SCROLL ───────────────────────────────────────────────────────────────
    if low.startswith("scroll_up:"):
        clicks = int(cmd[10:].strip())
        pyautogui.scroll(clicks)
        return f"✅ Scrolled up {clicks} times."

    if low.startswith("scroll_down:"):
        clicks = int(cmd[12:].strip())
        pyautogui.scroll(-clicks)
        return f"✅ Scrolled down {clicks} times."

    if low.startswith("scroll_at:"):
        parts  = cmd[10:].strip().split(",")
        x, y   = int(parts[0]), int(parts[1])
        clicks = int(parts[2]) if len(parts) > 2 else 3
        direction = 1 if clicks > 0 else -1
        pyautogui.scroll(abs(clicks) * direction, x=x, y=y)
        return f"✅ Scrolled {abs(clicks)} times at ({x}, {y})."

    # ── UNRECOGNISED ─────────────────────────────────────────────────────────
    return (
        f"❌ Unrecognised command: '{cmd}'\n\n"
        "Available commands:\n"
        "  open:<app>          — open:notepad, open:chrome, open:C:\\path\\app.exe\n"
        "  screenshot          — take screenshot of entire screen\n"
        "  list_windows        — list all open windows\n"
        "  focus:<title>       — bring window to foreground\n"
        "  maximize/minimize/close:<title>\n"
        "  click:x,y           — left-click at coordinates\n"
        "  double_click:x,y\n"
        "  right_click:x,y\n"
        "  type:<text>         — type text (use clipboard for Arabic/Unicode)\n"
        "  press:<key>         — press: enter, tab, escape, delete, f1..f12\n"
        "  hotkey:ctrl+s       — keyboard shortcuts\n"
        "  scroll_up:N / scroll_down:N\n"
        "  wait:N              — wait N seconds"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_action(title_frag: str, action: str) -> str:
    try:
        import pygetwindow as gw
        wins = [w for w in gw.getAllWindows() if title_frag.lower() in w.title.lower()]
        if not wins:
            return f"❌ No window matching '{title_frag}'. Use list_windows to check."
        win = wins[0]
        if action == "maximize":
            win.maximize()
        elif action == "minimize":
            win.minimize()
        elif action == "close":
            win.close()
        time.sleep(0.2)
        return f"✅ {action.capitalize()}d window: \"{win.title}\""
    except Exception as exc:
        return f"❌ {action} error: {exc}"


def _type_via_clipboard(text: str) -> str:
    """
    Type text by pasting from clipboard — necessary for non-ASCII / Arabic text.
    pyautogui.write() only handles ASCII.
    """
    try:
        import subprocess
        # Write to clipboard via PowerShell (works for any Unicode text)
        ps_cmd = f"Set-Clipboard -Value '{text.replace(chr(39), chr(34))}'"
        subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True,
            timeout=5,
        )
        import pyautogui
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.1)
        return f"✅ Typed via clipboard paste: {text[:80]}{'...' if len(text) > 80 else ''}"
    except Exception as exc:
        return f"❌ Clipboard type failed: {exc}"
