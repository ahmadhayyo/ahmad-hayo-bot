"""
Single source of truth for all tools the agent can invoke.

Import ALL_TOOLS from here instead of importing from individual modules —
this guarantees a consistent ordering and lets you easily disable a category
in one place.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from tools.app_tools import (
    close_app,
    focus_window,
    list_running_apps,
    open_app,
)
from tools.browser_tools import (
    browser_click,
    browser_close,
    browser_download_via_click,
    browser_eval_js,
    browser_fill,
    browser_get_text,
    browser_open,
    browser_press,
    browser_screenshot,
    browser_wait_for,
)
from tools.desktop_tools import (
    keyboard_hotkey,
    keyboard_type,
    list_windows,
    mouse_click,
    mouse_move,
    mouse_scroll,
    screen_screenshot,
    screen_size,
    wait,
)
from tools.file_tools import (
    append_file,
    copy_file,
    download_file,
    list_dir,
    make_dir,
    move_file,
    read_file,
    search_files,
    write_file,
)
from tools.system_tools import get_env, run_cmd, run_powershell

ALL_TOOLS: list[BaseTool] = [
    # Shell
    run_powershell,
    run_cmd,
    get_env,
    # File system
    read_file,
    write_file,
    append_file,
    list_dir,
    search_files,
    move_file,
    copy_file,
    download_file,
    make_dir,
    # Apps
    open_app,
    close_app,
    list_running_apps,
    focus_window,
    # Browser
    browser_open,
    browser_get_text,
    browser_click,
    browser_fill,
    browser_press,
    browser_screenshot,
    browser_download_via_click,
    browser_eval_js,
    browser_wait_for,
    browser_close,
    # Desktop
    screen_screenshot,
    screen_size,
    mouse_click,
    mouse_move,
    mouse_scroll,
    keyboard_type,
    keyboard_hotkey,
    list_windows,
    wait,
]

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
