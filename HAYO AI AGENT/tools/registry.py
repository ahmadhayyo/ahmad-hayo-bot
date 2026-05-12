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
from tools.clipboard_tools import (
    clipboard_get,
    clipboard_set,
    clipboard_append,
)
from tools.process_tools import (
    get_system_info,
    kill_process,
    list_processes,
    manage_service,
    scheduled_task,
)
from tools.network_tools import (
    check_port,
    dns_lookup,
    get_network_info,
    get_public_ip,
    ping_host,
    wifi_management,
)
from tools.audio_tools import (
    play_sound,
    show_notification,
    text_to_speech,
    volume_control,
)
from tools.office_tools import (
    excel_create,
    excel_read,
    excel_edit,
    excel_add_rows,
    excel_add_column,
    word_create,
    word_read,
    word_edit,
    pdf_read,
    pdf_create,
    pdf_merge,
    convert_excel_to_pdf,
    convert_word_to_pdf,
)
from tools.advanced_download import (
    download_with_progress,
    check_url_availability,
    get_file_hash,
)
from tools.chrome_management import (
    chrome_search_and_open,
    chrome_download_file_from_page,
    chrome_extract_download_links,
    chrome_handle_redirects,
    chrome_search_media_file,
    chrome_get_direct_download_url,
)
from tools.file_conversion import (
    convert_file,
    get_supported_formats,
    check_conversion_support,
)
from tools.replit_tools import (
    replit_open_project,
    replit_list_files,
    replit_read_file,
    replit_update_file,
    replit_git_commit,
    replit_git_sync,
    replit_run_project,
    replit_create_project_structure,
)

ALL_TOOLS: list[BaseTool] = [
    # ═══════════════════════════════════════════════════════════
    # SHELL & SYSTEM
    # ═══════════════════════════════════════════════════════════
    run_powershell,
    run_cmd,
    get_env,
    get_system_info,
    list_processes,
    kill_process,
    manage_service,
    scheduled_task,

    # ═══════════════════════════════════════════════════════════
    # FILE SYSTEM
    # ═══════════════════════════════════════════════════════════
    read_file,
    write_file,
    append_file,
    list_dir,
    search_files,
    move_file,
    copy_file,
    download_file,
    make_dir,

    # ═══════════════════════════════════════════════════════════
    # CLIPBOARD
    # ═══════════════════════════════════════════════════════════
    clipboard_get,
    clipboard_set,
    clipboard_append,

    # ═══════════════════════════════════════════════════════════
    # APPLICATIONS
    # ═══════════════════════════════════════════════════════════
    open_app,
    close_app,
    list_running_apps,
    focus_window,

    # ═══════════════════════════════════════════════════════════
    # BROWSER (Playwright persistent session)
    # ═══════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════
    # DESKTOP GUI (pyautogui — pixel-level control)
    # ═══════════════════════════════════════════════════════════
    screen_screenshot,
    screen_size,
    mouse_click,
    mouse_move,
    mouse_scroll,
    keyboard_type,
    keyboard_hotkey,
    list_windows,
    wait,

    # ═══════════════════════════════════════════════════════════
    # NETWORK
    # ═══════════════════════════════════════════════════════════
    get_network_info,
    get_public_ip,
    ping_host,
    check_port,
    wifi_management,
    dns_lookup,

    # ═══════════════════════════════════════════════════════════
    # AUDIO & NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════
    volume_control,
    text_to_speech,
    show_notification,
    play_sound,

    # ═══════════════════════════════════════════════════════════
    # OFFICE (Excel, Word, PDF)
    # ═══════════════════════════════════════════════════════════
    excel_create,
    excel_read,
    excel_edit,
    excel_add_rows,
    excel_add_column,
    word_create,
    word_read,
    word_edit,
    pdf_read,
    pdf_create,
    pdf_merge,
    convert_excel_to_pdf,
    convert_word_to_pdf,

    # ═══════════════════════════════════════════════════════════
    # ADVANCED DOWNLOAD (with progress, retry, integrity)
    # ═══════════════════════════════════════════════════════════
    download_with_progress,
    check_url_availability,
    get_file_hash,

    # ═══════════════════════════════════════════════════════════
    # CHROME MANAGEMENT (search, extract links, handle redirects)
    # ═══════════════════════════════════════════════════════════
    chrome_search_and_open,
    chrome_download_file_from_page,
    chrome_extract_download_links,
    chrome_handle_redirects,
    chrome_search_media_file,
    chrome_get_direct_download_url,

    # ═══════════════════════════════════════════════════════════
    # FILE CONVERSION (audio, video, docs, images)
    # ═══════════════════════════════════════════════════════════
    convert_file,
    get_supported_formats,
    check_conversion_support,

    # ═══════════════════════════════════════════════════════════
    # REPLIT INTEGRATION (project management, git sync, execution)
    # ═══════════════════════════════════════════════════════════
    replit_open_project,
    replit_list_files,
    replit_read_file,
    replit_update_file,
    replit_git_commit,
    replit_git_sync,
    replit_run_project,
    replit_create_project_structure,
]

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
