"""
Unified tool registry for HAYO AI Agent (Android Edition).
Single source of truth for all tools the agent can invoke.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from tools.system_tools import run_shell, run_root, get_env
from tools.file_tools import (
    read_file, write_file, append_file, list_dir,
    search_files, move_file, copy_file, make_dir, download_file,
)
from tools.screen_tools import (
    screen_screenshot, screen_tap, screen_swipe, screen_type_text,
    screen_key_event, screen_long_press, screen_size, screen_brightness,
    screen_rotate,
)
from tools.app_tools import (
    open_app, close_app, list_installed_apps, list_running_apps,
    install_apk, get_current_app,
)
from tools.device_tools import (
    get_device_info, get_battery_info, get_storage_info,
    get_running_processes, get_sensor_data,
    set_airplane_mode, set_wifi, set_bluetooth, set_mobile_data,
)
from tools.network_tools import (
    ping_host, get_network_info, get_public_ip,
    dns_lookup, check_port, wifi_scan, traceroute,
)
from tools.clipboard_tools import clipboard_get, clipboard_set, clipboard_append
from tools.audio_tools import (
    volume_control, text_to_speech, show_notification,
    play_sound, vibrate, torch_control,
)
from tools.web_tools import web_search, fetch_url, download_url
from tools.office_tools import (
    excel_create, excel_read, excel_edit, excel_add_rows, excel_add_column,
    word_create, word_read, word_edit,
    pdf_read, pdf_create, pdf_merge,
    convert_excel_to_pdf, convert_word_to_pdf,
)


ALL_TOOLS: list[BaseTool] = [
    # ═══════════════════════════════════════════════════════════
    # SHELL & SYSTEM
    # ═══════════════════════════════════════════════════════════
    run_shell,
    run_root,
    get_env,

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
    make_dir,
    download_file,

    # ═══════════════════════════════════════════════════════════
    # SCREEN CONTROL (ROOT)
    # ═══════════════════════════════════════════════════════════
    screen_screenshot,
    screen_tap,
    screen_swipe,
    screen_type_text,
    screen_key_event,
    screen_long_press,
    screen_size,
    screen_brightness,
    screen_rotate,

    # ═══════════════════════════════════════════════════════════
    # APPLICATIONS (ROOT)
    # ═══════════════════════════════════════════════════════════
    open_app,
    close_app,
    list_installed_apps,
    list_running_apps,
    install_apk,
    get_current_app,

    # ═══════════════════════════════════════════════════════════
    # DEVICE INFO & CONTROL
    # ═══════════════════════════════════════════════════════════
    get_device_info,
    get_battery_info,
    get_storage_info,
    get_running_processes,
    get_sensor_data,
    set_airplane_mode,
    set_wifi,
    set_bluetooth,
    set_mobile_data,

    # ═══════════════════════════════════════════════════════════
    # NETWORK
    # ═══════════════════════════════════════════════════════════
    ping_host,
    get_network_info,
    get_public_ip,
    dns_lookup,
    check_port,
    wifi_scan,
    traceroute,

    # ═══════════════════════════════════════════════════════════
    # CLIPBOARD
    # ═══════════════════════════════════════════════════════════
    clipboard_get,
    clipboard_set,
    clipboard_append,

    # ═══════════════════════════════════════════════════════════
    # AUDIO & HARDWARE
    # ═══════════════════════════════════════════════════════════
    volume_control,
    text_to_speech,
    show_notification,
    play_sound,
    vibrate,
    torch_control,

    # ═══════════════════════════════════════════════════════════
    # WEB
    # ═══════════════════════════════════════════════════════════
    web_search,
    fetch_url,
    download_url,

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
]

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
