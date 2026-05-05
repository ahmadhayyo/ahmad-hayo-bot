"""
Android app management tools — ROOT required for most operations.

Uses Android's activity manager (am), package manager (pm), and dumpsys.
"""

from __future__ import annotations

import subprocess
from langchain_core.tools import tool


def _root_cmd(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            f"su -c '{cmd}'", shell=True,
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out"
    except Exception as e:
        return f"[ERROR] {e}"


def _shell_cmd(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"[ERROR] {e}"


# ── Common app package names ────────────────────────────────────────────────
_KNOWN_APPS = {
    "chrome": "com.android.chrome/com.google.android.apps.chrome.Main",
    "youtube": "com.google.android.youtube/.HomeActivity",
    "camera": "com.sec.android.app.camera/.Camera",
    "settings": "com.android.settings/.Settings",
    "gallery": "com.sec.android.gallery3d/.app.activity.GalleryActivity",
    "phone": "com.samsung.android.dialer/.DialtactsActivity",
    "messages": "com.samsung.android.messaging/.ui.view.main.MainActivityWrapper",
    "calculator": "com.sec.android.app.popupcalculator/.Calculator",
    "calendar": "com.samsung.android.calendar/.list.CalendarListActivity",
    "clock": "com.sec.android.app.clockpackage/.ClockPackage",
    "contacts": "com.samsung.android.contacts/.contactslist.ContactsListActivity",
    "files": "com.sec.android.app.myfiles/.common.MainActivity",
    "whatsapp": "com.whatsapp/.Main",
    "telegram": "org.telegram.messenger/.DefaultIcon",
    "instagram": "com.instagram.android/.activity.MainTabActivity",
    "facebook": "com.facebook.katana/.LoginActivity",
    "twitter": "com.twitter.android/.StartActivity",
    "gmail": "com.google.android.gm/.ConversationListActivityGmail",
    "maps": "com.google.android.apps.maps/.MapsActivity",
    "playstore": "com.android.vending/.AssetBrowserActivity",
}


@tool
def open_app(name: str) -> str:
    """فتح تطبيق على الموبايل (يحتاج Root).
    يمكن استخدام اسم التطبيق بالإنجليزية أو اسم الحزمة الكامل.
    
    أمثلة: chrome, youtube, whatsapp, settings, camera, files,
    telegram, instagram, gmail, calculator, calendar
    
    Args:
        name: اسم التطبيق أو اسم الحزمة
    """
    # Check known apps
    lower = name.lower().strip()
    if lower in _KNOWN_APPS:
        activity = _KNOWN_APPS[lower]
        result = _root_cmd(f"am start -n {activity}")
        if "Error" not in result:
            return f"Opened: {lower} ({activity})"
        # Try monkey launch as fallback
        pkg = activity.split("/")[0]
        result = _root_cmd(f"monkey -p {pkg} -c android.intent.category.LAUNCHER 1")
        if "Events injected" in result:
            return f"Opened: {lower} ({pkg})"
        return f"[ERROR] Could not open {lower}: {result}"

    # Try as package name
    if "." in lower:
        result = _root_cmd(f"monkey -p {lower} -c android.intent.category.LAUNCHER 1")
        if "Events injected" in result:
            return f"Opened: {lower}"
        return f"[ERROR] Could not open package: {lower}"

    # Search for matching package
    packages = _root_cmd(f"pm list packages | grep -i {lower}")
    if packages:
        pkg = packages.split("\n")[0].replace("package:", "").strip()
        result = _root_cmd(f"monkey -p {pkg} -c android.intent.category.LAUNCHER 1")
        if "Events injected" in result:
            return f"Opened: {pkg}"
    return f"[ERROR] App not found: {name}. Use package name (e.g., com.whatsapp)"


@tool
def close_app(package_name: str) -> str:
    """إغلاق تطبيق بالقوة (يحتاج Root).
    
    Args:
        package_name: اسم حزمة التطبيق (مثل com.whatsapp) أو اسم مختصر
    """
    lower = package_name.lower().strip()
    if lower in _KNOWN_APPS:
        pkg = _KNOWN_APPS[lower].split("/")[0]
    else:
        pkg = package_name
    result = _root_cmd(f"am force-stop {pkg}")
    return result if result else f"Force stopped: {pkg}"


@tool
def list_installed_apps(filter_name: str = "") -> str:
    """عرض التطبيقات المثبتة على الموبايل.
    
    Args:
        filter_name: فلترة حسب الاسم (اختياري)
    """
    if filter_name:
        result = _root_cmd(f"pm list packages | grep -i {filter_name}")
    else:
        result = _root_cmd("pm list packages -3")  # Third-party only
    if not result:
        return "No matching apps found."
    packages = result.replace("package:", "").strip()
    lines = packages.split("\n")
    return f"Installed apps ({len(lines)}):\n" + "\n".join(f"  • {p.strip()}" for p in lines[:50])


@tool
def list_running_apps() -> str:
    """عرض التطبيقات قيد التشغيل حالياً."""
    result = _root_cmd("dumpsys activity recents | grep 'Recent #' | head -15")
    if not result:
        result = _root_cmd("ps -A | grep -v root | head -20")
    return result or "Could not retrieve running apps."


@tool
def install_apk(apk_path: str) -> str:
    """تثبيت تطبيق من ملف APK (يحتاج Root).
    
    Args:
        apk_path: مسار ملف APK
    """
    from core.safety import needs_human_approval
    flagged, _ = needs_human_approval(f"pm install {apk_path}")
    if flagged:
        return f"__HITL_REQUIRED__\nCommand: pm install {apk_path}"
    result = _root_cmd(f"pm install -r {apk_path}", timeout=120)
    return result or f"Installed: {apk_path}"


@tool
def get_current_app() -> str:
    """عرض التطبيق المفتوح حالياً في المقدمة."""
    result = _root_cmd("dumpsys window | grep mCurrentFocus")
    if result:
        return f"Current foreground: {result}"
    result = _root_cmd("dumpsys activity activities | grep mResumedActivity")
    return result or "Could not determine current app."
