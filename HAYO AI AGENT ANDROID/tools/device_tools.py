"""
Android device tools — battery, sensors, device info, storage.
Uses dumpsys, getprop, and other Android commands.
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


@tool
def get_device_info() -> str:
    """عرض معلومات شاملة عن الجهاز (الموديل، النظام، المعالج، الذاكرة)."""
    info = []
    
    model = _shell_cmd("getprop ro.product.model")
    brand = _shell_cmd("getprop ro.product.brand")
    android_ver = _shell_cmd("getprop ro.build.version.release")
    sdk = _shell_cmd("getprop ro.build.version.sdk")
    build = _shell_cmd("getprop ro.build.display.id")
    cpu = _shell_cmd("getprop ro.product.board")
    
    info.append(f"📱 الجهاز: {brand} {model}")
    info.append(f"🤖 Android: {android_ver} (SDK {sdk})")
    info.append(f"🏗️ Build: {build}")
    info.append(f"⚙️ المعالج: {cpu}")
    
    # Memory info
    mem = _shell_cmd("cat /proc/meminfo | head -4")
    info.append(f"\n📊 الذاكرة:\n{mem}")
    
    # Storage
    storage = _shell_cmd("df -h /sdcard 2>/dev/null || df -h /data")
    info.append(f"\n💾 التخزين:\n{storage}")
    
    # Uptime
    uptime = _shell_cmd("uptime")
    info.append(f"\n⏱️ وقت التشغيل: {uptime}")
    
    return "\n".join(info)


@tool
def get_battery_info() -> str:
    """عرض معلومات البطارية (النسبة، الحرارة، حالة الشحن)."""
    result = _root_cmd("dumpsys battery")
    if not result or "ERROR" in result:
        result = _shell_cmd("cat /sys/class/power_supply/battery/capacity 2>/dev/null")
        return f"Battery level: {result}%" if result else "[ERROR] Cannot read battery info"
    return f"🔋 معلومات البطارية:\n{result}"


@tool
def get_storage_info() -> str:
    """عرض معلومات التخزين المفصلة (المساحة المتاحة والمستخدمة)."""
    result = _shell_cmd("df -h 2>/dev/null | grep -E '(/sdcard|/data|/storage|/system|emulated)'")
    if not result:
        result = _shell_cmd("df -h")
    return f"💾 التخزين:\n{result}"


@tool
def get_running_processes(top_n: int = 15) -> str:
    """عرض العمليات الأكثر استهلاكاً للذاكرة.
    
    Args:
        top_n: عدد العمليات المعروضة (افتراضي 15)
    """
    result = _root_cmd(f"ps -A --sort=-rss | head -{top_n + 1}")
    if not result or "ERROR" in result:
        result = _shell_cmd(f"ps | head -{top_n + 1}")
    return f"📊 العمليات ({top_n} الأعلى):\n{result}"


@tool
def get_sensor_data() -> str:
    """قراءة بيانات المستشعرات المتاحة (بوصلة، تسارع، إلخ)."""
    result = _root_cmd("dumpsys sensorservice | head -30")
    return result or "[ERROR] Cannot read sensor data"


@tool
def set_airplane_mode(enabled: bool) -> str:
    """تفعيل/تعطيل وضع الطيران (يحتاج Root).
    
    Args:
        enabled: True لتفعيل، False لتعطيل
    """
    val = "1" if enabled else "0"
    _root_cmd(f"settings put global airplane_mode_on {val}")
    _root_cmd(f"am broadcast -a android.intent.action.AIRPLANE_MODE --ez state {str(enabled).lower()}")
    return f"Airplane mode: {'ON ✈️' if enabled else 'OFF'}"


@tool
def set_wifi(enabled: bool) -> str:
    """تفعيل/تعطيل Wi-Fi (يحتاج Root).
    
    Args:
        enabled: True لتفعيل، False لتعطيل
    """
    action = "enable" if enabled else "disable"
    result = _root_cmd(f"svc wifi {action}")
    return result if result else f"Wi-Fi: {'ON 📶' if enabled else 'OFF'}"


@tool
def set_bluetooth(enabled: bool) -> str:
    """تفعيل/تعطيل Bluetooth (يحتاج Root).
    
    Args:
        enabled: True لتفعيل، False لتعطيل
    """
    action = "enable" if enabled else "disable"
    result = _root_cmd(f"svc bluetooth {action}")
    return result if result else f"Bluetooth: {'ON 🔵' if enabled else 'OFF'}"


@tool
def set_mobile_data(enabled: bool) -> str:
    """تفعيل/تعطيل بيانات الهاتف (يحتاج Root).
    
    Args:
        enabled: True لتفعيل، False لتعطيل
    """
    action = "enable" if enabled else "disable"
    result = _root_cmd(f"svc data {action}")
    return result if result else f"Mobile data: {'ON 📱' if enabled else 'OFF'}"
