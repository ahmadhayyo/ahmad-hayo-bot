"""
Android screen tools — ROOT required.

Uses Android's built-in commands:
- screencap: لقطات شاشة
- input tap/swipe/text: التحكم باللمس والكتابة
- wm size/density: معلومات الشاشة
"""

from __future__ import annotations

import os
import subprocess
import base64
from pathlib import Path
from langchain_core.tools import tool


def _root_cmd(cmd: str, timeout: int = 30) -> str:
    """Execute a command as root."""
    try:
        r = subprocess.run(
            f"su -c '{cmd}'", shell=True,
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def screen_screenshot(save_path: str = "/sdcard/screenshot.png") -> str:
    """أخذ لقطة شاشة للموبايل (يحتاج Root).
    
    Args:
        save_path: مسار حفظ اللقطة (افتراضي: /sdcard/screenshot.png)
    """
    try:
        _root_cmd(f"screencap -p {save_path}")
        if os.path.exists(save_path):
            size = os.path.getsize(save_path)
            return f"Screenshot saved: {save_path} ({size} bytes)"
        return "[ERROR] Screenshot failed — file not created"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def screen_tap(x: int, y: int) -> str:
    """النقر على نقطة في الشاشة (يحتاج Root).
    
    Args:
        x: الإحداثي الأفقي (بالبكسل)
        y: الإحداثي العمودي (بالبكسل)
    """
    result = _root_cmd(f"input tap {x} {y}")
    return result if result else f"Tapped at ({x}, {y})"


@tool
def screen_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> str:
    """السحب على الشاشة من نقطة إلى أخرى (يحتاج Root).
    
    Args:
        x1: نقطة البداية X
        y1: نقطة البداية Y
        x2: نقطة النهاية X
        y2: نقطة النهاية Y
        duration_ms: مدة السحب بالميلي ثانية (افتراضي 300)
    """
    result = _root_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")
    return result if result else f"Swiped ({x1},{y1}) → ({x2},{y2}) in {duration_ms}ms"


@tool
def screen_type_text(text: str) -> str:
    """كتابة نص في حقل الإدخال الحالي (يحتاج Root).
    يجب أن يكون حقل إدخال مفعّلاً.
    
    Args:
        text: النص المراد كتابته
    """
    escaped = text.replace(" ", "%s").replace("'", "\\'")
    result = _root_cmd(f"input text '{escaped}'")
    return result if result else f"Typed: {text}"


@tool
def screen_key_event(keycode: str) -> str:
    """إرسال حدث مفتاح (يحتاج Root).
    أمثلة: KEYCODE_HOME, KEYCODE_BACK, KEYCODE_ENTER, KEYCODE_POWER,
    KEYCODE_VOLUME_UP, KEYCODE_VOLUME_DOWN, KEYCODE_RECENT_APPS
    
    Args:
        keycode: رمز المفتاح (مثل KEYCODE_HOME أو 3)
    """
    result = _root_cmd(f"input keyevent {keycode}")
    return result if result else f"Sent key event: {keycode}"


@tool
def screen_long_press(x: int, y: int, duration_ms: int = 1000) -> str:
    """ضغط مطوّل على نقطة في الشاشة (يحتاج Root).
    
    Args:
        x: الإحداثي الأفقي
        y: الإحداثي العمودي
        duration_ms: مدة الضغط بالميلي ثانية (افتراضي 1000)
    """
    result = _root_cmd(f"input swipe {x} {y} {x} {y} {duration_ms}")
    return result if result else f"Long pressed at ({x}, {y}) for {duration_ms}ms"


@tool
def screen_size() -> str:
    """عرض أبعاد وكثافة الشاشة."""
    size = _root_cmd("wm size")
    density = _root_cmd("wm density")
    return f"Size: {size}\nDensity: {density}"


@tool
def screen_brightness(level: int = -1) -> str:
    """تعديل سطوع الشاشة (0-255) أو قراءة القيمة الحالية.
    
    Args:
        level: مستوى السطوع (0-255). إذا كان -1 يعرض القيمة الحالية.
    """
    if level < 0:
        result = _root_cmd("settings get system screen_brightness")
        return f"Current brightness: {result}/255"
    level = max(0, min(255, level))
    _root_cmd(f"settings put system screen_brightness {level}")
    return f"Brightness set to {level}/255"


@tool
def screen_rotate(orientation: str = "auto") -> str:
    """تغيير اتجاه الشاشة.
    
    Args:
        orientation: auto, portrait, landscape, reverse_portrait, reverse_landscape
    """
    orientations = {
        "auto": ("0", "0"),
        "portrait": ("1", "0"),
        "landscape": ("1", "1"),
        "reverse_portrait": ("1", "2"),
        "reverse_landscape": ("1", "3"),
    }
    if orientation not in orientations:
        return f"[ERROR] Unknown orientation. Use: {', '.join(orientations.keys())}"
    locked, val = orientations[orientation]
    _root_cmd(f"settings put system accelerometer_rotation {0 if locked == '1' else 1}")
    if locked == "1":
        _root_cmd(f"settings put system user_rotation {val}")
    return f"Screen orientation set to: {orientation}"
