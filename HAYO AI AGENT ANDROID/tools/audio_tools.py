"""
Android audio tools — volume, TTS, notifications, media.
Uses termux-api and Android media commands.
"""

from __future__ import annotations

import subprocess
from langchain_core.tools import tool


def _shell_cmd(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"[ERROR] {e}"


def _root_cmd(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            f"su -c '{cmd}'", shell=True,
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def volume_control(stream: str = "media", level: int = -1) -> str:
    """التحكم بمستوى الصوت.
    
    Args:
        stream: نوع الصوت (media, ring, alarm, notification)
        level: المستوى (0-15). إذا كان -1 يعرض القيمة الحالية.
    """
    stream_map = {
        "media": "3",
        "ring": "2",
        "alarm": "4",
        "notification": "5",
    }
    stream_id = stream_map.get(stream, "3")
    
    if level < 0:
        result = _root_cmd(f"settings get system volume_music_speaker 2>/dev/null")
        if result and "ERROR" not in result:
            return f"Volume ({stream}): {result}/15"
        return _shell_cmd("termux-volume 2>/dev/null") or "[ERROR] Cannot read volume"
    
    level = max(0, min(15, level))
    # Try media command
    _root_cmd(f"media volume --stream {stream_id} --set {level}")
    return f"Volume ({stream}) set to {level}/15"


@tool
def text_to_speech(text: str, language: str = "ar") -> str:
    """قراءة نص بصوت عالٍ عبر TTS.
    
    Args:
        text: النص المراد قراءته
        language: اللغة (ar=عربي, en=إنجليزي)
    """
    # Try termux-tts-speak
    result = _shell_cmd(f'echo "{text}" | termux-tts-speak -l {language} 2>/dev/null')
    if "ERROR" not in (result or ""):
        return f"Speaking: {text[:50]}..."
    # Fallback: am broadcast
    _root_cmd(f'am broadcast -a android.intent.action.TTS_SPEAK --es text "{text}" 2>/dev/null')
    return f"TTS: {text[:50]}..."


@tool
def show_notification(title: str, content: str) -> str:
    """عرض إشعار على الجهاز.
    
    Args:
        title: عنوان الإشعار
        content: محتوى الإشعار
    """
    # Try termux-notification
    result = _shell_cmd(f'termux-notification --title "{title}" --content "{content}" 2>/dev/null')
    if "ERROR" not in (result or "ERROR"):
        return f"Notification sent: {title}"
    # Fallback: toast via root
    _root_cmd(f'am broadcast -a com.toast --es text "{title}: {content}" 2>/dev/null')
    return f"Notification: {title}"


@tool
def play_sound(type: str = "notification") -> str:
    """تشغيل صوت تنبيه.
    
    Args:
        type: نوع الصوت (notification, alarm, ringtone)
    """
    sounds = {
        "notification": "/system/media/audio/notifications/OnTheHunt.ogg",
        "alarm": "/system/media/audio/alarms/Alarm_Classic.ogg",
        "ringtone": "/system/media/audio/ringtones/Ring_Synth_04.ogg",
    }
    sound_file = sounds.get(type, sounds["notification"])
    # Check if file exists, try alternatives
    result = _root_cmd(f"ls {sound_file} 2>/dev/null")
    if not result or "ERROR" in result:
        result = _root_cmd("ls /system/media/audio/notifications/ | head -1")
        if result:
            sound_file = f"/system/media/audio/notifications/{result}"
    _root_cmd(f"am start -a android.intent.action.VIEW -d file://{sound_file} -t audio/* 2>/dev/null")
    return f"Playing sound: {type}"


@tool
def vibrate(duration_ms: int = 500) -> str:
    """تشغيل الاهتزاز.
    
    Args:
        duration_ms: مدة الاهتزاز بالميلي ثانية (افتراضي 500)
    """
    result = _shell_cmd(f"termux-vibrate -d {duration_ms} 2>/dev/null")
    if "ERROR" not in (result or "ERROR"):
        return f"Vibrated for {duration_ms}ms"
    _root_cmd(f'echo {duration_ms} > /sys/class/timed_output/vibrator/enable 2>/dev/null')
    return f"Vibrated for {duration_ms}ms"


@tool
def torch_control(enabled: bool) -> str:
    """تشغيل/إيقاف الكشاف (الفلاش).
    
    Args:
        enabled: True لتشغيل، False لإيقاف
    """
    result = _shell_cmd(f'termux-torch {"on" if enabled else "off"} 2>/dev/null')
    if "ERROR" not in (result or "ERROR"):
        return f"Torch: {'ON 🔦' if enabled else 'OFF'}"
    return f"[ERROR] Cannot control torch. Install termux-api: pkg install termux-api"
