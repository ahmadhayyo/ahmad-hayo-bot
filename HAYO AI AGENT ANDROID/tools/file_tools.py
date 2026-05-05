"""
Android file tools — read, write, copy, move, search, download.
Uses Linux paths. Downloads via curl/wget.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool


@tool
def read_file(path: str, max_chars: int = 20_000) -> str:
    """قراءة محتوى ملف نصي.
    
    Args:
        path: مسار الملف الكامل
        max_chars: الحد الأقصى للحروف (افتراضي 20000)
    """
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"[ERROR] File not found: {p}"
        if p.stat().st_size > 10_000_000:
            return f"[ERROR] File too large ({p.stat().st_size} bytes). Max 10 MB."
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n... (truncated, {len(text)} total chars)"
        return text
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def write_file(path: str, content: str) -> str:
    """كتابة محتوى إلى ملف (يستبدل المحتوى الحالي).
    
    Args:
        path: مسار الملف
        content: المحتوى المراد كتابته
    """
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {p}"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def append_file(path: str, content: str) -> str:
    """إلحاق محتوى بنهاية ملف موجود.
    
    Args:
        path: مسار الملف
        content: المحتوى المراد إلحاقه
    """
    try:
        p = Path(path).expanduser()
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} chars to {p}"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def list_dir(path: str = ".") -> str:
    """عرض محتويات مجلد.
    
    Args:
        path: مسار المجلد (افتراضي: المجلد الحالي)
    """
    try:
        p = Path(path).expanduser()
        if not p.is_dir():
            return f"[ERROR] Not a directory: {p}"
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        lines = [f"Listing of {p.resolve()}:"]
        for e in entries[:200]:
            kind = "DIR " if e.is_dir() else "FILE"
            size = e.stat().st_size if e.is_file() else 0
            lines.append(f"{kind} {size:>12}  {e.name}")
        if len(entries) > 200:
            lines.append(f"... and {len(entries) - 200} more entries")
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def search_files(root: str, pattern: str) -> str:
    """بحث عن ملفات بنمط معين.
    
    Args:
        root: المجلد الجذري للبحث
        pattern: نمط البحث (مثل *.txt أو *.jpg)
    """
    try:
        p = Path(root).expanduser()
        matches = list(p.rglob(pattern))[:100]
        if not matches:
            return f"No files matching '{pattern}' under {p}"
        return f"{len(matches)} match(es):\n" + "\n".join(str(m) for m in matches)
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def move_file(src: str, dst: str) -> str:
    """نقل أو إعادة تسمية ملف/مجلد.
    
    Args:
        src: المسار المصدر
        dst: المسار الهدف
    """
    try:
        shutil.move(str(Path(src).expanduser()), str(Path(dst).expanduser()))
        return f"Moved {src} → {dst}"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def copy_file(src: str, dst: str) -> str:
    """نسخ ملف أو مجلد.
    
    Args:
        src: المسار المصدر
        dst: المسار الهدف
    """
    try:
        s, d = Path(src).expanduser(), Path(dst).expanduser()
        if s.is_dir():
            shutil.copytree(str(s), str(d))
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(s), str(d))
        return f"Copied {src} → {dst}"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def make_dir(path: str) -> str:
    """إنشاء مجلد (مع المجلدات الأب إن لم تكن موجودة).
    
    Args:
        path: مسار المجلد
    """
    try:
        p = Path(path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {p}"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def download_file(url: str, save_path: str = "") -> str:
    """تحميل ملف من الإنترنت. يدعم ytsearch: لتحميل من YouTube.
    
    Args:
        url: رابط التحميل أو ytsearch:query للبحث في YouTube
        save_path: مسار الحفظ (افتراضي: /sdcard/Download/)
    """
    try:
        if not save_path:
            save_path = "/sdcard/Download/"
        dest = Path(save_path).expanduser()

        # YouTube / media download via yt-dlp
        if url.startswith("ytsearch:") or "youtube.com" in url or "youtu.be" in url:
            if dest.is_dir():
                out_template = str(dest / "%(title)s.%(ext)s")
            else:
                out_template = str(dest)
            cmd = [
                "yt-dlp", "--no-playlist",
                "-x", "--audio-format", "mp3",
                "-o", out_template, url,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                return f"Downloaded (yt-dlp) → {dest}\n{r.stdout[-500:]}"
            return f"[ERROR] yt-dlp failed:\n{r.stderr[-500:]}"

        # Regular HTTP download
        if dest.is_dir():
            dest = dest / url.split("/")[-1].split("?")[0]
        dest.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["curl", "-L", "-o", str(dest), url]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and dest.exists():
            return f"Downloaded → {dest} ({dest.stat().st_size} bytes)"
        return f"[ERROR] Download failed:\n{r.stderr[-500:]}"
    except Exception as e:
        return f"[ERROR] {e}"
