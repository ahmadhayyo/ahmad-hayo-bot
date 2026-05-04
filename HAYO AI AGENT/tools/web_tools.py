"""
tools/web_tools.py — Web Search and File Download Tools

Provides two powerful tools:
  1. web_search   — Search the web using DuckDuckGo (no API key needed).
  2. download_file — Download any file from a URL using multiple methods:

     Method 1: yt-dlp  — for YouTube, SoundCloud, Spotify previews, and 1000+
                          other sites. Converts to MP3 automatically. BEST for music.
     Method 2: requests — for direct file URLs (ends in .mp3, .pdf, .zip, etc.)
     Method 3: BITS     — PowerShell BitsTransfer (Windows-native, bypasses some blocks)
     Method 4: urllib   — final fallback

Strategy for music downloads:
  1. web_search for the song on YouTube (e.g. "tamer hosny song name youtube")
  2. download_file with the YouTube URL → yt-dlp handles it automatically
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_core.tools import tool

# ── Shared browser-like headers ───────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# Download timeout in seconds (large music files need time)
_DOWNLOAD_TIMEOUT = 120


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — web_search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str, num_results: int = 6) -> str:
    """
    Search the web using DuckDuckGo and return titles, URLs, and snippets.

    Use this to find direct download links, YouTube video URLs, research topics,
    or gather any information before taking an action.

    For music downloads: search like "tamer hosny [song name] youtube" to get
    a YouTube URL, then pass that URL to download_file.

    Args:
        query:       The search query (any language, including Arabic).
        num_results: How many results to return (default 6, max 10).

    Returns:
        Numbered list of results with title, URL, and description snippet.
    """
    import time

    last_error = ""
    # Try up to 3 times with backoff (DuckDuckGo rate-limits aggressively)
    for attempt in range(3):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=min(num_results, 10)))

            if results:
                lines = [f"🔍 Search results for: {query}\n"]
                for i, r in enumerate(results, 1):
                    title   = r.get("title", "No title")
                    url     = r.get("href",  "")
                    snippet = r.get("body",  "")[:250]
                    lines.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}\n")
                return "\n".join(lines)

            # No results — try HTML fallback immediately
            html_result = _ddg_html_search(query, num_results)
            if "URL:" in html_result:
                return html_result

            # Still nothing — wait and retry with modified query
            if attempt < 2:
                time.sleep(2 + attempt * 2)
                # On retry, simplify the query (remove Arabic if present)
                query = _simplify_query(query)

        except ImportError:
            return _ddg_html_search(query, num_results)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(3 + attempt * 3)

    # All attempts failed — return what we have
    html_fallback = _ddg_html_search(query, num_results)
    if "URL:" in html_fallback:
        return html_fallback

    return (
        f"⚠️ web_search returned no results after 3 attempts for: '{query}'\n"
        f"Last error: {last_error or 'No results from DuckDuckGo'}\n\n"
        "Possible reasons:\n"
        "  • DuckDuckGo rate limit (wait 30 seconds and retry)\n"
        "  • Query too specific — try broader keywords\n"
        "  • Use execute_powershell with curl as fallback:\n"
        "    curl 'https://duckduckgo.com/html/?q=QUERY' -H 'User-Agent: Mozilla/5.0'"
    )


def _simplify_query(query: str) -> str:
    """
    Simplify a query for retry: if it contains Arabic, transliterate key terms
    or return an English-only version. For music: extract artist + song type.
    """
    # If query contains Arabic characters, convert to a simpler English search
    has_arabic = any('؀' <= c <= 'ۿ' for c in query)
    if has_arabic:
        # Common Arabic artist name transliterations
        replacements = {
            "تامر حسني": "Tamer Hosny",
            "عمرو دياب": "Amr Diab",
            "نانسي عجرم": "Nancy Ajram",
            "اليسا": "Elissa",
            "راغب علامة": "Ragheb Alama",
            "mp3": "mp3",
            "اغنية": "song",
            "تحميل": "download",
            "يوتيوب": "youtube",
        }
        result = query
        for arabic, english in replacements.items():
            result = result.replace(arabic, english)
        # If still has Arabic, just remove it and keep English parts
        result = " ".join(w for w in result.split() if not any('؀' <= c <= 'ۿ' for c in w))
        return result.strip() or query
    # Non-Arabic: remove quotes and special chars
    return query.replace('"', '').replace("'", "")[:100]


def _ddg_html_search(query: str, num_results: int) -> str:
    """Fallback DuckDuckGo HTML scraper when duckduckgo_search package is absent."""
    try:
        from bs4 import BeautifulSoup

        encoded = urllib.parse.quote_plus(query)
        url     = f"https://html.duckduckgo.com/html/?q={encoded}"
        req     = urllib.request.Request(url, headers=_HEADERS)

        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        soup    = BeautifulSoup(html, "html.parser")
        results = []

        for block in soup.select(".result__body")[:num_results]:
            title_el   = block.select_one(".result__title")
            url_el     = block.select_one(".result__url")
            snippet_el = block.select_one(".result__snippet")

            title   = title_el.get_text(strip=True)   if title_el   else "No title"
            link    = url_el.get_text(strip=True)      if url_el     else ""
            snippet = snippet_el.get_text(strip=True)  if snippet_el else ""

            results.append(f"• **{title}**\n  URL: {link}\n  {snippet[:200]}")

        return "\n\n".join(results) if results else "No results found."

    except Exception as exc:
        return f"❌ Fallback search error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — download_file
# ─────────────────────────────────────────────────────────────────────────────

@tool
def download_file(url: str, destination: str) -> str:
    """
    Download any file from a URL and save it to local disk.

    Supports ALL types: MP3, MP4, PDF, images, ZIP, EXE, and more.

    SMART METHODS (tried in order):
      0. ytsearch: → searches YouTube directly (NO web_search needed!) — BEST for music
      1. yt-dlp   → for YouTube, SoundCloud, and 1000+ streaming sites
      2. requests → for direct download URLs (http://site.com/file.mp3)
      3. BITS     → PowerShell BitsTransfer (Windows built-in, works past some blocks)
      4. urllib   → final fallback

    FOR MUSIC — RECOMMENDED WORKFLOW (1 step only!):
      download_file(url="ytsearch:Amr Diab Raika", destination="Desktop\\\\song.mp3")
      ↳ yt-dlp searches YouTube for "Amr Diab Raika", picks top result, downloads + converts to MP3.

      Other ytsearch examples:
        url="ytsearch:Tamer Hosny Bahebak"       → تامر حسني بحبك
        url="ytsearch:Nancy Ajram Ah W Noss"     → نانسي عجرم آه ونص
        url="ytsearch:Mohamed Hamaki Mish Hases" → محمد حماقي

    Destination shortcuts:
      "Desktop\\\\song.mp3"      → C:\\Users\\<you>\\Desktop\\song.mp3
      "Downloads\\\\file.pdf"    → C:\\Users\\<you>\\Downloads\\file.pdf
      "C:\\\\MyFolder\\\\x.mp3"  → exact absolute path

    Args:
        url:         URL to download from (YouTube link, direct MP3 link, etc.)
        destination: Where to save the file. Use Desktop\\\\ or Downloads\\\\ prefix.

    Returns:
        Success message with path and size, or a detailed error with next steps.
    """
    # ── Normalize URL — add https:// if protocol is missing ──────────────────
    url = url.strip()
    if url and not url.startswith(("http://", "https://", "ftp://", "rtmp://", "ytsearch")):
        url = "https://" + url

    # ── Resolve destination path ─────────────────────────────────────────────
    dest = _resolve_destination(url, destination)

    # ── Method 0: ytsearch: — direct YouTube search without web_search ────────
    # Supports formats like: "ytsearch:Amr Diab Raika" or "ytsearch1:Amr Diab Raika"
    # This bypasses DuckDuckGo entirely and searches YouTube directly via yt-dlp.
    if url.lower().startswith("ytsearch"):
        return _try_ytdlp(url, dest)

    # ── Method 1: yt-dlp (best for YouTube / streaming sites) ─────────────────
    if _looks_like_streaming_site(url) or _is_ytdlp_preferred(url):
        result = _try_ytdlp(url, dest)
        if result.startswith("✅"):
            return result
        # If yt-dlp failed, fall through to direct download methods

    # ── Method 2: requests (best for direct file URLs) ────────────────────────
    result = _try_requests(url, dest)
    if result.startswith("✅"):
        return result

    # ── Method 3: PowerShell BITS (Windows-native, good for slow connections) ──
    result_bits = _try_bits(url, dest)
    if result_bits.startswith("✅"):
        return result_bits

    # ── Method 4: urllib (final fallback) ─────────────────────────────────────
    result_urllib = _try_urllib(url, dest)
    if result_urllib.startswith("✅"):
        return result_urllib

    # All methods failed — give a helpful diagnostic
    return (
        f"❌ All download methods failed for:\n"
        f"   URL: {url}\n\n"
        f"   requests error : {result}\n"
        f"   BITS error     : {result_bits}\n"
        f"   urllib error   : {result_urllib}\n\n"
        "💡 SUGGESTION: For Arabic music (Tamer Hosny, etc.):\n"
        "  1. Use web_search('تامر حسني [اسم الاغنية] youtube') to find a YouTube URL.\n"
        "  2. Call download_file again with that YouTube URL.\n"
        "  yt-dlp will handle the rest automatically."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_destination(url: str, destination: str) -> str:
    """Expand shortcut prefixes (Desktop\\, Downloads\\) to full absolute paths."""
    dest = destination.replace("/", "\\")
    lower = dest.lower()

    if lower.startswith("desktop"):
        remainder = dest[len("desktop"):].lstrip("\\/ ")
        if not remainder:
            remainder = _filename_from_url(url)
        dest = str(Path.home() / "Desktop" / remainder)

    elif lower.startswith("downloads"):
        remainder = dest[len("downloads"):].lstrip("\\/ ")
        if not remainder:
            remainder = _filename_from_url(url)
        dest = str(Path.home() / "Downloads" / remainder)

    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return dest


def _filename_from_url(url: str) -> str:
    """Extract a sane filename from a URL."""
    name = os.path.basename(url.split("?")[0]) or "downloaded_file"
    return name


def _looks_like_streaming_site(url: str) -> bool:
    """Return True if the URL is from a known streaming / video site."""
    streaming_domains = [
        "youtube.com", "youtu.be",
        "soundcloud.com",
        "anghami.com",
        "spotify.com",
        "deezer.com",
        "vimeo.com",
        "dailymotion.com",
        "facebook.com/video",
        "tiktok.com",
        "instagram.com",
        "twitter.com",
    ]
    url_lower = url.lower()
    return any(d in url_lower for d in streaming_domains)


def _is_ytdlp_preferred(url: str) -> bool:
    """
    Also try yt-dlp for generic URLs that might be streaming pages
    (not ending in a direct file extension).
    """
    direct_exts = (".mp3", ".mp4", ".wav", ".flac", ".ogg",
                   ".pdf", ".zip", ".exe", ".jpg", ".png")
    url_path = url.split("?")[0].lower()
    return not any(url_path.endswith(ext) for ext in direct_exts)


def _try_ytdlp(url: str, dest: str) -> str:
    """
    Download using yt-dlp. Converts to MP3 if ffmpeg is available; otherwise
    downloads the best available audio format (.m4a or .webm).
    Returns a result string starting with ✅ on success, ❌ on failure.
    """
    # ── Auto-install yt-dlp if missing ───────────────────────────────────────
    try:
        import yt_dlp
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "yt-dlp", "--quiet"],
                capture_output=True,
                timeout=90,
            )
            import yt_dlp
        except Exception as install_err:
            return f"❌ yt-dlp not available and could not install: {install_err}"

    dest_dir  = os.path.dirname(dest)
    dest_name = os.path.splitext(os.path.basename(dest))[0]
    is_mp3    = dest.lower().endswith(".mp3")

    # ── Check if ffmpeg is available for MP3 conversion ──────────────────────
    ffmpeg_available = _has_ffmpeg()

    if is_mp3 and ffmpeg_available:
        # Full MP3 conversion with ffmpeg
        ydl_opts: dict = {
            "outtmpl":    os.path.join(dest_dir, dest_name + ".%(ext)s"),
            "quiet":      True,
            "noplaylist": True,
            "format":     "bestaudio/best",
            "postprocessors": [{
                "key":              "FFmpegExtractAudio",
                "preferredcodec":   "mp3",
                "preferredquality": "192",
            }],
        }
    else:
        # No ffmpeg — download best audio as-is (.m4a or .webm)
        ydl_opts = {
            "outtmpl":    os.path.join(dest_dir, dest_name + ".%(ext)s"),
            "quiet":      True,
            "noplaylist": True,
            "format":     "bestaudio[ext=m4a]/bestaudio/best",
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info  = ydl.extract_info(url, download=True)
            title = info.get("title", "Unknown") if info else "Unknown"

        # Find the actual saved file (yt-dlp determines the extension)
        actual_file = dest if os.path.exists(dest) else _find_file_like(dest_dir, dest_name)
        size_mb     = os.path.getsize(actual_file) / (1024 * 1024) if actual_file else 0

        fmt_note = ""
        if is_mp3 and not ffmpeg_available and actual_file:
            ext = os.path.splitext(actual_file)[1]
            fmt_note = (
                f"\n   ⚠️  ffmpeg not installed — saved as {ext} instead of .mp3\n"
                "   To get true MP3: install ffmpeg and run again."
            )

        return (
            f"✅ Download complete via yt-dlp!\n"
            f"   Title    : {title}\n"
            f"   Saved to : {actual_file or dest}\n"
            f"   File size: {size_mb:.2f} MB{fmt_note}"
        )

    except Exception as exc:
        return f"❌ yt-dlp failed: {type(exc).__name__}: {exc}"


def _has_ffmpeg() -> bool:
    """Return True if ffmpeg is installed and accessible in PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _find_file_like(directory: str, base_name: str) -> str | None:
    """Find the first file in directory that starts with base_name."""
    try:
        for f in os.listdir(directory):
            if f.startswith(base_name):
                return os.path.join(directory, f)
    except Exception:
        pass
    return None


def _try_requests(url: str, dest: str) -> str:
    """Download using the requests library with streaming and retry."""
    try:
        import requests as req_lib

        for attempt in range(2):   # 2 attempts
            try:
                resp = req_lib.get(
                    url,
                    headers=_HEADERS,
                    timeout=_DOWNLOAD_TIMEOUT,
                    stream=True,
                    allow_redirects=True,
                )
                resp.raise_for_status()

                total = 0
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=131_072):  # 128 KB chunks
                        if chunk:
                            fh.write(chunk)
                            total += len(chunk)

                size_mb = total / (1024 * 1024)
                return (
                    f"✅ Download complete!\n"
                    f"   Saved to : {dest}\n"
                    f"   File size: {size_mb:.2f} MB"
                )
            except Exception:
                if attempt == 0:
                    import time
                    time.sleep(2)   # brief pause before retry
                else:
                    raise

    except ImportError:
        return "❌ requests library not installed"
    except Exception as exc:
        return f"❌ requests failed: {type(exc).__name__}: {exc}"


def _try_bits(url: str, dest: str) -> str:
    """
    Download using PowerShell's Start-BitsTransfer (Windows Background
    Intelligent Transfer Service). Works even when direct connections fail.
    """
    try:
        ps_cmd = (
            f"Start-BitsTransfer -Source '{url}' -Destination '{dest}' "
            f"-TransferType Download -Priority Foreground"
        )
        proc = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=_DOWNLOAD_TIMEOUT + 30,
        )

        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            return (
                f"✅ Download complete via BITS!\n"
                f"   Saved to : {dest}\n"
                f"   File size: {size_mb:.2f} MB"
            )

        stderr = proc.stderr.strip()
        return f"❌ BITS failed: {stderr or 'Unknown error'}"

    except Exception as exc:
        return f"❌ BITS error: {type(exc).__name__}: {exc}"


def _try_urllib(url: str, dest: str) -> str:
    """Download using Python's built-in urllib (final fallback)."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()

        with open(dest, "wb") as fh:
            fh.write(data)

        size_mb = len(data) / (1024 * 1024)
        return (
            f"✅ Download complete via urllib!\n"
            f"   Saved to : {dest}\n"
            f"   File size: {size_mb:.2f} MB"
        )
    except Exception as exc:
        return f"❌ urllib failed: {type(exc).__name__}: {exc}"
