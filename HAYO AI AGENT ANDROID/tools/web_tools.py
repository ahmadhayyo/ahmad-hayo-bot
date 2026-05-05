"""
Android web tools — search, browse text, download.
Uses DuckDuckGo for search, curl for fetching.
"""

from __future__ import annotations

import subprocess
from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """البحث في الويب عبر DuckDuckGo.
    
    Args:
        query: نص البحث
        max_results: عدد النتائج (افتراضي 5)
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'N/A')}")
            lines.append(f"   {r.get('href', '')}")
            lines.append(f"   {r.get('body', '')[:150]}")
            lines.append("")
        return "\n".join(lines)
    except ImportError:
        return "[ERROR] duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def fetch_url(url: str, max_chars: int = 10_000) -> str:
    """جلب محتوى صفحة ويب كنص.
    
    Args:
        url: رابط الصفحة
        max_chars: الحد الأقصى للنص (افتراضي 10000)
    """
    try:
        import requests
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        
        # Try to extract text from HTML
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            text = resp.text
        
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, {len(text)} total chars)"
        return text
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def download_url(url: str, save_path: str = "/sdcard/Download/") -> str:
    """تحميل ملف من رابط URL.
    
    Args:
        url: رابط التحميل
        save_path: مسار الحفظ (افتراضي: /sdcard/Download/)
    """
    try:
        from pathlib import Path
        dest = Path(save_path).expanduser()
        if dest.is_dir():
            filename = url.split("/")[-1].split("?")[0] or "downloaded_file"
            dest = dest / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        r = subprocess.run(
            ["curl", "-L", "-o", str(dest), url],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0 and dest.exists():
            return f"Downloaded → {dest} ({dest.stat().st_size} bytes)"
        return f"[ERROR] Download failed: {r.stderr[:200]}"
    except Exception as e:
        return f"[ERROR] {e}"
