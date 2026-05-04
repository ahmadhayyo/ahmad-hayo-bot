"""
Browser tools — Playwright with a persistent profile.

Why persistent context (not headless launch):
  - Cookies and logged-in sessions survive between agent runs (Gmail, banking,
    GitHub, etc. won't ask for 2FA on every task).
  - The profile is stored under .browser_profile/ inside the project dir.

Headful by default — the user wants to SEE the agent driving the browser.
Set BROWSER_HEADLESS=true in .env to flip.

The browser is a process-wide singleton (one window across the whole agent run).
Closing happens at agent shutdown via close_browser().
"""

from __future__ import annotations

import asyncio
import threading
from typing import Annotated, Optional

from langchain_core.tools import tool

from config import BROWSER_HEADLESS, BROWSER_USER_DATA_DIR, DOWNLOADS_DIR

# ── Module-level singletons ──────────────────────────────────────────────────
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_browser = None  # playwright BrowserContext (persistent)
_page = None  # active Page
_playwright = None
_init_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """
    Run an asyncio loop in a background thread so sync tools can call async
    Playwright APIs via asyncio.run_coroutine_threadsafe.
    """
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return _loop

    started = threading.Event()

    def runner():
        global _loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop = loop
        started.set()
        loop.run_forever()

    _loop_thread = threading.Thread(target=runner, daemon=True, name="playwright-loop")
    _loop_thread.start()
    started.wait(timeout=5.0)
    assert _loop is not None
    return _loop


def _run(coro):
    """Submit a coroutine to the background loop and wait for the result."""
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=120)


async def _ensure_browser():
    global _browser, _page, _playwright
    if _page is not None and not _page.is_closed():
        return
    if _playwright is None:
        from playwright.async_api import async_playwright

        _playwright = await async_playwright().start()
    if _browser is None:
        BROWSER_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        _browser = await _playwright.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_USER_DATA_DIR),
            headless=BROWSER_HEADLESS,
            accept_downloads=True,
            viewport={"width": 1280, "height": 800},
            args=["--start-maximized"],
        )
    if not _browser.pages:
        _page = await _browser.new_page()
    else:
        _page = _browser.pages[0]


@tool
def browser_open(
    url: Annotated[str, "Full URL including http(s)://"],
) -> str:
    """Open a URL in the persistent browser. The same window is reused across calls."""
    with _init_lock:

        async def _do():
            await _ensure_browser()
            await _page.goto(url, wait_until="domcontentloaded", timeout=45000)
            return _page.url

        try:
            final_url = _run(_do())
            return f"[OK] Loaded {final_url}"
        except Exception as exc:
            return f"[ERROR] {type(exc).__name__}: {exc}"


@tool
def browser_get_text(
    selector: Annotated[
        str, "CSS selector. Use 'body' for the whole page text."
    ] = "body",
    max_chars: Annotated[int, "Cap returned text length."] = 6000,
) -> str:
    """Read visible text from the current page (or a specific element)."""

    async def _do():
        await _ensure_browser()
        try:
            text = await _page.inner_text(selector, timeout=10000)
        except Exception:
            text = await _page.content()
        return text

    try:
        text = _run(_do())
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n...[truncated, total={len(text)} chars]"
    return text


@tool
def browser_click(
    selector: Annotated[str, "CSS or text= selector for the element to click."],
) -> str:
    """Click a button or link by selector. Use 'text=Submit' for text matching."""

    async def _do():
        await _ensure_browser()
        await _page.click(selector, timeout=15000)
        await _page.wait_for_load_state("domcontentloaded", timeout=15000)
        return _page.url

    try:
        url = _run(_do())
        return f"[OK] Clicked '{selector}'. Now at: {url}"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"


@tool
def browser_fill(
    selector: Annotated[str, "CSS selector for an input."],
    value: Annotated[str, "Value to type."],
) -> str:
    """Fill an input or textarea."""

    async def _do():
        await _ensure_browser()
        await _page.fill(selector, value, timeout=15000)

    try:
        _run(_do())
        return f"[OK] Filled '{selector}'"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"


@tool
def browser_press(
    key: Annotated[str, "Key to press, e.g. 'Enter', 'Escape', 'Tab', 'Control+a'."],
) -> str:
    """Send a single keystroke to the page."""

    async def _do():
        await _ensure_browser()
        await _page.keyboard.press(key)

    try:
        _run(_do())
        return f"[OK] Pressed {key}"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"


@tool
def browser_screenshot(
    path: Annotated[
        str, "Where to save the PNG. Defaults to <Desktop>/agent_screenshot.png"
    ] = "",
    full_page: Annotated[bool, "Capture full scrollable page."] = False,
) -> str:
    """Save a screenshot of the current browser page."""
    from pathlib import Path

    from config import DESKTOP_DIR

    target = Path(path) if path else DESKTOP_DIR / "agent_screenshot.png"
    target.parent.mkdir(parents=True, exist_ok=True)

    async def _do():
        await _ensure_browser()
        await _page.screenshot(path=str(target), full_page=full_page)

    try:
        _run(_do())
        return f"[OK] Saved screenshot -> {target}"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"


@tool
def browser_download_via_click(
    selector: Annotated[str, "Selector for the download button/link."],
    filename: Annotated[str, "Optional filename. Defaults to server-suggested name."] = "",
) -> str:
    """
    Click an element that triggers a file download, save to Downloads folder.
    Use this when the file isn't available as a direct URL (e.g. requires auth).
    """

    async def _do():
        await _ensure_browser()
        async with _page.expect_download(timeout=60000) as dl_info:
            await _page.click(selector)
        download = await dl_info.value
        suggested = download.suggested_filename
        target = DOWNLOADS_DIR / (filename or suggested)
        await download.save_as(str(target))
        return str(target)

    try:
        path = _run(_do())
        return f"[OK] Downloaded -> {path}"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"


@tool
def browser_eval_js(
    expression: Annotated[str, "JavaScript expression to evaluate in the page context."],
) -> str:
    """Execute JS in the page and return the JSON-serializable result."""

    async def _do():
        await _ensure_browser()
        return await _page.evaluate(expression)

    try:
        result = _run(_do())
        return f"[OK] {result!r}"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"


@tool
def browser_wait_for(
    selector: Annotated[str, "Selector that must appear before continuing."],
    timeout_ms: Annotated[int, "Max wait in milliseconds."] = 15000,
) -> str:
    """Wait for an element to appear in the DOM."""

    async def _do():
        await _ensure_browser()
        await _page.wait_for_selector(selector, timeout=timeout_ms)

    try:
        _run(_do())
        return f"[OK] Found '{selector}'"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"


@tool
def browser_close() -> str:
    """Close the browser (use sparingly — the persistent context is fine to keep open)."""
    global _browser, _page, _playwright

    async def _do():
        global _browser, _page, _playwright
        if _browser:
            await _browser.close()
            _browser = None
            _page = None
        if _playwright:
            await _playwright.stop()
            _playwright = None

    try:
        _run(_do())
        return "[OK] Browser closed"
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"
