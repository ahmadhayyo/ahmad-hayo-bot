"""
tools/web_and_cloud.py — Browser Automation & Git Operations

Two LangChain @tool functions:
  1. browser_automation — Playwright (headless=False) with Stealth mode and CAPTCHA detection.
  2. git_operations     — GitPython-based repository management.

CAPTCHA HANDLING
----------------
After navigating to a URL the tool inspects both the page <title> and the DOM for
well-known CAPTCHA / anti-bot fingerprints (Cloudflare, reCAPTCHA, hCaptcha, etc.).
If a CAPTCHA is detected the tool immediately returns the sentinel string
CAPTCHA_DETECTED_ACTION_REQUIRED without crashing.  The WorkerNode intercepts this
sentinel and calls LangGraph's interrupt() so the user can solve the CAPTCHA manually
in the visible browser window, then resume the agent.
"""

import os
from langchain_core.tools import tool

# ── Sentinel returned when CAPTCHA/anti-bot wall is detected ─────────────────
CAPTCHA_FLAG = "CAPTCHA_DETECTED_ACTION_REQUIRED"

# ── Page title substrings that indicate a bot-challenge page ─────────────────
CAPTCHA_TITLE_SIGNALS: list[str] = [
    "just a moment",
    "attention required",
    "security check",
    "captcha",
    "are you human",
    "verify you are human",
    "robot check",
    "ddos-guard",
    "access denied",
    "cloudflare",
]

# ── CSS selectors that indicate a CAPTCHA widget in the DOM ──────────────────
CAPTCHA_SELECTORS: list[str] = [
    "iframe[src*='recaptcha']",
    "iframe[src*='hcaptcha']",
    "iframe[src*='captcha']",
    ".g-recaptcha",
    ".h-captcha",
    "#cf-challenge-running",
    "#cf-spinner",
    ".cf-browser-verification",
    "#challenge-running",
    "#challenge-form",
    "[data-sitekey]",
]

# ── Default Downloads folder ──────────────────────────────────────────────────
DOWNLOADS_PATH: str = os.path.join(os.path.expanduser("~"), "Downloads")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — browser_automation
# ─────────────────────────────────────────────────────────────────────────────

@tool
def browser_automation(task: str, url: str = "") -> str:
    """
    Automate browser tasks using Playwright with stealth fingerprint masking.

    The browser always launches in VISIBLE (non-headless) mode so that if a
    CAPTCHA appears the user can see and solve it.  Playwright-stealth is applied
    immediately after page creation to minimise detection risk.

    CAPTCHA handling: if a bot-challenge page is detected after navigation, the
    function returns CAPTCHA_DETECTED_ACTION_REQUIRED.  The agent will then pause
    and wait for the user to solve the CAPTCHA before resuming.

    Args:
        task: Plain-English description of what to do (scrape, click, fill form, etc.).
        url:  Starting URL.  If empty the browser opens a blank page.

    Returns:
        Visible page text (≤ 6 000 chars), a status message, or the CAPTCHA sentinel.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    # Import stealth — optional but strongly recommended
    try:
        from playwright_stealth import stealth_sync
        _has_stealth = True
    except ImportError:
        _has_stealth = False

    with sync_playwright() as pw:
        # Launch visible Chromium with anti-detection flags
        browser = pw.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--start-maximized",
            ],
        )

        context = browser.new_context(
            accept_downloads=True,
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
        )

        page = context.new_page()

        # ── Apply stealth mask ───────────────────────────────────────────────
        if _has_stealth:
            stealth_sync(page)

        try:
            # ── Navigate ─────────────────────────────────────────────────────
            if url:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(1_500)   # allow JS to settle

                # ── CAPTCHA detection — title ────────────────────────────────
                title_lower = (page.title() or "").lower()
                for signal in CAPTCHA_TITLE_SIGNALS:
                    if signal in title_lower:
                        browser.close()
                        return CAPTCHA_FLAG

                # ── CAPTCHA detection — DOM selectors ────────────────────────
                for selector in CAPTCHA_SELECTORS:
                    try:
                        el = page.query_selector(selector)
                        if el:
                            browser.close()
                            return CAPTCHA_FLAG
                    except Exception:
                        pass

            # ── Extract visible text ─────────────────────────────────────────
            from bs4 import BeautifulSoup
            raw_html = page.content()
            soup = BeautifulSoup(raw_html, "html.parser")

            # Remove script/style noise
            for tag in soup(["script", "style", "noscript", "meta", "link"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Collapse blank lines
            lines = [ln for ln in text.splitlines() if ln.strip()]
            clean_text = "\n".join(lines)

            # Limit output to avoid flooding context window
            if len(clean_text) > 6_000:
                clean_text = clean_text[:6_000] + "\n\n[... TRUNCATED ...]"

            browser.close()
            return clean_text if clean_text else "✅ Page loaded but no visible text found."

        except PWTimeout:
            browser.close()
            return "❌ ERROR: Page navigation timed out (30 s). The site may be slow or blocked."
        except Exception as exc:
            browser.close()
            return f"❌ ERROR during browser automation: {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — git_operations
# ─────────────────────────────────────────────────────────────────────────────

@tool
def git_operations(repo_path: str, command: str) -> str:
    """
    Perform Git operations on a local repository using GitPython.

    Supported commands:
      - 'clone:<URL>'      Clone a remote repository into repo_path.
      - 'status'           Show working-tree status.
      - 'add_all'          Stage all changes (git add --all).
      - 'commit:<message>' Commit staged changes with the given message.
      - 'pull'             Pull latest changes from remote origin.
      - 'push'             Push committed changes to remote origin.
      - 'log'              Show last 10 commits (one-line format).
      - 'branch'           List all local branches.
      - 'checkout:<name>'  Switch to or create a branch.
      - 'diff'             Show unstaged diff of current changes.

    Args:
        repo_path: Absolute path to the local repository (or clone target).
        command:   One of the command strings above.

    Returns:
        Git output as a string, or an error message.
    """
    import git

    try:
        # ── clone is special: repo doesn't exist yet ─────────────────────────
        if command.startswith("clone:"):
            remote_url = command.split(":", 1)[1].strip()
            git.Repo.clone_from(remote_url, repo_path, progress=None)
            return f"✅ Repository cloned from '{remote_url}' into '{repo_path}'"

        # ── all other commands require an existing repo ───────────────────────
        repo = git.Repo(repo_path)

        if command == "status":
            return repo.git.status()

        elif command == "add_all":
            repo.git.add("--all")
            return "✅ All changes staged."

        elif command.startswith("commit:"):
            msg = command.split(":", 1)[1].strip()
            if not msg:
                return "❌ ERROR: Commit message cannot be empty."
            repo.git.commit("-m", msg)
            return f"✅ Committed with message: '{msg}'"

        elif command == "pull":
            output = repo.git.pull()
            return f"✅ Pull complete.\n{output}"

        elif command == "push":
            output = repo.git.push()
            return f"✅ Push complete.\n{output}"

        elif command == "log":
            return repo.git.log("--oneline", "-20")

        elif command == "branch":
            return repo.git.branch("-a")

        elif command.startswith("checkout:"):
            branch = command.split(":", 1)[1].strip()
            try:
                repo.git.checkout(branch)
            except git.exc.GitCommandError:
                # Branch doesn't exist — create it
                repo.git.checkout("-b", branch)
            return f"✅ Switched to branch '{branch}'"

        elif command == "diff":
            diff = repo.git.diff()
            return diff if diff else "✅ No unstaged changes."

        else:
            return (
                f"❌ ERROR: Unknown git command '{command}'.\n"
                "Valid commands: clone:<URL>, status, add_all, commit:<msg>, "
                "pull, push, log, branch, checkout:<name>, diff"
            )

    except git.exc.InvalidGitRepositoryError:
        return f"❌ ERROR: '{repo_path}' is not a valid Git repository."
    except git.exc.GitCommandError as exc:
        return f"❌ Git error: {exc}"
    except Exception as exc:
        return f"❌ ERROR: {type(exc).__name__}: {exc}"
