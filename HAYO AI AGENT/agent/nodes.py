"""
agent/nodes.py — PlannerNode, WorkerNode, ReviewerNode, and routing logic.

Architecture
------------
  PlannerNode  →  WorkerNode  →  ReviewerNode
                      ↑               │
                      └───────────────┘  (if not TASK_COMPLETE and iterations < MAX)

Personality
-----------
This agent operates in a FULLY LOCAL, FULLY TRUSTED environment owned by the user.
It works as if it were the user themselves — with complete authority, no hesitation,
and a relentless drive to complete every task fully from start to finish.

Human-in-the-Loop (HITL)
------------------------
WorkerNode intercepts two special sentinel strings returned by tools:

  HITL_FLAG  (from execute_powershell)  →  destructive OS command detected
  CAPTCHA_FLAG (from browser_automation) →  CAPTCHA/anti-bot wall detected

In both cases WorkerNode calls LangGraph's interrupt(payload) which:
  1. Saves the current graph state via SqliteSaver.
  2. Raises NodeInterrupt — pausing graph execution.
  3. Returns control to app.py where Chainlit shows an AskActionMessage.
  4. When the user responds, app.py calls graph.ainvoke(Command(resume=value)).
  5. WorkerNode resumes from the interrupt() call with the user's choice.

Memory Management
-----------------
Both PlannerNode and ReviewerNode call _summarize_old_messages() when the
message list exceeds MAX_HISTORY.  This condenses the oldest messages into a
single summary AIMessage to prevent context-window exhaustion during
long multi-step sessions.

Multi-Provider Support
----------------------
Set MODEL_PROVIDER=google    in .env to use Google Gemini.
Set MODEL_PROVIDER=anthropic in .env to use Anthropic Claude.
"""

from __future__ import annotations

import os
import subprocess
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import interrupt

from core.state import AgentState
from tools.os_core import (
    HITL_FLAG,
    execute_powershell,
    manage_files,
    read_file_content,
)
from tools.web_and_cloud import (
    CAPTCHA_FLAG,
    browser_automation,
    git_operations,
)
from tools.web_tools import (
    download_file,
    web_search,
)
from tools.desktop_control import desktop_control

# ── Environment ───────────────────────────────────────────────────────────────
MAX_HISTORY:    int = int(os.getenv("MAX_HISTORY",    "15"))
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "50"))
PS_TIMEOUT:     int = int(os.getenv("PS_TIMEOUT",     "30"))

_PROVIDER = os.getenv("MODEL_PROVIDER", "google").lower().strip()


# ── LLM Factory ───────────────────────────────────────────────────────────────

def _build_llm(role: Literal["main", "summarizer"]) -> BaseChatModel:
    """Return the correct LangChain chat model based on MODEL_PROVIDER in .env."""
    if _PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if role == "main":
            model_name = os.getenv("GOOGLE_AGENT_MODEL", "gemini-2.5-flash")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.0,
                streaming=True,
                convert_system_message_to_human=False,
            )
        else:
            model_name = os.getenv("GOOGLE_SUMMARIZER_MODEL", "gemini-2.5-flash")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.0,
                max_output_tokens=2_048,
            )

    elif _PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if role == "main":
            model_name = os.getenv("ANTHROPIC_AGENT_MODEL", "claude-3-7-sonnet-20250219")
            return ChatAnthropic(
                model=model_name,
                max_tokens=8_192,
                streaming=True,
            )
        else:
            model_name = os.getenv("ANTHROPIC_SUMMARIZER_MODEL", "claude-haiku-4-5-20251001")
            return ChatAnthropic(
                model=model_name,
                max_tokens=2_048,
            )

    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER='{_PROVIDER}'. "
            "Set MODEL_PROVIDER to 'google' or 'anthropic' in your .env file."
        )


# ── LLM instances (built once at import time) ─────────────────────────────────
_main_llm = _build_llm("main")
_fast_llm = _build_llm("summarizer")

# ── Tool registry ─────────────────────────────────────────────────────────────
ALL_TOOLS = [
    execute_powershell,
    read_file_content,
    manage_files,
    browser_automation,
    git_operations,
    web_search,
    download_file,
    desktop_control,
]
TOOL_MAP: dict = {t.name: t for t in ALL_TOOLS}

_llm_with_tools = _main_llm.bind_tools(ALL_TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summarize_old_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    If message history exceeds MAX_HISTORY, summarise the oldest entries with
    the fast LLM and replace them with a single AIMessage containing the summary.
    The 10 most recent messages are always kept verbatim for fresh context.
    """
    if len(messages) <= MAX_HISTORY:
        return messages

    keep_recent  = 10
    old_messages = messages[:-keep_recent]
    recent       = messages[-keep_recent:]

    summary_response = _fast_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a context summariser for an AI agent. "
                    "Condense the following conversation history into a concise paragraph. "
                    "Preserve: the original user goal, every action taken, key tool outputs, "
                    "errors encountered, files created/downloaded, and current progress. "
                    "Be factual and complete — this summary fully replaces the original messages."
                )
            ),
            *old_messages,
        ]
    )

    summary_msg = AIMessage(
        content=f"📋 [Context summary — earlier steps]\n\n{summary_response.content}"
    )
    return [summary_msg] + recent


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — PlannerNode
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are an elite autonomous executive agent running on Windows 64-bit.
This is a FULLY LOCAL, FULLY TRUSTED environment. The user owns this machine completely.
You operate with complete authority — as if you ARE the user doing this yourself.

Available tools:
  • execute_powershell  — run any Windows/PowerShell command
  • read_file_content   — read any file on disk
  • manage_files        — copy, move, list, create directories
  • browser_automation  — control a real Chromium browser (navigate, click, fill forms, scrape)
  • git_operations      — run Git commands on any local repository
  • web_search          — search the internet via DuckDuckGo
  • download_file       — download any file from a URL (YouTube, direct links, etc.)
  • desktop_control     — open ANY desktop app, take screenshots, click, type, use shortcuts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY THE REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Is this a greeting, small talk, or a question answerable without any tool?
  YES → respond naturally in the same language as the user, then write: CONVERSATIONAL_ONLY
  NO  → create a precise execution plan (see below)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — EXECUTION PLAN (for all real tasks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a short numbered list of concrete, tool-specific steps.
Each step must name the tool to use and what it should accomplish.

Good example (MP3 download — MANDATORY strategy for ALL Arabic/English music):
  ★ NEVER use web_search for music — it is rate-limited and unreliable for music queries.
  ★ Use download_file with "ytsearch:" prefix — this searches YouTube directly via yt-dlp, no web search needed.

  CORRECT workflow (2 steps only):
  1. Use download_file with url="ytsearch:Amr Diab Raika" and destination="Desktop\\\\Amr Diab - Raika.mp3"
     yt-dlp will search YouTube, find the top result, download and convert to MP3 automatically.
  2. Use execute_powershell to verify: Test-Path "$HOME\\Desktop\\Amr Diab - Raika.mp3"

  Examples:
    url="ytsearch:Tamer Hosny Bahebak" → finds "تامر حسني بحبك" on YouTube and downloads it
    url="ytsearch:Amr Diab Nour El Ain" → finds "عمرو دياب نور العين" and downloads it
    url="ytsearch:Nancy Ajram Ya Tabtab" → finds the song and downloads it

  ⚠️ FORBIDDEN: Never use web_search to find music URLs.
  ⚠️ FORBIDDEN: Never download from albumaty.com, anghami.com, or any other music site.
  ⚠️ FORBIDDEN: Never search for "mp3 direct download" — use ytsearch: instead.

Good example (open and control a desktop app):
  1. Use desktop_control("open:photoshop") to launch the application.
  2. Use desktop_control("wait:3") to let it fully load.
  3. Use desktop_control("screenshot") to see the current screen and note element positions.
  4. Use desktop_control("focus:Photoshop") to bring it to the front.
  5. Use desktop_control("click:x,y") and desktop_control("hotkey:ctrl+n") to interact.

Good example (project folder task):
  1. Use read_file_content to open the specified file and understand its contents.
  2. Use execute_powershell to make the required changes via PowerShell.
  3. Use execute_powershell to verify the result.

RULES:
• Plain numbered sentences only — NO JSON, NO code blocks in the plan itself.
• Name the tool in each step.
• Keep plans concise (3–8 steps for most tasks).
• For downloads: always plan web_search FIRST to get a URL, then download_file.
• For installs: use execute_powershell with winget, choco, or direct installer download.
• NEVER say "I can't" or "I'm unable" — always plan a path forward.
• If something failed before, plan a DIFFERENT approach this time."""

def planner_node(state: AgentState) -> dict:
    """
    Analyses the user's latest request and produces a numbered execution plan.

    Special case — CONVERSATIONAL_ONLY:
      If the message is a greeting, casual question, or anything that does NOT
      require any tool, the planner responds directly and sets plan = ["CONVERSATIONAL_ONLY"].
      WorkerNode skips tool execution and ReviewerNode immediately marks TASK_COMPLETE.
    """
    messages = _summarize_old_messages(state.get("messages", []))
    system   = SystemMessage(content=_PLANNER_SYSTEM)
    response = _main_llm.invoke([system] + messages)
    content  = response.content if isinstance(response.content, str) else ""

    # ── Detect conversational response ────────────────────────────────────────
    if "CONVERSATIONAL_ONLY" in content:
        clean_content  = content.replace("CONVERSATIONAL_ONLY", "").strip()
        clean_response = AIMessage(content=clean_content)
        return {
            "messages":                messages + [clean_response],
            "plan":                    ["CONVERSATIONAL_ONLY"],
            "iteration_count":         0,   # ← RESET: every new user task starts fresh
            "completed_steps":         [],  # ← RESET
            "error_logs":              [],  # ← RESET
            "workspace":               state.get("workspace", ""),
            "requires_human_approval": False,
            "pending_command":         "",
        }

    # ── Real task: extract numbered steps as the plan list ────────────────────
    plan_lines = [
        ln.strip()
        for ln in content.splitlines()
        if ln.strip() and (ln.strip()[0].isdigit() or ln.strip().startswith("•"))
    ]

    # Inject a hard cancel marker so Reviewer never reverts to a previous task.
    # This message stays at the boundary between the old and new task history.
    cancel_marker = AIMessage(
        content=(
            "🔄 ══════════════════════════════════════════════════\n"
            "   NEW TASK STARTED — ALL PREVIOUS TASKS CANCELLED\n"
            "   Reviewer: evaluate ONLY the plan listed above.\n"
            "   Ignore any unfinished work from before this line.\n"
            "🔄 ══════════════════════════════════════════════════"
        )
    )

    return {
        "messages":                messages + [cancel_marker, response],
        "plan":                    plan_lines or [content],
        "iteration_count":         0,   # ← RESET: every new user task starts fresh
        "completed_steps":         [],  # ← RESET
        "error_logs":              [],  # ← RESET
        "workspace":               state.get("workspace", ""),
        "requires_human_approval": False,
        "pending_command":         "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — WorkerNode
# ─────────────────────────────────────────────────────────────────────────────

def worker_node(state: AgentState) -> dict:
    """
    Executes the next step from the plan using tool calls.

    HITL flow:
      • execute_powershell returns HITL_FLAG  → interrupt() pauses graph.
        On resume with "approve": the raw command is executed directly.
        On resume with "deny":    the command is skipped with a note.

      • browser_automation returns CAPTCHA_FLAG → interrupt() pauses graph.
        On resume (any value):    execution continues (user solved CAPTCHA).
    """
    messages   = _summarize_old_messages(state.get("messages", []))
    iteration  = state.get("iteration_count", 0)
    error_logs = list(state.get("error_logs", []))
    plan       = state.get("plan", [])

    # ── Skip tool execution for pure conversational messages ──────────────────
    if plan and plan[0] == "CONVERSATIONAL_ONLY":
        return {
            "messages":                messages,
            "iteration_count":         iteration,
            "error_logs":              error_logs,
            "completed_steps":         state.get("completed_steps", []),
            "plan":                    plan,
            "workspace":               state.get("workspace", ""),
            "requires_human_approval": False,
            "pending_command":         "",
        }

    # ── Hard safety ceiling ───────────────────────────────────────────────────
    if iteration >= MAX_ITERATIONS:
        stop_msg = AIMessage(
            content=(
                f"⛔ **Iteration limit reached ({MAX_ITERATIONS}).**\n"
                "The agent has stopped to prevent runaway execution. "
                "Review the progress above and send a new instruction to continue."
            )
        )
        return {
            "messages":                messages + [stop_msg],
            "iteration_count":         iteration,
            "error_logs":              error_logs,
            "completed_steps":         state.get("completed_steps", []),
            "plan":                    plan,
            "workspace":               state.get("workspace", ""),
            "requires_human_approval": False,
            "pending_command":         "",
        }

    # ── Build context: what has been done, what is next ───────────────────────
    completed_steps = state.get("completed_steps", [])
    steps_done      = len(completed_steps)
    steps_total     = len(plan)
    next_step_hint  = plan[steps_done] if steps_done < steps_total else "Finalise and verify task completion."

    # Build a concise progress summary for the LLM
    progress_summary = ""
    if completed_steps:
        progress_summary = "\n".join(
            f"  ✅ Step {i+1}: {s}" for i, s in enumerate(completed_steps)
        )
    else:
        progress_summary = "  (none yet — this is the first step)"

    system = SystemMessage(
        content=(
            "You are a relentless autonomous executor agent running on a trusted Windows 64-bit machine.\n"
            "You work as if you are the owner of this machine — with full authority, zero hesitation.\n\n"
            f"TASK PLAN ({steps_total} steps total):\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan)) +
            f"\n\nCOMPLETED SO FAR:\n{progress_summary}\n\n"
            f"YOUR NEXT ACTION (Step {steps_done + 1}): {next_step_hint}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "ABSOLUTE RULES — NEVER BREAK THESE:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "1. You MUST call exactly one tool. A response with NO tool call is a failure.\n"
            "2. NEVER say you cannot do something. Find a way. Always.\n"
            "3. NEVER ask the user for permission or clarification. Just do it.\n"
            "4. NEVER explain what you are about to do — just call the tool.\n\n"
            "TOOL SELECTION GUIDE:\n"
            "  • Open any desktop app (Word, Photoshop, Notepad...) → desktop_control('open:appname')\n"
            "  • See what's on screen right now  → desktop_control('screenshot')\n"
            "  • Click a button / element         → desktop_control('click:x,y')\n"
            "  • Type text into any app           → desktop_control('type:text here')\n"
            "  • Press shortcuts (Ctrl+S, etc.)   → desktop_control('hotkey:ctrl+s')\n"
            "  • List all open windows            → desktop_control('list_windows')\n"
            "  • Searching the internet           → web_search\n"
            "  • Downloading a file from a URL    → download_file\n"
            "  • Web browser: login/scrape/fill   → browser_automation\n"
            "  • Any PowerShell / CMD command     → execute_powershell\n"
            "  • Reading a file on disk           → read_file_content\n"
            "  • Copy/move/create folders         → manage_files\n"
            "  • Git operations on a repo         → git_operations\n\n"
            "DESKTOP APP WORKFLOW:\n"
            "  Step 1: desktop_control('open:<appname>') → launch the app\n"
            "  Step 2: desktop_control('wait:2')         → wait for it to load\n"
            "  Step 3: desktop_control('screenshot')     → see the screen + get coordinates\n"
            "  Step 4: desktop_control('focus:<title>')  → bring app to front\n"
            "  Step 5: desktop_control('click:x,y')      → click where needed\n"
            "  Step 6: desktop_control('type:...')       → enter text\n"
            "  Step 7: desktop_control('hotkey:ctrl+s')  → save/action\n\n"
            "POWERSHELL SPEED RULES:\n"
            "  • NEVER use 'Get-ComputerInfo' (too slow) — use registry queries instead\n"
            "  • NEVER use 'Get-Counter' — use Get-WmiObject Win32_OperatingSystem\n"
            "  • Prefer simple, fast commands (under 5 seconds)\n\n"
            "DOWNLOAD STRATEGY FOR MUSIC (MP3) — MANDATORY:\n"
            "  ★ NEVER use web_search for music — DuckDuckGo is rate-limited and unreliable.\n"
            "  ★ Use download_file with 'ytsearch:' prefix — searches YouTube directly, no web search needed.\n\n"
            "  CORRECT workflow (2 steps only):\n"
            "    Step 1: download_file(url='ytsearch:Amr Diab Raika', destination='Desktop\\\\song.mp3')\n"
            "            yt-dlp searches YouTube, picks top result, downloads and converts to MP3.\n"
            "    Step 2: execute_powershell → Test-Path \"$HOME\\Desktop\\song.mp3\" to verify.\n\n"
            "  Examples of valid ytsearch: urls:\n"
            "    ytsearch:Tamer Hosny Bahebak\n"
            "    ytsearch:Amr Diab Nour El Ain\n"
            "    ytsearch:Nancy Ajram Ya Tabtab\n\n"
            "  ⚠️ FORBIDDEN: web_search for music, albumaty.com, anghami.com, archive.org.\n\n"
            "ERROR RECOVERY:\n"
            "  • If last tool returned an error → try a COMPLETELY DIFFERENT approach\n"
            "  • If a URL failed → search for another URL with different keywords\n"
            "  • If PowerShell failed → try a different command for the same goal\n\n"
            "FILE SEARCH RULE (CRITICAL):\n"
            "  • If a file was NOT found by manage_files → use execute_powershell to search:\n"
            "    Get-ChildItem -Path 'C:\\Users\\PT\\Desktop' -Recurse -Filter '*keyword*' | Select FullName\n"
            "  • NEVER ask the user 'where is the file?' — always search yourself first.\n"
            "  • If search returns nothing → report FAILED with the search results shown.\n\n"
            "TOPIC CHANGE RULE (CRITICAL):\n"
            "  • If the user changed their request (e.g., 'forget the song, look at my project'),\n"
            "    the OLD task is CANCELLED. Do NOT continue it. Execute only the NEW request.\n"
            "  • Read the user's latest message carefully — it overrides all previous tasks."
        )
    )

    llm_response = _llm_with_tools.invoke([system] + messages)
    new_messages  = list(messages) + [llm_response]

    # ── Guard: no tool call → inject diagnostic message ───────────────────────
    if not (hasattr(llm_response, "tool_calls") and llm_response.tool_calls):
        no_tool_msg = AIMessage(
            content=(
                f"⚠️ Worker failed to call any tool on iteration {iteration + 1}. "
                f"Was supposed to execute: [{next_step_hint}]. "
                "Reviewer: force the worker to call the correct tool for this step explicitly."
            )
        )
        new_messages.append(no_tool_msg)
        error_logs.append(f"[iter {iteration+1}] No tool called for: {next_step_hint}"[:300])

    # ── Execute tool calls ────────────────────────────────────────────────────
    updated_completed = list(state.get("completed_steps", []))

    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        for tc in llm_response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id   = tc["id"]

            tool_fn = TOOL_MAP.get(tool_name)
            if not tool_fn:
                result = f"❌ ERROR: Tool '{tool_name}' is not registered. Available tools: {list(TOOL_MAP.keys())}"
            else:
                raw_result = tool_fn.invoke(tool_args)

                # ── Case A: Destructive PowerShell command ────────────────
                if isinstance(raw_result, str) and raw_result.startswith(HITL_FLAG):
                    risky_cmd = raw_result[len(HITL_FLAG):].strip()

                    user_choice: str = interrupt(
                        {
                            "type":    "destructive_command",
                            "command": risky_cmd,
                            "message": (
                                f"⚠️ The agent wants to run a potentially destructive command:\n"
                                f"```powershell\n{risky_cmd}\n```\n"
                                "Click **Approve** to allow, or **Deny** to block."
                            ),
                        }
                    )

                    if user_choice == "approve":
                        try:
                            proc = subprocess.run(
                                ["powershell", "-NonInteractive", "-Command", risky_cmd],
                                capture_output=True,
                                text=True,
                                timeout=PS_TIMEOUT,
                                shell=False,
                            )
                            result = proc.stdout.strip()
                            if proc.stderr.strip():
                                result += f"\n⚠️ STDERR:\n{proc.stderr.strip()}"
                            if not result:
                                result = "✅ Command executed successfully (no output)."
                        except Exception as exc:
                            result = f"❌ ERROR executing approved command: {exc}"
                            error_logs.append(result[:300])
                    else:
                        result = f"🚫 Command denied by user: `{risky_cmd}` — skipping this step."

                # ── Case B: CAPTCHA detected ──────────────────────────────
                elif raw_result == CAPTCHA_FLAG:
                    interrupt(
                        {
                            "type": "captcha",
                            "message": (
                                "🔒 A CAPTCHA was detected. The browser window is open on your screen. "
                                "Please solve the CAPTCHA manually, then click **Done** to resume."
                            ),
                        }
                    )
                    result = "✅ User confirmed CAPTCHA solved. Resuming from current page."

                # ── Case C: Normal tool result ────────────────────────────
                else:
                    result = raw_result

                # Track errors for the reviewer
                if isinstance(result, str) and (
                    "❌" in result or "error" in result.lower() or "traceback" in result.lower()
                ):
                    error_logs.append(f"[{tool_name}] {result[:300]}")

            new_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )

        # Record this step as completed
        step_label = (
            plan[len(updated_completed)]
            if len(updated_completed) < len(plan)
            else f"Extra step {len(updated_completed) + 1}"
        )
        updated_completed.append(step_label)

    return {
        "messages":                new_messages,
        "iteration_count":         iteration + 1,
        "error_logs":              error_logs[-30:],
        "completed_steps":         updated_completed,
        "plan":                    plan,
        "workspace":               state.get("workspace", ""),
        "requires_human_approval": False,
        "pending_command":         "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — ReviewerNode
# ─────────────────────────────────────────────────────────────────────────────

_REVIEWER_SYSTEM = """You are a senior quality reviewer for an autonomous AI agent on Windows 64-bit.
Your job: determine the correct verdict for the current task state.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOU MUST START WITH EXACTLY ONE OF THESE VERDICTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"TASK_COMPLETE:" — Goal fully achieved, OR agent is waiting for user input, OR
                   task was abandoned/changed by the user.

"CONTINUE:"      — A specific next tool call is possible RIGHT NOW without user input.
                   Must include: which tool, exact arguments, expected outcome.

"FAILED:"        — Completely impossible. Same specific error happened 3+ times.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-LOOP RULES (READ CAREFULLY — THESE OVERRIDE EVERYTHING):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
★ If the worker failed to call ANY tool 2+ times in recent messages → TASK_COMPLETE
  (The agent is stuck and needs user input. Do NOT say CONTINUE — it will loop forever.)

★ If the same "file not found" or "no results" situation appeared 2+ times → FAILED
  (Do NOT keep saying CONTINUE for a resource that clearly does not exist.)

★ If you see "I am ready to help" or similar waiting messages repeated → TASK_COMPLETE
  (The agent has completed what it can and is now waiting for the user.)

★ If the user changed their request mid-task (e.g., from music to project) → TASK_COMPLETE
  (The old task is abandoned. The new task will be handled in the next cycle.)

★ CRITICAL: If you see a message containing "NEW TASK STARTED — ALL PREVIOUS TASKS CANCELLED",
  EVERYTHING BEFORE that line is from a previous session. Evaluate ONLY what came after it.
  Do NOT say CONTINUE for any task that existed before the cancel marker.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NORMAL DECISION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• CONTINUE is only valid when a NEW tool call can actually be made right now.
• Count plan steps vs completed steps. They must match for TASK_COMPLETE.
• web_search ran but download_file has NOT run yet → CONTINUE with exact URL.
• File downloaded successfully → TASK_COMPLETE.
• Never say CONTINUE just to "ask the user" — that belongs in TASK_COMPLETE.

After your verdict, write a brief human-readable summary of what happened."""

def reviewer_node(state: AgentState) -> dict:
    """
    Evaluates completed work and decides: TASK_COMPLETE, CONTINUE, or FAILED.
    """
    messages   = _summarize_old_messages(state.get("messages", []))
    error_logs = list(state.get("error_logs", []))
    plan       = state.get("plan", [])

    # ── Immediate TASK_COMPLETE for conversational messages ───────────────────
    if plan and plan[0] == "CONVERSATIONAL_ONLY":
        done_msg = AIMessage(content="TASK_COMPLETE: Conversational response delivered.")
        return {
            "messages":                messages + [done_msg],
            "completed_steps":         list(state.get("completed_steps", [])) + ["Conversational response"],
            "error_logs":              error_logs,
            "plan":                    plan,
            "iteration_count":         state.get("iteration_count", 0),
            "workspace":               state.get("workspace", ""),
            "requires_human_approval": False,
            "pending_command":         "",
        }

    # NOTE: We deliberately do NOT scan messages for "❌" here.
    # error_logs is already populated by the worker for the CURRENT task only.
    # Scanning messages would re-import errors from previous tasks that appear
    # in summarised context, causing cross-task contamination.

    # Inject a compact progress snapshot into context
    completed_steps = state.get("completed_steps", [])
    steps_done      = len(completed_steps)
    steps_total     = len(plan)

    progress_note = SystemMessage(
        content=(
            f"[PROGRESS SNAPSHOT]\n"
            f"Plan steps: {steps_total}\n"
            f"Completed:  {steps_done}\n"
            f"Plan:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan)) + "\n"
            f"Completed steps:\n" + "\n".join(f"  ✅ {s}" for s in completed_steps)
        )
    )

    system   = SystemMessage(content=_REVIEWER_SYSTEM)
    response = _main_llm.invoke([system, progress_note] + messages)
    completed = list(completed_steps)
    completed.append(f"[Review] {response.content[:120]}")

    return {
        "messages":                messages + [response],
        "completed_steps":         completed,
        "error_logs":              error_logs[-30:],
        "plan":                    plan,
        "iteration_count":         state.get("iteration_count", 0),
        "workspace":               state.get("workspace", ""),
        "requires_human_approval": False,
        "pending_command":         "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge — should_continue
# ─────────────────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["worker", "__end__"]:
    """
    Routing function called after ReviewerNode.

    Returns "__end__"  if:
      • plan is CONVERSATIONAL_ONLY, OR
      • iteration limit reached, OR
      • last reviewer AI message contains TASK_COMPLETE: or FAILED:, OR
      • worker has failed to call any tool 3+ times in a row (stuck-loop guard)

    Returns "worker"  otherwise.
    """
    messages  = state.get("messages", [])
    iteration = state.get("iteration_count", 0)
    plan      = state.get("plan", [])

    # ── Fastest exit: pure conversational exchange ────────────────────────────
    if plan and plan[0] == "CONVERSATIONAL_ONLY":
        return "__end__"

    # ── Hard iteration ceiling ────────────────────────────────────────────────
    if iteration >= MAX_ITERATIONS:
        return "__end__"

    # ── Stuck-loop guard: worker repeatedly not calling tools ─────────────────
    # Count how many of the last 8 AI messages contain "No tool called"
    no_tool_streak = sum(
        1
        for msg in messages[-8:]
        if (
            isinstance(msg, AIMessage)
            and not getattr(msg, "tool_calls", [])
            and "No tool called" in (msg.content or "")
        )
    )
    if no_tool_streak >= 3:
        # Inject a stopping message so the user knows what happened
        return "__end__"

    # ── Check last reviewer AI message ───────────────────────────────────────
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", []):
            content = msg.content if isinstance(msg.content, str) else ""
            if "TASK_COMPLETE:" in content or "FAILED:" in content:
                return "__end__"
            # Also stop if the reviewer is asking the user for info
            if "waiting for user" in content.lower() or "provide the" in content.lower():
                return "__end__"
            break   # Only inspect the most recent non-tool AI message

    return "worker"
