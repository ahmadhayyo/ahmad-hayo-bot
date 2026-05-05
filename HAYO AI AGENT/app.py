"""
app.py — Chainlit Web UI for the HAYO AI Agent.

Features:
  • Model selector: switch between Google Gemini, Claude, ChatGPT, DeepSeek
  • Persistent session IDs saved to disk → agent resumes after server restart.
  • "Continue" detection → typing أكمل / continue / استمر resumes last task.
  • Streaming intermediate steps via cl.Step.
  • File upload support (all formats and sizes).
  • Human-in-the-Loop (HITL) with fixed cl.Action payload API.
  • Chat memory with thinking/execution/progress display.
  • Screenshot integration for desktop capture.
"""
from __future__ import annotations
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # must run BEFORE agent imports read env vars

import chainlit as cl            # noqa: E402
from langchain_core.messages import AIMessageChunk, HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402
from agent.workflow import compile_graph  # noqa: E402

# ── Provider configuration ────────────────────────────────────────────────────
_PROVIDER = os.getenv("MODEL_PROVIDER", "google").lower().strip()

_PROVIDERS = {
    "google": {
        "label": "Google Gemini",
        "icon": "🟦",
        "model_var": "GOOGLE_AGENT_MODEL",
        "default_model": "gemini-2.5-flash",
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "icon": "🟠",
        "model_var": "ANTHROPIC_AGENT_MODEL",
        "default_model": "claude-sonnet-4-20250514",
    },
    "openai": {
        "label": "OpenAI ChatGPT",
        "icon": "🟢",
        "model_var": "OPENAI_AGENT_MODEL",
        "default_model": "gpt-4o",
    },
    "deepseek": {
        "label": "DeepSeek",
        "icon": "🔵",
        "model_var": "DEEPSEEK_AGENT_MODEL",
        "default_model": "deepseek-chat",
    },
    "groq": {
        "label": "Groq",
        "icon": "🟣",
        "model_var": "GROQ_AGENT_MODEL",
        "default_model": "llama-3.3-70b-versatile",
    },
}


def _get_model_display(provider: str | None = None) -> str:
    prov = provider or _PROVIDER
    info = _PROVIDERS.get(prov, {})
    model_name = os.getenv(info.get("model_var", ""), info.get("default_model", "unknown"))
    return f"{info.get('icon', '❓')} {info.get('label', prov)} — `{model_name}`"


# ── Session persistence file ──────────────────────────────────────────────────
_SESSION_FILE = os.path.join(os.path.dirname(__file__), "last_session.json")

# ── Continue keywords (Arabic + English) ─────────────────────────────────────
_CONTINUE_KEYWORDS = {
    "أكمل", "استمر", "اكمل", "كمّل", "كمل", "واصل",
    "continue", "resume", "go on", "keep going", "proceed",
    "أكمل من حيث توقفت", "استمر من حيث توقفت",
}

# ── Global graph variable ─────────────────────────────────────────────────────
# Lazy init: compiled on first use inside the event loop so that
# AsyncSqliteSaver can capture the running loop.
GRAPH = None


def _get_graph():
    """Return the compiled graph, creating it on first call."""
    global GRAPH
    if GRAPH is None:
        GRAPH = compile_graph()
    return GRAPH


# ─────────────────────────────────────────────────────────────────────────────
# Session persistence helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_session(thread_id: str, provider: str | None = None) -> None:
    """Save the active thread_id and provider to disk so it survives restarts."""
    try:
        data = {"thread_id": thread_id}
        if provider:
            data["provider"] = provider
        with open(_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _load_last_session() -> dict:
    """Return the last saved session data, or empty dict."""
    try:
        with open(_SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_continue_request(text: str) -> bool:
    """Return True if the user is asking to resume the previous task."""
    t = text.strip().lower()
    for kw in _CONTINUE_KEYWORDS:
        if kw in t:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Streaming helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extract_text_chunk(chunk: AIMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


async def _run_graph(input_or_command, config: dict) -> None:
    """Stream a single graph invocation and display steps in Chainlit."""
    global GRAPH
    if GRAPH is None:
        GRAPH = _get_graph()

    response_msg = cl.Message(content="")
    await response_msg.send()
    active_step: cl.Step | None = None

    NODE_LABELS = {
        "planner":  ("🧠", "يفكر... | Thinking..."),
        "worker":   ("⚡", "ينفذ... | Executing..."),
        "reviewer": ("🔍", "يراجع... | Reviewing..."),
    }

    current_node: str | None = None

    try:
        async for msg_chunk, metadata in GRAPH.astream(
            input_or_command, config=config, stream_mode="messages"
        ):
            node = metadata.get("langgraph_node", "")

            # Show / switch step indicator when the active node changes
            if node and node != current_node and node in NODE_LABELS:
                if active_step:
                    await active_step.__aexit__(None, None, None)
                emoji, label = NODE_LABELS[node]
                active_step = cl.Step(name=f"{emoji} {label}", type="run")
                await active_step.__aenter__()
                current_node = node

            # Stream text tokens
            if hasattr(msg_chunk, "content"):
                text = _extract_text_chunk(msg_chunk)
                if text:
                    await response_msg.stream_token(text)

    except Exception as exc:
        await cl.Message(
            content=f"❌ **خطأ في التنفيذ**: {type(exc).__name__}: {exc}"
        ).send()
    finally:
        if active_step:
            await active_step.__aexit__(None, None, None)
        await response_msg.update()


async def _handle_hitl_loop(config: dict) -> None:
    """Handle Human-in-the-Loop interrupts after each graph run."""
    global GRAPH
    if GRAPH is None:
        GRAPH = _get_graph()

    while True:
        state = await GRAPH.aget_state(config)
        if not state.next:
            break
        all_interrupts: list = []
        for task in state.tasks:
            if hasattr(task, "interrupts"):
                all_interrupts.extend(task.interrupts)
        if not all_interrupts:
            break

        interrupt_obj = all_interrupts[0]
        interrupt_data = getattr(interrupt_obj, "value", {})

        if isinstance(interrupt_data, dict):
            interrupt_type = interrupt_data.get("type", "generic")
            interrupt_message = interrupt_data.get("message", str(interrupt_data))
            risky_command = interrupt_data.get("command", "")
        else:
            interrupt_type = "generic"
            interrupt_message = str(interrupt_data)
            risky_command = ""

        # ── Case A: Destructive command ───────────────────────────────────────
        if interrupt_type == "destructive_command":
            res = await cl.AskActionMessage(
                content=(
                    "⚠️ **مطلوب موافقة المستخدم**\n\n"
                    f"الوكيل يريد تنفيذ أمر **قد يكون خطيراً**:\n\n"
                    f"```powershell\n{risky_command}\n```\n\n"
                    "هل تسمح بتنفيذ هذا الأمر؟"
                ),
                actions=[
                    cl.Action(name="approve", payload={"value": "approve"}, label="✅ موافق"),
                    cl.Action(name="deny", payload={"value": "deny"}, label="❌ رفض"),
                ],
                timeout=300,
            ).send()

            user_choice = (
                res.get("payload", {}).get("value", res.get("value", "deny"))
                if res else "deny"
            )
            status = "✅ تم الموافقة." if user_choice == "approve" else "❌ تم الرفض."
            await cl.Message(content=f"{status} جارٍ الاستئناف…").send()
            await _run_graph(Command(resume=user_choice), config)

        # ── Case B: CAPTCHA ───────────────────────────────────────────────────
        elif interrupt_type == "captcha":
            res = await cl.AskActionMessage(
                content=(
                    "🔒 **تم اكتشاف CAPTCHA**\n\n"
                    f"{interrupt_message}\n\n"
                    "يرجى حل الـ CAPTCHA في نافذة المتصفح، ثم اضغط **تم**."
                ),
                actions=[
                    cl.Action(name="done", payload={"value": "done"}, label="✅ تم"),
                    cl.Action(name="skip", payload={"value": "skip"}, label="⏭️ تخطي"),
                ],
                timeout=600,
            ).send()

            user_choice = (
                res.get("payload", {}).get("value", res.get("value", "done"))
                if res else "done"
            )
            await cl.Message(content="▶️ جارٍ الاستئناف بعد حل الـ CAPTCHA…").send()
            await _run_graph(Command(resume=user_choice), config)

        # ── Case C: Generic ───────────────────────────────────────────────────
        else:
            res = await cl.AskUserMessage(
                content=f"⏸️ **الوكيل متوقف مؤقتاً**\n\n{interrupt_message}\n\nاكتب ردك:",
                timeout=300,
            ).send()
            user_response = res.get("output", "continue") if res else "continue"
            await _run_graph(Command(resume=user_response), config)


# ─────────────────────────────────────────────────────────────────────────────
# Chainlit hooks
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start() -> None:
    """Generate or restore a session thread_id and set up model selector."""
    global GRAPH
    GRAPH = _get_graph()

    last_session = _load_last_session()
    last_thread = last_session.get("thread_id")
    saved_provider = last_session.get("provider", _PROVIDER)

    if last_thread:
        try:
            state = await GRAPH.aget_state({"configurable": {"thread_id": last_thread}})
            has_work = bool(state and state.values.get("messages"))
        except Exception:
            has_work = False
    else:
        has_work = False

    if has_work and last_thread:
        thread_id = last_thread
        resume_note = (
            "\n\n> 🔁 **تم استعادة الجلسة السابقة.** "
            "اكتب **أكمل** أو **continue** لاستئناف مهمتك الأخيرة.\n"
        )
    else:
        thread_id = str(uuid.uuid4())
        resume_note = ""

    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("current_provider", saved_provider)
    _save_session(thread_id, saved_provider)

    # Build available models list for display
    available_models = []
    for key, info in _PROVIDERS.items():
        api_key_var = {
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "groq": "GROQ_API_KEY",
        }.get(key, "")
        has_key = bool(os.getenv(api_key_var, "").strip())
        status = "✅" if has_key else "❌ (مفتاح API غير موجود)"
        available_models.append(f"  {info['icon']} **{info['label']}** — {status}")

    models_display = "\n".join(available_models)

    await cl.Message(
        content=(
            "# 🤖 HAYO AI Agent — وكيل ذكي خارق القدرات\n\n"
            f"**النموذج الحالي**: {_get_model_display(saved_provider)}\n"
            f"**الجلسة**: `{thread_id[:8]}…`{resume_note}\n\n"
            "---\n\n"
            "### القدرات المتاحة:\n"
            "🖥️ **النظام** — PowerShell, CMD, إدارة العمليات والخدمات\n"
            "📁 **الملفات** — قراءة، كتابة، نسخ، نقل، بحث، تحميل\n"
            "🌐 **المتصفح** — Chrome: تصفح، بحث، تحميل ملفات، ملء نماذج\n"
            "🖱️ **سطح المكتب** — فتح تطبيقات، لقطات شاشة، تحكم بالماوس والكيبورد\n"
            "📋 **الحافظة** — نسخ، لصق، إلحاق\n"
            "🌍 **الشبكة** — فحص الاتصال، DNS، Wi-Fi\n"
            "🔊 **الصوت** — تحكم بالمستوى، قراءة نص بصوت عالٍ\n"
            "🔧 **الإصلاح** — إصلاح مشاكل النظام، التسجيل، الخدمات\n\n"
            "---\n\n"
            "### النماذج المتاحة:\n"
            f"{models_display}\n\n"
            "💡 **لتغيير النموذج**: اكتب `/model google` أو `/model anthropic` أو `/model openai` أو `/model deepseek` أو `/model groq`\n\n"
            "---\n\n"
            "**أخبرني بما تريد — سأنفذ كل شيء بدقة.**"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    global GRAPH
    GRAPH = _get_graph()

    thread_id: str = cl.user_session.get("thread_id")
    config: dict = {"configurable": {"thread_id": thread_id}}
    _save_session(thread_id)

    user_text = message.content.strip()

    # ── Model switch command ──────────────────────────────────────────────────
    if user_text.lower().startswith("/model"):
        parts = user_text.split(maxsplit=1)
        if len(parts) == 2:
            new_provider = parts[1].strip().lower()
            if new_provider in _PROVIDERS:
                # Check if API key is available
                key_vars = {
                    "google": "GOOGLE_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "openai": "OPENAI_API_KEY",
                    "deepseek": "DEEPSEEK_API_KEY",
                    "groq": "GROQ_API_KEY",
                }
                api_key = os.getenv(key_vars.get(new_provider, ""), "").strip()
                if not api_key:
                    await cl.Message(
                        content=(
                            f"❌ **مفتاح API غير موجود لـ {_PROVIDERS[new_provider]['label']}**\n\n"
                            f"يرجى إضافة `{key_vars[new_provider]}` في ملف `.env` ثم أعد تشغيل الوكيل."
                        )
                    ).send()
                    return

                # Switch the provider
                from agent.nodes import switch_provider
                try:
                    switch_provider(new_provider)
                    cl.user_session.set("current_provider", new_provider)
                    _save_session(thread_id, new_provider)

                    # Start a new session for the new model
                    new_thread_id = str(uuid.uuid4())
                    cl.user_session.set("thread_id", new_thread_id)
                    _save_session(new_thread_id, new_provider)

                    await cl.Message(
                        content=(
                            f"✅ **تم تغيير النموذج بنجاح!**\n\n"
                            f"النموذج الحالي: {_get_model_display(new_provider)}\n"
                            f"جلسة جديدة: `{new_thread_id[:8]}…`\n\n"
                            "أخبرني بما تريد تنفيذه."
                        )
                    ).send()
                except Exception as exc:
                    await cl.Message(
                        content=f"❌ **خطأ في تغيير النموذج**: {exc}"
                    ).send()
            else:
                available = ", ".join(f"`{k}`" for k in _PROVIDERS.keys())
                await cl.Message(
                    content=f"❌ نموذج غير معروف: `{new_provider}`\n\nالنماذج المتاحة: {available}"
                ).send()
        else:
            current = cl.user_session.get("current_provider", _PROVIDER)
            await cl.Message(
                content=(
                    f"النموذج الحالي: {_get_model_display(current)}\n\n"
                    "للتغيير: `/model google` | `/model anthropic` | `/model openai` | `/model deepseek` | `/model groq`"
                )
            ).send()
        return

    # ── Screenshot command ────────────────────────────────────────────────────
    if user_text.lower() in ("/screenshot", "/لقطة", "لقطة شاشة", "screenshot"):
        await cl.Message(content="📸 جارٍ أخذ لقطة شاشة…").send()
        inputs = {
            "messages": [
                HumanMessage(content="Take a screenshot of the desktop using screen_screenshot() and show me the result.")
            ]
        }
        await _run_graph(inputs, config)
        await _handle_hitl_loop(config)
        return

    # ── Continue / Resume detection ───────────────────────────────────────────
    if _is_continue_request(user_text):
        try:
            state = await GRAPH.aget_state(config)
            if state and state.next:
                await cl.Message(content="▶️ جارٍ الاستئناف من حيث توقفت…").send()
                await _run_graph(Command(resume="continue"), config)
                await _handle_hitl_loop(config)
                return
            elif state and state.values.get("plan") and state.values.get("iteration_count", 0) > 0:  # noqa: E501
                await cl.Message(content="🔁 جارٍ متابعة المهمة السابقة…").send()
                inputs = {
                    "messages": [
                        HumanMessage(
                            content="Continue working on the task. Pick up from the last completed step and keep going until fully done."
                        )
                    ]
                }
                await _run_graph(inputs, config)
                await _handle_hitl_loop(config)
                return
        except Exception:
            pass

    # ── New task: reset state for clean execution ─────────────────────────────
    # When user sends a new task, start a fresh thread to avoid conflicts
    new_thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", new_thread_id)
    config = {"configurable": {"thread_id": new_thread_id}}
    _save_session(new_thread_id, cl.user_session.get("current_provider", _PROVIDER))

    # ── File upload processing ────────────────────────────────────────────────
    file_context = ""
    if message.elements:
        for element in message.elements:
            path = getattr(element, "path", None)
            name = getattr(element, "name", "unknown_file")
            mime = getattr(element, "mime", "")

            if path:
                # For non-text files, just note the path
                if mime and not mime.startswith("text/") and "json" not in mime and "xml" not in mime:
                    file_context += f"\n\n📎 **ملف مرفق: {name}** — المسار: `{path}` (نوع: {mime})"
                else:
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as fh:
                            raw = fh.read(80_000)
                        file_context += f"\n\n---\n📄 **ملف مرفق: {name}**\n```\n{raw}\n```"
                    except Exception as exc:
                        file_context += f"\n\n❌ تعذرت قراءة '{name}': {exc}"

    full_text = user_text + file_context

    # ── Initial graph run ─────────────────────────────────────────────────────
    inputs = {"messages": [HumanMessage(content=full_text)]}
    await _run_graph(inputs, config)
    await _handle_hitl_loop(config)

    # ── Post-run error summary ────────────────────────────────────────────────
    try:
        final_state = await GRAPH.aget_state(config)
        errors = final_state.values.get("error_logs", [])
        messages_list = final_state.values.get("messages", [])
        last_verdict = ""
        for msg in reversed(messages_list):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and ("FAILED:" in content or "TASK_COMPLETE:" in content):
                last_verdict = content[:20]
                break

        if errors and "FAILED:" in last_verdict:
            await cl.Message(
                content=(
                    f"📋 **سجل الأخطاء** — {len(errors)} مشكلة مسجلة:\n"
                    + "\n".join(f"  • {e[:200]}" for e in errors[-5:])
                )
            ).send()
    except Exception:
        pass
