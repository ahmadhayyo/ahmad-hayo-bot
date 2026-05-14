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
import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)  # must run BEFORE agent imports read env vars

import chainlit as cl            # noqa: E402
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage as _SysMsg  # noqa: E402
from langgraph.types import Command  # noqa: E402
from agent.workflow import compile_graph  # noqa: E402
from core.voice_system import (  # noqa: E402
    transcribe,
    text_to_speech,
    stt_available,
    VOICES,
)
from core.conversation_store import (  # noqa: E402
    get_conversation_store,
    derive_title,
    derive_summary,
)

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
# ── Internal tokens that must NEVER appear in user-facing output ──────────────
_INTERNAL_MARKERS = (
    "NEW TASK BOUNDARY",
    "CONVERSATIONAL_ONLY",
    "TASK_COMPLETE:",
    "TASK_COMPLETE.",
    "FAILED:",
    "CONTINUE:",
    "Reviewer:",
    "Worker:",
    "─── NEW TASK",
    "─────────────────────────",
    "━━━━━━━━━━━━━━━━━━━━━━━━━",
    "[Context summary",
    "[Review]",
    "[Duplicate response detected",
    "[Review was rephrased",
    "[Plan was slightly rephrased",
    "Conversational response delivered",
    "evaluate progress against the PLAN",
    "earlier messages remain available",
    "Worker failed to call any tool",
    "SKIPPED: Tool",
    "[INTERNAL",
    "DO NOT SHOW TO USER",
    "[PROGRESS SNAPSHOT]",
    "Plan steps:",
    "⏭️ SKIPPED:",
)


def _filter_internal_tokens(text: str) -> str:
    """Strip internal control tokens from text before showing to the user."""
    if not text:
        return text
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if any(marker in line for marker in _INTERNAL_MARKERS):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    # Collapse multiple blank lines
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result


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
    """Stream a single graph invocation and display steps in Chainlit.

    If voice mode is on for this session, the final assistant text is also
    synthesized to audio and attached to the response message.
    """
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
    final_text_buf: list[str] = []  # collect what becomes the spoken reply

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

            # Skip SystemMessage chunks — they are internal context only
            if isinstance(msg_chunk, _SysMsg):
                continue

            # Stream text tokens — filter internal control markers
            if hasattr(msg_chunk, "content"):
                text = _extract_text_chunk(msg_chunk)
                if text:
                    filtered = _filter_internal_tokens(text)
                    if filtered:
                        await response_msg.stream_token(filtered)
                        # Only buffer reviewer / planner text for TTS — worker
                        # output is mostly tool plumbing we don't want to read aloud.
                        if node in ("planner", "reviewer"):
                            final_text_buf.append(filtered)

    except Exception as exc:
        await cl.Message(
            content=f"❌ **خطأ في التنفيذ**: {type(exc).__name__}: {exc}"
        ).send()
    finally:
        if active_step:
            await active_step.__aexit__(None, None, None)
        await response_msg.update()

    # ── Voice mode: synthesize and attach audio ──────────────────────────────
    if cl.user_session.get("voice_mode") and final_text_buf:
        try:
            voice_pref = cl.user_session.get("voice_name")
            spoken_text = "".join(final_text_buf)
            audio_bytes = await text_to_speech(spoken_text, voice=voice_pref)
            if audio_bytes:
                audio_elem = cl.Audio(
                    name="reply.mp3",
                    content=audio_bytes,
                    auto_play=True,
                    mime="audio/mpeg",
                )
                await cl.Message(content="🔊", elements=[audio_elem]).send()
        except Exception as exc:
            await cl.Message(content=f"⚠️ TTS error: {exc}").send()


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

    # ── Voice defaults ────────────────────────────────────────────────────
    cl.user_session.set("voice_mode", False)
    cl.user_session.set("voice_name", "salma")  # ar-EG-SalmaNeural
    cl.user_session.set("audio_buffer", bytearray())
    cl.user_session.set("audio_mime", "audio/webm")

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

    stt_ok = stt_available()
    voice_status = "✅ جاهز" if stt_ok else "⚠️ غير متاح (يحتاج GROQ_API_KEY أو OPENAI_API_KEY صالح)"

    # ── Build recent-conversations preview ────────────────────────────────────
    history_block = ""
    try:
        store = get_conversation_store()
        recent = [c for c in store.list_recent(limit=5) if c["thread_id"] != thread_id]
        if recent:
            import datetime as _dt
            lines = ["### 📚 محادثاتك السابقة"]
            for c in recent:
                ts = _dt.datetime.fromtimestamp(c["updated_at"]).strftime("%Y-%m-%d %H:%M")
                tid_short = c["thread_id"][:8]
                title = c["title"] or "(بدون عنوان)"
                lines.append(f"  • `{tid_short}` — {title}  _{ts}_")
            lines.append("\nاكتب `/history` للقائمة الكاملة أو `/load <id>` لاستعادة محادثة.")
            history_block = "\n".join(lines) + "\n\n---\n\n"
    except Exception:
        pass

    await cl.Message(
        content=(
            "# 🤖 HAYO AI Agent — وكيل ذكي خارق القدرات\n\n"
            f"**النموذج الحالي**: {_get_model_display(saved_provider)}\n"
            f"**الجلسة**: `{thread_id[:8]}…`{resume_note}\n\n"
            "---\n\n"
            f"{history_block}"
            "### 🎙️ الدردشة الصوتية\n"
            f"الاستماع للصوت (STT): {voice_status}\n"
            "الرد الصوتي (TTS): ✅ جاهز — Edge TTS مجاني\n\n"
            "**الأوامر الصوتية**:\n"
            "  • `/voice on` / `off` — تفعيل/تعطيل الرد الصوتي\n"
            "  • `/voice <اسم>` — تغيير الصوت (salma, shakir, zariyah, hamed, aria, guy)\n"
            "  • اضغط 🎙️ في حقل الإدخال لتسجيل صوتك مباشرة\n\n"
            "**أوامر الذاكرة**:\n"
            "  • `/history` — قائمة المحادثات السابقة\n"
            "  • `/load <id>` — استعادة محادثة (مثال: `/load a1b2c3d4`)\n"
            "  • `/new` — بدء محادثة جديدة في نفس النافذة\n\n"
            "---\n\n"
            "### القدرات:\n"
            "🖥️ النظام · 📁 الملفات · 🌐 المتصفح · 🖱️ سطح المكتب · 📋 الحافظة\n"
            "🌍 الشبكة · 🔊 الصوت · 📊 Office · 🎬 تحويل ملفات · 🎵 تحميل أغاني YouTube\n"
            "🔗 GitHub · 📁 Google Drive · 🖼️ تحليل الصور\n\n"
            "---\n\n"
            "### النماذج المتاحة:\n"
            f"{models_display}\n\n"
            "💡 لتغيير النموذج: `/model google` | `/model anthropic` | `/model deepseek` | `/model groq`\n\n"
            "---\n\n"
            "**أخبرني بما تريد — سأتذكر كل شيء قلته في هذه المحادثة.**"
        )
    ).send()


# ─────────────────────────────────────────────────────────────────────────────
# Voice input handlers — receive audio chunks from the browser microphone
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_audio_start
async def on_audio_start() -> bool:
    """
    Called when the user presses the mic button.

    Return True to accept the audio stream, False to refuse.
    We refuse early if no STT provider is configured — otherwise the user
    speaks, expects something, and gets nothing.
    """
    if not stt_available():
        await cl.Message(
            content=(
                "❌ **لا يمكن استخدام الصوت** — لا يوجد مفتاح API صالح للتعرف على الصوت.\n\n"
                "أضف أحد المفاتيح التالية في `.env` ثم أعد تشغيل HAYO:\n"
                "  • `GROQ_API_KEY` — مجاني وسريع جداً (https://console.groq.com)\n"
                "  • أو `OPENAI_API_KEY` — مدفوع\n\n"
                "(الرد الصوتي TTS يعمل بدون مفاتيح — يمكنك الكتابة وستسمع الرد.)"
            )
        ).send()
        return False

    # Fresh buffer for this recording
    cl.user_session.set("audio_buffer", bytearray())
    cl.user_session.set("audio_mime", "audio/webm")
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk) -> None:
    """Stream raw audio bytes from the browser into our buffer."""
    buf = cl.user_session.get("audio_buffer")
    if buf is None:
        buf = bytearray()
        cl.user_session.set("audio_buffer", buf)
    buf.extend(chunk.data)
    if chunk.mimeType:
        cl.user_session.set("audio_mime", chunk.mimeType)


@cl.on_audio_end
async def on_audio_end() -> None:
    """
    User finished speaking. Transcribe the buffered audio and run it through
    the same flow as a typed message — so voice and text behave identically.

    IMPORTANT: Chainlit calls this with NO arguments. Adding any parameter
    silently breaks the callback (it just never fires).
    """
    buf = cl.user_session.get("audio_buffer")
    mime: str = cl.user_session.get("audio_mime") or "audio/webm"
    # Reset buffer for next recording
    cl.user_session.set("audio_buffer", bytearray())

    audio_bytes = bytes(buf) if buf else b""
    size_kb = len(audio_bytes) / 1024

    if not audio_bytes or size_kb < 0.5:  # less than 500 bytes ≈ no real audio
        await cl.Message(content=f"⚠️ لم يُسجَّل صوت كافٍ ({size_kb:.1f}KB). جرّب الإمساك بزر المايكروفون لفترة أطول.").send()
        return

    # Pick a filename extension matching the recorded MIME (for fallback when
    # the audio is already a container, e.g. webm/ogg). For raw PCM (Chainlit's
    # AudioWorklet) the transcribe() function auto-wraps it in WAV.
    ext = "wav"
    if "ogg" in mime: ext = "ogg"
    elif "webm" in mime: ext = "webm"
    elif "mp4" in mime or "m4a" in mime: ext = "m4a"
    elif "mpeg" in mime or "mp3" in mime: ext = "mp3"

    # Sample rate from .chainlit/config.toml — keep these in sync
    try:
        from chainlit.config import config as _cl_config
        sample_rate = _cl_config.features.audio.sample_rate or 24000
    except Exception:
        sample_rate = 24000

    thinking = cl.Message(content=f"🎙️ سمعتك ({size_kb:.0f}KB). جارٍ التحويل...")
    await thinking.send()

    try:
        transcript = await transcribe(audio_bytes, filename=f"voice.{ext}", sample_rate=sample_rate)
    except Exception as exc:
        thinking.content = (
            f"❌ **فشل التعرف على الصوت**\n\n```\n{exc}\n```\n\n"
            "تأكد من أن المفتاح في `.env` صحيح وفعّال:\n"
            "  • `GROQ_API_KEY` — مجاني من https://console.groq.com\n"
            "  • أو `OPENAI_API_KEY` — بديل مدفوع"
        )
        await thinking.update()
        return

    if not transcript.strip():
        thinking.content = "⚠️ لم أفهم ما قلته. حاول مرة أخرى بصوت أوضح."
        await thinking.update()
        return

    # Replace the "thinking" message with what we actually heard
    thinking.content = f"🎙️ **{transcript}**"
    await thinking.update()

    # Auto-enable voice reply since the user is clearly using voice
    cl.user_session.set("voice_mode", True)

    # Route through the normal text pipeline. @cl.on_message decorator returns
    # the original function unchanged, so calling it directly here works.
    fake = cl.Message(content=transcript, elements=[])
    await on_message(fake)


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

    # ── History / past-conversations commands ─────────────────────────────────
    if user_text.lower().startswith("/history"):
        store = get_conversation_store()
        items = store.list_recent(limit=15)
        if not items:
            await cl.Message(content="📭 لا توجد محادثات سابقة محفوظة بعد.").send()
            return
        lines = ["# 📚 المحادثات السابقة\n"]
        import datetime as _dt
        for i, c in enumerate(items, 1):
            ts = _dt.datetime.fromtimestamp(c["updated_at"]).strftime("%Y-%m-%d %H:%M")
            tid_short = c["thread_id"][:8]
            title = c["title"] or "(بدون عنوان)"
            count = c["message_count"]
            current_marker = " ← الحالية" if c["thread_id"] == thread_id else ""
            lines.append(f"**{i}.** `{tid_short}` — {title}{current_marker}")
            lines.append(f"   📅 {ts}  · 💬 {count} رسالة")
            if c["summary"]:
                summary_short = c["summary"][:150]
                lines.append(f"   📝 {summary_short}")
            lines.append("")
        lines.append("لاستعادة محادثة: `/load <id>` (الأحرف الـ 8 الأولى تكفي)")
        await cl.Message(content="\n".join(lines)).send()
        return

    if user_text.lower().startswith("/load"):
        parts = user_text.split(maxsplit=1)
        if len(parts) != 2:
            await cl.Message(content="استخدم: `/load <thread_id_prefix>`").send()
            return
        prefix = parts[1].strip().lower()
        store = get_conversation_store()
        candidates = [c for c in store.list_recent(limit=200) if c["thread_id"].startswith(prefix)]
        if not candidates:
            await cl.Message(content=f"❌ لم أجد محادثة تبدأ بـ `{prefix}`.").send()
            return
        if len(candidates) > 1:
            await cl.Message(
                content=f"⚠️ يوجد {len(candidates)} محادثات تبدأ بـ `{prefix}`. كن أكثر دقة."
            ).send()
            return
        target = candidates[0]
        cl.user_session.set("thread_id", target["thread_id"])
        _save_session(target["thread_id"], cl.user_session.get("current_provider", _PROVIDER))
        await cl.Message(
            content=(
                f"✅ **تم استعادة المحادثة**\n\n"
                f"العنوان: {target['title'] or '(بدون عنوان)'}\n"
                f"الجلسة: `{target['thread_id'][:8]}…` · {target['message_count']} رسالة\n\n"
                "تابع المحادثة من حيث توقفت — أو اكتب طلباً جديداً."
            )
        ).send()
        return

    if user_text.lower() in ("/new", "/جديد"):
        # Start a brand new conversation explicitly
        new_thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", new_thread_id)
        _save_session(new_thread_id, cl.user_session.get("current_provider", _PROVIDER))
        await cl.Message(
            content=f"🆕 **محادثة جديدة بدأت** — `{new_thread_id[:8]}…`"
        ).send()
        return

    # ── Voice mode command ────────────────────────────────────────────────────
    if user_text.lower().startswith("/voice"):
        parts = user_text.split(maxsplit=1)
        arg = parts[1].strip().lower() if len(parts) == 2 else ""

        if arg in ("on", "تفعيل", "شغل"):
            cl.user_session.set("voice_mode", True)
            current_voice = cl.user_session.get("voice_name", "salma")
            await cl.Message(
                content=f"🔊 **تم تفعيل الرد الصوتي**. الصوت الحالي: `{current_voice}`."
            ).send()
        elif arg in ("off", "تعطيل", "اطفي"):
            cl.user_session.set("voice_mode", False)
            await cl.Message(content="🔇 **تم تعطيل الرد الصوتي**.").send()
        elif arg in VOICES:
            cl.user_session.set("voice_name", arg)
            cl.user_session.set("voice_mode", True)
            await cl.Message(
                content=f"🎙️ **تم تغيير الصوت إلى**: `{arg}` ({VOICES[arg]})\nالرد الصوتي مفعّل الآن."
            ).send()
        elif arg == "":
            mode = "مفعّل ✅" if cl.user_session.get("voice_mode") else "معطّل ❌"
            current = cl.user_session.get("voice_name", "salma")
            voice_list = ", ".join(f"`{k}`" for k in VOICES.keys())
            await cl.Message(
                content=(
                    f"🎙️ **حالة الصوت**: {mode} · الصوت: `{current}`\n\n"
                    "**أوامر**:\n"
                    "  • `/voice on` — تفعيل\n"
                    "  • `/voice off` — تعطيل\n"
                    f"  • `/voice <اسم>` — تغيير الصوت\n\n"
                    f"**الأصوات المتاحة**: {voice_list}"
                )
            ).send()
        else:
            voice_list = ", ".join(f"`{k}`" for k in VOICES.keys())
            await cl.Message(
                content=f"❌ صوت غير معروف: `{arg}`\n\nالأصوات المتاحة: {voice_list}"
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

    # ── Keep the same thread_id across messages ───────────────────────────────
    # PREVIOUSLY we generated a fresh thread_id per message, which made the
    # agent forget everything between messages. Now we reuse the session's
    # thread_id so the LangGraph state (and full conversation history) carries
    # over. Task boundaries are signalled by the cancel_marker in planner_node,
    # which tells the reviewer to evaluate only the new plan.
    # `thread_id` and `config` are already set above from cl.user_session.
    _save_session(thread_id, cl.user_session.get("current_provider", _PROVIDER))

    # ── File upload processing ────────────────────────────────────────────────
    file_context = ""
    image_parts = []  # For multimodal image analysis
    if message.elements:
        for element in message.elements:
            path = getattr(element, "path", None)
            name = getattr(element, "name", "unknown_file")
            mime = getattr(element, "mime", "")

            if path:
                # Image files → send as multimodal content for vision analysis
                if mime and mime.startswith("image/"):
                    try:
                        import base64
                        with open(path, "rb") as img_file:
                            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        })
                        file_context += f"\n\n🖼️ **صورة مرفقة: {name}** — (تم إرسالها للتحليل البصري)"
                    except Exception as exc:
                        file_context += f"\n\n📎 **صورة مرفقة: {name}** — المسار: `{path}` (تعذر التحليل: {exc})"
                # Non-text, non-image files → just note the path
                elif mime and not mime.startswith("text/") and "json" not in mime and "xml" not in mime:
                    file_context += f"\n\n📎 **ملف مرفق: {name}** — المسار: `{path}` (نوع: {mime})"
                else:
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as fh:
                            raw = fh.read(80_000)
                        file_context += f"\n\n---\n📄 **ملف مرفق: {name}**\n```\n{raw}\n```"
                    except Exception as exc:
                        file_context += f"\n\n❌ تعذرت قراءة '{name}': {exc}"

    full_text = user_text + file_context

    # Build message content — multimodal if images are attached
    if image_parts:
        msg_content = [{"type": "text", "text": full_text}] + image_parts
    else:
        msg_content = full_text

    # ── Initial graph run ─────────────────────────────────────────────────────
    inputs = {"messages": [HumanMessage(content=msg_content)]}
    await _run_graph(inputs, config)
    await _handle_hitl_loop(config)

    # ── Post-run error summary + update conversation store ────────────────────
    try:
        final_state = await GRAPH.aget_state(config)
        errors = final_state.values.get("error_logs", [])
        messages_list = final_state.values.get("messages", [])
        last_verdict = ""
        for msg in reversed(messages_list):
            content = getattr(msg, "content", "")
            # Check metadata for original verdict (stripped from display content)
            meta = getattr(msg, "metadata", None) or {}
            original = meta.get("_original_verdict", "")
            check = f"{content} {original}" if isinstance(content, str) else str(original)
            if "FAILED:" in check or "TASK_COMPLETE:" in check:
                last_verdict = check[:40]
                break

        if errors and "FAILED:" in last_verdict:
            await cl.Message(
                content=(
                    f"📋 **سجل الأخطاء** — {len(errors)} مشكلة مسجلة:\n"
                    + "\n".join(f"  • {e[:200]}" for e in errors[-5:])
                )
            ).send()

        # ── Persist conversation summary for cross-session memory ─────────
        try:
            store = get_conversation_store()
            # Title comes from the very first user message of this thread.
            existing = store.get(thread_id)
            title = existing["title"] if (existing and existing["title"]) else derive_title(user_text)
            summary = derive_summary(messages_list)
            store.upsert(
                thread_id=thread_id,
                title=title,
                summary=summary,
                message_count=len(messages_list),
                provider=cl.user_session.get("current_provider", _PROVIDER),
            )
        except Exception:
            pass  # never let storage errors break the user-facing flow
    except Exception:
        pass