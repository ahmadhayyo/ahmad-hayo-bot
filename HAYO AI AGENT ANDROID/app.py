"""
HAYO AI Agent — Android Edition — Chainlit Web UI

Features:
  • Multi-provider support (Google, Anthropic, OpenAI, DeepSeek)
  • /model command for runtime provider switching
  • /screenshot command for device screenshots
  • New thread_id per task for clean isolation
  • Arabic/English bilingual UI
  • File upload support
  • HITL approval for dangerous commands
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

# ── Provider metadata ─────────────────────────────────────────────────────────
_PROVIDERS = {
    "google": {"label": "Google Gemini", "icon": "🟦", "model_var": "GOOGLE_AGENT_MODEL", "default_model": "gemini-2.5-flash"},
    "anthropic": {"label": "Anthropic Claude", "icon": "🟠", "model_var": "ANTHROPIC_AGENT_MODEL", "default_model": "claude-sonnet-4-20250514"},
    "openai": {"label": "OpenAI ChatGPT", "icon": "🟢", "model_var": "OPENAI_AGENT_MODEL", "default_model": "gpt-4o"},
    "deepseek": {"label": "DeepSeek", "icon": "🔵", "model_var": "DEEPSEEK_AGENT_MODEL", "default_model": "deepseek-chat"},
}

_SESSION_FILE = Path(__file__).parent / "last_session.json"


def _get_model_display(provider: str) -> str:
    info = _PROVIDERS.get(provider, {})
    model = os.getenv(info.get("model_var", ""), info.get("default_model", "unknown"))
    return f"{info.get('icon', '🤖')} {info.get('label', provider)} ({model})"


async def _init_graph():
    from agent.workflow import compile_graph
    return compile_graph()


def _save_session(thread_id: str, provider: str | None = None) -> None:
    try:
        data = {"thread_id": thread_id}
        if provider:
            data["provider"] = provider
        with open(_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _load_last_session() -> tuple[str | None, str | None]:
    try:
        with open(_SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("thread_id"), data.get("provider")
    except Exception:
        return None, None


def _is_continue_request(text: str) -> bool:
    lower = text.strip().lower()
    return lower in ("أكمل", "continue", "تابع", "استمر", "واصل", "كمل")


def _extract_text_chunk(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


# ─────────────────────────────────────────────────────────────────────────────
# Chainlit handlers
# ─────────────────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    graph = await _init_graph()
    cl.user_session.set("graph", graph)

    old_tid, saved_provider = _load_last_session()
    thread_id = old_tid or str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)

    provider = saved_provider or os.getenv("MODEL_PROVIDER", "google").lower()
    cl.user_session.set("provider", provider)

    # Build startup message
    lines = [
        "# 🤖 HAYO AI Agent — Android Edition",
        "",
        f"**النموذج الحالي:** {_get_model_display(provider)}",
        "",
        "### النماذج المتاحة:",
    ]
    for key, info in _PROVIDERS.items():
        key_var = info.get("model_var", "").replace("_AGENT_MODEL", "_API_KEY")
        has_key = bool(os.getenv(key_var))
        status = "✅" if has_key else "❌ (لا يوجد مفتاح)"
        lines.append(f"  {info['icon']} **{info['label']}** — {status}")

    lines += [
        "",
        "### الأوامر الخاصة:",
        "  `/model google` — تغيير النموذج",
        "  `/screenshot` — لقطة شاشة للموبايل",
        "  `أكمل` — استئناف مهمة سابقة",
        "",
        "---",
        "💬 أرسل طلبك وسأنفذه فوراً!",
    ]

    await cl.Message(content="\n".join(lines)).send()


@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or "").strip()
    graph = cl.user_session.get("graph")
    provider = cl.user_session.get("provider", "google")

    # ── /model command ────────────────────────────────────────────────────────
    if text.lower().startswith("/model"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            available = "\n".join(f"  `{k}` — {v['label']}" for k, v in _PROVIDERS.items())
            await cl.Message(content=f"اكتب اسم النموذج:\n{available}").send()
            return
        new_prov = parts[1].strip().lower()
        if new_prov not in _PROVIDERS:
            await cl.Message(content=f"❌ نموذج غير معروف: `{new_prov}`").send()
            return
        key_var = _PROVIDERS[new_prov]["model_var"].replace("_AGENT_MODEL", "_API_KEY")
        if not os.getenv(key_var):
            await cl.Message(content=f"❌ مفتاح `{key_var}` غير موجود في `.env`").send()
            return
        from agent.nodes import switch_provider
        switch_provider(new_prov)
        cl.user_session.set("provider", new_prov)
        new_thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", new_thread_id)
        _save_session(new_thread_id, new_prov)
        await cl.Message(content=f"✅ تم التبديل إلى {_get_model_display(new_prov)}").send()
        return

    # ── /screenshot command ───────────────────────────────────────────────────
    if text.lower() in ("/screenshot", "لقطة شاشة", "سكرين شوت"):
        from tools.screen_tools import screen_screenshot
        result = screen_screenshot.invoke({"save_path": "/sdcard/screenshot.png"})
        await cl.Message(content=f"📸 {result}").send()
        return

    # ── Continue request ──────────────────────────────────────────────────────
    is_continue = _is_continue_request(text)
    if not is_continue:
        new_thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", new_thread_id)
        _save_session(new_thread_id, provider)

    thread_id = cl.user_session.get("thread_id")

    # ── Handle file attachments ───────────────────────────────────────────────
    file_context = ""
    if message.elements:
        for el in message.elements:
            try:
                if hasattr(el, "path") and el.path:
                    p = Path(el.path)
                    if p.stat().st_size < 5_000_000:
                        try:
                            content = p.read_text(encoding="utf-8", errors="replace")
                            file_context += f"\n\n📎 ملف: {el.name}\n```\n{content[:10000]}\n```"
                        except Exception:
                            file_context += f"\n\n📎 ملف ثنائي: {el.name} ({p.stat().st_size} bytes) at {p}"
            except Exception:
                pass

    full_text = text + file_context

    # ── Run the graph ─────────────────────────────────────────────────────────
    thinking_msg = cl.Message(content="🧠 جاري التفكير...")
    await thinking_msg.send()

    await _run_graph(graph, full_text, thread_id, thinking_msg)


async def _run_graph(graph, user_text: str, thread_id: str, thinking_msg):
    """Stream graph execution and handle HITL interrupts."""
    config = {"configurable": {"thread_id": thread_id}}
    input_payload = {"messages": [HumanMessage(content=user_text)]}

    final_text = ""
    try:
        async for event in graph.astream_events(input_payload, config=config, version="v2"):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    text = _extract_text_chunk(chunk.content)
                    if text:
                        final_text += text
                        thinking_msg.content = final_text
                        await thinking_msg.update()

            elif kind == "on_tool_start":
                tool_name = event.get("name", "?")
                await cl.Message(content=f"⚙️ تنفيذ: `{tool_name}`...").send()

            elif kind == "on_tool_end":
                tool_output = event.get("data", {}).get("output", "")
                if isinstance(tool_output, str) and len(tool_output) > 500:
                    tool_output = tool_output[:500] + "..."
                await cl.Message(content=f"📋 النتيجة:\n```\n{tool_output}\n```").send()

    except Exception as e:
        err_str = str(e)
        if "interrupt" in err_str.lower() or "NodeInterrupt" in err_str:
            await _handle_hitl_loop(graph, thread_id)
        else:
            await cl.Message(content=f"❌ خطأ: {err_str[:500]}").send()

    if not final_text:
        thinking_msg.content = "✅ تم التنفيذ"
        await thinking_msg.update()


async def _handle_hitl_loop(graph, thread_id: str):
    """Handle Human-in-the-Loop interrupt from the graph."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await graph.aget_state(config)
        if not state or not state.tasks:
            return

        for task in state.tasks:
            payload = task.interrupts[0].value if task.interrupts else {}
            interrupt_type = payload.get("type", "unknown")
            msg_text = payload.get("message", "⚠️ يحتاج موافقتك")
            command = payload.get("command", "")

            if interrupt_type == "destructive_command":
                res = await cl.AskActionMessage(
                    content=f"{msg_text}",
                    actions=[
                        cl.Action(name="approve", payload={"value": "approve"}, label="✅ سماح"),
                        cl.Action(name="deny", payload={"value": "deny"}, label="❌ رفض"),
                    ],
                ).send()

                choice = "deny"
                if res and hasattr(res, "payload"):
                    choice = res.payload.get("value", "deny")
                elif res and hasattr(res, "name"):
                    choice = res.name

            elif interrupt_type == "captcha":
                await cl.AskActionMessage(
                    content=msg_text,
                    actions=[cl.Action(name="done", payload={"value": "done"}, label="✅ تم الحل")],
                ).send()
                choice = "done"
            else:
                choice = "approve"

            resume_msg = cl.Message(content="🔄 جاري الاستئناف...")
            await resume_msg.send()

            try:
                async for event in graph.astream_events(
                    Command(resume=choice), config=config, version="v2"
                ):
                    kind = event.get("event", "")
                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content"):
                            text = _extract_text_chunk(chunk.content)
                            if text:
                                resume_msg.content += text
                                await resume_msg.update()
            except Exception as inner_e:
                if "interrupt" in str(inner_e).lower():
                    await _handle_hitl_loop(graph, thread_id)

    except Exception as e:
        await cl.Message(content=f"❌ HITL error: {str(e)[:300]}").send()
