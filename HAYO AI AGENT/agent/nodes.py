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
WorkerNode intercepts special sentinel strings returned by tools:

  "__HITL_REQUIRED__" / "HITL_APPROVAL_REQUIRED:" → destructive OS command detected
  "CAPTCHA_DETECTED"                              → CAPTCHA/anti-bot wall detected

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
Set MODEL_PROVIDER=openai    in .env to use OpenAI ChatGPT.
Set MODEL_PROVIDER=deepseek  in .env to use DeepSeek.
Set MODEL_PROVIDER=groq      in .env to use Groq.

Provider can also be switched at runtime via the UI model selector.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import interrupt

from core.state import AgentState
from core.safety import needs_human_approval
from core.deduplication import is_duplicate_tool_call, is_duplicate_message, record_tool_call
from tools.registry import ALL_TOOLS, TOOLS_BY_NAME

# ── Environment ───────────────────────────────────────────────────────────────
MAX_HISTORY:    int = int(os.getenv("MAX_HISTORY",    "300"))  # Increased from 15 for longer conversations
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "500"))  # Increased from 50 to allow unlimited execution
PS_TIMEOUT:     int = int(os.getenv("PS_TIMEOUT",     "30"))

_PROVIDER = os.getenv("MODEL_PROVIDER", "google").lower().strip()


# ── LLM Factory ───────────────────────────────────────────────────────────────

def _build_llm(role: Literal["main", "summarizer"], provider: str | None = None) -> BaseChatModel:
    """Return the correct LangChain chat model based on provider.
    
    If provider is None, uses _PROVIDER (from .env MODEL_PROVIDER).
    This allows runtime model switching from the UI.
    """
    prov = (provider or _PROVIDER).lower().strip()

    if prov == "google":
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
            model_name = os.getenv("GOOGLE_SUMMARIZER_MODEL", "gemini-2.0-flash")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.0,
                max_output_tokens=2_048,
            )

    elif prov == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if role == "main":
            model_name = os.getenv("ANTHROPIC_AGENT_MODEL", "claude-sonnet-4-20250514")
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

    elif prov == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY") or "sk-placeholder"
        if role == "main":
            model_name = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.0,
                streaming=True,
            )
        else:
            model_name = os.getenv("OPENAI_SUMMARIZER_MODEL", "gpt-4o-mini")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.0,
                max_tokens=2_048,
            )

    elif prov == "deepseek":
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-placeholder"
        if role == "main":
            model_name = os.getenv("DEEPSEEK_AGENT_MODEL", "deepseek-chat")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                streaming=True,
            )
        else:
            model_name = os.getenv("DEEPSEEK_SUMMARIZER_MODEL", "deepseek-chat")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                max_tokens=2_048,
            )

    elif prov == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY") or ""
        if role == "main":
            model_name = os.getenv("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=0.0,
                streaming=True,
            )
        else:
            model_name = os.getenv("GROQ_SUMMARIZER_MODEL", "llama-3.1-8b-instant")
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=0.0,
                max_tokens=2_048,
            )

    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER='{prov}'. "
            "Set MODEL_PROVIDER to 'google', 'anthropic', 'openai', 'deepseek', or 'groq'."
        )


def switch_provider(provider: str) -> None:
    """Switch the LLM provider at runtime (called from Chainlit UI)."""
    global _main_llm, _fast_llm, _llm_with_tools, _PROVIDER
    _PROVIDER = provider.lower().strip()
    _main_llm = _build_llm("main", _PROVIDER)
    _fast_llm = _build_llm("summarizer", _PROVIDER)
    _llm_with_tools = _main_llm.bind_tools(ALL_TOOLS)


def _ensure_provider_match() -> None:
    """Rebuild LLM if provider changed in environment."""
    global _main_llm, _fast_llm, _llm_with_tools, _PROVIDER
    current_provider = os.getenv("MODEL_PROVIDER", "google").lower().strip()
    if current_provider != _PROVIDER:
        _PROVIDER = current_provider
        _main_llm = _build_llm("main", _PROVIDER)
        _fast_llm = _build_llm("summarizer", _PROVIDER)
        _llm_with_tools = _main_llm.bind_tools(ALL_TOOLS)


# ── LLM instances (built once at import time) ─────────────────────────────────
_main_llm = _build_llm("main")
_fast_llm = _build_llm("summarizer")

# ── Tool registry (unified from tools/registry.py) ───────────────────────────
TOOL_MAP: dict = TOOLS_BY_NAME

_llm_with_tools = _main_llm.bind_tools(ALL_TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tool_call_id(tc) -> str | None:
    """Extract a tool_call id whether the entry is a dict or an object."""
    if isinstance(tc, dict):
        return tc.get("id")
    return getattr(tc, "id", None)


def _sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Enforce strict OpenAI-compatible tool message sequencing.

    OpenAI-style APIs (DeepSeek, OpenAI, etc.) raise 400 if:
      A) A ToolMessage has no preceding AIMessage with a matching tool_call_id
      B) An AIMessage has tool_calls whose responses never appear

    This sanitizer fixes both:
      1. Drops orphan ToolMessages (case A)
      2. Re-anchors each ToolMessage right after its declaring AIMessage
      3. Synthesizes placeholder ToolMessages for any unanswered tool_calls (case B)

    The result is a message sequence where every AIMessage(tool_calls=[a,b,c])
    is immediately followed by ToolMessage(a), ToolMessage(b), ToolMessage(c)
    in any order. No orphans, no missing responses.
    """
    if not messages:
        return messages

    # Index every ToolMessage by its tool_call_id so we can re-anchor them later
    tool_responses: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_responses[msg.tool_call_id] = msg

    result: list[BaseMessage] = []
    used_response_ids: set[str] = set()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Strip from natural position — we'll re-insert after the matching AIMessage
            continue

        result.append(msg)

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tid = _tool_call_id(tc)
                if not tid or tid in used_response_ids:
                    continue
                used_response_ids.add(tid)
                if tid in tool_responses:
                    result.append(tool_responses[tid])
                else:
                    # Placeholder for missing response — prevents API 400
                    result.append(ToolMessage(
                        content="[Tool result missing — execution likely failed]",
                        tool_call_id=tid,
                    ))

    return result


def _summarize_old_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    If message history exceeds MAX_HISTORY, summarise the oldest entries with
    the fast LLM and replace them with a single AIMessage containing the summary.

    Strategy:
    - Keep last 20 messages (not 10) for richer context
    - Only summarize up to the 30 messages before those (avoid huge summarization)
    - Filter out previous summaries to avoid redundancy
    - Always sanitize the result to remove orphaned ToolMessages
    """
    if len(messages) <= MAX_HISTORY:
        return _sanitize_messages(messages)

    keep_recent  = min(20, len(messages) // 2)  # Keep more recent messages
    old_messages = messages[:-keep_recent]
    recent       = messages[-keep_recent:]

    # Only take the last 30 of the old messages for summarization (avoid context explosion)
    messages_to_summarize = old_messages[-30:] if len(old_messages) > 30 else old_messages

    # Filter out any existing context summaries to avoid nesting
    messages_to_summarize = [
        m for m in messages_to_summarize
        if not (isinstance(m, AIMessage) and "Context summary" in str(m.content))
    ]

    # Sanitize messages_to_summarize too before sending to LLM
    messages_to_summarize = _sanitize_messages(messages_to_summarize)

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
            *messages_to_summarize,
        ]
    )

    summary_msg = AIMessage(
        content=f"[Context summary — earlier steps]\n\n{summary_response.content}"
    )

    # Return: filtered old messages (excluding the ones we summarized) + summary + recent
    remaining_old = old_messages[:-len(messages_to_summarize)] if messages_to_summarize else []
    combined = remaining_old + [summary_msg] + recent

    # Always sanitize to drop any orphaned ToolMessages that lost their AIMessage pair
    return _sanitize_messages(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — PlannerNode
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """أنت وكيل تنفيذي ذكي خارق القدرات يعمل على نظام Windows 64-bit.
هذا بيئة محلية موثوقة بالكامل. المستخدم يملك هذا الجهاز بالكامل.
تعمل بصلاحيات كاملة — كما لو كنت أنت المستخدم نفسه.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ قواعد ذكية تحسّن الأداء والكفاءة
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ تجنب تكرار استدعاءات الأدوات:
   • لا تستدعِ نفس الأداة بنفس المعاملات مرتين متتاليتين
   • إذا كنت تريد بيانات مختلفة، استخدم معاملات مختلفة
   • مثال: إذا استدعيت browser_click مرة، لا تستدعِها بنفس الـ selector مرة أخرى

🎵 تحميل أغاني / فيديو / صوت — استخدم yt-dlp مباشرة (لا تستخدم Google!):
   ⭐ download_audio_by_search(query, dest='desktop:'): يبحث في YouTube ويحمّل
      أفضل جودة صوت. مثال:
        • "حمّل أغنية بعيش لتامر حسني" → download_audio_by_search(query='بعيش تامر حسني', dest='desktop:')
        • "نزّل Hotel California" → download_audio_by_search(query='Hotel California Eagles', dest='desktop:')
   ⭐ download_audio_from_url(url, dest=): تحميل صوت من URL محدد (YouTube link)
   ⭐ download_video_from_url(url, dest=): تحميل فيديو كامل من URL

   ❌ لا تبحث في Google ثم تحاول النقر على مواقع mp3 — هذا سيفشل دائماً
   ❌ لا تستخدم browser_open + browser_fill لتحميل أغنية — استخدم yt-dlp مباشرة

✅ أدوات التحميل العامة (لروابط ملفات مباشرة فقط):
   • download_file(): تحميل ملف من URL مباشر معروف
   • download_with_progress(): تحميل مع إعادة محاولة تلقائية وتتبع السرعة
   • check_url_availability(): التحقق من URL قبل التحميل

✅ أدوات المتصفح (للتفاعل مع صفحات الويب):
   • browser_open(): فتح URL
   • browser_fill(): ملء حقل إدخال (مواقع عادية)
   • browser_react_fill(): ملء حقل في React/Vue/Angular SPA (Replit, GitHub, Google)
   • browser_click(): نقر على زر/رابط
   • browser_press(key='Enter'): إرسال نموذج بـ Enter
   • browser_get_text(): قراءة محتوى صفحة
   ⚠️ لا تستخدم browser_eval_js للنقر أو الكتابة — استخدم browser_click/browser_press

✅ أدوات تحويل الملفات (قوية وسريعة):
   • convert_file(): تحويل بين صيغ (mp3↔wav, pdf↔docx, png↔jpg, إلخ)
   • get_supported_formats(): عرض الصيغ المدعومة والمتطلبات
   • check_conversion_support(): التحقق من إمكانية تحويل معين

✅ إدارة الذاكرة الذكية:
   • الوكيل يحتفظ بآخر 300 رسالة فقط (يحذف القديمة الزائدة تلقائياً)
   • الرسائل القديمة تُلخّص تلقائياً للحفاظ على السياق
   • لا تقلق بشأن مدة الجلسة — الذاكرة مُدارة بكفاءة

═══════════════════════════════════════════════
الأدوات المتاحة لك (جميعها تعمل فعلياً):
═══════════════════════════════════════════════

🖥️ النظام والأوامر:
  • run_powershell     — تنفيذ أي أمر PowerShell
  • run_cmd            — تنفيذ أوامر CMD الكلاسيكية
  • get_env            — قراءة متغيرات البيئة
  • get_system_info    — معلومات النظام (OS, CPU, RAM, Disk)
  • list_processes     — عرض العمليات الجارية مرتبة حسب CPU/الذاكرة
  • kill_process       — إيقاف عملية بالاسم أو PID
  • manage_service     — إدارة خدمات Windows (start/stop/restart/status)
  • scheduled_task     — إدارة المهام المجدولة

📁 نظام الملفات:
  • read_file          — قراءة أي ملف نصي
  • write_file         — إنشاء أو كتابة ملف
  • append_file        — إلحاق نص بملف موجود
  • list_dir           — عرض محتويات مجلد
  • search_files       — بحث glob في المجلدات
  • move_file          — نقل أو إعادة تسمية ملف/مجلد
  • copy_file          — نسخ ملف أو مجلد
  • download_file      — تحميل ملف من URL إلى القرص
  • make_dir           — إنشاء مجلد جديد

📋 الحافظة:
  • clipboard_get      — قراءة محتوى الحافظة
  • clipboard_set      — تعيين نص في الحافظة
  • clipboard_append   — إلحاق نص بالحافظة

🚀 التطبيقات:
  • open_app           — فتح أي تطبيق (chrome, word, vscode, ...)
  • close_app          — إغلاق تطبيق
  • list_running_apps  — عرض التطبيقات المفتوحة
  • focus_window       — جلب نافذة للأمام

🌐 المتصفح (Playwright — جلسة دائمة مع ملفات تعريف الارتباط):
  • browser_open        — فتح URL
  • browser_get_text    — قراءة نص من الصفحة
  • browser_click       — النقر على عنصر/زر (استخدم هذا لإرسال النماذج!)
  • browser_fill        — ملء حقل إدخال (مواقع عادية)
  • browser_react_fill  — ملء حقل في React/Vue/Angular SPA (Replit, GitHub, Google...)
  • browser_press       — ضغط مفتاح (Enter لإرسال النموذج)
  • browser_screenshot  — لقطة شاشة للمتصفح (مرة واحدة فقط لكل خطوة!)
  • browser_download_via_click — تحميل ملف بالنقر على زر تحميل
  • browser_download_to_desktop — تحميل ملف من URL مباشر عبر المتصفح (يتعامل مع الجلسات والكوكيز)
  • browser_eval_js     — تنفيذ JavaScript (للقراءة فقط — لا تستخدمه للنقر أو الكتابة!)
  • browser_wait_for    — انتظار ظهور عنصر
  • browser_new_tab     — فتح تبويب جديد
  • browser_switch_tab  — التبديل بين التبويبات
  • browser_list_tabs   — عرض كل التبويبات المفتوحة
  • browser_close_tab   — إغلاق تبويب
  • browser_login       — تسجيل دخول تلقائي لأي موقع (URL + اسم مستخدم + كلمة سر)
  • browser_get_cookies — عرض ملفات تعريف الارتباط (لتصحيح مشاكل تسجيل الدخول)

🖱️ التحكم بسطح المكتب (pyautogui):
  • screen_screenshot  — لقطة شاشة لسطح المكتب بالكامل
  • screen_size        — أبعاد الشاشة
  • mouse_click        — نقر في إحداثيات محددة
  • mouse_move         — تحريك المؤشر
  • mouse_scroll       — تمرير
  • keyboard_type      — كتابة نص
  • keyboard_hotkey    — اختصارات لوحة المفاتيح (ctrl+s, alt+f4, ...)
  • list_windows       — عرض النوافذ المفتوحة
  • wait               — انتظار (ثوانٍ)

🌍 الشبكة:
  • get_network_info   — معلومات الشبكة (IP, DNS, Gateway)
  • get_public_ip      — عنوان IP العام
  • ping_host          — اختبار الاتصال بمضيف
  • check_port         — فحص منفذ TCP
  • wifi_management    — إدارة Wi-Fi
  • dns_lookup         — استعلام DNS

🔊 الصوت والإشعارات:
  • volume_control     — التحكم بمستوى الصوت
  • text_to_speech     — قراءة نص بصوت عالٍ
  • show_notification  — إظهار إشعار Windows
  • play_sound         — تشغيل صوت

📊 المستندات المكتبية (Excel, Word, PDF):
  • excel_create       — إنشاء ملف Excel جديد من بيانات JSON
  • excel_read         — قراءة محتوى ملف Excel
  • excel_edit         — تعديل خلية في ملف Excel
  • excel_add_rows     — إضافة صفوف جديدة لملف Excel
  • excel_add_column   — إضافة عمود جديد (صيغة أو قيم)
  • word_create        — إنشاء ملف Word جديد
  • word_read          — قراءة محتوى ملف Word
  • word_edit          — البحث والاستبدال في ملف Word
  • pdf_read           — قراءة واستخراج نص من PDF
  • pdf_create         — إنشاء ملف PDF من نص
  • pdf_merge          — دمج عدة ملفات PDF
  • convert_excel_to_pdf — تحويل Excel إلى PDF
  • convert_word_to_pdf  — تحويل Word إلى PDF

🔗 GitHub (إدارة المستودعات والمشاريع):
  • github_clone        — استنساخ مستودع من GitHub
  • github_status       — حالة المستودع المحلي (الفرع، الملفات المعدلة، آخر commits)
  • github_commit_push  — حفظ التعديلات ورفعها إلى GitHub (stage + commit + push)
  • github_pull         — سحب آخر التحديثات من المستودع البعيد
  • github_create_repo  — إنشاء مستودع جديد على GitHub
  • github_branch       — إدارة الفروع (list, create, switch, delete)

📁 Google Drive (رفع وتحميل وعرض الملفات):
  • gdrive_list         — عرض الملفات في Google Drive
  • gdrive_download     — تحميل ملف من Google Drive
  • gdrive_upload       — رفع ملف إلى Google Drive

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الخطوة 1 — تصنيف الطلب
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
هل هذا تحية أو محادثة عادية أو سؤال يمكن الإجابة عليه بدون أداة؟
  نعم → أجب بشكل ودود وطبيعي بنفس لغة المستخدم (كأنك صديق يساعده، ليس روبوت رسمي)،
        ردك سيُقرأ صوتياً أحياناً، فاجعله نبرة محادثة دافئة، بدون markdown أو رموز،
        ثم اكتب: CONVERSATIONAL_ONLY
  لا  → أنشئ خطة تنفيذ دقيقة (انظر أدناه)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الخطوة 2 — خطة التنفيذ (لجميع المهام الحقيقية)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
اكتب قائمة مرقمة قصيرة من الخطوات المحددة مع ذكر الأداة في كل خطوة.

مثال (فتح تطبيق والعمل فيه):
  1. open_app('word') → فتح Microsoft Word
  2. wait(seconds=3) → انتظار التحميل
  3. screen_screenshot → رؤية الشاشة
  4. focus_window('Word') → جلب النافذة للأمام
  5. keyboard_type('نص الرسالة') → كتابة النص
  6. keyboard_hotkey('ctrl,s') → حفظ

مثال (تحميل ملف من الإنترنت):
  1. browser_open(url='https://example.com/file.pdf')
  2. download_file(url='...', dest='desktop:file.pdf')
  3. run_powershell → Test-Path للتحقق

مثال (إصلاح مشكلة في النظام):
  1. get_system_info → فهم حالة النظام
  2. list_processes(sort_by='memory') → فحص العمليات
  3. run_powershell → تنفيذ أوامر الإصلاح
  4. run_powershell → التحقق من النتيجة

مثال (إنشاء جدول Excel بالبيانات):
  1. excel_create(path='Desktop/report.xlsx', data='[{"الاسم":"أحمد","الراتب":5000}]')
  2. excel_add_column(path='...', header='الضريبة', formula_or_values='=B{row}*0.1')
  3. excel_read(path='...') → عرض النتيجة

مثال (قراءة PDF واستخراج بيانات إلى Excel):
  1. pdf_read(path='report.pdf') → استخراج النص
  2. excel_create(path='data.xlsx', data='...') → إنشاء جدول بالبيانات المستخرجة

القواعد:
• جمل مرقمة فقط — بدون JSON أو كود في الخطة نفسها.
• سمِّ الأداة في كل خطوة.
• اجعل الخطط مختصرة (3-8 خطوات لمعظم المهام).
• لا تقل أبداً "لا أستطيع" — دائماً خطط لمسار أمامي.
• إذا فشل شيء سابقاً، خطط لنهج مختلف هذه المرة.
• أجب دائماً بنفس لغة المستخدم (عربي أو إنجليزي)."""

def planner_node(state: AgentState) -> dict:
    """
    Analyses the user's latest request and produces a numbered execution plan.

    Special case — CONVERSATIONAL_ONLY:
      If the message is a greeting, casual question, or anything that does NOT
      require any tool, the planner responds directly and sets plan = ["CONVERSATIONAL_ONLY"].
      WorkerNode skips tool execution and ReviewerNode immediately marks TASK_COMPLETE.
    """
    _ensure_provider_match()  # Ensure correct LLM provider is being used
    messages = _summarize_old_messages(state.get("messages", []))
    system   = SystemMessage(content=_PLANNER_SYSTEM)
    response = _main_llm.invoke([system] + messages)
    content  = response.content if isinstance(response.content, str) else ""

    # ── Detect conversational response ────────────────────────────────────────
    if "CONVERSATIONAL_ONLY" in content:
        clean_content  = content.replace("CONVERSATIONAL_ONLY", "").strip()
        clean_response = AIMessage(content=clean_content)

        # ── Check for duplicate message ──────────────────────────────────────
        if is_duplicate_message(clean_response, messages, min_length=50):
            clean_response = AIMessage(
                content=(
                    clean_content + "\n\n"
                    "⚠️ [Duplicate response detected — message was slightly rephrased to avoid exact duplication]"
                )
            )
        return {
            "messages":                messages + [clean_response],
            "plan":                    ["CONVERSATIONAL_ONLY"],
            "iteration_count":         0,   # ← RESET: every new user task starts fresh
            "completed_steps":         [],  # ← RESET
            "error_logs":              [],  # ← RESET
            "workspace":               state.get("workspace", ""),
            "requires_human_approval": False,
            "pending_command":         "",
            "tool_call_history":       [],  # ← RESET
            "last_tool_name":          "",  # ← RESET
            "last_tool_args":          {},  # ← RESET
            "last_message_content":    "",  # ← RESET
            "task_id":                 str(uuid.uuid4()),  # ← NEW
        }

    # ── Real task: extract numbered steps as the plan list ────────────────────
    plan_lines = [
        ln.strip()
        for ln in content.splitlines()
        if ln.strip() and (ln.strip()[0].isdigit() or ln.strip().startswith("•"))
    ]

    # ── Check for duplicate plan response ────────────────────────────────────
    if is_duplicate_message(response, messages, min_length=50):
        content = (
            content + "\n\n"
            "⚠️ [Plan was slightly rephrased to avoid exact duplication with previous response]"
        )
        response = AIMessage(content=content)

    # Insert a soft task-boundary marker. The Reviewer should focus on the
    # CURRENT plan, but the agent must STILL be able to read earlier messages
    # if the new request refers back to them ("download the file we discussed",
    # "النسخة الثانية من ذاك التقرير", etc.).
    task_id = str(uuid.uuid4())
    cancel_marker = AIMessage(
        content=(
            "─── NEW TASK BOUNDARY ───\n"
            "Reviewer: evaluate progress against the PLAN BELOW, not any older plan.\n"
            "Worker: earlier messages remain available as conversational context — "
            "consult them when the user's request refers back to prior tasks "
            "(e.g. 'the file we downloaded', 'continue from where you left off').\n"
            "─────────────────────────"
        ),
        metadata={
            "type": "task_cancel",
            "task_id": task_id,
            "timestamp": __import__("time").time(),
        }
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
        "tool_call_history":       [],  # ← RESET: track last 20 tool calls
        "last_tool_name":          "",  # ← RESET
        "last_tool_args":          {},  # ← RESET
        "last_message_content":    "",  # ← RESET
        "task_id":                 task_id,  # ← NEW: unique task identifier
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
    _ensure_provider_match()  # Ensure correct LLM provider is being used
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
            "tool_call_history":       state.get("tool_call_history", []),
            "last_tool_name":          state.get("last_tool_name", ""),
            "last_tool_args":          state.get("last_tool_args", {}),
            "last_message_content":    state.get("last_message_content", ""),
            "task_id":                 state.get("task_id", str(uuid.uuid4())),
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
            "tool_call_history":       state.get("tool_call_history", []),
            "last_tool_name":          state.get("last_tool_name", ""),
            "last_tool_args":          state.get("last_tool_args", {}),
            "last_message_content":    state.get("last_message_content", ""),
            "task_id":                 state.get("task_id", str(uuid.uuid4())),
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
            "أنت وكيل تنفيذي ذكي خارق القدرات يعمل على جهاز Windows 64-bit موثوق بالكامل.\n"
            "تعمل كما لو كنت مالك هذا الجهاز — بصلاحيات كاملة، بدون تردد.\n\n"
            f"خطة المهمة ({steps_total} خطوات):\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan)) +
            f"\n\nتم إنجازه حتى الآن:\n{progress_summary}\n\n"
            f"إجراؤك التالي (الخطوة {steps_done + 1}): {next_step_hint}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "قواعد مطلقة — لا تكسرها أبداً:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "1. يجب أن تستدعي أداة واحدة على الأقل. استجابة بدون أداة = فشل.\n"
            "2. لا تقل أبداً أنك لا تستطيع. ابحث عن طريقة. دائماً.\n"
            "3. لا تطلب إذن المستخدم أو توضيح. فقط نفذ.\n"
            "4. لا تشرح ما ستفعله — فقط استدعِ الأداة.\n"
            "5. لا تكرر نفس الأداة بنفس المعاملات. إذا فشلت — جرب نهجاً مختلفاً تماماً.\n\n"
            "🎵 تحميل أغاني/فيديو — استخدم yt-dlp مباشرة (الطريق الأقصر والأنجح):\n"
            "   مثال: 'حمّل أغنية بعيش لتامر حسني' →\n"
            "     download_audio_by_search(query='بعيش تامر حسني', dest='desktop:')\n"
            "   مثال: 'حمّل فيديو من YouTube link' →\n"
            "     download_video_from_url(url='https://youtube.com/...', dest='desktop:')\n"
            "   ❌ لا تبحث في Google لتحميل أغاني — yt-dlp أفضل بكثير!\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🌐 قواعد المتصفح — تسجيل الدخول وإرسال النماذج:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "✅ تسجيل الدخول في أي موقع (Gmail، Replit، GitHub، إلخ):\n"
            "   1. browser_open(url='https://site.com/login')\n"
            "   2. browser_fill(selector='input[name=\"email\"]', value='user@example.com')\n"
            "   3. browser_fill(selector='input[name=\"password\"]', value='mypassword')\n"
            "   4. browser_press(key='Enter')  ← أو: browser_click(selector='button[type=\"submit\"]')\n"
            "   5. browser_wait_for(selector='[class*=\"dashboard\"]', timeout_ms=8000) ← تأكيد الدخول\n\n"
            "⚠️ قواعد حاسمة لتسجيل الدخول:\n"
            "   ❌ لا تستخدم أبداً: browser_eval_js('document.querySelector(...).click()')\n"
            "      → هذا لا يعمل في React/Vue/Angular (Replit, GitHub, Google...)\n"
            "   ✅ استخدم دائماً: browser_click(selector='...') أو browser_press(key='Enter')\n"
            "   ✅ إذا فشل browser_fill، جرب: browser_react_fill(selector='...', value='...')\n"
            "      → browser_react_fill مخصص لمواقع SPA التي تستخدم React/Vue/Angular\n\n"
            "✅ مثال تسجيل دخول Replit:\n"
            "   1. browser_open(url='https://replit.com/login')\n"
            "   2. browser_fill(selector='input[name=\"username\"]', value='email@gmail.com')\n"
            "   3. browser_fill(selector='input[name=\"password\"]', value='كلمة_السر')\n"
            "   4. browser_press(key='Enter')\n"
            "   5. browser_wait_for(selector='.replit-ui-theme-root', timeout_ms=10000)\n\n"
            "✅ مثال تسجيل دخول Gmail:\n"
            "   1. browser_open(url='https://mail.google.com')\n"
            "   2. browser_fill(selector='input[type=\"email\"]', value='user@gmail.com')\n"
            "   3. browser_press(key='Enter')\n"
            "   4. browser_wait_for(selector='input[type=\"password\"]', timeout_ms=5000)\n"
            "   5. browser_fill(selector='input[type=\"password\"]', value='كلمة_السر')\n"
            "   6. browser_press(key='Enter')\n\n"
            "دليل اختيار الأدوات:\n"
            "  📂 الملفات:\n"
            "    • قراءة ملف                    → read_file(path='...')\n"
            "    • كتابة ملف                    → write_file(path='...', content='...')\n"
            "    • عرض محتويات مجلد              → list_dir(path='...')\n"
            "    • بحث عن ملفات                  → search_files(root='...', pattern='*.pdf')\n"
            "    • نسخ/نقل ملف                   → copy_file / move_file\n"
            "    • تحميل من URL مباشر             → download_file(url='...', dest='desktop:file.pdf')\n"
            "    • إنشاء مجلد                    → make_dir(path='...')\n\n"
            "  🚀 التطبيقات:\n"
            "    • فتح تطبيق                     → open_app(name='chrome')\n"
            "    • إغلاق تطبيق                   → close_app(name='notepad')\n"
            "    • جلب نافذة للأمام               → focus_window(title='...')\n\n"
            "  🖱️ سطح المكتب (GUI):\n"
            "    • لقطة شاشة                     → screen_screenshot()\n"
            "    • نقر في إحداثيات               → mouse_click(x=100, y=200)\n"
            "    • كتابة نص                      → keyboard_type(text='...')\n"
            "    • اختصار لوحة مفاتيح            → keyboard_hotkey(keys='ctrl,s')\n"
            "    • قائمة النوافذ                  → list_windows()\n"
            "    • انتظار                        → wait(seconds=2)\n\n"
            "  🌐 المتصفح:\n"
            "    • فتح صفحة                      → browser_open(url='https://...')\n"
            "    • قراءة نص الصفحة               → browser_get_text()\n"
            "    • نقر على عنصر/زر               → browser_click(selector='button[type=\"submit\"]')\n"
            "    • ملء حقل عادي                  → browser_fill(selector='input[name=x]', value='y')\n"
            "    • ملء حقل React/Vue/Angular     → browser_react_fill(selector='...', value='...')\n"
            "    • ضغط مفتاح (مثل Enter)         → browser_press(key='Enter')\n"
            "    • لقطة شاشة المتصفح (مرة واحدة)→ browser_screenshot()\n\n"
            "  💻 الأوامر:\n"
            "    • PowerShell                    → run_powershell(command='...')\n"
            "    • CMD                           → run_cmd(command='...')\n\n"
            "  🔧 النظام:\n"
            "    • معلومات النظام                → get_system_info()\n"
            "    • العمليات الجارية               → list_processes(sort_by='memory')\n"
            "    • إيقاف عملية                   → kill_process(target='chrome')\n"
            "    • إدارة خدمة                    → manage_service(service_name='...', action='status')\n\n"
            "  📋 الحافظة:\n"
            "    • قراءة الحافظة                 → clipboard_get()\n"
            "    • كتابة في الحافظة              → clipboard_set(text='...')\n\n"
            "  🌍 الشبكة:\n"
            "    • معلومات الشبكة                → get_network_info()\n"
            "    • فحص اتصال                     → ping_host(host='google.com')\n"
            "    • فحص منفذ                      → check_port(host='...', port=80)\n\n"
            "  🔊 الصوت:\n"
            "    • التحكم بالصوت                 → volume_control(action='set', level=50)\n"
            "    • قراءة نص بصوت عالٍ            → text_to_speech(text='...')\n"
            "    • إشعار Windows                 → show_notification(title='...', message='...')\n\n"
            "استراتيجية العمل مع التطبيقات:\n"
            "  1. open_app('appname') → فتح التطبيق\n"
            "  2. wait(seconds=3)     → انتظار التحميل\n"
            "  3. screen_screenshot() → رؤية الشاشة والإحداثيات\n"
            "  4. focus_window('...')  → جلب النافذة للأمام\n"
            "  5. mouse_click(x,y)    → نقر\n"
            "  6. keyboard_type('...')→ كتابة\n"
            "  7. keyboard_hotkey('ctrl,s') → حفظ\n\n"
            "قواعد سرعة PowerShell:\n"
            "  • لا تستخدم Get-ComputerInfo — استخدم get_system_info() بدلاً منها\n"
            "  • لا تستخدم Get-Counter — استخدم Get-WmiObject\n\n"
            "استعادة الأخطاء:\n"
            "  • إذا فشلت الأداة → جرب نهجاً مختلفاً تماماً\n"
            "  • إذا لم يُعثر على ملف → استخدم search_files أو run_powershell للبحث\n"
            "  • لا تسأل المستخدم أبداً 'أين الملف؟' — ابحث بنفسك أولاً\n\n"
            "تغيير الموضوع:\n"
            "  • إذا غيّر المستخدم طلبه → المهمة القديمة ملغاة. نفذ الطلب الجديد فقط."
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
    tool_call_history = list(state.get("tool_call_history", []))
    task_id = state.get("task_id", "unknown")

    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        for tc in llm_response.tool_calls:
            # Defensive: extract fields safely — DeepSeek may emit tool_calls
            # with unexpected shapes
            try:
                if isinstance(tc, dict):
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {}) or {}
                    tool_id   = tc.get("id", "")
                else:
                    tool_name = getattr(tc, "name", "")
                    tool_args = getattr(tc, "args", {}) or {}
                    tool_id   = getattr(tc, "id", "")
            except Exception as exc:
                # Malformed tool_call — skip but log
                error_logs.append(f"[iter {iteration+1}] Malformed tool_call: {exc}")
                continue

            # Always ensure tool_id exists so we can produce a ToolMessage
            if not tool_id:
                tool_id = f"missing_id_{uuid.uuid4()}"

            # ── Check for duplicate tool call ────────────────────────────────
            if is_duplicate_tool_call(tool_name, tool_args, tool_call_history, recent_count=2):
                # Give the model actionable guidance based on which tool is looping
                if tool_name in ("browser_screenshot", "screen_screenshot"):
                    skip_hint = (
                        "You already have the screenshot from the previous step. "
                        "Do NOT take another screenshot — instead, act on what you already know. "
                        "If the form didn't submit, use browser_press(key='Enter') or "
                        "browser_click(selector='button[type=\"submit\"]')."
                    )
                elif tool_name == "browser_get_text":
                    skip_hint = (
                        "You already read the page text. "
                        "Do NOT read it again — act on the content you already have. "
                        "If login failed, try browser_react_fill() or browser_press(key='Enter')."
                    )
                elif tool_name in ("browser_fill", "browser_react_fill"):
                    skip_hint = (
                        "You already filled this field. "
                        "Move to the next step: use browser_press(key='Enter') or "
                        "browser_click(selector='button[type=\"submit\"]') to submit."
                    )
                else:
                    skip_hint = (
                        "Avoid repeating the same tool call. "
                        "Try a completely different approach or tool."
                    )
                result = (
                    f"⏭️ SKIPPED: Tool '{tool_name}' with identical args was already called recently. "
                    f"{skip_hint}"
                )
                new_messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                continue

            tool_fn = TOOL_MAP.get(tool_name)
            if not tool_fn:
                result = f"❌ ERROR: Tool '{tool_name}' is not registered. Available tools: {list(TOOL_MAP.keys())}"
            else:
                try:
                    raw_result = tool_fn.invoke(tool_args)
                except Exception as exc:
                    raw_result = f"❌ ERROR running {tool_name}: {type(exc).__name__}: {exc}"
                    error_logs.append(f"[task:{task_id}][{tool_name}] {raw_result[:300]}")

                # ── Case A: Destructive command detected ─────────────────
                _hitl_sentinels = ("HITL_APPROVAL_REQUIRED:", "__HITL_REQUIRED__")
                _is_hitl = isinstance(raw_result, str) and any(
                    raw_result.startswith(s) or s in raw_result for s in _hitl_sentinels
                )

                if _is_hitl:
                    # Extract the risky command from the sentinel
                    risky_cmd = str(raw_result)
                    for s in _hitl_sentinels:
                        risky_cmd = risky_cmd.replace(s, "")
                    # Try to extract "Command:" line if present
                    for line in raw_result.splitlines():
                        if line.strip().startswith("Command:"):
                            risky_cmd = line.split("Command:", 1)[1].strip()
                            break
                    risky_cmd = risky_cmd.strip()

                    user_choice: str = interrupt(
                        {
                            "type":    "destructive_command",
                            "command": risky_cmd,
                            "message": (
                                f"⚠️ الوكيل يريد تنفيذ أمر قد يكون خطيراً:\n"
                                f"```powershell\n{risky_cmd}\n```\n"
                                "اضغط **موافق** للسماح، أو **رفض** للمنع."
                            ),
                        }
                    )

                    if user_choice == "approve":
                        try:
                            proc = subprocess.run(
                                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", risky_cmd],
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
                            error_logs.append(f"[task:{task_id}] {result[:300]}")
                    else:
                        result = f"🚫 Command denied by user: `{risky_cmd}` — skipping this step."

                # ── Case B: CAPTCHA detected ──────────────────────────────
                elif isinstance(raw_result, str) and "CAPTCHA_DETECTED" in raw_result:
                    interrupt(
                        {
                            "type": "captcha",
                            "message": (
                                "🔒 تم اكتشاف CAPTCHA. نافذة المتصفح مفتوحة على شاشتك. "
                                "يرجى حل الـ CAPTCHA يدوياً، ثم اضغط **تم** للاستئناف."
                            ),
                        }
                    )
                    result = "✅ User confirmed CAPTCHA solved. Resuming from current page."

                # ── Case C: Normal tool result ────────────────────────────
                else:
                    result = raw_result

                # Track errors for the reviewer with task context
                if isinstance(result, str) and (
                    "❌" in result or "error" in result.lower() or "traceback" in result.lower()
                ):
                    error_logs.append(f"[task:{task_id}][{tool_name}] {result[:300]}")

            new_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )

            # ── Record tool call in history ──────────────────────────────────
            tool_call_history = record_tool_call(
                tool_name=tool_name,
                tool_args=tool_args,
                result=result,
                tool_history=tool_call_history,
                max_history=20,
            )

        # Record this step as completed
        step_label = (
            plan[len(updated_completed)]
            if len(updated_completed) < len(plan)
            else f"Extra step {len(updated_completed) + 1}"
        )
        updated_completed.append(step_label)

    # Get the last tool name and args if any were called
    last_tool_name = ""
    last_tool_args = {}
    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        last_tc = llm_response.tool_calls[-1]
        last_tool_name = last_tc.get("name", "")
        last_tool_args = last_tc.get("args", {})

    # Final defensive sanitization — guarantees no invalid tool sequences
    # reach the reducer, even if something above missed a ToolMessage.
    new_messages = _sanitize_messages(new_messages)

    return {
        "messages":                new_messages,
        "iteration_count":         iteration + 1,
        "error_logs":              error_logs[-30:],
        "completed_steps":         updated_completed,
        "plan":                    plan,
        "workspace":               state.get("workspace", ""),
        "requires_human_approval": False,
        "pending_command":         "",
        "tool_call_history":       tool_call_history,  # Updated history
        "last_tool_name":          last_tool_name,     # Track last tool
        "last_tool_args":          last_tool_args,     # Track last args
        "last_message_content":    state.get("last_message_content", ""),
        "task_id":                 state.get("task_id", str(uuid.uuid4())),
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
★ If the same tool was called (or SKIPPED for duplication) 3+ times → FAILED
  (The worker is stuck looping. Stop it now and report the failure.)

★ If you see "SKIPPED: Tool ... was already called" 2+ times → FAILED with a
  message telling the user to try a different approach (suggest the right
  yt-dlp tool: download_audio_by_search for songs).

★ If the worker failed to call ANY tool 2+ times in recent messages → TASK_COMPLETE
  (The agent is stuck and needs user input. Do NOT say CONTINUE — it will loop forever.)

★ If the worker is trying to scrape Google/sketchy mp3 sites for a song → REPLAN
  Tell it: "Use download_audio_by_search(query='song title artist', dest='desktop:')
  instead — that's the correct tool for downloading songs."

★ If the same "file not found" or "no results" situation appeared 2+ times → FAILED
  (Do NOT keep saying CONTINUE for a resource that clearly does not exist.)

★ If you see "I am ready to help" or similar waiting messages repeated → TASK_COMPLETE
  (The agent has completed what it can and is now waiting for the user.)

★ If the user changed their request mid-task (e.g., from music to project) → TASK_COMPLETE
  (The old task is abandoned. The new task will be handled in the next cycle.)

★ CRITICAL: If you see a "NEW TASK BOUNDARY" line, your VERDICT scope is the
  plan that comes AFTER it. Do NOT say CONTINUE based on unfinished work from
  an earlier plan. BUT — the agent's MEMORY still includes earlier messages,
  so summaries, tool results, and downloaded files from before remain valid
  context (e.g. "the song we just downloaded" is a real artifact, even though
  the download task itself is closed).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NORMAL DECISION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• CONTINUE is only valid when a NEW tool call can actually be made right now.
• Count plan steps vs completed steps. They must match for TASK_COMPLETE.
• web_search ran but download_file has NOT run yet → CONTINUE with exact URL.
• File downloaded successfully → TASK_COMPLETE.
• Never say CONTINUE just to "ask the user" — that belongs in TASK_COMPLETE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY STYLE (this is read aloud in voice mode — keep it human):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After your verdict, write a SHORT human-friendly summary (1–3 sentences):
  • Match the user's language (Arabic if they wrote in Arabic, English otherwise)
  • Sound natural and warm, like a helpful friend — NOT a status report
  • Skip technical jargon, file paths, tool names, and bullet points
  • Use contractions and conversational phrasing
  • Express emotion when fitting: relief on success, regret on failure
  • No markdown formatting (no **, no #, no ```) — plain prose only

Good (Arabic): "تمام! حملت الملف على سطح المكتب. تقدر تفتحه دلوقتي."
Good (English): "Done! The file's on your desktop, ready when you are."
Bad: "**TASK_COMPLETE:** File `song.mp3` downloaded to `C:\\Users\\...\\Desktop\\song.mp3` via `download_file()`."
"""

def reviewer_node(state: AgentState) -> dict:
    """
    Evaluates completed work and decides: TASK_COMPLETE, CONTINUE, or FAILED.
    """
    _ensure_provider_match()  # Ensure correct LLM provider is being used
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
            "tool_call_history":       state.get("tool_call_history", []),
            "last_tool_name":          state.get("last_tool_name", ""),
            "last_tool_args":          state.get("last_tool_args", {}),
            "last_message_content":    state.get("last_message_content", ""),
            "task_id":                 state.get("task_id", str(uuid.uuid4())),
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

    # ── Check for duplicate review message ───────────────────────────────────
    if is_duplicate_message(response, messages, min_length=50):
        response = AIMessage(
            content=(
                response.content + "\n\n"
                "⚠️ [Review was rephrased to avoid exact duplication]"
            )
        )

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
        "tool_call_history":       state.get("tool_call_history", []),
        "last_tool_name":          state.get("last_tool_name", ""),
        "last_tool_args":          state.get("last_tool_args", {}),
        "last_message_content":    state.get("last_message_content", ""),
        "task_id":                 state.get("task_id", str(uuid.uuid4())),
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
        return "__end__"

    # ── Tool-loop guard: same tool being called or SKIPPED repeatedly ─────────
    # Counts how many SKIPPED-duplicate messages we've seen recently.
    # Threshold lowered to 2 (was 3) — catches loops earlier.
    skipped_streak = sum(
        1
        for msg in messages[-12:]
        if (
            isinstance(msg, ToolMessage)
            and "SKIPPED: Tool" in (msg.content or "")
        )
    )
    if skipped_streak >= 2:
        return "__end__"

    # Also: if the LAST 3 tool calls all targeted the same tool, we're looping.
    # Threshold lowered to 3 (was 5) — catches browser_screenshot loops sooner.
    tool_history = state.get("tool_call_history", [])
    if len(tool_history) >= 3:
        recent_names = [call.get("name", "") for call in tool_history[-3:]]
        if len(set(recent_names)) == 1 and recent_names[0]:
            # Same tool 3 times in a row — stop
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
