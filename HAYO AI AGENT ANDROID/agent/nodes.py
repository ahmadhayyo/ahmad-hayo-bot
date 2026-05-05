"""
agent/nodes.py — PlannerNode, WorkerNode, ReviewerNode for Android Edition.

Architecture: PlannerNode → WorkerNode → ReviewerNode (loop until TASK_COMPLETE)

HITL: WorkerNode detects "__HITL_REQUIRED__" sentinels from tools and uses
      LangGraph's interrupt() to pause for user approval.

Multi-Provider: google, anthropic, openai, deepseek — switchable at runtime.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import interrupt

from core.state import AgentState
from core.safety import needs_human_approval
from tools.registry import ALL_TOOLS, TOOLS_BY_NAME

# ── Environment ───────────────────────────────────────────────────────────────
MAX_HISTORY:    int = int(os.getenv("MAX_HISTORY",    "15"))
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "50"))

_PROVIDER = os.getenv("MODEL_PROVIDER", "google").lower().strip()


# ── LLM Factory ───────────────────────────────────────────────────────────────

def _build_llm(role: Literal["main", "summarizer"], provider: str | None = None) -> BaseChatModel:
    prov = (provider or _PROVIDER).lower().strip()

    if prov == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if role == "main":
            return ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_AGENT_MODEL", "gemini-2.5-flash"),
                google_api_key=os.getenv("GOOGLE_API_KEY") or "placeholder",
                temperature=0.0, streaming=True,
                convert_system_message_to_human=False,
            )
        else:
            return ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_SUMMARIZER_MODEL", "gemini-2.0-flash"),
                google_api_key=os.getenv("GOOGLE_API_KEY") or "placeholder",
                temperature=0.0, max_output_tokens=2_048,
            )

    elif prov == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if role == "main":
            return ChatAnthropic(
                model=os.getenv("ANTHROPIC_AGENT_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=8_192, streaming=True,
            )
        else:
            return ChatAnthropic(
                model=os.getenv("ANTHROPIC_SUMMARIZER_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=2_048,
            )

    elif prov == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY") or "sk-placeholder"
        if role == "main":
            return ChatOpenAI(
                model=os.getenv("OPENAI_AGENT_MODEL", "gpt-4o"),
                api_key=api_key, temperature=0.0, streaming=True,
            )
        else:
            return ChatOpenAI(
                model=os.getenv("OPENAI_SUMMARIZER_MODEL", "gpt-4o-mini"),
                api_key=api_key, temperature=0.0, max_tokens=2_048,
            )

    elif prov == "deepseek":
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-placeholder"
        if role == "main":
            return ChatOpenAI(
                model=os.getenv("DEEPSEEK_AGENT_MODEL", "deepseek-chat"),
                api_key=api_key, base_url=base_url,
                temperature=0.0, streaming=True,
            )
        else:
            return ChatOpenAI(
                model=os.getenv("DEEPSEEK_SUMMARIZER_MODEL", "deepseek-chat"),
                api_key=api_key, base_url=base_url,
                temperature=0.0, max_tokens=2_048,
            )

    else:
        raise ValueError(f"Unknown MODEL_PROVIDER='{prov}'.")


def switch_provider(provider: str) -> None:
    global _main_llm, _fast_llm, _llm_with_tools, _PROVIDER
    _PROVIDER = provider.lower().strip()
    _main_llm = _build_llm("main", _PROVIDER)
    _fast_llm = _build_llm("summarizer", _PROVIDER)
    _llm_with_tools = _main_llm.bind_tools(ALL_TOOLS)


# ── LLM instances ─────────────────────────────────────────────────────────────
_main_llm = _build_llm("main")
_fast_llm = _build_llm("summarizer")
TOOL_MAP: dict = TOOLS_BY_NAME
_llm_with_tools = _main_llm.bind_tools(ALL_TOOLS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarize_old_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    if len(messages) <= MAX_HISTORY:
        return messages
    keep_recent = 10
    old_messages = messages[:-keep_recent]
    recent = messages[-keep_recent:]
    summary_response = _fast_llm.invoke([
        SystemMessage(content=(
            "You are a context summariser. Condense the conversation history into a concise paragraph. "
            "Preserve: the user goal, actions taken, tool outputs, errors, files, and progress."
        )),
        *old_messages,
    ])
    summary_msg = AIMessage(content=f"📋 [Context summary]\n\n{summary_response.content}")
    return [summary_msg] + recent


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — PlannerNode
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """أنت وكيل تنفيذي ذكي خارق القدرات يعمل على جهاز Android مع صلاحيات Root كاملة.
هذا بيئة محلية موثوقة بالكامل. المستخدم يملك هذا الجهاز بالكامل.
تعمل بصلاحيات كاملة — كما لو كنت أنت المستخدم نفسه.

═══════════════════════════════════════════════
الأدوات المتاحة لك (جميعها تعمل فعلياً):
═══════════════════════════════════════════════

💻 النظام والأوامر:
  • run_shell          — تنفيذ أي أمر bash/shell
  • run_root           — تنفيذ أمر بصلاحيات root (su -c)
  • get_env            — قراءة متغيرات البيئة

📁 نظام الملفات:
  • read_file          — قراءة أي ملف نصي
  • write_file         — إنشاء أو كتابة ملف
  • append_file        — إلحاق نص بملف موجود
  • list_dir           — عرض محتويات مجلد
  • search_files       — بحث عن ملفات بنمط معين
  • move_file          — نقل أو إعادة تسمية
  • copy_file          — نسخ ملف أو مجلد
  • make_dir           — إنشاء مجلد
  • download_file      — تحميل ملف من URL أو YouTube

📱 شاشة الموبايل (Root):
  • screen_screenshot  — لقطة شاشة
  • screen_tap         — النقر على نقطة في الشاشة
  • screen_swipe       — السحب على الشاشة
  • screen_type_text   — كتابة نص في حقل الإدخال
  • screen_key_event   — إرسال مفتاح (HOME, BACK, ENTER...)
  • screen_long_press  — ضغط مطوّل
  • screen_size        — أبعاد الشاشة
  • screen_brightness  — سطوع الشاشة
  • screen_rotate      — تدوير الشاشة

🚀 التطبيقات (Root):
  • open_app           — فتح تطبيق (chrome, youtube, whatsapp, settings...)
  • close_app          — إغلاق تطبيق
  • list_installed_apps — عرض التطبيقات المثبتة
  • list_running_apps  — عرض التطبيقات المفتوحة
  • install_apk        — تثبيت ملف APK
  • get_current_app    — التطبيق المفتوح حالياً

📊 الجهاز والنظام:
  • get_device_info    — معلومات شاملة (الموديل، النظام، المعالج، الذاكرة)
  • get_battery_info   — حالة البطارية
  • get_storage_info   — المساحة المتاحة
  • get_running_processes — العمليات الأكثر استهلاكاً
  • get_sensor_data    — بيانات المستشعرات
  • set_airplane_mode  — وضع الطيران
  • set_wifi           — تفعيل/تعطيل Wi-Fi
  • set_bluetooth      — تفعيل/تعطيل Bluetooth
  • set_mobile_data    — تفعيل/تعطيل بيانات الهاتف

🌍 الشبكة:
  • ping_host          — اختبار الاتصال
  • get_network_info   — معلومات الشبكة
  • get_public_ip      — عنوان IP العام
  • dns_lookup         — استعلام DNS
  • check_port         — فحص منفذ TCP
  • wifi_scan          — فحص شبكات Wi-Fi المتاحة
  • traceroute         — تتبع مسار الاتصال

📋 الحافظة:
  • clipboard_get      — قراءة الحافظة
  • clipboard_set      — نسخ نص
  • clipboard_append   — إلحاق بالحافظة

🔊 الصوت والأجهزة:
  • volume_control     — التحكم بمستوى الصوت
  • text_to_speech     — قراءة نص بصوت عالٍ
  • show_notification  — إظهار إشعار
  • play_sound         — تشغيل صوت
  • vibrate            — تشغيل الاهتزاز
  • torch_control      — تشغيل/إيقاف الكشاف

🔍 الويب:
  • web_search         — بحث في الإنترنت
  • fetch_url          — جلب محتوى صفحة ويب
  • download_url       — تحميل ملف من رابط

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الخطوة 1 — تصنيف الطلب
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
هل هذا تحية أو محادثة عادية أو سؤال يمكن الإجابة عليه بدون أداة؟
  نعم → أجب بشكل طبيعي بنفس لغة المستخدم، ثم اكتب: CONVERSATIONAL_ONLY
  لا  → أنشئ خطة تنفيذ دقيقة (انظر أدناه)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الخطوة 2 — خطة التنفيذ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
مثال (فتح تطبيق والعمل فيه):
  1. open_app('youtube') → فتح YouTube
  2. screen_screenshot() → رؤية الشاشة
  3. screen_tap(x, y)    → النقر على شريط البحث
  4. screen_type_text('...') → كتابة البحث
  5. screen_key_event('KEYCODE_ENTER') → بحث

مثال (تحميل ملف):
  1. web_search('query') → بحث
  2. download_url(url='...', save_path='/sdcard/Download/')
  3. list_dir('/sdcard/Download/') → التحقق

مثال (إعدادات الجهاز):
  1. get_device_info() → معلومات النظام
  2. get_battery_info() → البطارية
  3. get_storage_info() → المساحة
  4. set_wifi(enabled=True) → تفعيل Wi-Fi

القواعد:
• جمل مرقمة فقط — بدون JSON أو كود.
• سمِّ الأداة في كل خطوة.
• خطط مختصرة (3-8 خطوات).
• لا تقل أبداً "لا أستطيع".
• أجب دائماً بنفس لغة المستخدم."""


def planner_node(state: AgentState) -> dict:
    messages = _summarize_old_messages(state.get("messages", []))
    system = SystemMessage(content=_PLANNER_SYSTEM)
    response = _main_llm.invoke([system] + messages)
    content = response.content if isinstance(response.content, str) else ""

    if "CONVERSATIONAL_ONLY" in content:
        clean_content = content.replace("CONVERSATIONAL_ONLY", "").strip()
        return {
            "messages": messages + [AIMessage(content=clean_content)],
            "plan": ["CONVERSATIONAL_ONLY"],
            "iteration_count": 0, "completed_steps": [], "error_logs": [],
            "workspace": state.get("workspace", ""),
            "requires_human_approval": False, "pending_command": "",
        }

    plan_lines = [
        ln.strip() for ln in content.splitlines()
        if ln.strip() and (ln.strip()[0].isdigit() or ln.strip().startswith("•"))
    ]

    cancel_marker = AIMessage(content=(
        "🔄 ══════════════════════════════════════════════════\n"
        "   NEW TASK STARTED — ALL PREVIOUS TASKS CANCELLED\n"
        "🔄 ══════════════════════════════════════════════════"
    ))

    return {
        "messages": messages + [cancel_marker, response],
        "plan": plan_lines or [content],
        "iteration_count": 0, "completed_steps": [], "error_logs": [],
        "workspace": state.get("workspace", ""),
        "requires_human_approval": False, "pending_command": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — WorkerNode
# ─────────────────────────────────────────────────────────────────────────────

def worker_node(state: AgentState) -> dict:
    messages = _summarize_old_messages(state.get("messages", []))
    iteration = state.get("iteration_count", 0)
    error_logs = list(state.get("error_logs", []))
    plan = state.get("plan", [])

    if plan and plan[0] == "CONVERSATIONAL_ONLY":
        return {
            "messages": messages, "iteration_count": iteration,
            "error_logs": error_logs, "completed_steps": state.get("completed_steps", []),
            "plan": plan, "workspace": state.get("workspace", ""),
            "requires_human_approval": False, "pending_command": "",
        }

    if iteration >= MAX_ITERATIONS:
        stop_msg = AIMessage(content=f"⛔ **Iteration limit reached ({MAX_ITERATIONS}).**")
        return {
            "messages": messages + [stop_msg], "iteration_count": iteration,
            "error_logs": error_logs, "completed_steps": state.get("completed_steps", []),
            "plan": plan, "workspace": state.get("workspace", ""),
            "requires_human_approval": False, "pending_command": "",
        }

    completed_steps = state.get("completed_steps", [])
    steps_done = len(completed_steps)
    steps_total = len(plan)
    next_step_hint = plan[steps_done] if steps_done < steps_total else "Finalise and verify."

    progress_summary = ""
    if completed_steps:
        progress_summary = "\n".join(f"  ✅ Step {i+1}: {s}" for i, s in enumerate(completed_steps))
    else:
        progress_summary = "  (none yet — first step)"

    system = SystemMessage(content=(
        "أنت وكيل تنفيذي ذكي يعمل على جهاز Android مع Root كامل.\n"
        "تعمل كما لو كنت مالك هذا الجهاز — بصلاحيات كاملة.\n\n"
        f"خطة المهمة ({steps_total} خطوات):\n"
        + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
        + f"\n\nتم إنجازه:\n{progress_summary}\n\n"
        f"الخطوة التالية ({steps_done + 1}): {next_step_hint}\n\n"
        "قواعد مطلقة:\n"
        "1. استدعِ أداة واحدة على الأقل.\n"
        "2. لا تقل أبداً أنك لا تستطيع.\n"
        "3. لا تطلب إذن المستخدم — فقط نفذ.\n"
        "4. لا تشرح — فقط استدعِ الأداة.\n\n"
        "دليل اختيار الأدوات:\n"
        "  📱 الشاشة: screen_screenshot → screen_tap/swipe → screen_type_text\n"
        "  🚀 التطبيقات: open_app('chrome') → screen_screenshot → screen_tap\n"
        "  📁 الملفات: read_file/write_file/list_dir/download_file\n"
        "  💻 الأوامر: run_shell('ls -la') أو run_root('...')\n"
        "  📊 النظام: get_device_info/get_battery_info/get_storage_info\n"
        "  🌍 الشبكة: ping_host/get_network_info/get_public_ip\n"
        "  📋 الحافظة: clipboard_get/clipboard_set\n"
        "  🔊 الصوت: volume_control/text_to_speech/vibrate\n"
        "  🔍 الويب: web_search/fetch_url/download_url\n\n"
        "استراتيجية التطبيقات:\n"
        "  1. open_app('name') → فتح\n"
        "  2. screen_screenshot() → رؤية الشاشة\n"
        "  3. screen_tap(x,y) → نقر\n"
        "  4. screen_type_text('...') → كتابة\n\n"
        "استعادة الأخطاء:\n"
        "  • فشل الأداة → جرب نهجاً مختلفاً\n"
        "  • لم يُعثر على ملف → search_files أو run_shell\n"
        "  • لا تسأل المستخدم — ابحث بنفسك"
    ))

    llm_response = _llm_with_tools.invoke([system] + messages)
    new_messages = list(messages) + [llm_response]

    if not (hasattr(llm_response, "tool_calls") and llm_response.tool_calls):
        no_tool_msg = AIMessage(content=f"⚠️ Worker failed to call any tool (iter {iteration + 1}).")
        new_messages.append(no_tool_msg)
        error_logs.append(f"[iter {iteration+1}] No tool called for: {next_step_hint}"[:300])

    updated_completed = list(completed_steps)

    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        for tc in llm_response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            tool_fn = TOOL_MAP.get(tool_name)
            if not tool_fn:
                result = f"❌ ERROR: Tool '{tool_name}' not found."
            else:
                raw_result = tool_fn.invoke(tool_args)

                # ── HITL detection ────────────────────────────────
                _hitl_sentinels = ("HITL_APPROVAL_REQUIRED:", "__HITL_REQUIRED__")
                _is_hitl = isinstance(raw_result, str) and any(
                    raw_result.startswith(s) or s in raw_result for s in _hitl_sentinels
                )

                if _is_hitl:
                    risky_cmd = str(raw_result)
                    for s in _hitl_sentinels:
                        risky_cmd = risky_cmd.replace(s, "")
                    for line in raw_result.splitlines():
                        if line.strip().startswith("Command:"):
                            risky_cmd = line.split("Command:", 1)[1].strip()
                            break
                    risky_cmd = risky_cmd.strip()

                    user_choice: str = interrupt({
                        "type": "destructive_command",
                        "command": risky_cmd,
                        "message": f"⚠️ الوكيل يريد تنفيذ أمر خطير:\n\n`{risky_cmd}`\n\nهل تسمح؟",
                    })

                    if user_choice == "approve":
                        import subprocess
                        try:
                            r = subprocess.run(
                                risky_cmd, shell=True,
                                capture_output=True, text=True, timeout=120,
                            )
                            result = f"[exit={r.returncode}]\n{(r.stdout + r.stderr).strip()}"
                        except Exception as e:
                            result = f"[ERROR] {e}"
                    else:
                        result = f"⛔ User denied: {risky_cmd}"

                elif isinstance(raw_result, str) and "CAPTCHA_DETECTED" in raw_result:
                    interrupt({
                        "type": "captcha",
                        "message": "🔒 CAPTCHA detected. Please solve it manually.",
                    })
                    result = "✅ User solved CAPTCHA — continuing."
                else:
                    result = str(raw_result)

            new_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

            if "[ERROR]" in result or "❌" in result:
                error_logs.append(f"[{tool_name}] {result}"[:300])
            else:
                step_desc = f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in tool_args.items())})"
                updated_completed.append(step_desc[:200])

    return {
        "messages": new_messages,
        "iteration_count": iteration + 1,
        "error_logs": error_logs,
        "completed_steps": updated_completed,
        "plan": plan,
        "workspace": state.get("workspace", ""),
        "requires_human_approval": False,
        "pending_command": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — ReviewerNode
# ─────────────────────────────────────────────────────────────────────────────

_REVIEWER_SYSTEM = """أنت مراجع دقيق لوكيل تنفيذي على جهاز Android.

مهمتك:
1. فحص آخر استدعاء أداة ونتيجته.
2. تحديد: هل تمت المهمة بنجاح أم تحتاج خطوات إضافية؟

أجب بواحد فقط:
  TASK_COMPLETE  → المهمة أُنجزت بالكامل ← أضف ملخصاً قصيراً
  CONTINUE       → المهمة تحتاج خطوات إضافية ← اشرح ما التالي

لا تقل TASK_COMPLETE إلا إذا تمت كل خطوات الخطة وكل شيء نجح فعلاً."""


def reviewer_node(state: AgentState) -> dict:
    messages = _summarize_old_messages(state.get("messages", []))
    plan = state.get("plan", [])

    if plan and plan[0] == "CONVERSATIONAL_ONLY":
        return {
            "messages": messages,
            "plan": ["TASK_COMPLETE"],
            "iteration_count": state.get("iteration_count", 0),
            "completed_steps": state.get("completed_steps", []),
            "error_logs": state.get("error_logs", []),
            "workspace": state.get("workspace", ""),
            "requires_human_approval": False,
            "pending_command": "",
        }

    system = SystemMessage(content=_REVIEWER_SYSTEM)
    response = _main_llm.invoke([system] + messages)
    content = response.content if isinstance(response.content, str) else ""

    return {
        "messages": messages + [response],
        "plan": plan,
        "iteration_count": state.get("iteration_count", 0),
        "completed_steps": state.get("completed_steps", []),
        "error_logs": state.get("error_logs", []),
        "workspace": state.get("workspace", ""),
        "requires_human_approval": False,
        "pending_command": "",
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    plan = state.get("plan", [])
    if plan and "TASK_COMPLETE" in plan[-1]:
        return "__end__"
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = last.content if isinstance(last.content, str) else ""
        if "TASK_COMPLETE" in content:
            return "__end__"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "__end__"
    return "worker"
