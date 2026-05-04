# 🤖 HAYO AI AGENT

وكيل ذكاء اصطناعي محلي خارق القدرات — يعمل على جهازك بصلاحيات كاملة.

## ما الذي يستطيع فعله

- **🌐 المتصفح**: يفتح Chrome، يتصفح المواقع، يسجل دخول، يملأ النماذج، يحمل الملفات
- **🖥️ سطح المكتب**: يفتح أي تطبيق (Word, Photoshop, VSCode...)، يلتقط لقطات شاشة، ينقر، يكتب، يستخدم اختصارات لوحة المفاتيح
- **📁 الملفات**: قراءة، كتابة، نسخ، نقل، بحث، تحميل من الإنترنت إلى سطح المكتب
- **💻 PowerShell**: تنفيذ أي أمر Windows مع حماية مدمجة من الأوامر المدمرة
- **🎵 الموسيقى**: تحميل من YouTube مباشرة بصيغة MP3 عبر `ytsearch:`
- **🔍 Git**: عمليات على المستودعات المحلية
- **🔎 بحث الويب**: عبر DuckDuckGo بدون مفتاح API

## المعمارية

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PLANNER    │───►│   WORKER    │───►│  REVIEWER   │
│ (يخطط)      │    │ (ينفذ)      │    │ (يراجع)     │
└─────────────┘    └─────▲───────┘    └─────┬───────┘
                         │                  │
                         └──────────────────┘
                              CONTINUE
```

- **LangGraph** كبنية أساسية للحالة والتوجيه
- **SQLite Checkpointer** لحفظ المحادثات بين تشغيلات الخادم
- **Human-in-the-Loop** للأوامر المدمرة و CAPTCHA — يتوقف الوكيل ويسأل
- **متعدد المزودين** — Anthropic Claude أو Google Gemini عبر متغير `.env`

## التثبيت السريع (لمرة واحدة)

```powershell
cd "C:\HAYO AI AGENT"

# 1) تفعيل البيئة
.\venv\Scripts\activate

# 2) تثبيت التبعيات
pip install -r requirements.txt

# 3) تثبيت متصفح Playwright (مرة واحدة)
playwright install chromium

# 4) إعداد المفاتيح
copy .env.example .env
notepad .env
# عدّل GOOGLE_API_KEY أو ANTHROPIC_API_KEY
```

## التشغيل

### الواجهة الرسومية (الأسهل)
انقر مرتين على `START.bat` — سيفتح المتصفح تلقائياً على `http://localhost:8000`.

### سطر الأوامر
```powershell
python main.py
# أو نفّذ طلبًا واحدًا واخرج:
python main.py --once "افتح Chrome وحمّل ملف PDF من https://example.com/file.pdf إلى سطح المكتب"
```

## أمثلة طلبات

| الطلب | ما سيفعله الوكيل |
|---|---|
| "افتح Word واكتب فيه رسالة شكر" | open_app('word') → desktop_control type |
| "حمّل أغنية عمرو دياب نور العين على سطح المكتب" | download_file('ytsearch:Amr Diab Nour El Ain') |
| "ابحث في GitHub عن مشاريع LangGraph وافتح أعلى نتيجة" | browser_open + browser_click |
| "سجّل دخول إلى جيميل وأرسل بريد إلى x@y.com" | browser_automation (يحفظ الجلسة) |
| "اعرض المساحة المتبقية في القرص C" | execute_powershell |

## الأمان

الوكيل لا يستطيع تنفيذ هذه الأوامر بدون موافقتك الصريحة:
- `Remove-Item -Recurse`, `rm -rf`, `format`
- `shutdown`, `Restart-Computer`
- `Set-ExecutionPolicy`, `reg delete`
- `Add-LocalGroupMember`, `New-LocalUser`

عند اكتشاف نمط مدمر، الوكيل **يتوقف** ويعرض زر موافقة/رفض في الواجهة.

## هيكل المشروع

```
HAYO AI AGENT/
├── app.py                 # واجهة Chainlit (الافتراضية)
├── main.py                # واجهة CLI بديلة
├── START.bat              # تشغيل بنقرة واحدة
├── .env                   # مفاتيح API (لا ترفعه)
├── requirements.txt
│
├── agent/
│   ├── workflow.py        # تجميع LangGraph (compile_graph)
│   ├── nodes.py           # PlannerNode + WorkerNode + ReviewerNode
│   └── graph.py           # shim للتوافق
│
├── core/
│   └── state.py           # AgentState TypedDict
│
└── tools/
    ├── os_core.py         # PowerShell + ملفات + manage_files
    ├── web_and_cloud.py   # Playwright browser + Git
    ├── web_tools.py       # web_search + download_file (yt-dlp)
    └── desktop_control.py # pyautogui + pygetwindow + لقطات شاشة
```

## استكشاف الأخطاء

| مشكلة | الحل |
|---|---|
| `MODEL_PROVIDER='google' but GOOGLE_API_KEY is empty` | عدّل `.env` وضع المفتاح من https://aistudio.google.com/app/apikey |
| `playwright not installed` | `playwright install chromium` |
| `ModuleNotFoundError: pyautogui` | `pip install -r requirements.txt` |
| الوكيل يدور بلا توقف | المشروع به حد `MAX_ITERATIONS=20` يُجبره على التوقف. ارفعه في `.env` إن احتجت. |
| المتصفح يفتح بدون كوكيز | أول مرة يفتح profile جديد. سجّل دخول ستحفظه `.browser_profile/` تلقائياً. |

## ترقيات مقترحة لاحقاً

- إضافة وحدة `tools/code_tools.py` لإدارة Python venv وتشغيل scripts
- استبدال DuckDuckGo بـ Tavily/Brave API للبحث الأقوى
- ربط ذاكرة طويلة الأمد (Mem0 / pgvector) لتذكر تفضيلاتك بين الجلسات
