# 🤖 HAYO AI Agent — Android Edition

وكيل ذكاء اصطناعي خارق القدرات لجهاز Android مع صلاحيات Root كاملة.

---

## 📱 المتطلبات

| المتطلب | التفاصيل |
|---------|----------|
| **جهاز** | Samsung أو أي جهاز Android |
| **نظام** | Android 8.0+ (Samsung Note 8 مدعوم) |
| **Root** | مطلوب — عبر Magisk |
| **Termux** | تحميل من [F-Droid](https://f-droid.org/packages/com.termux/) |
| **Termux:API** | تحميل من [F-Droid](https://f-droid.org/packages/com.termux.api/) |
| **Chrome** | للوصول إلى واجهة الوكيل |
| **مفتاح API** | واحد على الأقل (Google/Claude/ChatGPT/DeepSeek) |

---

## 🚀 التثبيت (مرة واحدة فقط)

### الخطوة 1: تحميل التطبيقات المطلوبة

1. **Termux** — من [F-Droid](https://f-droid.org/packages/com.termux/) (لا تستخدم نسخة Play Store!)
2. **Termux:API** — من [F-Droid](https://f-droid.org/packages/com.termux.api/)

### الخطوة 2: نقل المجلد إلى الموبايل

انقل مجلد `HAYO AI AGENT ANDROID` بالكامل إلى:
```
/sdcard/HAYO AI AGENT ANDROID/
```

يمكنك نقله عبر USB أو Bluetooth أو Google Drive.

### الخطوة 3: تشغيل التثبيت

افتح **Termux** والصق هذه الأوامر:

```bash
# الدخول إلى مجلد المشروع
cd /sdcard/HAYO\ AI\ AGENT\ ANDROID/

# إعطاء صلاحية التنفيذ
chmod +x setup.sh start.sh stop.sh

# تشغيل التثبيت (قد يأخذ 5-10 دقائق)
bash setup.sh
```

### الخطوة 4: إضافة مفتاح API

```bash
nano .env
```

أضف مفتاح API واحد على الأقل:
```
GOOGLE_API_KEY=your-key-here
```

اضغط `Ctrl+X` ثم `Y` ثم `Enter` للحفظ.

---

## ▶️ التشغيل اليومي

### الطريقة 1 (الأسهل):
```bash
hayo
```
ثم افتح Chrome على: `http://localhost:8000`

### الطريقة 2:
```bash
cd /sdcard/HAYO\ AI\ AGENT\ ANDROID/
bash start.sh
```

### الإيقاف:
```bash
hayo-stop
```
أو اضغط `Ctrl+C` في Termux.

---

## 🛠️ القدرات (60+ أداة)

### 📱 شاشة الموبايل (Root)
| الأداة | الوظيفة |
|--------|---------|
| `screen_screenshot` | لقطة شاشة |
| `screen_tap` | النقر على الشاشة |
| `screen_swipe` | السحب على الشاشة |
| `screen_type_text` | كتابة نص |
| `screen_key_event` | أزرار (HOME, BACK, ENTER) |
| `screen_long_press` | ضغط مطوّل |
| `screen_brightness` | سطوع الشاشة |
| `screen_rotate` | تدوير الشاشة |

### 🚀 التطبيقات (Root)
| الأداة | الوظيفة |
|--------|---------|
| `open_app` | فتح أي تطبيق (chrome, whatsapp, youtube...) |
| `close_app` | إغلاق تطبيق |
| `list_installed_apps` | عرض التطبيقات المثبتة |
| `install_apk` | تثبيت APK |
| `get_current_app` | التطبيق المفتوح حالياً |

### 💻 النظام والأوامر
| الأداة | الوظيفة |
|--------|---------|
| `run_shell` | تنفيذ أمر bash |
| `run_root` | تنفيذ أمر بصلاحيات root |
| `get_device_info` | معلومات الجهاز الشاملة |
| `get_battery_info` | حالة البطارية |
| `get_storage_info` | المساحة المتاحة |

### 📁 الملفات
| الأداة | الوظيفة |
|--------|---------|
| `read_file` | قراءة ملف |
| `write_file` | كتابة ملف |
| `list_dir` | محتويات مجلد |
| `search_files` | بحث عن ملفات |
| `download_file` | تحميل ملف/فيديو YouTube |
| `move_file` / `copy_file` | نقل/نسخ |

### 📊 التحكم بالجهاز (Root)
| الأداة | الوظيفة |
|--------|---------|
| `set_wifi` | تفعيل/تعطيل Wi-Fi |
| `set_bluetooth` | تفعيل/تعطيل Bluetooth |
| `set_airplane_mode` | وضع الطيران |
| `set_mobile_data` | بيانات الهاتف |

### 🌍 الشبكة
| الأداة | الوظيفة |
|--------|---------|
| `ping_host` | اختبار اتصال |
| `get_network_info` | معلومات الشبكة |
| `wifi_scan` | فحص شبكات Wi-Fi |
| `traceroute` | تتبع المسار |

### 🔊 الصوت والأجهزة
| الأداة | الوظيفة |
|--------|---------|
| `volume_control` | مستوى الصوت |
| `text_to_speech` | قراءة نص بصوت عالٍ |
| `vibrate` | اهتزاز |
| `torch_control` | الكشاف |
| `show_notification` | إشعار |

### 🔍 الويب
| الأداة | الوظيفة |
|--------|---------|
| `web_search` | بحث في الإنترنت |
| `fetch_url` | جلب محتوى صفحة |
| `download_url` | تحميل ملف من رابط |

---

## 🤖 النماذج المدعومة

| النموذج | المزود | المفتاح |
|---------|--------|---------|
| 🟦 Gemini 2.5 Flash | Google | `GOOGLE_API_KEY` |
| 🟠 Claude Sonnet 4 | Anthropic | `ANTHROPIC_API_KEY` |
| 🟢 GPT-4o | OpenAI | `OPENAI_API_KEY` |
| 🔵 DeepSeek Chat | DeepSeek | `DEEPSEEK_API_KEY` |

لتبديل النموذج أثناء الاستخدام:
```
/model google
/model anthropic
/model openai
/model deepseek
```

---

## 🔒 الأمان

- ✅ الوكيل يطلب **إذنك** قبل تنفيذ أوامر خطرة (حذف ملفات، rm -rf، reboot...)
- ✅ لا يحذف ملفات النظام الأساسية
- ✅ لا يسجل أو يرسل مفاتيح API
- ✅ يعمل محلياً بالكامل — لا بيانات ترسل لأطراف ثالثة

---

## 🔧 حل المشاكل

### "خطأ في التثبيت"
```bash
pkg install -y python python-pip
pip install --upgrade pip setuptools wheel
```

### "الوكيل لا يتصل بالأدوات"
تأكد أن Root يعمل:
```bash
su -c "whoami"
# يجب أن يظهر: root
```

### "Termux:API لا يعمل"
تأكد من تثبيته من F-Droid (ليس Play Store):
```bash
pkg install termux-api
```

### "لا أستطيع الوصول إلى الملفات"
```bash
termux-setup-storage
```

### "الصفحة لا تفتح في Chrome"
تأكد أن الوكيل يعمل:
```bash
curl http://localhost:8000
```

---

## 📂 هيكل المشروع

```
HAYO AI AGENT ANDROID/
├── app.py              ← واجهة Chainlit الرئيسية
├── config.py           ← الإعدادات والمتغيرات
├── requirements.txt    ← مكتبات Python
├── setup.sh            ← سكربت التثبيت (مرة واحدة)
├── start.sh            ← تشغيل الوكيل
├── stop.sh             ← إيقاف الوكيل
├── .env.example        ← نموذج ملف التكوين
├── .chainlit           ← إعدادات واجهة Chainlit
├── chainlit.md         ← صفحة ترحيب الواجهة
├── agent/
│   ├── nodes.py        ← عقد التخطيط والتنفيذ والمراجعة
│   └── workflow.py     ← رسم LangGraph البياني
├── core/
│   ├── state.py        ← حالة الوكيل
│   └── safety.py       ← حماية ضد الأوامر الخطرة
└── tools/
    ├── registry.py     ← سجل الأدوات الموحد (60+ أداة)
    ├── system_tools.py ← أوامر Shell و Root
    ├── file_tools.py   ← قراءة/كتابة/بحث الملفات
    ├── screen_tools.py ← تحكم بالشاشة (Root)
    ├── app_tools.py    ← فتح/إغلاق التطبيقات (Root)
    ├── device_tools.py ← معلومات الجهاز والتحكم
    ├── network_tools.py← أدوات الشبكة
    ├── clipboard_tools.py ← الحافظة
    ├── audio_tools.py  ← الصوت والإشعارات
    └── web_tools.py    ← بحث وتحميل من الويب
```

---

## 📝 أمثلة استخدام

```
"افتح YouTube وابحث عن أغنية"
"ما هي معلومات جهازي؟"
"حمّل هذا الملف: https://example.com/file.pdf"
"أظهر نسبة البطارية"
"افتح WhatsApp"
"ابحث في الويب عن أحدث أخبار التقنية"
"خذ لقطة شاشة"
"فعّل Wi-Fi"
"أطفئ Bluetooth"
"اعرض التطبيقات المثبتة"
"ابحث عن ملفات PDF في التحميلات"
```

---

> **Made with ❤️ by HAYO AI**
