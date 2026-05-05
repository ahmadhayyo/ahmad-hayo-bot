#!/data/data/com.termux/files/usr/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  HAYO AI Agent — Android Setup Script
#  Run this ONCE in Termux to install everything
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║     🤖 HAYO AI Agent — Android Setup             ║"
echo "║     جاري التثبيت... يرجى الانتظار                ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Update Termux packages ──────────────────────────────────────────
echo "📦 [1/6] تحديث الحزم..."
pkg update -y && pkg upgrade -y

# ── Step 2: Install Python and essentials ────────────────────────────────────
echo "🐍 [2/6] تثبيت Python والأدوات الأساسية..."
pkg install -y python python-pip git curl wget openssl libffi rust binutils

# ── Step 3: Install termux-api for hardware access ──────────────────────────
echo "📱 [3/6] تثبيت Termux API..."
pkg install -y termux-api

# ── Step 4: Install yt-dlp for media downloads ──────────────────────────────
echo "🎵 [4/6] تثبيت yt-dlp..."
pip install --upgrade yt-dlp 2>/dev/null || echo "  ⚠️ yt-dlp optional — skipping"

# ── Step 5: Install Python dependencies ─────────────────────────────────────
echo "📚 [5/6] تثبيت مكتبات Python..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
pip install --upgrade pip
pip install -r requirements.txt

# ── Step 6: Setup .env file ─────────────────────────────────────────────────
echo "⚙️ [6/6] إعداد ملف التكوين..."
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "  ✅ تم إنشاء .env — يرجى تعديله وإضافة مفتاح API"
    echo ""
    echo "  لتعديل الملف اكتب:"
    echo "    nano $SCRIPT_DIR/.env"
else
    echo "  ✅ ملف .env موجود بالفعل"
fi

# ── Step 7: Storage permission ──────────────────────────────────────────────
echo "📂 إعداد صلاحيات التخزين..."
termux-setup-storage 2>/dev/null || echo "  ⚠️ Run 'termux-setup-storage' manually if needed"

# ── Step 8: Create quick-start alias ────────────────────────────────────────
echo ""
echo "📝 إنشاء اختصارات التشغيل..."

# Add to .bashrc for easy access
BASHRC="$HOME/.bashrc"
ALIAS_LINE="alias hayo='cd \"$SCRIPT_DIR\" && bash start.sh'"
ALIAS_STOP="alias hayo-stop='bash \"$SCRIPT_DIR/stop.sh\"'"

grep -q "alias hayo=" "$BASHRC" 2>/dev/null || echo "$ALIAS_LINE" >> "$BASHRC"
grep -q "alias hayo-stop=" "$BASHRC" 2>/dev/null || echo "$ALIAS_STOP" >> "$BASHRC"

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║     ✅ اكتمل التثبيت بنجاح!                      ║"
echo "╠═══════════════════════════════════════════════════╣"
echo "║                                                   ║"
echo "║  الخطوة التالية:                                   ║"
echo "║  1. عدّل ملف .env وأضف مفتاح API:                ║"
echo "║     nano $SCRIPT_DIR/.env                         ║"
echo "║                                                   ║"
echo "║  2. لتشغيل الوكيل اكتب:                           ║"
echo "║     hayo                                          ║"
echo "║     أو: bash $SCRIPT_DIR/start.sh                 ║"
echo "║                                                   ║"
echo "║  3. افتح Chrome على:                               ║"
echo "║     http://localhost:8000                          ║"
echo "║                                                   ║"
echo "║  4. لإيقاف الوكيل:                                ║"
echo "║     hayo-stop                                     ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
