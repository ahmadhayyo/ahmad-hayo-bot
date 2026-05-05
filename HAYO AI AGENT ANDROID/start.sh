#!/data/data/com.termux/files/usr/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  HAYO AI Agent — Android — Start Script
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║     🤖 HAYO AI Agent — Android                   ║"
echo "║     جاري التشغيل...                               ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# Check .env exists
if [ ! -f ".env" ]; then
    echo "❌ ملف .env غير موجود!"
    echo "   اكتب: cp .env.example .env && nano .env"
    exit 1
fi

# Check API key
source <(grep -v '^#' .env | sed 's/^/export /')
HAS_KEY=0
[ -n "$GOOGLE_API_KEY" ] && HAS_KEY=1
[ -n "$ANTHROPIC_API_KEY" ] && HAS_KEY=1
[ -n "$OPENAI_API_KEY" ] && HAS_KEY=1
[ -n "$DEEPSEEK_API_KEY" ] && HAS_KEY=1

if [ "$HAS_KEY" -eq 0 ]; then
    echo "❌ لا يوجد مفتاح API!"
    echo "   عدّل .env وأضف مفتاح واحد على الأقل"
    echo "   nano .env"
    exit 1
fi

# Get provider
PROVIDER="${MODEL_PROVIDER:-google}"
echo "🤖 النموذج: $PROVIDER"
echo "🌐 الرابط: http://localhost:8000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📱 افتح Chrome وادخل: http://localhost:8000"
echo "  ⛔ لإيقاف الوكيل: اضغط Ctrl+C أو اكتب hayo-stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Open Chrome automatically (if termux-open is available)
(sleep 3 && termux-open-url "http://localhost:8000" 2>/dev/null) &

# Run Chainlit
exec chainlit run app.py --host 0.0.0.0 --port 8000
