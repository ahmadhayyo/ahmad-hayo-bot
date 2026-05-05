#!/data/data/com.termux/files/usr/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  HAYO AI Agent — Android — Stop Script
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "⛔ إيقاف HAYO AI Agent..."

# Kill chainlit process
pkill -f "chainlit run app.py" 2>/dev/null
pkill -f "python.*app.py" 2>/dev/null

# Kill any process on port 8000
fuser -k 8000/tcp 2>/dev/null

echo "✅ تم إيقاف الوكيل"
echo ""
