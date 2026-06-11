#!/bin/bash
# Lyra-2 UI 快速启动脚本

# 默认配置
HOST="${LYRA_UI_HOST:-127.0.0.1}"
PORT="${LYRA_UI_PORT:-7860}"

echo "🚀 Starting Lyra-2 Demo UI..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo ""
echo "📝 Access the UI at: http://$HOST:$PORT"
echo ""
echo "💡 Tips:"
echo "   - Press Ctrl+C to stop the server"
echo "   - Use environment variables to customize:"
echo "     LYRA_UI_HOST=0.0.0.0 LYRA_UI_PORT=8080 ./start_ui.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 启动UI
python3 lyra2_demo_ui.py --host "$HOST" --port "$PORT"
