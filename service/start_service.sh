#!/bin/bash

# 服装属性识别服务启动脚本

# 设置环境变量
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8000"}
export DEVICE=${DEVICE:-"cuda"}
export WORKERS=${WORKERS:-"1"}

echo "================================================"
echo "启动服装属性识别服务"
echo "================================================"
echo "主机: $HOST"
echo "端口: $PORT"
echo "设备: $DEVICE"
echo "Worker数量: $WORKERS"
echo "================================================"

# 切换到服务目录
cd "$(dirname "$0")"

# 检查依赖
if ! python -c "import fastapi" 2>/dev/null; then
    echo "错误: 未安装FastAPI，请运行: pip install -r requirements_service.txt"
    exit 1
fi

# 启动服务
if [ "$WORKERS" -gt 1 ]; then
    # 多worker模式
    uvicorn app:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS"
else
    # 单worker开发模式
    python app.py
fi
