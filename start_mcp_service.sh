#!/bin/bash

# Mobile Agent MCP 服务启动脚本

echo "=================================="
echo "Mobile Agent MCP Service Launcher"
echo "=================================="
echo ""

# 检查配置文件
if [ ! -f ".config.yaml" ]; then
    echo "❌ 配置文件 .config.yaml 不存在！"
    echo ""
    echo "请执行以下步骤："
    echo "1. 复制模板配置文件："
    echo "   cp config.yaml .config.yaml"
    echo ""
    echo "2. 编辑 .config.yaml 并填写："
    echo "   - mcp_endpoint: 你的 MCP WebSocket 端点"
    echo "   - api_key: 你的视觉模型 API Key"
    echo ""
    exit 1
fi

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装！"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python -c "import yaml, fastmcp, phone_agent" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  部分依赖未安装，正在安装..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败！"
        exit 1
    fi
fi

# 检查 ADB
echo "🔌 检查 ADB..."
if ! command -v adb &> /dev/null; then
    echo "⚠️  ADB 未安装或不在 PATH 中"
    echo "请参考 README.md 安装 ADB"
else
    # 检查设备连接
    DEVICE_COUNT=$(adb devices | grep -c "device$")
    if [ "$DEVICE_COUNT" -eq 0 ]; then
        echo "⚠️  未检测到已连接的设备"
        echo "请确保："
        echo "  1. 设备已通过 USB 连接"
        echo "  2. 已开启 USB 调试"
        echo "  3. 已授权电脑调试"
        echo ""
        echo "或使用 WiFi 连接："
        echo "  python main.py --connect <IP>:5555"
        echo ""
    else
        echo "✅ 检测到 $DEVICE_COUNT 个已连接的设备"
    fi
fi

echo ""
echo "🚀 启动 Mobile Agent MCP 服务..."
echo ""

# 启动服务
python mcp_pipe.py mobile_agent_server.py

# 捕获退出信号
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 服务启动失败！"
    echo "请查看上方错误信息并修正"
    exit 1
fi
