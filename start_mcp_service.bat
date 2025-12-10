@echo off
REM Mobile Agent MCP 服务启动脚本 (Windows)

echo ==================================
echo Mobile Agent MCP Service Launcher
echo ==================================
echo.

REM 检查配置文件
if not exist ".config.yaml" (
    echo ❌ 配置文件 .config.yaml 不存在！
    echo.
    echo 请执行以下步骤：
    echo 1. 复制模板配置文件：
    echo    copy config.yaml .config.yaml
    echo.
    echo 2. 编辑 .config.yaml 并填写：
    echo    - mcp_endpoint: 你的 MCP WebSocket 端点
    echo    - api_key: 你的视觉模型 API Key
    echo.
    pause
    exit /b 1
)

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装！
    pause
    exit /b 1
)

REM 检查依赖
echo 📦 检查依赖...
python -c "import yaml, fastmcp, phone_agent" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  部分依赖未安装，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败！
        pause
        exit /b 1
    )
)

REM 检查 ADB
echo 🔌 检查 ADB...
where adb >nul 2>&1
if errorlevel 1 (
    echo ⚠️  ADB 未安装或不在 PATH 中
    echo 请参考 README.md 安装 ADB
) else (
    REM 检查设备连接
    for /f "tokens=*" %%i in ('adb devices ^| find /c "device"') do set DEVICE_COUNT=%%i
    if %DEVICE_COUNT% EQU 1 (
        echo ⚠️  未检测到已连接的设备
        echo 请确保：
        echo   1. 设备已通过 USB 连接
        echo   2. 已开启 USB 调试
        echo   3. 已授权电脑调试
        echo.
        echo 或使用 WiFi 连接：
        echo   python main.py --connect ^<IP^>:5555
        echo.
    ) else (
        echo ✅ 检测到已连接的设备
    )
)

echo.
echo 🚀 启动 Mobile Agent MCP 服务...
echo.

REM 设置 UTF-8 编码
set PYTHONIOENCODING=utf-8

REM 启动服务
python mcp_pipe.py mobile_agent_server.py

if errorlevel 1 (
    echo.
    echo ❌ 服务启动失败！
    echo 请查看上方错误信息并修正
    pause
    exit /b 1
)
