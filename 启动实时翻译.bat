@echo off
chcp 65001 >nul
title 实时英日翻译器

echo ===================================================
echo           实时英日翻译器 · 启动中...
echo ===================================================

cd /d "%~dp0"

REM 检查是否有 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 系统中未找到 Python！请先安装 Python 3.10 或更高版本。
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查或创建虚拟环境
if not exist ".venv\Scripts\python.exe" (
    echo [首次运行] 正在创建 Python 虚拟环境并安装依赖...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

REM 项目更新后补齐 WtP 断句依赖
python -c "import wtpsplit" >nul 2>nul
if %errorlevel% neq 0 (
    echo [准备] 正在安装 WtP 断句依赖...
    pip install -r requirements.txt
)

REM 启动引导器
.venv\Scripts\python.exe launcher.py %*

if %errorlevel% neq 0 (
    echo.
    echo 程序发生异常退出。
    pause
)
