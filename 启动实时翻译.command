#!/bin/bash
cd "$(dirname "$0")"

echo "==================================================="
echo "          实时英日翻译器 · 启动中..."
echo "==================================================="

# 确保有可执行虚拟环境
if [ ! -x .venv/bin/python ]; then
    echo "[首次运行] 正在初始化虚拟环境..."
    if which uv >/dev/null 2>&1; then
        uv venv --python python3 .venv
        uv pip install --python .venv/bin/python -r requirements.txt
    else
        python3 -m venv .venv
        .venv/bin/pip install --upgrade pip
        .venv/bin/pip install -r requirements.txt
    fi
fi

# 项目更新后，已存在的虚拟环境也可能缺少新的断句依赖。
if ! .venv/bin/python -c "import wtpsplit" >/dev/null 2>&1; then
    echo "[准备] 正在安装 WtP 断句依赖..."
    if which uv >/dev/null 2>&1; then
        uv pip install --python .venv/bin/python -r requirements.txt
    else
        .venv/bin/pip install -r requirements.txt
    fi
fi

# 启动向导与主程序
.venv/bin/python launcher.py "$@"
