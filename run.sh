#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"

if [ ! -x .venv/bin/python ]; then
  uv venv --python python3 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt

if [ ! -f .venv/.llama-cpp-cpu-only ]; then
  uv pip install --python .venv/bin/python pip
  .venv/bin/python install_llama_cpu.py
  touch .venv/.llama-cpp-cpu-only
fi

if [ ! -f models/Qwen3-4B-Instruct-2507-GGUF/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf ]; then
  .venv/bin/python download_model.py --backend cpu
fi

exec .venv/bin/python translate.py "$@"
