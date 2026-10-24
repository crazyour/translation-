#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"

if [ ! -x .venv/bin/python ]; then
  uv venv --python python3 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt

translation_backend=mlx
translation_model=quality
previous_arg=
for arg in "$@"; do
  if [ "$previous_arg" = "--device" ] && [ "$arg" = "cpu" ]; then
    translation_backend=cpu
  fi
  if [ "$previous_arg" = "--translation-model" ]; then
    translation_model=$arg
  fi
  case "$arg" in
    --device=cpu) translation_backend=cpu ;;
    --translation-model=*) translation_model=${arg#--translation-model=} ;;
  esac
  previous_arg=$arg
done

case "$translation_model" in
  nllb-600m|nllb-1.3b) needs_qwen_backend=false ;;
  *) needs_qwen_backend=true ;;
esac

if [ "$translation_backend" = "cpu" ] && [ "$needs_qwen_backend" = true ]; then
  if [ ! -f .venv/.llama-cpp-cpu-only ]; then
    uv pip install --python .venv/bin/python pip
    .venv/bin/python install_llama_cpu.py
    touch .venv/.llama-cpp-cpu-only
  fi
fi

# realtime.py performs one unified, validated first-run download for every
# required model and shows item-by-item progress before opening the window.
exec .venv/bin/python realtime.py "$@"
