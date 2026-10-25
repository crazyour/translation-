#!/bin/sh
set -eu

mode=${1:-realtime}
if [ "$#" -gt 0 ]; then
    shift
fi

case "$mode" in
    realtime)
        exec python realtime.py \
            --device cpu \
            --translation-model quality \
            "$@"
        ;;
    translate)
        model_path=/app/models/Qwen3-4B-Instruct-2507-GGUF/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
        python download_model.py --backend cpu
        exec python translate.py \
            --device cpu \
            --model "$model_path" \
            "$@"
        ;;
    list-mics)
        exec python realtime.py --list-mics
        ;;
    shell)
        exec /bin/sh "$@"
        ;;
    *)
        # Allow advanced users to run normal container commands too.
        exec "$mode" "$@"
        ;;
esac
