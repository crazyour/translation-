from __future__ import annotations

import os
from pathlib import Path

from runtime_paths import MODELS_DIR

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from huggingface_hub import snapshot_download


MODEL_ID = "benjamin/wtp-bert-mini"
MODEL_DIR = MODELS_DIR / "wtp-bert-mini"


def main(arguments: list[str] | None = None) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    required = tuple(MODEL_DIR / name for name in (
        "config.json", "model.onnx", "mixtures.skops"
    ))
    if all(path.is_file() for path in required):
        print(f"✓ WtP 断句模型已存在：{MODEL_DIR}", flush=True)
        return
    print("↓ 正在下载 WtP 语义断句模型（约 16 MB）", flush=True)
    print(f"  保存位置：{MODEL_DIR}", flush=True)
    print("  下方会持续显示文件下载进度。", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=("config.json", "model.onnx", "mixtures.skops"),
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("WtP 下载结束，但模型文件不完整")
    print(f"✓ WtP 断句模型下载完成：{MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
