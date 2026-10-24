from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from pathlib import Path

from runtime_paths import MODELS_DIR

from funasr import AutoModel


MODEL_ID = "iic/SenseVoiceSmall"
MODEL_DIR = MODELS_DIR / "SenseVoiceSmall"


def main(arguments: list[str] | None = None) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (MODEL_DIR / "model.pt").exists():
        print(f"✓ SenseVoiceSmall 已存在：{MODEL_DIR}", flush=True)
        return
    print(f"↓ 正在下载 SenseVoiceSmall 语音识别模型（约 900 MB）", flush=True)
    print(f"  保存位置：{MODEL_DIR}", flush=True)
    print("  下方会持续显示文件下载进度，请不要关闭窗口。", flush=True)
    try:
        from modelscope import snapshot_download
        snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR))
    except Exception as e:
        print(f"  ModelScope 下载失败 ({e})，正在尝试备用方式…", flush=True)
        AutoModel(
            model=MODEL_ID,
            cache_dir=str(MODEL_DIR.parent),
        )
    if not (MODEL_DIR / "model.pt").is_file():
        raise RuntimeError("SenseVoiceSmall 下载结束，但未找到 model.pt")
    print("✓ SenseVoiceSmall 下载完成。", flush=True)


if __name__ == "__main__":
    main()
