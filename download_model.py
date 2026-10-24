from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parent
from runtime_paths import MODELS_DIR
MLX_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MLX_MODEL_DIR = MODELS_DIR / MLX_MODEL_ID.split("/")[-1]
GGUF_MODEL_ID = "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF"
GGUF_FILENAME = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
GGUF_MODEL_DIR = MODELS_DIR / "Qwen3-4B-Instruct-2507-GGUF"
FAST_GGUF_MODEL_ID = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
FAST_GGUF_FILENAME = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
FAST_GGUF_MODEL_DIR = MODELS_DIR / "Qwen2.5-1.5B-Instruct-GGUF"
FAST_MLX_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
FAST_MLX_MODEL_DIR = MODELS_DIR / FAST_MLX_MODEL_ID.split("/")[-1]
NLLB_600M_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_600M_MODEL_DIR = MODELS_DIR / "nllb-200-distilled-600M-ja-zh"
NLLB_1_3B_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
NLLB_1_3B_MODEL_DIR = MODELS_DIR / "nllb-200-distilled-1.3B"


def _has_snapshot(model_dir: Path) -> bool:
    weights = list(model_dir.glob("*.safetensors")) + list(
        model_dir.glob("pytorch_model*.bin")
    )
    base_ready = (
        (model_dir / "config.json").is_file()
        and (model_dir / "tokenizer_config.json").is_file()
        and bool(weights)
    )
    if not base_ready:
        return False
    for index_name in (
        "model.safetensors.index.json", "pytorch_model.bin.index.json"
    ):
        index_path = model_dir / index_name
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))[
                "weight_map"
            ]
            return all((model_dir / name).is_file() for name in set(weight_map.values()))
        except (OSError, KeyError, json.JSONDecodeError):
            return False
    return True


def _download_snapshot(
    model_id: str, model_dir: Path, display_name: str, approximate_size: str
) -> None:
    if _has_snapshot(model_dir):
        print(f"✓ {display_name} 已存在：{model_dir}", flush=True)
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"↓ 正在下载 {display_name}（约 {approximate_size}）", flush=True)
    print(f"  保存位置：{model_dir}", flush=True)
    print("  下方会持续显示文件下载进度，大模型可能需要较长时间。", flush=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        ignore_patterns=["*.md", ".gitattributes", "*.h5", "*.msgpack"],
    )
    if not _has_snapshot(model_dir):
        raise RuntimeError(f"{display_name} 下载结束，但模型文件不完整")
    print(f"✓ {display_name} 下载完成。", flush=True)


def _download_gguf(
    model_id: str, filename: str, model_dir: Path,
    display_name: str, approximate_size: str,
) -> None:
    target = model_dir / filename
    if target.is_file():
        print(f"✓ {display_name} 已存在：{target}", flush=True)
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"↓ 正在下载 {display_name}（约 {approximate_size}）", flush=True)
    print(f"  保存位置：{target}", flush=True)
    print("  下方会持续显示文件下载进度，请不要关闭窗口。", flush=True)
    path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=model_dir,
    )
    if not target.is_file():
        raise RuntimeError(f"{display_name} 下载结束，但未找到 {filename}")
    print(f"✓ {display_name} 下载完成：{path}", flush=True)


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=(
            "mlx", "mlx-fast", "cpu", "cpu-fast",
            "nllb-600m", "nllb-1.3b", "all",
        ),
        default="mlx",
    )
    args = parser.parse_args(arguments)

    if args.backend in {"mlx", "all"}:
        _download_snapshot(
            MLX_MODEL_ID, MLX_MODEL_DIR, "Qwen3 4B MLX 翻译模型", "2.2 GB"
        )

    if args.backend in {"cpu", "all"}:
        _download_gguf(
            GGUF_MODEL_ID, GGUF_FILENAME, GGUF_MODEL_DIR,
            "Qwen3 4B GGUF 翻译模型", "2.4 GB",
        )

    if args.backend in {"cpu-fast", "all"}:
        _download_gguf(
            FAST_GGUF_MODEL_ID, FAST_GGUF_FILENAME, FAST_GGUF_MODEL_DIR,
            "Qwen2.5 1.5B GGUF 翻译模型", "940 MB",
        )

    if args.backend in {"mlx-fast", "all"}:
        _download_snapshot(
            FAST_MLX_MODEL_ID, FAST_MLX_MODEL_DIR,
            "Qwen2.5 1.5B MLX 翻译模型", "850 MB",
        )

    if args.backend in {"nllb-600m", "all"}:
        _download_snapshot(
            NLLB_600M_MODEL_ID, NLLB_600M_MODEL_DIR,
            "NLLB 600M 翻译模型", "2.4 GB",
        )

    if args.backend in {"nllb-1.3b", "all"}:
        _download_snapshot(
            NLLB_1_3B_MODEL_ID, NLLB_1_3B_MODEL_DIR,
            "NLLB 1.3B 翻译模型", "5.2 GB",
        )


if __name__ == "__main__":
    main()
