from __future__ import annotations

import json
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from runtime_paths import CONFIG_FILE, DATA_DIR, MODELS_DIR, RESOURCE_DIR

ROOT = RESOURCE_DIR
SENSEVOICE_DIR = MODELS_DIR / "SenseVoiceSmall"
WTP_MODEL_DIR = MODELS_DIR / "wtp-bert-mini"


def _snapshot_ready(model_dir: Path) -> bool:
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


def _translation_model_spec(cfg: dict) -> tuple[str, str, Path, Callable[[], bool]]:
    dev = cfg.get("device", "auto")
    model_kind = cfg.get("translation_model", "quality")
    use_mlx = (
        sys.platform == "darwin"
        and (
            dev == "mps"
            or (dev == "auto" and platform.machine() == "arm64")
        )
    )

    if model_kind == "fast" and use_mlx:
        model_dir = MODELS_DIR / "Qwen2.5-1.5B-Instruct-4bit"
        return "Qwen2.5 1.5B MLX 翻译模型", "约 850 MB", model_dir, lambda: _snapshot_ready(model_dir)
    if model_kind == "quality" and use_mlx:
        model_dir = MODELS_DIR / "Qwen3-4B-Instruct-2507-4bit"
        return "Qwen3 4B MLX 翻译模型", "约 2.2 GB", model_dir, lambda: _snapshot_ready(model_dir)
    if model_kind == "fast":
        target = MODELS_DIR / "Qwen2.5-1.5B-Instruct-GGUF" / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        return "Qwen2.5 1.5B GGUF 翻译模型", "约 940 MB", target, target.is_file
    if model_kind == "quality":
        target = MODELS_DIR / "Qwen3-4B-Instruct-2507-GGUF" / "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        return "Qwen3 4B GGUF 翻译模型", "约 2.4 GB", target, target.is_file
    if model_kind == "nllb-1.3b":
        model_dir = MODELS_DIR / "nllb-200-distilled-1.3B"
        return "NLLB 1.3B 翻译模型", "约 5.2 GB", model_dir, lambda: _snapshot_ready(model_dir)
    model_dir = MODELS_DIR / "nllb-200-distilled-600M-ja-zh"
    return "NLLB 600M 翻译模型", "约 2.4 GB", model_dir, lambda: _snapshot_ready(model_dir)


def _translation_download_backend(cfg: dict) -> str:
    dev = cfg.get("device", "auto")
    model_kind = cfg.get("translation_model", "quality")
    use_mlx = (
        sys.platform == "darwin"
        and (
            dev == "mps"
            or (dev == "auto" and platform.machine() == "arm64")
        )
    )
    if model_kind == "nllb-600m":
        return "nllb-600m"
    if model_kind == "nllb-1.3b":
        return "nllb-1.3b"
    if model_kind == "fast":
        return "mlx-fast" if use_mlx else "cpu-fast"
    return "mlx" if use_mlx else "cpu"


def detect_system_hardware() -> dict:
    """自动检测系统平台与显卡加速硬件"""
    os_name = platform.system()
    result = {
        "os": os_name,
        "is_mac": os_name == "Darwin",
        "is_win": os_name == "Windows",
        "is_linux": os_name == "Linux",
        "apple_silicon": False,
        "has_cuda": False,
        "cuda_name": "",
        "has_amd": False,
        "amd_name": "",
        "recommended": "cpu",
    }

    # 1. 检测 Apple Silicon (Mac MPS/MLX)
    if result["is_mac"]:
        if platform.machine() == "arm64":
            result["apple_silicon"] = True
            result["recommended"] = "mps"
            return result

    # 2. 检测 NVIDIA CUDA
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            # 某些 Linux ROCm 环境下 torch.cuda 会显示 AMD 显卡名
            if any(k in device_name.lower() for k in ["amd", "radeon"]):
                result["has_amd"] = True
                result["amd_name"] = device_name
                result["recommended"] = "rocm"
            else:
                result["has_cuda"] = True
                result["cuda_name"] = device_name
                result["recommended"] = "cuda"
            return result
    except Exception:
        pass

    # 3. 命令行补充检测 NVIDIA
    if shutil.which("nvidia-smi"):
        result["has_cuda"] = True
        result["recommended"] = "cuda"
        return result

    # 4. 补充检测 AMD GPU (Windows / Linux)
    if result["is_win"]:
        try:
            cmd = "powershell -Command \"Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name\""
            out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                line_lower = line.lower()
                if "radeon" in line_lower or "amd" in line_lower:
                    result["has_amd"] = True
                    result["amd_name"] = line.strip()
                    result["recommended"] = "rocm"
                    break
                elif "nvidia" in line_lower or "geforce" in line_lower:
                    result["has_cuda"] = True
                    result["cuda_name"] = line.strip()
                    result["recommended"] = "cuda"
                    break
        except Exception:
            pass
    elif result["is_linux"]:
        if shutil.which("rocminfo") or shutil.which("rocm-smi"):
            result["has_amd"] = True
            result["recommended"] = "rocm"

    return result


def load_config() -> dict:
    if CONFIG_FILE.is_file():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(cfg: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置文件失败：{e}", file=sys.stderr)


def first_time_wizard(hw: dict) -> dict:
    print("=" * 60)
    print("       实时英日翻译器 · 首次启动硬件配置向导")
    print("=" * 60)
    print(f"操作系统: {hw['os']} ({platform.machine()})")
    
    detected_desc = []
    if hw["apple_silicon"]:
        detected_desc.append("Apple Silicon 芯片 (支持 Metal / MLX 极速加速)")
    if hw["has_cuda"]:
        detected_desc.append(f"NVIDIA 显卡: {hw['cuda_name'] or 'CUDA 支持'}")
    if hw["has_amd"]:
        detected_desc.append(f"AMD 显卡: {hw['amd_name'] or 'ROCm/Vulkan 支持'}")
    if not detected_desc:
        detected_desc.append("通用 CPU 计算")
        
    print(f"检测到的硬件环境: {', '.join(detected_desc)}")
    print(f"系统推荐运行设备: 【{hw['recommended'].upper()}】")
    print("-" * 60)
    print("请选择你想使用的运算加速设备：")
    print(f"  [1] 自动推荐 ({hw['recommended'].upper()}) （直接回车）")
    print("  [2] NVIDIA 显卡 (CUDA)")
    print("  [3] AMD 显卡 (ROCm / Vulkan)")
    print("  [4] Apple 芯片 (Metal / MLX)")
    print("  [5] CPU 纯处理器模式")
    
    choice = input("\n请输入选项编号 [默认 1]: ").strip()
    device_map = {
        "1": hw["recommended"],
        "": hw["recommended"],
        "2": "cuda",
        "3": "rocm",
        "4": "mps",
        "5": "cpu",
    }
    selected_device = device_map.get(choice, hw["recommended"])

    print("-" * 60)
    print("请选择翻译模型（响应速度与翻译精度的平衡）：")
    print("  [1] 高质量版 (Qwen3 4B)    - 默认模型，语境理解更丰富（直接回车）")
    print("  [2] 极速版 (Qwen2.5 1.5B)  - 更低延迟和内存占用")
    print("  [3] 经典翻译 (NLLB 600M)   - 专用神经翻译模型")
    
    m_choice = input("\n请输入选项编号 [默认 1]: ").strip()
    model_map = {
        "1": "quality",
        "": "quality",
        "2": "fast",
        "3": "nllb-600m",
    }
    selected_model = model_map.get(m_choice, "quality")

    cfg = {
        "device": selected_device,
        "translation_model": selected_model,
    }
    save_config(cfg)
    print("=" * 60)
    print(f"配置已保存！设备: {selected_device} | 模型: {selected_model}")
    print("=" * 60)
    return cfg


def ensure_models(cfg: dict, announce: bool = True) -> None:
    """检查并自动下载当前配置需要的全部模型。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    translation_name, translation_size, translation_path, translation_ready = (
        _translation_model_spec(cfg)
    )
    tasks = [
        (
            "SenseVoiceSmall 语音识别模型", "约 900 MB",
            SENSEVOICE_DIR,
            lambda: (SENSEVOICE_DIR / "model.pt").is_file(),
            ("download_asr", []),
        ),
        (
            "WtP 语义断句模型", "约 16 MB", WTP_MODEL_DIR,
            lambda: all((WTP_MODEL_DIR / name).is_file() for name in (
                "config.json", "model.onnx", "mixtures.skops"
            )),
            ("download_sentence_model", []),
        ),
        (
            translation_name, translation_size, translation_path,
            translation_ready,
            ("download_model", ["--backend", _translation_download_backend(cfg)]),
        ),
    ]
    missing = [task for task in tasks if not task[3]()]
    if not missing and not announce:
        return

    print("\n" + "=" * 62, flush=True)
    print("模型准备检查（首次运行会自动下载）", flush=True)
    print("下载可以断点续传；请保持网络连接并不要关闭窗口。", flush=True)
    print("=" * 62, flush=True)

    for index, (name, size, path, ready, download_args) in enumerate(tasks, 1):
        if ready():
            print(f"[{index}/3] ✓ {name}：已就绪", flush=True)
            continue
        print(f"\n[{index}/3] ↓ {name} 需要下载（{size}）", flush=True)
        print(f"      目标：{path}", flush=True)
        try:
            module_name, arguments = download_args
            if getattr(sys, "frozen", False):
                downloader = importlib.import_module(module_name)
                downloader.main(arguments)
            else:
                subprocess.run(
                    [sys.executable, "-m", module_name, *arguments], check=True
                )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"{name} 下载失败（退出码 {exc.returncode}）。"
                "请检查上方错误信息、网络和磁盘空间后重新启动，"
                "已下载部分会自动续传。"
            ) from exc
        if not ready():
            raise RuntimeError(f"{name} 下载后校验失败：模型文件不完整")
        print(f"[{index}/3] ✓ {name}：准备完成", flush=True)

    print("\n✓ 当前配置需要的全部模型已准备完成。", flush=True)


def _effective_config(cfg: dict, arguments: list[str]) -> dict:
    """Apply model/device CLI overrides before deciding what to download."""
    effective = dict(cfg)
    previous = ""
    for argument in arguments:
        if previous == "--device":
            effective["device"] = argument
        elif previous == "--translation-model":
            effective["translation_model"] = argument
        elif argument.startswith("--device="):
            effective["device"] = argument.split("=", 1)[1]
        elif argument.startswith("--translation-model="):
            effective["translation_model"] = argument.split("=", 1)[1]
        previous = argument
    return effective


def main():
    hw = detect_system_hardware()
    cfg = load_config()

    # 如果没有配置或命令行要求重新配置
    if not cfg and getattr(sys, "frozen", False) and "--setup" not in sys.argv:
        cfg = {
            "device": hw["recommended"],
            "translation_model": "quality",
        }
        save_config(cfg)
        print("首次启动：已根据本机硬件选择推荐设备与 Qwen3 4B 翻译模型。", flush=True)
    elif not cfg or "--setup" in sys.argv:
        cfg = first_time_wizard(hw)

    passthrough_args = [a for a in sys.argv[1:] if a != "--setup"]
    effective_cfg = _effective_config(cfg, passthrough_args)

    # 确保当前命令行实际选中的模型齐备
    try:
        ensure_models(effective_cfg)
    except Exception as exc:
        print(f"\n[错误] 首次模型准备未完成：{exc}", file=sys.stderr)
        print("重新启动程序即可从已下载部分继续。", file=sys.stderr)
        return

    # 组装启动参数
    cmd = [
        sys.executable,
        str(ROOT / "realtime.py"),
        "--device", effective_cfg.get("device", "auto"),
        "--translation-model", effective_cfg.get("translation_model", "quality"),
    ]

    # 透传其他命令行参数（如 --capture-system-audio, --mic 等）
    cmd.extend(passthrough_args)

    print("\n正在启动实时英日翻译器...")
    print("【提示】程序启动后麦克风默认处于关闭状态，准备就绪后点击“开始收音”即可。\n")
    if getattr(sys, "frozen", False):
        from realtime import main as realtime_main

        sys.argv = [sys.argv[0], *cmd[2:]]
        realtime_main()
    else:
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
