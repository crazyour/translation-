from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch

from runtime_paths import MODELS_DIR


MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_DIR = MODELS_DIR / MODEL_ID.split("/")[-1]
GGUF_MODEL_PATH = (
    MODELS_DIR / "Qwen3-4B-Instruct-2507-GGUF/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)
FAST_GGUF_MODEL_PATH = (
    MODELS_DIR / "Qwen2.5-1.5B-Instruct-GGUF/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)
FAST_MLX_MODEL_DIR = (
    MODELS_DIR / "Qwen2.5-1.5B-Instruct-4bit"
)
SOURCE_LANG = "jpn_Jpan"
TARGET_LANG = "zho_Hans"
MAX_OUTPUT_CHARS = 500

SYSTEM_PROMPT = (
    "你是一名实时同传翻译。输入可能是日语、英语，也可能两种语言混合。"
    "请自动识别并只输出简明准确的单行中文译文，不要任何解释或原文。"
)

def _generate_chat(messages, tokenizer, model, max_new_tokens: int) -> str:
    if tokenizer is None:
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        return response["choices"][0]["message"]["content"] or ""

    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return mlx_generate(
        model, tokenizer, prompt=prompt, verbose=False,
        max_tokens=max_new_tokens,
        sampler=make_sampler(temp=0.0),
    )


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            print("未检测到有效 CUDA / ROCm 设备，自动回退到 CPU", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前设备不支持 Apple MPS，请改用 --device cpu")
    return torch.device(device_name)


def load_model(model_dir: Path, device: torch.device):
    is_gguf = model_dir.suffix == ".gguf" or not (model_dir / "config.json").is_file()
    if is_gguf or device.type == "cpu":
        gguf_path = model_dir if model_dir.suffix == ".gguf" else (
            FAST_GGUF_MODEL_PATH if "1.5B" in str(model_dir) else GGUF_MODEL_PATH
        )
        if not gguf_path.is_file():
            raise FileNotFoundError(
                f"GGUF 模型不存在：{gguf_path}\n"
                "请先运行：python download_model.py --backend cpu-fast"
            )
        from llama_cpp import Llama

        threads = max(1, (os.cpu_count() or 4) - 2)
        gpu_layers = -1 if device.type == "cuda" else 0
        model = Llama(
            model_path=str(gguf_path),
            n_ctx=2048,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        return None, model

    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"模型不存在：{model_dir}\n请先运行：python download_model.py"
        )
    if sys.platform == "darwin":
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(str(model_dir))
        return tokenizer, model
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        return tokenizer, model


def translate(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
    context: str = "",
) -> str:
    text = text.strip()[-max_input_length:]
    context = context.strip()
    if context:
        user_content = f"上下文：\n{context}\n\n当前发言：\n{text}"
    else:
        user_content = text
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    output = _generate_chat(
        messages, tokenizer, model, min(max_new_tokens, 128)
    )
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.S)
    output = re.sub(r"^(译文|翻译|中文)\s*[：:]\s*", "", output.strip())
    output = " ".join(output.splitlines()).strip()
    return output[:MAX_OUTPUT_CHARS]


def interactive(
    tokenizer,
    model,
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
) -> None:
    print("日文或英文输入翻译为中文，输入 q、quit 或 exit 退出。")
    while True:
        try:
            text = input("日文> ").strip()
        except EOFError:
            print()
            return
        if text.lower() in {"q", "quit", "exit"}:
            return
        if not text:
            continue
        try:
            print(f"中文> {translate(text, tokenizer, model, device, max_input_length, max_new_tokens)}")
        except Exception as error:
            print(f"翻译失败：{error}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Qwen 进行日语/英语译中文命令行翻译")
    parser.add_argument("text", nargs="?", help="一次性输入日文或英文；省略时进入交互模式")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR,
        help="本地模型目录",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="cpu",
        help="推理设备，默认为 cpu",
    )
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.max_input_length < 1 or args.max_new_tokens < 1:
        parser.error("长度参数必须大于 0")

    device = choose_device(args.device)
    active_model = (
        args.model if device.type == "cpu" and args.model.suffix == ".gguf"
        else GGUF_MODEL_PATH if device.type == "cpu"
        else args.model
    )
    print(f"正在加载模型：{active_model}", file=sys.stderr)
    tokenizer, model = load_model(args.model, device)
    backend = "CPU / llama.cpp" if device.type == "cpu" else "MLX / Apple Silicon"
    print(f"推理设备：{backend}", file=sys.stderr)

    if args.text is not None:
        print(translate(args.text, tokenizer, model, device, args.max_input_length, args.max_new_tokens))
        return

    interactive(tokenizer, model, device, args.max_input_length, args.max_new_tokens)


if __name__ == "__main__":
    main()
