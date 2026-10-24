from __future__ import annotations

import argparse
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import tkinter as tk
import torch
from funasr import AutoModel

from runtime_paths import DATA_DIR, MODELS_DIR, RESOURCE_DIR

from translate import (
    FAST_GGUF_MODEL_PATH,
    FAST_MLX_MODEL_DIR,
    MAX_OUTPUT_CHARS,
    MODEL_DIR as NLLB_MODEL_DIR,
    SOURCE_LANG,
    TARGET_LANG,
    choose_device,
    load_model as load_nllb,
    translate as nllb_translate,
)


SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall"
SENSEVOICE_MODEL_DIR = MODELS_DIR / "SenseVoiceSmall"
WTP_MODEL_DIR = MODELS_DIR / "wtp-bert-mini"
HELP_DOCUMENT_PATH = RESOURCE_DIR / "设置说明.md"
NLLB_600M_MODEL_DIR = (
    MODELS_DIR / "nllb-200-distilled-600M-ja-zh"
)
NLLB_1_3B_MODEL_DIR = (
    MODELS_DIR / "nllb-200-distilled-1.3B"
)
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_MS = 100
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

# Endpointing is deliberately a little patient: Japanese speakers often pause
# mid-sentence, so a short silence must not immediately become a new subtitle.
ENERGY_THRESHOLD = 0.002
MIN_SPEECH_MS = 400
END_SILENCE_MS = 800
MAX_SEGMENT_S = 5
MAX_SENTENCE_CHARS = 60
PRE_ROLL_MS = 300

TAG_RE = re.compile(r"<\|[^|]+?\|>")
SENTENCE_END_RE = re.compile(r"[。！？!?\.]")
UNTRUSTED_ASR_PUNCTUATION = frozenset("。！？!?.,、，；;：:")
JAPANESE_DANGLING_END_RE = re.compile(
    r"(?:"
    r"について(?:は|も|ですが)?|"
    r"までに|ですが|ますが|だが|ですけど|"
    r"ますけど|けど|けれど|けれども|ので|のに|"
    r"ため(?:に)?|場合|なら|としたら|によると|"
    r"ことを|ものを|とき|時|"
    r"という|といった|ような|ように|ながら|つつ|"
    r"は|が|を|に|へ|で|と|も|の|や|から|より"
    r")$"
)

# FunASR and Transformers both manipulate PyTorch model-loading state. Loading
# them concurrently can leave a module on the meta device, so initialization
# must be serialized even though inference runs in separate worker threads.
MODEL_LOAD_LOCK = threading.Lock()


@dataclass
class UiEvent:
    kind: str
    text: str = ""
    source: str = ""


@dataclass
class TranslationRequest:
    kind: str = "append"
    text: str = ""


def _join_transcript(current: str, addition: str) -> str:
    current = current.rstrip()
    addition = addition.strip()
    if not current:
        return addition
    if not addition:
        return current
    # Preserve the ASR chunk boundary as a soft hint. This is essential for
    # unpunctuated Japanese: concatenating two chunks without whitespace turns
    # e.g. "...ください" + "それから..." into an ambiguous character run.
    # The organizer may still join across this newline when the first chunk is
    # semantically incomplete, and prefix validation ignores whitespace.
    return current + "\n" + addition


def _content_characters(text: str) -> str:
    return "".join(
        char for char in text
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "Z"))
    )


def _validated_source_prefix(raw: str, organized: str):
    """Accept punctuation edits only and return (display source, remainder)."""
    target = _content_characters(organized)
    if not target:
        return None
    raw_positions = [
        (char, index) for index, char in enumerate(raw)
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "Z"))
    ]
    raw_content = "".join(char for char, _ in raw_positions)
    if not raw_content.startswith(target):
        return None
    boundary = raw_positions[len(target) - 1][1] + 1
    while boundary < len(raw):
        char = raw[boundary]
        if not char.isspace() and not unicodedata.category(char).startswith(("P", "Z")):
            break
        boundary += 1
    consumed = raw[:boundary]
    if _content_characters(consumed) != target:
        return None
    return organized.strip(), raw[boundary:].lstrip()


def _looks_semantically_incomplete(text: str) -> bool:
    """Reject common Japanese clause/topic endings that still need a predicate."""
    normalized = "".join(
        char for char in text
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "Z"))
    )
    return bool(JAPANESE_DANGLING_END_RE.search(normalized))


def _flatten_transcript(text: str) -> str:
    """Remove internal chunk markers while preserving needed English spaces."""
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts:
        return ""
    flattened = parts[0]
    for part in parts[1:]:
        needs_space = (
            flattened[-1].isascii()
            and part[0].isascii()
            and flattened[-1].isalnum()
            and part[0].isalnum()
        )
        flattened += (" " if needs_space else "") + part
    return flattened


def _contains_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))


def _boundary_text(text: str) -> tuple[str, str, str]:
    """Return punctuation-free model text, language code, and look-ahead text."""
    flattened = _flatten_transcript(text)
    if _contains_japanese(flattened):
        normalized = "".join(
            char for char in flattened
            if not char.isspace() and char not in UNTRUSTED_ASR_PUNCTUATION
        )
        return normalized, "ja", "これは次の文です"

    # Keep word spaces for English, but do not let possibly unstable ASR
    # punctuation dictate sentence boundaries. WtP decides those itself.
    chars = [
        " " if char.isspace() or char in UNTRUSTED_ASR_PUNCTUATION
        else char
        for char in flattened
    ]
    normalized = re.sub(r"\s+", " ", "".join(chars)).strip()
    return normalized, "en", " This is the next sentence"


def _format_source(raw_prefix: str, language: str) -> str:
    text = _flatten_transcript(raw_prefix)
    raw_terminal = re.search(r"([。！？!?\.])\s*$", raw_prefix)
    # Sentence stops emitted in the middle of an ASR chunk are not trusted;
    # WtP established the real boundary independently. Preserve commas,
    # percentages, quotation marks, and all recognized words/numbers.
    text = SENTENCE_END_RE.sub("", text)
    if language == "ja":
        text = "".join(char for char in text if not char.isspace())
    else:
        text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    terminal = (
        raw_terminal.group(1) if raw_terminal
        else ("。" if language == "ja" else ".")
    )
    return text + terminal


class SentenceBoundaryDetector:
    """Extract stable sentences before they are handed to the translator.

    WtP needs a small amount of right context to decide whether the current
    final words form a sentence boundary. The synthetic look-ahead is used
    only for inference and is never displayed or translated.
    """

    def __init__(self, model_dir: Path, max_sentence_chars: int):
        self.model_dir = model_dir
        self.max_sentence_chars = max(
            20, min(200, int(max_sentence_chars))
        )
        self.model = None

    def load(self):
        if not (self.model_dir / "model.onnx").is_file():
            raise FileNotFoundError(
                f"断句模型不存在：{self.model_dir}\n"
                "请先运行：python download_sentence_model.py"
            )
        from wtpsplit import WtP

        self.model = WtP(
            str(self.model_dir),
            ort_providers=["CPUExecutionProvider"],
            ignore_legacy_warning=True,
        )
        return self

    def update_max_length(self, characters: int):
        self.max_sentence_chars = max(20, min(200, int(characters)))

    def _semantic_boundaries(
        self, text: str, language: str, look_ahead: str
    ) -> list[int]:
        augmented = text + look_ahead
        parts = self.model.split(
            augmented,
            lang_code=language,
            style="ud",
            algorithm="viterbi",
        )
        boundaries = []
        offset = 0
        for part in parts[:-1]:
            offset += len(part)
            boundary = offset
            # English WtP keeps the separator space at the end of the
            # preceding sentence. It belongs to the synthetic look-ahead, not
            # to the real buffered text.
            while boundary > 0 and augmented[boundary - 1].isspace():
                boundary -= 1
            if boundary <= len(text):
                boundaries.append(boundary)
        return boundaries

    def _length_boundary(self, text: str, language: str) -> int | None:
        if len(text) <= self.max_sentence_chars:
            return None
        minimum = max(8, self.max_sentence_chars // 3)
        target = max(minimum, round(self.max_sentence_chars * 0.75))
        parts = self.model.split(
            text,
            lang_code=language,
            style="ud",
            min_length=minimum,
            max_length=self.max_sentence_chars,
            prior_type="gaussian",
            prior_kwargs={"target_length": target},
            algorithm="viterbi",
        )
        if len(parts) < 2:
            return None
        boundary = len(parts[0])
        return boundary if 0 < boundary < len(text) else None

    def pop_ready(self, raw: str) -> tuple[str, str] | None:
        if self.model is None:
            raise RuntimeError("断句模型尚未加载")
        text, language, look_ahead = _boundary_text(raw)
        if not text:
            return None

        boundaries = self._semantic_boundaries(text, language, look_ahead)
        boundary = boundaries[0] if boundaries else None

        # A semantic sentence may itself be too long for a useful subtitle.
        # In that case, ask the same model to choose a natural earlier split.
        length_input = text[:boundary] if boundary is not None else text
        length_boundary = self._length_boundary(length_input, language)
        if length_boundary is not None:
            boundary = length_boundary
        if boundary is None:
            return None

        candidate = text[:boundary]
        if not candidate:
            return None
        if length_boundary is None and _looks_semantically_incomplete(candidate):
            return None

        validated = _validated_source_prefix(raw, candidate)
        if validated is None:
            return None
        _, remainder = validated
        consumed_length = len(raw) - len(remainder)
        raw_prefix = raw[:consumed_length]
        return _format_source(raw_prefix, language), remainder


class MicSegmenter:
    def __init__(self, on_segment, threshold=ENERGY_THRESHOLD, sr=SAMPLE_RATE,
                 block_ms=BLOCK_MS, device=None, on_activity=None,
                 max_segment_s=MAX_SEGMENT_S, segment_mode="fixed",
                 min_speech_ms=MIN_SPEECH_MS,
                 end_silence_ms=END_SILENCE_MS,
                 pre_roll_ms=PRE_ROLL_MS):
        self.on_segment = on_segment
        self.threshold = threshold
        self.sr = sr
        self.block_ms = block_ms
        self.device = device
        self.on_activity = on_activity
        self.max_segment_s = max_segment_s
        self.segment_mode = segment_mode
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.pre_roll_ms = pre_roll_ms

        self.stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._silent_blocks = 0
        self._speech_blocks = 0
        self._is_speech = False
        self._segment_start_ms = 0
        self._elapsed_ms = 0
        self._pre_roll = deque(maxlen=max(0, pre_roll_ms // block_ms))
        self._activity_blocks = 0

    @property
    def is_running(self) -> bool:
        return self.stream is not None

    def update_settings(self, *, max_segment_s, segment_mode, min_speech_ms,
                        end_silence_ms, pre_roll_ms, threshold):
        with self._lock:
            self.max_segment_s = max_segment_s
            self.segment_mode = segment_mode
            self.min_speech_ms = min_speech_ms
            self.end_silence_ms = end_silence_ms
            self.pre_roll_ms = pre_roll_ms
            self.threshold = threshold
            previous = list(self._pre_roll)
            self._pre_roll = deque(
                previous,
                maxlen=max(0, pre_roll_ms // self.block_ms),
            )

    def start(self):
        if self.stream is not None:
            return
        self._buffer.clear()
        self._silent_blocks = 0
        self._speech_blocks = 0
        self._is_speech = False
        self._elapsed_ms = 0
        self._pre_roll.clear()
        self._activity_blocks = 0
        sd.check_input_settings(
            device=self.device, channels=CHANNELS, samplerate=self.sr,
            dtype="float32",
        )
        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            device=self.device,
            callback=self._on_audio,
        )
        self.stream.start()

    def stop(self):
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None
        self._flush_buffer()

    def _flush_buffer(self):
        segment = None
        with self._lock:
            if self._buffer and self._speech_blocks * self.block_ms >= self.min_speech_ms:
                segment = np.concatenate(self._buffer)
            self._buffer.clear()
            self._is_speech = False
            self._silent_blocks = 0
            self._speech_blocks = 0
        if segment is not None:
            threading.Thread(
                target=self.on_segment, args=(segment,), daemon=True
            ).start()

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            sys.stderr.write(f"[audio] {status}\n")
        audio = indata[:, 0].astype(np.float32).copy()
        self.feed(audio)

    def feed(self, audio: np.ndarray):
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        self._activity_blocks += 1
        if self.on_activity and self._activity_blocks % max(1, 500 // self.block_ms) == 0:
            self.on_activity(rms)
        block_ms = self.block_ms
        completed_segment = None
        with self._lock:
            self._elapsed_ms += block_ms
            if rms >= self.threshold:
                self._speech_blocks += 1
                self._silent_blocks = 0
                if not self._is_speech:
                    self._is_speech = True
                    # Keep a little audio before the threshold crossing. This
                    # prevents quiet consonants and the first syllable from
                    # being clipped by the energy gate.
                    self._buffer.extend(self._pre_roll)
                    self._pre_roll.clear()
                    # The selected cutoff is measured from the first detected
                    # speech block.  Pre-roll protects the first syllable but
                    # must not make a 5-second setting expire 300 ms early.
                    self._segment_start_ms = self._elapsed_ms - block_ms
                self._buffer.append(audio)
                cur_len_ms = self._elapsed_ms - self._segment_start_ms
                if cur_len_ms >= self.max_segment_s * 1000:
                    completed_segment = np.concatenate(self._buffer)
                    self._buffer.clear()
                    self._is_speech = False
                    self._silent_blocks = 0
                    self._speech_blocks = 0
            else:
                if self._is_speech:
                    self._buffer.append(audio)
                else:
                    self._pre_roll.append(audio)
                self._silent_blocks += 1
                speech_ms = self._speech_blocks * block_ms
                silence_ms = self._silent_blocks * block_ms
                cur_len_ms = self._elapsed_ms - self._segment_start_ms
                reached_max = cur_len_ms >= self.max_segment_s * 1000
                if self.segment_mode == "smart":
                    should_end = speech_ms >= self.min_speech_ms and (
                        silence_ms >= self.end_silence_ms or reached_max
                    )
                else:
                    # Fixed mode does not finish early just because the
                    # speaker pauses. Stopping capture still flushes the rest.
                    should_end = speech_ms >= self.min_speech_ms and reached_max
                if should_end and self._buffer:
                    completed_segment = np.concatenate(self._buffer)
                    self._buffer.clear()
                    self._is_speech = False
                    self._silent_blocks = 0
                    self._speech_blocks = 0
        if completed_segment is not None:
            threading.Thread(
                target=self.on_segment,
                args=(completed_segment.copy(),),
                daemon=True,
            ).start()


class SystemAudioSegmenter:
    def __init__(self, on_segment, on_activity=None, on_error=None,
                 max_segment_s=MAX_SEGMENT_S, segment_mode="fixed",
                 min_speech_ms=MIN_SPEECH_MS,
                 end_silence_ms=END_SILENCE_MS,
                 pre_roll_ms=PRE_ROLL_MS,
                 threshold=ENERGY_THRESHOLD):
        self.processor = MicSegmenter(
            on_segment, on_activity=on_activity,
            max_segment_s=max_segment_s, segment_mode=segment_mode,
            min_speech_ms=min_speech_ms,
            end_silence_ms=end_silence_ms,
            pre_roll_ms=pre_roll_ms, threshold=threshold,
        )
        self.on_error = on_error
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.error_thread: threading.Thread | None = None

    @property
    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        if platform.system() != "Darwin":
            raise RuntimeError("当前原生系统音频采集仅支持 macOS；Windows 需使用 WASAPI Loopback")
        source = RESOURCE_DIR / "system_audio_capture.m"
        build_dir = DATA_DIR / ".build"
        executable = build_dir / "system_audio_capture"
        build_dir.mkdir(exist_ok=True)
        if not executable.exists() or executable.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(
                [
                    "clang", "-fobjc-arc", str(source), "-o", str(executable),
                    "-framework", "Foundation",
                    "-framework", "ScreenCaptureKit",
                    "-framework", "CoreMedia",
                    "-framework", "CoreAudio",
                ],
                check=True,
            )
        self.process = subprocess.Popen(
            [str(executable)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.thread = threading.Thread(target=self._read_audio, daemon=True)
        self.thread.start()
        self.error_thread = threading.Thread(target=self._read_errors, daemon=True)
        self.error_thread.start()

    def _read_errors(self):
        process = self.process
        if process is None or process.stderr is None:
            return
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("ERROR:") and self.on_error:
                self.on_error(line.removeprefix("ERROR:").strip())

    def _read_audio(self):
        assert self.process is not None and self.process.stdout is not None
        pending = b""
        frame_bytes = BLOCK_SIZE * np.dtype(np.float32).itemsize
        while self.is_running:
            chunk = self.process.stdout.read(frame_bytes)
            if not chunk:
                break
            pending += chunk
            while len(pending) >= frame_bytes:
                frame, pending = pending[:frame_bytes], pending[frame_bytes:]
                self.processor.feed(np.frombuffer(frame, dtype=np.float32).copy())

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self.processor._flush_buffer()


def get_translation_path(model_kind: str, device: torch.device) -> Path:
    use_mlx = sys.platform == "darwin" and device.type == "mps"
    if model_kind == "fast":
        return FAST_MLX_MODEL_DIR if use_mlx else FAST_GGUF_MODEL_PATH
    if model_kind == "quality":
        return NLLB_MODEL_DIR if use_mlx else FAST_GGUF_MODEL_PATH
    if model_kind == "nllb-600m":
        return NLLB_600M_MODEL_DIR
    if model_kind == "nllb-1.3b":
        return NLLB_1_3B_MODEL_DIR
    return FAST_MLX_MODEL_DIR if use_mlx else FAST_GGUF_MODEL_PATH


class AsrWorker(threading.Thread):
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue, device: str = "cpu"):
        super().__init__(daemon=True, name="asr")
        self.in_q = in_q
        self.out_q = out_q
        self.device = device
        self.model: AutoModel | None = None

    def run(self):
        def _load(dev: str):
            os.environ.setdefault("MODELSCOPE_CACHE", str(SENSEVOICE_MODEL_DIR))
            with MODEL_LOAD_LOCK:
                return AutoModel(
                    model=str(SENSEVOICE_MODEL_DIR),
                    quantize=False,
                    disable_update=True,
                    device=dev,
                )

        try:
            self.out_q.put(UiEvent("status", f"ASR 加载中…（{self.device}）"))
            self.model = _load(self.device)
            self.out_q.put(UiEvent("asr_device", self.device))
            self.out_q.put(UiEvent("status", f"ASR 就绪（{self.device}）"))
        except Exception as e:
            if self.device != "cpu":
                self.out_q.put(UiEvent(
                    "status", f"ASR {self.device} 失败，回退 CPU…"))
                try:
                    self.model = _load("cpu")
                    self.device = "cpu"
                    self.out_q.put(UiEvent("asr_device", self.device))
                    self.out_q.put(UiEvent("status", "ASR 就绪（cpu）"))
                except Exception as e2:
                    self.out_q.put(UiEvent("error", f"ASR 加载失败：{e2}"))
                    return
            else:
                self.out_q.put(UiEvent("error", f"ASR 加载失败：{e}"))
                return

        while True:
            item = self.in_q.get()
            if item is None:
                return
            audio: np.ndarray = item
            try:
                result = self.model.generate(
                    input=audio,
                    language="auto",
                    use_itn=True,
                    disable_pbar=True,
                )
                text = ""
                if result:
                    text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
                text = TAG_RE.sub("", text).strip()
                if text:
                    self.out_q.put(UiEvent("asr_text", text))
            except Exception as e:
                self.out_q.put(UiEvent("error", f"识别失败：{e}"))


class TranslateWorker(threading.Thread):
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue, device: torch.device,
                 model_path: Path, model_kind: str = "quality",
                 max_sentence_chars: int = MAX_SENTENCE_CHARS):
        super().__init__(daemon=True, name="translate")
        self.in_q = in_q
        self.out_q = out_q
        self.device = device
        self.model_path = model_path
        self.model_kind = model_kind
        self.history: list[tuple[str, str]] = []
        self.pending_source = ""
        self.segmenter = SentenceBoundaryDetector(
            WTP_MODEL_DIR, max_sentence_chars
        )

    def update_max_sentence_chars(self, characters: int):
        self.segmenter.update_max_length(characters)

    def _extract_ready_sentences(self, addition: str) -> list[str]:
        self.pending_source = _join_transcript(self.pending_source, addition)
        ready = []
        for _ in range(24):
            result = self.segmenter.pop_ready(self.pending_source)
            if result is None:
                break
            source, remainder = result
            if remainder == self.pending_source:
                break
            ready.append(source)
            self.pending_source = remainder
        return ready

    def run(self):
        try:
            self.out_q.put(UiEvent("status", "断句模型加载中…（CPU）"))
            self.segmenter.load()
        except Exception as e:
            self.out_q.put(UiEvent("error", f"断句模型加载失败：{e}"))
            return

        if self.model_kind.startswith("nllb-"):
            self._run_nllb()
            return

        try:
            self.out_q.put(UiEvent("status", "翻译模型加载中…"))
            with MODEL_LOAD_LOCK:
                tokenizer, model = load_nllb(self.model_path, self.device)
            self.out_q.put(UiEvent("status", "就绪"))
        except Exception as e:
            self.out_q.put(UiEvent("error", f"翻译模型加载失败：{e}"))
            return

        while True:
            item = self.in_q.get()
            if item is None:
                return
            if item.kind == "clear":
                self.pending_source = ""
                self.history.clear()
                continue
            if item.kind != "append":
                continue
            try:
                ready = self._extract_ready_sentences(item.text)
                if ready:
                    self.out_q.put(UiEvent(
                        "status", "翻译中…（CPU 模式可能需要较长时间）"
                    ))
                for source in ready:
                    context = "\n".join(
                        f"原文：{previous}\n译文：{zh}"
                        for previous, zh in self.history[-6:]
                    )
                    zh = nllb_translate(
                        source, tokenizer, model, self.device, 1024, 384,
                        context=context,
                    )
                    self._emit_pair(source, zh)
                self.out_q.put(UiEvent("status", "正在听…"))
            except Exception as e:
                self.out_q.put(UiEvent("error", f"翻译失败：{e}"))

    def _emit_pair(self, source: str, zh: str):
        if not source or not zh:
            return
        self.history.append((source, zh))
        self.history = self.history[-6:]
        self.out_q.put(UiEvent("translation_pair", zh, source=source))

    def _run_nllb(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.out_q.put(UiEvent("status", "NLLB 翻译模型加载中…"))
            with MODEL_LOAD_LOCK:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    src_lang=SOURCE_LANG,
                )
                load_kwargs = {"local_files_only": True}
                if self.device.type == "mps":
                    load_kwargs["torch_dtype"] = torch.float16
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path, **load_kwargs
                ).to(self.device)
                model.eval()
            target_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)
            self.out_q.put(UiEvent("status", "就绪"))
        except Exception as e:
            self.out_q.put(UiEvent("error", f"NLLB 模型加载失败：{e}"))
            return

        while True:
            item = self.in_q.get()
            if item is None:
                return
            if item.kind == "clear":
                self.pending_source = ""
                self.history.clear()
                continue
            if item.kind != "append":
                continue
            try:
                ready = self._extract_ready_sentences(item.text)
                if ready:
                    self.out_q.put(UiEvent("status", "NLLB 翻译中…"))
                for source in ready:
                    # NLLB needs an explicit source language. Japanese script
                    # is unambiguous; other Latin input is treated as English.
                    tokenizer.src_lang = (
                        "jpn_Jpan"
                        if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", source)
                        else "eng_Latn"
                    )
                    inputs = tokenizer(
                        source,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {
                        key: value.to(self.device) for key, value in inputs.items()
                    }
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            forced_bos_token_id=target_id,
                            max_new_tokens=192,
                        )
                    translated = tokenizer.batch_decode(
                        generated, skip_special_tokens=True
                    )[0].strip()
                    self._emit_pair(source, translated[:MAX_OUTPUT_CHARS])
                self.out_q.put(UiEvent("status", "正在听…"))
            except Exception as e:
                self.out_q.put(UiEvent("error", f"NLLB 翻译失败：{e}"))


class RealtimeApp:
    def __init__(
        self,
        device: torch.device,
        mic_index: int | None,
        system_audio_index: int | None,
        capture_system_audio: bool,
        translation_model: str,
        device_key: str = "auto",
        segment_seconds: int = MAX_SEGMENT_S,
        segment_mode: str = "smart",
        max_sentence_chars: int = MAX_SENTENCE_CHARS,
    ):
        self.device = device
        self.translation_model = translation_model
        self.device_key = device_key
        self.mic_index = mic_index
        self.system_audio_index = system_audio_index
        self.capture_system_audio = capture_system_audio
        self.segment_seconds = max(1, min(30, segment_seconds))
        self.segment_mode = segment_mode if segment_mode in ("fixed", "smart") else "smart"
        self.max_sentence_chars = max(20, min(200, int(max_sentence_chars)))
        self.min_speech_ms = MIN_SPEECH_MS
        self.end_silence_ms = END_SILENCE_MS
        self.pre_roll_ms = PRE_ROLL_MS
        self.energy_threshold = ENERGY_THRESHOLD

        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.text_q: queue.Queue[TranslationRequest | None] = queue.Queue()
        self.ui_q: queue.Queue[UiEvent] = queue.Queue()
        self._workers_started = False
        self._listen_session = 0
        self._last_audio_frame_at = 0.0
        self._last_audio_rms = 0.0
        self._session_peak_rms = 0.0
        self._audio_warning = False
        self._listen_state = "stopped"
        self._heard_reset_id = None
        self._asr_device_name = _funasr_device(device)

        if capture_system_audio and mic_index is not None:
            init_source = "both"
        elif capture_system_audio or system_audio_index is not None:
            init_source = "system"
        else:
            init_source = "mic"
        self.audio_source_key = init_source

        self.root = tk.Tk()
        self.audio_source_var = tk.StringVar(value=self.AUDIO_SOURCE_LABELS[init_source])
        self.segment_mode_var = tk.StringVar(
            value=self.SEGMENT_MODE_LABELS[self.segment_mode]
        )
        self.segment_seconds_var = tk.StringVar(value=str(self.segment_seconds))
        self.max_sentence_chars_var = tk.StringVar(
            value=str(self.max_sentence_chars)
        )
        self.segmenters = self._create_segmenters(self.audio_source_key)

        self.asr = AsrWorker(self.audio_q, self.ui_q, device=_funasr_device(device))
        translation_path = get_translation_path(translation_model, device)
        self.translator = TranslateWorker(
            self.text_q, self.ui_q, self.device, translation_path,
            translation_model, self.max_sentence_chars,
        )
        self._translation_path = translation_path

        self._build_ui()
        self._refresh_device_labels()

    def _create_segmenters(self, source_key: str):
        segmenters = []
        if source_key in ("mic", "both"):
            segmenters.append(
                MicSegmenter(
                    self._on_audio_segment, device=self.mic_index,
                    on_activity=self._on_audio_activity,
                    max_segment_s=self.segment_seconds,
                    segment_mode=self.segment_mode,
                    min_speech_ms=self.min_speech_ms,
                    end_silence_ms=self.end_silence_ms,
                    pre_roll_ms=self.pre_roll_ms,
                    threshold=self.energy_threshold,
                )
            )
        if source_key in ("system", "both"):
            if platform.system() == "Darwin":
                segmenters.append(SystemAudioSegmenter(
                    self._on_audio_segment,
                    on_activity=self._on_audio_activity,
                    on_error=self._on_audio_error,
                    max_segment_s=self.segment_seconds,
                    segment_mode=self.segment_mode,
                    min_speech_ms=self.min_speech_ms,
                    end_silence_ms=self.end_silence_ms,
                    pre_roll_ms=self.pre_roll_ms,
                    threshold=self.energy_threshold,
                ))
            elif self.system_audio_index is not None:
                segmenters.append(
                    MicSegmenter(
                        self._on_audio_segment, device=self.system_audio_index,
                        on_activity=self._on_audio_activity,
                        max_segment_s=self.segment_seconds,
                        segment_mode=self.segment_mode,
                        min_speech_ms=self.min_speech_ms,
                        end_silence_ms=self.end_silence_ms,
                        pre_roll_ms=self.pre_roll_ms,
                        threshold=self.energy_threshold,
                    )
                )
        return segmenters

    def _on_segment_mode_change(self, event=None):
        selected = self.segment_mode_var.get()
        self.segment_mode = next(
            (key for key, label in self.SEGMENT_MODE_LABELS.items() if label == selected),
            "fixed",
        )
        self._apply_segmenter_settings()

    def _on_segment_seconds_change(self, event=None):
        try:
            seconds = int(self.segment_seconds_var.get().split()[0])
        except (ValueError, IndexError):
            seconds = MAX_SEGMENT_S
        self.segment_seconds = max(1, min(30, seconds))
        self.segment_seconds_var.set(str(self.segment_seconds))
        self._apply_segmenter_settings()

    def _on_max_sentence_chars_change(self, event=None):
        try:
            characters = int(self.max_sentence_chars_var.get().split()[0])
        except (ValueError, IndexError):
            characters = MAX_SENTENCE_CHARS
        self.max_sentence_chars = max(20, min(200, characters))
        self.max_sentence_chars_var.set(str(self.max_sentence_chars))
        self.translator.update_max_sentence_chars(self.max_sentence_chars)

    def _apply_segmenter_settings(self):
        for segmenter in self.segmenters:
            processor = getattr(segmenter, "processor", segmenter)
            processor.update_settings(
                max_segment_s=self.segment_seconds,
                segment_mode=self.segment_mode,
                min_speech_ms=self.min_speech_ms,
                end_silence_ms=self.end_silence_ms,
                pre_roll_ms=self.pre_roll_ms,
                threshold=self.energy_threshold,
            )

    def _on_audio_source_change(self, event=None):
        val = self.audio_source_var.get()
        source_key = next(
            (k for k, v in self.AUDIO_SOURCE_LABELS.items() if v == val), "mic"
        )
        if source_key == self.audio_source_key and self.segmenters:
            return

        was_running = any(s.is_running for s in self.segmenters)
        for s in self.segmenters:
            try:
                s.stop()
            except Exception:
                pass

        self.audio_source_key = source_key
        self.segmenters = self._create_segmenters(source_key)

        if was_running:
            try:
                if not self.segmenters:
                    raise RuntimeError("当前声音来源没有可用的输入设备")
                for s in self.segmenters:
                    s.start()
                self._begin_listen_session()
                self._set_listen_state("active")
                self._refresh_audio_device_label()
            except Exception as e:
                for s in self.segmenters:
                    try:
                        s.stop()
                    except Exception:
                        pass
                self.status_var.set(f"音频切换错误：{e}")
        else:
            self._refresh_audio_device_label()

    def _on_audio_segment(self, audio: np.ndarray):
        self.audio_q.put(audio)
        self.ui_q.put(UiEvent("speech", ""))

    def _on_audio_activity(self, rms: float):
        self.ui_q.put(UiEvent("audio_level", f"{rms:.6f}"))

    def _on_audio_error(self, message: str):
        lowered = message.lower()
        if "tcc" in lowered or "declined" in lowered or "拒绝" in message:
            message = (
                "macOS 已拒绝电脑声音采集权限。请到“系统设置 → "
                "隐私与安全性 → 屏幕与系统音频录制”允许“实时英日翻译”，"
                "然后完全退出并重新打开应用"
            )
        else:
            message = f"音频采集失败：{message}"
        self.ui_q.put(UiEvent("audio_error", message))

    def _begin_listen_session(self):
        self._listen_session += 1
        session = self._listen_session
        self._last_audio_frame_at = 0.0
        self._last_audio_rms = 0.0
        self._session_peak_rms = 0.0
        self._audio_warning = False
        self.root.after(3500, lambda: self._check_audio_input(session))

    def _set_listen_state(self, state: str):
        """Render the three normal listening states in one place."""
        self._listen_state = state
        if state == "stopped":
            text, color = "停止", self.IDLE
            bg, fg, source_fg = self.IDLE_SOFT, self.TEXT_DIM, self.TEXT_MUTE
        elif state == "heard":
            text, color = "听到了", self.SPEAK
            bg, fg, source_fg = self.SPEAK_SOFT, "#A15F11", "#9A7445"
        else:
            text, color = "开启", self.OK
            bg, fg, source_fg = self.OK_SOFT, "#25745F", "#5F8C80"
        self.status_var.set(text)
        self.status_bar.configure(bg=bg)
        self.status_inner.configure(bg=bg)
        self.status_label.configure(bg=bg, fg=fg)
        self.source_label.configure(bg=bg, fg=source_fg)
        self.dot.configure(bg=bg)
        self.dot.itemconfigure(self._dot_id, fill=color)

    def _show_heard_state(self):
        if not any(segmenter.is_running for segmenter in self.segmenters):
            return
        if self._heard_reset_id is not None:
            try:
                self.root.after_cancel(self._heard_reset_id)
            except Exception:
                pass
        self._set_listen_state("heard")
        self._heard_reset_id = self.root.after(700, self._return_to_active_state)

    def _return_to_active_state(self):
        self._heard_reset_id = None
        if any(segmenter.is_running for segmenter in self.segmenters):
            self._set_listen_state("active")

    def _compute_device_label(self) -> str:
        if self.device.type == "mps":
            return "Apple MPS"
        if self.device.type == "cuda":
            if self.device_key == "rocm":
                return "AMD ROCm"
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return "CUDA"
        return "CPU"

    @staticmethod
    def _asr_device_label(device_name: str) -> str:
        if device_name == "mps":
            return "Apple MPS"
        if device_name.startswith("cuda"):
            return "CUDA"
        return "CPU"

    @staticmethod
    def _input_device_name(index: int | None) -> str:
        try:
            info = sd.query_devices(index, "input")
            return str(info["name"])
        except Exception:
            return "系统默认麦克风" if index is None else f"输入设备 {index}"

    def _audio_device_label(self) -> str:
        labels = []
        if self.audio_source_key in ("mic", "both"):
            labels.append(self._input_device_name(self.mic_index))
        if self.audio_source_key in ("system", "both"):
            if platform.system() == "Darwin":
                labels.append("系统内录 (ScreenCaptureKit)")
            elif self.system_audio_index is not None:
                labels.append(self._input_device_name(self.system_audio_index))
            else:
                labels.append("未配置系统音频设备")
        return " + ".join(labels) or "无可用输入设备"

    def _refresh_audio_device_label(self):
        self.source_var.set(f"输入：{self._audio_device_label()}")

    def _refresh_device_labels(self):
        self.compute_device_var.set(
            f"翻译：{self._compute_device_label()}  ·  "
            f"语音：{self._asr_device_label(self._asr_device_name)}  ·  "
            "断句：CPU"
        )
        self._refresh_audio_device_label()

    def _check_audio_input(self, session: int):
        if session != self._listen_session:
            return
        if not any(segmenter.is_running for segmenter in self.segmenters):
            return
        # Do not hide a model-loading/error message; check again shortly.
        if "加载" in self.status_var.get():
            self.root.after(2000, lambda: self._check_audio_input(session))
            return
        if self._last_audio_frame_at == 0.0:
            self._audio_warning = True
            self.status_var.set("已开启，但未收到音频数据；请检查声音来源和系统权限")
            self.dot.itemconfigure(self._dot_id, fill=self.DANGER)
        elif self._session_peak_rms < self.energy_threshold:
            self._audio_warning = True
            self.status_var.set("音频已连接，但当前静音或音量过低")
            self.dot.itemconfigure(self._dot_id, fill=self.IDLE)

    # ----- palette: a calm, light workspace -----
    BG          = "#F3F6FB"
    SURFACE     = "#FFFFFF"
    SURFACE_HI  = "#F7F9FD"
    BORDER      = "#E1E7F0"
    BORDER_HI   = "#CBD5E1"
    TEXT        = "#172033"
    TEXT_DIM    = "#526178"
    TEXT_MUTE   = "#8793A6"
    ACCENT      = "#5869E8"
    ACCENT_HOV  = "#4557D6"
    ACCENT_SOFT = "#ECEFFF"
    DANGER      = "#DF5964"
    DANGER_HOV  = "#C94855"
    NEUTRAL     = "#E9EEF5"
    NEUTRAL_HOV = "#DCE4EE"
    SPEAK       = "#F4A340"
    SPEAK_SOFT  = "#FFF2DF"
    OK          = "#18A17B"
    OK_SOFT     = "#E8F7F2"
    IDLE        = "#A8B2C1"
    IDLE_SOFT   = "#E9EEF5"
    JA          = "#25314A"
    ZH          = "#4B5FD4"
    SELECT      = "#DCE2FF"

    MODEL_LABELS = {
        "quality": "Qwen3 · 4B",
        "fast": "Qwen2.5 · 1.5B",
        "nllb-600m": "NLLB · 600M",
        "nllb-1.3b": "NLLB · 1.3B",
    }
    AUDIO_SOURCE_LABELS = {
        "mic": "麦克风",
        "system": "电脑声音 (系统内录)",
        "both": "全部 (麦克风 + 电脑声音)",
    }
    SEGMENT_MODE_LABELS = {
        "fixed": "固定",
        "smart": "智能",
    }
    SENSITIVITY_THRESHOLDS = {
        "高（轻声）": 0.001,
        "中（推荐）": ENERGY_THRESHOLD,
        "低（降噪）": 0.004,
    }

    def _build_ui(self):
        from tkinter import ttk

        self.root.title("实时英日翻译  Realtime EN/JA→ZH")
        self.root.geometry("640x560+80+80")
        self.root.minsize(520, 430)
        self.root.configure(bg=self.BG)
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass

        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass
        self._configure_styles()

        shell = tk.Frame(self.root, bg=self.BG)
        shell.pack(fill="both", expand=True)
        self._shell = shell

        self._build_header(shell)
        self._build_status(shell)
        # Reserve the action bar before the expanding subtitle card. This keeps
        # the primary controls visible even when the window is made compact.
        self._build_footer(shell)
        self._build_cards(shell)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self):
        from tkinter import ttk
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Light.TCombobox",
            fieldbackground=self.SURFACE_HI,
            background=self.SURFACE_HI,
            foreground=self.TEXT,
            arrowcolor=self.TEXT_DIM,
            bordercolor=self.BORDER,
            lightcolor=self.SURFACE_HI,
            darkcolor=self.SURFACE_HI,
            borderwidth=1,
            relief="flat",
            padding=(10, 6),
            focuscolor=self.ACCENT,
        )
        style.map(
            "Light.TCombobox",
            foreground=[("readonly", self.TEXT), ("disabled", self.TEXT_MUTE)],
            fieldbackground=[("readonly", self.SURFACE_HI)],
            selectbackground=[("readonly", self.SURFACE_HI)],
            selectforeground=[("readonly", self.TEXT)],
        )
        # dropdown list colors (works under clam on macOS too)
        self.root.option_add("*TCombobox*Listbox*Background", self.SURFACE_HI)
        self.root.option_add("*TCombobox*Listbox*Foreground", self.TEXT)
        self.root.option_add("*TCombobox*Listbox*selectBackground", self.ACCENT)
        self.root.option_add("*TCombobox*Listbox*selectForeground", "#FFFFFF")
        self.root.option_add("*TCombobox*Listbox*font", ("SF Pro Text", 11))
        self.root.option_add("*TCombobox*Listbox*borderWidth", 0)

    def _build_header(self, parent):
        bar = tk.Frame(parent, bg=self.BG)
        bar.pack(fill="x", padx=22, pady=(22, 14))

        mark = tk.Label(
            bar, text="译", bg=self.ACCENT, fg="#FFFFFF",
            font=("SF Pro Display", 16, "bold"), width=2, height=1,
        )
        mark.pack(side="left", ipadx=3, ipady=5, padx=(0, 11))

        titles = tk.Frame(bar, bg=self.BG)
        titles.pack(side="left")

        title = tk.Label(
            titles, text="实时英日翻译", bg=self.BG, fg=self.TEXT,
            font=("SF Pro Display", 19, "bold"), anchor="w",
        )
        title.pack(anchor="w")

        sub = tk.Label(
            titles, text="英语、日语自动识别并翻译成中文", bg=self.BG,
            fg=self.TEXT_MUTE, font=("SF Pro Text", 10), anchor="w",
        )
        sub.pack(anchor="w", pady=(2, 0))

        meta = tk.Frame(bar, bg=self.BG)
        meta.pack(side="right")

        language = tk.Label(
            meta, text="EN / JA   →   ZH", bg=self.ACCENT_SOFT, fg=self.ACCENT,
            font=("SF Pro Text", 10, "bold"), padx=13, pady=5,
        )
        language.pack(anchor="e")

        self.compute_device_var = tk.StringVar(value="设备检测中…")
        device_label = tk.Label(
            meta, textvariable=self.compute_device_var, bg=self.BG,
            fg=self.TEXT_MUTE, font=("SF Pro Text", 9), anchor="e",
        )
        device_label.pack(anchor="e", pady=(4, 0))

        # window drag handle
        for w in (bar, mark, titles, title, sub, meta, language, device_label):
            w.bind("<Button-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

    def _build_status(self, parent):
        bar = tk.Frame(parent, bg=self.IDLE_SOFT)
        bar.pack(fill="x", padx=22, pady=(0, 10), ipady=8)
        self.status_bar = bar

        inner = tk.Frame(bar, bg=self.IDLE_SOFT)
        inner.pack(fill="x", padx=13)
        self.status_inner = inner

        self.dot = tk.Canvas(inner, width=12, height=12, bg=self.IDLE_SOFT,
                             highlightthickness=0, bd=0)
        self._dot_id = self.dot.create_oval(1, 1, 11, 11, fill=self.IDLE, outline="")
        self.dot.pack(side="left")

        self.status_var = tk.StringVar(value="初始化中…")
        self.status_label = tk.Label(
            inner, textvariable=self.status_var, bg=self.IDLE_SOFT, fg=self.TEXT_DIM,
            font=("SF Pro Text", 10, "bold"), anchor="w",
        )
        self.status_label.pack(side="left", padx=(8, 0))

        self.source_var = tk.StringVar(value="")
        source_label = tk.Label(
            inner, textvariable=self.source_var, bg=self.IDLE_SOFT, fg=self.TEXT_MUTE,
            font=("SF Pro Text", 10), anchor="e",
        )
        source_label.pack(side="right")
        self.source_label = source_label

    def _build_cards(self, parent):
        body = tk.Frame(parent, bg=self.BG)
        self._cards_frame = body
        body.pack(fill="both", expand=True, padx=22, pady=(0, 10))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        card = self._make_card(body)
        card.grid(row=0, column=0, sticky="nsew")
        self._add_card_header(card, "实时字幕", "English / 日本語 → 中文", self.ACCENT)

        self.text = tk.Text(
            card, bg=self.SURFACE, fg=self.TEXT,
            insertbackground=self.TEXT, relief="flat", bd=0,
            font=("SF Pro Text", 13), wrap="word",
            width=1, height=1,
            padx=14, pady=10, spacing1=3, spacing3=4,
            selectbackground=self.SELECT, selectforeground=self.TEXT,
        )
        self.text.pack(fill="both", expand=True, padx=4, pady=(0, 8))
        self.text.configure(state="disabled")

        # tag styles — JA is the reading color, pending is muted, and ZH uses
        # the accent color so the translation is easy to scan.
        # lmargin1/2 indent ZH and pending under the JA line like subtitles
        self.text.tag_configure(
            "ja", foreground=self.JA,
            font=("SF Pro Text", 13),
        )
        self.text.tag_configure(
            "pending", foreground=self.TEXT_MUTE,
            font=("SF Pro Text", 12),
            lmargin1=18, lmargin2=18,
        )
        self.text.tag_configure(
            "zh", foreground=self.ZH,
            font=("SF Pro Text", 12, "bold"),
            lmargin1=18, lmargin2=18,
            spacing1=2,
        )
        self.text.tag_configure(
            "empty", foreground=self.TEXT_MUTE,
            font=("SF Pro Text", 12), justify="center",
            spacing1=10, spacing3=5,
        )
        self._is_empty = True
        self.text.configure(state="normal")
        self.text.insert(
            "1.0",
            "\n等待英语或日语语音…\n识别原文和中文翻译会依次显示在这里",
            "empty",
        )
        self.text.configure(state="disabled")


    def _make_card(self, parent):
        card = tk.Frame(parent, bg=self.SURFACE, highlightthickness=1,
                        highlightbackground=self.BORDER, highlightcolor=self.BORDER)
        return card

    def _add_card_header(self, card, title, sub, accent):
        head = tk.Frame(card, bg=self.SURFACE)
        head.pack(fill="x", padx=15, pady=(12, 8))

        marker = tk.Canvas(
            head, width=8, height=8, bg=self.SURFACE,
            highlightthickness=0, bd=0,
        )
        marker.create_oval(1, 1, 7, 7, fill=accent, outline="")
        marker.pack(side="left", padx=(0, 7))

        tk.Label(
            head, text=title, bg=self.SURFACE, fg=self.TEXT,
            font=("SF Pro Display", 12, "bold"), anchor="w",
        ).pack(side="left")

        tk.Label(
            head, text=sub, bg=self.SURFACE, fg=self.TEXT_MUTE,
            font=("SF Pro Text", 9), anchor="e",
        ).pack(side="right")

        sep = tk.Frame(card, bg=self.BORDER, height=1)
        sep.pack(fill="x", padx=15, pady=(0, 6))

    def _build_footer(self, parent):
        bar = tk.Frame(parent, bg=self.BG)
        bar.pack(side="bottom", fill="x", padx=22, pady=(2, 20))

        # Keep the everyday controls visible. Less common tuning lives in the
        # advanced dialog so the main translation surface stays quiet.
        from tkinter import ttk
        settings_row = tk.Frame(bar, bg=self.BG)
        settings_row.pack(fill="x", pady=(0, 10))

        src_frame = tk.Frame(settings_row, bg=self.BG)
        src_frame.pack(side="left")

        tk.Label(
            src_frame, text="声音来源", bg=self.BG, fg=self.TEXT_MUTE,
            font=("SF Pro Text", 10, "bold"),
        ).pack(side="left", padx=(0, 6))

        self.audio_source_combo = ttk.Combobox(
            src_frame, textvariable=self.audio_source_var,
            values=tuple(self.AUDIO_SOURCE_LABELS.values()),
            state="readonly", style="Light.TCombobox", width=15,
        )
        self.audio_source_combo.pack(side="left")
        self.audio_source_combo.bind("<<ComboboxSelected>>", self._on_audio_source_change)

        self.advanced_btn = self._make_button(
            settings_row, "高级设置", self._open_advanced_settings,
            bg=self.NEUTRAL, hover=self.NEUTRAL_HOV, fg=self.TEXT_DIM,
            font=("SF Pro Text", 10, "bold"), padx=10,
        )
        self.advanced_btn.pack(side="right")

        length_frame = tk.Frame(settings_row, bg=self.BG)
        length_frame.pack(side="right", padx=(8, 0))
        tk.Label(
            length_frame, text="最长句长", bg=self.BG, fg=self.TEXT_MUTE,
            font=("SF Pro Text", 10, "bold"),
        ).pack(side="left", padx=(0, 5))
        self.max_sentence_chars_combo = ttk.Combobox(
            length_frame, textvariable=self.max_sentence_chars_var,
            values=("40", "60", "80", "120"),
            state="normal", style="Light.TCombobox", width=4,
        )
        self.max_sentence_chars_combo.pack(side="left")
        for event_name in ("<<ComboboxSelected>>", "<Return>", "<FocusOut>"):
            self.max_sentence_chars_combo.bind(
                event_name, self._on_max_sentence_chars_change
            )
        tk.Label(
            length_frame, text="字", bg=self.BG, fg=self.TEXT_MUTE,
            font=("SF Pro Text", 10),
        ).pack(side="left", padx=(2, 0))

        action_row = tk.Frame(bar, bg=self.BG)
        action_row.pack(fill="x")

        self.toggle_btn = self._make_button(
            action_row, "开始收音", self._toggle,
            bg=self.ACCENT, hover=self.ACCENT_HOV, fg="white",
            font=("SF Pro Text", 12, "bold"), padx=20,
        )
        self.toggle_btn.pack(side="left")

        self.clear_btn = self._make_button(
            action_row, "清空字幕", self._clear,
            bg=self.NEUTRAL, hover=self.NEUTRAL_HOV, fg=self.TEXT_DIM,
            font=("SF Pro Text", 12), padx=16,
        )
        self.clear_btn.pack(side="right")

        self.help_btn = self._make_button(
            action_row, "设置说明", self._open_help_document,
            bg=self.SURFACE, hover=self.NEUTRAL, fg=self.TEXT_DIM,
            font=("SF Pro Text", 12), padx=16,
        )
        self.help_btn.configure(
            highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        self.help_btn.pack(side="right", padx=(0, 8))

    def _make_button(self, parent, text, command, bg, hover, fg, font, padx):
        btn = tk.Label(
            parent, text=text, bg=bg, fg=fg, font=font,
            padx=padx, pady=8, cursor="hand2",
        )
        btn._bg = bg
        btn._hover = hover
        btn._command = command

        def on_enter(_):
            btn.configure(bg=btn._hover)

        def on_leave(_):
            btn.configure(bg=btn._bg)

        def on_click(_):
            command()

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.bind("<Button-1>", on_click)
        return btn
    def _open_help_document(self):
        from tkinter import ttk

        existing = getattr(self, "_help_window", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        window = tk.Toplevel(self.root)
        self._help_window = window
        window.title("设置说明")
        window.geometry("620x650")
        window.minsize(480, 420)
        window.configure(bg=self.BG)
        window.transient(self.root)

        header = tk.Frame(window, bg=self.BG)
        header.pack(fill="x", padx=26, pady=(22, 14))
        tk.Label(
            header, text="设置说明", bg=self.BG, fg=self.TEXT,
            font=("SF Pro Display", 18, "bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            header, text="声音来源、分段规则和声音检测参数", bg=self.BG,
            fg=self.TEXT_MUTE, font=("SF Pro Text", 10), anchor="w",
        ).pack(anchor="w", pady=(3, 0))

        reading = tk.Frame(
            window, bg=self.SURFACE, highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        reading.pack(fill="both", expand=True, padx=26, pady=(0, 14))

        scrollbar = ttk.Scrollbar(reading, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        document = tk.Text(
            reading, bg=self.SURFACE, fg=self.TEXT, relief="flat", bd=0,
            wrap="word", padx=20, pady=16, spacing1=2, spacing3=7,
            font=("SF Pro Text", 11), yscrollcommand=scrollbar.set,
            selectbackground=self.SELECT, selectforeground=self.TEXT,
        )
        document.pack(fill="both", expand=True)
        scrollbar.configure(command=document.yview)

        document.tag_configure(
            "heading", foreground=self.TEXT,
            font=("SF Pro Display", 14, "bold"),
            spacing1=13, spacing3=7,
        )
        document.tag_configure(
            "body", foreground=self.TEXT_DIM,
            font=("SF Pro Text", 11),
            lmargin1=2, lmargin2=2,
        )
        document.tag_configure(
            "bullet", foreground=self.TEXT_DIM,
            font=("SF Pro Text", 11),
            lmargin1=4, lmargin2=22,
        )
        document.tag_configure(
            "error", foreground=self.DANGER,
            font=("SF Pro Text", 11, "bold"),
        )

        try:
            help_text = HELP_DOCUMENT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            document.insert(
                "end",
                f"无法读取说明文档：{exc}\n\n请确认 {HELP_DOCUMENT_PATH.name} 与程序位于同一目录。",
                "error",
            )
        else:
            for line in help_text.splitlines():
                if line.startswith("# "):
                    # The window header already carries the document title.
                    continue
                if line.startswith("## "):
                    document.insert("end", line[3:] + "\n", "heading")
                elif line.startswith("- "):
                    document.insert("end", "•  " + line[2:] + "\n", "bullet")
                elif line:
                    document.insert("end", line + "\n", "body")
                else:
                    document.insert("end", "\n", "body")
        document.configure(state="disabled")

        footer = tk.Frame(window, bg=self.BG)
        footer.pack(fill="x", padx=26, pady=(0, 20))

        def close_window(event=None):
            self._help_window = None
            window.destroy()

        close_btn = self._make_button(
            footer, "关闭", close_window,
            bg=self.NEUTRAL, hover=self.NEUTRAL_HOV, fg=self.TEXT_DIM,
            font=("SF Pro Text", 11), padx=18,
        )
        close_btn.pack(side="right")
        window.protocol("WM_DELETE_WINDOW", close_window)
        window.bind("<Escape>", close_window)
        window.focus_force()
    def _open_advanced_settings(self):
        from tkinter import ttk

        existing = getattr(self, "_advanced_window", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        window = tk.Toplevel(self.root)
        self._advanced_window = window
        window.title("高级分段设置")
        window.geometry("440x540")
        window.resizable(False, False)
        window.configure(bg=self.BG)
        window.transient(self.root)

        shell = tk.Frame(window, bg=self.BG)
        shell.pack(fill="both", expand=True, padx=24, pady=22)

        tk.Label(
            shell, text="高级分段设置", bg=self.BG, fg=self.TEXT,
            font=("SF Pro Display", 17, "bold"), anchor="w",
        ).pack(fill="x")

        summary_var = tk.StringVar()
        tk.Label(
            shell, textvariable=summary_var, bg=self.BG, fg=self.TEXT_MUTE,
            font=("SF Pro Text", 10), anchor="w", justify="left",
            wraplength=390,
        ).pack(fill="x", pady=(5, 16))

        mode_var = tk.StringVar(value=self.SEGMENT_MODE_LABELS[self.segment_mode])
        max_var = tk.StringVar(value=str(self.segment_seconds))
        min_var = tk.StringVar(value=f"{self.min_speech_ms / 1000:g}")
        pause_var = tk.StringVar(value=f"{self.end_silence_ms / 1000:g}")
        sentence_length_var = tk.StringVar(value=str(self.max_sentence_chars))
        pre_roll_var = tk.StringVar(value=str(self.pre_roll_ms))
        sensitivity_var = tk.StringVar(value=min(
            self.SENSITIVITY_THRESHOLDS,
            key=lambda label: abs(
                self.SENSITIVITY_THRESHOLDS[label] - self.energy_threshold
            ),
        ))

        def add_section(title):
            row = tk.Frame(shell, bg=self.BG)
            row.pack(fill="x", pady=(0, 8))
            tk.Label(
                row, text=title, bg=self.BG, fg=self.TEXT_DIM,
                font=("SF Pro Text", 11, "bold"), anchor="w",
            ).pack(side="left")
            tk.Frame(row, bg=self.BORDER, height=1).pack(
                side="left", fill="x", expand=True, padx=(10, 0)
            )

        def add_field(parent, row, label, variable, suffix="", values=None):
            tk.Label(
                parent, text=label, bg=self.SURFACE, fg=self.TEXT_DIM,
                font=("SF Pro Text", 10), anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(14, 10), pady=7)
            if values is None:
                widget = ttk.Entry(parent, textvariable=variable, width=10)
            else:
                widget = ttk.Combobox(
                    parent, textvariable=variable, values=values,
                    state="readonly", style="Light.TCombobox", width=13,
                )
            widget.grid(row=row, column=1, sticky="e", pady=7)
            tk.Label(
                parent, text=suffix, bg=self.SURFACE, fg=self.TEXT_MUTE,
                font=("SF Pro Text", 9), anchor="w", width=8,
            ).grid(row=row, column=2, sticky="w", padx=(7, 12), pady=7)
            return widget

        add_section("分段规则")
        segment_form = tk.Frame(
            shell, bg=self.SURFACE, highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        segment_form.pack(fill="x", pady=(0, 14))
        segment_form.columnconfigure(0, weight=1)
        mode_widget = add_field(
            segment_form, 0, "分段方式", mode_var,
            values=tuple(self.SEGMENT_MODE_LABELS.values()),
        )
        add_field(segment_form, 1, "最长分段", max_var, "秒")
        add_field(segment_form, 2, "最短语音", min_var, "秒")
        pause_widget = add_field(segment_form, 3, "停顿截断", pause_var, "秒")
        add_field(segment_form, 4, "最长句长", sentence_length_var, "字")

        add_section("声音检测")
        audio_form = tk.Frame(
            shell, bg=self.SURFACE, highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        audio_form.pack(fill="x")
        audio_form.columnconfigure(0, weight=1)
        add_field(
            audio_form, 0, "收音灵敏度", sensitivity_var,
            values=tuple(self.SENSITIVITY_THRESHOLDS),
        )
        add_field(audio_form, 1, "语音前置保留", pre_roll_var, "毫秒")

        error_var = tk.StringVar(value="")
        tk.Label(
            shell, textvariable=error_var, bg=self.BG, fg=self.DANGER,
            font=("SF Pro Text", 9), anchor="w",
        ).pack(fill="x", pady=(8, 0))

        def refresh_mode(event=None):
            smart = mode_var.get() == self.SEGMENT_MODE_LABELS["smart"]
            pause_widget.configure(state="normal" if smart else "disabled")
            if smart:
                summary_var.set("遇到设定长度的停顿就提交，同时受最长分段限制。")
            else:
                summary_var.set("按设定秒数提交，中途的短暂停顿不会提前截断。")

        mode_widget.bind("<<ComboboxSelected>>", refresh_mode)
        refresh_mode()

        def close_window():
            try:
                window.grab_release()
            except Exception:
                pass
            self._advanced_window = None
            window.destroy()

        def apply_settings():
            try:
                max_seconds = int(max_var.get())
                min_seconds = float(min_var.get())
                pause_seconds = float(pause_var.get())
                sentence_characters = int(sentence_length_var.get())
                pre_roll_ms = int(pre_roll_var.get())
            except ValueError:
                error_var.set("请在设置中输入有效数字")
                return

            try:
                if not 1 <= max_seconds <= 30:
                    raise ValueError("最长分段需要在 1–30 秒之间")
                if not 0.1 <= min_seconds <= 5:
                    raise ValueError("最短语音需要在 0.1–5 秒之间")
                if min_seconds > max_seconds:
                    raise ValueError("最短语音不能长于最长分段")
                if not 0.2 <= pause_seconds <= 3:
                    raise ValueError("停顿截断需要在 0.2–3 秒之间")
                if not 20 <= sentence_characters <= 200:
                    raise ValueError("最长句长需要在 20–200 字之间")
                if not 0 <= pre_roll_ms <= 1000:
                    raise ValueError("前置保留需要在 0–1000 毫秒之间")
            except ValueError as exc:
                error_var.set(str(exc))
                return

            self.segment_mode = next(
                (key for key, label in self.SEGMENT_MODE_LABELS.items()
                 if label == mode_var.get()),
                "fixed",
            )
            self.segment_seconds = max_seconds
            self.min_speech_ms = round(min_seconds * 1000)
            self.end_silence_ms = round(pause_seconds * 1000)
            self.max_sentence_chars = sentence_characters
            self.pre_roll_ms = pre_roll_ms
            self.energy_threshold = self.SENSITIVITY_THRESHOLDS[sensitivity_var.get()]
            self.segment_mode_var.set(self.SEGMENT_MODE_LABELS[self.segment_mode])
            self.segment_seconds_var.set(str(self.segment_seconds))
            self.max_sentence_chars_var.set(str(self.max_sentence_chars))
            self.translator.update_max_sentence_chars(
                self.max_sentence_chars
            )
            self._apply_segmenter_settings()
            close_window()

        actions = tk.Frame(shell, bg=self.BG)
        actions.pack(side="bottom", fill="x", pady=(12, 0))
        cancel_btn = self._make_button(
            actions, "取消", close_window,
            bg=self.NEUTRAL, hover=self.NEUTRAL_HOV, fg=self.TEXT_DIM,
            font=("SF Pro Text", 11), padx=16,
        )
        cancel_btn.pack(side="right")
        apply_btn = self._make_button(
            actions, "应用设置", apply_settings,
            bg=self.ACCENT, hover=self.ACCENT_HOV, fg="white",
            font=("SF Pro Text", 11, "bold"), padx=16,
        )
        apply_btn.pack(side="right", padx=(0, 8))

        window.protocol("WM_DELETE_WINDOW", close_window)
        window.grab_set()
        window.focus_force()

    # ----- window drag -----
    def _drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _drag_move(self, event):
        x = self.root.winfo_pointerx() - self._drag_x
        y = self.root.winfo_pointery() - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _set_text(self, text):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self._is_empty = not bool(text)
        if text:
            self.text.insert("1.0", text, "ja")
        else:
            self.text.insert(
                "1.0",
                "\n等待英语或日语语音…\n识别原文和中文翻译会依次显示在这里",
                "empty",
            )
        self.text.configure(state="disabled")

    def _append_translation_pair(self, source: str, zh: str):
        self.text.configure(state="normal")
        if self._is_empty:
            self.text.delete("1.0", "end")
            self._is_empty = False
        self.text.insert("end", source.rstrip() + "\n", "ja")
        self.text.insert("end", "  " + zh.strip() + "\n", "zh")
        self.text.see("end")
        self.text.configure(state="disabled")

    def _toggle(self):
        if any(segmenter.is_running for segmenter in self.segmenters):
            self._listen_session += 1
            if self._heard_reset_id is not None:
                try:
                    self.root.after_cancel(self._heard_reset_id)
                except Exception:
                    pass
                self._heard_reset_id = None
            for segmenter in self.segmenters:
                segmenter.stop()
            self.toggle_btn.configure(text="开始收音", bg=self.ACCENT)
            self.toggle_btn._bg = self.ACCENT
            self.toggle_btn._hover = self.ACCENT_HOV
            self._set_listen_state("stopped")
            self._refresh_audio_device_label()
        else:
            try:
                if not self.segmenters:
                    raise RuntimeError("当前声音来源没有可用的输入设备")
                for segmenter in self.segmenters:
                    segmenter.start()
            except Exception as e:
                for segmenter in self.segmenters:
                    segmenter.stop()
                self.status_var.set(f"音频设备错误：{e}")
                return
            self.toggle_btn.configure(text="停止收音", bg=self.DANGER)
            self.toggle_btn._bg = self.DANGER
            self.toggle_btn._hover = self.DANGER_HOV
            self._set_listen_state("active")
            self._refresh_audio_device_label()
            self._begin_listen_session()

    def _clear(self):
        self.text_q.put(TranslationRequest(kind="clear"))
        self._set_text("")

    def _on_close(self):
        for segmenter in self.segmenters:
            segmenter.stop()
        try:
            self.audio_q.put_nowait(None)
            self.text_q.put_nowait(None)
        except Exception:
            pass
        self.root.destroy()

    def _poll_ui(self):
        try:
            while True:
                ev = self.ui_q.get_nowait()
                if ev.kind == "status":
                    # Worker progress should not overwrite the three-state
                    # listening indicator while audio capture is active.
                    if not any(s.is_running for s in self.segmenters):
                        if "加载" in ev.text or "回退" in ev.text:
                            self.status_var.set(ev.text)
                        else:
                            self._set_listen_state("stopped")
                elif ev.kind == "asr_device":
                    self._asr_device_name = ev.text
                    self._refresh_device_labels()
                elif ev.kind in ("error", "audio_error"):
                    self.status_var.set(ev.text)
                    self.dot.itemconfigure(self._dot_id, fill=self.DANGER)
                    if ev.kind == "audio_error":
                        self._listen_session += 1
                        for segmenter in self.segmenters:
                            try:
                                segmenter.stop()
                            except Exception:
                                pass
                        self.toggle_btn.configure(text="开始收音", bg=self.ACCENT)
                        self.toggle_btn._bg = self.ACCENT
                        self.toggle_btn._hover = self.ACCENT_HOV
                    self.text.configure(state="normal")
                    self.text.insert("end", f"\n[!] {ev.text}\n", "pending")
                    self.text.see("end")
                    self.text.configure(state="disabled")
                elif ev.kind == "speech":
                    self._show_heard_state()
                elif ev.kind == "audio_level":
                    rms = float(ev.text)
                    self._last_audio_frame_at = time.monotonic()
                    self._last_audio_rms = rms
                    self._session_peak_rms = max(self._session_peak_rms, rms)
                    if rms >= self.energy_threshold:
                        self._show_heard_state()
                    if self._audio_warning and rms >= self.energy_threshold:
                        self._audio_warning = False
                        self._set_listen_state("heard")
                elif ev.kind == "asr_text":
                    self.text_q.put(TranslationRequest(text=ev.text))
                elif ev.kind == "translation_pair":
                    self._append_translation_pair(ev.source, ev.text)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_ui)

    def run(self):
        if not self._workers_started:
            self.asr.start()
            self.translator.start()
            self._workers_started = True
            self.root.after(80, self._poll_ui)
        self.root.mainloop()


def _funasr_device(device: torch.device) -> str:
    if device.type == "mps":
        return "mps"
    if device.type == "cuda":
        return "cuda:0"
    return "cpu"


def list_mics():
    print("可用输入设备：")
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            print(f"  [{i}] {dev['name']}  (channels={dev['max_input_channels']})")


def main():
    parser = argparse.ArgumentParser(description="实时英语/日语语音 → 中文翻译浮窗")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "rocm", "cpu"), default="auto")
    parser.add_argument(
        "--translation-model",
        choices=("quality", "fast", "nllb-600m", "nllb-1.3b"),
        default="quality",
        help=(
            "quality=Qwen3 4B；fast=Qwen2.5 1.5B；"
            "nllb-600m/nllb-1.3b=NLLB"
        ),
    )
    parser.add_argument("--mic", type=int, default=None, help="输入设备 index（默认系统默认）")
    parser.add_argument(
        "--system-audio",
        type=int,
        default=None,
        help="电脑音频虚拟输入设备 index（例如 BlackHole，可与 --mic 同时使用）",
    )
    parser.add_argument(
        "--capture-system-audio",
        action="store_true",
        help="使用 macOS ScreenCaptureKit 直接捕获电脑声音（无需虚拟声卡）",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        choices=range(1, 31),
        default=MAX_SEGMENT_S,
        metavar="1-30",
        help=f"连续语音最长分段时间（秒，默认 {MAX_SEGMENT_S}）",
    )
    parser.add_argument(
        "--segment-mode",
        choices=("fixed", "smart"),
        default="smart",
        help="分段方式：fixed=固定时间，smart=智能停顿（默认 smart）",
    )
    parser.add_argument(
        "--max-sentence-chars",
        type=int,
        choices=range(20, 201),
        default=MAX_SENTENCE_CHARS,
        metavar="20-200",
        help=(
            "语义句超过该长度时由断句模型选择自然分割点"
            f"（字符，默认 {MAX_SENTENCE_CHARS}）"
        ),
    )
    parser.add_argument("--list-mics", action="store_true", help="列出可用麦克风后退出")
    args = parser.parse_args()

    if args.list_mics:
        list_mics()
        return

    # No matter which start entry is used, first-run preparation checks the
    # complete model files and provides the same visible progress output.
    from launcher import ensure_models
    try:
        ensure_models(
            {
                "device": args.device,
                "translation_model": args.translation_model,
            },
            announce=False,
        )
    except Exception as err:
        print(f"模型自动准备失败：{err}", file=sys.stderr)
        print("请检查网络后重新启动，下载会自动续传。", file=sys.stderr)
        return

    device = choose_device(args.device)
    translation_path = get_translation_path(args.translation_model, device)

    asr_device = _funasr_device(device)

    print("─" * 54, file=sys.stderr)
    print(f"  翻译模型    : {args.translation_model:<8} → {translation_path.name}", file=sys.stderr)
    print(f"  翻译设备    : {device}", file=sys.stderr)
    print(f"  ASR 模型    : SenseVoiceSmall → {SENSEVOICE_MODEL_DIR.name}", file=sys.stderr)
    print(f"  ASR 设备    : {asr_device}", file=sys.stderr)
    print(f"  断句模型   : WtP → {WTP_MODEL_DIR.name} (CPU)", file=sys.stderr)
    print("─" * 54, file=sys.stderr)

    if args.system_audio is not None and args.system_audio == args.mic:
        parser.error("--mic 和 --system-audio 不能选择同一个设备")

    app = RealtimeApp(
        device, args.mic, args.system_audio, args.capture_system_audio,
        args.translation_model, args.device, args.segment_seconds,
        args.segment_mode, args.max_sentence_chars,
    )
    app.run()


if __name__ == "__main__":
    main()
