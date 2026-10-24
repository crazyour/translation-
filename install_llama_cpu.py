from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


VERSION = "0.3.35"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llama-cpu-build-") as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                "--no-binary", "llama-cpp-python", "--no-deps",
                "-d", str(tmp_path), f"llama-cpp-python=={VERSION}",
            ],
            check=True,
        )
        archive = next(tmp_path.glob("llama_cpp_python-*.tar.gz"))
        with tarfile.open(archive) as tar:
            tar.extractall(tmp_path, filter="data")
        source = next(tmp_path.glob("llama_cpp_python-*/CMakeLists.txt"))
        cmake = source.read_text()
        cmake = cmake.replace(
            'set(GGML_METAL "ON" CACHE BOOL "ggml: enable Metal" FORCE)',
            'set(GGML_METAL "OFF" CACHE BOOL "ggml: disable Metal for CPU mode" FORCE)',
        ).replace(
            'set(GGML_METAL_EMBED_LIBRARY "ON" CACHE BOOL "ggml: embed metal library" FORCE)',
            'set(GGML_METAL_EMBED_LIBRARY "OFF" CACHE BOOL "ggml: do not embed Metal" FORCE)',
        )
        source.write_text(cmake)
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--force-reinstall",
                "--no-deps", "--no-cache-dir", str(source.parent),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
