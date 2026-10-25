from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


VERSION = "0.3.35"


def _extract_source(archive: tarfile.TarFile, destination: Path) -> None:
    """Safely extract on both Python 3.10/3.11 and Python 3.12+."""
    try:
        archive.extractall(destination, filter="data")
        return
    except TypeError:
        # The filter argument was added in Python 3.12. Reject path traversal
        # explicitly before using the older API.
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(
                    f"源码压缩包包含不安全路径：{member.name}"
                )
        archive.extractall(destination)


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
            _extract_source(tar, tmp_path)
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
                "--no-cache-dir", str(source.parent),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
