# syntax=docker/dockerfile:1

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# Use Debian's Python so that the matching python3-tk package provides the
# native Tk extension required by realtime.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libasound2-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libopenblas-dev \
        ninja-build \
        pkg-config \
        portaudio19-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-tk \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

# Keep the expensive dependency layer reusable while application code changes.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.6.0" "torchaudio==2.6.0" \
    && python -m pip install -r requirements.txt

COPY install_llama_cpu.py ./
RUN --mount=type=cache,target=/root/.cache/pip \
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a" \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    python install_llama_cpu.py

COPY . .
RUN chmod +x docker/entrypoint.sh

# Models are downloaded on first use and should survive container replacement.
VOLUME ["/app/models"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["realtime"]
