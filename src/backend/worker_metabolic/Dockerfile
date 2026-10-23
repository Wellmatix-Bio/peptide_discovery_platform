# syntax=docker/dockerfile:1.7
# Build from the repository root. Match the CUDA 12.8 PyTorch wheel below.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
# LD_LIBRARY_PATH: Vertex mounts the host NVIDIA driver under /usr/local/nvidia; without it PyTorch cannot find libcuda and sees no GPU.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEV_MODE=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64 \
    VIRTUAL_ENV=/opt/venv
# Ubuntu 24.04 provides Python 3.12. Keep pip installs out of system Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    build-essential git libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /app
# Cache the large PyTorch installation independently of application dependencies.
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /app/requirements.txt
# torch stays on the CUDA 12.8 build installed above; the pin in requirements.txt would replace it with a CUDA 13 build.
RUN grep -v '^torch==' requirements.txt > /tmp/requirements.notorch.txt && \
    python -m pip install -r /tmp/requirements.notorch.txt && python -m pip check
# propy3 (GPL-2.0-only) is installed here, not in requirements.txt, so the aggregation predictor runs in the worker image (docs/LICENSING.md).
RUN python -m pip install propy3 && python -m pip check
COPY src/ ./src/
COPY model_store/ ./model_store/
COPY configs/ ./configs/

# Non-root runtime user (Google container security baseline).
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser /app
USER appuser

# Import check intentionally avoids inference and downloading weights.
RUN python -c "import runner; import backend.worker_metabolic.worker"
ENTRYPOINT ["python", "-m", "backend.worker_metabolic.worker"]
