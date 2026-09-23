# Build from the repository root.
# CUDA runtime comes from the base image; torch is installed from plain
# PyPI (no cu128 index) so it links against system CUDA instead of
# bundling its own copy.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEV_MODE=false \
    PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && python -m pip check

COPY src/ ./src/
COPY model_store/ ./model_store/
COPY configs/ ./configs/

# Non-root runtime user (Google container security baseline).
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser /app
USER appuser

# Import check intentionally avoids inference and downloading weights.
RUN python -c "import runner; import backend.worker_e2e.worker"
ENTRYPOINT ["python", "-m", "backend.worker_e2e.worker"]
