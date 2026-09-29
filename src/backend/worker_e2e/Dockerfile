# syntax=docker/dockerfile:1.7
# Build from the repository root. Match the CUDA 12.8 PyTorch wheel below.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEV_MODE=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
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
RUN python -m pip install -r requirements.txt && python -m pip check
COPY src/ ./src/
COPY model_store/ ./model_store/
COPY configs/ ./configs/

# Non-root runtime user (Google container security baseline).
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser /app
USER appuser

# Import check intentionally avoids inference and downloading weights.
RUN python -c "import runner; import backend.worker_e2e.worker"
ENTRYPOINT ["python", "-m", "backend.worker_e2e.worker"]
