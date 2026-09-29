# Build from the repository root. CUDA runtime libraries come with PyTorch's wheel.
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEV_MODE=false \
    PYTHONPATH=/app/src:/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && \
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
