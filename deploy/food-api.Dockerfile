# The job API for running the food pipeline on this computer (DEV_MODE): it starts the food worker as a subprocess in the same container, on CPU.
# Build from the repository root (deploy/compose.food.yaml does). The model weights are NOT in the image; compose mounts them read-only at /app/model_store/model_weights.
# propy3 (GPL-2.0-only, optional) is installed so the aggregation screen runs, as the worker image does; installing it is your acceptance of its terms (docs/LICENSING.md).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEV_MODE=true PYTHONPATH=/app/src:/app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# The CPU PyTorch wheel, installed first so it is cached apart from the other dependencies; requirements.txt pins the default build, which is dropped here.
RUN python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/requirements.txt
RUN grep -v '^torch==' requirements.txt > /tmp/requirements.notorch.txt && python -m pip install -r /tmp/requirements.notorch.txt
RUN python -m pip install propy3 && python -m pip check
COPY src ./src
COPY model_store ./model_store
COPY configs ./configs
# Runs are written under /app/artifacts (a volume in compose); the user owns it so the volume starts writable.
RUN useradd --uid 10001 --create-home app && mkdir -p /app/artifacts && chown app /app/artifacts
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"
CMD ["python", "-m", "uvicorn", "backend.api_e2e.api:app", "--host", "0.0.0.0", "--port", "8080"]
