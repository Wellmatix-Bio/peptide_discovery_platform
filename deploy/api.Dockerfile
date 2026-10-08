# The job API. It has NO AUTHENTICATION of its own and publishes no host port in
# deploy/compose.yaml: it binds 0.0.0.0 inside the compose network, and the accounts proxy is the
# only route to it. That, not the web app, is what makes "the backend has no auth" safe.
#
# Built from the checkout rather than a wheel: the project ships no package metadata for the
# backend, and /api/v1/models reads model_store/*/predictor.py and model_card.json off disk.

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --uid 10001 --create-home app
WORKDIR /app

COPY deploy/api.requirements.txt ./
RUN pip install -r api.requirements.txt

COPY src ./src
# Read, never imported: the manifest endpoint digests each predictor.py and reads its model card.
# The WEIGHTS are not here and must not be -- they are synced to the worker from VERTEX_MODEL_STORE
# at run time, which is why the manifest says its digests cover code and not weights.
COPY model_store ./model_store

ENV PYTHONPATH=/app/src
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"
# One worker. The API only submits jobs and reads artifacts; the GPU work is Vertex's.
# Shell form so Cloud Run's $PORT is honoured; compose leaves PORT unset, so 8080 is used there.
CMD python -m uvicorn backend.api_e2e.api:app --host 0.0.0.0 --port ${PORT:-8080}
