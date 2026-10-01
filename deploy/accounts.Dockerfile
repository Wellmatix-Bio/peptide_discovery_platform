# The accounts service and authenticating proxy -- the ONLY route to the job API.
# Standard-library security code; FastAPI, httpx and uvicorn only for HTTP. Runs as a NON-ROOT user, which is exactly why a bind-mounted data
# directory (created root:root by Docker) fails - the service checks writability at startup and
# prints the chown to run. The named volume in compose.yaml avoids it: a fresh named volume takes
# the ownership of /data in this image.

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --uid 10001 --create-home app
WORKDIR /srv
COPY services/accounts/requirements.txt ./
RUN pip install -r requirements.txt
COPY services/accounts/accounts ./accounts
RUN mkdir -p /data && chown app:app /data
ENV ACCOUNTS_DATA_DIR=/data
VOLUME /data
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"
# uvicorn reads WEB_CONCURRENCY for its worker count. Several workers are fine - every piece of
# shared state is in SQLite. Several REPLICAS are not: SQLite means one instance.
CMD ["python", "-m", "uvicorn", "accounts.app:build", "--factory", "--host", "0.0.0.0", "--port", "8080"]
