FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DEV_MODE=false PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src/common/ ./src/common/
COPY src/schemas/ ./src/schemas/
COPY src/backend/api_e2e/ ./src/backend/api_e2e/
EXPOSE 8080
ENTRYPOINT ["uvicorn", "backend.api_e2e.api:app", "--host", "0.0.0.0", "--port", "8080"]
