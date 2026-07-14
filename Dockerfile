# Tenet backend — deployable to Alibaba Cloud (Function Compute container or ECS+Docker).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Run as non-root (semgrep dockerfile.security.missing-user); /data holds the sqlite store.
RUN useradd -m -u 10001 tenet && mkdir -p /data && chown tenet /data
ENV TENET_DB_PATH=/data/tenet.db
USER tenet

# DASHSCOPE_API_KEY + (optional) Alibaba Cloud OSS creds are injected as env vars
# at deploy time — never baked into the image.
EXPOSE 8000
CMD ["uvicorn", "tenet.api:app", "--host", "0.0.0.0", "--port", "8000"]
