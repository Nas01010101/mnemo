# Mnemo backend — deployable to Alibaba Cloud (Function Compute container or ECS+Docker).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
ENV PYTHONUNBUFFERED=1

# DASHSCOPE_API_KEY + (optional) Alibaba Cloud OSS creds are injected as env vars
# at deploy time — never baked into the image.
EXPOSE 8000
WORKDIR /app/src
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
