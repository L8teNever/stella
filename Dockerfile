FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY stella /app/stella

RUN pip install --no-cache-dir .

RUN mkdir -p /data

ENV STELLA_DB_PATH=/data/stella.db \
    STELLA_HOST=0.0.0.0 \
    STELLA_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["stella", "serve"]
