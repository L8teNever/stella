# Claude Code CLI (used by the frag_ida tool). Pin the version; bump deliberately.
FROM node:22-slim AS claude
RUN npm install -g @anthropic-ai/claude-code@2.1.287

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY stella /app/stella

COPY --from=claude /usr/local/bin/node /usr/local/bin/node
COPY --from=claude /usr/local/lib/node_modules/@anthropic-ai /usr/local/lib/node_modules/@anthropic-ai
RUN ln -s /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude \
    && claude --version

RUN pip install --no-cache-dir .

RUN mkdir -p /data

ENV STELLA_DB_PATH=/data/stella.db \
    HOME=/data \
    CLAUDE_CONFIG_DIR=/data/claude \
    STELLA_HOST=0.0.0.0 \
    STELLA_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["stella", "serve"]
