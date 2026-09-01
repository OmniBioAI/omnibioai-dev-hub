# ── Stage 1: Build Vite UI ────────────────────────────────────────────────────
# Explicitly target linux/arm64 for aarch64 host
FROM --platform=linux/arm64 node:20-bookworm-slim AS ui-builder
WORKDIR /ui
COPY omnibioai-dev-hub-ui/package*.json ./
RUN npm ci
COPY omnibioai-dev-hub-ui/ ./
RUN npm run build

# ── Stage 2: Python API + nginx ───────────────────────────────────────────────
FROM --platform=linux/arm64 ghcr.io/omnibioai/omnibioai-base:latest AS backend

LABEL org.opencontainers.image.source=https://github.com/man4ish/omnibioai

# curl needed for Ollama readiness check; nginx for UI serving
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# faiss-cpu must be installed BEFORE sentence-transformers to avoid conflicts
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir faiss-cpu && \
    pip install --no-cache-dir fastapi uvicorn requests && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY api/        ./api/
COPY embeddings/ ./embeddings/
COPY index/      ./index/
COPY ingestion/  ./ingestion/
COPY processing/ ./processing/
COPY retrieval/  ./retrieval/
COPY rag/        ./rag/
COPY scripts/    ./scripts/
COPY configs/    ./configs/

# Copy built UI from builder stage
COPY --from=ui-builder /ui/dist /usr/share/nginx/html

# Remove the default nginx site; the real devhub.conf is generated at
# container start (see CMD) so it can embed the runtime JWT_SECRET value.
RUN rm -f /etc/nginx/sites-enabled/default \
          /etc/nginx/sites-available/default

# ── Non-root runtime user ──────────────────────────────────────────────────
# Runtime writes this container actually performs, and what each needs:
#   /app/data/faiss_index/  -- VectorStore.save() (index.faiss, metadata.pkl),
#                              written by scripts/build_index.py on first boot
#   /app/logs/, /app/cache/ -- reserved for app use (currently unwritten, but
#                              pre-created + owned so a future write doesn't
#                              silently hit a permission error)
#   /var/log/nginx/         -- access.log / error.log (nginx opens these at
#                              startup, not lazily -- must be writable then)
#   /var/lib/nginx/         -- client_body/proxy/fastcgi/... temp dirs, created
#                              by nginx workers on first request
#   /etc/nginx/conf.d/      -- devhub.conf is generated here by the CMD script
#                              at container start (embeds the runtime JWT_SECRET)
# The stock nginx.conf also sets `user www-data;` and `pid /run/nginx.pid;`,
# both of which assume a root master process (setuid to www-data, write to
# /run which is root:root here) -- since the master now runs as appuser,
# the `user` directive is dropped (nginx just runs workers as whoever
# started it) and the pid file is redirected to /tmp (world-writable).
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/faiss_index /app/logs /app/cache \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /var/log/nginx /var/lib/nginx /etc/nginx/conf.d \
    && sed -i '/^user www-data;$/d' /etc/nginx/nginx.conf \
    && sed -i 's#pid /run/nginx.pid;#pid /tmp/nginx.pid;#' /etc/nginx/nginx.conf

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    REPO_BASE=/repos

USER appuser

EXPOSE 8082 5173

# Startup sequence:
#   1. Wait for Ollama to be ready (it starts in parallel via depends_on)
#   2. Build FAISS index on first run only (skipped if index already exists)
#   3. Generate nginx's devhub.conf with the runtime JWT_SECRET baked in as
#      the internal proxy header (see api/auth.py) -- can't be done at
#      build time since the secret is only known at container start
#   4. Start nginx (serves UI on 5173)
#   5. Start FastAPI (serves API on 8082)
CMD ["bash", "-c", "\
  echo '⏳ Waiting for Ollama...' && \
  until curl -sf http://ollama:11434/api/tags > /dev/null 2>&1; do \
    echo '  ollama not ready, retrying in 3s...'; sleep 3; \
  done && \
  echo '✅ Ollama is ready' && \
  if [ ! -f /app/data/faiss_index/index.faiss ]; then \
    echo '🚀 Building FAISS index...' && \
    python scripts/build_index.py; \
  else \
    echo '✅ Index already exists, skipping build'; \
  fi && \
  printf 'server {\n\
    listen 5173;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / { try_files $uri $uri/ /index.html; }\n\
    location /api/ { proxy_pass http://127.0.0.1:8082; proxy_set_header Host $host; }\n\
    location /rag/ { proxy_pass http://127.0.0.1:8082; proxy_set_header X-Devhub-Internal \"%s\"; }\n\
    location /health { proxy_pass http://127.0.0.1:8082; }\n\
    location /status { proxy_pass http://127.0.0.1:8082; }\n\
    location /docs   { proxy_pass http://127.0.0.1:8082; }\n\
}\n' \"${JWT_SECRET:-change-me}\" > /etc/nginx/conf.d/devhub.conf && \
  nginx && \
  echo '🌐 nginx started on port 5173' && \
  uvicorn api.main:app --host 0.0.0.0 --port 8082"]