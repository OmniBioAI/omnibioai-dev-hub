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
COPY utils/      ./utils/
COPY configs/    ./configs/

# Copy built UI from builder stage
COPY --from=ui-builder /ui/dist /usr/share/nginx/html

# nginx config — serve UI on 5173, proxy API routes to FastAPI on 8082
RUN printf 'server {\n\
    listen 5173;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / { try_files $uri $uri/ /index.html; }\n\
    location /api/ { proxy_pass http://127.0.0.1:8082; proxy_set_header Host $host; }\n\
    location /rag/ { proxy_pass http://127.0.0.1:8082; }\n\
    location /health { proxy_pass http://127.0.0.1:8082; }\n\
    location /status { proxy_pass http://127.0.0.1:8082; }\n\
    location /docs   { proxy_pass http://127.0.0.1:8082; }\n\
}\n' > /etc/nginx/conf.d/devhub.conf && \
    rm -f /etc/nginx/sites-enabled/default \
          /etc/nginx/sites-available/default

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    REPO_BASE=/repos

EXPOSE 8082 5173

# Startup sequence:
#   1. Wait for Ollama to be ready (it starts in parallel via depends_on)
#   2. Build FAISS index on first run only (skipped if index already exists)
#   3. Start nginx (serves UI on 5173)
#   4. Start FastAPI (serves API on 8082)
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
  nginx && \
  echo '🌐 nginx started on port 5173' && \
  uvicorn api.main:app --host 0.0.0.0 --port 8082"]