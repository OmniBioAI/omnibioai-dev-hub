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

# NOT `USER appuser` here (unlike before): /app/data is a host bind mount
# (see docker-compose.yml), so it doesn't exist yet at build time and the
# chown -R appuser:appuser /app above never actually reaches it -- it
# gets whatever ownership the host directory has once mounted at runtime,
# which was uid 1000 (the host user), not appuser's uid 10001, causing
# build_index.py's index write to fail with Permission denied (see
# docker-entrypoint.sh's own comment for the full incident). The image
# now starts as root so the entrypoint can chown that specific bind
# mount at container start, then drops to appuser itself before running
# anything else -- see docker-entrypoint.sh.
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8082 5173

# Startup sequence:
#   1. Fail fast if AUTH_ENABLED=true but JWT_SECRET is unset/empty --
#      refuse to start rather than silently baking a well-known "change-me"
#      placeholder into nginx's X-Devhub-Internal header (the internal
#      UI-proxy auth path). This runs BEFORE nginx starts, and before the
#      Ollama wait / index build, so a misconfigured container fails
#      immediately instead of coming up looking like it's working. Mirrors
#      api/auth.py's validate_auth_config() check on the FastAPI side --
#      same AUTH_ENABLED=true + JWT_SECRET-unset condition, same fail-closed
#      intent, enforced here because this is an architecturally separate
#      code path (shell/nginx-config-generation vs. Python/JWT-validation).
#   2. Wait for Ollama to be ready (it starts in parallel via depends_on)
#   3. Build FAISS index on first run only (skipped if index already exists)
#   4. Generate nginx's devhub.conf with the runtime JWT_SECRET baked in as
#      the internal proxy header (see api/auth.py) -- can't be done at
#      build time since the secret is only known at container start
#   5. Start nginx (serves UI on 5173)
#   6. Start FastAPI (serves API on 8082)
# (Now lives in docker-entrypoint.sh, unchanged in order or behavior --
# moved out of an inline CMD string so it could gain a root-first step
# for the /app/data chown above, without nesting another layer of shell
# quoting inside a Dockerfile CMD array.)
CMD ["/app/docker-entrypoint.sh"]