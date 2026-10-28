# OmniBioAI — Developer Hub
# Purpose: Build the developer hub API and nginx-served web UI.
# Author: Manish Kumar <manish@omnibioai.org>

# ── Stage 1: Build Vite UI ────────────────────────────────────────────────────
# Always build on the host (BuildKit) platform, never the target platform:
# the UI build output (static JS/CSS/HTML) is architecture-independent, so
# building it once per target arch would be wasted native-runner work. Same
# pattern already proven in omnibioai-control-center's own Dockerfile.
FROM --platform=$BUILDPLATFORM node:20-bookworm-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS ui-builder
WORKDIR /ui
ARG VITE_STUDIO_ORIGIN=
ENV VITE_STUDIO_ORIGIN=${VITE_STUDIO_ORIGIN}
COPY omnibioai-dev-hub-ui/package*.json ./
COPY omnibioai-dev-hub-ui/vendor/omnibioai-design-tokens/ ./vendor/omnibioai-design-tokens/
RUN npm ci
# Application source
COPY omnibioai-dev-hub-ui/ ./
RUN npm run build

# ── Stage 2: Python API + nginx ───────────────────────────────────────────────
# No --platform override here: the native per-architecture CI build invokes
# buildx with --platform linux/amd64 or linux/arm64 on a runner that is
# natively that architecture (no QEMU) -- the previous hardcoded
# --platform=linux/arm64 meant this stage could only ever be cross-built
# from an aarch64 host, and would fail outright on a native amd64 runner.
FROM ghcr.io/omnibioai/omnibioai-base:1.0.0@sha256:ded14114d89200c118682db5018fdb1775fd5375648bcd6f4abb04d70296e1ca AS backend

ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source=https://github.com/OmniBioAI/omnibioai-dev-hub \
      org.opencontainers.image.revision=${SOURCE_REVISION}

# curl needed for Ollama readiness check; nginx for UI serving
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx=1.26.3-3+deb13u9 \
        curl=8.14.1-2+deb13u7 \
    && apt-get purge -y --no-install-recommends \
        docker-ce-cli \
        libmariadb-dev \
        libmariadb-dev-compat \
        libmariadb3 \
        mariadb-common \
        linux-libc-dev \
        libxml2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# faiss-cpu must be installed BEFORE sentence-transformers to avoid conflicts
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy==1.26.4" "faiss-cpu==1.15.0" && \
    pip install --no-cache-dir -r requirements.txt

# The shared base also carries development/test tooling and an unrelated
# Django application stack. They are not runtime dependencies of Dev Hub;
# remove them from this service image after installing the pinned runtime set.
RUN python -m pip uninstall -y Django gunicorn pip setuptools wheel

# Copy application source
COPY api/        ./api/
COPY embeddings/ ./embeddings/
COPY index/      ./index/
COPY ingestion/  ./ingestion/
# api/routes/discovery.py imports integrations.tes_discovery (and the
# other integrations.* modules) -- this line was missing entirely, so
# the image would crash on startup with ModuleNotFoundError as soon as
# api.main imported api.routes.discovery. Confirmed by a real native
# ARM64 build + run: this was the next failure surfaced immediately
# after fixing the unrelated cryptography SIGILL below.
COPY integrations/ ./integrations/
COPY processing/ ./processing/
COPY retrieval/  ./retrieval/
COPY rag/        ./rag/
COPY scripts/    ./scripts/
COPY configs/    ./configs/

# Copy built UI from builder stage
COPY --from=ui-builder /ui/dist /usr/share/nginx/html

# Remove the default nginx site; runtime config is generated from the
# file-backed internal-proxy credential.
RUN rm -f /etc/nginx/sites-enabled/default \
          /etc/nginx/sites-available/default

# ── Non-root runtime user ──────────────────────────────────────────────────
# Runtime writes this container actually performs, and what each needs:
#   /app/data/faiss_index/  -- mounted read-only; startup verifies and app reads
#   /app/logs/, /app/cache/ -- private application scratch locations
#   /var/log/nginx/         -- access.log / error.log (nginx opens these at
#                              startup, not lazily -- must be writable then)
#   /var/lib/nginx/         -- client_body/proxy/fastcgi/... temp dirs, created
#                              by nginx workers on first request
#   /etc/nginx/conf.d/      -- runtime config is generated here from the secret file
# nginx runs as appuser, so the `user` directive is dropped and its pid is
# redirected to /tmp.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/faiss_index /app/logs /app/cache \
    && chown -R appuser:appuser /app \
    && chmod -R go-w /app \
    && chown -R appuser:appuser /var/log/nginx /var/lib/nginx /etc/nginx/conf.d \
    && sed -i '/^user www-data;$/d' /etc/nginx/nginx.conf \
    && sed -i 's#pid /run/nginx.pid;#pid /tmp/nginx.pid;#' /etc/nginx/nginx.conf

# Runtime configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    REPO_BASE=/repos

# The image defaults to UID/GID 10001. Compose independently pins the same
# identity and mounts only the production index directory read-only.
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod 0755 /app/docker-entrypoint.sh

# Remove upstream test-only private keys from the runtime artifact.
RUN rm -f /usr/local/lib/python3.12/site-packages/tornado/test/test.key \
          /usr/local/lib/python3.12/site-packages/twisted/test/server.pem

USER 10001:10001

EXPOSE 8082 5173

# Startup sequence:
#   1. Validate JWT settings and the independent file-backed proxy secret.
#   2. Validate the canonical, PUBLIC-only promoted index; never build or promote.
#   3. Wait for Ollama, then start nginx and FastAPI.
CMD ["/app/docker-entrypoint.sh"]
