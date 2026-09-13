#!/bin/bash
set -euo pipefail

# Root-then-drop-to-appuser wrapper around the container's real startup
# sequence (below). Needed specifically for /app/data: it's a host bind
# mount (${MACHINE_DIR}/omnibioai-dev-hub/data:/app/data, see
# docker-compose.yml), so it doesn't exist at image-build time and
# overlays whatever the Dockerfile's own build-time
# `chown -R appuser:appuser /app` set -- with whatever ownership the host
# directory actually has. That host directory predates appuser's uid
# 10001 (owned by the host user's uid 1000), so build_index.py's final
# faiss.write_index() step was hitting "Permission denied" writing
# index.faiss -- confirmed live 2026-09-13: the index build ran to full
# completion (every document loaded and embedded) and only failed at
# this last step, so it looked like a slow/stuck build rather than a
# permission bug, and reproduced identically on every restart-on-failure
# retry since the index was never actually persisted.
#
# Same fix this Dockerfile already applies to /var/log/nginx,
# /var/lib/nginx, /etc/nginx/conf.d (see its own chown -R
# appuser:appuser comment) -- those are baked into the image, so a
# build-time chown is enough. /app/data can't be fixed at build time
# because the bind mount isn't there yet, so it has to happen here, at
# container start, as root, before dropping to appuser for everything
# else (index build, nginx, uvicorn) -- self-re-execs this same script
# as appuser once the chown is done, rather than requiring gosu/su-exec
# (neither is installed in the base image; `runuser` already is).
if [ "$(id -u)" = "0" ]; then
  chown -R appuser:appuser /app/data
  exec runuser -u appuser -- "$0" "$@"
fi

# Everything below is unchanged from the previous inline CMD -- same
# steps, same order, same behavior, now running as appuser (uid 10001)
# after the re-exec above.
_auth_enabled=$(printf '%s' "${AUTH_ENABLED:-}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
if [ "$_auth_enabled" = 'true' ] && [ -z "${JWT_SECRET:-}" ]; then
  echo '❌ AUTH_ENABLED is true but JWT_SECRET is not set -- refusing to start with the internal UI-proxy auth header silently falling back to a placeholder.' >&2
  echo '   Set JWT_SECRET before starting this container.' >&2
  exit 1
fi

echo '⏳ Waiting for Ollama...'
until curl -sf http://ollama:11434/api/tags > /dev/null 2>&1; do
  echo '  ollama not ready, retrying in 3s...'
  sleep 3
done
echo '✅ Ollama is ready'

if [ ! -f /app/data/faiss_index/index.faiss ]; then
  echo '🚀 Building FAISS index...'
  python scripts/build_index.py
else
  echo '✅ Index already exists, skipping build'
fi

# Single line, deliberately: this used to be a Dockerfile CMD JSON
# string, where a trailing `\` before a newline is Docker's own
# line-continuation syntax for the string literal -- Docker's parser
# strips it before bash ever runs, so the multi-line-looking version in
# the old Dockerfile was actually always one line by the time bash saw
# it. A real script file has no such collapsing: bash's single quotes
# preserve a trailing `\` + real newline byte-for-byte, which landed
# straight in devhub.conf and made nginx choke on a stray `\` (confirmed
# live 2026-09-13: "unknown directive \"\\\" ... devhub.conf:3", crash-
# looped the container). Keeping this on one line reproduces the
# original, correct, single-line-with-\n-escapes output exactly.
printf 'server {\n    listen 5173;\n    root /usr/share/nginx/html;\n    index index.html;\n    location / { try_files $uri $uri/ /index.html; }\n    location /api/ { proxy_pass http://127.0.0.1:8082; proxy_set_header Host $host; }\n    location /rag/ { proxy_pass http://127.0.0.1:8082; proxy_set_header X-Devhub-Internal "%s"; }\n    location /health { proxy_pass http://127.0.0.1:8082; }\n    location /status { proxy_pass http://127.0.0.1:8082; }\n    location /docs   { proxy_pass http://127.0.0.1:8082; }\n}\n' "${JWT_SECRET:-}" > /etc/nginx/conf.d/devhub.conf

nginx
echo '🌐 nginx started on port 5173'
exec uvicorn api.main:app --host 0.0.0.0 --port 8082
