#!/bin/bash
set -euo pipefail

# Production always runs directly as UID 10001. This entrypoint never
# changes ownership or permissions on mounted host artifacts.
if [ "$(id -u)" != "10001" ]; then
  echo 'Dev Hub must run as UID 10001; refusing startup.' >&2
  exit 1
fi

# Fail closed if invoked under a different identity; no root re-exec path.
_auth_enabled=$(printf '%s' "${AUTH_ENABLED:-}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
if [ "$_auth_enabled" = 'true' ]; then
  if [ -z "${JWT_SECRET:-}" ]; then
    echo '❌ AUTH_ENABLED is true but JWT_SECRET is not set -- refusing to start.' >&2
    echo '   Set JWT_SECRET before starting this container.' >&2
    exit 1
  fi
  case "${JWT_SECRET}" in
    change-me|change-me-in-production)
      echo '❌ JWT_SECRET uses an insecure placeholder -- refusing to start.' >&2
      exit 1
      ;;
  esac
  if ! python -c 'from api.secret_config import load_proxy_secret; load_proxy_secret()' >/dev/null 2>&1; then
    echo 'Dev Hub proxy secret file is missing or invalid; refusing startup.' >&2
    exit 1
  fi
fi

# Production serves a pre-built, explicitly promoted, PUBLIC-only index. Startup
# NEVER builds, rebuilds or promotes one: construction and promotion are explicit
# lifecycle operations (index.lifecycle), not side effects of a restart. It used
# to run build_index.py inline here when index.faiss was missing, which blocked
# the service for ~110 minutes and, since Phase 18, produced a staging candidate
# that was never served anyway.
#   valid index present     -> start normally
#   index absent            -> fail fast, clear error
#   index invalid/incomplete-> fail fast (legacy, tampered, non-PUBLIC-only, rejected...)
#   unpromoted candidate    -> ignored; only /app/data/faiss_index is ever served
# NB: with a restart policy of on-failure a failing start will loop, so freeze
# restarts before any index maintenance (see the promotion runbook).
_index_dir=/app/data/faiss_index
python -c 'from index.runtime_path import validate_runtime_index_root; validate_runtime_index_root("/app/data/faiss_index")' || exit 1
if [ ! -f "$_index_dir/index.faiss" ]; then
  echo "❌ No production index at $_index_dir (index.faiss missing)." >&2
  echo '   Refusing to start: the service never builds or promotes an index implicitly.' >&2
  echo '   Promote a validated PUBLIC-only candidate with index.lifecycle.promote_candidate.' >&2
  exit 1
fi
if ! python scripts/verify_index.py "$_index_dir" --require-public-only; then
  echo "❌ $_index_dir failed validation (see INDEX INVALID above). Refusing to start." >&2
  echo '   Roll back with index.lifecycle.rollback, or promote a valid PUBLIC-only candidate.' >&2
  exit 1
fi

echo '⏳ Waiting for Ollama...'
until curl -sf http://ollama:11434/api/tags > /dev/null 2>&1; do
  echo '  ollama not ready, retrying in 3s...'
  sleep 3
done
echo '✅ Ollama is ready'

_proxy_secret=$(python -c 'from api.secret_config import load_proxy_secret; print(load_proxy_secret(), end="")')
_frame_policy="frame-ancestors 'self' https://webstudio.omnibioai.org"
umask 077
printf 'server {\n    listen 5173;\n    root /usr/share/nginx/html;\n    index index.html;\n    add_header Content-Security-Policy "%s" always;\n    location / { try_files $uri $uri/ /index.html; }\n    location /api/ { proxy_pass http://127.0.0.1:8082; proxy_set_header Host $host; }\n    location /rag/ { proxy_pass http://127.0.0.1:8082; proxy_set_header X-Devhub-Internal "%s"; }\n    location /health { proxy_pass http://127.0.0.1:8082; }\n    location /status { proxy_pass http://127.0.0.1:8082; }\n    location /docs   { proxy_pass http://127.0.0.1:8082; }\n}\n' "$_frame_policy" "$_proxy_secret" > /etc/nginx/conf.d/devhub.conf
unset _proxy_secret _frame_policy

nginx
echo '🌐 nginx started on port 5173'
exec uvicorn api.main:app --host 0.0.0.0 --port 8082
