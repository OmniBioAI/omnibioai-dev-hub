#!/usr/bin/env bash
# Native, post-publication runtime smoke for one architecture of the
# Dev Hub release image. Resolves the exact architecture digest from the
# REAL published OCI index (never a cached build-job digest), pulls and
# runs exactly that immutable artifact, asserts the container is actually
# running on the architecture it claims, and exercises the smallest
# meaningful, safe, no-production-credential checks.
#
# Dev Hub's own docker-entrypoint.sh refuses to start without BOTH:
#   1. a pre-built, valid, PUBLIC-only FAISS index at /app/data/faiss_index
#      (fails closed, never builds one itself -- by design)
#   2. a reachable Ollama at http://ollama:11434/api/tags (waits forever
#      otherwise)
# Rather than weaken either fail-closed check to make smoke-testing
# easier, this script satisfies both for real:
#   - builds a tiny, real, valid PUBLIC-only index INSIDE the pulled
#     image itself (reusing its own index.vector_store/index.lifecycle
#     code -- the exact same construction already proven in
#     tests/test_startup_safety.py's MAKE_INDEX fixture -- so no extra
#     Python/numpy/faiss install is needed on the runner), written into
#     a disposable named volume mounted at /app/data
#   - runs a real, disposable `ollama/ollama` sidecar (no model pulled --
#     the entrypoint only checks that /api/tags responds, which it does
#     immediately on boot) on a dedicated bridge network so the
#     container's internal DNS lookup of hostname "ollama" resolves
#
# No production credentials, no production database, no production
# platform dependency of any kind.
#
# Usage:
#   native_runtime_smoke.sh <image-repo> <ref-tag> <expect-uname>
#
# Requires: docker buildx, jq, curl.

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <image-repo> <ref-tag> <expect-uname>" >&2
  exit 2
fi

IMAGE="$1"
REF_TAG="$2"
EXPECT_UNAME="$3"
REF="${IMAGE}:${REF_TAG}"

case "$EXPECT_UNAME" in
  x86_64|aarch64) ;;
  *) echo "FAIL: unknown expect-uname '$EXPECT_UNAME' (expected x86_64 or aarch64)" >&2; exit 1 ;;
esac

SUFFIX="$$"
NETWORK="devhub-smoke-net-${SUFFIX}"
VOLUME="devhub-smoke-data-${SUFFIX}"
OLLAMA_CONTAINER="devhub-smoke-ollama-${SUFFIX}"
APP_CONTAINER="devhub-smoke-app-${SUFFIX}"
HOST_PORT=$((20000 + RANDOM % 10000))

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

cleanup() {
  docker logs "$APP_CONTAINER" 2>&1 | tail -150 || true
  docker rm -f "$APP_CONTAINER" "$OLLAMA_CONTAINER" >/dev/null 2>&1 || true
  docker volume rm -f "$VOLUME" >/dev/null 2>&1 || true
  docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== Resolving native linux digest from ${REF} for expected uname ${EXPECT_UNAME} =="
RAW_INDEX="$(docker buildx imagetools inspect "$REF" --raw)" || fail "could not inspect ${REF}"
case "$EXPECT_UNAME" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
esac
DIGEST="$(jq -r --arg arch "$ARCH" '
  .manifests[]
  | select(.platform.architecture == $arch and .platform.os == "linux")
  | select((.annotations["vnd.docker.reference.type"] // "") != "attestation-manifest")
  | .digest
' <<<"$RAW_INDEX" | head -1)"
[ -n "$DIGEST" ] || fail "no linux/${ARCH} runtime manifest found in ${REF}"
echo "Resolved digest: ${DIGEST}"

PULL_REF="${IMAGE}@${DIGEST}"
echo "== Pulling exact immutable artifact ${PULL_REF} =="
docker pull "$PULL_REF" || fail "pull failed for ${PULL_REF}"

echo "== Setting up disposable network and data volume =="
docker network create "$NETWORK" >/dev/null
docker volume create "$VOLUME" >/dev/null

echo "== Building a tiny, real, valid PUBLIC-only fixture index inside the pulled image =="
# Reuses the exact construction proven in tests/test_startup_safety.py's
# MAKE_INDEX -- real VectorStore.add()/.save() + a manifest explicitly
# marked visibility_policy: PUBLIC_ONLY, 3 random 768-dim vectors, zero
# real document content.
FIXTURE_SCRIPT='
import numpy as np
from index.vector_store import VectorStore
from index.lifecycle import artifact_hashes, write_manifest
out = "/app/data/faiss_index"
visibilities = ["PUBLIC", "PUBLIC", "PUBLIC"]
meta = [{"text": f"t{i}", "source": f"repo:d{i}.md@abc", "repo": "repo", "relative_path": f"d{i}.md",
         "source_revision": "abc", "document_id": f"doc{i}", "chunk_id": f"c{i}", "content_hash": "h", "chunk_hash": "ch",
         "content_state": "CURRENT", "verification_state": "CONFIGURED", "bundle": None,
         "citation": {"repository": "repo"}, "visibility": v} for i, v in enumerate(visibilities)]
vs = VectorStore()
vs.add(np.random.default_rng(0).normal(size=(len(meta), 768)).astype("float32"), meta)
vs.save(out)
counts = {k: visibilities.count(k) for k in ("PUBLIC", "INTERNAL", "REVIEW_REQUIRED")}
manifest = {"build_id": "smoke", "build_status": "CLEAN", "artifact_hashes": artifact_hashes(out),
            "visibility_counts": counts, "visibility_policy": "PUBLIC_ONLY"}
write_manifest(out, manifest)
print("fixture index written")
'
docker run --rm -v "${VOLUME}:/app/data" "$PULL_REF" python -c "$FIXTURE_SCRIPT" \
  || fail "fixture index construction failed inside ${PULL_REF}"

echo "== Starting disposable Ollama sidecar (no model pulled; only /api/tags is checked) =="
docker run -d --name "$OLLAMA_CONTAINER" --network "$NETWORK" --network-alias ollama ollama/ollama:latest >/dev/null \
  || fail "failed to start disposable Ollama sidecar"

echo "== Starting disposable Dev Hub container (no production config/secrets) =="
docker run -d --name "$APP_CONTAINER" \
  --network "$NETWORK" \
  -p "${HOST_PORT}:8082" \
  -v "${VOLUME}:/app/data" \
  -e AUTH_ENABLED=false \
  "$PULL_REF" >/dev/null \
  || fail "docker run failed for ${PULL_REF}"

echo "== Asserting native architecture inside the running container =="
ACTUAL_UNAME="$(docker exec "$APP_CONTAINER" uname -m)" || fail "could not exec uname -m in ${APP_CONTAINER}"
[ "$ACTUAL_UNAME" = "$EXPECT_UNAME" ] \
  || fail "container reports '${ACTUAL_UNAME}', expected native '${EXPECT_UNAME}' -- possible emulation"
echo "Native architecture confirmed inside container: ${ACTUAL_UNAME}"

echo "== Waiting for GET /health =="
tries=60
until curl -sf -o /dev/null "http://127.0.0.1:${HOST_PORT}/health"; do
  if ! docker ps --filter "name=${APP_CONTAINER}" --filter status=running -q | grep -q .; then
    fail "${APP_CONTAINER} exited before becoming ready (see captured logs above)"
  fi
  tries=$((tries - 1))
  [ "$tries" -gt 0 ] || fail "Dev Hub /health never became ready"
  sleep 2
done

HEALTH_BODY="$(curl -sf "http://127.0.0.1:${HOST_PORT}/health")"
echo "$HEALTH_BODY" | jq -e '.status == "ok" and .service == "omnibioai-dev-hub"' >/dev/null \
  || fail "GET /health returned unexpected body: ${HEALTH_BODY}"
echo "GET /health -> 200 {\"status\":\"ok\",\"service\":\"omnibioai-dev-hub\",...} OK"

echo "PASS: native linux/${ARCH} Dev Hub runtime smoke succeeded for ${PULL_REF}"
