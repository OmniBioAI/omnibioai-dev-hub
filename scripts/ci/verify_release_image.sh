#!/usr/bin/env bash
# Structured, fail-closed remote verification of one published Dev Hub
# release image. Proves, from the REAL published OCI index -- never from
# a mutable tag's human-readable `imagetools inspect` text, always from
# --raw JSON -- that:
#
#   1. the ref resolves to an OCI index
#   2. exactly one linux/amd64 runtime manifest exists in that index
#   3. exactly one linux/arm64 runtime manifest exists in that index
#   4. runtime manifests are distinguished from attestation manifests
#      (an attestation-manifest annotation, not a guess from count/order)
#   5. each runtime manifest's own image config carries the exact
#      org.opencontainers.image.source / .revision / .version expected
#   6. .revision matches ^[0-9a-f]{40}$ and equals the expected commit
#   7. each runtime manifest has at least one SBOM attestation and at
#      least one provenance attestation, each exactly subject-bound to
#      that runtime manifest's own digest (not merely present somewhere
#      in the index)
#
# REF_TAG (the registry tag actually inspected) and EXPECTED_VERSION (the
# org.opencontainers.image.version label value asserted on each runtime
# manifest) are deliberately independent arguments -- they are the SAME
# string only when verifying an already-published :X.Y.Z tag; they differ
# when verifying a pre-publication staging-index-<sha> tag, whose runtime
# images are already labeled with the real target version even though the
# tag itself is not that version.
#
# Any missing or mismatched piece of evidence is a hard failure (exit 1).
# This script intentionally has no "warn and continue" path.
#
# Usage:
#   verify_release_image.sh <image-repo> <ref-tag> <expected-source-url> <expected-revision-sha> <expected-version>
#
# Requires: docker buildx, jq.

set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 <image-repo> <ref-tag> <expected-source-url> <expected-revision-sha> <expected-version>" >&2
  exit 2
fi

IMAGE="$1"
REF_TAG="$2"
EXPECTED_SOURCE="$3"
EXPECTED_REVISION="$4"
EXPECTED_VERSION="$5"
REF="${IMAGE}:${REF_TAG}"

if ! [[ "$EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FAIL: expected-revision-sha argument '$EXPECTED_REVISION' is not an exact 40-character hex SHA" >&2
  exit 1
fi

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

echo "== Verifying ${REF} =="

RAW_INDEX="$(docker buildx imagetools inspect "$REF" --raw)" || fail "could not resolve/inspect ${REF}"

INDEX_MEDIA_TYPE="$(jq -r '.mediaType // empty' <<<"$RAW_INDEX")"
case "$INDEX_MEDIA_TYPE" in
  application/vnd.oci.image.index.v1+json|application/vnd.docker.distribution.manifest.list.v2+json) ;;
  *) fail "${REF} is not an OCI/Docker multi-arch index (mediaType='${INDEX_MEDIA_TYPE}')" ;;
esac

runtime_digest_for() {
  local arch="$1"
  jq -r --arg arch "$arch" '
    .manifests[]
    | select(.platform.architecture == $arch and .platform.os == "linux")
    | select((.annotations["vnd.docker.reference.type"] // "") != "attestation-manifest")
    | .digest
  ' <<<"$RAW_INDEX"
}

AMD64_DIGESTS="$(runtime_digest_for amd64)"
ARM64_DIGESTS="$(runtime_digest_for arm64)"

AMD64_COUNT="$(grep -c . <<<"$AMD64_DIGESTS" || true)"
ARM64_COUNT="$(grep -c . <<<"$ARM64_DIGESTS" || true)"

[ "$AMD64_COUNT" -eq 1 ] || fail "expected exactly 1 linux/amd64 runtime manifest in ${REF}, found ${AMD64_COUNT}"
[ "$ARM64_COUNT" -eq 1 ] || fail "expected exactly 1 linux/arm64 runtime manifest in ${REF}, found ${ARM64_COUNT}"

AMD64_DIGEST="$(echo "$AMD64_DIGESTS" | head -1)"
ARM64_DIGEST="$(echo "$ARM64_DIGESTS" | head -1)"

echo "linux/amd64 runtime digest: ${AMD64_DIGEST}"
echo "linux/arm64 runtime digest: ${ARM64_DIGEST}"

verify_runtime_labels() {
  local arch="$1" digest="$2"
  local labels
  labels="$(docker buildx imagetools inspect "${IMAGE}@${digest}" --format '{{json .Image.Config.Labels}}')" \
    || fail "could not inspect config for ${arch} runtime digest ${digest}"

  local source revision version
  source="$(jq -r '.["org.opencontainers.image.source"] // empty' <<<"$labels")"
  revision="$(jq -r '.["org.opencontainers.image.revision"] // empty' <<<"$labels")"
  version="$(jq -r '.["org.opencontainers.image.version"] // empty' <<<"$labels")"

  [ "$source" = "$EXPECTED_SOURCE" ] \
    || fail "${arch}: org.opencontainers.image.source='${source}', expected '${EXPECTED_SOURCE}'"
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] \
    || fail "${arch}: org.opencontainers.image.revision='${revision}' is not an exact 40-character hex SHA"
  [ "$revision" = "$EXPECTED_REVISION" ] \
    || fail "${arch}: org.opencontainers.image.revision='${revision}', expected '${EXPECTED_REVISION}'"
  [ "$version" = "$EXPECTED_VERSION" ] \
    || fail "${arch}: org.opencontainers.image.version='${version}', expected '${EXPECTED_VERSION}'"

  echo "${arch}: OCI source/revision/version OK"
}

verify_runtime_labels amd64 "$AMD64_DIGEST"
verify_runtime_labels arm64 "$ARM64_DIGEST"

# Attestation manifests: each carries vnd.docker.reference.digest (the
# runtime manifest it is ABOUT) and vnd.docker.reference.type=attestation-manifest.
# Classify each by its in-toto predicate type, then assert, per runtime
# digest, that both an SBOM and a provenance attestation are present AND
# that the attestation's own `subject.digest` is exactly bound to that
# runtime digest -- existence alone is not accepted.
ATTESTATION_ENTRIES="$(jq -c '
  .manifests[]
  | select((.annotations["vnd.docker.reference.type"] // "") == "attestation-manifest")
  | {digest: .digest, subject_of: .annotations["vnd.docker.reference.digest"]}
' <<<"$RAW_INDEX")"

declare -A HAS_SBOM
declare -A HAS_PROVENANCE

while IFS= read -r entry; do
  [ -z "$entry" ] && continue
  att_digest="$(jq -r '.digest' <<<"$entry")"
  subject_of="$(jq -r '.subject_of' <<<"$entry")"

  att_raw="$(docker buildx imagetools inspect "${IMAGE}@${att_digest}" --raw)" \
    || fail "could not inspect attestation manifest ${att_digest}"

  subject_digest="$(jq -r '.subject.digest // empty' <<<"$att_raw")"
  [ "$subject_digest" = "$subject_of" ] \
    || fail "attestation ${att_digest} subject.digest='${subject_digest}' does not match its own reference-digest annotation '${subject_of}' -- untrusted binding"

  predicate_types="$(jq -r '.layers[]?.annotations["in-toto.io/predicate-type"] // empty' <<<"$att_raw")"
  while IFS= read -r predicate; do
    [ -z "$predicate" ] && continue
    case "$predicate" in
      *spdx*) HAS_SBOM["$subject_of"]=1 ;;
      https://slsa.dev/provenance/*) HAS_PROVENANCE["$subject_of"]=1 ;;
    esac
  done <<<"$predicate_types"
done <<<"$ATTESTATION_ENTRIES"

# Explicit per-architecture calls (no arch:digest packing/splitting) --
# AMD64_DIGEST/ARM64_DIGEST are already distinct variables holding the
# complete "sha256:<hex>" digest; passing each straight through avoids
# ever having to split a string that itself contains a colon.
verify_subject_binding() {
  local arch="$1" digest="$2"

  [ "${HAS_SBOM[$digest]:-0}" = "1" ] \
    || fail "${arch} (${digest}): no SPDX SBOM attestation subject-bound to this runtime digest"
  [ "${HAS_PROVENANCE[$digest]:-0}" = "1" ] \
    || fail "${arch} (${digest}): no SLSA provenance attestation subject-bound to this runtime digest"

  echo "${arch}: SBOM and provenance subject-binding OK"
}

verify_subject_binding amd64 "$AMD64_DIGEST"
verify_subject_binding arm64 "$ARM64_DIGEST"

echo "PASS: ${REF} satisfies structured remote verification (both architectures, source/revision/version, SBOM + provenance subject-binding)"
