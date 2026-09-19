"""Build a Phase 18 trusted Dev Hub candidate index.

This script writes only to a candidate directory unless the caller explicitly
uses lifecycle promotion helpers separately. It does not overwrite the current
index.

No checkpoint/resume: the candidate directory is written only once, after
embedding finishes (VectorStore.save + write_manifest), so an interrupted run
leaves no partial candidate artifact to accidentally resume or promote. A
killed/crashed build simply restarts from scratch under a new build_id.
Mid-build resume was considered and deliberately not implemented -- tying a
resumed batch back to an identical source corpus, repository revision set,
schema version, embedding identity and dimension is real complexity that
isn't worth taking on for a build that completes in minutes, and a subtly
wrong resume is a worse outcome than a clean restart.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath("."))

from index.lifecycle import artifact_hashes, candidate_dir, validate_index_directory, write_manifest
from index.vector_store import CANONICAL_DIM, VectorStore
from ingestion.trusted import (
    SourcePolicy,
    build_manifest,
    chunks_for_document,
    discover_documents,
)
from rag.engine import ollama_embed

EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_IDENTITY = "ollama:nomic-embed-text:revision-unknown"

# Conservative defaults for bulk embedding against the local Ollama instance.
# A sustained serial run without pauses was observed to degrade into near-100%
# first/second-attempt failures after a few minutes (consistent with the
# transient CUDA-context pressure ollama_embed's retry already documents),
# even though short bursts and idle-recovery runs were clean. Pausing briefly
# every N chunks lets that pressure dissipate instead of compounding.
#
# These are deliberately NOT solved by adding concurrency: the degradation is
# a function of sustained *load*, and concurrent requests would only load the
# same local Ollama instance harder. Serial embedding is preserved.
DEFAULT_EMBED_BATCH_SIZE = 25
DEFAULT_EMBED_COOLDOWN_SECONDS = 5.0


def normalize_vector(vec):
    arr = np.array(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != CANONICAL_DIM:
        raise ValueError(f"Embedding dimension mismatch: got {arr.shape[0]} expected {CANONICAL_DIM}")
    return arr


def embed_metadata(
    metadata: list[dict],
    *,
    model: str = EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    cooldown_seconds: float = DEFAULT_EMBED_COOLDOWN_SECONDS,
    embed_fn=ollama_embed,
    progress: bool = True,
    second_chance: bool = True,
    second_chance_cooldown_seconds: float | None = None,
) -> dict:
    """Serially embed chunk metadata with bounded batching/backpressure.

    Never increases concurrency against Ollama -- see the module-level
    comment on DEFAULT_EMBED_BATCH_SIZE for why. Every vector's dimension is
    validated as it is produced (not only after the full build), and every
    embedding attempt -- success or failure, first try or retry -- is counted
    so the candidate manifest carries real backpressure telemetry rather than
    just a final pass/fail count. No chunk text is logged.

    Chunks that still fail after ollama_embed's own bounded per-request retry
    (3 attempts) get exactly one further bounded second chance: after a
    longer cooldown (default 3x the batch cooldown, letting whatever
    sustained-load pressure caused the failures fully dissipate), the failed
    subset -- and only that subset -- is retried once through the same
    batched path. This is a second, independent bounded retry layer on top
    of ollama_embed's existing one, not a change to it. Chunks that fail
    again after the second chance are final and are never retried further.
    """
    total_attempts = 0
    retry_count = 0

    def on_attempt(attempt_number: int, ok: bool) -> None:  # noqa: ARG001 -- ok kept for signature clarity
        nonlocal total_attempts, retry_count
        total_attempts += 1
        if attempt_number > 1:
            retry_count += 1

    def run_pass(metas: list[dict], label: str) -> tuple[list[np.ndarray], list[dict], list[dict]]:
        pass_vectors: list[np.ndarray] = []
        pass_indexed: list[dict] = []
        pass_failures: list[dict] = []
        total = len(metas)
        processed = 0
        start = time.monotonic()
        for meta in metas:
            try:
                vec = normalize_vector(embed_fn(meta["text"], model=model, on_attempt=on_attempt))
                pass_vectors.append(vec)
                pass_indexed.append(meta)
            except Exception as exc:  # noqa: BLE001 -- one failed chunk must not corrupt the candidate
                pass_failures.append({"chunk_id": meta["chunk_id"], "source": meta["source"], "error": str(exc)})
            processed += 1

            batch_boundary = processed % batch_size == 0
            finished = processed == total
            if progress and (batch_boundary or finished):
                elapsed = time.monotonic() - start
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (total - processed) / rate if rate > 0 else float("inf")
                print(
                    f"  [embed:{label}] batch {(processed - 1) // batch_size + 1}: "
                    f"{processed}/{total} ({processed / total:.1%})  "
                    f"ok={len(pass_indexed)} failed={len(pass_failures)} retries={retry_count}  "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.0f}s",
                    flush=True,
                )
            if batch_boundary and not finished:
                time.sleep(cooldown_seconds)
        return pass_vectors, pass_indexed, pass_failures

    seen_chunk_ids: set[str] = set()
    duplicate_count = 0
    deduped: list[dict] = []
    for meta in metadata:
        if meta["chunk_id"] in seen_chunk_ids:
            duplicate_count += 1
            continue
        seen_chunk_ids.add(meta["chunk_id"])
        deduped.append(meta)

    embed_start = time.monotonic()
    vectors, indexed_meta, embedding_failures = run_pass(deduped, "pass1")

    second_chance_recovered = 0
    if second_chance and embedding_failures:
        recovery_cooldown = (
            second_chance_cooldown_seconds if second_chance_cooldown_seconds is not None else cooldown_seconds * 3
        )
        if progress:
            print(
                f"  [embed] {len(embedding_failures)} chunk(s) failed the first pass; "
                f"cooling down {recovery_cooldown:.1f}s before a bounded second-chance retry",
                flush=True,
            )
        time.sleep(recovery_cooldown)
        failed_ids = {f["chunk_id"] for f in embedding_failures}
        retry_metas = [m for m in deduped if m["chunk_id"] in failed_ids]
        retry_vectors, retry_indexed, retry_failures = run_pass(retry_metas, "retry")
        vectors.extend(retry_vectors)
        indexed_meta.extend(retry_indexed)
        second_chance_recovered = len(retry_indexed)
        embedding_failures = retry_failures  # only chunks still failing after the second chance are final

    elapsed_total = time.monotonic() - embed_start
    return {
        "vectors": vectors,
        "indexed_meta": indexed_meta,
        "embedding_failures": embedding_failures,
        "stats": {
            "total_embedding_attempts": total_attempts,
            "successful_embeddings": len(indexed_meta),
            "failed_embeddings": len(embedding_failures),
            "retry_count": retry_count,
            "second_chance_recovered": second_chance_recovered,
            "duplicate_chunks_removed": duplicate_count,
            "batch_size": batch_size,
            "cooldown_seconds": cooldown_seconds,
            "embedding_elapsed_seconds": round(elapsed_total, 2),
        },
    }


def build_candidate(repo_base: str, staging_root: str, build_id: str | None = None, repo_names: list[str] | None = None,
                     *, batch_size: int = DEFAULT_EMBED_BATCH_SIZE, cooldown_seconds: float = DEFAULT_EMBED_COOLDOWN_SECONDS) -> dict:
    build_id = build_id or f"devhub-{uuid.uuid4().hex[:12]}"
    out_dir = candidate_dir(staging_root, build_id)
    if out_dir.exists():
        raise RuntimeError(f"candidate directory already exists: {out_dir}")
    build_start = time.monotonic()

    policy = SourcePolicy(repository_names=repo_names) if repo_names is not None else SourcePolicy()
    docs, discovery_stats = discover_documents(repo_base, policy)
    metadata = []
    for doc in docs:
        metadata.extend(chunks_for_document(doc, build_id, EMBEDDING_MODEL))

    manifest = build_manifest(
        build_id,
        metadata,
        discovery_stats,
        embedding_provider=EMBEDDING_PROVIDER,
        embedding_model=EMBEDDING_MODEL,
        embedding_identity=EMBEDDING_IDENTITY,
        embedding_dimension=CANONICAL_DIM,
    )
    manifest["parsed"] = len(docs)
    manifest["chunked"] = len(metadata)

    embed_result = embed_metadata(metadata, model=EMBEDDING_MODEL, batch_size=batch_size, cooldown_seconds=cooldown_seconds)
    vectors = embed_result["vectors"]
    indexed_meta = embed_result["indexed_meta"]
    embedding_failures = embed_result["embedding_failures"]
    embed_stats = embed_result["stats"]

    if not vectors:
        raise RuntimeError("no vectors generated; candidate index not written")

    store = VectorStore()
    store.add(np.vstack(vectors).astype(np.float32), indexed_meta)
    store.save(str(out_dir))

    manifest["embedded"] = len(indexed_meta)
    manifest["indexed"] = len(indexed_meta)
    manifest["retrievable"] = sum(1 for m in indexed_meta if m.get("visibility") == "PUBLIC")
    manifest["selected_chunk_count"] = len(indexed_meta)
    manifest["embedding_failures"] = embedding_failures
    manifest["embedding_telemetry"] = embed_stats
    manifest["duplicate_chunks_removed"] = embed_stats["duplicate_chunks_removed"]
    # Strict policy for this trusted rebuild: a candidate with ANY unresolved
    # embedding failure is never CLEAN, regardless of how small a fraction of
    # the corpus it represents. required chunks == successfully embedded
    # chunks, or the candidate does not become promotable.
    if embedding_failures:
        manifest["build_status"] = "EMBEDDING_FAILURES"
    elif manifest.get("ingestion_failures"):
        manifest["build_status"] = "INGESTION_FAILURES"
    elif manifest.get("missing_repositories"):
        manifest["build_status"] = "MISSING_REPOSITORIES"
    else:
        manifest["build_status"] = "CLEAN"
    manifest["artifact_hashes"] = artifact_hashes(out_dir)
    manifest["build_elapsed_seconds"] = round(time.monotonic() - build_start, 2)
    write_manifest(out_dir, manifest)

    validation = validate_index_directory(out_dir)
    if not validation["ok"]:
        raise RuntimeError(f"candidate integrity failed: {validation['reason']}")
    manifest["artifact_hashes"] = artifact_hashes(out_dir)
    write_manifest(out_dir, manifest)
    return {"candidate_dir": str(out_dir), "manifest": manifest, "validation": validation}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-base", default=os.environ.get("REPO_BASE", "/home/manish/Desktop/machine"))
    parser.add_argument("--staging-root", default=str(Path(__file__).resolve().parents[1] / "data" / "faiss_candidates"))
    parser.add_argument("--build-id")
    parser.add_argument(
        "--embed-batch-size", type=int,
        default=int(os.environ.get("DEVHUB_EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE)),
    )
    parser.add_argument(
        "--embed-cooldown-seconds", type=float,
        default=float(os.environ.get("DEVHUB_EMBED_COOLDOWN_SECONDS", DEFAULT_EMBED_COOLDOWN_SECONDS)),
    )
    args = parser.parse_args()
    result = build_candidate(
        args.repo_base, args.staging_root, args.build_id,
        batch_size=args.embed_batch_size, cooldown_seconds=args.embed_cooldown_seconds,
    )
    print(result["candidate_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
