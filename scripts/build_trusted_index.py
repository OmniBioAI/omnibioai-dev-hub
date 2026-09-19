"""Build a Phase 18 trusted Dev Hub candidate index.

This script writes only to a candidate directory unless the caller explicitly
uses lifecycle promotion helpers separately. It does not overwrite the current
index.
"""

from __future__ import annotations

import argparse
import os
import sys
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


def normalize_vector(vec):
    arr = np.array(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != CANONICAL_DIM:
        raise ValueError(f"Embedding dimension mismatch: got {arr.shape[0]} expected {CANONICAL_DIM}")
    return arr


def build_candidate(repo_base: str, staging_root: str, build_id: str | None = None, repo_names: list[str] | None = None) -> dict:
    build_id = build_id or f"devhub-{uuid.uuid4().hex[:12]}"
    out_dir = candidate_dir(staging_root, build_id)
    if out_dir.exists():
        raise RuntimeError(f"candidate directory already exists: {out_dir}")

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

    vectors = []
    indexed_meta = []
    seen_chunk_ids: set[str] = set()
    duplicate_count = 0
    embedding_failures = []
    for meta in metadata:
        if meta["chunk_id"] in seen_chunk_ids:
            duplicate_count += 1
            continue
        seen_chunk_ids.add(meta["chunk_id"])
        try:
            vectors.append(normalize_vector(ollama_embed(meta["text"], model=EMBEDDING_MODEL)))
            indexed_meta.append(meta)
        except Exception as exc:  # noqa: BLE001 -- one failed chunk must not corrupt the candidate
            embedding_failures.append({"chunk_id": meta["chunk_id"], "source": meta["source"], "error": str(exc)})

    if not vectors:
        raise RuntimeError("no vectors generated; candidate index not written")

    store = VectorStore()
    store.add(np.vstack(vectors).astype(np.float32), indexed_meta)
    store.save(str(out_dir))

    manifest["embedded"] = len(indexed_meta)
    manifest["indexed"] = len(indexed_meta)
    manifest["retrievable"] = sum(1 for m in indexed_meta if m.get("visibility") == "PUBLIC")
    manifest["selected_chunk_count"] = len(indexed_meta)
    manifest["duplicate_chunks_removed"] = duplicate_count
    manifest["embedding_failures"] = embedding_failures
    if embedding_failures:
        manifest["build_status"] = "EMBEDDING_FAILURES"
    elif manifest.get("ingestion_failures"):
        manifest["build_status"] = "INGESTION_FAILURES"
    elif manifest.get("missing_repositories"):
        manifest["build_status"] = "MISSING_REPOSITORIES"
    else:
        manifest["build_status"] = "CLEAN"
    manifest["artifact_hashes"] = artifact_hashes(out_dir)
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
    args = parser.parse_args()
    result = build_candidate(args.repo_base, args.staging_root, args.build_id)
    print(result["candidate_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
