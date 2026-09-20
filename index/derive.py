"""Derive a PUBLIC-only production artifact from a validated mixed-visibility candidate.

Defence in depth: the serving path already filters to PUBLIC, but a filter is
one bug away from a leak. A derived artifact that physically contains only
PUBLIC chunks cannot leak INTERNAL or REVIEW_REQUIRED content whatever the
serving code does.

Vectors are copied bit-for-bit from the source index (never re-embedded), so
the derived artifact's retrieval quality is exactly the source's on PUBLIC rows.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from index.lifecycle import (
    MANIFEST_FILE,
    PUBLIC_ONLY_POLICY,
    VALID_VISIBILITIES,
    artifact_hashes,
    read_manifest,
    validate_index_directory,
    write_manifest,
)
from index.vector_store import VectorStore
from ingestion.trusted import sha256_file, utc_now_iso


def derive_public_only(source_dir: str | Path, out_root: str | Path, build_id: str | None = None) -> dict[str, Any]:
    """Build `<out_root>/<build_id>` containing only the PUBLIC chunks of `source_dir`."""
    source_dir = Path(source_dir)
    out_root = Path(out_root)

    validation = validate_index_directory(source_dir)
    if not validation["ok"]:
        raise RuntimeError(f"source candidate is not valid, refusing to derive from it: {validation['reason']}")
    source_manifest = read_manifest(source_dir)
    if source_manifest.get("visibility_policy") == PUBLIC_ONLY_POLICY:
        raise RuntimeError("source candidate is already PUBLIC_ONLY; derive from the mixed candidate")

    build_id = build_id or f"{source_manifest['build_id']}-public"
    final_dir = out_root / build_id
    if final_dir.exists():
        raise RuntimeError(f"derived candidate already exists: {final_dir}")
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_root / f".{build_id}.partial"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    source = VectorStore()
    if not source.load(str(source_dir)):
        raise RuntimeError("source vector store did not load")

    # Fail closed: anything that is not exactly a valid visibility is an error, never "skipped".
    unknown = [i for i, m in enumerate(source.metadata) if m.get("visibility") not in VALID_VISIBILITIES]
    if unknown:
        raise RuntimeError(f"source has {len(unknown)} chunks with missing/unknown visibility (first rows {unknown[:5]})")
    public_rows = [i for i, m in enumerate(source.metadata) if m["visibility"] == "PUBLIC"]
    if not public_rows:
        raise RuntimeError("source contains no PUBLIC chunks; nothing to derive")

    all_vectors = source.index.reconstruct_n(0, source.index.ntotal)
    vectors = np.ascontiguousarray(all_vectors[public_rows], dtype=np.float32)
    metadata = [source.metadata[i] for i in public_rows]

    derived = VectorStore()
    derived.add(vectors, metadata)
    derived.save(str(tmp_dir))

    excluded_docs: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in sorted(VALID_VISIBILITIES - {"PUBLIC"})}
    excluded_chunks = collections.Counter()
    for m in source.metadata:
        if m["visibility"] != "PUBLIC":
            excluded_chunks[m["visibility"]] += 1
            excluded_docs[m["visibility"]][m["document_id"]] = {"repo": m["repo"], "relative_path": m["relative_path"],
                                                                "source_revision": m["source_revision"]}
    docs = {m["document_id"] for m in metadata}
    repos = sorted({m["repo"] for m in metadata})
    src_hashes = artifact_hashes(source_dir)
    manifest: dict[str, Any] = {
        "schema": "devhub.index-manifest.v1",
        "build_id": build_id,
        "build_timestamp": utc_now_iso(),
        "derived": True,
        "derivation": "public-only-filter; vectors copied bit-for-bit from the source candidate, not re-embedded",
        "derived_from": {
            "build_id": source_manifest["build_id"],
            "manifest_sha256": sha256_file(source_dir / MANIFEST_FILE),
            "artifact_hashes": src_hashes,
            "build_status": source_manifest["build_status"],
            "chunk_count": len(source.metadata),
            "visibility_counts": source_manifest.get("visibility_counts"),
        },
        "visibility_policy": PUBLIC_ONLY_POLICY,
        "metadata_schema_version": source_manifest.get("metadata_schema_version"),
        "source_selection_policy_version": source_manifest.get("source_selection_policy_version"),
        "embedding_provider": source_manifest.get("embedding_provider"),
        "embedding_model": source_manifest.get("embedding_model"),
        "embedding_identity": source_manifest.get("embedding_identity"),
        "embedding_dimension": source_manifest.get("embedding_dimension"),
        "vector_backend": source_manifest.get("vector_backend"),
        "repositories": repos,
        "repository_revisions": {r: next(m["source_revision"] for m in metadata if m["repo"] == r) for r in repos},
        "parsed": len(docs),
        "chunked": len(metadata),
        "embedded": len(metadata),
        "indexed": len(metadata),
        "retrievable": len(metadata),
        "selected_document_count": len(docs),
        "selected_chunk_count": len(metadata),
        "visibility_counts": {"PUBLIC": len(metadata), "INTERNAL": 0, "REVIEW_REQUIRED": 0},
        "excluded_by_policy": {v: {"documents": len(excluded_docs[v]), "chunks": excluded_chunks[v]} for v in excluded_docs},
        "embedding_failures": [],
        "ingestion_failures": source_manifest.get("ingestion_failures", []),
        "missing_repositories": source_manifest.get("missing_repositories", []),
        "embedding_telemetry": {**(source_manifest.get("embedding_telemetry") or {}), "note": "telemetry of the SOURCE build; this artifact was not embedded"},
        "artifact_hashes": artifact_hashes(tmp_dir),
        "evaluation_status": "NOT_EVALUATED",
        "promotion_status": "CANDIDATE",
        "build_status": "CLEAN",
    }
    write_manifest(tmp_dir, manifest)

    # Verify before it can be used: full contract, PUBLIC-only, and bit-exact vector preservation.
    check = validate_index_directory(tmp_dir, require_public_only=True)
    if not check["ok"]:
        shutil.rmtree(tmp_dir)
        raise RuntimeError(f"derived artifact failed validation: {check['reason']}")
    reloaded = VectorStore()
    reloaded.load(str(tmp_dir))
    if not np.array_equal(reloaded.index.reconstruct_n(0, reloaded.index.ntotal), vectors):
        shutil.rmtree(tmp_dir)
        raise RuntimeError("derived vectors are not bit-identical to the source vectors")
    if reloaded.metadata != metadata:
        shutil.rmtree(tmp_dir)
        raise RuntimeError("derived metadata is not identical to the source metadata for the PUBLIC rows")

    os.replace(tmp_dir, final_dir)
    report = {
        "derived_dir": str(final_dir),
        "source_dir": str(source_dir),
        "public_documents": len(docs),
        "public_chunks": len(metadata),
        "excluded": {v: {"documents": len(excluded_docs[v]), "chunks": excluded_chunks[v]} for v in excluded_docs},
        "excluded_documents": {v: sorted(excluded_docs[v].values(), key=lambda d: (d["repo"], d["relative_path"])) for v in excluded_docs},
        "vectors_bit_identical": True,
        "validation": {k: check[k] for k in ("faiss_ntotal", "metadata_count", "dimension", "visibility_counts", "public_only")},
    }
    (out_root / f"{build_id}.derivation-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
