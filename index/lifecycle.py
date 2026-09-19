"""Index staging, validation, promotion and rollback helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ingestion.trusted import sha256_file, utc_now_iso
from index.vector_store import CANONICAL_DIM, VectorStore

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.pkl"
MANIFEST_FILE = "manifest.json"
POINTER_FILE = "CURRENT"


def artifact_hashes(directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    hashes: dict[str, str] = {}
    for name in (INDEX_FILE, METADATA_FILE):
        path = directory / name
        if path.exists():
            hashes[name] = sha256_file(path)
    return hashes


def write_manifest(directory: str | Path, manifest: dict[str, Any]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(directory: str | Path) -> dict[str, Any]:
    return json.loads((Path(directory) / MANIFEST_FILE).read_text(encoding="utf-8"))


def validate_index_directory(directory: str | Path, *, require_public_only: bool = False) -> dict[str, Any]:
    directory = Path(directory)
    missing = [name for name in (INDEX_FILE, METADATA_FILE, MANIFEST_FILE) if not (directory / name).exists()]
    if missing:
        return {"ok": False, "reason": f"missing artifacts: {missing}"}

    manifest = read_manifest(directory)
    if manifest.get("build_status", "CLEAN") != "CLEAN":
        return {"ok": False, "reason": f"candidate build status is {manifest.get('build_status')}"}

    store = VectorStore()
    if not store.load(str(directory)):
        return {"ok": False, "reason": "vector store did not load"}
    ntotal = store.index.ntotal if store.index else 0
    metadata_count = len(store.metadata)
    if store.dim != CANONICAL_DIM:
        return {"ok": False, "reason": f"dimension mismatch: {store.dim}"}
    if ntotal != metadata_count:
        return {"ok": False, "reason": f"ntotal metadata mismatch: {ntotal} != {metadata_count}"}

    bad_visibility = []
    missing_identity = []
    absolute_only = []
    for i, meta in enumerate(store.metadata):
        visibility = meta.get("visibility")
        if visibility not in {"PUBLIC", "INTERNAL", "REVIEW_REQUIRED"}:
            bad_visibility.append(i)
        if require_public_only and visibility != "PUBLIC":
            bad_visibility.append(i)
        for field in ("repo", "relative_path", "source_revision", "document_id", "chunk_id", "content_hash", "chunk_hash"):
            if not meta.get(field):
                missing_identity.append({"index": i, "field": field})
        if meta.get("source", "").startswith("/"):
            absolute_only.append(i)
    if bad_visibility:
        return {"ok": False, "reason": f"invalid visibility metadata: {bad_visibility[:10]}"}
    if missing_identity:
        return {"ok": False, "reason": f"missing identity metadata: {missing_identity[:10]}"}
    if absolute_only:
        return {"ok": False, "reason": f"absolute-path-only source identities: {absolute_only[:10]}"}
    return {
        "ok": True,
        "faiss_ntotal": ntotal,
        "metadata_count": metadata_count,
        "dimension": store.dim,
        "artifact_hashes": artifact_hashes(directory),
        "manifest_hash": sha256_file(directory / MANIFEST_FILE),
    }


def candidate_dir(staging_root: str | Path, build_id: str) -> Path:
    return Path(staging_root) / build_id


def promote_candidate(candidate: str | Path, current_dir: str | Path, previous_root: str | Path) -> dict[str, Any]:
    candidate = Path(candidate)
    current_dir = Path(current_dir)
    previous_root = Path(previous_root)
    validation = validate_index_directory(candidate)
    if not validation["ok"]:
        raise RuntimeError(f"candidate validation failed: {validation['reason']}")

    previous_root.mkdir(parents=True, exist_ok=True)
    build_id = read_manifest(candidate).get("build_id", candidate.name)
    previous_target = None
    if current_dir.exists():
        previous_target = previous_root / f"previous-{utc_now_iso().replace(':', '').replace('-', '')}"
        os.replace(current_dir, previous_target)
    os.replace(candidate, current_dir)

    manifest = read_manifest(current_dir)
    manifest["promotion_status"] = "PROMOTED"
    manifest["promotion_timestamp"] = utc_now_iso()
    manifest["previous_index"] = str(previous_target) if previous_target else None
    write_manifest(current_dir, manifest)
    return {"promoted_build_id": build_id, "previous_index": str(previous_target) if previous_target else None}


def rollback(current_dir: str | Path, previous_dir: str | Path, rollback_root: str | Path) -> dict[str, Any]:
    current_dir = Path(current_dir)
    previous_dir = Path(previous_dir)
    rollback_root = Path(rollback_root)
    validation = validate_index_directory(previous_dir)
    if not validation["ok"]:
        raise RuntimeError(f"rollback target validation failed: {validation['reason']}")
    rollback_root.mkdir(parents=True, exist_ok=True)
    moved_current = None
    if current_dir.exists():
        moved_current = rollback_root / f"rolled-forward-{utc_now_iso().replace(':', '').replace('-', '')}"
        os.replace(current_dir, moved_current)
    os.replace(previous_dir, current_dir)
    manifest = read_manifest(current_dir)
    manifest["promotion_status"] = "ROLLED_BACK"
    manifest["rollback_timestamp"] = utc_now_iso()
    manifest["rolled_forward_index"] = str(moved_current) if moved_current else None
    write_manifest(current_dir, manifest)
    return {"rollback_build_id": manifest.get("build_id"), "moved_current": str(moved_current) if moved_current else None}


def failed_build_preserves_current(candidate: str | Path, current_dir: str | Path) -> bool:
    candidate = Path(candidate)
    current_dir = Path(current_dir)
    current_hashes = artifact_hashes(current_dir) if current_dir.exists() else {}
    if candidate.exists():
        shutil.rmtree(candidate)
    return current_hashes == (artifact_hashes(current_dir) if current_dir.exists() else {})
