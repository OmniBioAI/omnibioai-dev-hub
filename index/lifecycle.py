"""Index staging, validation, promotion and rollback helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from index.vector_store import CANONICAL_DIM, VectorStore
from ingestion.trusted import sha256_file, utc_now_iso

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.pkl"
MANIFEST_FILE = "manifest.json"
POINTER_FILE = "CURRENT"
REJECTED_MARKER = "REJECTED"
PUBLIC_ONLY_POLICY = "PUBLIC_ONLY"
VALID_VISIBILITIES = {"PUBLIC", "INTERNAL", "REVIEW_REQUIRED"}
# Fields every indexed chunk must carry for the retrieval/citation contract.
REQUIRED_CHUNK_FIELDS = (
    "repo", "relative_path", "source_revision", "document_id", "chunk_id", "content_hash", "chunk_hash",
    "content_state", "verification_state", "citation",
)


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
    tmp = directory / f"{MANIFEST_FILE}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, directory / MANIFEST_FILE)  # a crash mid-write never leaves a torn manifest


def read_manifest(directory: str | Path) -> dict[str, Any]:
    return json.loads((Path(directory) / MANIFEST_FILE).read_text(encoding="utf-8"))


def _validate_legacy(directory: Path) -> dict[str, Any]:
    """Minimal integrity check for a pre-Phase-18 index (no manifest, no visibility)."""
    if not (directory / INDEX_FILE).exists() or not (directory / METADATA_FILE).exists():
        return {"ok": False, "reason": f"legacy index missing artifacts in {directory}"}
    store = VectorStore()
    if not store.load(str(directory)):
        return {"ok": False, "reason": "legacy vector store did not load"}
    ntotal = store.index.ntotal if store.index else 0
    if store.dim != CANONICAL_DIM or ntotal != len(store.metadata):
        return {"ok": False, "reason": f"legacy index inconsistent: dim={store.dim} ntotal={ntotal} metadata={len(store.metadata)}"}
    return {"ok": True, "legacy": True, "faiss_ntotal": ntotal, "metadata_count": len(store.metadata),
            "dimension": store.dim, "artifact_hashes": artifact_hashes(directory)}


def validate_index_directory(directory: str | Path, *, require_public_only: bool = False,
                             allow_legacy: bool = False) -> dict[str, Any]:
    """Validate an index directory against the Phase 18 contract.

    A manifest declaring visibility_policy == PUBLIC_ONLY makes the artifact
    enforce that itself. allow_legacy accepts a pre-Phase-18 index (only used
    for rollback targets; never for promotion or serving).
    """
    directory = Path(directory)
    rejected = directory / REJECTED_MARKER
    if rejected.exists():
        reason = rejected.read_text(encoding="utf-8").strip().splitlines()[:1]
        return {"ok": False, "reason": f"candidate is marked {REJECTED_MARKER}: {reason[0] if reason else 'no reason recorded'}"}
    if allow_legacy and not (directory / MANIFEST_FILE).exists():
        return _validate_legacy(directory)
    missing = [name for name in (INDEX_FILE, METADATA_FILE, MANIFEST_FILE) if not (directory / name).exists()]
    if missing:
        return {"ok": False, "reason": f"missing artifacts: {missing}"}

    manifest = read_manifest(directory)
    if manifest.get("build_status", "CLEAN") != "CLEAN":
        return {"ok": False, "reason": f"candidate build status is {manifest.get('build_status')}"}
    if manifest.get("visibility_policy") == PUBLIC_ONLY_POLICY:
        require_public_only = True

    actual_hashes = artifact_hashes(directory)
    expected_hashes = manifest.get("artifact_hashes")
    if expected_hashes and expected_hashes != actual_hashes:
        return {"ok": False, "reason": "artifact hashes do not match the manifest (index or metadata changed after build)"}

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
    counts = {v: 0 for v in sorted(VALID_VISIBILITIES)}
    for i, meta in enumerate(store.metadata):
        visibility = meta.get("visibility")
        if visibility not in VALID_VISIBILITIES:
            bad_visibility.append(i)
        else:
            counts[visibility] += 1
            if require_public_only and visibility != "PUBLIC":
                bad_visibility.append(i)
        for field in REQUIRED_CHUNK_FIELDS:
            if not meta.get(field):
                missing_identity.append({"index": i, "field": field})
        if "bundle" not in meta:  # may be None (root-level file) but the key must exist for scope filtering
            missing_identity.append({"index": i, "field": "bundle"})
        if meta.get("source", "").startswith("/"):
            absolute_only.append(i)
    if bad_visibility:
        return {"ok": False, "reason": f"invalid visibility metadata: {bad_visibility[:10]}"}
    if missing_identity:
        return {"ok": False, "reason": f"missing identity metadata: {missing_identity[:10]}"}
    if absolute_only:
        return {"ok": False, "reason": f"absolute-path-only source identities: {absolute_only[:10]}"}
    declared = manifest.get("visibility_counts")
    if declared is not None and {k: declared.get(k, 0) for k in counts} != counts:
        return {"ok": False, "reason": f"manifest visibility_counts {declared} do not match the index {counts}"}
    return {
        "ok": True,
        "faiss_ntotal": ntotal,
        "metadata_count": metadata_count,
        "dimension": store.dim,
        "visibility_counts": counts,
        "public_only": require_public_only,
        "artifact_hashes": actual_hashes,
        "manifest_hash": sha256_file(directory / MANIFEST_FILE),
    }


def candidate_dir(staging_root: str | Path, build_id: str) -> Path:
    return Path(staging_root) / build_id


def _swap_in(source: Path, current_dir: Path, previous_target: Path | None) -> None:
    """Second half of a swap: move `source` into place; if that fails, put the old index back.

    Handles ordinary failures (permissions, disk, missing path). A hard crash
    (SIGKILL, power loss) between the renames is NOT handled -- see
    promote_candidate.
    """
    try:
        os.replace(source, current_dir)
    except Exception as exc:
        if previous_target is not None:
            try:
                os.replace(previous_target, current_dir)
            except Exception as restore_exc:  # noqa: BLE001 -- report the exact recovery state whatever failed
                raise RuntimeError(
                    f"swap failed ({exc!r}) AND restoring the previous index failed ({restore_exc!r}); "
                    f"NO current index at {current_dir}; previous index is at {previous_target}, "
                    f"the new index is still at {source}. Recover by hand: mv {previous_target} {current_dir}"
                ) from exc
        raise


def promote_candidate(candidate: str | Path, current_dir: str | Path, previous_root: str | Path) -> dict[str, Any]:
    """Validate, then swap `candidate` in as `current_dir`, retaining the previous index.

    NOT ATOMIC. Each os.replace is atomic on its own, but the swap is two of
    them (current -> previous, candidate -> current), so:
      * there is a brief moment with no `current_dir`;
      * an ordinary failure of the second rename is undone (the old index is
        moved back), but a HARD crash between the renames (SIGKILL, power loss)
        leaves no `current_dir`: the old index is intact under `previous_root`
        and the candidate intact where it was, and nothing here detects or
        repairs that -- an operator must finish or undo it by hand
        (mv <previous_root>/previous-* <current_dir>  to undo, or
         mv <candidate> <current_dir>  to finish);
      * the manifest is rewritten after the swap (atomically, but after), so a
        crash there leaves a promoted directory still labelled CANDIDATE.
    A loaded process is unaffected (the index is read into memory at startup),
    but anything that starts inside the window sees no index. Both source and
    destination must be on one filesystem. Run it with the service stopped or
    with restarts frozen.
    """
    candidate = Path(candidate)
    current_dir = Path(current_dir)
    previous_root = Path(previous_root)
    validation = validate_index_directory(candidate)
    if not validation["ok"]:
        raise RuntimeError(f"candidate validation failed: {validation['reason']}")

    previous_root.mkdir(parents=True, exist_ok=True)
    build_id = read_manifest(candidate).get("build_id", candidate.name)
    previous_target = None
    previous_hashes: dict[str, str] = {}
    previous_legacy = False
    if current_dir.exists():
        previous_hashes = artifact_hashes(current_dir)
        previous_legacy = not (current_dir / MANIFEST_FILE).exists()
        previous_target = previous_root / f"previous-{utc_now_iso().replace(':', '').replace('-', '')}"
        os.replace(current_dir, previous_target)
    _swap_in(candidate, current_dir, previous_target)

    manifest = read_manifest(current_dir)
    manifest["promotion_status"] = "PROMOTED"
    manifest["promotion_timestamp"] = utc_now_iso()
    manifest["previous_index"] = str(previous_target) if previous_target else None
    manifest["previous_index_artifact_hashes"] = previous_hashes
    manifest["previous_index_legacy"] = previous_legacy
    write_manifest(current_dir, manifest)
    return {"promoted_build_id": build_id, "previous_index": str(previous_target) if previous_target else None}


def rollback(current_dir: str | Path, previous_dir: str | Path, rollback_root: str | Path) -> dict[str, Any]:
    """Swap `previous_dir` back in as `current_dir`, keeping what it replaces.

    Same two-rename, non-atomic caveats as promote_candidate. The target may be
    a legacy (pre-Phase-18) index, which has no manifest; it must still load and
    be self-consistent, and must be byte-identical to what promotion retained
    (hashes recorded in the current manifest).
    """
    current_dir = Path(current_dir)
    previous_dir = Path(previous_dir)
    rollback_root = Path(rollback_root)
    validation = validate_index_directory(previous_dir, allow_legacy=True)
    if not validation["ok"]:
        raise RuntimeError(f"rollback target validation failed: {validation['reason']}")
    if (current_dir / MANIFEST_FILE).exists():
        expected = read_manifest(current_dir).get("previous_index_artifact_hashes") or {}
        if expected and expected != artifact_hashes(previous_dir):
            raise RuntimeError("rollback target validation failed: the retained previous index no longer matches the hashes recorded at promotion")
    rollback_root.mkdir(parents=True, exist_ok=True)
    moved_current = None
    if current_dir.exists():
        moved_current = rollback_root / f"rolled-forward-{utc_now_iso().replace(':', '').replace('-', '')}"
        os.replace(current_dir, moved_current)
    _swap_in(previous_dir, current_dir, moved_current)
    manifest_path = current_dir / MANIFEST_FILE
    if manifest_path.exists():
        manifest = read_manifest(current_dir)
    else:  # legacy index: record the rollback without touching its index files
        manifest = {"schema": "devhub.legacy-index-record.v1", "build_id": "legacy-pre-phase18", "legacy": True,
                    "artifact_hashes": artifact_hashes(current_dir)}
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
