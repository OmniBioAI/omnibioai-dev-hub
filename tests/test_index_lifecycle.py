"""Tests for Phase 18 staging, integrity, promotion and rollback helpers."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

mock_faiss = MagicMock()
sys.modules["faiss"] = mock_faiss

from index.lifecycle import (
    failed_build_preserves_current,
    promote_candidate,
    read_manifest,
    rollback,
    validate_index_directory,
    write_manifest,
)


def _index_dir(path, build_id="b1", metadata=None):
    path.mkdir(parents=True)
    (path / "index.faiss").write_bytes(b"index")
    (path / "metadata.pkl").write_bytes(b"metadata")
    write_manifest(path, {"build_id": build_id, "promotion_status": "CANDIDATE"})
    return path


def _valid_metadata(visibility="PUBLIC"):
    return [{
        "repo": "repo",
        "relative_path": "README.md",
        "source_revision": "abc",
        "document_id": "doc",
        "chunk_id": "chunk",
        "content_hash": "content",
        "chunk_hash": "chunkhash",
        "content_state": "CURRENT",
        "verification_state": "CONFIGURED",
        "citation": {"repository": "repo"},
        "bundle": None,
        "visibility": visibility,
        "text": "hello",
        "source": "repo:README.md@abc",
    }]


def test_validate_index_directory_rejects_metadata_count_mismatch(tmp_path):
    directory = _index_dir(tmp_path / "candidate")
    store = MagicMock()
    store.index.ntotal = 2
    store.metadata = _valid_metadata()
    store.dim = 768
    with patch("index.lifecycle.VectorStore") as cls:
        cls.return_value.load.return_value = True
        cls.return_value.index = store.index
        cls.return_value.metadata = store.metadata
        cls.return_value.dim = store.dim
        result = validate_index_directory(directory)
    assert not result["ok"]
    assert "ntotal metadata mismatch" in result["reason"]


def test_validate_index_directory_rejects_missing_visibility(tmp_path):
    directory = _index_dir(tmp_path / "candidate")
    with patch("index.lifecycle.VectorStore") as cls:
        cls.return_value.load.return_value = True
        cls.return_value.index.ntotal = 1
        cls.return_value.metadata = [{**_valid_metadata()[0], "visibility": None}]
        cls.return_value.dim = 768
        result = validate_index_directory(directory)
    assert not result["ok"]
    assert "invalid visibility" in result["reason"]


def test_validate_index_directory_passes_complete_metadata(tmp_path):
    directory = _index_dir(tmp_path / "candidate")
    with patch("index.lifecycle.VectorStore") as cls:
        cls.return_value.load.return_value = True
        cls.return_value.index.ntotal = 1
        cls.return_value.metadata = _valid_metadata()
        cls.return_value.dim = 768
        result = validate_index_directory(directory)
    assert result["ok"]
    assert result["faiss_ntotal"] == 1
    assert result["metadata_count"] == 1


def test_promote_candidate_retains_previous_and_rollback_restores(tmp_path):
    current = _index_dir(tmp_path / "current", build_id="old")
    candidate = _index_dir(tmp_path / "candidate", build_id="new")
    previous_root = tmp_path / "previous"
    rollback_root = tmp_path / "rolled"

    with patch("index.lifecycle.validate_index_directory", return_value={"ok": True}):
        promoted = promote_candidate(candidate, current, previous_root)
        assert promoted["promoted_build_id"] == "new"
        assert current.exists()
        assert previous_root.exists()
        assert read_manifest(current)["promotion_status"] == "PROMOTED"

        rolled = rollback(current, promoted["previous_index"], rollback_root)
        assert rolled["rollback_build_id"] == "old"
        assert read_manifest(current)["promotion_status"] == "ROLLED_BACK"


def test_failed_build_preserves_current(tmp_path):
    current = _index_dir(tmp_path / "current", build_id="old")
    candidate = _index_dir(tmp_path / "candidate", build_id="bad")

    assert failed_build_preserves_current(candidate, current)
    assert current.exists()
    assert not candidate.exists()


def test_validate_index_directory_rejects_non_clean_build_status(tmp_path):
    directory = _index_dir(tmp_path / "candidate")
    manifest = read_manifest(directory)
    manifest["build_status"] = "EMBEDDING_FAILURES"
    write_manifest(directory, manifest)

    result = validate_index_directory(directory)

    assert not result["ok"]
    assert "EMBEDDING_FAILURES" in result["reason"]


def test_rejected_candidate_is_refused_and_current_is_untouched(tmp_path):
    current = _index_dir(tmp_path / "current", build_id="old")
    candidate = _index_dir(tmp_path / "candidate", build_id="bad")
    (candidate / "REJECTED").write_text("incomplete metadata (no bundle field)\n")
    before = read_manifest(current)

    result = validate_index_directory(candidate)
    assert not result["ok"] and "REJECTED" in result["reason"] and "incomplete metadata" in result["reason"]
    with pytest.raises(RuntimeError, match="REJECTED"):
        promote_candidate(candidate, current, tmp_path / "previous")
    assert read_manifest(current) == before and candidate.exists()


def test_rejected_marker_also_blocks_use_as_rollback_target(tmp_path):
    current = _index_dir(tmp_path / "current", build_id="new")
    previous = _index_dir(tmp_path / "previous", build_id="rejected-one")
    (previous / "REJECTED").write_text("do not use\n")
    with pytest.raises(RuntimeError, match="REJECTED"):
        rollback(current, previous, tmp_path / "rolled")
    assert current.exists() and previous.exists()


# ---- Phase 18 hardening: hashes, PUBLIC_ONLY policy, legacy rollback, failed swap --------

import os

from index.lifecycle import artifact_hashes


def _patched_store(metadata, ntotal=None):
    p = patch("index.lifecycle.VectorStore")
    cls = p.start()
    cls.return_value.load.return_value = True
    cls.return_value.index.ntotal = len(metadata) if ntotal is None else ntotal
    cls.return_value.metadata = metadata
    cls.return_value.dim = 768
    return p


def test_validate_rejects_artifacts_that_changed_after_the_manifest_was_written(tmp_path):
    directory = _index_dir(tmp_path / "candidate")
    manifest = read_manifest(directory)
    manifest["artifact_hashes"] = artifact_hashes(directory)
    write_manifest(directory, manifest)
    (directory / "index.faiss").write_bytes(b"tampered")
    p = _patched_store(_valid_metadata())
    try:
        result = validate_index_directory(directory)
    finally:
        p.stop()
    assert not result["ok"] and "artifact hashes do not match" in result["reason"]


def test_public_only_manifest_enforces_itself_and_rejects_any_internal_chunk(tmp_path):
    directory = _index_dir(tmp_path / "candidate")
    manifest = read_manifest(directory)
    manifest["visibility_policy"] = "PUBLIC_ONLY"
    write_manifest(directory, manifest)
    meta = _valid_metadata() + _valid_metadata("INTERNAL")
    p = _patched_store(meta)
    try:
        result = validate_index_directory(directory)
    finally:
        p.stop()
    assert not result["ok"] and "invalid visibility" in result["reason"]


def test_public_only_manifest_accepts_all_public(tmp_path):
    directory = _index_dir(tmp_path / "candidate")
    manifest = read_manifest(directory)
    manifest.update({"visibility_policy": "PUBLIC_ONLY", "visibility_counts": {"PUBLIC": 1, "INTERNAL": 0, "REVIEW_REQUIRED": 0}})
    write_manifest(directory, manifest)
    p = _patched_store(_valid_metadata())
    try:
        result = validate_index_directory(directory)
    finally:
        p.stop()
    assert result["ok"] and result["public_only"] and result["visibility_counts"]["INTERNAL"] == 0


def test_validate_rejects_manifest_visibility_counts_that_do_not_match(tmp_path):
    directory = _index_dir(tmp_path / "candidate")
    manifest = read_manifest(directory)
    manifest["visibility_counts"] = {"PUBLIC": 5, "INTERNAL": 0, "REVIEW_REQUIRED": 0}
    write_manifest(directory, manifest)
    p = _patched_store(_valid_metadata())
    try:
        result = validate_index_directory(directory)
    finally:
        p.stop()
    assert not result["ok"] and "visibility_counts" in result["reason"]


@pytest.mark.parametrize("field", ["content_state", "verification_state", "citation", "bundle"])
def test_validate_requires_the_retrieval_contract_fields(tmp_path, field):
    directory = _index_dir(tmp_path / "candidate")
    meta = _valid_metadata()
    del meta[0][field]
    p = _patched_store(meta)
    try:
        result = validate_index_directory(directory)
    finally:
        p.stop()
    assert not result["ok"] and field in result["reason"]


def _legacy_dir(path):
    path.mkdir(parents=True)
    (path / "index.faiss").write_bytes(b"legacy-index")
    (path / "metadata.pkl").write_bytes(b"legacy-meta")
    return path


def test_rollback_to_a_legacy_manifestless_index_works_and_preserves_its_files(tmp_path):
    legacy = _legacy_dir(tmp_path / "current")
    legacy_hashes = artifact_hashes(legacy)
    candidate = _index_dir(tmp_path / "candidate", build_id="new")
    p = _patched_store(_valid_metadata())
    try:
        promoted = promote_candidate(candidate, legacy, tmp_path / "previous")
        manifest = read_manifest(legacy)
        assert manifest["previous_index_legacy"] is True and manifest["previous_index_artifact_hashes"] == legacy_hashes
        rolled = rollback(legacy, promoted["previous_index"], tmp_path / "rolled")
    finally:
        p.stop()
    assert rolled["rollback_build_id"] == "legacy-pre-phase18"
    assert artifact_hashes(legacy) == legacy_hashes  # legacy index files byte-identical
    assert read_manifest(legacy)["promotion_status"] == "ROLLED_BACK"


def test_rollback_refuses_a_retained_previous_index_that_changed_since_promotion(tmp_path):
    legacy = _legacy_dir(tmp_path / "current")
    candidate = _index_dir(tmp_path / "candidate", build_id="new")
    p = _patched_store(_valid_metadata())
    try:
        promoted = promote_candidate(candidate, legacy, tmp_path / "previous")
        (tmp_path / "previous" / os.path.basename(promoted["previous_index"]) / "index.faiss").write_bytes(b"corrupted")
        with pytest.raises(RuntimeError, match="no longer matches"):
            rollback(legacy, promoted["previous_index"], tmp_path / "rolled")
    finally:
        p.stop()
    assert read_manifest(legacy)["promotion_status"] == "PROMOTED"  # nothing was swapped


def test_failed_second_rename_puts_the_old_index_back(tmp_path):
    current = _index_dir(tmp_path / "current", build_id="old")
    candidate = _index_dir(tmp_path / "candidate", build_id="new")
    real_replace = os.replace

    p = _patched_store(_valid_metadata())
    try:
        boom = lambda s, d: (_ for _ in ()).throw(PermissionError("boom")) if str(s) == str(candidate) else real_replace(s, d)
        with patch("index.lifecycle.os.replace", side_effect=boom), pytest.raises(PermissionError):
            promote_candidate(candidate, current, tmp_path / "previous")
    finally:
        p.stop()
    assert read_manifest(current)["build_id"] == "old"  # old index restored to current
    assert candidate.exists()  # candidate untouched


def test_failed_second_rename_and_failed_restore_reports_exact_recovery_state(tmp_path):
    current = _index_dir(tmp_path / "current", build_id="old")
    candidate = _index_dir(tmp_path / "candidate", build_id="new")
    real_replace = os.replace

    def fail_all_after_first(s, d):
        if str(d) == str(current):
            raise PermissionError("cannot write current")
        return real_replace(s, d)

    p = _patched_store(_valid_metadata())
    try:
        with (
            patch("index.lifecycle.os.replace", side_effect=fail_all_after_first),
            pytest.raises(RuntimeError, match="NO current index.*Recover by hand: mv"),
        ):
            promote_candidate(candidate, current, tmp_path / "previous")
    finally:
        p.stop()
    assert not current.exists()
    assert len(list((tmp_path / "previous").iterdir())) == 1 and candidate.exists()  # both indexes intact


def test_manifest_write_is_atomic_file_replace(tmp_path):
    write_manifest(tmp_path / "d", {"a": 1})
    assert not (tmp_path / "d" / "manifest.json.tmp").exists()
    assert json.loads((tmp_path / "d" / "manifest.json").read_text()) == {"a": 1}
