"""Tests for Phase 18 staging, integrity, promotion and rollback helpers."""

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
