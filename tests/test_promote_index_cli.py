"""Tests for the operator-facing promote/rollback command (real FAISS, subprocess)."""

import json
import subprocess
import sys
from pathlib import Path

from tests.test_startup_safety import _make_index

ROOT = Path(__file__).resolve().parents[1]
CLI = str(ROOT / "scripts" / "promote_index.py")


def _cli(*args):
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True, timeout=120, check=False)


def _hashes(d: Path):
    import hashlib
    return {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("index.faiss", "metadata.pkl")}


def _legacy(data: Path):
    live = data / "faiss_index"
    _make_index(live, "PUBLIC,PUBLIC")
    (live / "manifest.json").unlink()  # pre-Phase-18 index: no manifest
    return live


def test_promote_then_rollback_restores_the_legacy_index_byte_for_byte(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    legacy_hashes = _hashes(live)
    _make_index(src, "PUBLIC,PUBLIC,PUBLIC")

    r = _cli("promote", "--source", str(src), "--data-dir", str(data))
    assert r.returncode == 0, r.stderr
    assert "PROMOTED" in r.stdout and "hashes verified" in r.stdout
    assert _hashes(live) == _hashes(src)
    assert json.loads((live / "manifest.json").read_text())["promotion_status"] == "PROMOTED"
    prev = list((data / "faiss_previous").iterdir())
    assert len(prev) == 1 and _hashes(prev[0]) == legacy_hashes  # previous retained

    r = _cli("rollback", "--data-dir", str(data))
    assert r.returncode == 0, r.stderr
    assert _hashes(live) == legacy_hashes
    assert list((data / "faiss_rolled").iterdir())  # the promoted index is retained too


def test_promote_refuses_a_mixed_visibility_candidate_and_changes_nothing(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    before = _hashes(live)
    _make_index(src, "PUBLIC,INTERNAL", policy="mixed")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data))
    assert r.returncode == 1 and "REFUSED" in r.stderr
    assert _hashes(live) == before and not (data / "faiss_candidates").exists()


def test_promote_refuses_a_rejected_candidate(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    before = _hashes(live)
    _make_index(src, "PUBLIC")
    (src / "REJECTED").write_text("never\n")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data))
    assert r.returncode == 1 and "REJECTED" in r.stderr and _hashes(live) == before


def test_promote_refuses_when_the_candidate_was_already_staged(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    before = _hashes(live)
    _make_index(src, "PUBLIC")
    (data / "faiss_candidates" / "b").mkdir(parents=True)  # manifest build_id from the helper is "b"
    r = _cli("promote", "--source", str(src), "--data-dir", str(data))
    assert r.returncode == 1 and "already exists" in r.stderr and _hashes(live) == before


def test_rollback_without_a_recorded_previous_index_is_refused(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_index(data / "faiss_index", "PUBLIC")
    r = _cli("rollback", "--data-dir", str(data))
    assert r.returncode == 1 and "no previous index" in r.stderr
