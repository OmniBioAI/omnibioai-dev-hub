"""F-S4-1 regression tests: an index is only promotable or servable with a
complete, valid integrity receipt (manifest["artifact_hashes"] covering exactly
index.faiss and metadata.pkl with matching lowercase sha256 digests).

Before V12I, validate_index_directory() skipped the hash comparison whenever
the manifest carried no hashes, so a manifest of `{}` was accepted and the
service started on it. These tests drive the REAL verifier CLI
(scripts/verify_index.py), the REAL promotion CLI, and -- inside the UID-10001
qualification container -- the REAL docker-entrypoint.sh, with synthetic
indexes only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from index.lifecycle import REQUIRED_ARTIFACTS, receipt_problem
from tests.startup_harness import (
    CANONICAL_INDEX,
    NGINX_CONF,
    ROOT,
    make_index,
    make_sandbox,
    run_entrypoint,
)

VERIFY = str(ROOT / "scripts" / "verify_index.py")
PROMOTE = str(ROOT / "scripts" / "promote_index.py")
GOOD = "a" * 64


def _verify(path: Path):
    return subprocess.run([sys.executable, VERIFY, str(path), "--require-public-only"],
                          capture_output=True, text=True, timeout=120, check=False)


def _edit_manifest(index: Path, edit) -> None:
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    edit(manifest)
    manifest_path.write_text(json.dumps(manifest))


def _set_hashes(value):
    return lambda m: m.__setitem__("artifact_hashes", value)


def _valid_hashes(index: Path):
    return json.loads((index / "manifest.json").read_text())["artifact_hashes"]


# --- pure contract -----------------------------------------------------------

@pytest.mark.parametrize("manifest, reason", [
    ([], "not a JSON object"),
    ("receipt", "not a JSON object"),
    ({}, "no artifact hashes"),
    ({"artifact_hashes": None}, "no artifact hashes"),
    ({"artifact_hashes": {}}, "no artifact hashes"),
    ({"artifact_hashes": []}, "no artifact hashes"),
    ({"artifact_hashes": "deadbeef"}, "no artifact hashes"),
    ({"artifact_hashes": {"index.faiss": GOOD}}, "missing required artifacts: ['metadata.pkl']"),
    ({"artifact_hashes": {"metadata.pkl": GOOD}}, "missing required artifacts: ['index.faiss']"),
    ({"artifact_hashes": {"index.faiss": GOOD, "metadata.pkl": GOOD, "../etc/passwd": GOOD}}, "unexpected artifacts"),
    ({"artifact_hashes": {"index.faiss": GOOD, "metadata.pkl": GOOD, "/app/data/x": GOOD}}, "unexpected artifacts"),
    ({"artifact_hashes": {"index.faiss": GOOD, "metadata.pkl": GOOD, "manifest.json": GOOD}}, "unexpected artifacts"),
    ({"artifact_hashes": {"./index.faiss": GOOD, "metadata.pkl": GOOD}}, "unexpected artifacts"),
    ({"artifact_hashes": {"index.faiss": "a" * 63, "metadata.pkl": GOOD}}, "malformed sha256"),
    ({"artifact_hashes": {"index.faiss": "A" * 64, "metadata.pkl": GOOD}}, "malformed sha256"),
    ({"artifact_hashes": {"index.faiss": "g" * 64, "metadata.pkl": GOOD}}, "malformed sha256"),
    ({"artifact_hashes": {"index.faiss": GOOD + "\n", "metadata.pkl": GOOD}}, "malformed sha256"),
    ({"artifact_hashes": {"index.faiss": 12345, "metadata.pkl": GOOD}}, "malformed sha256"),
    ({"artifact_hashes": {"index.faiss": GOOD, "metadata.pkl": "b" * 64}}, "do not match"),
])
def test_receipt_contract_fails_closed(manifest, reason):
    actual = {"index.faiss": GOOD, "metadata.pkl": GOOD}
    assert reason in receipt_problem(manifest, actual)


def test_receipt_contract_accepts_exactly_the_complete_matching_receipt():
    actual = {"index.faiss": GOOD, "metadata.pkl": "b" * 64}
    assert receipt_problem({"artifact_hashes": dict(actual)}, actual) is None
    assert REQUIRED_ARTIFACTS == ("index.faiss", "metadata.pkl")


# --- real verifier CLI on real synthetic indexes ----------------------------

@pytest.mark.parametrize("case, mutate, reason", [
    ("hashless_empty_manifest", lambda d: (d / "manifest.json").write_text("{}"), "manifest carries no artifact hashes"),
    ("hashes_key_removed", lambda d: _edit_manifest(d, lambda m: m.pop("artifact_hashes")), "manifest carries no artifact hashes"),
    ("hashes_empty", lambda d: _edit_manifest(d, _set_hashes({})), "manifest carries no artifact hashes"),
    ("hashes_not_a_mapping", lambda d: _edit_manifest(d, _set_hashes(["index.faiss"])), "manifest carries no artifact hashes"),
    ("entry_missing", lambda d: _edit_manifest(d, lambda m: m["artifact_hashes"].pop("metadata.pkl")), "missing required artifacts: ['metadata.pkl']"),
    ("unsafe_path_entry", lambda d: _edit_manifest(d, lambda m: m["artifact_hashes"].__setitem__("../../etc/passwd", GOOD)), "unexpected artifacts"),
    ("digest_bad_format", lambda d: _edit_manifest(d, lambda m: m["artifact_hashes"].__setitem__("index.faiss", "xyz")), "malformed sha256"),
    ("digest_uppercase", lambda d: _edit_manifest(d, lambda m: m["artifact_hashes"].__setitem__("index.faiss", m["artifact_hashes"]["index.faiss"].upper())), "malformed sha256"),
    ("digest_mismatch", lambda d: _edit_manifest(d, lambda m: m["artifact_hashes"].__setitem__("metadata.pkl", GOOD)), "do not match the manifest"),
    ("artifact_missing", lambda d: (d / "metadata.pkl").unlink(), "missing artifacts: ['metadata.pkl']"),
    ("receipt_malformed_json", lambda d: (d / "manifest.json").write_text("{not json"), "could not inspect"),
    ("receipt_not_an_object", lambda d: (d / "manifest.json").write_text("[]"), "manifest is not a JSON object"),
    ("receipt_inconsistent_counts", lambda d: _edit_manifest(d, lambda m: m.__setitem__("visibility_counts", {"PUBLIC": 9})), "do not match the index"),
])
def test_verifier_cli_refuses_incomplete_or_invalid_receipts(tmp_path, case, mutate, reason):
    index = tmp_path / "idx"
    make_index(index, "PUBLIC,PUBLIC")
    assert _verify(index).returncode == 0  # the unmodified synthetic index is valid
    mutate(index)
    r = _verify(index)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "INDEX INVALID: " in r.stderr and reason in r.stderr, r.stderr


def test_verifier_cli_accepts_a_complete_receipt(tmp_path):
    index = tmp_path / "idx"
    make_index(index, "PUBLIC,PUBLIC,PUBLIC")
    r = _verify(index)
    assert r.returncode == 0 and "INDEX OK: 3 vectors" in r.stdout
    assert set(_valid_hashes(index)) == set(REQUIRED_ARTIFACTS)


def test_public_only_enforcement_is_unchanged_with_a_valid_receipt(tmp_path):
    index = tmp_path / "idx"
    make_index(index, "PUBLIC,INTERNAL", policy="mixed")  # hashes valid; content not PUBLIC-only
    r = _verify(index)
    assert r.returncode == 1 and "invalid visibility metadata: [1]" in r.stderr


# --- real promotion CLI -------------------------------------------------------

def test_promotion_refuses_a_hashless_candidate_and_leaves_the_live_index_untouched(tmp_path):
    data, live, src = tmp_path / "data", tmp_path / "data" / "faiss_index", tmp_path / "src"
    data.mkdir()
    make_index(live, "PUBLIC")
    before = {p.name: p.read_bytes() for p in live.iterdir()}
    make_index(src, "PUBLIC,PUBLIC")
    _edit_manifest(src, lambda m: m.pop("artifact_hashes"))
    r = subprocess.run([sys.executable, PROMOTE, "promote", "--source", str(src), "--data-dir", str(data)],
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode != 0 and "REFUSED" in r.stderr, r.stdout + r.stderr
    assert {p.name: p.read_bytes() for p in live.iterdir()} == before
    assert not (data / "faiss_candidates").exists() or not any((data / "faiss_candidates").iterdir())


# --- real entrypoint at the canonical path (UID-10001 container) ------------

@pytest.mark.parametrize("case, mutate, reason", [
    ("empty_manifest", lambda d: (d / "manifest.json").write_text("{}"), "manifest carries no artifact hashes"),
    ("hashes_removed", lambda d: _edit_manifest(d, lambda m: m.pop("artifact_hashes")), "manifest carries no artifact hashes"),
    ("hashes_incomplete", lambda d: _edit_manifest(d, lambda m: m["artifact_hashes"].pop("index.faiss")), "integrity receipt is missing required artifacts: ['index.faiss']"),
])
def test_entrypoint_refuses_to_serve_an_index_without_a_complete_receipt(tmp_path, runtime_container, case, mutate, reason):
    make_index(CANONICAL_INDEX, "PUBLIC,PUBLIC")
    mutate(CANONICAL_INDEX)
    sb = make_sandbox(tmp_path)
    r = run_entrypoint(sb)
    calls = sb.calls()
    assert r.returncode == 1, r.stderr
    assert f"python scripts/verify_index.py {CANONICAL_INDEX} --require-public-only" in calls
    assert f"INDEX INVALID: {reason}" in r.stderr
    assert "nginx" not in calls and "uvicorn" not in calls and not NGINX_CONF.exists()
