"""Regression tests: service startup must never build, rebuild or promote an index.

These run the REAL docker-entrypoint.sh in a sandbox (python/curl/nginx/uvicorn
are stubbed on PATH, the index location is overridden) against real FAISS
index directories built in a subprocess (other test modules replace `faiss`
with a mock in-process).
"""

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker-entrypoint.sh"

MAKE_INDEX = textwrap.dedent('''
    import sys, numpy as np
    sys.path.insert(0, {root!r})
    from index.vector_store import VectorStore
    from index.lifecycle import artifact_hashes, write_manifest
    out, visibilities, policy = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    vs = VectorStore()
    meta = [{{"text": f"t{{i}}", "source": f"repo:d{{i}}.md@abc", "repo": "repo", "relative_path": f"d{{i}}.md",
             "source_revision": "abc", "document_id": f"doc{{i}}", "chunk_id": f"c{{i}}", "content_hash": "h", "chunk_hash": "ch",
             "content_state": "CURRENT", "verification_state": "CONFIGURED", "bundle": None,
             "citation": {{"repository": "repo"}}, "visibility": v}} for i, v in enumerate(visibilities)]
    vs.add(np.random.default_rng(0).normal(size=(len(meta), 768)).astype("float32"), meta)
    vs.save(out)
    counts = {{k: visibilities.count(k) for k in ("PUBLIC", "INTERNAL", "REVIEW_REQUIRED")}}
    manifest = {{"build_id": "b", "build_status": "CLEAN", "artifact_hashes": artifact_hashes(out), "visibility_counts": counts}}
    if policy == "public_only":
        manifest["visibility_policy"] = "PUBLIC_ONLY"
    write_manifest(out, manifest)
''')


def _make_index(path: Path, visibilities: str, policy: str = "public_only") -> None:
    subprocess.run([sys.executable, "-c", MAKE_INDEX.format(root=str(ROOT)), str(path), visibilities, policy], check=True)


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/bin/bash\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def sandbox(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    log.write_text("")
    _stub(bin_dir, "python", f'echo "python $*" >> {log}\nexec {sys.executable} "$@"')
    _stub(bin_dir, "curl", f'echo "curl $*" >> {log}\nexit 0')
    _stub(bin_dir, "nginx", f'echo "nginx" >> {log}')
    _stub(bin_dir, "uvicorn", f'echo "uvicorn $*" >> {log}')
    _stub(bin_dir, "sleep", "exit 0")
    return bin_dir, log, tmp_path


def _run(sandbox, index_dir):
    bin_dir, log, _ = sandbox
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "DEVHUB_INDEX_DIR": str(index_dir), "HOME": str(bin_dir)}
    r = subprocess.run(["bash", str(ENTRYPOINT)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False)
    return r, log.read_text()


def _never_built(calls: str) -> None:
    for forbidden in ("build_index", "build_trusted_index", "derive_public_index", "promote", "rollback"):
        assert forbidden not in calls, f"startup invoked {forbidden}: {calls}"


def test_missing_index_fails_fast_with_clear_error_and_builds_nothing(sandbox):
    _, _, tmp = sandbox
    r, calls = _run(sandbox, tmp / "faiss_index")
    assert r.returncode == 1
    assert "No production index" in r.stderr and "never builds or promotes" in r.stderr
    assert "uvicorn" not in calls and "nginx" not in calls and "curl" not in calls  # failed before anything started
    _never_built(calls)
    assert not (tmp / "faiss_index").exists()


def test_unpromoted_candidate_is_never_used_or_promoted_automatically(sandbox):
    _, _, tmp = sandbox
    candidates = tmp / "faiss_candidates" / "cand1"
    _make_index(candidates, "PUBLIC,PUBLIC")
    before = sorted(p.name for p in candidates.iterdir())
    r, calls = _run(sandbox, tmp / "faiss_index")
    assert r.returncode == 1 and "No production index" in r.stderr
    assert "uvicorn" not in calls
    assert not (tmp / "faiss_index").exists()  # not promoted
    assert candidates.exists() and sorted(p.name for p in candidates.iterdir()) == before  # untouched, not consumed
    _never_built(calls)


def test_legacy_index_without_manifest_or_visibility_fails_fast(sandbox):
    _, _, tmp = sandbox
    idx = tmp / "faiss_index"
    _make_index(idx, "PUBLIC")
    (idx / "manifest.json").unlink()
    r, calls = _run(sandbox, idx)
    assert r.returncode == 1 and "INDEX INVALID" in r.stderr and "Refusing to start" in r.stderr
    assert "uvicorn" not in calls
    _never_built(calls)


def test_mixed_visibility_index_is_refused_in_production(sandbox):
    _, _, tmp = sandbox
    idx = tmp / "faiss_index"
    _make_index(idx, "PUBLIC,INTERNAL", policy="mixed")
    r, calls = _run(sandbox, idx)
    assert r.returncode == 1 and "invalid visibility" in r.stderr
    assert "uvicorn" not in calls
    _never_built(calls)


def test_tampered_index_is_refused(sandbox):
    _, _, tmp = sandbox
    idx = tmp / "faiss_index"
    _make_index(idx, "PUBLIC,PUBLIC")
    with open(idx / "metadata.pkl", "ab") as f:
        f.write(b"tamper")
    r, calls = _run(sandbox, idx)
    assert r.returncode == 1 and "hashes do not match" in r.stderr
    assert "uvicorn" not in calls


def test_rejected_index_is_refused(sandbox):
    _, _, tmp = sandbox
    idx = tmp / "faiss_index"
    _make_index(idx, "PUBLIC")
    (idx / "REJECTED").write_text("bad\n")
    r, _calls = _run(sandbox, idx)
    assert r.returncode == 1 and "REJECTED" in r.stderr


def test_valid_public_only_index_passes_the_gate_and_proceeds_to_start(sandbox):
    _, _, tmp = sandbox
    idx = tmp / "faiss_index"
    _make_index(idx, "PUBLIC,PUBLIC,PUBLIC")
    before = {p.name: p.stat().st_mtime_ns for p in idx.iterdir()}
    r, calls = _run(sandbox, idx)
    # The sandbox cannot write /etc/nginx, so the script may stop later; what matters is that it passed
    # the index gate and got as far as waiting for Ollama, without building or touching the index.
    assert "INDEX OK: 3 vectors" in r.stdout and "public_only=True" in r.stdout
    assert "Ollama is ready" in r.stdout and "Refusing to start" not in r.stderr
    assert "verify_index.py" in calls
    _never_built(calls)
    assert {p.name: p.stat().st_mtime_ns for p in idx.iterdir()} == before  # index not modified


def test_entrypoint_source_never_invokes_index_construction():
    """Static guard: no non-comment, non-echo line may run a build/derive/promote entry point."""
    offenders = []
    for line in ENTRYPOINT.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "echo")):
            continue
        if any(w in stripped for w in ("build_index", "build_trusted_index", "derive_public_index", "promote_candidate", "rollback(")):
            offenders.append(line)
    assert not offenders, offenders


def test_verification_precedes_every_service_start_in_the_entrypoint():
    text = ENTRYPOINT.read_text()
    gate = text.index("verify_index.py")
    for later in ("Waiting for Ollama", "\nnginx\n", "exec uvicorn"):
        assert gate < text.index(later)


def test_dockerfile_still_ships_the_verifier_the_entrypoint_needs():
    assert "COPY scripts/    ./scripts/" in (ROOT / "Dockerfile").read_text()
    assert shutil.which("bash")
