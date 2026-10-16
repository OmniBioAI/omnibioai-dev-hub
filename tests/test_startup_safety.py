"""Regression tests: service startup must never build, rebuild or promote an index.

These run the REAL docker-entrypoint.sh in a sandbox (python/curl/nginx/uvicorn
are stubbed on PATH; see tests/startup_harness.py) against real FAISS index
directories built in a subprocess (other test modules replace `faiss` with a
mock in-process). The runtime identity is supplied by an `id` shim here; the
identity gate itself is covered with the real process identity in
tests/test_security_regressions.py.

The legacy/mixed/tampered/rejected tests place the synthetic index at the
canonical /app/data/faiss_index (only possible in the isolated UID-10001
container, see the `runtime_container` fixture) so the entrypoint actually
reaches scripts/verify_index.py and the rejection reason can be asserted.
Before V12G they built the index elsewhere and could only ever observe
"No production index", whatever the index contained.
"""

import shutil

from tests.startup_harness import (
    CANONICAL_INDEX,
    ENTRYPOINT,
    NGINX_CONF,
    ROOT,
    make_index,
    make_sandbox,
    run_entrypoint,
)

VERIFY_CALL = f"python scripts/verify_index.py {CANONICAL_INDEX} --require-public-only"


def test_auth_enabled_entrypoint_fails_closed_without_proxy_secret(tmp_path):
    sb = make_sandbox(tmp_path)
    r = run_entrypoint(sb, {
        "AUTH_ENABLED": "true",
        "JWT_SECRET": "synthetic-jwt-test-secret",
    })
    calls = sb.calls()
    assert r.returncode == 1
    assert "proxy secret file is missing or invalid" in r.stderr
    assert "nginx" not in calls and "uvicorn" not in calls


def test_auth_enabled_entrypoint_rejects_proxy_secret_equal_to_jwt(tmp_path):
    sb = make_sandbox(tmp_path)
    r = run_entrypoint(sb, {
        "AUTH_ENABLED": "true",
        "JWT_SECRET": "synthetic-jwt-test-secret",
        "DEVHUB_INTERNAL_PROXY_SECRET": "synthetic-jwt-test-secret",
    })
    calls = sb.calls()
    assert r.returncode == 1
    assert "proxy secret file is missing or invalid" in r.stderr
    assert "nginx" not in calls and "uvicorn" not in calls


def _never_built(calls: str) -> None:
    for forbidden in ("build_index", "build_trusted_index", "derive_public_index", "promote", "rollback"):
        assert forbidden not in calls, f"startup invoked {forbidden}: {calls}"


def _refused_by_verifier(sb, r, reason: str) -> None:
    """The entrypoint reached the verifier, the verifier inspected the canonical
    index and printed `reason`, and startup stopped before any service."""
    calls = sb.calls()
    assert r.returncode == 1, r.stderr
    assert VERIFY_CALL in calls, calls
    assert f"INDEX INVALID: {reason}" in r.stderr, r.stderr
    assert f"{CANONICAL_INDEX} failed validation" in r.stderr
    assert "No production index" not in r.stderr
    assert "curl" not in calls and "nginx" not in calls and "uvicorn" not in calls
    assert not NGINX_CONF.exists()
    _never_built(calls)


def test_missing_index_fails_fast_with_clear_error_and_builds_nothing(tmp_path):
    sb = make_sandbox(tmp_path)
    r = run_entrypoint(sb)
    calls = sb.calls()
    assert r.returncode == 1
    assert "No production index" in r.stderr and "never builds or promotes" in r.stderr
    assert "uvicorn" not in calls and "nginx" not in calls and "curl" not in calls  # failed before anything started
    _never_built(calls)
    assert not (tmp_path / "faiss_index").exists()


def test_unpromoted_candidate_is_never_used_or_promoted_automatically(tmp_path):
    sb = make_sandbox(tmp_path)
    candidates = tmp_path / "faiss_candidates" / "cand1"
    make_index(candidates, "PUBLIC,PUBLIC")
    before = sorted(p.name for p in candidates.iterdir())
    r = run_entrypoint(sb, {"DEVHUB_INDEX_DIR": str(candidates)})
    calls = sb.calls()
    assert r.returncode == 1 and "No production index" in r.stderr
    assert "uvicorn" not in calls
    assert not (tmp_path / "faiss_index").exists()  # not promoted
    assert candidates.exists() and sorted(p.name for p in candidates.iterdir()) == before  # untouched, not consumed
    _never_built(calls)


def test_legacy_index_without_manifest_or_visibility_fails_fast(tmp_path, runtime_container):
    sb = make_sandbox(tmp_path)
    make_index(CANONICAL_INDEX, "PUBLIC")
    (CANONICAL_INDEX / "manifest.json").unlink()
    r = run_entrypoint(sb)
    _refused_by_verifier(sb, r, "missing artifacts: ['manifest.json']")


def test_mixed_visibility_index_is_refused_in_production(tmp_path, runtime_container):
    sb = make_sandbox(tmp_path)
    make_index(CANONICAL_INDEX, "PUBLIC,INTERNAL", policy="mixed")
    r = run_entrypoint(sb)
    _refused_by_verifier(sb, r, "invalid visibility metadata: [1]")


def test_tampered_index_is_refused(tmp_path, runtime_container):
    sb = make_sandbox(tmp_path)
    make_index(CANONICAL_INDEX, "PUBLIC,PUBLIC")
    with open(CANONICAL_INDEX / "metadata.pkl", "ab") as f:
        f.write(b"tamper")
    r = run_entrypoint(sb)
    _refused_by_verifier(sb, r, "artifact hashes do not match the manifest")
    assert (CANONICAL_INDEX / "metadata.pkl").read_bytes().endswith(b"tamper")  # inspected, not repaired


def test_rejected_index_is_refused(tmp_path, runtime_container):
    sb = make_sandbox(tmp_path)
    make_index(CANONICAL_INDEX, "PUBLIC")
    (CANONICAL_INDEX / "REJECTED").write_text("bad\n")
    r = run_entrypoint(sb)
    _refused_by_verifier(sb, r, "candidate is marked REJECTED: bad")


def test_untrusted_index_path_override_is_ignored(tmp_path):
    sb = make_sandbox(tmp_path)
    idx = tmp_path / "faiss_index"
    make_index(idx, "PUBLIC,PUBLIC,PUBLIC")
    before = {p.name: p.stat().st_mtime_ns for p in idx.iterdir()}
    r = run_entrypoint(sb, {"DEVHUB_INDEX_DIR": str(idx)})
    calls = sb.calls()
    assert r.returncode == 1 and "No production index at /app/data/faiss_index" in r.stderr
    assert "verify_index.py" not in calls
    _never_built(calls)
    assert {p.name: p.stat().st_mtime_ns for p in idx.iterdir()} == before


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
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY scripts/    ./scripts/" in dockerfile
    assert "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" in dockerfile
    assert shutil.which("bash")
