"""V12A security-control regression tests (S1 non-root, S2 no startup chown,
S4 verifier enforcement, S5 proxy-secret handling).

Identity-dependent tests use the REAL process identity: on a developer host
(any UID other than 10001) they assert refusal; in the isolated UID-10001
qualification container they assert acceptance. Shim-based tests cover
identity values that cannot be produced unprivileged (0, malformed output).
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from api.runtime_identity import RUNTIME_UID, require_runtime_uid
from tests.startup_harness import (
    CANONICAL_DATA,
    CANONICAL_INDEX,
    ENTRYPOINT,
    NGINX_CONF,
    PROXY_SECRET_PATH,
    ROOT,
    SYNTHETIC_JWT_SECRET,
    SYNTHETIC_PROXY_SECRET,
    UID_REFUSAL,
    make_index,
    make_sandbox,
    rehash_manifest,
    run_entrypoint,
    runtime_container_available,
    tree_snapshot,
)

VERIFY_CALL = f"python scripts/verify_index.py {CANONICAL_INDEX} --require-public-only"
OWNERSHIP_CALL = re.compile(r"^(chown|chgrp|chmod|setfacl|find|install) ", re.MULTILINE)


# ---------------------------------------------------------------------------
# S1 -- non-root runtime identity
# ---------------------------------------------------------------------------

def test_s1_entrypoint_enforces_the_real_process_identity(tmp_path):
    """No shim: the entrypoint sees this process's real UID."""
    sb = make_sandbox(tmp_path, uid=None)
    r = run_entrypoint(sb)
    calls = sb.calls()
    if os.geteuid() == RUNTIME_UID:
        assert UID_REFUSAL not in r.stderr
        assert "python -c from index.runtime_path import validate_runtime_index_root" in calls  # passed the gate
    else:
        assert r.returncode == 1
        assert r.stderr.strip() == UID_REFUSAL
        assert calls == ""  # nothing -- not even python -- ran under the wrong identity


@pytest.mark.parametrize("reported_uid", ["0", "1000", "65534", "", "abc", "010001", "10001 ", " 10001", "-10001", "100010"])
def test_s1_entrypoint_refuses_every_identity_other_than_10001(tmp_path, reported_uid):
    sb = make_sandbox(tmp_path, uid=reported_uid)
    r = run_entrypoint(sb, {"AUTH_ENABLED": "true", "JWT_SECRET": SYNTHETIC_JWT_SECRET})
    assert r.returncode == 1
    assert r.stderr.strip() == UID_REFUSAL
    assert sb.calls() == ""


def test_s1_entrypoint_accepts_exactly_10001(tmp_path):
    sb = make_sandbox(tmp_path, uid="10001")
    r = run_entrypoint(sb)
    assert UID_REFUSAL not in r.stderr
    assert "validate_runtime_index_root" in sb.calls()


def test_s1_entrypoint_fails_closed_when_identity_lookup_fails(tmp_path):
    sb = make_sandbox(tmp_path, uid=None)
    (sb.bin_dir / "id").write_text("#!/bin/bash\nexit 1\n")
    (sb.bin_dir / "id").chmod(0o755)
    r = run_entrypoint(sb)
    assert r.returncode == 1 and r.stderr.strip() == UID_REFUSAL
    assert sb.calls() == ""


def test_s1_entrypoint_fails_closed_when_id_is_unavailable(tmp_path):
    sb = make_sandbox(tmp_path, uid=None)
    r = run_entrypoint(sb, path=str(sb.bin_dir))  # stubs only: no `id` anywhere on PATH
    assert r.returncode == 1 and UID_REFUSAL in r.stderr
    assert sb.calls() == ""


@pytest.mark.parametrize("euid", [0, 1000, 65534, 10000, 10002, "10001", 10001.0, None])
def test_s1_require_runtime_uid_rejects_everything_but_int_10001(monkeypatch, euid):
    if euid is None:
        monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(RuntimeError, match="must run as UID 10001"):
        require_runtime_uid(euid)


def test_s1_require_runtime_uid_accepts_10001(monkeypatch):
    assert require_runtime_uid(10001) == 10001
    monkeypatch.setattr(os, "geteuid", lambda: 10001)
    assert require_runtime_uid() == 10001


@pytest.mark.parametrize("euid", [0, 1000])
def test_s1_app_startup_refuses_before_any_other_startup_work(monkeypatch, euid):
    from api import main
    monkeypatch.setattr(os, "geteuid", lambda: euid)
    with patch.object(main, "validate_auth_config") as auth_check, \
         patch.object(main, "init_control_plane", new_callable=AsyncMock) as init, \
         pytest.raises(RuntimeError, match="must run as UID 10001"):
        asyncio.run(main.startup_event())
    auth_check.assert_not_called()
    init.assert_not_called()


def _launch_app_with_uvicorn():
    """Start the real application the way an alternate entrypoint would
    (`uvicorn api.main:app`), loopback only, ephemeral port. Returns (proc, text)
    once startup completed or the process ended."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1",
        "AUTH_ENABLED": "true", "JWT_SECRET": SYNTHETIC_JWT_SECRET, "JWT_AUDIENCE": "devhub", "JWT_ISSUER": "v12g",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "0", "--lifespan", "on"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    output = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        output.append(line)
        if "Application startup complete" in line:
            break
    return proc, "".join(output)


def _stop(proc) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)


def test_s1_alternate_entrypoint_uvicorn_enforces_the_real_process_identity(runtime_container_or_host):
    """`docker run <image> uvicorn api.main:app` bypasses docker-entrypoint.sh
    (the image uses CMD); the application lifespan must still enforce the UID."""
    proc, text = _launch_app_with_uvicorn()
    try:
        if os.geteuid() == RUNTIME_UID:
            assert "Application startup complete" in text, text
            assert "must run as UID" not in text
        else:
            proc.wait(timeout=30)
            text += proc.stdout.read()
            assert proc.returncode != 0
            assert "must run as UID 10001" in text, text
            assert "Application startup complete" not in text
            assert "Uvicorn running" not in text
    finally:
        _stop(proc)


@pytest.fixture
def runtime_container_or_host(request):
    """In the qualification container the alternate-entrypoint acceptance path
    needs the synthetic proxy secret; on a host the identity check fires first."""
    ok, _ = runtime_container_available()
    if ok:
        request.getfixturevalue("runtime_container")
    yield


def test_s1_dockerfile_runs_the_gated_entrypoint_as_10001():
    lines = [ln.strip() for ln in (ROOT / "Dockerfile").read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    users = [ln for ln in lines if ln.startswith("USER ")]
    assert users and users[-1] == "USER 10001:10001"
    assert not any(ln.startswith("ENTRYPOINT") for ln in lines)
    cmds = [ln for ln in lines if ln.startswith("CMD ")]
    assert cmds == ['CMD ["/app/docker-entrypoint.sh"]']
    assert lines.index(users[-1]) < lines.index(cmds[0])
    entry = ENTRYPOINT.read_text()
    assert entry.index('if [ "$(id -u)" != "10001" ]') < min(entry.index("python"), entry.index("nginx"))
    assert not re.search(r"^\s*(exec\s+)?(su|sudo|gosu|su-exec|setpriv|runuser)\b", entry, re.MULTILINE)


# ---------------------------------------------------------------------------
# S2 -- startup never changes ownership/permissions of mounted data
# ---------------------------------------------------------------------------

_SHELL_OWNERSHIP = [
    re.compile(r"\b(chown|chgrp|lchown|setfacl|chattr)\b"),
    re.compile(r"\bchmod\b.*(\s-[a-zA-Z]*R|--recursive)"),
    re.compile(r"\bfind\b.*-exec(dir)?\s+\S*(chown|chgrp|chmod)"),
    re.compile(r"\binstall\b.*\s-(o|g)\s"),
]
_PY_OWNERSHIP_NAMES = {"chown", "lchown", "fchown", "chmod", "lchmod", "fchmod"}


def _python_ownership_calls(source: str) -> list[str]:
    """AST scan: catches os.chown, aliased modules (import os as x; x.chown),
    `from os import chown as c` and getattr(os, "chown") -- not just one spelling."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and any(a.name in _PY_OWNERSHIP_NAMES for a in node.names):
            found.append(f"line {node.lineno}: from {node.module} import {[a.name for a in node.names]}")
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name in _PY_OWNERSHIP_NAMES:
                found.append(f"line {node.lineno}: {name}(...)")
            elif name == "getattr" and any(isinstance(a, ast.Constant) and a.value in _PY_OWNERSHIP_NAMES for a in node.args):
                found.append(f"line {node.lineno}: getattr(..., {node.args[1].value!r})")
    return found
_STARTUP_PY_DIRS = ("api", "index", "rag", "integrations", "processing", "ingestion", "retrieval", "embeddings")


def _code_lines(text: str):
    for n, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield n, stripped


def test_s2_entrypoint_contains_no_ownership_or_recursive_permission_change():
    offenders = [(n, ln) for n, ln in _code_lines(ENTRYPOINT.read_text())
                 if any(p.search(ln) for p in _SHELL_OWNERSHIP)]
    assert not offenders, offenders


def test_s2_application_startup_code_never_changes_ownership():
    offenders = []
    for d in _STARTUP_PY_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            offenders += [f"{path.relative_to(ROOT)} {hit}" for hit in _python_ownership_calls(path.read_text())]
    assert not offenders, offenders


def test_s2_structural_guard_detects_known_recursive_ownership_forms():
    """The guard itself must recognise the forms it exists to stop."""
    for bad in ("chown -R appuser:appuser /app/data", "/usr/bin/chown -R 10001 /app/data",
                "find /app/data -exec chown 10001 {} +", "chmod -R 0777 /app/data", "setfacl -R -m u:1:rwx /app/data"):
        assert any(p.search(bad) for p in _SHELL_OWNERSHIP), bad
    for bad_py in ("import os\nos.chown(p, 1, 1)", "import os as _o\n_o.chown(p, 1, 1)",
                   "from os import chown as c\nc(p, 1, 1)", "import shutil\nshutil.chown(p, 'u')",
                   "import os\ngetattr(os, 'chown')(p, 1, 1)", "import os\nos.chmod(p, 0o777)"):
        assert _python_ownership_calls(bad_py), bad_py


@pytest.mark.parametrize("scenario", ["missing_index", "auth_without_secret", "identity_refused"])
def test_s2_startup_paths_never_invoke_ownership_tools(tmp_path, scenario):
    sb = make_sandbox(tmp_path, uid="1000" if scenario == "identity_refused" else "10001")
    env = {"AUTH_ENABLED": "true", "JWT_SECRET": SYNTHETIC_JWT_SECRET} if scenario == "auth_without_secret" else {}
    r = run_entrypoint(sb, env)
    assert r.returncode == 1
    assert not OWNERSHIP_CALL.search(sb.calls()), sb.calls()


def test_s2_full_startup_leaves_mounted_data_ownership_untouched(tmp_path, runtime_container):
    sb = make_sandbox(tmp_path)
    make_index(CANONICAL_INDEX, "PUBLIC,PUBLIC")
    (CANONICAL_DATA / "tenant_marker").write_text("synthetic\n")
    before = tree_snapshot(CANONICAL_DATA)
    time.sleep(0.01)  # any chown/chmod during startup would now produce a distinct ctime
    r = run_entrypoint(sb)
    assert r.returncode == 0, r.stderr
    assert "uvicorn api.main:app" in sb.calls()  # the full startup path really ran
    assert not OWNERSHIP_CALL.search(sb.calls()), sb.calls()
    assert tree_snapshot(CANONICAL_DATA) == before


def test_s2_application_startup_leaves_mounted_data_untouched(runtime_container):
    """Runs the REAL application lifespan (not a stub) over a synthetic mounted
    data tree and fingerprints ownership/mode/ctime before and after."""
    make_index(CANONICAL_INDEX, "PUBLIC,PUBLIC")
    (CANONICAL_DATA / "tenant").mkdir()
    (CANONICAL_DATA / "tenant" / "dataset.csv").write_text("synthetic\n")
    before = tree_snapshot(CANONICAL_DATA)
    time.sleep(0.01)
    proc, text = _launch_app_with_uvicorn()
    try:
        assert "Application startup complete" in text, text
        assert tree_snapshot(CANONICAL_DATA) == before
    finally:
        _stop(proc)


# ---------------------------------------------------------------------------
# S4 -- the entrypoint enforces the PUBLIC-only verifier result
# ---------------------------------------------------------------------------

def _refused(sb, r, reason_regex: str) -> None:
    calls = sb.calls()
    assert r.returncode == 1, r.stderr
    assert VERIFY_CALL in calls, calls
    assert re.search(reason_regex, r.stderr), r.stderr
    assert f"{CANONICAL_INDEX} failed validation" in r.stderr
    assert "Waiting for Ollama" not in r.stdout
    assert "curl" not in calls and "nginx" not in calls and "uvicorn" not in calls
    assert not NGINX_CONF.exists()


def test_s4_valid_public_only_index_is_accepted_and_services_start(tmp_path, runtime_container):
    sb = make_sandbox(tmp_path)
    make_index(CANONICAL_INDEX, "PUBLIC,PUBLIC,PUBLIC")
    r = run_entrypoint(sb)
    calls = sb.calls()
    assert r.returncode == 0, r.stderr
    assert "INDEX OK: 3 vectors, dim 768" in r.stdout and "public_only=True" in r.stdout
    assert calls.index(VERIFY_CALL) < calls.index("nginx") < calls.index("uvicorn api.main:app --host 0.0.0.0 --port 8082")


@pytest.mark.parametrize("case, build, reason", [
    ("legacy_missing_manifest", lambda: (make_index(CANONICAL_INDEX, "PUBLIC"), (CANONICAL_INDEX / "manifest.json").unlink()),
     r"INDEX INVALID: missing artifacts: \['manifest.json'\]"),
    ("mixed_public_internal", lambda: make_index(CANONICAL_INDEX, "PUBLIC,INTERNAL", policy="mixed"),
     r"INDEX INVALID: invalid visibility metadata: \[1\]"),
    ("review_required_chunk", lambda: make_index(CANONICAL_INDEX, "PUBLIC,REVIEW_REQUIRED", policy="mixed"),
     r"INDEX INVALID: invalid visibility metadata: \[1\]"),
    ("private_visibility_value", lambda: make_index(CANONICAL_INDEX, "PUBLIC,PRIVATE", policy="mixed"),
     r"INDEX INVALID: invalid visibility metadata: \[1\]"),
    ("tampered_index_bytes", lambda: (make_index(CANONICAL_INDEX, "PUBLIC,PUBLIC"),
                                      (CANONICAL_INDEX / "index.faiss").open("ab").write(b"x")),
     r"INDEX INVALID: artifact hashes do not match the manifest"),
    ("rejected_marker", lambda: (make_index(CANONICAL_INDEX, "PUBLIC"), (CANONICAL_INDEX / "REJECTED").write_text("qa failed\n")),
     r"INDEX INVALID: candidate is marked REJECTED: qa failed"),
    ("incomplete_missing_metadata", lambda: (make_index(CANONICAL_INDEX, "PUBLIC"), (CANONICAL_INDEX / "metadata.pkl").unlink()),
     r"INDEX INVALID: missing artifacts: \['metadata.pkl'\]"),
    ("failed_build_status", lambda: make_index(CANONICAL_INDEX, "PUBLIC", policy="failed_build"),
     r"INDEX INVALID: candidate build status is FAILED"),
    ("missing_identity_metadata", lambda: make_index(CANONICAL_INDEX, "PUBLIC,PUBLIC", drop_field="chunk_id"),
     r"INDEX INVALID: missing identity metadata"),
    ("unreadable_metadata_with_consistent_hashes", lambda: (make_index(CANONICAL_INDEX, "PUBLIC"),
                                                            (CANONICAL_INDEX / "metadata.pkl").write_bytes(b"not a pickle"),
                                                            rehash_manifest(CANONICAL_INDEX)),
     r"INDEX INVALID: (vector store did not load|could not inspect)"),
])
def test_s4_entrypoint_refuses_every_invalid_canonical_index(tmp_path, runtime_container, case, build, reason):
    sb = make_sandbox(tmp_path)
    build()
    r = run_entrypoint(sb)
    _refused(sb, r, reason)


def _verifier_stub(log: Path, behaviour: str) -> str:
    """python stub that replaces ONLY the verifier invocation."""
    return (f'echo "python $*" >> {log}\n'
            f'if [ "$1" = "scripts/verify_index.py" ]; then {behaviour}; fi\n'
            f'exec {sys.executable} "$@"')


@pytest.mark.parametrize("behaviour, status", [
    ("echo 'INDEX INVALID: synthetic verifier failure' >&2; exit 3", 3),
    ("kill -SEGV $$", 139),
    ("exit 1", 1),
])
def test_s4_any_nonzero_or_crashed_verifier_stops_startup_even_for_a_valid_index(tmp_path, runtime_container, behaviour, status):
    """`status` documents the verifier exit the entrypoint observes (139 = SIGSEGV)."""
    sb = make_sandbox(tmp_path, python_body=_verifier_stub(tmp_path / "calls.log", behaviour))
    make_index(CANONICAL_INDEX, "PUBLIC,PUBLIC")  # valid: only the verifier result can stop startup
    r = run_entrypoint(sb)
    calls = sb.calls()
    assert r.returncode == 1, r.stderr
    assert VERIFY_CALL in calls
    assert f"{CANONICAL_INDEX} failed validation" in r.stderr
    assert "nginx" not in calls and "uvicorn" not in calls and "curl" not in calls
    assert not NGINX_CONF.exists()


def test_s4_no_production_index_fails_closed_without_running_the_verifier(tmp_path, runtime_container):
    sb = make_sandbox(tmp_path)
    r = run_entrypoint(sb)
    assert r.returncode == 1
    assert f"No production index at {CANONICAL_INDEX}" in r.stderr
    assert "verify_index.py" not in sb.calls() and "uvicorn" not in sb.calls()


def test_s4_symlinked_canonical_index_is_refused(tmp_path, runtime_container):
    real = CANONICAL_DATA / "elsewhere"
    make_index(real, "PUBLIC")
    CANONICAL_INDEX.symlink_to(real, target_is_directory=True)
    sb = make_sandbox(tmp_path)
    r = run_entrypoint(sb)
    assert r.returncode == 1
    assert "runtime index path cannot contain symlinks" in r.stderr
    assert "verify_index.py" not in sb.calls() and "uvicorn" not in sb.calls()


# ---------------------------------------------------------------------------
# S5 -- proxy secret handling during startup
# ---------------------------------------------------------------------------

def test_s5_rendered_nginx_config_is_private_and_carries_only_the_proxy_secret(tmp_path, runtime_container):
    sb = make_sandbox(tmp_path)
    make_index(CANONICAL_INDEX, "PUBLIC")
    r = run_entrypoint(sb, {"AUTH_ENABLED": "true", "JWT_SECRET": SYNTHETIC_JWT_SECRET,
                            "JWT_AUDIENCE": "devhub", "JWT_ISSUER": "v12g"})
    assert r.returncode == 0, r.stderr
    st = NGINX_CONF.stat()
    assert st.st_uid == RUNTIME_UID and (st.st_mode & 0o777) == 0o600
    conf = NGINX_CONF.read_text()
    assert f'proxy_set_header X-Devhub-Internal "{SYNTHETIC_PROXY_SECRET}";' in conf
    assert SYNTHETIC_JWT_SECRET not in conf
    # never in argv of any started process, nor in startup output
    assert SYNTHETIC_PROXY_SECRET not in sb.calls()
    assert SYNTHETIC_PROXY_SECRET not in r.stdout + r.stderr
    assert (PROXY_SECRET_PATH.stat().st_mode & 0o777) == 0o400


def test_s5_entrypoint_never_reads_the_proxy_secret_from_the_environment(tmp_path, runtime_container):
    PROXY_SECRET_PATH.unlink()
    sb = make_sandbox(tmp_path)
    make_index(CANONICAL_INDEX, "PUBLIC")
    r = run_entrypoint(sb, {"DEVHUB_INTERNAL_PROXY_SECRET": SYNTHETIC_PROXY_SECRET})
    assert r.returncode != 0
    assert "nginx" not in sb.calls() and "uvicorn" not in sb.calls()
    assert not NGINX_CONF.exists()


def test_harness_paths_are_the_production_constants():
    """Guard against the harness drifting from the production contract."""
    from api import secret_config
    from index.runtime_path import CANONICAL_INDEX_ROOT
    assert CANONICAL_INDEX == CANONICAL_INDEX_ROOT == Path("/app/data/faiss_index")
    assert PROXY_SECRET_PATH == secret_config.PROXY_SECRET_FILE
    assert f"> {NGINX_CONF}" in ENTRYPOINT.read_text()
