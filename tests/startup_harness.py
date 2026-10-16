"""Shared harness for running the REAL docker-entrypoint.sh in a sandbox.

python/curl/nginx/uvicorn/sleep (and the ownership tools) are stubbed on PATH
and every invocation is logged. `id` is either the real binary (real process
identity) or a shim answering `id -u` with a chosen value.

Tests that need the canonical production paths (/app/data/faiss_index,
/run/secrets, /etc/nginx/conf.d) use the `runtime_container` fixture: those
paths are root-owned territory on a developer host, so the fixture only runs
inside the isolated UID-10001 qualification container, where they are
disposable tmpfs mounts. Elsewhere it skips with an explicit reason -- it
never redirects the entrypoint to a different index path.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker-entrypoint.sh"
BASH = shutil.which("bash")
RUNTIME_UID = 10001

CANONICAL_INDEX = Path("/app/data/faiss_index")
CANONICAL_DATA = CANONICAL_INDEX.parent
PROXY_SECRET_PATH = Path("/run/secrets/devhub_internal_proxy_secret")
NGINX_CONF = Path("/etc/nginx/conf.d/devhub.conf")

SYNTHETIC_PROXY_SECRET = "synthetic-v12g-independent-proxy-secret-0123456789"
SYNTHETIC_JWT_SECRET = "synthetic-v12g-jwt-signing-secret-0123456789abcdef"

UID_REFUSAL = "Dev Hub must run as UID 10001; refusing startup."
OWNERSHIP_TOOLS = ("chown", "chgrp", "chmod", "setfacl", "find", "install")

MAKE_INDEX = textwrap.dedent('''
    import json, sys, numpy as np
    sys.path.insert(0, {root!r})
    from index.vector_store import VectorStore
    from index.lifecycle import artifact_hashes, write_manifest
    out, visibilities, policy, drop_field = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4]
    vs = VectorStore()
    meta = [{{"text": f"t{{i}}", "source": f"repo:d{{i}}.md@abc", "repo": "repo", "relative_path": f"d{{i}}.md",
             "source_revision": "abc", "document_id": f"doc{{i}}", "chunk_id": f"c{{i}}", "content_hash": "h", "chunk_hash": "ch",
             "content_state": "CURRENT", "verification_state": "CONFIGURED", "bundle": None,
             "citation": {{"repository": "repo"}}, "visibility": v}} for i, v in enumerate(visibilities)]
    if drop_field:
        for m in meta:
            m.pop(drop_field, None)
    vs.add(np.random.default_rng(0).normal(size=(len(meta), 768)).astype("float32"), meta)
    vs.save(out)
    counts = {{k: visibilities.count(k) for k in ("PUBLIC", "INTERNAL", "REVIEW_REQUIRED")}}
    manifest = {{"build_id": "b", "build_status": "CLEAN", "artifact_hashes": artifact_hashes(out), "visibility_counts": counts}}
    if policy == "public_only":
        manifest["visibility_policy"] = "PUBLIC_ONLY"
    if policy == "failed_build":
        manifest["visibility_policy"] = "PUBLIC_ONLY"
        manifest["build_status"] = "FAILED"
    write_manifest(out, manifest)
''')


def make_index(path: Path, visibilities: str, policy: str = "public_only", drop_field: str = "") -> None:
    """Build a real, synthetic FAISS index (768-d) with a manifest, in a subprocess
    (other test modules replace `faiss` with a mock in-process)."""
    subprocess.run(
        [sys.executable, "-c", MAKE_INDEX.format(root=str(ROOT)), str(path), visibilities, policy, drop_field],
        check=True,
    )


REHASH_MANIFEST = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, {root!r})
    from index.lifecycle import artifact_hashes, read_manifest, write_manifest
    m = read_manifest(sys.argv[1]); m["artifact_hashes"] = artifact_hashes(sys.argv[1]); write_manifest(sys.argv[1], m)
''')


def rehash_manifest(path: Path) -> None:
    """Re-sign artifact hashes after a deliberate corruption, so validation gets
    past the hash check and reaches the load/metadata branches."""
    subprocess.run([sys.executable, "-c", REHASH_MANIFEST.format(root=str(ROOT)), str(path)], check=True)


@dataclass
class Sandbox:
    bin_dir: Path
    log: Path
    tmp: Path

    def calls(self) -> str:
        return self.log.read_text()


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/bin/bash\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def make_sandbox(tmp: Path, *, uid: str | None = str(RUNTIME_UID), python_body: str | None = None) -> Sandbox:
    """uid=None keeps the REAL `id`; otherwise an `id` shim answers `id -u` with `uid`."""
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    log = tmp / "calls.log"
    log.write_text("")
    _stub(bin_dir, "python", python_body or f'echo "python $*" >> {log}\nexec {sys.executable} "$@"')
    _stub(bin_dir, "curl", f'echo "curl $*" >> {log}\nexit 0')
    _stub(bin_dir, "nginx", f'echo "nginx $*" >> {log}')
    _stub(bin_dir, "uvicorn", f'echo "uvicorn $*" >> {log}')
    _stub(bin_dir, "sleep", "exit 0")
    for tool in OWNERSHIP_TOOLS:
        _stub(bin_dir, tool, f'echo "{tool} $*" >> {log}\nexit 0')
    if uid is not None:
        _stub(bin_dir, "id", f'if [ "$*" = "-u" ]; then printf "%s\\n" {uid!r}; else exec /usr/bin/id "$@"; fi')
    _assert_stubs_are_live(bin_dir, log)
    return Sandbox(bin_dir, log, tmp)


def _assert_stubs_are_live(bin_dir: Path, log: Path) -> None:
    """Fail loudly if PATH stubs cannot execute (e.g. a noexec /tmp): bash would
    silently fall through to the real binaries and every "X not in calls"
    assertion would pass vacuously."""
    probe = subprocess.run([BASH, "-c", "curl --v12g-harness-probe"], env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
                           capture_output=True, text=True, check=False)
    assert probe.returncode == 0 and log.read_text() == "curl --v12g-harness-probe\n", (
        f"sandbox stubs in {bin_dir} are not executable: {probe.stderr}")
    log.write_text("")


def run_entrypoint(sb: Sandbox, extra_env: dict | None = None, *, path: str | None = None):
    env = {
        "PATH": path if path is not None else f"{sb.bin_dir}:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(sb.bin_dir),
        # An attempted override must be ignored: only /app/data/faiss_index is served.
        "DEVHUB_INDEX_DIR": str(sb.tmp / "untrusted_override_index"),
    }
    env.update(extra_env or {})
    return subprocess.run([BASH, str(ENTRYPOINT)], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=120, check=False)


def tree_snapshot(root: Path) -> dict:
    """Ownership/permission/ctime fingerprint of every entry under root.
    chown(2) updates ctime even when the owner is unchanged, so a no-op
    recursive chown is still visible here."""
    snap = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [""] + dirnames + filenames:
            p = Path(dirpath) / name if name else Path(dirpath)
            st = p.lstat()
            snap[str(p)] = (st.st_uid, st.st_gid, stat.S_IMODE(st.st_mode), st.st_ctime_ns, st.st_mtime_ns)
    return snap


def _clear(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


def runtime_container_available() -> tuple[bool, str]:
    if os.geteuid() != RUNTIME_UID:
        return False, f"effective UID {os.geteuid()} is not the runtime UID {RUNTIME_UID}"
    for p in (CANONICAL_DATA, PROXY_SECRET_PATH.parent, NGINX_CONF.parent):
        if not (p.is_dir() and os.access(p, os.W_OK)):
            return False, f"{p} is not a writable disposable mount"
    return True, ""


@pytest.fixture
def runtime_uid():
    """Real runtime identity: files this test creates are genuinely owned by UID 10001."""
    if os.geteuid() != RUNTIME_UID:
        pytest.skip(f"RUNTIME_UID_REQUIRED: effective UID {os.geteuid()} cannot create files owned by UID "
                    f"{RUNTIME_UID}; executed in the isolated UID-10001 container")


@pytest.fixture
def runtime_container():
    """Disposable canonical paths inside the UID-10001 qualification container.

    Provisions a synthetic proxy secret (0400, owned by the runtime UID) so the
    only thing that can stop startup in an index test is the index gate itself.
    """
    ok, why = runtime_container_available()
    if not ok:
        pytest.skip(f"RUNTIME_CONTAINER_REQUIRED: {why}; executed in the isolated UID-10001 container")
    for p in (CANONICAL_DATA, PROXY_SECRET_PATH.parent, NGINX_CONF.parent):
        _clear(p)
    PROXY_SECRET_PATH.write_text(SYNTHETIC_PROXY_SECRET + "\n")
    PROXY_SECRET_PATH.chmod(0o400)
    try:
        yield
    finally:
        for p in (CANONICAL_DATA, PROXY_SECRET_PATH.parent, NGINX_CONF.parent):
            _clear(p)
