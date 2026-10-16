"""F-UID-1 regression tests: Dev Hub must never serve HTTP under any identity but
UID 10001, or with an invalid auth configuration -- independent of the ASGI
lifespan protocol (`--lifespan off` skips startup events entirely).

Every test here starts a REAL server process with the REAL process identity
(no monkeypatching): on a developer host (UID 1000) and in UID 0 / 65534
qualification containers they assert refusal; in the UID 10001 container they
assert service. "Refusal" means the process exits non-zero before it ever binds
(nothing accepts a TCP connection on the port), not a warning while serving.
"""

from __future__ import annotations

import http.client
import importlib
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from api.runtime_identity import RUNTIME_UID
from tests.startup_harness import (
    PROXY_SECRET_PATH,
    ROOT,
    SYNTHETIC_JWT_SECRET,
    SYNTHETIC_PROXY_SECRET,
)

BIN = Path(sys.executable).parent
IDENTITY_REFUSAL = "must run as UID 10001"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port: int, path: str, headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)  # direct: no proxy environment involved
    try:
        conn.request("GET", path, headers=headers or {})
        return conn.getresponse().status
    finally:
        conn.close()


def _accepts_connections(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return True
    except OSError:
        return False


LAUNCHERS = {
    "uvicorn_cli_lifespan_on": lambda p: [str(BIN / "uvicorn"), "api.main:app", "--host", "127.0.0.1", "--port", str(p)],
    "uvicorn_cli_lifespan_off": lambda p: [str(BIN / "uvicorn"), "api.main:app", "--host", "127.0.0.1", "--port", str(p), "--lifespan", "off"],
    "python_m_uvicorn_lifespan_off": lambda p: [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(p), "--lifespan", "off"],
    "uvicorn_run_import_string_lifespan_off": lambda p: [sys.executable, "-c", f"import uvicorn; uvicorn.run('api.main:app', host='127.0.0.1', port={p}, lifespan='off')"],
    "uvicorn_run_app_object_lifespan_off": lambda p: [sys.executable, "-c", f"import uvicorn, api.main as m; uvicorn.run(m.app, host='127.0.0.1', port={p}, lifespan='off')"],
}


class Server:
    def __init__(self, argv, env_extra):
        env = {"PATH": f"{BIN}:/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "PYTHONDONTWRITEBYTECODE": "1"}
        env.update(env_extra)
        self.proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.lines: list[str] = []
        threading.Thread(target=lambda: self.lines.extend(self.proc.stdout), daemon=True).start()

    @property
    def output(self) -> str:
        return "".join(self.lines)

    def wait_ready_or_exit(self, timeout=60) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end and self.proc.poll() is None and "Uvicorn running" not in self.output:
            time.sleep(0.1)
        time.sleep(0.3)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=30)


def _launch(name: str, env_extra: dict | None = None):
    port = _free_port()
    server = Server(LAUNCHERS[name](port), env_extra or {})
    server.wait_ready_or_exit()
    return server, port


def _assert_refused_before_serving(server: Server, port: int, reason: str) -> None:
    try:
        server.proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pytest.fail(f"server still running; refusal expected: {server.output[-800:]}")
    assert server.proc.returncode != 0, server.output
    assert reason in server.output, server.output[-1500:]
    assert "Uvicorn running" not in server.output and "Application startup complete" not in server.output
    assert not _accepts_connections(port)


# ---------------------------------------------------------------------------
# Identity: real process UID, every supported launch path, lifespan on and off
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("launcher", sorted(LAUNCHERS))
def test_every_launch_path_enforces_the_real_process_identity(launcher):
    server, port = _launch(launcher)
    try:
        if os.geteuid() == RUNTIME_UID:
            assert server.proc.poll() is None, server.output
            assert _get(port, "/health") == 200
        else:
            _assert_refused_before_serving(server, port, IDENTITY_REFUSAL)
    finally:
        server.stop()


def test_importing_the_application_module_enforces_the_real_process_identity():
    r = subprocess.run([sys.executable, "-c", "import api.main"], cwd=ROOT, capture_output=True, text=True,
                       env={"PATH": f"{BIN}:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}, timeout=120, check=False)
    if os.geteuid() == RUNTIME_UID:
        assert r.returncode == 0, r.stderr
    else:
        assert r.returncode != 0 and IDENTITY_REFUSAL in r.stderr


def test_import_time_gate_checks_identity_before_auth_configuration(monkeypatch):
    """In-process complement: under UID 0 with a broken auth config, the identity
    refusal is what stops the import -- identity is checked first."""
    import api.main as main_module
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    try:
        with patch("os.geteuid", return_value=0), pytest.raises(RuntimeError, match=IDENTITY_REFUSAL):
            importlib.reload(main_module)
        with patch("os.geteuid", return_value=RUNTIME_UID), pytest.raises(RuntimeError, match="AUTH_ENABLED is true but JWT_SECRET.* is not set"):
            importlib.reload(main_module)
    finally:
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        with patch("os.geteuid", return_value=RUNTIME_UID):
            importlib.reload(main_module)


# ---------------------------------------------------------------------------
# Auth configuration: enforced with lifespan OFF (real UID 10001 container)
# ---------------------------------------------------------------------------

VALID_AUTH = {"AUTH_ENABLED": "true", "JWT_SECRET": SYNTHETIC_JWT_SECRET, "JWT_AUDIENCE": "devhub", "JWT_ISSUER": "v12i"}


def _rewrite_proxy_secret(value: str | None) -> None:
    if PROXY_SECRET_PATH.exists():
        PROXY_SECRET_PATH.unlink()
    if value is not None:
        PROXY_SECRET_PATH.write_text(value + "\n")
        PROXY_SECRET_PATH.chmod(0o400)


@pytest.mark.parametrize("case, env_changes, proxy_secret, reason", [
    ("jwt_secret_missing", {"JWT_SECRET": None}, SYNTHETIC_PROXY_SECRET, "JWT_SECRET is not set"),
    ("jwt_audience_missing", {"JWT_AUDIENCE": None}, SYNTHETIC_PROXY_SECRET, "JWT_AUDIENCE is not set"),
    ("jwt_issuer_missing", {"JWT_ISSUER": None}, SYNTHETIC_PROXY_SECRET, "JWT_ISSUER is not set"),
    ("jwt_secret_placeholder", {"JWT_SECRET": "change-me"}, SYNTHETIC_PROXY_SECRET, "insecure placeholder"),
    ("proxy_secret_file_missing", {}, None, "independent proxy secret is invalid"),
    ("proxy_secret_equals_jwt_secret", {}, SYNTHETIC_JWT_SECRET, "independent proxy secret is invalid"),
    ("proxy_secret_too_short", {}, "short-secret", "independent proxy secret is invalid"),
])
def test_invalid_auth_configuration_refuses_service_even_with_lifespan_off(runtime_container, case, env_changes, proxy_secret, reason):
    _rewrite_proxy_secret(proxy_secret)
    env = {k: v for k, v in {**VALID_AUTH, **env_changes}.items() if v is not None}
    server, port = _launch("uvicorn_cli_lifespan_off", env)
    try:
        _assert_refused_before_serving(server, port, reason)
    finally:
        server.stop()


@pytest.mark.parametrize("launcher", ["uvicorn_cli_lifespan_on", "uvicorn_cli_lifespan_off"])
def test_valid_auth_configuration_serves_and_still_enforces_authentication(runtime_container, launcher):
    server, port = _launch(launcher, VALID_AUTH)
    try:
        assert server.proc.poll() is None, server.output
        assert _get(port, "/health") == 200
        assert _get(port, "/api/discovery/categories") == 401
        assert _get(port, "/api/discovery/categories", {"X-Devhub-Internal": SYNTHETIC_JWT_SECRET}) == 401
        assert _get(port, "/api/discovery/categories", {"Authorization": "Bearer not.a.jwt"}) == 401
        assert _get(port, "/api/discovery/categories", {"X-Devhub-Internal": SYNTHETIC_PROXY_SECRET}) != 401
    finally:
        server.stop()
