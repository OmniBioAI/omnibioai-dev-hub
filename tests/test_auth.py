"""Unit tests for api/auth.py's JWT dependency.

These tests exercise api/auth.py's current behavior as written, including
validate_auth_config() -- the startup guard that makes the app refuse to
start when AUTH_ENABLED=true but JWT_SECRET is unset/empty (previously a
silent request-time auth bypass; see the tests near the bottom of this
file, and the PR description for before/after).
"""

import asyncio
import time

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.auth import (
    AuthError,
    extract_token,
    require_auth,
    validate_auth_config,
    validate_token,
)

SECRET = "test-secret-value"


def _make_token(secret=SECRET, sub="alice", exp_delta=3600, **extra_claims):
    payload = {"sub": sub, "exp": int(time.time()) + exp_delta, **extra_claims}
    return jwt.encode(payload, secret, algorithm="HS256")


# Minimal throwaway app exercising the real require_auth dependency,
# isolated from api/routes/rag.py's control-plane wiring.
app = FastAPI()


@app.get("/protected")
def protected(actor: str = Depends(require_auth)):
    return {"actor": actor}


client = TestClient(app)


# =========================================================
# extract_token() -- unit tests
# =========================================================

def test_extract_token_valid():
    assert extract_token("Bearer abc.def.ghi") == "abc.def.ghi"


def test_extract_token_missing_header():
    with pytest.raises(AuthError, match="Authorization header is missing"):
        extract_token(None)


def test_extract_token_missing_bearer_prefix():
    with pytest.raises(AuthError, match="must be 'Bearer <token>'"):
        extract_token("sometoken")


def test_extract_token_wrong_scheme():
    with pytest.raises(AuthError, match="must be 'Bearer <token>'"):
        extract_token("Basic abc.def.ghi")


# =========================================================
# validate_token() -- unit tests
# =========================================================

def test_validate_token_valid():
    token = _make_token()
    payload = validate_token(token, SECRET)
    assert payload["sub"] == "alice"


def test_validate_token_expired():
    token = _make_token(exp_delta=-3600)  # expired an hour ago
    with pytest.raises(AuthError, match="Token has expired"):
        validate_token(token, SECRET)


def test_validate_token_malformed_garbage():
    with pytest.raises(AuthError, match="Invalid token"):
        validate_token("this-is-not-a-jwt", SECRET)


def test_validate_token_wrong_secret_rejected():
    token = _make_token(secret=SECRET)
    with pytest.raises(AuthError, match="Invalid token"):
        validate_token(token, "a-different-secret")


# =========================================================
# require_auth() -- end to end via TestClient (real header parsing)
# =========================================================

def test_require_auth_disabled_no_ops(monkeypatch):
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)

    resp = client.get("/protected")

    assert resp.status_code == 200
    assert resp.json()["actor"] == "system"


def test_require_auth_disabled_ignores_bad_internal_header(monkeypatch):
    # AUTH_ENABLED=false must no-op unconditionally -- confirm it doesn't
    # accidentally start caring about X-Devhub-Internal's value (that check
    # only exists in the AUTH_ENABLED=true branch), which would be a
    # behavior change from what's documented.
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.setenv("JWT_SECRET", SECRET)

    resp = client.get("/protected", headers={"X-Devhub-Internal": "totally-wrong-value"})

    assert resp.status_code == 200
    assert resp.json()["actor"] == "system"


def test_require_auth_enabled_valid_jwt_accepted(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = _make_token(sub="alice")

    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json()["actor"] == "alice"


def test_require_auth_enabled_jwt_without_identity_claim_falls_back_to_unknown(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    # A structurally valid, correctly-signed token that carries none of the
    # identity fields require_auth() checks (sub/service/username/email).
    payload = {"exp": int(time.time()) + 3600, "role": "irrelevant"}
    token = jwt.encode(payload, SECRET, algorithm="HS256")

    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json()["actor"] == "unknown"


def test_require_auth_enabled_expired_jwt_rejected(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = _make_token(exp_delta=-3600)

    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 401
    assert "expired" in resp.json()["detail"].lower()


def test_require_auth_enabled_garbage_token_rejected(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)

    resp = client.get("/protected", headers={"Authorization": "Bearer garbage.not.a.jwt"})

    assert resp.status_code == 401
    assert "invalid token" in resp.json()["detail"].lower()


def test_require_auth_enabled_missing_authorization_header_rejected(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)

    resp = client.get("/protected")

    assert resp.status_code == 401
    assert "missing" in resp.json()["detail"].lower()


def test_require_auth_internal_header_bypasses_jwt(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)

    # No Authorization header at all -- only the matching internal header.
    resp = client.get("/protected", headers={"X-Devhub-Internal": SECRET})

    assert resp.status_code == 200
    assert resp.json()["actor"] == "devhub-ui"


def test_require_auth_internal_header_absent_falls_through_to_jwt(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = _make_token(sub="bob")

    # No X-Devhub-Internal header sent -- a valid JWT should still work.
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json()["actor"] == "bob"


def test_require_auth_internal_header_wrong_value_falls_through_and_fails(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)

    # Wrong internal-header value and no Authorization header -- falls
    # through to the JWT path, which then has nothing to validate.
    resp = client.get("/protected", headers={"X-Devhub-Internal": "wrong-value"})

    assert resp.status_code == 401


# =========================================================
# JWT_SECRET unset -- documents ACTUAL current behavior.
# =========================================================

def test_require_auth_unset_secret_internal_header_path_is_unreachable(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)

    # internal_secret resolves to "" (falsy), so `if internal_secret and
    # x_devhub_internal == internal_secret` can never be true here, even if
    # a caller sends an empty-string header value that would otherwise
    # equality-match "" -- the `and` short-circuits first.
    resp = client.get("/protected", headers={"X-Devhub-Internal": ""})

    assert resp.status_code == 401  # falls through to the JWT path, no Authorization header


# =========================================================
# validate_auth_config() -- startup guard against the empty-secret bypass.
#
# Previously, AUTH_ENABLED=true with JWT_SECRET unset/empty silently fell
# through at request time to validate_token(token, "") -- and PyJWT does
# not reject an empty HMAC secret, so anyone could forge a token signed
# with secret="" (e.g. jwt.encode({"sub": "attacker"}, "", algorithm=
# "HS256")) and it would pass validation. That let a deployment look
# auth-protected while actually being wide open. These tests now confirm
# the app refuses to start in that configuration instead of silently
# serving requests with auth effectively disabled.
# =========================================================

def test_validate_auth_config_raises_when_enabled_without_secret(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="AUTH_ENABLED is true but JWT_SECRET is not set"):
        validate_auth_config()


def test_validate_auth_config_raises_when_enabled_with_blank_secret(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "   ")  # whitespace-only -> strips to ""

    with pytest.raises(RuntimeError, match="AUTH_ENABLED is true but JWT_SECRET is not set"):
        validate_auth_config()


def test_validate_auth_config_passes_when_enabled_with_secret_set(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)

    validate_auth_config()  # must not raise


def test_validate_auth_config_ignores_missing_secret_when_auth_disabled(monkeypatch):
    # Requirement: AUTH_ENABLED=false must be completely unaffected by this
    # check -- JWT_SECRET stays irrelevant when auth is off.
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)

    validate_auth_config()  # must not raise


def test_startup_refuses_when_auth_enabled_without_secret(monkeypatch):
    """End-to-end version of the guard: api.main's real startup_event()
    (not just validate_auth_config() in isolation) refuses to start in
    this configuration."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)

    from api.main import startup_event

    with pytest.raises(RuntimeError, match="AUTH_ENABLED is true but JWT_SECRET is not set"):
        asyncio.run(startup_event())
