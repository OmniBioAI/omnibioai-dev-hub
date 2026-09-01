"""JWT auth dependency for /rag/* routes -- same shared-secret style as
omnibioai-model-registry's require_auth (Authorization: Bearer <jwt>,
HS256, verified against JWT_SECRET, gated by AUTH_ENABLED so it stays a
no-op until a deployment turns it on)."""

import os
from typing import Annotated

import jwt as _jwt
from fastapi import Header, HTTPException, Request


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


def _auth_enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "").strip().lower() == "true"


def _jwt_secret() -> str:
    return os.getenv("JWT_SECRET", "").strip()


def validate_auth_config() -> None:
    """Fail loudly at startup if auth is enabled but misconfigured.

    AUTH_ENABLED=true with an unset/empty JWT_SECRET used to fall through
    at request time to validate_token(token, "") -- and PyJWT does not
    reject an empty HMAC secret, so anyone could forge a token signed with
    secret="" (e.g. jwt.encode({"sub": "attacker"}, "", algorithm="HS256"))
    and it would pass validation. That left a deployment looking secured
    while actually being wide open. Refusing to start is the fix: a loud
    crash at boot beats a silent bypass in production. Call this from the
    app's startup path (see api/main.py) -- not from require_auth(), so a
    misconfigured deployment never serves a single request.
    """
    if _auth_enabled() and not _jwt_secret():
        raise RuntimeError(
            "AUTH_ENABLED is true but JWT_SECRET is not set -- "
            "refusing to start with auth silently disabled."
        )


def extract_token(authorization_header: str | None) -> str:
    if not authorization_header:
        raise AuthError("Authorization header is missing", 401)
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Authorization header must be 'Bearer <token>'", 401)
    return parts[1]


def validate_token(token: str, secret: str) -> dict:
    try:
        return _jwt.decode(token, secret, algorithms=["HS256"])
    except _jwt.ExpiredSignatureError:
        raise AuthError("Token has expired", 401)
    except _jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid token: {exc}", 401)


# nginx (devhub.conf, same container) injects this exact header when it
# proxies the built UI's same-origin requests to FastAPI -- see Dockerfile.
# A request arriving straight from outside the container (bypassing nginx)
# has no way to attach it and must present a real Bearer JWT instead.
_INTERNAL_HEADER = "x-devhub-internal"


async def require_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_devhub_internal: Annotated[str | None, Header()] = None,
) -> str:
    """FastAPI dependency. Returns actor string. No-op ("system") unless
    AUTH_ENABLED=true. When enabled, accepts either a valid Bearer JWT or
    the internal proxy header nginx sets for the first-party UI."""
    if not _auth_enabled():
        return "system"

    internal_secret = _jwt_secret()
    if internal_secret and x_devhub_internal == internal_secret:
        return "devhub-ui"

    try:
        token = extract_token(authorization)
        payload = validate_token(token, internal_secret)
        for field in ("sub", "service", "username", "email"):
            value = payload.get(field)
            if value:
                return str(value)
        return "unknown"
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
