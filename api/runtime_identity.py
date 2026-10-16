"""Runtime process-identity contract: Dev Hub serves only as UID 10001.

docker-entrypoint.sh enforces this before anything starts, but the image's
CMD can be replaced (e.g. ``docker run <image> uvicorn api.main:app``), which
skips the shell gate entirely. The FastAPI startup event therefore enforces
the same contract, so every ASGI launch path that runs the application's
lifespan refuses to serve under any other identity -- including root.
"""
import os

RUNTIME_UID = 10001


def require_runtime_uid(euid: int | None = None) -> int:
    """Return the effective UID if it is exactly RUNTIME_UID, else refuse."""
    actual = os.geteuid() if euid is None else euid
    if type(actual) is not int or actual != RUNTIME_UID:
        raise RuntimeError(
            f"Dev Hub must run as UID {RUNTIME_UID} (effective UID {actual!r}); refusing to start."
        )
    return actual
