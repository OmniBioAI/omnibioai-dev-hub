"""Strict file-backed Dev Hub internal proxy secret loading."""
import hmac
import os
import stat
from pathlib import Path

PROXY_SECRET_FILE = Path("/run/secrets/devhub_internal_proxy_secret")
PLACEHOLDERS = {"change-me", "change-me-in-production", "", "null", "none"}


def load_proxy_secret() -> str:
    path = PROXY_SECRET_FILE
    if path.is_symlink():
        raise RuntimeError("Dev Hub proxy secret must be a regular, non-symlink file")
    try:
        st = path.stat()
    except OSError as exc:
        raise RuntimeError("Dev Hub proxy secret file is unavailable") from exc
    if not stat.S_ISREG(st.st_mode) or st.st_uid != 10001 or stat.S_IMODE(st.st_mode) != 0o400:
        raise RuntimeError("Dev Hub proxy secret file must be regular, owned by UID 10001, mode 0400")
    value = path.read_text(encoding="utf-8").strip()
    if len(value) < 32 or value.lower() in PLACEHOLDERS:
        raise RuntimeError("Dev Hub proxy secret file is empty, too short, or a placeholder")
    jwt_secret = os.getenv("JWT_SECRET", "").strip()
    if jwt_secret and hmac.compare_digest(jwt_secret, value):
        raise RuntimeError("Dev Hub proxy secret must be independent of JWT_SECRET")
    return value
