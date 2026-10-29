from pathlib import Path
from types import SimpleNamespace

import pytest

from index.runtime_path import CANONICAL_INDEX_ROOT, validate_runtime_index_root
from api import secret_config


def _set_runtime_owned_secret(monkeypatch, path):
    """Point the loader at a fixture while preserving its UID 10001 contract."""
    original_stat = Path.stat

    def synthetic_owner(self, *args, **kwargs):
        result = original_stat(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_mode=result.st_mode, st_uid=10001)
        return result

    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", path)
    monkeypatch.setattr(Path, "stat", synthetic_owner)


def test_runtime_path_accepts_only_canonical_root():
    assert validate_runtime_index_root() == CANONICAL_INDEX_ROOT
    with pytest.raises(ValueError):
        validate_runtime_index_root("/app/data/faiss_candidates/public")
    with pytest.raises(ValueError):
        validate_runtime_index_root("/app/data/faiss_index/../faiss_candidates/public")


def test_runtime_path_rejects_symlink_substitution(tmp_path):
    target = tmp_path / "approved"
    target.mkdir()
    substitute = tmp_path / "substitute"
    substitute.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_runtime_index_root(substitute, approved_root=substitute)


def test_proxy_secret_requires_secure_file_and_does_not_accept_environment(monkeypatch, tmp_path):
    path = tmp_path / "proxy-secret"
    path.write_text("synthetic-independent-proxy-secret-value-0123456789\n")
    path.chmod(0o400)
    _set_runtime_owned_secret(monkeypatch, path)
    monkeypatch.setenv("DEVHUB_INTERNAL_PROXY_SECRET", "synthetic-legacy-environment-value-not-used")
    # The legacy environment variable is ignored; the file remains the source.
    assert secret_config.load_proxy_secret() == "synthetic-independent-proxy-secret-value-0123456789"


def test_proxy_secret_rejects_group_or_world_permissions(monkeypatch, tmp_path):
    path = tmp_path / "proxy-secret"
    path.write_text("synthetic-independent-proxy-secret-value-0123456789\n")
    path.chmod(0o440)
    _set_runtime_owned_secret(monkeypatch, path)
    with pytest.raises(RuntimeError, match="mode 0400"):
        secret_config.load_proxy_secret()


def test_proxy_secret_rejects_symlink(monkeypatch, tmp_path):
    real = tmp_path / "real"
    real.write_text("synthetic-independent-proxy-secret-value-0123456789\n")
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", link)
    with pytest.raises(RuntimeError, match="non-symlink"):
        secret_config.load_proxy_secret()


@pytest.mark.parametrize("value, expected", [
    ("change-me", "too short"),
    ("synthetic-jwt-test-secret-value-0123456789", "independent"),
])
def test_proxy_secret_rejects_placeholder_or_jwt_reuse(monkeypatch, tmp_path, value, expected):
    path = tmp_path / "proxy-secret"
    path.write_text(value + "\n")
    path.chmod(0o400)
    _set_runtime_owned_secret(monkeypatch, path)
    monkeypatch.setenv("JWT_SECRET", value if expected == "independent" else "a-different-jwt-secret-value-0123456789")
    with pytest.raises(RuntimeError, match=expected):
        secret_config.load_proxy_secret()
