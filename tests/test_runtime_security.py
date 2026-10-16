from pathlib import Path
from types import SimpleNamespace

import pytest

from api import secret_config
from index.runtime_path import CANONICAL_INDEX_ROOT, validate_runtime_index_root


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


def test_proxy_secret_requires_secure_file_and_does_not_accept_environment(monkeypatch, tmp_path, runtime_uid):
    path = tmp_path / "proxy-secret"
    path.write_text("synthetic-independent-proxy-secret-value-0123456789\n")
    path.chmod(0o400)
    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", path)
    monkeypatch.setenv("DEVHUB_INTERNAL_PROXY_SECRET", "synthetic-legacy-environment-value-not-used")
    # The file is owned by the runtime UID in the isolated test container.
    # The legacy environment variable is ignored; the file remains the source.
    assert secret_config.load_proxy_secret() == "synthetic-independent-proxy-secret-value-0123456789"


def test_proxy_secret_rejects_group_or_world_permissions(monkeypatch, tmp_path):
    path = tmp_path / "proxy-secret"
    path.write_text("synthetic-independent-proxy-secret-value-0123456789\n")
    path.chmod(0o440)
    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", path)
    # Isolate permission-bit validation from host UID ownership.
    original_stat = Path.stat
    def synthetic_owner(self, *args, **kwargs):
        result = original_stat(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_mode=result.st_mode, st_uid=10001)
        return result
    monkeypatch.setattr(Path, "stat", synthetic_owner)
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
def test_proxy_secret_rejects_placeholder_or_jwt_reuse(monkeypatch, tmp_path, runtime_uid, value, expected):
    path = tmp_path / "proxy-secret"
    path.write_text(value + "\n")
    path.chmod(0o400)
    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", path)
    monkeypatch.setenv("JWT_SECRET", value if expected == "independent" else "a-different-jwt-secret-value-0123456789")
    with pytest.raises(RuntimeError, match=expected):
        secret_config.load_proxy_secret()


def _owned_by_runtime_uid(monkeypatch, path):
    """Host-runnable twin of the real-UID tests above: only st_uid is synthetic."""
    original_stat = Path.stat

    def synthetic_owner(self, *args, **kwargs):
        result = original_stat(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_mode=result.st_mode, st_uid=10001)
        return result
    monkeypatch.setattr(Path, "stat", synthetic_owner)


def test_proxy_secret_logic_accepts_secure_file_and_ignores_environment(monkeypatch, tmp_path):
    path = tmp_path / "proxy-secret"
    path.write_text("synthetic-independent-proxy-secret-value-0123456789\n")
    path.chmod(0o400)
    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", path)
    monkeypatch.setenv("DEVHUB_INTERNAL_PROXY_SECRET", "synthetic-legacy-environment-value-not-used")
    _owned_by_runtime_uid(monkeypatch, path)
    assert secret_config.load_proxy_secret() == "synthetic-independent-proxy-secret-value-0123456789"


@pytest.mark.parametrize("value, expected", [
    ("change-me", "too short"),
    ("x" * 31, "too short"),
    ("change-me-in-production", "too short"),
    ("synthetic-jwt-test-secret-value-0123456789", "independent"),
])
def test_proxy_secret_logic_rejects_short_placeholder_or_jwt_reuse(monkeypatch, tmp_path, value, expected):
    path = tmp_path / "proxy-secret"
    path.write_text(value + "\n")
    path.chmod(0o400)
    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", path)
    monkeypatch.setenv("JWT_SECRET", value if expected == "independent" else "a-different-jwt-secret-value-0123456789")
    _owned_by_runtime_uid(monkeypatch, path)
    with pytest.raises(RuntimeError, match=expected):
        secret_config.load_proxy_secret()


def test_proxy_secret_rejects_file_owned_by_another_uid(monkeypatch, tmp_path):
    path = tmp_path / "proxy-secret"
    path.write_text("synthetic-independent-proxy-secret-value-0123456789\n")
    path.chmod(0o400)
    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", path)
    original_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: SimpleNamespace(st_mode=original_stat(self, *a, **k).st_mode, st_uid=0) if self == path else original_stat(self, *a, **k))
    with pytest.raises(RuntimeError, match="owned by UID 10001"):
        secret_config.load_proxy_secret()


def test_proxy_secret_missing_file_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(secret_config, "PROXY_SECRET_FILE", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="unavailable"):
        secret_config.load_proxy_secret()
