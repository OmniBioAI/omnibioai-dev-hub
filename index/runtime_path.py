"""Canonical, read-only production index path validation."""
from pathlib import Path

CANONICAL_INDEX_ROOT = Path("/app/data/faiss_index")


def validate_runtime_index_root(
    path: str | Path = CANONICAL_INDEX_ROOT,
    *,
    approved_root: str | Path = CANONICAL_INDEX_ROOT,
) -> Path:
    """Accept only the canonical path and reject symlinks and escapes."""
    candidate = Path(path)
    approved = Path(approved_root)
    if not candidate.is_absolute() or candidate != approved:
        raise ValueError("runtime index path must be the canonical production index root")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("runtime index path cannot contain symlinks")
    resolved = candidate.resolve(strict=False)
    if resolved != approved:
        raise ValueError("runtime index path resolves outside the canonical production index root")
    return resolved
