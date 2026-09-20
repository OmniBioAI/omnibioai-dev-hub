"""Exact filtered nearest-neighbour search over a flat FAISS index.

Visibility (and scope) filters are applied after FAISS ranks vectors. With a
fixed top-k that starves the caller whenever most of the index is excluded:
the ordinary-caller path is PUBLIC-only while ~93% of the Dev Hub corpus is
INTERNAL, so the k nearest chunks are usually all INTERNAL and all filtered
out, returning nothing even though relevant PUBLIC chunks exist.

This widens k (x4 per round, capped at ntotal) until `want` accepted results
are found or the whole index has been ranked. Results are always re-checked
by `accept`, so widening can never admit a chunk the filter rejects.
No pure-python import of faiss so it is usable from engine code and tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def search_allowed(
    index: Any,
    metadata: list[dict],
    query_vec: Any,
    want: int,
    accept: Callable[[dict], bool],
    initial_k: int | None = None,
) -> list[tuple[float, int]]:
    """Return up to `want` (score, row) pairs, best first, whose metadata passes `accept`."""
    total = index.ntotal
    if total == 0 or want <= 0:
        return []
    k = min(max(initial_k or want, want), total)
    while True:
        scores, indices = index.search(query_vec, k)
        found: list[tuple[float, int]] = []
        for score, row in zip(scores[0], indices[0], strict=False):
            if row < 0 or row >= len(metadata):
                continue
            if accept(metadata[row]):
                found.append((float(score), int(row)))
                if len(found) >= want:
                    return found
        if k >= total:
            return found
        k = min(k * 4, total)
