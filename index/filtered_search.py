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

import numpy as np


def cosine_relevance(index: Any, row: int, query_vec: Any) -> float | None:
    """Cosine similarity between the query and the stored vector at `row`.

    The index scores by raw inner product over UN-normalized vectors (stored
    norms range ~15.7-23.5), so its score is not a usable relevance measure.
    Returns None if the index cannot reconstruct vectors.
    """
    try:
        stored = np.asarray(index.reconstruct(int(row)), dtype=np.float32).reshape(-1)
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(stored) * np.linalg.norm(q))
        return float(np.dot(stored, q) / denom) if denom else None
    except Exception:  # noqa: BLE001 -- an index that cannot reconstruct vectors simply has no relevance
        return None


def search_allowed(
    index: Any,
    metadata: list[dict],
    query_vec: Any,
    want: int,
    accept: Callable[[dict], bool],
    initial_k: int | None = None,
    min_relevance: float | None = None,
) -> list[tuple[float, int, float | None]]:
    """Return up to `want` (score, row, cosine) triples, best first, whose metadata passes `accept`.

    With `min_relevance`, a chunk whose cosine similarity to the query is below
    it (or cannot be computed) is skipped, so a query nothing in the allowed
    set is genuinely relevant to returns nothing instead of nearest neighbours.
    """
    total = index.ntotal
    if total == 0 or want <= 0:
        return []
    k = min(max(initial_k or want, want), total)
    while True:
        scores, indices = index.search(query_vec, k)
        found: list[tuple[float, int, float | None]] = []
        for score, row in zip(scores[0], indices[0], strict=False):
            if row < 0 or row >= len(metadata):
                continue
            if not accept(metadata[row]):
                continue
            relevance = cosine_relevance(index, row, query_vec)
            if min_relevance is not None and (relevance is None or relevance < min_relevance):
                continue
            found.append((float(score), int(row), relevance))
            if len(found) >= want:
                return found
        if k >= total:
            return found
        k = min(k * 4, total)
