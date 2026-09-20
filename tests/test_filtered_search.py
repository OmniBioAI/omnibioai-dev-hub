"""Regression tests: visibility filtering must not starve results.

Visibility is applied after nearest-neighbour ranking. If the k nearest chunks
are all excluded (e.g. the INTERNAL majority on a PUBLIC-only path), a fixed
top-k returns nothing although relevant allowed chunks exist further down.
"""

from unittest.mock import patch

import numpy as np

from index.filtered_search import search_allowed
from index.vector_store import VectorStore
from rag.engine import RAGEngine

DIM = 768


class FakeIndex:
    """Brute-force inner-product index with the faiss search() contract (-1 padding)."""

    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ntotal = len(self.vectors)
        self.search_calls = []

    def search(self, q, k):
        self.search_calls.append(k)
        scores = self.vectors @ np.asarray(q, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores)[:k]
        pad = k - len(order)
        return (
            np.concatenate([scores[order], np.full(pad, -1e9)])[None, :],
            np.concatenate([order, np.full(pad, -1)])[None, :],
        )


def _corpus(n_internal=200, n_public=5):
    """INTERNAL chunks are all closer to the query than every PUBLIC chunk."""
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    vecs, meta = [], []
    for i in range(n_internal):
        v = np.zeros(DIM, dtype=np.float32)
        v[0], v[1] = 0.99 - i * 1e-4, 0.1
        vecs.append(v)
        meta.append({"text": f"internal-{i}", "source": "s", "visibility": "INTERNAL", "repo": "r", "bundle": "b"})
    for i in range(n_public):
        v = np.zeros(DIM, dtype=np.float32)
        v[0], v[1] = 0.5 - i * 1e-3, 0.5
        vecs.append(v)
        meta.append({"text": f"public-{i}", "source": "s", "visibility": "PUBLIC", "repo": "r", "bundle": "b"})
    return q, FakeIndex(vecs), meta


def _store(index, meta):
    vs = VectorStore()
    vs.index, vs.metadata, vs.dim = index, meta, DIM
    return vs


def test_search_allowed_widens_past_excluded_neighbours():
    q, index, meta = _corpus()
    hits = search_allowed(index, meta, q, 5, lambda m: m["visibility"] == "PUBLIC")
    assert [meta[row]["text"] for _, row in hits] == [f"public-{i}" for i in range(5)]
    assert index.search_calls[0] == 5 and index.search_calls[-1] > 5  # started narrow, widened


def test_search_allowed_returns_fewer_when_index_has_fewer_allowed():
    q, index, meta = _corpus(n_public=2)
    hits = search_allowed(index, meta, q, 5, lambda m: m["visibility"] == "PUBLIC")
    assert len(hits) == 2
    assert index.search_calls[-1] == index.ntotal  # exhausted the whole index before giving up


def test_search_allowed_never_admits_rejected_chunks():
    q, index, meta = _corpus()
    hits = search_allowed(index, meta, q, 50, lambda m: m["visibility"] == "PUBLIC")
    assert all(meta[row]["visibility"] == "PUBLIC" for _, row in hits)


def test_search_allowed_empty_index_and_zero_want():
    _, index, meta = _corpus()
    assert search_allowed(index, meta, np.ones(DIM), 0, lambda m: True) == []
    assert search_allowed(FakeIndex(np.zeros((0, DIM))), [], np.ones(DIM), 5, lambda m: True) == []


def test_vector_store_search_public_not_starved_by_internal_majority():
    q, index, meta = _corpus()
    results = _store(index, meta).search(q, top_k=5, allowed_visibilities={"PUBLIC"})
    assert len(results) == 5
    assert {r["visibility"] for r in results} == {"PUBLIC"}


def test_vector_store_filter_search_public_scope_not_starved():
    q, index, meta = _corpus()
    results = _store(index, meta).filter_search(
        q, top_k=5, field="bundle", value="b", allowed_visibilities={"PUBLIC"}
    )
    assert len(results) == 5
    assert {r["visibility"] for r in results} == {"PUBLIC"}


def test_vector_store_search_without_allowed_set_is_unfiltered():
    q, index, meta = _corpus()
    results = _store(index, meta).search(q, top_k=5)
    assert {r["visibility"] for r in results} == {"INTERNAL"}


def test_engine_retrieve_default_public_not_starved_and_never_leaks():
    q, index, meta = _corpus()
    engine = RAGEngine(_store(index, meta))
    with patch("rag.engine.ollama_embed", return_value=q):
        results = engine.retrieve("anything", top_k=5)
    assert len(results) == 5
    assert {r["visibility"] for r in results} == {"PUBLIC"}


def test_engine_retrieve_internal_scope_only_when_explicitly_allowed():
    q, index, meta = _corpus()
    engine = RAGEngine(_store(index, meta))
    with patch("rag.engine.ollama_embed", return_value=q):
        results = engine.retrieve("anything", top_k=5, allowed_visibilities={"PUBLIC", "INTERNAL"})
    assert {r["visibility"] for r in results} == {"INTERNAL"}  # nearest are INTERNAL when permitted
