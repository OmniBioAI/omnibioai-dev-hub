"""Unit tests for VectorStore with faiss mocked out: index initialization, dimension checks, add and
search, persistence, and metadata-filtered search.

Developer: Manish Kumar <manish@omnibioai.org>
"""

import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# Mock faiss
mock_faiss = MagicMock()
mock_faiss.IndexFlatIP = MagicMock()
sys.modules['faiss'] = mock_faiss

from index.vector_store import VectorStore


@pytest.fixture
def vs():
    """Provide a fresh VectorStore backed by the mocked faiss module."""
    return VectorStore()

def test_init_index_success(vs):
    """Create a flat inner-product index of the supported 768 dimensions."""
    vs._init_index(768)
    assert vs.dim == 768
    mock_faiss.IndexFlatIP.assert_called_with(768)

def test_init_index_failure(vs):
    """Reject any embedding dimension other than the supported nomic-embed-text size."""
    with pytest.raises(ValueError, match="Only nomic-embed-text"):
        vs._init_index(512)

def test_coerce_shape(vs):
    """Coerce supported vector shapes into a (1, 768) array and reject shapes that cannot be
    coerced."""
    # 1D
    v1 = [0.1] * 768
    c1 = vs._coerce_shape(v1)
    assert c1.shape == (1, 768)
    
    # 2D
    v2 = [[0.1] * 768]
    c2 = vs._coerce_shape(v2)
    assert c2.shape == (1, 768)
    
    # 3D
    v3 = [[[0.1] * 768]]
    c3 = vs._coerce_shape(v3)
    assert c3.shape == (1, 768)
    
    # Invalid
    with pytest.raises(ValueError, match="Cannot coerce"):
        vs._coerce_shape(np.zeros((1, 1, 1, 768)))

def test_add_success(vs):
    """Add embeddings to the index, set the index dimension, and store their metadata."""
    mock_index = MagicMock()
    mock_faiss.IndexFlatIP.return_value = mock_index
    
    vs.add([[0.1] * 768], [{"text": "t1"}])
    assert vs.dim == 768
    mock_index.add.assert_called_once()
    assert len(vs.metadata) == 1

def test_add_dim_mismatch(vs):
    """Reject embeddings whose dimension differs from the index dimension."""
    vs.dim = 768
    vs.index = MagicMock()
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        vs.add([[0.1] * 512], [])

def test_search_empty(vs):
    """Return no results when the store holds no index."""
    assert vs.search([0.1]*768) == []

def test_search_success(vs):
    """Return the metadata of the nearest indexed vectors as search results."""
    vs.dim = 768
    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
    vs.index = mock_index
    vs.metadata = [{"text": "t1", "source": "s1"}]
    
    res = vs.search([0.1]*768)
    assert len(res) == 1
    assert res[0]["text"] == "t1"

def test_search_dim_mismatch(vs):
    """Reject a query whose dimension differs from the index dimension."""
    vs.dim = 768
    vs.index = MagicMock()
    vs.index.ntotal = 1
    with pytest.raises(ValueError, match="Query dimension mismatch"):
        vs.search([0.1]*512)

def test_search_invalid_indices(vs):
    """Return no results when the index reports indices that have no metadata."""
    vs.dim = 768
    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.search.return_value = (np.array([[0.9]]), np.array([[-1]]))
    vs.index = mock_index
    vs.metadata = []

    assert vs.search([0.1]*768) == []


# ------------------------------------------------------------------
# save / load
# ------------------------------------------------------------------

def test_save_raises_when_empty(vs):
    """Refuse to save an empty index."""
    with pytest.raises(RuntimeError, match="index is empty"):
        vs.save("/tmp/vs_test")


def test_save_success(vs):
    """Create the target directory, write the index, and dump the metadata when saving."""
    mock_index = MagicMock()
    mock_index.ntotal = 2
    vs.index = mock_index
    vs.dim = 768
    vs.metadata = [{"text": "a"}, {"text": "b"}]

    with patch("os.makedirs") as mock_mkdirs, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("pickle.dump") as mock_pdump:
        mock_faiss.write_index = MagicMock()
        vs.save("/tmp/vs_test")

    mock_mkdirs.assert_called_once_with("/tmp/vs_test", exist_ok=True)
    mock_faiss.write_index.assert_called_once()
    mock_file.assert_called_once()
    mock_pdump.assert_called_once()


def test_load_returns_false_when_missing(vs):
    """Return False when the saved index directory does not exist."""
    with patch("os.path.exists", return_value=False):
        result = vs.load("/tmp/no_such_dir")
    assert result is False


def test_load_success(vs):
    """Restore the index, metadata, and dimension from saved data and return True."""
    saved_data = {"metadata": [{"text": "x", "source": "s1"}], "dim": 768}
    mock_loaded_index = MagicMock()
    mock_loaded_index.ntotal = 1

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open()), \
         patch("pickle.load", return_value=saved_data) as mock_pload:
        mock_faiss.read_index = MagicMock(return_value=mock_loaded_index)
        result = vs.load("/tmp/vs_test")

    mock_pload.assert_called_once()
    assert result is True
    assert vs.dim == 768
    assert vs.metadata == saved_data["metadata"]
    assert vs.index is mock_loaded_index


# ------------------------------------------------------------------
# filter_search
# ------------------------------------------------------------------

def _vs_with_metadata(metadata):
    """Return a VectorStore wired with a mock index and given metadata."""
    v = VectorStore()
    v.dim = 768
    mock_idx = MagicMock()
    mock_idx.ntotal = len(metadata)
    # Return scores/indices covering all rows
    scores = np.array([[1.0 - i * 0.1 for i in range(len(metadata))]])
    indices = np.array([[i for i in range(len(metadata))]])
    mock_idx.search.return_value = (scores, indices)
    v.index = mock_idx
    v.metadata = metadata
    return v


def test_filter_search_no_filter_delegates_to_search(vs):
    """Delegate to plain search when no filter is given."""
    vs.dim = 768
    mock_idx = MagicMock()
    mock_idx.ntotal = 1
    mock_idx.search.return_value = (np.array([[0.9]]), np.array([[0]]))
    vs.index = mock_idx
    vs.metadata = [{"text": "t1", "source": "s1"}]

    # No field/value → behaves like search()
    res = vs.filter_search([0.1] * 768)
    assert len(res) == 1
    assert res[0]["text"] == "t1"


def test_filter_search_empty_index(vs):
    """Return no results from a filtered search over an empty index."""
    assert vs.filter_search([0.1] * 768, field="bundle", value="x") == []


def test_filter_search_by_bundle(vs):
    """Return only the entries whose bundle matches the filter value."""
    meta = [
        {"text": "t1", "source": "s1", "repo": "repo-a", "bundle": "bx"},
        {"text": "t2", "source": "s2", "repo": "repo-a", "bundle": "by"},
        {"text": "t3", "source": "s3", "repo": "repo-b", "bundle": "bx"},
    ]
    v = _vs_with_metadata(meta)
    res = v.filter_search([0.1] * 768, top_k=5, field="bundle", value="bx")
    assert len(res) == 2
    texts = [r["text"] for r in res]
    assert "t1" in texts
    assert "t3" in texts
    assert "t2" not in texts


def test_filter_search_by_repo(vs):
    """Return only the entries whose repo matches the filter value."""
    meta = [
        {"text": "t1", "source": "s1", "repo": "repo-a", "bundle": "bx"},
        {"text": "t2", "source": "s2", "repo": "repo-b", "bundle": "bx"},
    ]
    v = _vs_with_metadata(meta)
    res = v.filter_search([0.1] * 768, top_k=5, field="repo", value="repo-a")
    assert len(res) == 1
    assert res[0]["text"] == "t1"


def test_filter_search_respects_top_k(vs):
    """Cap filtered results at top_k."""
    meta = [{"text": f"t{i}", "source": f"s{i}", "repo": "r", "bundle": "b"} for i in range(10)]
    v = _vs_with_metadata(meta)
    res = v.filter_search([0.1] * 768, top_k=3, field="bundle", value="b")
    assert len(res) == 3


def test_filter_search_no_matches_returns_empty(vs):
    """Return an empty list when no entry matches the filter value."""
    meta = [{"text": "t1", "source": "s1", "repo": "repo-a", "bundle": "bx"}]
    v = _vs_with_metadata(meta)
    res = v.filter_search([0.1] * 768, top_k=5, field="bundle", value="nonexistent")
    assert res == []


def test_filter_search_returns_repo_and_bundle_fields(vs):
    """Include the repo and bundle fields in filtered results."""
    meta = [{"text": "t1", "source": "s1", "repo": "my-repo", "bundle": "my-bundle"}]
    v = _vs_with_metadata(meta)
    res = v.filter_search([0.1] * 768, top_k=1, field="bundle", value="my-bundle")
    assert len(res) == 1
    assert res[0]["repo"] == "my-repo"
    assert res[0]["bundle"] == "my-bundle"


def test_filter_search_dim_mismatch_raises(vs):
    """Reject a filtered query whose dimension differs from the index dimension."""
    vs.dim = 768
    vs.index = MagicMock()
    vs.index.ntotal = 1
    with pytest.raises(ValueError, match="Query dimension mismatch"):
        vs.filter_search([0.1] * 512, field="bundle", value="x")


def test_filter_search_skips_invalid_indices(vs):
    """Skip out-of-range indices returned by the index while keeping the valid hit."""
    meta = [{"text": "t1", "source": "s1", "bundle": "bx"}]
    v = _vs_with_metadata(meta)
    # Inject an out-of-range index into the mock search result
    v.index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 99]]))
    res = v.filter_search([0.1] * 768, top_k=5, field="bundle", value="bx")
    assert len(res) == 1  # idx 99 is skipped
