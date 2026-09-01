import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock sentence_transformers
mock_st = MagicMock()
mock_st.SentenceTransformer = MagicMock()
sys.modules['sentence_transformers'] = mock_st

from embeddings.embedder import Embedder


@pytest.fixture
def embedder():
    with patch("embeddings.embedder.SentenceTransformer") as mock_model:
        model_instance = MagicMock()
        mock_model.return_value = model_instance
        yield Embedder()

def test_encode_single(embedder):
    embedder.model.encode.return_value = np.array([[0.1, 0.2]])
    res = embedder.encode_single("text")
    assert len(res) == 2
    assert isinstance(res, list)

def test_encode_empty(embedder):
    assert embedder.encode([]) == []

def test_normalize_zero_norm(embedder):
    # Test division by zero protection
    vecs = np.array([[0.0, 0.0]])
    normed = embedder._normalize(vecs)
    assert np.all(normed == 0.0)

def test_encode_string(embedder):
    embedder.model.encode.return_value = np.array([[0.1, 0.2]])
    # Passing string should hit line 34
    res = embedder.encode("text")
    assert len(res) == 1
