"""Unit tests for Retriever with a mocked vector store and a mocked embedder.

Developer: Manish Kumar <manish@omnibioai.org>
"""

from unittest.mock import MagicMock, patch

import pytest

from retrieval.retriever import Retriever


@pytest.fixture
def mock_vs():
    """Provide a MagicMock standing in for the vector store."""
    return MagicMock()

def test_retriever(mock_vs):
    """Encode the query with the embedder and return the vector store's hits."""
    # Mock Embedder during init
    with patch("retrieval.retriever.Embedder") as mock_embedder_cls:
        mock_emb = MagicMock()
        mock_embedder_cls.return_value = mock_emb
        mock_emb.encode.return_value = [[0.1]*768]
        mock_vs.search.return_value = [{"text": "hit"}]
        
        r = Retriever(mock_vs)
        res = r.retrieve("query")
        
        assert res == [{"text": "hit"}]
        mock_emb.encode.assert_called_once_with("query")
