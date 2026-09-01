from unittest.mock import MagicMock, patch

import pytest

from retrieval.retriever import Retriever


@pytest.fixture
def mock_vs():
    return MagicMock()

def test_retriever(mock_vs):
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
