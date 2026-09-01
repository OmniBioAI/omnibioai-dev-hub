from unittest.mock import mock_open, patch

from ingestion.doc_loader import load_documents


def test_load_documents():
    fake_walk = [("/repo", [], ["README.md", "notes.txt"])]
    with patch("os.path.exists", return_value=True), \
         patch("os.walk", return_value=iter(fake_walk)), \
         patch("builtins.open", mock_open(read_data="some content")):
        docs = load_documents(["/repo"])
    assert len(docs) == 1
    assert docs[0]["text"] == "some content"
    assert docs[0]["source"] == "/repo/README.md"

def test_load_documents_not_found():
    with patch("os.path.exists", return_value=False):
        docs = load_documents(["/repo"])
        assert len(docs) == 0
