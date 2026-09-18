"""Unit tests for load_documents with the filesystem mocked: markdown discovery, missing roots, the
work-subtree exclusion, and unreadable files.

Developer: Manish Kumar <manish@omnibioai.org>
"""

from unittest.mock import mock_open, patch

from ingestion.doc_loader import load_documents


def test_load_documents():
    """Load only markdown files, recording each document's text and source path."""
    fake_walk = [("/repo", [], ["README.md", "notes.txt"])]
    with patch("os.path.exists", return_value=True), \
         patch("os.walk", return_value=iter(fake_walk)), \
         patch("builtins.open", mock_open(read_data="some content")):
        docs = load_documents(["/repo"])
    assert len(docs) == 1
    assert docs[0]["text"] == "some content"
    assert docs[0]["source"] == "/repo/README.md"

def test_load_documents_not_found():
    """Return no documents when the given root does not exist."""
    with patch("os.path.exists", return_value=False):
        docs = load_documents(["/repo"])
        assert len(docs) == 0


def test_load_documents_skips_work_subtree():
    """Skip files under a work/ subtree so shadowed copies are not loaded."""
    # omnibioai/work/ holds UUID- and wftest_*/sweep_*-named runtime copies
    # of bundle READMEs that would otherwise shadow the canonical
    # omnibioai-workflow-bundles/ paths -- see doc_loader.py's module
    # docstring. Any .md under a "work" path segment must never be loaded.
    fake_walk = [
        ("/repo", [], ["README.md"]),
        ("/repo/work/run-1234", [], ["shadowed.md"]),
    ]
    with patch("os.path.exists", return_value=True), \
         patch("os.walk", return_value=iter(fake_walk)), \
         patch("builtins.open", mock_open(read_data="real content")):
        docs = load_documents(["/repo"])
    assert len(docs) == 1
    assert docs[0]["source"] == "/repo/README.md"


def test_load_documents_continues_past_unreadable_file():
    """Keep loading the remaining files when one file cannot be read."""
    fake_walk = [("/repo", [], ["bad.md", "good.md"])]
    call_count = {"n": 0}

    def _open(path, *args, **kwargs):
        call_count["n"] += 1
        if "bad.md" in path:
            raise OSError("permission denied")
        return mock_open(read_data="readable content").return_value

    with patch("os.path.exists", return_value=True), \
         patch("os.walk", return_value=iter(fake_walk)), \
         patch("builtins.open", side_effect=_open):
        docs = load_documents(["/repo"])

    assert len(docs) == 1
    assert docs[0]["source"] == "/repo/good.md"
