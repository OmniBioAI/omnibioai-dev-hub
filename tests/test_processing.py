"""Unit tests for chunk_text's plain-text and chunk_size-argument behavior.

Developer: Manish Kumar <manish@omnibioai.org>
"""

from processing.chunker import chunk_text


def test_chunk_text_plain():
    """Return plain text without headers as a single unchanged chunk."""
    # Plain text with no headers is returned as a single chunk
    text = "hello world foo bar"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_accepts_chunk_size_kwarg():
    """Accept a chunk_size keyword on the processing package's chunk_text and return a single chunk
    for short text."""
    # chunk_size is kept for API compatibility and must not raise
    result = chunk_text("some text", chunk_size=2)
    assert isinstance(result, list)
    assert len(result) == 1
