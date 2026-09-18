"""Unit tests for MemoryStoreV4's bounded conversation memory.

Developer: Manish Kumar <manish@omnibioai.org>
"""

from rag.memory_store import MemoryStoreV4


def test_memory_store():
    """Evict the oldest turn once max_len is exceeded and render only the remaining turns in the
    context."""
    ms = MemoryStoreV4(max_len=2)
    ms.add("user", "hi")
    ms.add("assistant", "hello")
    ms.add("user", "bye") # Should eject "hi"
    
    assert len(ms.memory) == 2
    ctx = ms.get_context()
    assert "assistant: hello" in ctx
    assert "user: bye" in ctx
    assert "user: hi" not in ctx
