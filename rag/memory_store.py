"""
OmniBioAI rag.memory_store.

Purpose:
    Defines MemoryStoreV4 with add and get_context methods for rag.memory_store.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from collections import deque


class MemoryStoreV4:
    """
    lightweight working memory + conversation history
    """

    def __init__(self, max_len=20):
        self.memory = deque(maxlen=max_len)

    def add(self, role: str, text: str):
        self.memory.append({
            "role": role,
            "text": text
        })

    def get_context(self) -> str:
        return "\n".join(
            f"{m['role']}: {m['text']}" for m in self.memory
        )