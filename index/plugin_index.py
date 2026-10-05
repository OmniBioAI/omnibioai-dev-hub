"""
OmniBioAI index.plugin_index.

Purpose:
    Defines PluginIndex with add and search methods for index.plugin_index.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

class PluginIndex:
    def __init__(self, plugin_docs=None):
        """
        plugin_docs format:
        [
            {"text": "...", "plugin": "name"},
            ...
        ]
        """
        self.docs = plugin_docs or []

    def add(self, docs):
        self.docs.extend(docs)

    def search(self, query: str):
        """
        Simple keyword match for plugin knowledge.
        Later upgrade:
        - plugin embeddings
        - metadata filtering
        - plugin capability routing
        """

        results = []
        query = query.lower()

        for doc in self.docs:
            text = doc.get("text", "")

            if query in text.lower():
                results.append({
                    "text": text,
                    "source": "plugin"
                })

        return results