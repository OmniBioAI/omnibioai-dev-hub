from embeddings.embedder import Embedder


class Retriever:
    def __init__(self, vector_store):
        self.embedder = Embedder()
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5):
        query_vec = self.embedder.encode(query)[0]
        results = self.vector_store.search(query_vec, top_k)
        return results