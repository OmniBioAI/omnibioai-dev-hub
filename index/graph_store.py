from collections import defaultdict, deque
from typing import Any

# =========================================================
# GRAPH STORE V4 - OMNIBIOAI
# RAG-ready semantic graph + expansion engine
# =========================================================

class GraphStore:
    """
    Lightweight knowledge graph for RAG augmentation.

    V4 upgrades:
    - bidirectional-ready design
    - BFS expansion with depth control
    - relevance scoring
    - cycle protection
    - UI-friendly structured output
    """

    def __init__(self):
        # adjacency list: node -> [(neighbor, relation)]
        self.edges: dict[str, list[tuple[str, str]]] = defaultdict(list)

    # =========================================================
    # ADD EDGE
    # =========================================================
    def add_edge(self, a: str, b: str, relation: str = "related"):
        """
        Add directed edge a → b
        """

        if not a or not b:
            return

        self.edges[a].append((b, relation))

    # =========================================================
    # MAIN SEARCH ENTRY
    # =========================================================
    def search(self, query: str, max_depth: int = 2) -> list[dict[str, Any]]:
        """
        Semantic graph expansion search (V4)

        Returns:
            List of graph paths with scores (UI-ready)
        """

        query = (query or "").lower().strip()

        if not query:
            return []

        visited: set[str] = set()
        results: list[dict[str, Any]] = []

        seed_nodes = self._find_seed_nodes(query)

        for seed in seed_nodes:
            results.extend(
                self._bfs_expand(seed, query, max_depth, visited)
            )

        # final ranking
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return results[:30]  # UI-safe cap

    # =========================================================
    # SEED NODE DETECTION
    # =========================================================
    def _find_seed_nodes(self, query: str) -> list[str]:

        seeds = []

        q_tokens = set(query.split())

        for node in self.edges:

            node_l = node.lower()
            node_tokens = set(node_l.split())

            # strong match
            if query in node_l:
                seeds.append(node)
                continue

            # token overlap match
            if q_tokens & node_tokens:
                seeds.append(node)

        return seeds[:5]

    # =========================================================
    # BFS EXPANSION ENGINE
    # =========================================================
    def _bfs_expand(
        self,
        start: str,
        query: str,
        max_depth: int,
        visited: set[str]
    ) -> list[dict[str, Any]]:

        queue = deque([(start, 0)])
        results = []

        while queue:

            node, depth = queue.popleft()

            if node in visited or depth > max_depth:
                continue

            visited.add(node)

            neighbors = self.edges.get(node, [])

            for neighbor, rel in neighbors:

                score = self._score_match(query, node, neighbor)

                results.append({
                    "text": f"{node} → {neighbor} ({rel})",
                    "source": "graph",
                    "score": score,
                    "depth": depth,
                    "node": node,
                    "neighbor": neighbor,
                    "relation": rel
                })

                queue.append((neighbor, depth + 1))

        return results

    # =========================================================
    # SCORING ENGINE
    # =========================================================
    def _score_match(self, query: str, a: str, b: str) -> float:

        q_tokens = set(query.split())
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())

        overlap_score = len(q_tokens & a_tokens) + len(q_tokens & b_tokens)

        direct_match = 1.0 if query in a.lower() else 0.0

        return float(overlap_score * 0.5 + direct_match * 1.5)

    # =========================================================
    # GRAPH STATS (FOR UI)
    # =========================================================
    def size(self) -> dict[str, int]:

        node_count = len(self.edges)
        edge_count = sum(len(v) for v in self.edges.values())

        return {
            "nodes": node_count,
            "edges": edge_count
        }

    # =========================================================
    # DEBUG EXPORT (UI GRAPH VISUALIZATION READY)
    # =========================================================
    def export(self) -> dict[str, Any]:

        nodes = list(self.edges.keys())
        edges = []

        for src, neighbors in self.edges.items():
            for dst, rel in neighbors:
                edges.append({
                    "from": src,
                    "to": dst,
                    "relation": rel
                })

        return {
            "nodes": nodes,
            "edges": edges
        }