"""Unit tests for the in-memory GraphStore: edge insertion, seed matching, scoring, breadth-first
expansion, search, size, and export.

Developer: Manish Kumar <manish@omnibioai.org>
"""

import pytest

from index.graph_store import GraphStore


@pytest.fixture
def gs():
    """Provide a fresh, empty in-memory GraphStore."""
    return GraphStore()

def test_add_edge(gs):
    """Record a directed labeled edge and ignore an edge whose source node name is empty."""
    gs.add_edge("A", "B", "rel")
    assert ("B", "rel") in gs.edges["A"]
    
    # Invalid
    gs.add_edge("", "B")
    assert "" not in gs.edges

def test_find_seed_nodes(gs):
    """Match seed nodes by case-insensitive substring of the query and return every node for an
    empty query."""
    gs.add_edge("OmniBioAI", "Engine")
    gs.add_edge("RAG System", "Vector")
    
    seeds = gs._find_seed_nodes("omnibioai")
    assert "OmniBioAI" in seeds
    
    seeds = gs._find_seed_nodes("system")
    assert "RAG System" in seeds
    
    # "" in any string is True, so it finds all nodes
    assert len(gs._find_seed_nodes("")) == 2

def test_score_match(gs):
    """Score a matching query above zero and a non-matching query at exactly zero."""
    score = gs._score_match("omni", "OmniBioAI", "Engine")
    assert score > 0
    
    score0 = gs._score_match("none", "A", "B")
    assert score0 == 0.0

def test_bfs_expand(gs):
    """Expand breadth-first through a cyclic graph, reporting each traversed edge in order without
    revisiting nodes."""
    gs.add_edge("A", "B", "r1")
    gs.add_edge("B", "C", "r2")
    gs.add_edge("C", "A", "r3") # Cycle
    
    results = gs._bfs_expand("A", "A", max_depth=2, visited=set())
    # A -> B (depth 0), B -> C (depth 1), C -> A (depth 2)
    assert len(results) == 3
    assert results[0]["node"] == "A"
    assert results[0]["neighbor"] == "B"
    assert results[1]["node"] == "B"
    assert results[1]["neighbor"] == "C"
    assert results[2]["node"] == "C"
    assert results[2]["neighbor"] == "A"

def test_search(gs):
    """Return edges for nodes matching the query, matching node first, and nothing for a None query."""
    gs.add_edge("OmniBioAI", "Engine", "powers")
    
    res = gs.search("omni")
    assert len(res) > 0
    assert res[0]["node"] == "OmniBioAI"
    
    assert gs.search(None) == []

def test_size(gs):
    """Report the node and edge counts after one edge is added."""
    gs.add_edge("A", "B")
    stats = gs.size()
    assert stats["nodes"] == 1
    assert stats["edges"] == 1

def test_export(gs):
    """Export the graph as its nodes plus edges carrying their from and to endpoints."""
    gs.add_edge("A", "B", "rel")
    data = gs.export()
    assert "A" in data["nodes"]
    assert data["edges"][0]["from"] == "A"
    assert data["edges"][0]["to"] == "B"

def test_bfs_expand_visited_and_depth(gs):
    """Return no results when the starting node is already visited or the depth already exceeds
    max_depth."""
    gs.add_edge("A", "B")
    # if A is already visited, it should skip
    results = gs._bfs_expand("A", "A", max_depth=2, visited={"A"})
    assert results == []
    
    # if depth > max_depth, it should skip
    results = gs._bfs_expand("A", "A", max_depth=-1, visited=set())
    assert results == []

def test_find_seed_nodes_overlap(gs):
    """Match a node through word overlap with the query even when the whole query is not a substring
    of the node name."""
    gs.add_edge("Bio Informatic", "Engine")
    # Query "Bio Engine" has "Bio" overlap with "Bio Informatic"
    # but "Bio Engine" is NOT a substring of "Bio Informatic"
    seeds = gs._find_seed_nodes("bio engine")
    assert "Bio Informatic" in seeds
