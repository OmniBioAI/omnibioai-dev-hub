import pytest

from index.graph_store import GraphStore


@pytest.fixture
def gs():
    return GraphStore()

def test_add_edge(gs):
    gs.add_edge("A", "B", "rel")
    assert ("B", "rel") in gs.edges["A"]
    
    # Invalid
    gs.add_edge("", "B")
    assert "" not in gs.edges

def test_find_seed_nodes(gs):
    gs.add_edge("OmniBioAI", "Engine")
    gs.add_edge("RAG System", "Vector")
    
    seeds = gs._find_seed_nodes("omnibioai")
    assert "OmniBioAI" in seeds
    
    seeds = gs._find_seed_nodes("system")
    assert "RAG System" in seeds
    
    # "" in any string is True, so it finds all nodes
    assert len(gs._find_seed_nodes("")) == 2

def test_score_match(gs):
    score = gs._score_match("omni", "OmniBioAI", "Engine")
    assert score > 0
    
    score0 = gs._score_match("none", "A", "B")
    assert score0 == 0.0

def test_bfs_expand(gs):
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
    gs.add_edge("OmniBioAI", "Engine", "powers")
    
    res = gs.search("omni")
    assert len(res) > 0
    assert res[0]["node"] == "OmniBioAI"
    
    assert gs.search(None) == []

def test_size(gs):
    gs.add_edge("A", "B")
    stats = gs.size()
    assert stats["nodes"] == 1
    assert stats["edges"] == 1

def test_export(gs):
    gs.add_edge("A", "B", "rel")
    data = gs.export()
    assert "A" in data["nodes"]
    assert data["edges"][0]["from"] == "A"
    assert data["edges"][0]["to"] == "B"

def test_bfs_expand_visited_and_depth(gs):
    gs.add_edge("A", "B")
    # if A is already visited, it should skip
    results = gs._bfs_expand("A", "A", max_depth=2, visited={"A"})
    assert results == []
    
    # if depth > max_depth, it should skip
    results = gs._bfs_expand("A", "A", max_depth=-1, visited=set())
    assert results == []

def test_find_seed_nodes_overlap(gs):
    gs.add_edge("Bio Informatic", "Engine")
    # Query "Bio Engine" has "Bio" overlap with "Bio Informatic"
    # but "Bio Engine" is NOT a substring of "Bio Informatic"
    seeds = gs._find_seed_nodes("bio engine")
    assert "Bio Informatic" in seeds
