"""Unit tests for the in-memory PluginIndex: construction, adding documents, and search.

Developer: Manish Kumar <manish@omnibioai.org>
"""

from index.plugin_index import PluginIndex


def test_plugin_index():
    """Index plugin documents at construction and on add, return the matching document from search,
    and return nothing for a non-matching query."""
    pi = PluginIndex([{"text": "t1", "plugin": "p1"}])
    assert len(pi.docs) == 1
    
    pi.add([{"text": "t2", "plugin": "p2"}])
    assert len(pi.docs) == 2
    
    res = pi.search("t1")
    assert len(res) == 1
    assert res[0]["text"] == "t1"
    
    assert pi.search("none") == []
