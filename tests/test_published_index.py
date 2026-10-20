"""Tests for the published-on-the-docs-site check and the published-subset derivation."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from index.published import (
    documents,
    load_sitemap_routes,
    parse_sitemap,
    route_map_from_checkout,
    unpublished_documents,
)

ROOT = Path(__file__).resolve().parents[1]

SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.example.org/</loc></url>
  <url><loc>https://docs.example.org/user/start/</loc></url>
  <url><loc> https://docs.example.org/tools </loc></url>
</urlset>
"""


def _meta(path, route=None, repo="omnibioai-docs"):
    return {"repo": repo, "relative_path": path, **({"route": route} if route else {})}


def test_parse_sitemap_keeps_normalized_paths_only():
    assert parse_sitemap(SITEMAP) == {"/", "/user/start/", "/tools/"}


def test_load_sitemap_from_file_and_refuse_an_empty_one(tmp_path):
    good, empty = tmp_path / "s.xml", tmp_path / "e.xml"
    good.write_text(SITEMAP)
    empty.write_text("<urlset></urlset>")
    assert "/tools/" in load_sitemap_routes(str(good))
    with pytest.raises(ValueError, match="lists no routes"):
        load_sitemap_routes(str(empty))


def test_unpublished_documents_reports_missing_and_unknown_routes():
    routes = parse_sitemap(SITEMAP)
    meta = [_meta("site/docs/start.md", "/user/start"), _meta("site/docs/start.md", "/user/start/"),
            _meta("site/docs/admin/ops.md", "/admin/ops/"), _meta("COMPAT.md"), _meta("README.md", repo="omnibioai-tes")]
    bad = unpublished_documents(meta, routes)
    assert [(d["repo"], d["relative_path"], d["reason"]) for d in bad] == [
        ("omnibioai-docs", "COMPAT.md", "no-route"),
        ("omnibioai-docs", "site/docs/admin/ops.md", "not-in-sitemap"),
        ("omnibioai-tes", "README.md", "no-route"),
    ]
    assert documents(meta)[("omnibioai-docs", "site/docs/start.md")]["chunks"] == 2


def test_route_map_from_a_checkout_fills_routes_for_older_indexes(tmp_path):
    (tmp_path / "site/docs").mkdir(parents=True)
    (tmp_path / "site/docs/t.md").write_text("---\nslug: /tools\n---\n# Tools")
    (tmp_path / "site/docs/n.md").write_text("# no slug")
    route_map = route_map_from_checkout(tmp_path, {"site/docs/t.md", "site/docs/n.md", "site/docs/gone.md"})
    assert route_map == {"site/docs/t.md": "/tools/"}
    meta = [_meta("site/docs/t.md"), _meta("site/docs/n.md")]
    assert [d["relative_path"] for d in unpublished_documents(meta, parse_sitemap(SITEMAP), route_map)] == ["site/docs/n.md"]


# Real FAISS runs in a subprocess: other test modules replace `faiss` with a mock in-process.
HELPER = textwrap.dedent('''
    import json, sys, numpy as np
    sys.path.insert(0, {root!r})
    from index.vector_store import VectorStore
    from index.lifecycle import artifact_hashes, write_manifest, read_manifest, validate_index_directory
    from index.derive import derive_published_subset
    from index.published import parse_sitemap
    work, mode = sys.argv[1], sys.argv[2]
    docs = [("site/docs/start.md", "/user/start/"), ("site/docs/start.md", "/user/start/"), ("site/docs/admin/ops.md", "/admin/ops/"),
            ("COMPAT.md", None), ("site/docs/tools.md", "/tools/")]
    if mode == "nothing_published": docs = [("COMPAT.md", None)]
    md = [{{"text": f"t{{i}}", "source": f"omnibioai-docs:{{p}}@rev", "repo": "omnibioai-docs", "relative_path": p, "route": r,
           "source_revision": "rev", "document_id": f"doc-{{p}}", "chunk_id": f"c{{i}}", "content_hash": "h", "chunk_hash": "ch",
           "content_state": "CURRENT", "verification_state": "CONFIGURED", "bundle": "site", "citation": {{"repository": "omnibioai-docs"}},
           "visibility": "PUBLIC"}} for i, (p, r) in enumerate(docs)]
    vecs = np.random.default_rng(3).normal(size=(len(md), 768)).astype("float32")
    src = work + "/src"
    vs = VectorStore(); vs.add(vecs, md); vs.save(src)
    manifest = {{"build_id": "pub", "build_status": "CLEAN", "artifact_hashes": artifact_hashes(src),
                "visibility_counts": {{"PUBLIC": len(md), "INTERNAL": 0, "REVIEW_REQUIRED": 0}}, "promotion_status": "PROMOTED"}}
    if mode != "not_public_only": manifest["visibility_policy"] = "PUBLIC_ONLY"
    write_manifest(src, manifest)
    routes = parse_sitemap(open(sys.argv[3]).read()) if mode != "no_routes" else set()
    out = {{}}
    try:
        rep = derive_published_subset(src, work + "/out", routes, sitemap_source="test-sitemap")
        d = rep["derived_dir"]; m = read_manifest(d); dv = VectorStore(); dv.load(d)
        kept = [i for i, x in enumerate(md) if x["relative_path"] in {{y["relative_path"] for y in dv.metadata}}]
        out = {{"ok": True, "report": rep, "manifest": m, "paths": [x["relative_path"] for x in dv.metadata],
               "bit_identical": bool(np.array_equal(dv.index.reconstruct_n(0, dv.index.ntotal), vecs[kept])),
               "valid": validate_index_directory(d)["ok"]}}
    except Exception as e:
        out = {{"ok": False, "error": str(e)}}
    print(json.dumps(out))
''')


def _derive(tmp_path, mode="ok"):
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(SITEMAP)
    r = subprocess.run([sys.executable, "-c", HELPER.format(root=str(ROOT)), str(tmp_path), mode, str(sitemap)],
                       capture_output=True, text=True, timeout=120, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_derive_published_subset_keeps_only_published_documents_bit_for_bit(tmp_path):
    out = _derive(tmp_path)
    assert out["ok"], out
    assert sorted(set(out["paths"])) == ["site/docs/start.md", "site/docs/tools.md"]
    assert out["bit_identical"] and out["valid"]
    m = out["manifest"]
    assert m["visibility_policy"] == "PUBLIC_ONLY" and m["promotion_status"] == "CANDIDATE"
    assert m["selected_chunk_count"] == 3 and m["derived_from"]["chunk_count"] == 5
    assert m["published_check"]["sitemap"] == "test-sitemap"
    excluded = {d["relative_path"]: (d["reason"], d["chunks"]) for d in m["excluded_not_published"]["documents"]}
    assert excluded == {"COMPAT.md": ("no-route", 1), "site/docs/admin/ops.md": ("not-in-sitemap", 1)}
    assert (tmp_path / "out" / "pub-published.derivation-report.json").exists()
    assert not list((tmp_path / "out").glob(".*partial"))


@pytest.mark.parametrize("mode, error", [
    ("not_public_only", "not PUBLIC_ONLY"),
    ("nothing_published", "nothing to derive"),
    ("no_routes", "no sitemap routes"),
])
def test_derive_published_subset_refuses(tmp_path, mode, error):
    out = _derive(tmp_path, mode)
    assert not out["ok"] and error in out["error"]
    assert not (tmp_path / "out" / "pub-published").exists()
