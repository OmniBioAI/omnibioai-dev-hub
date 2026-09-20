"""Tests for deriving the PUBLIC-only production artifact (real FAISS, run in a subprocess)."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HELPER = textwrap.dedent('''
    import json, sys, pickle, numpy as np
    sys.path.insert(0, {root!r})
    from index.vector_store import VectorStore
    from index.lifecycle import artifact_hashes, write_manifest, read_manifest, validate_index_directory
    from index.derive import derive_public_only
    import faiss
    work, mode = sys.argv[1], sys.argv[2]
    vis = ["PUBLIC", "INTERNAL", "PUBLIC", "REVIEW_REQUIRED", "INTERNAL", "PUBLIC"]
    if mode == "unknown_visibility": vis[1] = "SECRET"
    if mode == "no_public": vis = ["INTERNAL", "INTERNAL"]
    if mode == "already_public": vis = ["PUBLIC", "PUBLIC"]
    def meta(i, v):
        return {{"text": f"t{{i}}", "source": f"r:{{i}}.md@rev", "repo": "r" if v == "PUBLIC" else "priv", "relative_path": f"{{i}}.md",
                "source_revision": f"rev{{i}}", "document_id": f"doc{{i}}", "chunk_id": f"c{{i}}", "content_hash": "h", "chunk_hash": "ch",
                "content_state": "CURRENT", "verification_state": "CONFIGURED", "bundle": f"b{{i}}", "citation": {{"repository": "r"}}, "visibility": v}}
    md = [meta(i, v) for i, v in enumerate(vis)]
    vecs = np.random.default_rng(1).normal(size=(len(md), 768)).astype("float32") * 17
    src = work + "/src"
    vs = VectorStore(); vs.add(vecs, md); vs.save(src)
    counts = {{k: vis.count(k) for k in ("PUBLIC", "INTERNAL", "REVIEW_REQUIRED")}}
    status = "EMBEDDING_FAILURES" if mode == "not_clean" else "CLEAN"
    write_manifest(src, {{"build_id": "srcbuild", "build_status": status, "artifact_hashes": artifact_hashes(src), "visibility_counts": counts,
                        "embedding_model": "nomic-embed-text", "embedding_dimension": 768, "metadata_schema_version": "v1"}})
    if mode == "already_public":
        m = read_manifest(src); m["visibility_policy"] = "PUBLIC_ONLY"; write_manifest(src, m)
    out = {{}}
    try:
        rep = derive_public_only(src, work + "/out")
        d = rep["derived_dir"]
        m = read_manifest(d); dv = VectorStore(); dv.load(d)
        rows = [i for i, v in enumerate(vis) if v == "PUBLIC"]
        out = {{"ok": True, "manifest": m, "ntotal": dv.index.ntotal, "visibilities": sorted({{x["visibility"] for x in dv.metadata}}),
               "chunk_ids": [x["chunk_id"] for x in dv.metadata], "bit_identical": bool(np.array_equal(dv.index.reconstruct_n(0, dv.index.ntotal), vecs[rows])),
               "validation_ok": validate_index_directory(d)["ok"], "report": rep,
               "pickle_text": open(d + "/metadata.pkl", "rb").read().count(b"priv"), "bundle_kept": [x["bundle"] for x in dv.metadata]}}
    except Exception as e:
        out = {{"ok": False, "error": str(e)}}
    print("RESULT" + json.dumps(out))
''')


def _run(tmp_path, mode="normal"):
    r = subprocess.run([sys.executable, "-c", HELPER.format(root=str(ROOT)), str(tmp_path), mode],
                       capture_output=True, text=True, timeout=180, check=True)
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


def test_derived_artifact_contains_only_public_and_no_internal_identity(tmp_path):
    out = _run(tmp_path)
    assert out["ok"], out
    assert out["ntotal"] == 3 and out["visibilities"] == ["PUBLIC"]
    assert out["chunk_ids"] == ["c0", "c2", "c5"]  # order preserved; c1,c3,c4 (INTERNAL/REVIEW_REQUIRED) absent
    assert out["pickle_text"] == 0  # not even the excluded documents' repo name is in the artifact bytes


def test_vectors_are_copied_bit_for_bit_and_metadata_preserved(tmp_path):
    out = _run(tmp_path)
    assert out["bit_identical"] and out["validation_ok"]
    assert out["bundle_kept"] == ["b0", "b2", "b5"]


def test_manifest_is_self_describing_and_traces_back_to_the_source(tmp_path):
    m = _run(tmp_path)["manifest"]
    assert m["visibility_policy"] == "PUBLIC_ONLY" and m["derived"] is True and m["build_status"] == "CLEAN"
    assert m["derived_from"]["build_id"] == "srcbuild" and m["derived_from"]["chunk_count"] == 6
    assert m["visibility_counts"] == {"PUBLIC": 3, "INTERNAL": 0, "REVIEW_REQUIRED": 0}
    assert m["excluded_by_policy"] == {"INTERNAL": {"documents": 2, "chunks": 2}, "REVIEW_REQUIRED": {"documents": 1, "chunks": 1}}
    assert m["promotion_status"] == "CANDIDATE" and m["evaluation_status"] == "NOT_EVALUATED"
    assert "not re-embedded" in m["derivation"] and set(m["artifact_hashes"]) == {"index.faiss", "metadata.pkl"}
    assert "priv" not in json.dumps(m)  # manifest never names excluded repositories/paths


def test_report_lists_excluded_documents_outside_the_artifact(tmp_path):
    rep = _run(tmp_path)["report"]
    assert rep["excluded"]["INTERNAL"]["documents"] == 2 and rep["vectors_bit_identical"]
    assert {d["relative_path"] for d in rep["excluded_documents"]["INTERNAL"]} == {"1.md", "4.md"}
    assert (tmp_path / "out" / "srcbuild-public.derivation-report.json").exists()


def test_refuses_source_with_unknown_visibility_fail_closed(tmp_path):
    out = _run(tmp_path, "unknown_visibility")
    assert not out["ok"] and "invalid visibility" in out["error"]
    assert not (tmp_path / "out" / "srcbuild-public").exists()


def test_refuses_a_source_that_is_not_clean(tmp_path):
    out = _run(tmp_path, "not_clean")
    assert not out["ok"] and "EMBEDDING_FAILURES" in out["error"]


def test_refuses_when_nothing_is_public(tmp_path):
    out = _run(tmp_path, "no_public")
    assert not out["ok"] and "no PUBLIC chunks" in out["error"]


def test_refuses_to_derive_from_an_already_public_only_source(tmp_path):
    out = _run(tmp_path, "already_public")
    assert not out["ok"] and "already PUBLIC_ONLY" in out["error"]


def test_a_public_only_manifest_over_a_source_containing_internal_chunks_is_refused_by_validation(tmp_path):
    # Not derivable, and not servable: the mixed source fails validation before anything is copied.
    out = _run(tmp_path, "unknown_visibility")
    assert not out["ok"]
