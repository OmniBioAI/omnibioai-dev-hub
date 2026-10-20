"""Tests for the operator-facing promote/rollback command (real FAISS, subprocess)."""

import json
import subprocess
import sys
from pathlib import Path

from tests.test_startup_safety import _make_index

ROOT = Path(__file__).resolve().parents[1]
CLI = str(ROOT / "scripts" / "promote_index.py")


def _cli(*args, skip_published_check=True):
    # Tests not about the published check opt out of it explicitly, as an operator would have to.
    if skip_published_check and args[0] == "promote" and "--sitemap" not in args:
        args = (*args, "--skip-published-check")
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True, timeout=120, check=False)


def _hashes(d: Path):
    import hashlib
    return {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("index.faiss", "metadata.pkl")}


def _legacy(data: Path):
    live = data / "faiss_index"
    _make_index(live, "PUBLIC,PUBLIC")
    (live / "manifest.json").unlink()  # pre-Phase-18 index: no manifest
    return live


def test_promote_then_rollback_restores_the_legacy_index_byte_for_byte(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    legacy_hashes = _hashes(live)
    _make_index(src, "PUBLIC,PUBLIC,PUBLIC")

    r = _cli("promote", "--source", str(src), "--data-dir", str(data))
    assert r.returncode == 0, r.stderr
    assert "PROMOTED" in r.stdout and "hashes verified" in r.stdout
    assert _hashes(live) == _hashes(src)
    assert json.loads((live / "manifest.json").read_text())["promotion_status"] == "PROMOTED"
    prev = list((data / "faiss_previous").iterdir())
    assert len(prev) == 1 and _hashes(prev[0]) == legacy_hashes  # previous retained

    r = _cli("rollback", "--data-dir", str(data))
    assert r.returncode == 0, r.stderr
    assert _hashes(live) == legacy_hashes
    assert list((data / "faiss_rolled").iterdir())  # the promoted index is retained too


def test_promote_refuses_a_mixed_visibility_candidate_and_changes_nothing(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    before = _hashes(live)
    _make_index(src, "PUBLIC,INTERNAL", policy="mixed")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data))
    assert r.returncode == 1 and "REFUSED" in r.stderr
    assert _hashes(live) == before and not (data / "faiss_candidates").exists()


def test_promote_refuses_a_rejected_candidate(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    before = _hashes(live)
    _make_index(src, "PUBLIC")
    (src / "REJECTED").write_text("never\n")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data))
    assert r.returncode == 1 and "REJECTED" in r.stderr and _hashes(live) == before


def test_promote_refuses_when_the_candidate_was_already_staged(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    before = _hashes(live)
    _make_index(src, "PUBLIC")
    (data / "faiss_candidates" / "b").mkdir(parents=True)  # manifest build_id from the helper is "b"
    r = _cli("promote", "--source", str(src), "--data-dir", str(data))
    assert r.returncode == 1 and "already exists" in r.stderr and _hashes(live) == before


def test_first_activation_accepts_the_canonical_staged_candidate(tmp_path):
    data = tmp_path / "data"
    src = data / "faiss_candidates" / "b"
    _make_index(src, "PUBLIC,PUBLIC")

    r = _cli("promote", "--source", str(src), "--data-dir", str(data))

    assert r.returncode == 0, r.stderr
    assert "using staged b; hashes verified" in r.stdout
    assert not src.exists()
    assert _hashes(data / "faiss_index")
    assert json.loads((data / "faiss_index" / "manifest.json").read_text())["promotion_status"] == "PROMOTED"


def test_stale_interrupted_partial_does_not_block_a_clean_retry(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    _make_index(src, "PUBLIC")
    stale = data / "faiss_candidates" / ".b.partial-abandoned"
    stale.mkdir(parents=True)
    (stale / "incomplete").write_text("partial copy\n")

    r = _cli("promote", "--source", str(src), "--data-dir", str(data))

    assert r.returncode == 0, r.stderr
    assert (data / "faiss_index" / "index.faiss").exists()
    assert stale.exists()  # unknown crash debris is preserved for operator review, not deleted implicitly


def test_promote_refuses_a_corrupted_candidate_and_leaves_active_untouched(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    before = _hashes(live)
    _make_index(src, "PUBLIC")
    (src / "index.faiss").write_bytes(b"corrupted")

    r = _cli("promote", "--source", str(src), "--data-dir", str(data))

    assert r.returncode == 1 and "artifact hashes do not match" in r.stderr
    assert _hashes(live) == before


def test_rollback_without_a_recorded_previous_index_is_refused(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_index(data / "faiss_index", "PUBLIC")
    r = _cli("rollback", "--data-dir", str(data))
    assert r.returncode == 1 and "no previous index" in r.stderr


def _sitemap(tmp_path, *routes):
    p = tmp_path / "sitemap.xml"
    p.write_text("<urlset>" + "".join(f"<url><loc>https://docs.example.org{r}</loc></url>" for r in routes) + "</urlset>")
    return str(p)


def _docs_checkout(tmp_path, count):
    """Front matter slugs for the helper index's d<i>.md documents (which carry no `route`)."""
    repo = tmp_path / "docs-repo"
    repo.mkdir()
    for i in range(count):
        (repo / f"d{i}.md").write_text(f"---\nslug: /d{i}/\n---\n# d{i}")
    return str(repo)


def test_promote_refuses_without_a_sitemap_unless_explicitly_skipped(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    _make_index(src, "PUBLIC")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data), skip_published_check=False)
    assert r.returncode == 1 and "--sitemap is required" in r.stderr
    assert not (data / "faiss_index").exists() and not (data / "faiss_candidates").exists()

    r = _cli("promote", "--source", str(src), "--data-dir", str(data))  # helper adds --skip-published-check
    assert r.returncode == 0, r.stderr
    assert json.loads((data / "faiss_index" / "manifest.json").read_text())["promotion_published_check"]["status"] == "SKIPPED"


def test_promote_refuses_a_candidate_with_an_unpublished_document(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    live = _legacy(data)
    before = _hashes(live)
    _make_index(src, "PUBLIC,PUBLIC,PUBLIC")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data),
             "--sitemap", _sitemap(tmp_path, "/d0/", "/d1/"), "--docs-repo", _docs_checkout(tmp_path, 3))
    assert r.returncode == 1 and "1 indexed documents are not published" in r.stderr and "d2.md route=/d2/" in r.stderr
    assert _hashes(live) == before and not (data / "faiss_candidates").exists()


def test_promote_refuses_documents_without_a_route_when_no_checkout_is_given(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    _make_index(src, "PUBLIC")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data), "--sitemap", _sitemap(tmp_path, "/d0/"))
    assert r.returncode == 1 and "no-route" in r.stderr


def test_promote_refuses_an_unreadable_sitemap(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    _make_index(src, "PUBLIC")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data), "--sitemap", str(tmp_path / "missing.xml"))
    assert r.returncode == 1 and "cannot read sitemap" in r.stderr


def test_promote_records_a_passed_published_check(tmp_path):
    data, src = tmp_path / "data", tmp_path / "src"
    data.mkdir()
    _make_index(src, "PUBLIC,PUBLIC")
    sitemap = _sitemap(tmp_path, "/", "/d0/", "/d1/")
    r = _cli("promote", "--source", str(src), "--data-dir", str(data), "--sitemap", sitemap, "--docs-repo", _docs_checkout(tmp_path, 2))
    assert r.returncode == 0, r.stderr
    assert "all 2 documents are in the sitemap (3 routes)" in r.stdout
    check = json.loads((data / "faiss_index" / "manifest.json").read_text())["promotion_published_check"]
    assert check["status"] == "PASSED" and check["sitemap"] == sitemap and check["route_source"] == "metadata+checkout"
