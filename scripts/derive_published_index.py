"""One-off: derive a candidate holding only docs live on docs.omnibioai.org (no re-embedding).

Mirrors index/derive.py: vectors copied bit-for-bit, full validation, atomic rename.
Usage: derive_published.py SOURCE_DIR OUT_ROOT BUILD_ID ALLOWLIST EXCLUDED
"""
import collections, json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, "/app")
import numpy as np
from index.lifecycle import MANIFEST_FILE, PUBLIC_ONLY_POLICY, artifact_hashes, read_manifest, validate_index_directory, write_manifest
from index.vector_store import VectorStore
from ingestion.trusted import sha256_file, utc_now_iso

source_dir, out_root, build_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
allow = set(Path(sys.argv[4]).read_text().split())
expected_excluded = set(Path(sys.argv[5]).read_text().split())

pre = validate_index_directory(source_dir, require_public_only=True)
assert pre["ok"], pre
src_manifest = read_manifest(source_dir)
final_dir, tmp_dir = out_root / build_id, out_root / f".{build_id}.partial"
assert not final_dir.exists(), f"{final_dir} exists"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)

src = VectorStore(); assert src.load(str(source_dir))
src_docs = {m["relative_path"] for m in src.metadata}
assert allow <= src_docs, f"allowlist has docs not in source: {sorted(allow - src_docs)}"
assert src_docs - allow == expected_excluded, f"unexpected exclusion set: {sorted((src_docs - allow) ^ expected_excluded)}"

rows = [i for i, m in enumerate(src.metadata) if m["relative_path"] in allow]
vectors = np.ascontiguousarray(src.index.reconstruct_n(0, src.index.ntotal)[rows], dtype=np.float32)
metadata = [src.metadata[i] for i in rows]
out = VectorStore(); out.add(vectors, metadata); out.save(str(tmp_dir))

excluded_chunks = collections.Counter(m["relative_path"] for m in src.metadata if m["relative_path"] not in allow)
docs = {m["document_id"] for m in metadata}
manifest = dict(src_manifest)
for k in ("promotion_status", "promotion_timestamp", "previous_index", "previous_index_artifact_hashes", "previous_index_legacy"):
    manifest.pop(k, None)
manifest.update({
    "build_id": build_id,
    "build_timestamp": utc_now_iso(),
    "derived": True,
    "derivation": "published-on-docs.omnibioai.org filter (live sitemap match) of a PUBLIC_ONLY candidate; vectors copied bit-for-bit, not re-embedded",
    "derived_from": {"build_id": src_manifest["build_id"], "manifest_sha256": sha256_file(source_dir / MANIFEST_FILE),
                     "artifact_hashes": artifact_hashes(source_dir), "build_status": src_manifest["build_status"],
                     "chunk_count": len(src.metadata), "visibility_counts": src_manifest.get("visibility_counts"),
                     "derived_from": src_manifest.get("derived_from")},
    "excluded_not_published": {"documents": sorted(excluded_chunks), "chunks": sum(excluded_chunks.values())},
    "visibility_policy": PUBLIC_ONLY_POLICY,
    "parsed": len(docs), "chunked": len(metadata), "embedded": len(metadata), "indexed": len(metadata),
    "retrievable": len(metadata), "selected_document_count": len(docs), "selected_chunk_count": len(metadata),
    "visibility_counts": {"PUBLIC": len(metadata), "INTERNAL": 0, "REVIEW_REQUIRED": 0},
    "artifact_hashes": artifact_hashes(tmp_dir),
    "evaluation_status": "NOT_EVALUATED", "promotion_status": "CANDIDATE", "build_status": "CLEAN",
})
write_manifest(tmp_dir, manifest)

check = validate_index_directory(tmp_dir)  # full contract; manifest PUBLIC_ONLY enforces public-only
re = VectorStore(); re.load(str(tmp_dir))
ok = (check["ok"] and np.array_equal(re.index.reconstruct_n(0, re.index.ntotal), vectors) and re.metadata == metadata
      and not ({m["relative_path"] for m in re.metadata} & expected_excluded))
if not ok:
    shutil.rmtree(tmp_dir)
    raise SystemExit(f"FAILED: {check}")
os.replace(tmp_dir, final_dir)
print("VALIDATION", json.dumps(validate_index_directory(final_dir)))
print(f"before: chunks={len(src.metadata)} docs={len(src_docs)}  after: chunks={len(metadata)} docs={len(docs)}")
print("excluded docs -> chunks in source / chunks in new index:")
new_counts = collections.Counter(m["relative_path"] for m in re.metadata)
for p in sorted(expected_excluded):
    print(f"  {p}: {excluded_chunks[p]} / {new_counts.get(p, 0)}")
print("vectors bit-identical: True")
