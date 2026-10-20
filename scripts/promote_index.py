#!/usr/bin/env python3
"""Controlled, explicit promotion and rollback of the production index.

Run it as the service's own uid (10001) with only the data directory mounted,
so no host permission changes are needed:

    promote   check every indexed document is published on the docs site (--sitemap;
              routes from chunk metadata or --docs-repo front matter), copy a validated
              PUBLIC-only candidate under <data-dir>/faiss_candidates,
              verify the copied hashes, validate it, then swap it in as
              <data-dir>/faiss_index, retaining the previous index under
              <data-dir>/faiss_previous
    rollback  swap the retained previous index back in

NOT ATOMIC (see index.lifecycle.promote_candidate). Run with the service stopped.
Never invoked by the service itself; startup only verifies (scripts/verify_index.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.lifecycle import (
    artifact_hashes,
    promote_candidate,
    read_manifest,
    rollback,
    validate_index_directory,
    write_manifest,
)
from index.published import (
    load_sitemap_routes,
    route_map_from_checkout,
    unpublished_documents,
)
from index.vector_store import VectorStore
from ingestion.trusted import utc_now_iso

UNPUBLISHED_SHOWN = 20


def _state(data: Path) -> dict:
    return {p.name: sorted(c.name for c in p.iterdir()) for p in sorted(data.iterdir()) if p.is_dir()}


def _stage_candidate(source: Path, dest: Path, expected_hashes: dict[str, str]) -> None:
    """Copy an external candidate into the data filesystem without exposing a partial final path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{dest.name}.partial-", dir=dest.parent))
    try:
        shutil.copytree(source, partial, dirs_exist_ok=True)
        copied_hashes = artifact_hashes(partial)
        if copied_hashes != artifact_hashes(source) or copied_hashes != expected_hashes:
            raise RuntimeError("copied artifact hashes do not match the source/manifest")
        check = validate_index_directory(partial, require_public_only=True)
        if not check["ok"]:
            raise RuntimeError(f"copied candidate invalid: {check['reason']}")
        os.replace(partial, dest)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def _published_check(args: argparse.Namespace, source: Path) -> dict | None:
    """Every indexed document must be on the docs site. Returns the check record, or None after printing REFUSED."""
    if args.skip_published_check:
        print("WARNING: published check SKIPPED (--skip-published-check); recorded in the manifest", file=sys.stderr)
        return {"status": "SKIPPED", "checked_at": utc_now_iso()}
    if not args.sitemap:
        print("REFUSED: --sitemap is required: every indexed document must be published on the docs site "
              "(pass --skip-published-check to override deliberately)", file=sys.stderr)
        return None
    try:
        routes = load_sitemap_routes(args.sitemap)
    except Exception as exc:  # noqa: BLE001 -- any unreadable sitemap refuses promotion
        print(f"REFUSED: cannot read sitemap {args.sitemap}: {exc}", file=sys.stderr)
        return None
    store = VectorStore()
    if not store.load(str(source)):
        print("REFUSED: candidate vector store did not load", file=sys.stderr)
        return None
    paths = {m.get("relative_path", "") for m in store.metadata}
    route_map = route_map_from_checkout(args.docs_repo, paths) if args.docs_repo else None
    bad = unpublished_documents(store.metadata, routes, route_map)
    if bad:
        print(f"REFUSED: {len(bad)} indexed documents are not published on the docs site:", file=sys.stderr)
        for d in bad[:UNPUBLISHED_SHOWN]:
            print(f"  {d['repo']}:{d['relative_path']} route={d['route']} ({d['reason']}, {d['chunks']} chunks)", file=sys.stderr)
        if len(bad) > UNPUBLISHED_SHOWN:
            print(f"  ... and {len(bad) - UNPUBLISHED_SHOWN} more", file=sys.stderr)
        return None
    print(f"published check: all {len(paths)} documents are in the sitemap ({len(routes)} routes)")
    return {"status": "PASSED", "sitemap": args.sitemap, "sitemap_routes": len(routes), "documents": len(paths),
            "route_source": "metadata+checkout" if route_map else "metadata", "checked_at": utc_now_iso()}


def cmd_promote(args: argparse.Namespace) -> int:
    source, data = Path(args.source), Path(args.data_dir)
    manifest = read_manifest(source)
    build_id = manifest["build_id"]
    dest = data / "faiss_candidates" / build_id
    source_is_staged = source.resolve() == dest.resolve()
    if dest.exists() and not source_is_staged:
        print(f"REFUSED: {dest} already exists", file=sys.stderr)
        return 1
    check = validate_index_directory(source, require_public_only=True)
    if not check["ok"]:
        print(f"REFUSED: source candidate invalid: {check['reason']}", file=sys.stderr)
        return 1
    published = _published_check(args, source)
    if published is None:
        return 1
    if source_is_staged:
        if artifact_hashes(source) != manifest.get("artifact_hashes"):
            print("REFUSED: staged artifact hashes do not match the manifest", file=sys.stderr)
            return 1
        print(f"using staged {build_id}; hashes verified: {artifact_hashes(source)}")
    else:
        try:
            _stage_candidate(source, dest, manifest.get("artifact_hashes") or {})
        except Exception as exc:  # noqa: BLE001 -- report staging failure; active index is still untouched
            print(f"REFUSED: staging failed: {exc}", file=sys.stderr)
            return 1
        print(f"copied {build_id}; hashes verified: {artifact_hashes(dest)}")
    result = promote_candidate(dest, data / "faiss_index", data / "faiss_previous")
    promoted = read_manifest(data / "faiss_index")
    promoted["promotion_published_check"] = published
    write_manifest(data / "faiss_index", promoted)
    print("PROMOTED " + json.dumps(result))
    print("state " + json.dumps(_state(data)))
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    data = Path(args.data_dir)
    previous = read_manifest(data / "faiss_index").get("previous_index")
    if not previous:
        print("REFUSED: current manifest records no previous index", file=sys.stderr)
        return 1
    result = rollback(data / "faiss_index", previous, data / "faiss_rolled")
    print("ROLLED BACK " + json.dumps(result))
    print("state " + json.dumps(_state(data)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--source", required=True, help="validated PUBLIC-only candidate directory (read-only mount is fine)")
    p.add_argument("--data-dir", default="/app/data")
    p.add_argument("--sitemap", help="docs sitemap.xml (URL or file); every indexed document must be listed in it")
    p.add_argument("--docs-repo", help="docs checkout to read route slugs from, for indexes without `route` metadata")
    p.add_argument("--skip-published-check", action="store_true",
                   help="promote without the sitemap check (recorded as SKIPPED in the promoted manifest)")
    p.set_defaults(fn=cmd_promote)
    r = sub.add_parser("rollback")
    r.add_argument("--data-dir", default="/app/data")
    r.set_defaults(fn=cmd_rollback)
    args = ap.parse_args()
    try:
        return args.fn(args)
    except Exception as exc:  # noqa: BLE001 -- surface the exact failure state to the operator
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
