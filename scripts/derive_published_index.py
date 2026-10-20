#!/usr/bin/env python3
"""Derive a candidate holding only the documents published on the docs site (no re-embedding).

Input is a PUBLIC_ONLY candidate (scripts/derive_public_index.py output). A
document is kept only if its route is in the docs sitemap; routes come from
chunk metadata or, for older indexes, from front matter slugs in --docs-repo.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.derive import derive_published_subset
from index.lifecycle import read_manifest
from index.published import load_sitemap_routes, route_map_from_checkout
from index.vector_store import VectorStore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="validated PUBLIC_ONLY candidate directory")
    ap.add_argument("--out-root", required=True, help="staging root to create the derived candidate under")
    ap.add_argument("--sitemap", required=True, help="docs sitemap.xml: URL or local file")
    ap.add_argument("--docs-repo", help="docs checkout to read route slugs from, for indexes without `route` metadata")
    ap.add_argument("--build-id")
    args = ap.parse_args()
    routes = load_sitemap_routes(args.sitemap)
    route_map = None
    if args.docs_repo:
        store = VectorStore()
        if not store.load(args.source):
            print(f"REFUSED: cannot load {args.source}", file=sys.stderr)
            return 1
        route_map = route_map_from_checkout(args.docs_repo, {m.get("relative_path", "") for m in store.metadata})
    r = derive_published_subset(args.source, args.out_root, routes, build_id=args.build_id, route_map=route_map,
                                sitemap_source=args.sitemap)
    print(f"derived: {r['derived_dir']} (from {read_manifest(args.source)['build_id']})")
    print(f"chunks {r['source_chunks']} -> {r['published_chunks']}; published documents={r['published_documents']}")
    for d in r["excluded_not_published"]["documents"]:
        print(f"  excluded {d['repo']}:{d['relative_path']} ({d['reason']}, {d['chunks']} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
