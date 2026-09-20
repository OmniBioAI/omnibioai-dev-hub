#!/usr/bin/env python3
"""Verify an index directory against the Phase 18 contract. Exit 0 = valid, 1 = not.

Used by the container entrypoint so the service never starts on a missing,
incomplete, legacy, rejected, tampered or non-PUBLIC-only index -- and never
tries to build one.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.lifecycle import validate_index_directory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index_dir")
    ap.add_argument("--require-public-only", action="store_true")
    args = ap.parse_args()
    try:
        result = validate_index_directory(args.index_dir, require_public_only=args.require_public_only)
    except Exception as exc:  # noqa: BLE001 -- any failure to even inspect the index is "not valid"
        print(f"INDEX INVALID: could not inspect {args.index_dir}: {exc!r}", file=sys.stderr)
        return 1
    if not result["ok"]:
        print(f"INDEX INVALID: {result['reason']}", file=sys.stderr)
        return 1
    print(f"INDEX OK: {result['faiss_ntotal']} vectors, dim {result['dimension']}, visibility {result['visibility_counts']}, public_only={result['public_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
