#!/usr/bin/env python3
"""Derive the PUBLIC-only production candidate from a validated mixed candidate (no re-embedding)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.derive import derive_public_only


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="validated mixed-visibility candidate directory")
    ap.add_argument("--out-root", required=True, help="staging root to create the derived candidate under")
    ap.add_argument("--build-id")
    args = ap.parse_args()
    r = derive_public_only(args.source, args.out_root, args.build_id)
    print(f"derived: {r['derived_dir']}")
    print(f"public documents={r['public_documents']} chunks={r['public_chunks']} excluded={r['excluded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
