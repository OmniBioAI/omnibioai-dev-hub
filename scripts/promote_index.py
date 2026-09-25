#!/usr/bin/env python3
"""Controlled, explicit promotion and rollback of the production index.

Run it as the service's own uid (10001) with only the data directory mounted,
so no host permission changes are needed:

    promote   copy a validated PUBLIC-only candidate under <data-dir>/faiss_candidates,
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
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.lifecycle import (
    artifact_hashes,
    promote_candidate,
    read_manifest,
    rollback,
    validate_index_directory,
)


def _state(data: Path) -> dict:
    return {p.name: sorted(c.name for c in p.iterdir()) for p in sorted(data.iterdir()) if p.is_dir()}


def cmd_promote(args: argparse.Namespace) -> int:
    source, data = Path(args.source), Path(args.data_dir)
    manifest = read_manifest(source)
    build_id = manifest["build_id"]
    dest = data / "faiss_candidates" / build_id
    if dest.exists():
        print(f"REFUSED: {dest} already exists", file=sys.stderr)
        return 1
    check = validate_index_directory(source, require_public_only=True)
    if not check["ok"]:
        print(f"REFUSED: source candidate invalid: {check['reason']}", file=sys.stderr)
        return 1
    (data / "faiss_candidates").mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    if artifact_hashes(dest) != artifact_hashes(source) or artifact_hashes(dest) != manifest.get("artifact_hashes"):
        shutil.rmtree(dest)
        print("REFUSED: copied artifact hashes do not match the source/manifest; copy removed", file=sys.stderr)
        return 1
    print(f"copied {build_id}; hashes verified: {artifact_hashes(dest)}")
    result = promote_candidate(dest, data / "faiss_index", data / "faiss_previous")
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
