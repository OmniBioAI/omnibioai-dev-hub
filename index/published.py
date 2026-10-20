"""Is every indexed document actually published on the docs site?

The visibility label PUBLIC is a classification; this module checks the
outcome: each document's route must be in the docs site's sitemap. Routes come
from chunk metadata (`route`, recorded at ingestion from the page's front
matter slug) or, for indexes built before that field existed, from the slug in
a docs checkout. A document with no resolvable route is unpublished (fail
closed).
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ingestion.trusted import front_matter, normalize_route

SITEMAP_TIMEOUT_SECONDS = 30
_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


def parse_sitemap(xml: str) -> set[str]:
    """Routes (paths only, normalized) listed in a sitemap.xml document."""
    return {normalize_route(urlparse(loc).path or "/") for loc in _LOC.findall(xml)}


def load_sitemap_routes(source: str) -> set[str]:
    """Read a sitemap from an http(s) URL or a local file. An empty sitemap is an error."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "omnibioai-devhub-published-check"})
        with urllib.request.urlopen(req, timeout=SITEMAP_TIMEOUT_SECONDS) as resp:
            xml = resp.read().decode("utf-8")
    else:
        xml = Path(source).read_text(encoding="utf-8")
    routes = parse_sitemap(xml)
    if not routes:
        raise ValueError(f"sitemap {source} lists no routes")
    return routes


def route_map_from_checkout(docs_repo: str | Path, relative_paths: set[str]) -> dict[str, str]:
    """relative_path -> route from front matter slugs in a docs checkout (for indexes without `route`)."""
    docs_repo = Path(docs_repo)
    routes: dict[str, str] = {}
    for rel in relative_paths:
        path = docs_repo / rel
        if not path.is_file():
            continue
        slug = front_matter(path.read_text(encoding="utf-8", errors="ignore")).get("slug")
        if isinstance(slug, str) and slug.strip():
            routes[rel] = normalize_route(slug)
    return routes


def documents(metadata: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """One entry per (repo, relative_path): its route (first non-empty) and chunk count."""
    docs: dict[tuple[str, str], dict[str, Any]] = {}
    for m in metadata:
        key = (m.get("repo", ""), m.get("relative_path", ""))
        d = docs.setdefault(key, {"repo": key[0], "relative_path": key[1], "route": None, "chunks": 0})
        d["chunks"] += 1
        if not d["route"] and m.get("route"):
            d["route"] = normalize_route(m["route"])
    return docs


def unpublished_documents(metadata: list[dict[str, Any]], sitemap_routes: set[str],
                          route_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Documents whose route is unknown or not in the sitemap, sorted by repo and path."""
    route_map = route_map or {}
    out = []
    for d in documents(metadata).values():
        route = d["route"] or route_map.get(d["relative_path"])
        if route is None:
            out.append({**d, "reason": "no-route"})
        elif route not in sitemap_routes:
            out.append({**d, "route": route, "reason": "not-in-sitemap"})
    return sorted(out, key=lambda d: (d["repo"], d["relative_path"]))
