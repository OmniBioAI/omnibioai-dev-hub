import os

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache", "obsolete", "production-readiness"}

# Path-based exclusion: skip any subtree whose path contains one of these
# directory names *relative to the repo root*.  "work" covers omnibioai/work/,
# which holds UUID-named and wftest_*/sweep_* runtime copies of bundle READMEs
# that would otherwise shadow the canonical omnibioai-workflow-bundles/ paths.
# Checked across all 19 repos — only omnibioai has a work/ dir, and it contains
# only execution artifacts, never source documentation.
SKIP_PATH_SEGMENTS = {"work"}


def load_documents(repo_paths):
    docs = []
    for repo in repo_paths:
        if not os.path.exists(repo):
            print(f"⚠️  Skipping missing repo: {repo}")
            continue
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            # Skip directories that live under a path segment we want to exclude.
            # Using relpath keeps the check repo-relative so "work" in one repo
            # doesn't accidentally affect a different repo that has a legitimate
            # subdirectory with the same name.
            rel = os.path.relpath(root, repo)
            if any(part in SKIP_PATH_SEGMENTS for part in rel.split(os.sep)):
                dirs.clear()
                continue

            for fname in files:
                if fname.endswith(".md"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read().strip()
                        if text:
                            docs.append({
                                "id": fpath,
                                "text": text,
                                "source": fpath
                            })
                    except Exception as e:  # noqa: BLE001 -- one unreadable file must not abort the whole indexing walk
                        print(f"⚠️  Could not read {fpath}: {e}")
    print(f"📄 Loaded {len(docs)} documents")
    return docs