#!/bin/bash
# check_and_reindex.sh
# Checks omnibioai-studio for a new release tag; if found, rebuilds the FAISS index.

LOG="/home/manish/.reindex.log"
STATE_FILE="/home/manish/.last_indexed_tag"

source /home/manish/miniconda3/etc/profile.d/conda.sh
conda activate chemoinfo

echo "[$(date)] Checking for new omnibioai-studio release..." >> "$LOG"

cd /home/manish/Desktop/machine/omnibioai-studio || { echo "[$(date)] ERROR: studio repo not found" >> "$LOG"; exit 1; }
git fetch --tags >> "$LOG" 2>&1

LATEST=$(git tag --sort=-creatordate | head -1)
LAST_INDEXED=$(cat "$STATE_FILE" 2>/dev/null || echo "")

if [ "$LATEST" != "$LAST_INDEXED" ] && [ -n "$LATEST" ]; then
    echo "[$(date)] New release detected: $LATEST (was: $LAST_INDEXED). Reindexing..." >> "$LOG"

    cd /home/manish/Desktop/machine/omnibioai-dev-hub || { echo "[$(date)] ERROR: dev-hub repo not found" >> "$LOG"; exit 1; }

    export REPO_BASE=/home/manish/Desktop/machine
    rm -rf data/faiss_index/*

    if python scripts/build_index.py >> "$LOG" 2>&1; then
        echo "$LATEST" > "$STATE_FILE"
        echo "[$(date)] Reindex complete for $LATEST." >> "$LOG"

        # restart API so it picks up the new index — adjust to your actual process manager
        sudo systemctl restart omnibioai-dev-hub 2>> "$LOG" || echo "[$(date)] WARNING: restart failed, restart manually" >> "$LOG"
    else
        echo "[$(date)] ERROR: build_index.py failed, index NOT updated, tag not advanced." >> "$LOG"
    fi
else
    echo "[$(date)] No new release ($LATEST already indexed)." >> "$LOG"
fi