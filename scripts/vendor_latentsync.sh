#!/usr/bin/env bash
# Clone (or verify) the pinned LatentSync checkout under vendor/.
# Checkpoints are fetched separately by gpu/lipsync_latentsync.py on first run
# (or via `python -m gpu.lipsync_latentsync --download-only`), since they're
# large and belong in weights/ or Drive, never in vendor/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/vendor/LatentSync"
# Pinned for reproducibility. Bump deliberately, not accidentally.
PIN="a229c3948406bc2cf6eaf4873e662e70c6a04746"

if [ -d "$DEST/.git" ]; then
  have="$(git -C "$DEST" rev-parse HEAD)"
  if [ "$have" = "$PIN" ]; then
    echo "vendor/LatentSync already at pinned commit $PIN"
    exit 0
  fi
  echo "vendor/LatentSync is at $have, expected $PIN — leaving it alone."
  echo "Delete vendor/LatentSync and re-run this script to reset to the pin."
  exit 1
fi

echo "Cloning bytedance/LatentSync @ $PIN"
git clone --quiet https://github.com/bytedance/LatentSync.git "$DEST"
git -C "$DEST" checkout --quiet "$PIN"
echo "Vendored at $DEST"
