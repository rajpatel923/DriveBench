#!/usr/bin/env bash
# =============================================================================
# 01_build_recovery.sh  —  Build temporal neighbor manifest + all recovered
#                          image directories.
#
# Works on macOS and Linux (no GPU required — pure Python / PIL).
#
# Usage:
#   bash script/01_build_recovery.sh
#   bash script/01_build_recovery.sh --limit 20   # pilot: first 20 frames only
#
# Prerequisites:
#   data/nuscenes/v1.0-trainval_meta/   nuScenes metadata
#   data/nuscenes/v1.0-trainval01_blobs/ (or more blobs) for neighbor images
# =============================================================================
set -euo pipefail

# ---------- editable config --------------------------------------------------
META_DIR="data/nuscenes/v1.0-trainval_meta/v1.0-trainval"
BLOB_DIR="data/nuscenes/v1.0-trainval01_blobs"   # root with samples/ & sweeps/
NEIGHBOR_DIR="data/nuscenes/temporal_neighbors"
STEPS=2          # sweep hops on each side of the target keyframe
LIMIT=""         # leave empty for all frames; --limit 20 for a quick pilot
WITH_RIFE=0      # set to 1 via --with-rife (requires weights in recovery/rife/train_log/)
RIFE_WEIGHTS="recovery/rife/train_log"
WITH_LIDAR=0     # set to 1 via --with-lidar (requires sweeps/LIDAR_TOP/ in BLOB_DIR)
# -----------------------------------------------------------------------------

source "$(dirname "$0")/../env.sh"

# Parse CLI flags
for arg in "$@"; do
  [[ "$arg" == "--limit" ]]     && LIMIT_FLAG="--limit" && continue
  [[ -n "${LIMIT_FLAG:-}" ]]    && LIMIT="$arg" && unset LIMIT_FLAG && continue
  [[ "$arg" == "--with-rife" ]]  && WITH_RIFE=1 && continue
  [[ "$arg" == "--rife-weights" ]] && RIFE_W_FLAG=1 && continue
  [[ -n "${RIFE_W_FLAG:-}" ]]   && RIFE_WEIGHTS="$arg" && unset RIFE_W_FLAG && continue
  [[ "$arg" == "--with-lidar" ]] && WITH_LIDAR=1 && continue
done
LIMIT_ARGS=${LIMIT:+--limit $LIMIT}

echo "========================================"
echo " Step 1/2  Fetch temporal neighbors"
echo "========================================"
python tools/fetch_nuscenes_temporal_neighbors.py \
    --meta-dir     "$META_DIR" \
    --nuscenes-root "$BLOB_DIR" \
    --dest         "$NEIGHBOR_DIR" \
    --steps        "$STEPS"

echo ""
echo "========================================"
echo " Step 2/2  Build recovery conditions"
echo "========================================"
MANIFEST="$NEIGHBOR_DIR/neighbor_manifest.json"

for strategy in previous nearest linear_blend; do
    # Capitalise first letter and strip underscores for directory name
    dir_name="Recovered_$(echo "$strategy" | sed 's/_//g' | awk '{print toupper(substr($0,1,1)) substr($0,2)}')"
    dest="data/corruption/$dir_name"
    echo ""
    echo "  -> strategy: $strategy  dest: $dest"
    python recovery/temporal_recovery.py \
        --manifest      "$MANIFEST" \
        --neighbors-dir "$NEIGHBOR_DIR" \
        --dest          "$dest" \
        --strategy      "$strategy" \
        ${LIMIT_ARGS}
done

if [[ $WITH_RIFE -eq 1 ]]; then
    echo ""
    echo "  -> strategy: rife  dest: data/corruption/Recovered_RIFE"
    echo "     (weights: $RIFE_WEIGHTS)"
    python recovery/temporal_recovery.py \
        --manifest      "$MANIFEST" \
        --neighbors-dir "$NEIGHBOR_DIR" \
        --dest          "data/corruption/Recovered_RIFE" \
        --strategy      rife \
        --rife-weights  "$RIFE_WEIGHTS" \
        ${LIMIT_ARGS}
else
    echo ""
    echo "  [RIFE skipped — pass --with-rife to include it]"
    echo "  (run 'bash script/00_setup_rife.sh' first to download weights)"
fi

if [[ $WITH_LIDAR -eq 1 ]]; then
    echo ""
    echo "  -> strategy: lidar  dest: data/corruption/Recovered_LiDAR"
    python recovery/lidar_recovery.py \
        --meta-dir      "$META_DIR" \
        --nuscenes-root "$BLOB_DIR" \
        --dest          "data/corruption/Recovered_LiDAR" \
        --manifest      "$MANIFEST" \
        ${LIMIT_ARGS}
else
    echo ""
    echo "  [LiDAR skipped — pass --with-lidar to include it]"
    echo "  (requires sweeps/LIDAR_TOP/ in $BLOB_DIR)"
fi

echo ""
echo "Done. Recovered directories:"
ls data/corruption/ | grep Recovered

echo ""
echo "Next step: bash script/02_validate_recovery.sh"
