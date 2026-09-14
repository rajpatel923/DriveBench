#!/usr/bin/env bash
# =============================================================================
# 02_validate_recovery.sh  —  Measure image quality (PSNR / SSIM / L1) for
#                             each recovery method vs. the original frames.
#
# Works on macOS and Linux.  No VLM or GPU required.
# Requires: scikit-image  (pip install scikit-image)
#
# Usage:
#   bash script/02_validate_recovery.sh
#   bash script/02_validate_recovery.sh --limit 20
# =============================================================================
set -euo pipefail

# ---------- editable config --------------------------------------------------
ORIGINALS="data/nuscenes/samples"
FRAMELOST="data/corruption/FrameLost"
OUTPUT="data/recovery_validation.json"
LIMIT=""
# -----------------------------------------------------------------------------

source "$(dirname "$0")/../env.sh"

for arg in "$@"; do
  [[ "$arg" == "--limit" ]] && LIMIT_FLAG="--limit" && continue
  [[ -n "${LIMIT_FLAG:-}" ]] && LIMIT="$arg" && unset LIMIT_FLAG
done
LIMIT_ARGS=${LIMIT:+--limit $LIMIT}

# Collect all Recovered_* directories that exist
RECOVERED_DIRS=()
for d in data/corruption/Recovered_Previous \
          data/corruption/Recovered_Nearest \
          data/corruption/Recovered_LinearBlend \
          data/corruption/Recovered_RIFE; do
    [[ -d "$d" ]] && RECOVERED_DIRS+=("$d")
done

if [[ ${#RECOVERED_DIRS[@]} -eq 0 ]]; then
    echo "No Recovered_* directories found. Run 01_build_recovery.sh first."
    exit 1
fi

echo "Validating ${#RECOVERED_DIRS[@]} recovery method(s) against originals..."
python tools/validate_recovery.py \
    --originals     "$ORIGINALS" \
    --framelost-dir "$FRAMELOST" \
    --recovered     "${RECOVERED_DIRS[@]}" \
    --output        "$OUTPUT" \
    ${LIMIT_ARGS}

echo ""
echo "Results written to $OUTPUT"
echo "Next step: bash script/03_run_inference_mac.sh   (macOS)"
echo "        or bash script/03_run_inference_linux.sh  (Linux/CUDA)"
