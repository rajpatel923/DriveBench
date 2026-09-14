#!/usr/bin/env bash
# =============================================================================
# 03_run_inference_mac.sh  —  Run Qwen2.5-VL-7B inference via MLX-VLM
#                             (Apple Silicon / macOS only).
#
# Runs 5 conditions in sequence: clean, noimage, previous, nearest, linearblen
# Supports --pilot to run on the 18-frame blob-01 subset instead of the full
# 200-frame dataset.
#
# Usage:
#   bash script/03_run_inference_mac.sh            # full dataset
#   bash script/03_run_inference_mac.sh --pilot    # 18-frame pilot subset
#
# Requires: pip install mlx-vlm
# =============================================================================
set -euo pipefail

# ---------- editable config --------------------------------------------------
MODEL="mlx-community/Qwen2.5-VL-7B-Instruct-bf16"
SYSTEM_PROMPT="prompt.txt"
OUT_DIR="res/qwen2.5-vl-7b-mlx"
# -----------------------------------------------------------------------------

source "$(dirname "$0")/../env.sh"

PILOT=0
WITH_RIFE=0
WITH_LIDAR=0
for arg in "$@"; do
    [[ "$arg" == "--pilot" ]]      && PILOT=1
    [[ "$arg" == "--with-rife" ]]  && WITH_RIFE=1
    [[ "$arg" == "--with-lidar" ]] && WITH_LIDAR=1
done

if [[ $PILOT -eq 1 ]]; then
    DATA="data/blob01_subset.json"
    PREFIX="pilot_"
    echo "Mode: PILOT (18-frame subset)"
else
    DATA="data/drivebench-test-final.json"
    PREFIX=""
    echo "Mode: FULL dataset (200 frames, 1461 questions)"
fi

mkdir -p "$OUT_DIR"

run_condition() {
    local label="$1"
    local corruption="$2"
    local output="$OUT_DIR/${PREFIX}${label}.json"

    echo ""
    echo "----------------------------------------"
    echo " Condition: $label"
    echo " Output:    $output"
    echo "----------------------------------------"

    local args=(
        --model        "$MODEL"
        --data         "$DATA"
        --output       "$output"
        --system_prompt "$SYSTEM_PROMPT"
    )
    [[ -n "$corruption" ]] && args+=(--corruption "$corruption")

    python inference/qwen2vl_mlx.py "${args[@]}"
    echo "Done: $label"
}

run_condition "clean"        ""
run_condition "noimage"      "NoImage"
run_condition "previous"     "Recovered_Previous"
run_condition "nearest"      "Recovered_Nearest"
run_condition "linearblen"   "Recovered_LinearBlend"

if [[ $WITH_RIFE -eq 1 ]]; then
    run_condition "rife" "Recovered_RIFE"
else
    echo ""
    echo "[RIFE skipped — pass --with-rife to include it]"
fi

if [[ $WITH_LIDAR -eq 1 ]]; then
    run_condition "lidar" "Recovered_LiDAR"
else
    echo ""
    echo "[LiDAR skipped — pass --with-lidar to include it]"
fi

N_COND=5
[[ $WITH_RIFE  -eq 1 ]] && N_COND=$((N_COND+1))
[[ $WITH_LIDAR -eq 1 ]] && N_COND=$((N_COND+1))
echo ""
echo "========================================"
echo " All $N_COND conditions complete."
echo " Results in: $OUT_DIR/"
echo " Next step:  bash script/04_compare_conditions.sh --model qwen2.5-vl-7b-mlx${PILOT:+ --pilot}${WITH_RIFE:+ --with-rife}${WITH_LIDAR:+ --with-lidar}"
echo "========================================"
