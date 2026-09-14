#!/usr/bin/env bash
# =============================================================================
# 04_compare_conditions.sh  —  Compare VLM scores across all 5 conditions
#                              (clean, noimage, previous, nearest, linearblen)
#
# Works on macOS and Linux.
#
# Usage:
#   bash script/04_compare_conditions.sh --model qwen2.5-vl-7b-mlx
#   bash script/04_compare_conditions.sh --model qwen2.5-vl-7b-mlx --pilot
#   bash script/04_compare_conditions.sh --model qwen2.5-vl-7b       # Linux
#
# Note: GPT-based scoring requires OPENAI_API_KEY in your environment.
#       Omit --eval-gpt to skip GPT scoring and use language metrics only.
# =============================================================================
set -euo pipefail

source "$(dirname "$0")/../env.sh"

MODEL=""
PILOT=0
EVAL_GPT=""
WITH_RIFE=0
WITH_LIDAR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --pilot)      PILOT=1; shift ;;
        --eval-gpt)   EVAL_GPT="--eval-gpt"; shift ;;
        --with-rife)   WITH_RIFE=1; shift ;;
        --with-lidar)  WITH_LIDAR=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash script/04_compare_conditions.sh --model <model-dir-name> [--pilot] [--eval-gpt]"
    echo "  model-dir-name: subdirectory under res/, e.g. qwen2.5-vl-7b-mlx"
    exit 1
fi

PREFIX=$([[ $PILOT -eq 1 ]] && echo "pilot_" || echo "")
CONDITIONS=("${PREFIX}clean" "${PREFIX}noimage" "${PREFIX}previous" "${PREFIX}nearest" "${PREFIX}linearblen")
[[ $WITH_RIFE  -eq 1 ]] && CONDITIONS+=("${PREFIX}rife")
[[ $WITH_LIDAR -eq 1 ]] && CONDITIONS+=("${PREFIX}lidar")

# Check which conditions have result files
echo "Checking available result files in res/$MODEL/ ..."
AVAILABLE=()
for c in "${CONDITIONS[@]}"; do
    if [[ -f "res/$MODEL/${c}.json" ]]; then
        n=$(python3 -c "import json; print(len(json.load(open('res/$MODEL/${c}.json'))))" 2>/dev/null)
        echo "  [OK] ${c}.json  ($n results)"
        AVAILABLE+=("$c")
    else
        echo "  [--] ${c}.json  (missing)"
    fi
done

if [[ ${#AVAILABLE[@]} -lt 2 ]]; then
    echo ""
    echo "Need at least 2 conditions to compare. Run 03_run_inference_*.sh first."
    exit 1
fi

echo ""
echo "========================================"
echo " Condition comparison"
echo " Model: $MODEL"
echo " Baseline: ${AVAILABLE[0]}"
echo "========================================"

python tools/compare_conditions.py \
    --model      "$MODEL" \
    --conditions "${AVAILABLE[@]}" \
    --baseline   "${AVAILABLE[0]}"

echo ""
echo "========================================"
echo " Image quality summary (from validate_recovery.py)"
echo "========================================"
if [[ -f "data/recovery_validation.json" ]]; then
    python3 -c "
import json
data = json.load(open('data/recovery_validation.json'))
methods = set()
for row in data:
    methods.update(row['methods'].keys())
print(f'Frames evaluated: {len(data)}')
print()
for m in sorted(methods):
    rows = [r['methods'][m] for r in data if r['methods'].get(m, {}).get('available')]
    if not rows: continue
    n = len(rows)
    print(f'{m} (n={n}):')
    print(f'  PSNR: {sum(r[\"psnr\"] for r in rows)/n:.2f} dB   SSIM: {sum(r[\"ssim\"] for r in rows)/n:.4f}   L1: {sum(r[\"l1\"] for r in rows)/n:.2f}')
    print(f'  PSNR (crop): {sum(r[\"psnr_crop\"] for r in rows)/n:.2f} dB   SSIM (crop): {sum(r[\"ssim_crop\"] for r in rows)/n:.4f}')
"
else
    echo "  (no recovery_validation.json — run 02_validate_recovery.sh to generate)"
fi
