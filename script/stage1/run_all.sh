#!/usr/bin/env bash
# Stage 1 ("Pixels or Words?") inference runs for Qwen2.5-VL-7B, greedy.
#
# Usage:
#   bash script/stage1/run_all.sh              # the 400 scored MCQs per run
#   bash script/stage1/run_all.sh --limit 20   # smoke test
#   bash script/stage1/run_all.sh --only "P_lidar S05 S10"   # just these labels
#
# A run is skipped if <label>.json already exists, so the script can be
# re-run after a crash. Failed runs are appended to failures.txt and the
# script carries on. Run tools/preflight.py first.

set -uo pipefail

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATA="${DATA:-data/drivebench-mcq.json}"   # the 400 scored MCQs; see tools/make_mcq_data.py
SYSTEM_PROMPT="prompt.txt"
TEXT_DIR="data/sensor_text"
OUT_DIR="results/stage1/qwen2.5-vl-7b_mcq"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$NUM_GPUS" -ge 1 ]] || NUM_GPUS=1

LIMIT=""
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --only)  ONLY=" $2 "; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$LIMIT" ]] && OUT_DIR="${OUT_DIR}_limit${LIMIT}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p "$OUT_DIR"
FAILURES="$OUT_DIR/failures.txt"

run_condition() {
    local label="$1"
    local corruption="$2"
    local sensor_text="${3:-}"
    # No .json suffix here: qwen2vl.py's ray.data.write_json() treats --output
    # as a directory, then re-saves the merged result to <output>.json itself.
    local output="$OUT_DIR/$label"

    if [[ -n "$ONLY" && "$ONLY" != *" $label "* ]]; then
        return
    fi

    # Skip only complete runs; a crash can leave an empty or truncated .json.
    if python -c "import json,sys; sys.exit(len(json.load(open('$output.json'))) == 0)" 2>/dev/null; then
        echo "Skip (exists): $label"
        return
    fi
    rm -rf "$output" "$output.json"

    echo ""
    echo "----------------------------------------"
    echo " Condition:   $label"
    echo " Corruption:  ${corruption:-<clean>}"
    echo " Sensor text: ${sensor_text:-<none>}"
    echo " Output:      $output"
    echo "----------------------------------------"

    local args=(
        --model         "$MODEL"
        --data          "$DATA"
        --output        "$output"
        --system_prompt "$SYSTEM_PROMPT"
        --max_model_len 16384
        --num_processes "$NUM_GPUS"
        --temperature   0.0
        --top_p         1.0
        --seed          0
    )
    [[ -n "$corruption" ]]  && args+=(--corruption "$corruption")
    [[ -n "$sensor_text" ]] && args+=(--sensor_text "$TEXT_DIR/$sensor_text")
    [[ -n "$LIMIT" ]]       && args+=(--limit "$LIMIT")

    if python inference/qwen2vl.py "${args[@]}" && [[ -s "$output.json" ]]; then
        echo "Done: $label"
    else
        echo "FAILED: $label"
        echo "$(date -Iseconds) $label" >> "$FAILURES"
    fi
}

#             label              corruption        sensor text
run_condition C0_clean           ""
run_condition C1_blank           NoImage
run_condition W_oracle           NoImage           oracle.json
run_condition W_shuffled         NoImage           oracle_shuffled.json
run_condition P_lidar            Recovered_LiDAR
run_condition W_cluster          NoImage           cluster.json
run_condition C0b_clean_rerun    ""
run_condition S05                Stale_0.5s
run_condition S10                Stale_1.0s
run_condition S20                Stale_2.0s
run_condition S10_cluster        Stale_1.0s        cluster.json
# Added after the first full run: 10-sweep LiDAR + radar image, and ego motion.
# LiDAR conditions get a one-line legend automatically (--image_note auto).
run_condition P_lidar2           Recovered_LiDAR_Radar
run_condition E_blank_ego        NoImage           ego.json
run_condition W_oracle_ego       NoImage           oracle_ego.json
run_condition W_cluster_ego      NoImage           cluster_ego.json
run_condition P_lidar2_ego       Recovered_LiDAR_Radar ego.json

echo ""
if [[ -s "$FAILURES" ]]; then
    echo "Some runs failed; see $FAILURES"
    exit 1
fi
echo "All runs complete: $OUT_DIR"
