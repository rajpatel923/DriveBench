#!/usr/bin/env bash
# =============================================================================
# 00b_download_vlm_weights.sh  —  Pre-download the three VLM checkpoints used
#                                  by this repo's GPU/Linux inference scripts,
#                                  so the weights are cached before you kick
#                                  off a full 17-condition sweep.
#
# Models (see claude/IMPLEMENTATION_GUIDE.md VLM selection table):
#   - Qwen/Qwen2.5-VL-7B-Instruct   (primary backbone)
#   - OpenGVLab/InternVL2_5-8B      (secondary comparison)
#   - llava-hf/llava-1.5-7b-hf      (baseline)
#
# Usage:
#   bash script/00b_download_vlm_weights.sh            # all three
#   bash script/00b_download_vlm_weights.sh qwen        # just one
#   bash script/00b_download_vlm_weights.sh qwen intern
#
# Requires: huggingface_hub (already in .venv). If any repo is gated, run
#   huggingface-cli login
# first and accept the license on the model's HF page.
# =============================================================================
set -euo pipefail

declare -A MODELS=(
    [qwen]="Qwen/Qwen2.5-VL-7B-Instruct"
    [intern]="OpenGVLab/InternVL2_5-8B"
    [llava]="llava-hf/llava-1.5-7b-hf"
)

TARGETS=("$@")
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(qwen intern llava)

for key in "${TARGETS[@]}"; do
    repo="${MODELS[$key]:-}"
    if [[ -z "$repo" ]]; then
        echo "Unknown target: $key (expected: qwen, intern, llava)"
        exit 1
    fi
    echo ""
    echo "========================================"
    echo " Downloading $repo"
    echo "========================================"
    huggingface-cli download "$repo"
done

echo ""
echo "All requested weights cached to \$HF_HOME (default: ~/.cache/huggingface/hub)."
