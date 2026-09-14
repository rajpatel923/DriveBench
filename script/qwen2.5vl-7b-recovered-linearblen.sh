#!/usr/bin/env bash
# Qwen2.5-VL-7B on the linear-blend recovered condition (Linux/vLLM).
#
# Prerequisite: populate data/corruption/Recovered_LinearBlend/ first:
#   python recovery/temporal_recovery.py \
#       --manifest data/nuscenes/temporal_neighbors/neighbor_manifest.json \
#       --neighbors-dir data/nuscenes/temporal_neighbors \
#       --dest data/corruption/Recovered_LinearBlend \
#       --strategy linear_blend

source env.linux.sh

python inference/qwen2vl.py \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --data data/drivebench-test-final.json \
    --output "res/qwen2.5-vl-7b/recovered_linearblen.json" \
    --system_prompt prompt.txt \
    --corruption "Recovered_LinearBlend"
