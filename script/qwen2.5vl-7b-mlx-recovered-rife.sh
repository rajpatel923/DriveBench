#!/usr/bin/env bash
# Qwen2.5-VL-7B on the RIFE-interpolated recovered condition (Mac/MLX).
#
# Prerequisite: populate data/corruption/Recovered_RIFE/ first:
#   python recovery/temporal_recovery.py \
#       --manifest data/nuscenes/temporal_neighbors/neighbor_manifest.json \
#       --neighbors-dir data/nuscenes/temporal_neighbors \
#       --dest data/corruption/Recovered_RIFE \
#       --strategy rife \
#       --rife-weights recovery/rife/train_log

python inference/qwen2vl_mlx.py \
    --model "mlx-community/Qwen2.5-VL-7B-Instruct-bf16" \
    --data data/drivebench-test-final.json \
    --output "res/qwen2.5-vl-7b-mlx/recovered_rife.json" \
    --system_prompt prompt.txt \
    --corruption "Recovered_RIFE"
