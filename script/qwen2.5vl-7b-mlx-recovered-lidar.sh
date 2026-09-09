#!/usr/bin/env bash
# Qwen2.5-VL-7B on the LiDAR-projection recovered condition (Mac/MLX).
#
# Prerequisite: populate data/corruption/Recovered_LiDAR/ first:
#   python recovery/lidar_recovery.py \
#       --meta-dir /mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval \
#       --nuscenes-root <PATH_TO_NUSCENES_WITH_SWEEPS> \
#       --dest data/corruption/Recovered_LiDAR

python inference/qwen2vl_mlx.py \
    --model "mlx-community/Qwen2.5-VL-7B-Instruct-bf16" \
    --data data/drivebench-test-final.json \
    --output "res/qwen2.5-vl-7b-mlx/recovered_lidar.json" \
    --system_prompt prompt.txt \
    --corruption "Recovered_LiDAR"
