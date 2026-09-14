#!/usr/bin/env bash
# Qwen2.5-VL-7B recovered-frames condition via MLX-VLM (Apple Silicon).
#
# Prerequisite: data/corruption/Recovered/<CAM>/ must already be populated.
# See script/llava1.5-7b-recovered.sh for the full prerequisite chain.

python inference/qwen2vl_mlx.py \
    --model "mlx-community/Qwen2.5-VL-7B-Instruct-bf16" \
    --data data/drivebench-test-final.json \
    --output "res/qwen2.5-vl-7b-mlx/recovered.json" \
    --system_prompt prompt.txt \
    --corruption "Recovered"
