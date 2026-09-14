#!/usr/bin/env bash
# Qwen2.5-VL-7B recovered-frames condition via vLLM (GPU / CUDA).
#
# Prerequisite: data/corruption/Recovered/<CAM>/ must already be populated.
# See script/llava1.5-7b-recovered.sh for the full prerequisite chain.

GPU=$1

python inference/qwen2vl.py \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --data data/drivebench-test-final.json \
    --output "res/qwen2.5-vl-7b/recovered" \
    --system_prompt prompt.txt \
    --num_processes "${GPU}" \
    --max_model_len 8192 \
    --corruption "Recovered"
