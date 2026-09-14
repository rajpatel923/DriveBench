#!/usr/bin/env bash
# Qwen2.5-VL-7B inference via vLLM (GPU / CUDA).
# Uses the existing generic inference/qwen2vl.py — no code changes needed.
# Usage:  bash script/qwen2.5vl-7b.sh <num_gpus>

GPU=$1

corruptions=(
    ""
    "NoImage"
    "BitError"
    "CameraCrash"
    "Fog"
    "H256ABRCompression"
    "LowLight"
    "Rain"
    "Snow"
    "Brightness"
    "ColorQuant"
    "FrameLost"
    "LensObstacleCorruption"
    "MotionBlur"
    "Saturate"
    "ZoomBlur"
    "WaterSplashCorruption"
)

outputs=(
    "clean"
    "noimage"
    "biterror"
    "cameracrash"
    "fog"
    "h256"
    "lowlight"
    "rain"
    "snow"
    "bright"
    "colorquant"
    "framelost"
    "lens"
    "motion"
    "saturate"
    "zoom"
    "water"
)

for i in "${!outputs[@]}"; do
    python inference/qwen2vl.py \
        --model "Qwen/Qwen2.5-VL-7B-Instruct" \
        --data data/drivebench-test-final.json \
        --output "res/qwen2.5-vl-7b/${outputs[i]}" \
        --system_prompt prompt.txt \
        --num_processes "${GPU}" \
        --max_model_len 8192 \
        --corruption "${corruptions[i]}"
done
