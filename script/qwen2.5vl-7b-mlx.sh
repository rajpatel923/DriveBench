#!/usr/bin/env bash
# Qwen2.5-VL-7B inference via MLX-VLM (Apple Silicon).
# Runs the full 17-condition DriveBench sweep using inference/qwen2vl_mlx.py.
# Usage:  bash script/qwen2.5vl-7b-mlx.sh

corruptions=(
    ""              # clean
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
    python inference/qwen2vl_mlx.py \
        --model "mlx-community/Qwen2.5-VL-7B-Instruct-bf16" \
        --data data/drivebench-test-final.json \
        --output "res/qwen2.5-vl-7b-mlx/${outputs[i]}.json" \
        --system_prompt prompt.txt \
        --corruption "${corruptions[i]}"
done
