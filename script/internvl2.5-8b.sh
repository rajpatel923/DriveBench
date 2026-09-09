#!/usr/bin/env bash
# InternVL2.5-8B inference — full 17-condition DriveBench sweep.
# Works on both Mac (MPS) and GPU (CUDA) — device is auto-detected in inference/internvl.py.
# Usage:  bash script/internvl2.5-8b.sh

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
    python inference/internvl.py \
        --model "OpenGVLab/InternVL2_5-8B" \
        --data data/drivebench-test-final.json \
        --output "res/internvl2.5-8b/${outputs[i]}.json" \
        --system_prompt prompt.txt \
        --corruption "${corruptions[i]}"
done
