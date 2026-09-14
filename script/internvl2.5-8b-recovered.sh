#!/usr/bin/env bash
# InternVL2.5-8B recovered-frames condition.
#
# Prerequisite: data/corruption/Recovered/<CAM>/ must already be populated.
# See script/llava1.5-7b-recovered.sh for the full prerequisite chain.

python inference/internvl.py \
    --model "OpenGVLab/InternVL2_5-8B" \
    --data data/drivebench-test-final.json \
    --output "res/internvl2.5-8b/recovered.json" \
    --system_prompt prompt.txt \
    --corruption "Recovered"
