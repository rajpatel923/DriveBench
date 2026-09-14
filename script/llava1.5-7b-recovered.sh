GPU=$1

# Runs inference for the "Recovered" condition only -- pairs with the
# existing clean/framelost/noimage runs from script/llava1.5-7b.sh for the
# temporal-recovery comparison (see claude/TEMPORAL_RECOVERY_RESEARCH.md).
#
# Prerequisite: data/corruption/Recovered/<CAM>/ must already be populated,
# e.g. via:
#   python tools/fetch_nuscenes_temporal_neighbors.py --meta-dir <...> --nuscenes-root <...> --dest data/nuscenes/temporal_neighbors
#   python recovery/temporal_recovery.py --manifest data/nuscenes/temporal_neighbors/neighbor_manifest.json \
#       --neighbors-dir data/nuscenes/temporal_neighbors --dest data/corruption/Recovered
#
# No inference code change is needed for this: inference/llava1.5.py's
# generic corruption-swap logic (img_path.replace('nuscenes/samples',
# f'corruption/{corruption}')) already reads from data/corruption/Recovered/
# for --corruption Recovered.

python inference/llava1.5.py \
    --model 'llava-hf/llava-1.5-7b-hf' \
    --data data/drivebench-test-final.json \
    --output "res/llava-1.5-7b/recovered" \
    --system_prompt prompt.txt \
    --num_processes "${GPU}" \
    --max_model_len 4096 \
    --corruption "Recovered"
