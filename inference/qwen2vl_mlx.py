"""
Qwen2-VL inference via MLX-VLM, for running on Apple Silicon (e.g. M3 Max) instead
of vLLM/CUDA. This is a sequential, single-machine counterpart to inference/qwen2vl.py
(which uses vLLM + Ray for GPU clusters) - MLX-VLM has no multi-request batching API,
so this simply loops over the dataset.

Run this on the Mac itself, not on the CUDA machine - requires `pip install mlx-vlm`.
Output format matches inference/llava1.5.py and inference/qwen2vl.py so the same
comparison scripts work unchanged.

Usage:
    python inference/qwen2vl_mlx.py \
        --model mlx-community/Qwen2-VL-7B-Instruct-bf16 \
        --data data/drivebench-test-final.json \
        --output res/qwen2-vl-7b-mlx/clean.json \
        --system_prompt prompt.txt \
        --corruption ""
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add this script's own directory first so `utils` resolves to inference/utils.py
# rather than any site-package named `inference` (e.g. roboflow inference-sdk).
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from utils import replace_system_prompt


def _load_checkpoint(ckpt_path: str) -> tuple:
    """Load a JSONL checkpoint and return (completed_results, completed_keys)."""
    results, keys = [], set()
    if not os.path.exists(ckpt_path):
        return results, keys
    with open(ckpt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                results.append(item)
                keys.add((item["scene_token"], item["frame_token"], item["question"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return results, keys


def parse_arguments():
    parser = argparse.ArgumentParser(description='VLM Inference via MLX-VLM (Apple Silicon)')
    parser.add_argument('--model', type=str, required=True, help='MLX-VLM model id or local path')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file (written directly, no temp folder)')
    parser.add_argument('--system_prompt', type=str, required=True,
                        help='System prompt file')
    parser.add_argument('--corruption', type=str, default='',
                        help='Corruption type')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run on the first N entries (for smoke testing)')
    return parser.parse_args()


def load_images(sample_filenames, corruption):
    image_paths = [p for p in sample_filenames.values() if p is not None]
    images = []
    for filename in image_paths:
        img_path = filename
        if corruption and len(corruption) > 1 and corruption != 'NoImage':
            img_path = img_path.replace('nuscenes/samples', f'corruption/{corruption}')
        if corruption == 'NoImage':
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image: {img_path}, error: {e}")
                exit(1)
        images.append(img)
    return image_paths, images


def main():
    args = parse_arguments()

    with open(args.system_prompt, 'r') as f:
        base_system_prompt = f.read()

    with open(args.data, 'r') as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]

    print(f"Loading {args.model} via MLX-VLM...")
    model, processor = load(args.model)
    config = load_config(args.model)

    # Write checkpoints to /tmp to avoid OneDrive sync timeouts on flush()
    _ckpt_name = os.path.basename(args.output) + ".ckpt.jsonl"
    ckpt_path = os.path.join("/tmp", _ckpt_name)
    results, done_keys = _load_checkpoint(ckpt_path)
    if done_keys:
        print(f"Resuming: {len(done_keys)} entries already done, {len(data) - len(done_keys)} remaining")
    ckpt_f = open(ckpt_path, "a")

    for entry in tqdm(data, desc=f"corruption={args.corruption or 'clean'}"):
        entry_key = (entry["scene_token"], entry["frame_token"], entry["question"])
        if entry_key in done_keys:
            continue

        image_paths, images = load_images(entry['image_path'], args.corruption)
        system_prompt = replace_system_prompt(base_system_prompt, image_paths)

        full_prompt = system_prompt + "\n" + entry['question']
        formatted_prompt = apply_chat_template(
            processor, config, full_prompt, num_images=len(images)
        )

        try:
            output = generate(
                model,
                processor,
                formatted_prompt,
                image=images,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=False,
            )
            pred = getattr(output, 'text', output)
        except Exception as e:
            print(f"\nWARN: generate() failed ({type(e).__name__}), skipping entry. Error: {e}")
            continue

        result = dict(entry)
        result['prompts'] = full_prompt
        result['pred'] = pred
        results.append(result)
        ckpt_f.write(json.dumps(result) + "\n")
        ckpt_f.flush()

    ckpt_f.close()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == '__main__':
    main()
