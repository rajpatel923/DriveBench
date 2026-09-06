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
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from inference.utils import replace_system_prompt


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

    results = []
    for entry in tqdm(data, desc=f"corruption={args.corruption or 'clean'}"):
        image_paths, images = load_images(entry['image_path'], args.corruption)
        system_prompt = replace_system_prompt(base_system_prompt, image_paths)

        full_prompt = system_prompt + "\n" + entry['question']
        formatted_prompt = apply_chat_template(
            processor, config, full_prompt, num_images=len(images)
        )

        output = generate(
            model,
            processor,
            formatted_prompt,
            image=images,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=False,
        )

        result = dict(entry)
        result['prompts'] = full_prompt
        # mlx_vlm.generate returns a GenerationResult with a .text attribute in
        # recent versions; fall back to str() if a plain string is returned instead.
        result['pred'] = getattr(output, 'text', output)
        results.append(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == '__main__':
    main()
