"""
InternVL2.5-8B inference for DriveBench.

Sequential single-machine runner (Mac MPS or CUDA GPU), following the same structure
as inference/qwen2vl_mlx.py.  InternVL is not supported by vLLM at time of writing, so
there is no Ray/vLLM variant — use this file on both Mac and GPU machines.

Output format matches inference/llava1.5.py and inference/qwen2vl_mlx.py so the same
evaluate/ scripts work unchanged.

Requirements:
    pip install transformers torch torchvision

Usage (Mac):
    python inference/internvl.py \\
        --model OpenGVLab/InternVL2_5-8B \\
        --data data/drivebench-test-final.json \\
        --output res/internvl2.5-8b/clean.json \\
        --system_prompt prompt.txt \\
        --corruption ""

Usage (GPU):
    Same command — device is auto-detected (CUDA > MPS > CPU).
"""

import os
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from inference.utils import replace_system_prompt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def pil_to_pixel_values(pil_img, input_size=448):
    """Return a (1, 3, H, W) tensor for one image (single-patch, fixed resolution)."""
    return build_transform(input_size)(pil_img).unsqueeze(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='InternVL2.5 Inference (Sequential)')
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL2_5-8B',
                        help='HuggingFace model ID or local path')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--system_prompt', type=str, required=True,
                        help='System prompt file')
    parser.add_argument('--corruption', type=str, default='',
                        help='Corruption type')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run the first N entries (smoke test)')
    return parser.parse_args()


def main():
    args = parse_arguments()

    if torch.cuda.is_available():
        device = "cuda"
        use_flash_attn = True
    elif torch.backends.mps.is_available():
        device = "mps"
        use_flash_attn = False
    else:
        device = "cpu"
        use_flash_attn = False

    dtype = torch.bfloat16

    with open(args.system_prompt, 'r') as f:
        base_system_prompt = f.read()
    with open(args.data, 'r') as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]

    print(f"Loading {args.model} on {device} (flash_attn={use_flash_attn})...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False
    )
    model_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=use_flash_attn,
    )
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(args.model, **model_kwargs).eval()
    else:
        model = AutoModel.from_pretrained(args.model, **model_kwargs).eval().to(device)

    generation_config = dict(
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=args.temperature,
    )

    results = []
    for entry in tqdm(data, desc=f"corruption={args.corruption or 'clean'}"):
        raw_paths = [p for p in entry['image_path'].values() if p is not None]
        system_prompt = replace_system_prompt(base_system_prompt, raw_paths)

        pil_images = []
        for img_path in raw_paths:
            if args.corruption and len(args.corruption) > 1 and args.corruption != 'NoImage':
                img_path = img_path.replace('nuscenes/samples', f'corruption/{args.corruption}')
            if args.corruption == 'NoImage':
                pil_images.append(Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8)))
            else:
                try:
                    pil_images.append(Image.open(img_path).convert('RGB'))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    exit(1)

        tensors = [pil_to_pixel_values(img).to(dtype).to(device) for img in pil_images]
        pixel_values = torch.cat(tensors, dim=0)          # (N_images, 3, 448, 448)
        num_patches_list = [1] * len(pil_images)          # one patch per image

        # InternVL expects one <image>\n per image in the question string.
        # Prepend the DriveBench system prompt as leading context in the user turn.
        image_tokens = '<image>\n' * len(pil_images)
        question = system_prompt + "\n" + image_tokens + entry['question']

        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

        result = dict(entry)
        result['prompts'] = question
        result['pred'] = response
        results.append(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == '__main__':
    main()
