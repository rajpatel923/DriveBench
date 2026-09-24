"""
Qwen2-VL variant of inference/llava1.5.py, adapted to test whether DriveBench's
missing-frame hallucination finding generalizes beyond LLaVA.

Unlike LLaVA's raw string prompt with hand-rolled "<image>" placeholders, Qwen2-VL
needs its own chat template and vision tokens, so this uses vLLM's LLM.chat() with
PIL image content parts instead of LLM.generate() with a manually built prompt string.
"""

import os
import warnings
import argparse
from typing import Any, Dict
from packaging.version import Version

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

from inference.utils import replace_system_prompt, load_json_files, save_json
from inference.conditions import load_images, load_sensor_text, build_user_text, resolve_image_note


assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"


def parse_arguments():
    parser = argparse.ArgumentParser(description='VLM Multi-GPU Inference (Qwen2-VL)')
    parser.add_argument('--model', type=str, required=True, help='VLMs')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--system_prompt', type=str, required=True,
                        help='System prompt file')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--max_model_len', type=int, default=16384,
                        help='Maximum model length. 8192 is too small for a full '
                             '6-camera prompt at nuScenes native resolution '
                             '(~11.2k image+text tokens observed) -- raised to 16384.')
    parser.add_argument('--num_images_per_prompt', type=int, default=6,
                        help='Maximum number of images per prompt')
    parser.add_argument('--corruption', type=str, default='',
                        help='Corruption type')
    parser.add_argument('--sensor_text', type=str, default='',
                        help='Optional {frame_token: text} JSON; its report is '
                             'prepended to every question (see inference/conditions.py)')
    parser.add_argument('--image_note', type=str, default='auto',
                        help="Fixed line before every question saying how non-camera images "
                             "are drawn: 'auto' (IMAGE_NOTES for --corruption), 'none', or text")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling (0.0 = greedy)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p for sampling')
    parser.add_argument('--seed', type=int, default=0,
                        help='Sampling seed')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run on the first N entries (for smoke testing)')
    return parser.parse_args()


class LLMPredictor:
    def __init__(self, model_name, system_prompt, sampling_params,
                 num_images_per_prompt, max_model_len, tensor_parallel_size, corruption,
                 sensor_text_path, image_note=''):
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": num_images_per_prompt},
            tensor_parallel_size=tensor_parallel_size
        )
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params
        self.corruption = corruption
        self.sensor_text = load_sensor_text(sensor_text_path)
        self.image_note = resolve_image_note(image_note, corruption)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        questions = batch['question']
        filenames = batch['image_path']
        frame_tokens = batch['frame_token']
        batch_size = len(questions)

        conversations = []
        system_prompts = [self.system_prompt] * batch_size
        user_texts = []

        for idx, sample_filenames in enumerate(filenames):
            image_paths, images = load_images(sample_filenames, self.corruption)
            system_prompts[idx] = replace_system_prompt(system_prompts[idx], image_paths)
            user_text = build_user_text(questions[idx], frame_tokens[idx], self.sensor_text,
                                        self.image_note)
            user_texts.append(user_text)

            content = [{"type": "image_pil", "image_pil": img} for img in images]
            content.append({"type": "text", "text": user_text})

            conversations.append([
                {"role": "system", "content": system_prompts[idx]},
                {"role": "user", "content": content},
            ])

        outputs = self.llm.chat(
            conversations,
            self.sampling_params,
            use_tqdm=False
        )

        batch['prompts'] = system_prompts
        batch['user_text'] = user_texts
        if outputs is None:
            warnings.warn("[Warning]: outputs is None")
            batch['pred'] = None
            return batch

        generated_text = []
        for output in outputs:
            generated_text.append(output.outputs[0].text)
        batch['pred'] = generated_text
        return batch


def main():
    args = parse_arguments()

    with open(args.system_prompt, 'r') as f:
        system_prompt = f.read()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_tokens
    )

    tensor_parallel_size = 1
    num_instances = args.num_processes

    ds = ray.data.read_json(args.data)
    if args.limit:
        ds = ds.limit(args.limit)

    def scheduling_strategy_fn():
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        resources_kwarg["num_gpus"] = 1
    else:
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_args=(
            args.model,
            system_prompt,
            sampling_params,
            args.num_images_per_prompt,
            args.max_model_len,
            tensor_parallel_size,
            args.corruption,
            args.sensor_text,
            args.image_note
        ),
        concurrency=num_instances,
        batch_size=1,
        **resources_kwarg,
    )

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    ds.write_json(args.output)

    res = load_json_files(args.output)
    save_json(res, args.output + ".json")
    os.system(f"rm -r {args.output}")


if __name__ == '__main__':
    main()
