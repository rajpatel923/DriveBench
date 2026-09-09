"""
Three recovery strategies for reconstructing a DriveBench NoImage/FrameLost
frame from real temporal neighbors or cross-modal LiDAR data.

Strategies
----------
nearest  Zero-ML baseline: substitute the closest-in-time real sweep frame.
         No model required. Run this first — it is the empirical floor that
         rife must beat.

rife     Temporal interpolation via RIFE (ECCV 2022). Synthesises the missing
         frame from the bracketing prev/next sweep frames. Requires cloning
         github.com/hzwer/ECCV2022-RIFE to recovery/rife/ and passing
         --rife-weights to main().

lidar    Cross-modal fallback: projects the nuScenes LiDAR point cloud into
         camera image space and renders a depth-colourised image. Used when
         no temporal neighbours exist. Implemented in recovery/lidar_recovery.py.

Run order: nearest → rife → lidar (validate each before adding the next).
"""

import argparse
import json
import os
from typing import Optional

from PIL import Image


def _nearest_pair(neighbors: list, timestamp: int):
    """Return (nearest, nearest_prev, nearest_next) by actual timestamp distance."""
    if not neighbors:
        return None, None, None

    nearest = min(neighbors, key=lambda n: abs(n["timestamp"] - timestamp))

    prevs = [n for n in neighbors if n["side"] == "prev"]
    nexts = [n for n in neighbors if n["side"] == "next"]
    nearest_prev = max(prevs, key=lambda n: n["timestamp"]) if prevs else None
    nearest_next = min(nexts, key=lambda n: n["timestamp"]) if nexts else None

    return nearest, nearest_prev, nearest_next


def _load_bracketing_pair(manifest_entry: dict, neighbors_dir: str):
    """
    Load the nearest bracketing (prev, next) PIL images and compute the
    fractional interpolation timestep t ∈ (0, 1).
    Returns (prev_img, next_img, t) or (None, None, 0.5) on failure.
    """
    timestamp = manifest_entry["timestamp"]
    _, prev_n, next_n = _nearest_pair(manifest_entry["neighbors"], timestamp)

    if prev_n is None or next_n is None:
        return None, None, 0.5

    prev_path = os.path.join(neighbors_dir, prev_n["filename"])
    next_path = os.path.join(neighbors_dir, next_n["filename"])
    if not (os.path.exists(prev_path) and os.path.exists(next_path)):
        return None, None, 0.5

    prev_img = Image.open(prev_path).convert("RGB")
    next_img = Image.open(next_path).convert("RGB")
    span = next_n["timestamp"] - prev_n["timestamp"]
    t = (timestamp - prev_n["timestamp"]) / span if span else 0.5
    return prev_img, next_img, float(t)


def nearest_sweep_substitution(manifest_entry: dict, neighbors_dir: str) -> Optional[Image.Image]:
    """Zero-ML baseline: substitute the closest-in-time real sweep frame."""
    timestamp = manifest_entry["timestamp"]
    nearest, _, _ = _nearest_pair(manifest_entry["neighbors"], timestamp)
    if nearest is None:
        return None
    path = os.path.join(neighbors_dir, nearest["filename"])
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")


def rife_interpolate(manifest_entry: dict, neighbors_dir: str,
                     rife_model=None) -> Optional[Image.Image]:
    """
    Temporal interpolation via RIFE (Real-Time Intermediate Flow Estimation,
    ECCV 2022 — hzwer/ECCV2022-RIFE). Synthesises the missing keyframe from
    the nearest bracketing prev/next 12 Hz sweep frames. Falls back to
    nearest_sweep_substitution when only one side is available.

    rife_model: loaded RIFE Model instance (see _load_rife_model() and main()).
    """
    if rife_model is None:
        raise RuntimeError(
            "rife_interpolate requires a loaded RIFE model. "
            "Clone github.com/hzwer/ECCV2022-RIFE to recovery/rife/ and "
            "pass --rife-weights <path/to/train_log> to this script."
        )

    prev_img, next_img, t = _load_bracketing_pair(manifest_entry, neighbors_dir)
    if prev_img is None or next_img is None:
        return nearest_sweep_substitution(manifest_entry, neighbors_dir)

    import torch
    import torchvision.transforms.functional as TF

    device = next(rife_model.net.parameters()).device
    I0 = TF.to_tensor(prev_img).unsqueeze(0).to(device)
    I1 = TF.to_tensor(next_img).unsqueeze(0).to(device)

    with torch.no_grad():
        middle = rife_model.inference(I0, I1, timestep=t)

    return TF.to_pil_image(middle.squeeze(0).clamp(0, 1).cpu())


def _load_rife_model(weights_dir: str):
    """
    Load a RIFE HDv3 model from a cloned copy of hzwer/ECCV2022-RIFE at
    recovery/rife/. Weights live in the train_log/ subdirectory of that repo.

    Returns: loaded, eval-mode RIFE Model instance.
    """
    import sys
    import torch

    rife_repo = os.path.join(os.path.dirname(__file__), "rife")
    if not os.path.isdir(rife_repo):
        raise RuntimeError(
            f"RIFE repo not found at {rife_repo}.\n"
            "  git clone https://github.com/hzwer/ECCV2022-RIFE recovery/rife\n"
            "  # then download HDv3 weights into recovery/rife/train_log/"
        )
    sys.path.insert(0, rife_repo)

    from model.RIFE_HDv3 import Model  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.load_model(weights_dir, -1)
    model.eval()
    model.device()  # moves net to device
    print(f"RIFE loaded from {weights_dir} on {device}")
    return model


STRATEGIES = {
    "nearest": nearest_sweep_substitution,
    "rife": rife_interpolate,
}


def build_recovered_dataset(manifest_path: str, neighbors_dir: str, data_dir: str,
                             dest: str, strategy: str = "nearest", model=None,
                             limit: Optional[int] = None):
    """
    Populate dest/<CAM>/<filename> for every (frame, camera) pair in the
    manifest using the chosen strategy.

    Passing --corruption <dest_basename> to any inference script will then
    read from this directory via the generic img_path.replace() swap without
    any inference-code changes.

    Output filenames match the original keyframe basenames so the path-swap
    logic resolves them correctly.
    """
    recover_fn = STRATEGIES[strategy]
    manifest = json.load(open(manifest_path))

    frame_to_cam_filename: dict = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        entries = json.load(open(os.path.join(data_dir, fname)))
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "frame_token" not in entry:
                continue
            frame_to_cam_filename.setdefault(entry["frame_token"], {})
            for cam, path in entry.get("image_path", {}).items():
                if path:
                    frame_to_cam_filename[entry["frame_token"]][cam] = os.path.basename(path)

    written, skipped = 0, []
    items = list(manifest.items())
    if limit:
        items = items[:limit]

    for frame_token, per_cam in items:
        for cam, manifest_entry in per_cam.items():
            out_filename = frame_to_cam_filename.get(frame_token, {}).get(cam)
            if not out_filename:
                skipped.append((frame_token, cam, "no matching QA image_path entry"))
                continue

            if strategy == "rife":
                img = recover_fn(manifest_entry, neighbors_dir, rife_model=model)
            else:
                img = recover_fn(manifest_entry, neighbors_dir)

            if img is None:
                skipped.append((frame_token, cam, "neighbor image(s) not found locally"))
                continue

            out_dir = os.path.join(dest, cam)
            os.makedirs(out_dir, exist_ok=True)
            img.resize((224, 224)).save(os.path.join(out_dir, out_filename))
            written += 1

    return written, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                        help="Path to neighbor_manifest.json (from tools/fetch_nuscenes_temporal_neighbors.py)")
    parser.add_argument("--neighbors-dir", required=True,
                        help="Directory the neighbor images were copied into")
    parser.add_argument("--data-dir", default="data",
                        help="Directory with DriveBench QA JSON files (default: data)")
    parser.add_argument("--dest", required=True,
                        help="Output directory, e.g. data/corruption/Recovered_RIFE")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="nearest")
    parser.add_argument("--rife-weights", default="recovery/rife/train_log",
                        help="RIFE weights directory (only for --strategy rife)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N frames (smoke test)")
    args = parser.parse_args()

    model = None
    if args.strategy == "rife":
        model = _load_rife_model(args.rife_weights)

    written, skipped = build_recovered_dataset(
        args.manifest, args.neighbors_dir, args.data_dir, args.dest,
        args.strategy, model=model, limit=args.limit,
    )

    print(f"Wrote {written} recovered images to {args.dest} (strategy='{args.strategy}')")
    if skipped:
        print(f"Skipped {len(skipped)} (frame, camera) pairs:")
        for frame_token, cam, reason in skipped[:10]:
            print(f"  {frame_token} / {cam}: {reason}")
        if len(skipped) > 10:
            print(f"  … and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
