"""
Two recovery strategies for reconstructing a DriveBench NoImage/FrameLost
frame from its real temporal neighbors (see
tools/fetch_nuscenes_temporal_neighbors.py for how those neighbors are
resolved, and claude/TEMPORAL_RECOVERY_RESEARCH.md for the rationale).

Both strategies implement the same interface -- given a manifest entry for
one (frame, camera) and the directory the neighbor images were copied into,
produce a recovered PIL.Image -- so `build_recovered_dataset` below can swap
between them without touching the rest of the pipeline.

Build and validate `nearest_sweep_substitution` FIRST: it needs no model and
is the empirical floor the interpolation approach has to beat. Only reach
for `film_interpolate` once the baseline shows the hypothesis has legs.
"""

import argparse
import json
import os
from typing import Optional

from PIL import Image


def _nearest_pair(neighbors: list, timestamp: int):
    """Pick the single nearest neighbor overall, and the nearest bracketing
    pair (one prev, one next) if both sides are available -- prev/next are
    not always equidistant in real sweep data, so this picks by actual
    timestamp distance rather than assuming symmetry.
    """
    if not neighbors:
        return None, None, None

    nearest = min(neighbors, key=lambda n: abs(n["timestamp"] - timestamp))

    prevs = [n for n in neighbors if n["side"] == "prev"]
    nexts = [n for n in neighbors if n["side"] == "next"]
    nearest_prev = max(prevs, key=lambda n: n["timestamp"]) if prevs else None
    nearest_next = min(nexts, key=lambda n: n["timestamp"]) if nexts else None

    return nearest, nearest_prev, nearest_next


def nearest_sweep_substitution(manifest_entry: dict, neighbors_dir: str) -> Optional[Image.Image]:
    """Baseline, zero-ML recovery: just use the single closest-in-time real
    frame as the reconstruction. Cheap, fast, no training or GPU beyond what
    inference already needs -- the right first experiment to run before
    investing in an interpolation model.
    """
    timestamp = manifest_entry["timestamp"]
    nearest, _, _ = _nearest_pair(manifest_entry["neighbors"], timestamp)
    if nearest is None:
        return None
    path = os.path.join(neighbors_dir, nearest["filename"])
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")


def film_interpolate(manifest_entry: dict, neighbors_dir: str, film_model=None) -> Optional[Image.Image]:
    """Interpolation-based recovery using a pretrained FILM checkpoint
    (arXiv:2202.04901 -- chosen over AMT for its large-motion robustness,
    see claude/TEMPORAL_RECOVERY_RESEARCH.md section 2). Synthesizes the
    frame at the missing timestamp from the nearest bracketing real prev/next
    frames.

    NOT wired up to a concrete checkpoint yet -- no interpolation model is
    installed in either conda env (verified in claude/PLAN.md). Fill in
    `film_model` with a loaded FILM inference wrapper (e.g. following
    https://github.com/google-research/frame-interpolation, or a maintained
    PyPI port) once the nearest_sweep_substitution baseline has shown the
    hypothesis is worth the extra investment.
    """
    if film_model is None:
        raise NotImplementedError(
            "film_interpolate needs a loaded FILM model passed as `film_model`. "
            "No interpolation model is installed yet -- see the docstring above "
            "and claude/TEMPORAL_RECOVERY_RESEARCH.md section 2."
        )

    timestamp = manifest_entry["timestamp"]
    _, prev_n, next_n = _nearest_pair(manifest_entry["neighbors"], timestamp)
    if prev_n is None or next_n is None:
        # Can't bracket the missing frame -- fall back to nearest-neighbor
        # substitution rather than fail outright.
        return nearest_sweep_substitution(manifest_entry, neighbors_dir)

    prev_path = os.path.join(neighbors_dir, prev_n["filename"])
    next_path = os.path.join(neighbors_dir, next_n["filename"])
    if not (os.path.exists(prev_path) and os.path.exists(next_path)):
        return None

    prev_img = Image.open(prev_path).convert("RGB")
    next_img = Image.open(next_path).convert("RGB")
    span = next_n["timestamp"] - prev_n["timestamp"]
    t = (timestamp - prev_n["timestamp"]) / span if span else 0.5

    return film_model.interpolate(prev_img, next_img, t)


STRATEGIES = {
    "nearest": nearest_sweep_substitution,
    "film": film_interpolate,
}


def build_recovered_dataset(manifest_path: str, neighbors_dir: str, data_dir: str,
                             dest: str, strategy: str = "nearest"):
    """Populate data/corruption/Recovered/<CAM>/<filename> for every frame in
    the manifest, using the chosen strategy. Once this directory is
    populated, no inference code changes are needed -- passing
    `--corruption Recovered` to inference/llava1.5.py reads straight from it
    via the existing generic corruption-swap logic (see
    inference/llava1.5.py:84-87 and claude/TEMPORAL_RECOVERY_RESEARCH.md).
    Output filenames match the originals in data/nuscenes/samples/<CAM>/ so
    the path-substitution logic finds them unmodified.
    """
    recover_fn = STRATEGIES[strategy]
    manifest = json.load(open(manifest_path))

    # Recovered images must be saved under the *original* keyframe filename
    # (not the neighbor's), since that's the name inference/llava1.5.py looks
    # up after its path-substitution. Get that mapping from the QA data.
    frame_to_cam_filename = {}
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
    for frame_token, per_cam in manifest.items():
        for cam, manifest_entry in per_cam.items():
            out_filename = frame_to_cam_filename.get(frame_token, {}).get(cam)
            if not out_filename:
                skipped.append((frame_token, cam, "no matching QA image_path entry"))
                continue
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                         help="Path to neighbor_manifest.json from tools/fetch_nuscenes_temporal_neighbors.py")
    parser.add_argument("--neighbors-dir", required=True,
                         help="Directory the neighbor images were copied into (the --dest of that script)")
    parser.add_argument("--data-dir", default="data",
                         help="Directory with DriveBench QA JSON files (default: data)")
    parser.add_argument("--dest", default="data/corruption/Recovered",
                         help="Output directory (default: data/corruption/Recovered, so it plugs into the existing corruption-swap mechanism)")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="nearest",
                         help="Recovery strategy (default: nearest, the zero-ML baseline)")
    args = parser.parse_args()

    written, skipped = build_recovered_dataset(
        args.manifest, args.neighbors_dir, args.data_dir, args.dest, args.strategy)

    print(f"Wrote {written} recovered images to {args.dest} using strategy='{args.strategy}'")
    if skipped:
        print(f"Skipped {len(skipped)} (frame, camera) pairs:")
        for frame_token, cam, reason in skipped[:10]:
            print(f"  {frame_token} / {cam}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
