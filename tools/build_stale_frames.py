"""
Build a "stale camera" condition: for every DriveBench frame, substitute the
real keyframe images from k keyframes earlier (nuScenes keyframes are 2 Hz, so
k=1 is 0.5 s stale, k=2 is 1.0 s, k=4 is 2.0 s).

Files are written under the *target* frame's filename so the inference
loader's `nuscenes/samples -> corruption/<cond>` path swap finds them:

    data/corruption/Stale_<X>s/<CAM>/<target basename>

At the start of a scene there are fewer than k earlier keyframes; the earliest
available one is used, and the frame is marked truncated in the manifest
(with its actual staleness in seconds).

Usage:
    python tools/build_stale_frames.py --k 1   # -> Stale_0.5s
    python tools/build_stale_frames.py --k 2   # -> Stale_1.0s
    python tools/build_stale_frames.py --k 4   # -> Stale_2.0s
"""

import os
import sys
import glob
import json
import shutil
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.fetch_nuscenes_temporal_neighbors import CAMERAS, index_sample_data


def walk_back_samples(samples: dict, token: str, k: int) -> tuple:
    """Return (sample_token, hops) after walking `prev` up to k times."""
    hops = 0
    while hops < k and samples[token]["prev"]:
        token = samples[token]["prev"]
        hops += 1
    return token, hops


def find_file(rel_path: str, roots: list) -> str:
    for root in roots:
        path = os.path.join(root, rel_path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{rel_path} not found under any of {roots}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, required=True,
                        help="Keyframes back (1 keyframe = 0.5 s)")
    parser.add_argument("--meta-dir", default="data/nuscenes/v1.0-trainval_meta/v1.0-trainval")
    parser.add_argument("--nuscenes-root", nargs="+",
                        default=sorted(glob.glob("data/nuscenes/v1.0-trainval*_blobs")))
    parser.add_argument("--data", default="data/drivebench-test-final.json")
    parser.add_argument("--dest-root", default="data/corruption")
    parser.add_argument("--manifest-dir", default="results/stage1/manifests")
    args = parser.parse_args()

    name = f"Stale_{0.5 * args.k:.1f}s"
    dest = os.path.join(args.dest_root, name)

    samples = {s["token"]: s for s in
               json.load(open(os.path.join(args.meta_dir, "sample.json")))}
    by_token, by_sample_and_channel = index_sample_data(args.meta_dir)

    # frame_token -> {cam: target basename}
    targets = {}
    for entry in json.load(open(args.data)):
        for cam, path in entry["image_path"].items():
            if path:
                targets.setdefault(entry["frame_token"], {})[cam] = os.path.basename(path)

    manifest, n_truncated, written = {}, 0, 0
    for frame_token, cams in sorted(targets.items()):
        stale_token, hops = walk_back_samples(samples, frame_token, args.k)
        truncated = hops < args.k
        n_truncated += truncated
        manifest[frame_token] = {}
        for cam in CAMERAS:
            target_rec = by_token[by_sample_and_channel[(frame_token, cam)]]
            stale_rec = by_token[by_sample_and_channel[(stale_token, cam)]]
            dt = (target_rec["timestamp"] - stale_rec["timestamp"]) / 1e6
            out_name = cams.get(cam, os.path.basename(target_rec["filename"]))
            out_dir = os.path.join(dest, cam)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(find_file(stale_rec["filename"], args.nuscenes_root),
                            os.path.join(out_dir, out_name))
            written += 1
            manifest[frame_token][cam] = {"src": stale_rec["filename"],
                                          "dt_s": round(dt, 3),
                                          "hops": hops,
                                          "truncated": truncated}

    os.makedirs(args.manifest_dir, exist_ok=True)
    manifest_path = os.path.join(args.manifest_dir, f"{name}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    shutil.copyfile(manifest_path, os.path.join(dest, "manifest.json"))

    dts = [c["dt_s"] for cams in manifest.values() for c in cams.values()]
    print(f"{name}: wrote {written} images for {len(manifest)} frames to {dest}")
    print(f"  truncated at scene start: {n_truncated}/{len(manifest)} frames")
    print(f"  staleness (s): min {min(dts):.2f}  mean {sum(dts) / len(dts):.2f}  max {max(dts):.2f}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
