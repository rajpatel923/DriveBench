"""
Copy only the nuScenes camera images DriveBench actually needs out of a
downloaded nuScenes v1.0-trainval extraction, instead of keeping the full
(much larger) sensor dataset.

Usage:
    python tools/fetch_nuscenes_subset.py <nuscenes_root> data/nuscenes/samples

<nuscenes_root> should be wherever you extracted the nuScenes blob(s) from
nuscenes.org, i.e. the directory that itself contains a `samples/` folder
with CAM_FRONT/, CAM_BACK/, etc. subfolders.
"""

import argparse
import json
import os
import shutil


def collect_needed_filenames(data_dir: str) -> set:
    needed = set()
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for path in entry.get("image_path", {}).values():
                if "nuscenes/samples" in path:
                    needed.add(os.path.basename(path))
    return needed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nuscenes_root", help="Path to extracted nuScenes download (contains samples/)")
    parser.add_argument("dest", help="Destination, e.g. data/nuscenes/samples")
    parser.add_argument("--data-dir", default="data", help="Directory with DriveBench QA JSON files")
    args = parser.parse_args()

    needed = collect_needed_filenames(args.data_dir)
    print(f"Need {len(needed)} unique images")

    src_samples = os.path.join(args.nuscenes_root, "samples")
    if not os.path.isdir(src_samples):
        raise SystemExit(f"No samples/ directory found under {args.nuscenes_root}")

    found = 0
    missing = []
    for cam in os.listdir(src_samples):
        cam_dir = os.path.join(src_samples, cam)
        if not os.path.isdir(cam_dir):
            continue
        dest_cam_dir = os.path.join(args.dest, cam)
        for fname in os.listdir(cam_dir):
            if fname in needed:
                os.makedirs(dest_cam_dir, exist_ok=True)
                shutil.copy(os.path.join(cam_dir, fname), os.path.join(dest_cam_dir, fname))
                found += 1

    print(f"Copied {found} images into {args.dest}")
    if found < len(needed):
        print(f"WARNING: {len(needed) - found} images were not found under {src_samples}.")
        print("This nuScenes blob may not cover all the scenes DriveBench needs;")
        print("you may need to download additional trainval blob(s) from nuscenes.org.")


if __name__ == "__main__":
    main()
