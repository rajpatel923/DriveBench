"""
Preflight for script/stage1/run_all.sh: catch missing files before burning
GPU hours. For every run it checks that
  - every image the dataset references resolves for that condition,
  - the sensor-text JSON (if any) covers every frame token,
  - load_images() returns 672x378 images (checked on a few questions),
and prints one example user_text. Exits non-zero on any failure.

Usage:
    python tools/preflight.py
    python tools/preflight.py --runs C1_blank W_oracle
"""

import os
import sys
import json
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from inference.conditions import (IMG_SIZE, resolve_image_path, load_images, load_sensor_text,
                                  build_user_text, resolve_image_note)

# Keep in sync with script/stage1/run_all.sh
RUNS = {
    "C0_clean": ("", None),
    "C1_blank": ("NoImage", None),
    "W_oracle": ("NoImage", "oracle.json"),
    "W_shuffled": ("NoImage", "oracle_shuffled.json"),
    "P_lidar": ("Recovered_LiDAR", None),
    "W_cluster": ("NoImage", "cluster.json"),
    "C0b_clean_rerun": ("", None),
    "S05": ("Stale_0.5s", None),
    "S10": ("Stale_1.0s", None),
    "S20": ("Stale_2.0s", None),
    "S10_cluster": ("Stale_1.0s", "cluster.json"),
    "P_lidar2": ("Recovered_LiDAR_Radar", None),
    "E_blank_ego": ("NoImage", "ego.json"),
    "W_oracle_ego": ("NoImage", "oracle_ego.json"),
    "W_cluster_ego": ("NoImage", "cluster_ego.json"),
    "P_lidar2_ego": ("Recovered_LiDAR_Radar", "ego.json"),
}


def check_run(label, condition, text_file, data, text_dir, n_load):
    errors = []
    missing = set()
    for q in data:
        for p in q["image_path"].values():
            r = resolve_image_path(p, condition)
            if r is not None and not os.path.exists(r):
                missing.add(r)
    if missing:
        errors.append(f"{len(missing)} missing images, e.g. {sorted(missing)[0]}")

    sensor_text = None
    if text_file:
        path = os.path.join(text_dir, text_file)
        if not os.path.exists(path):
            errors.append(f"sensor text not found: {path}")
        else:
            sensor_text = load_sensor_text(path)
            uncovered = {q["frame_token"] for q in data} - set(sensor_text)
            if uncovered:
                errors.append(f"sensor text missing {len(uncovered)} frames")

    if not missing:
        for q in data[:n_load]:
            _, images = load_images(q["image_path"], condition)
            bad = [im.size for im in images if im.size != IMG_SIZE]
            if bad:
                errors.append(f"image size {bad[0]} != {IMG_SIZE}")
                break

    print(f"[{'FAIL' if errors else 'ok'}] {label} (corruption={condition or '<clean>'}, text={text_file})")
    for e in errors:
        print(f"    - {e}")
    if not errors:
        q = data[0]
        example = build_user_text(q["question"], q["frame_token"], sensor_text,
                                  resolve_image_note("auto", condition))
        lines = example.splitlines()
        shown = lines[:6] + (["    ..."] + lines[-3:] if len(lines) > 9 else lines[6:])
        print("    example user_text:\n      " + "\n      ".join(shown))
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/drivebench-mcq.json")
    parser.add_argument("--text-dir", default="data/sensor_text")
    parser.add_argument("--runs", nargs="*", default=list(RUNS))
    parser.add_argument("--n-load", type=int, default=5, help="questions per run to actually load")
    args = parser.parse_args()

    data = json.load(open(args.data))
    ok = True
    for label in args.runs:
        condition, text_file = RUNS[label]
        ok &= check_run(label, condition, text_file, data, args.text_dir, args.n_load)
    print("\nPreflight " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
