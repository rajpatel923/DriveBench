"""
Resolve and (optionally) copy the real nuScenes camera frames temporally
adjacent to each DriveBench test frame, for use as input to a temporal frame
-recovery method (see claude/TEMPORAL_RECOVERY_RESEARCH.md).

Scope: only the two true missing-frame conditions (NoImage, FrameLost) need
this — the other 14 corruption types already have the clean original frame
on disk under data/nuscenes/samples/, so they don't need temporal recovery.

DriveBench's `frame_token` is the nuScenes `sample_token` (verified directly:
resolving a frame_token from data/drivebench-test-final.json against
sample.json returns an exact match). Each nuScenes sample_data record forms
its own prev/next linked list per (scene, camera channel) at the full sensor
rate (~12Hz for cameras), which is denser and better suited to interpolation
than walking sample-level (2Hz keyframe) prev/next. This script walks that
sample_data-level chain `--steps` hops in each direction per camera.

Two modes:
  --manifest-only   Resolve neighbor filenames and write a JSON manifest.
                     Works with just the metadata dir (already present in
                     this repo's setup) -- no nuScenes image download needed.
  (default)          Also copy the resolved neighbor images out of
                     <nuscenes_root>/sweeps (and /samples, for keyframe
                     neighbors) into --dest, mirroring
                     tools/fetch_nuscenes_subset.py's copy pattern.
                     Requires the nuScenes sweep blob(s) to be downloaded and
                     extracted to <nuscenes_root> first -- this script does
                     not download anything itself.

Usage:
    # Step 1 (no download needed yet): see what's needed.
    python tools/fetch_nuscenes_temporal_neighbors.py --meta-dir /path/to/v1.0-trainval \
        --manifest-only --dest data/nuscenes/temporal_neighbors

    # Step 2 (after downloading+extracting the nuScenes sweep blob(s)):
    python tools/fetch_nuscenes_temporal_neighbors.py --meta-dir /path/to/v1.0-trainval \
        --nuscenes-root /path/to/extracted/blob --dest data/nuscenes/temporal_neighbors
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def load_frame_tokens(data_dir: str) -> set:
    """Collect the unique frame_tokens (== nuScenes sample_tokens) that
    DriveBench's NoImage/FrameLost conditions need recovery for -- i.e. every
    frame in the test set, since those two corruptions blank out every frame.
    """
    frame_tokens = set()
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            entries = json.load(open(path))
        except json.JSONDecodeError:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and "frame_token" in entry:
                frame_tokens.add(entry["frame_token"])
    return frame_tokens


def index_sample_data(meta_dir: str):
    """Build lookup structures from nuScenes sample_data.json:
      - by_token: token -> record
      - by_sample_and_channel: (sample_token, channel) -> token, for the
        keyframe sample_data record of that (sample, camera).
    """
    sample_data = json.load(open(os.path.join(meta_dir, "sample_data.json")))
    sensor = json.load(open(os.path.join(meta_dir, "sensor.json")))
    calibrated_sensor = json.load(open(os.path.join(meta_dir, "calibrated_sensor.json")))

    # calibrated_sensor -> sensor channel name
    cs_to_sensor_token = {cs["token"]: cs["sensor_token"] for cs in calibrated_sensor}
    sensor_token_to_channel = {s["token"]: s["channel"] for s in sensor}

    by_token = {}
    by_sample_and_channel = {}
    for rec in sample_data:
        by_token[rec["token"]] = rec
        sensor_token = cs_to_sensor_token.get(rec["calibrated_sensor_token"])
        channel = sensor_token_to_channel.get(sensor_token)
        rec["_channel"] = channel
        if rec.get("is_key_frame") and channel in CAMERAS:
            by_sample_and_channel[(rec["sample_token"], channel)] = rec["token"]

    return by_token, by_sample_and_channel


def walk_neighbors(by_token: dict, start_token: str, steps: int, direction: str):
    """Walk `direction` ('prev' or 'next') up to `steps` hops in the
    sample_data-level chain for one camera channel, returning the records
    encountered (nearest first). Stops early if the chain runs out.
    """
    out = []
    token = by_token[start_token].get(direction)
    for _ in range(steps):
        if not token or token not in by_token:
            break
        rec = by_token[token]
        out.append(rec)
        token = rec.get(direction)
    return out


def resolve_neighbors(frame_tokens: set, meta_dir: str, steps: int):
    """For every frame_token, resolve up to `steps` real neighbor frames on
    each side, per camera. Returns
    {frame_token: {camera: {"timestamp": int, "neighbors": [{"filename", "timestamp", "side"}...]}}}
    (neighbors ordered nearest-first within each side), plus a flat set of
    all needed filenames (for the copy step). Including timestamps lets a
    recovery method pick the true nearest neighbor (prev vs. next aren't
    always equidistant) and compute an interpolation fraction for methods
    like FILM.
    """
    by_token, by_sample_and_channel = index_sample_data(meta_dir)

    manifest = {}
    needed_filenames = set()
    missing = []

    for frame_token in sorted(frame_tokens):
        manifest[frame_token] = {}
        for cam in CAMERAS:
            start = by_sample_and_channel.get((frame_token, cam))
            if start is None:
                missing.append((frame_token, cam))
                continue
            current = by_token[start]
            prev_recs = walk_neighbors(by_token, start, steps, "prev")
            next_recs = walk_neighbors(by_token, start, steps, "next")
            neighbors = (
                [{"filename": r["filename"], "timestamp": r["timestamp"], "side": "prev"} for r in prev_recs]
                + [{"filename": r["filename"], "timestamp": r["timestamp"], "side": "next"} for r in next_recs]
            )
            manifest[frame_token][cam] = {
                "timestamp": current["timestamp"],
                "neighbors": neighbors,
            }
            needed_filenames.update(n["filename"] for n in neighbors)

    return manifest, needed_filenames, missing


def copy_images(nuscenes_root: str, needed_filenames: set, dest: str):
    """Copy resolved neighbor images out of the extracted nuScenes blob.
    nuScenes `sample_data.filename` is already relative to the blob root
    (e.g. 'sweeps/CAM_FRONT/xyz.jpg' or 'samples/CAM_FRONT/xyz.jpg'), so this
    just joins it against nuscenes_root and mirrors the same relative path
    under dest.
    """
    found, missing = 0, []
    for rel_path in sorted(needed_filenames):
        src = os.path.join(nuscenes_root, rel_path)
        dst = os.path.join(dest, rel_path)
        if not os.path.exists(src):
            missing.append(rel_path)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        found += 1
    return found, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-dir", required=True,
                         help="Path to nuScenes v1.0-trainval metadata dir (contains sample_data.json, sensor.json, calibrated_sensor.json)")
    parser.add_argument("--data-dir", default="data",
                         help="Directory with DriveBench QA JSON files (default: data)")
    parser.add_argument("--dest", required=True,
                         help="Destination directory for resolved neighbor images / manifest")
    parser.add_argument("--nuscenes-root", default=None,
                         help="Path to extracted nuScenes blob containing samples/ and sweeps/ (required unless --manifest-only)")
    parser.add_argument("--steps", type=int, default=2,
                         help="How many sample_data hops to resolve on each side of the missing frame (default: 2, i.e. up to ~166ms away at 12Hz)")
    parser.add_argument("--manifest-only", action="store_true",
                         help="Only resolve filenames and write the manifest; don't copy images (no nuscenes-root needed)")
    args = parser.parse_args()

    if not args.manifest_only and not args.nuscenes_root:
        parser.error("--nuscenes-root is required unless --manifest-only is set")

    frame_tokens = load_frame_tokens(args.data_dir)
    print(f"Resolving temporal neighbors for {len(frame_tokens)} frames x {len(CAMERAS)} cameras "
          f"(+/-{args.steps} sample_data steps each side)...")

    manifest, needed_filenames, missing = resolve_neighbors(frame_tokens, args.meta_dir, args.steps)

    os.makedirs(args.dest, exist_ok=True)
    manifest_path = os.path.join(args.dest, "neighbor_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Need {len(needed_filenames)} unique neighbor images "
          f"(manifest written to {manifest_path})")
    if missing:
        print(f"WARNING: {len(missing)} (frame_token, camera) pairs had no matching "
              f"keyframe sample_data entry in the metadata -- check --meta-dir covers "
              f"the right scenes:")
        for frame_token, cam in missing[:10]:
            print(f"  {frame_token} / {cam}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if args.manifest_only:
        print("--manifest-only set: not copying images. Re-run with --nuscenes-root "
              "once the corresponding nuScenes sweep blob(s) are downloaded and extracted.")
        return

    found, copy_missing = copy_images(args.nuscenes_root, needed_filenames, args.dest)
    print(f"Copied {found} images into {args.dest}")
    if copy_missing:
        print(f"WARNING: {len(copy_missing)} images were not found under {args.nuscenes_root}.")
        print("This nuScenes blob may not include sweeps, or may not cover all needed scenes; "
              "you may need to download additional blob(s) from nuscenes.org.")


if __name__ == "__main__":
    main()
