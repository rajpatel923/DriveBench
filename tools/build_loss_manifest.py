"""
Build a per-method source-eligibility manifest for frame-loss experiments.

For each (frame_token, camera) pair marked as lost, this script records
which neighbor frames each recovery method is *permitted* to use.  This
prevents a method from accidentally drawing on a frame that is also lost
— critical for consecutive-loss and multi-camera-loss scenarios.

Method          Permitted sources
-----------     -------------------------------------------------------
previous        Past frames only (side == "prev"); real-time compatible.
nearest         Any neighbor (past or future); requires buffering.
linear_blend    Must have ONE prev AND ONE next; marks "insufficient"
rife            Same requirement as linear_blend.

Usage:
    # Inspect what would be lost (FrameLost corruption directory):
    python tools/build_loss_manifest.py \\
        --neighbor-manifest data/nuscenes/temporal_neighbors/neighbor_manifest.json \\
        --framelost-dir data/corruption/FrameLost \\
        --output data/loss_manifest.json

    # With an explicit lost-tokens file (one frame_token per line):
    python tools/build_loss_manifest.py \\
        --neighbor-manifest data/nuscenes/temporal_neighbors/neighbor_manifest.json \\
        --lost-tokens-file lost_tokens.txt \\
        --output data/loss_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def _collect_lost_tokens_from_dir(framelost_dir: str) -> set[str]:
    """
    Scan the FrameLost corruption directory to find which frame_tokens
    have at least one camera image present.  The filename stem encodes the
    token via the nuScenes naming convention.

    Falls back gracefully if the directory is missing.
    """
    tokens: set[str] = set()
    if not os.path.isdir(framelost_dir):
        return tokens
    for cam in os.listdir(framelost_dir):
        cam_dir = os.path.join(framelost_dir, cam)
        if not os.path.isdir(cam_dir):
            continue
        for fname in os.listdir(cam_dir):
            stem = os.path.splitext(fname)[0]
            tokens.add(stem)
    return tokens


def build_loss_manifest(
    neighbor_manifest: dict,
    lost_tokens: set[str],
) -> dict:
    """
    For each (frame_token, camera) in the neighbor manifest, produce a
    per-method eligibility record.

    Output format per entry:
    {
      "timestamp": <int>,
      "methods": {
        "previous":     {"eligible": [{"filename", "timestamp"}], "insufficient": false},
        "nearest":      {"eligible": [...], "insufficient": false},
        "linear_blend": {"eligible": [], "insufficient": true},  // if one side missing
        "rife":         {"eligible": [], "insufficient": true},
      },
      "consecutive_loss_neighbors": [filenames also marked as lost]
    }
    """
    manifest_out: dict = {}

    for frame_token, cameras in neighbor_manifest.items():
        is_lost = frame_token in lost_tokens
        entry_out: dict = {}

        for cam, entry in cameras.items():
            neighbors = entry.get("neighbors", [])
            ts = entry.get("timestamp", 0)

            # Identify which neighbors are ALSO lost (cannot be used as source)
            also_lost = [
                n for n in neighbors
                if os.path.splitext(os.path.basename(n["filename"]))[0] in lost_tokens
            ]
            also_lost_filenames = {n["filename"] for n in also_lost}

            available = [n for n in neighbors if n["filename"] not in also_lost_filenames]

            prevs = sorted(
                [n for n in available if n["side"] == "prev"],
                key=lambda n: n["timestamp"],
                reverse=True,
            )
            nexts = sorted(
                [n for n in available if n["side"] == "next"],
                key=lambda n: n["timestamp"],
            )
            all_avail = sorted(available, key=lambda n: abs(n["timestamp"] - ts))

            def _as_src(n: dict) -> dict:
                return {"filename": n["filename"], "timestamp": n["timestamp"], "side": n["side"]}

            entry_out[cam] = {
                "timestamp": ts,
                "is_lost": is_lost,
                "methods": {
                    "previous": {
                        "eligible": [_as_src(prevs[0])] if prevs else [],
                        "insufficient": len(prevs) == 0,
                    },
                    "nearest": {
                        "eligible": [_as_src(all_avail[0])] if all_avail else [],
                        "insufficient": len(all_avail) == 0,
                    },
                    "linear_blend": {
                        "eligible": (
                            [_as_src(prevs[0]), _as_src(nexts[0])]
                            if prevs and nexts else []
                        ),
                        "insufficient": not (prevs and nexts),
                    },
                    "rife": {
                        "eligible": (
                            [_as_src(prevs[0]), _as_src(nexts[0])]
                            if prevs and nexts else []
                        ),
                        "insufficient": not (prevs and nexts),
                    },
                },
                "blocked_neighbors": [n["filename"] for n in also_lost],
            }

        manifest_out[frame_token] = entry_out

    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--neighbor-manifest",
        required=True,
        help="Path to neighbor_manifest.json from tools/fetch_nuscenes_temporal_neighbors.py",
    )
    parser.add_argument(
        "--framelost-dir",
        default=None,
        help="data/corruption/FrameLost directory (used to detect lost frame tokens)",
    )
    parser.add_argument(
        "--lost-tokens-file",
        default=None,
        help="Plain-text file with one lost frame_token per line (alternative to --framelost-dir)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path, e.g. data/loss_manifest.json",
    )
    args = parser.parse_args()

    with open(args.neighbor_manifest) as f:
        neighbor_manifest = json.load(f)

    lost_tokens: set[str] = set()
    if args.lost_tokens_file:
        with open(args.lost_tokens_file) as f:
            lost_tokens = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(lost_tokens)} lost tokens from {args.lost_tokens_file}")
    elif args.framelost_dir:
        lost_tokens = _collect_lost_tokens_from_dir(args.framelost_dir)
        print(f"Detected {len(lost_tokens)} lost frame_tokens from {args.framelost_dir}")
    else:
        print("Warning: no --framelost-dir or --lost-tokens-file given; "
              "all frames treated as lost (full manifest).")
        lost_tokens = set(neighbor_manifest)

    manifest_out = build_loss_manifest(neighbor_manifest, lost_tokens)

    # Summary stats
    insufficient: dict[str, int] = defaultdict(int)
    blocked_total = 0
    for frame_token, cameras in manifest_out.items():
        for cam, entry in cameras.items():
            if not entry["is_lost"]:
                continue
            for method, info in entry["methods"].items():
                if info["insufficient"]:
                    insufficient[method] += 1
            blocked_total += len(entry["blocked_neighbors"])

    n_lost = sum(
        1 for ft, cams in manifest_out.items()
        for cam, e in cams.items()
        if e["is_lost"]
    )
    print(f"\nLoss manifest summary:")
    print(f"  Total (frame, camera) pairs marked lost: {n_lost}")
    print(f"  Neighbors blocked (also lost): {blocked_total}")
    for method in ["previous", "nearest", "linear_blend", "rife"]:
        n = insufficient.get(method, 0)
        print(f"  {method}: insufficient sources for {n}/{n_lost} lost pairs")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"\nWrote loss manifest to {args.output}")


if __name__ == "__main__":
    main()
