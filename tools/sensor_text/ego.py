"""
Ego-motion sensor text: the vehicle's own speed, turning rate and
acceleration from odometry (nuScenes ego poses), past-only.

Poses come from the LIDAR_TOP sample_data chain (20 Hz): the keyframe, the
sweep ~WINDOW s earlier, and the one ~2*WINDOW s earlier. Nothing after the
keyframe is read, so the text says what the car has been doing, not what it
does next (which is what the behavior MCQs ask). A frame too close to the
start of its scene gets "ego motion unknown".

Writes:
    data/sensor_text/ego.json          {frame: ego line}
    data/sensor_text/oracle_ego.json   ego line + oracle report
    data/sensor_text/cluster_ego.json  ego line + cluster report

Usage:
    python tools/sensor_text/ego.py
"""

import os
import sys
import json
import argparse
from typing import Dict, Optional

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.sensor_text.common import META_DIR, DATA, OUT_DIR, Meta, load_frames
from tools.sensor_text.oracle import yaw

WINDOW = 0.5            # s between the poses compared
MAX_WINDOW_ERROR = 0.1  # s; the sweep found must be this close to WINDOW
STOPPED_SPEED = 0.3     # m/s
STRAIGHT_RATE = 2.0     # deg/s
STEADY_ACCEL = 0.3      # m/s^2

COMPOSED = ("oracle", "cluster")


def pose_back(rec: dict, seconds: float, sd_by_token: Dict[str, dict]) -> Optional[dict]:
    """The earlier sweep in rec's chain closest to `seconds` before rec."""
    t0, best = rec["timestamp"], None
    while rec["prev"]:
        rec = sd_by_token[rec["prev"]]
        err = abs((t0 - rec["timestamp"]) / 1e6 - seconds)
        if best is not None and err > best[0]:
            break
        best = (err, rec)
    return best[1] if best and best[0] <= MAX_WINDOW_ERROR else None


def motion(a: dict, b: dict, poses: Dict[str, dict]):
    """(speed m/s, yaw rate deg/s +left) from sweep a to the later sweep b."""
    pa, pb = poses[a["ego_pose_token"]], poses[b["ego_pose_token"]]
    dt = (b["timestamp"] - a["timestamp"]) / 1e6
    speed = np.hypot(*(np.array(pb["translation"][:2]) - np.array(pa["translation"][:2]))) / dt
    d = (yaw(pb["rotation"]) - yaw(pa["rotation"]) + np.pi) % (2 * np.pi) - np.pi
    return float(speed), float(np.degrees(d) / dt), dt


def ego_state(meta: Meta, token: str, sd_by_token: Dict[str, dict]) -> Optional[dict]:
    now = meta.keyframe(token, "LIDAR_TOP")
    before = pose_back(now, WINDOW, sd_by_token)
    if before is None:
        return None
    speed, yaw_rate, dt = motion(before, now, meta.ego_poses)
    state = {"speed": speed, "yaw_rate": yaw_rate, "accel": None}
    earlier = pose_back(before, WINDOW, sd_by_token)
    if earlier is not None:
        prev_speed, _, prev_dt = motion(earlier, before, meta.ego_poses)
        state["accel"] = (speed - prev_speed) / ((dt + prev_dt) / 2)
    return state


def format_ego(state: Optional[dict]) -> str:
    head = f"Ego vehicle (odometry, past {WINDOW:.1f} s): "
    if state is None:
        return head + "ego motion unknown."
    if state["speed"] < STOPPED_SPEED:
        parts = ["stopped"]
    else:
        parts = [f"speed {state['speed']:.1f} m/s"]
        r = state["yaw_rate"]
        parts.append("going straight" if abs(r) < STRAIGHT_RATE else
                     f"turning {'left' if r > 0 else 'right'} ({abs(r):.0f} deg/s)")
    a = state["accel"]
    if a is not None:
        parts.append("steady speed" if abs(a) < STEADY_ACCEL else
                     f"{'accelerating' if a > 0 else 'slowing down'} ({abs(a):.1f} m/s^2)")
    return head + ", ".join(parts) + "."


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-dir", default=META_DIR)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    frames = load_frames(args.data)
    meta = Meta(args.meta_dir)
    sd_by_token = {r["token"]: r for recs in meta.sample_data.values() for r in recs}
    states = {tok: ego_state(meta, tok, sd_by_token) for tok in frames}
    ego = {tok: format_ego(s) for tok, s in states.items()}

    def dump(name, obj):
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(obj, f, indent=1)

    dump("ego", ego)
    dump("ego.states", states)
    for source in COMPOSED:
        reports = json.load(open(os.path.join(args.out_dir, f"{source}.json")))
        dump(f"{source}_ego", {tok: f"{ego[tok]}\n{reports[tok]}" for tok in frames})

    unknown = sum(s is None for s in states.values())
    print(f"Wrote ego.json + {', '.join(f'{s}_ego.json' for s in COMPOSED)} "
          f"for {len(frames)} frames ({unknown} unknown)")


if __name__ == "__main__":
    main()
