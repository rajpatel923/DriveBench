"""
Oracle sensor text: nuScenes ground-truth boxes, as if LiDAR/radar
perception were perfect. Upper bound for the "words" arm.

Kept boxes: at least one LiDAR or radar return (num_lidar_pts +
num_radar_pts > 0), within 50 m. Velocity and turning rate come from the
previous annotation only (dropped if more than 1.5 s back), as a tracker
would have them; the next annotation would leak the object's future motion,
which is what the perception MCQs ask about. Velocity is rotated into the
ego frame.

Usage:
    python tools/sensor_text/oracle.py   # -> data/sensor_text/oracle.json
"""

import os
import sys
import json
import argparse

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.sensor_text.common import META_DIR, DATA, OUT_DIR, Meta, FrameContext, load_frames
from tools.sensor_text.format import DEFAULT_MAX_OBJECTS, write_source

# nuScenes category name prefix -> label shown to the model (first match wins)
LABELS = [
    ("vehicle.car", "car"),
    ("vehicle.truck", "truck"),
    ("vehicle.bus", "bus"),
    ("vehicle.trailer", "trailer"),
    ("vehicle.construction", "construction vehicle"),
    ("vehicle.emergency", "emergency vehicle"),
    ("vehicle.motorcycle", "motorcycle"),
    ("vehicle.bicycle", "bicycle"),
    ("human.pedestrian", "pedestrian"),
    ("animal", "animal"),
    ("movable_object.barrier", "barrier"),
    ("movable_object.trafficcone", "traffic cone"),
    ("movable_object.pushable_pullable", "pushable object"),
    ("movable_object.debris", "debris"),
    ("static_object.bicycle_rack", "bicycle rack"),
]

MAX_TIME_DIFF = 1.5  # s, as in the devkit's box_velocity()


def label_for(category: str) -> str:
    return next((lab for prefix, lab in LABELS if category.startswith(prefix)), "object")


def box_velocity(ann: dict, anns: dict, sample_ts: dict):
    """Global-frame (vx, vy, vz) from the previous box to this one, or None.
    Unlike NuScenes.box_velocity() this never looks at the next box, which
    would leak where the object goes next."""
    if not ann["prev"]:
        return None
    prev = anns[ann["prev"]]
    dt = (sample_ts[ann["sample_token"]] - sample_ts[prev["sample_token"]]) / 1e6
    if dt <= 0 or dt > MAX_TIME_DIFF:
        return None
    return (np.array(ann["translation"]) - np.array(prev["translation"])) / dt


def yaw(q) -> float:
    """Heading about z (rad) from a nuScenes [w, x, y, z] quaternion."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_rate(ann: dict, anns: dict, sample_ts: dict):
    """Turning rate in deg/s (+ = left) from the previous box to this one, or
    None. Past-only, like a tracker would have it: using the next box would
    leak the "going ahead / turn left / turn right" answer."""
    if not ann["prev"]:
        return None
    prev = anns[ann["prev"]]
    dt = (sample_ts[ann["sample_token"]] - sample_ts[prev["sample_token"]]) / 1e6
    if dt <= 0 or dt > MAX_TIME_DIFF:
        return None
    d = (yaw(ann["rotation"]) - yaw(prev["rotation"]) + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(d) / dt)


def build(meta_dir: str, data: str) -> dict:
    frames = load_frames(data)
    print("Loading nuScenes metadata (sample_annotation.json is ~650 MB)...")
    meta = Meta(meta_dir)
    load = lambda name: json.load(open(os.path.join(meta_dir, f"{name}.json")))
    categories = {c["token"]: c["name"] for c in load("category")}
    instance_cat = {i["token"]: categories[i["category_token"]] for i in load("instance")}
    sample_ts = {s["token"]: s["timestamp"] for s in load("sample")}
    anns = {a["token"]: a for a in load("sample_annotation")}

    by_sample = {}
    for a in anns.values():
        if a["sample_token"] in frames:
            by_sample.setdefault(a["sample_token"], []).append(a)

    objects_by_frame = {}
    for tok in frames:
        ctx = FrameContext(meta, tok)
        objs = []
        for a in by_sample.get(tok, []):
            if a["num_lidar_pts"] + a["num_radar_pts"] <= 0:
                continue
            x, y, z = ctx.global_to_ego_points(np.array(a["translation"]))[0]
            obj = {"label": label_for(instance_cat[a["instance_token"]]),
                   "x": float(x), "y": float(y), "z": float(z),
                   "speed": None, "vx": None, "vy": None,
                   "cam": None, "u": None, "v": None,
                   "annotation_token": a["token"]}
            v = box_velocity(a, anns, sample_ts)
            if v is not None:
                vx, vy = ctx.global_to_ego_vectors(v)[0][:2]
                obj.update(vx=float(vx), vy=float(vy), speed=float(np.hypot(vx, vy)))
            obj["yaw_rate"] = yaw_rate(a, anns, sample_ts)
            proj = ctx.project((x, y, z))
            if proj:
                obj["cam"], obj["u"], obj["v"] = proj[0], float(proj[1]), float(proj[2])
            objs.append(obj)
        objects_by_frame[tok] = objs
    return objects_by_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-dir", default=META_DIR)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS)
    args = parser.parse_args()
    write_source("oracle", build(args.meta_dir, args.data), args.max_objects, args.out_dir)


if __name__ == "__main__":
    main()
