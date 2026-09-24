"""
Deployable sensor text: unlabelled LiDAR clusters with radar velocities.
No learned detector and no annotations; only what raw LiDAR + radar give.

LiDAR (LIDAR_TOP keyframe, ego frame):
  1. Keep points 0.3-2.5 m above ground, within 50 m, outside the ego box.
  2. Rasterize to a 0.3 m BEV occupancy grid and label 8-connected
     components (scipy.ndimage.label).
  3. Keep components with >= 10 points whose footprint (cv2.minAreaRect) is
     at most 12 x 4 m and not a thin line (> 5 m long, < 0.6 m wide), which
     rejects walls, fences and kerbs.
  4. Size-only labels: pedestrian-sized / cyclist-sized / car-sized /
     truck-or-bus-sized / small object.

Radar (all five RADAR_* keyframes, ego-motion-compensated velocities):
  - A cluster's velocity is the median of radar returns within 2 m (BEV) of
    any of its points; no nearby return -> "speed unknown".
  - Moving returns (> 1 m/s) not near any cluster are merged on a 2 m grid
    and reported as "radar return (not resolved by LiDAR)".

Usage:
    python tools/sensor_text/cluster.py   # -> data/sensor_text/cluster.json
"""

import os
import sys
import argparse

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.sensor_text.common import (META_DIR, DATA, OUT_DIR, MAX_RANGE, Meta, FrameContext,
                                      load_frames, default_roots)
from tools.sensor_text.format import DEFAULT_MAX_OBJECTS, write_source

Z_MIN, Z_MAX = 0.3, 2.5
EGO_BOX = (-1.5, 4.0, 1.2)       # x_min, x_max, |y| max (ego origin = rear axle)
CELL = 0.3
MIN_POINTS = 10
MAX_LENGTH, MAX_WIDTH = 12.0, 4.0
RADAR_ASSOC = 2.0
RADAR_ONLY_SPEED = 1.0
RADAR_ONLY_CELL = 2.0


def size_label(length: float, width: float, height: float) -> str:
    if length <= 1.2 and width <= 1.2:
        return "pedestrian-sized object" if height >= 1.0 else "small object"
    if length <= 2.5:
        return "cyclist-sized object"
    if length <= 6.0:
        return "car-sized object"
    return "truck-or-bus-sized object"


def lidar_clusters(pts: np.ndarray) -> list:
    """Return a list of (points [M,3], length, width) clusters."""
    d = np.hypot(pts[:, 0], pts[:, 1])
    in_ego = ((pts[:, 0] > EGO_BOX[0]) & (pts[:, 0] < EGO_BOX[1])
              & (np.abs(pts[:, 1]) < EGO_BOX[2]))
    keep = (pts[:, 2] >= Z_MIN) & (pts[:, 2] <= Z_MAX) & (d <= MAX_RANGE) & ~in_ego
    pts = pts[keep]
    if len(pts) == 0:
        return []

    n = int(np.ceil(2 * MAX_RANGE / CELL)) + 1
    ix = ((pts[:, 0] + MAX_RANGE) / CELL).astype(int)
    iy = ((pts[:, 1] + MAX_RANGE) / CELL).astype(int)
    grid = np.zeros((n, n), dtype=bool)
    grid[ix, iy] = True
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3)))
    point_labels = labels[ix, iy]

    clusters = []
    for lab in np.unique(point_labels):
        cpts = pts[point_labels == lab]
        if len(cpts) < MIN_POINTS:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(cpts[:, :2].astype(np.float32))
        length, width = max(w, h, CELL), max(min(w, h), CELL)
        if length > MAX_LENGTH or width > MAX_WIDTH or (length > 5.0 and width < 0.6):
            continue
        clusters.append((cpts, length, width))
    return clusters


def radar_only_groups(xyz: np.ndarray, v: np.ndarray) -> list:
    """Merge unassociated moving radar returns on a coarse grid."""
    if len(xyz) == 0:
        return []
    keys = np.floor(xyz[:, :2] / RADAR_ONLY_CELL).astype(int)
    groups = {}
    for i, k in enumerate(map(tuple, keys)):
        groups.setdefault(k, []).append(i)
    return [(xyz[idx].mean(axis=0), np.median(v[idx], axis=0)) for idx in groups.values()]


def frame_objects(ctx: FrameContext, roots: list) -> list:
    clusters = lidar_clusters(ctx.lidar_points_ego(roots))
    rxyz, rv = ctx.radar_points_ego(roots)
    rxyz_valid = np.hypot(rxyz[:, 0], rxyz[:, 1]) <= MAX_RANGE
    rxyz, rv = rxyz[rxyz_valid], rv[rxyz_valid]

    # Associate each radar return to the nearest cluster point within 2 m.
    assoc = np.full(len(rxyz), -1)
    if clusters and len(rxyz):
        all_pts = np.vstack([c[0][:, :2] for c in clusters])
        owner = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(clusters)])
        dist, nn = cKDTree(all_pts).query(rxyz[:, :2], distance_upper_bound=RADAR_ASSOC)
        hit = np.isfinite(dist)
        assoc[hit] = owner[nn[hit]]

    objs = []
    for i, (cpts, length, width) in enumerate(clusters):
        cx, cy = np.median(cpts[:, 0]), np.median(cpts[:, 1])
        cz = float(np.median(cpts[:, 2]))
        obj = {"label": size_label(length, width, float(cpts[:, 2].max())),
               "x": float(cx), "y": float(cy), "z": cz,
               "speed": None, "vx": None, "vy": None,
               "cam": None, "u": None, "v": None,
               "n_points": int(len(cpts)), "length": float(length), "width": float(width)}
        mine = assoc == i
        if mine.any():
            vx, vy = np.median(rv[mine], axis=0)
            obj.update(vx=float(vx), vy=float(vy), speed=float(np.hypot(vx, vy)))
        proj = ctx.project((cx, cy, cz))
        if proj:
            obj["cam"], obj["u"], obj["v"] = proj[0], float(proj[1]), float(proj[2])
        objs.append(obj)

    moving = (assoc < 0) & (np.hypot(rv[:, 0], rv[:, 1]) > RADAR_ONLY_SPEED)
    for (x, y, z), (vx, vy) in radar_only_groups(rxyz[moving], rv[moving]):
        obj = {"label": "radar return (not resolved by LiDAR)",
               "x": float(x), "y": float(y), "z": 0.5,
               "vx": float(vx), "vy": float(vy), "speed": float(np.hypot(vx, vy)),
               "cam": None, "u": None, "v": None}
        proj = ctx.project((x, y, 0.5))
        if proj:
            obj["cam"], obj["u"], obj["v"] = proj[0], float(proj[1]), float(proj[2])
        objs.append(obj)
    return objs


def build(meta_dir: str, data: str, roots: list) -> dict:
    frames = load_frames(data)
    print("Loading nuScenes metadata...")
    meta = Meta(meta_dir)
    return {tok: frame_objects(FrameContext(meta, tok), roots) for tok in frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-dir", default=META_DIR)
    parser.add_argument("--nuscenes-root", nargs="+", default=default_roots())
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS)
    args = parser.parse_args()
    write_source("cluster", build(args.meta_dir, args.data, args.nuscenes_root),
                 args.max_objects, args.out_dir)


if __name__ == "__main__":
    main()
