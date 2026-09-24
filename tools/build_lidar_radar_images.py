"""
LiDAR + radar images for the "pixels" route (condition Recovered_LiDAR_Radar).

Differs from recovery/lidar_recovery.py (Recovered_LiDAR, kept as the
single-sweep ablation) in two ways:

  1. Multi-sweep LiDAR: the keyframe plus the previous sweeps (default 10 in
     total, ~0.45 s), each moved into the keyframe ego frame with its own ego
     pose, as in the devkit's LidarPointCloud.from_file_multisweep() and the
     10-sweep input of CenterPoint/BEVFusion. Only past sweeps are used.
  2. Radar motion: every radar return moving faster than MIN_SPEED is drawn
     as a white vertical pillar (CRF-Net style), with a white arrow to where
     its ego-motion-compensated velocity puts it ARROW_SECONDS later whenever
     that arrow is long enough to see. The words route already carries the
     same radar velocities as text, so both routes get the same sensors.

Colours match Recovered_LiDAR (depth: red near, green ~25 m, blue 50 m+) on
black; radar marks are white, which the depth ramp never produces. Images are
native 1600x900; the inference loader resizes them like every other condition.

Usage:
    python tools/build_lidar_radar_images.py            # all DriveBench frames
    python tools/build_lidar_radar_images.py --limit 3  # quick look
"""

import os
import sys
import json
import argparse
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from recovery.lidar_recovery import _load_lidar_points, _make_transform, _depth_to_colour
from tools.sensor_text.common import (META_DIR, DATA, IMG_W, IMG_H, MAX_RANGE,
                                      Meta, FrameContext, default_roots, find_file)

CONDITION = "Recovered_LiDAR_Radar"
MAX_SWEEP_AGE = 0.6   # s; a gap longer than this ends the sweep chain
MIN_POINT_DIST = 1.0  # m from the sensor; drops returns off the ego vehicle (devkit remove_close)
MIN_DEPTH = 1.0       # m in front of the camera, as in Recovered_LiDAR
MIN_SPEED = 0.5       # m/s; slower radar returns get no arrow (same threshold as "stationary" in the text)
ARROW_COLOUR = (255, 255, 255)
PILLAR_HEIGHT = 2.0  # m; radar has no usable elevation, so returns become pillars
MIN_ARROW_PX = 25    # px at 1600x900; shorter motion arrows are left out

# Set in the parent before the Pool forks, so workers share it copy-on-write.
_META: Optional[Meta] = None
_SD_BY_TOKEN: Dict[str, dict] = {}


def lidar_sweeps(ctx: FrameContext, n_sweeps: int, roots: List[str]) -> Tuple[np.ndarray, int]:
    """Keyframe + up to n_sweeps-1 previous LIDAR_TOP sweeps, all in the
    keyframe ego frame. Returns (xyz [N,3], sweeps used)."""
    clouds, rec, t0 = [], ctx.lidar, ctx.lidar["timestamp"]
    while rec is not None and len(clouds) < n_sweeps:
        if (t0 - rec["timestamp"]) / 1e6 > MAX_SWEEP_AGE:
            break
        pts = _load_lidar_points(find_file(rec["filename"], roots))
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) > MIN_POINT_DIST]
        pose = _META.ego_poses[rec["ego_pose_token"]]
        T = (ctx.T_ego_global
             @ _make_transform(pose["rotation"], pose["translation"])
             @ ctx.sensor_to_ego(rec))
        clouds.append((T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3])
        rec = _SD_BY_TOKEN.get(rec["prev"]) if rec["prev"] else None
    return np.vstack(clouds), len(clouds)


def to_camera(xyz_ego: np.ndarray, T_cam_ego: np.ndarray, K: np.ndarray):
    """Pixel coords and depth of ego-frame points (no filtering)."""
    pc = (T_cam_ego @ np.hstack([xyz_ego, np.ones((len(xyz_ego), 1))]).T)[:3]
    depth = pc[2]
    uv = (K @ pc)[:2] / np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    return uv.T, depth


def render(lidar_ego: np.ndarray, radar_xyz: np.ndarray, radar_v: np.ndarray,
           T_cam_ego: np.ndarray, K: np.ndarray, point_radius: int,
           arrow_seconds: float, arrow_thickness: int) -> Tuple[Image.Image, int, int]:
    canvas = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    uv, depth = to_camera(lidar_ego, T_cam_ego, K)
    keep = ((depth > MIN_DEPTH) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W)
            & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H))
    uv, depth = uv[keep].astype(int), depth[keep]
    colours = (_depth_to_colour(depth) * 255).astype(np.uint8)
    for i in np.argsort(-depth):  # far first, so near points stay on top
        cv2.circle(canvas, (int(uv[i, 0]), int(uv[i, 1])), point_radius,
                   tuple(int(c) for c in colours[i]), -1)

    # Radar measures Doppler (radial) velocity, so for traffic ahead the motion
    # vector points along the line of sight and projects to a few pixels. Each
    # moving return is therefore drawn as a vertical pillar from the ground up
    # to PILLAR_HEIGHT (the radar-to-image encoding of CRF-Net, Nobis et al.
    # 2019), plus an arrow from its mid-height where the motion is visible.
    n_marks = 0
    speed = np.hypot(radar_v[:, 0], radar_v[:, 1])
    moving = (speed > MIN_SPEED) & (np.hypot(radar_xyz[:, 0], radar_xyz[:, 1]) <= MAX_RANGE)
    if moving.any():
        base = radar_xyz[moving].copy()
        base[:, 2] = 0.0
        top = base.copy()
        top[:, 2] = PILLAR_HEIGHT
        mid = base.copy()
        mid[:, 2] = PILLAR_HEIGHT / 2
        ahead = mid.copy()
        ahead[:, :2] += radar_v[moving] * arrow_seconds
        (uv_b, d_b), (uv_t, _), (uv_m, _), (uv_a, d_a) = (
            to_camera(p, T_cam_ego, K) for p in (base, top, mid, ahead))
        for i in range(len(base)):
            if d_b[i] <= MIN_DEPTH or not (0 <= uv_b[i, 0] < IMG_W and 0 <= uv_t[i, 1] < IMG_H):
                continue
            pt = lambda uv: (int(uv[i, 0]), int(uv[i, 1]))
            cv2.line(canvas, pt(uv_b), pt(uv_t), ARROW_COLOUR, arrow_thickness, cv2.LINE_AA)
            if arrow_seconds > 0 and d_a[i] > MIN_DEPTH and np.hypot(*(uv_a[i] - uv_m[i])) >= MIN_ARROW_PX:
                cv2.arrowedLine(canvas, pt(uv_m), pt(uv_a), ARROW_COLOUR,
                                arrow_thickness, cv2.LINE_AA, tipLength=0.25)
            n_marks += 1
    return Image.fromarray(canvas), int(keep.sum()), n_marks


def build_frame(job) -> List[Tuple[str, str, int, int, int]]:
    token, cam_paths, dest_root, opts = job
    roots = opts["roots"]
    ctx = FrameContext(_META, token)
    lidar_ego, n_sweeps = lidar_sweeps(ctx, opts["sweeps"], roots)
    radar_xyz, radar_v = ctx.radar_points_ego(roots)
    out = []
    for cam, src in cam_paths.items():
        if cam not in ctx.cameras:
            continue
        T_cam_ego, K = ctx.cameras[cam]
        img, n_pts, n_arrows = render(lidar_ego, radar_xyz, radar_v, T_cam_ego, K,
                                      opts["point_radius"], opts["arrow_seconds"],
                                      opts["arrow_thickness"])
        dst = os.path.join(dest_root, cam, os.path.basename(src))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img.save(dst, quality=95)
        out.append((token, cam, n_pts, n_arrows, n_sweeps))
    return out


def main():
    global _META, _SD_BY_TOKEN
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-dir", default=META_DIR)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--dest", default=f"data/corruption/{CONDITION}",
                        help="must be data/corruption/<condition> so the loader finds it")
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--point-radius", type=int, default=3,
                        help="px at 1600x900 (Recovered_LiDAR uses 4 for its sparser cloud)")
    parser.add_argument("--arrow-seconds", type=float, default=1.0,
                        help="length of motion arrows in seconds of motion; 0 = no arrows")
    parser.add_argument("--arrow-thickness", type=int, default=4)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None, help="first N frames only")
    args = parser.parse_args()

    frames: Dict[str, Dict[str, str]] = {}
    for e in json.load(open(args.data)):
        frames.setdefault(e["frame_token"], {}).update(
            {cam: p for cam, p in e["image_path"].items() if p})
    jobs = list(frames.items())[:args.limit]

    print("Loading nuScenes metadata...")
    _META = Meta(args.meta_dir)
    _SD_BY_TOKEN = {r["token"]: r for recs in _META.sample_data.values() for r in recs}
    opts = {"roots": default_roots(), "sweeps": args.sweeps, "point_radius": args.point_radius,
            "arrow_seconds": args.arrow_seconds, "arrow_thickness": args.arrow_thickness}
    jobs = [(tok, cams, args.dest, opts) for tok, cams in jobs]

    with Pool(args.workers) as pool:
        rows = [r for out in pool.imap_unordered(build_frame, jobs, chunksize=2) for r in out]

    pts = np.array([r[2] for r in rows])
    arrows = np.array([r[3] for r in rows])
    sweeps = np.array([r[4] for r in rows])
    print(f"Wrote {len(rows)} images for {len(jobs)} frames to {args.dest}")
    print(f"  LiDAR points/image: mean {pts.mean():.0f}, min {pts.min()}")
    print(f"  sweeps/frame: mean {sweeps.mean():.1f}, min {sweeps.min()}")
    print(f"  radar marks/image: mean {arrows.mean():.1f}, images with >=1 mark "
          f"{(arrows > 0).mean():.0%}")


if __name__ == "__main__":
    main()
