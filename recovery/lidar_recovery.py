"""
Cross-modal frame recovery via LiDAR-to-camera projection.

When no temporal sweep neighbours exist for a frame (first/last clip frame,
network gap), this module projects the nuScenes LIDAR_TOP point cloud into
each camera's image plane using the nuScenes calibration data and renders a
depth-colourised image.  The output is a 224×224 PIL image with a black
background and points coloured by depth (near = red, far = blue).

This is intentionally lightweight — it is NOT a GAN/diffusion LiDAR→RGB
synthesis (those require large model training).  The goal is to give the VLM
structural scene information (object positions, depths) when temporal
interpolation is unavailable, which is enough to outperform a blank frame for
spatial-reasoning questions.

Requirements
------------
  pip install numpy scipy Pillow
  nuScenes metadata JSON files (calibrated_sensor.json, sample_data.json,
  ego_pose.json) — already present at the path used in tools/.

Usage
-----
  from recovery.lidar_recovery import build_lidar_recovered_dataset

  build_lidar_recovered_dataset(
      meta_dir="/mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval",
      nuscenes_root="/path/to/nuscenes",
      data_dir="data",
      dest="data/corruption/Recovered_LiDAR",
  )
"""

import json
import os
from typing import Dict, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# nuScenes calibration helpers (pure-numpy, no nuscenes-devkit required)
# ---------------------------------------------------------------------------

def _quat_to_rotation_matrix(q: list) -> np.ndarray:
    """Convert a nuScenes quaternion [w, x, y, z] to a 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _make_transform(rotation_quat: list, translation: list) -> np.ndarray:
    """Return a 4×4 rigid-body transform from sensor to ego frame."""
    T = np.eye(4)
    T[:3, :3] = _quat_to_rotation_matrix(rotation_quat)
    T[:3, 3] = translation
    return T


def _load_calibrations(meta_dir: str) -> Dict[str, dict]:
    """
    Parse calibrated_sensor.json and return a dict keyed by token.
    Each value includes rotation, translation, camera_intrinsic (for cams).
    """
    path = os.path.join(meta_dir, "calibrated_sensor.json")
    records = json.load(open(path))
    return {r["token"]: r for r in records}


def _load_ego_poses(meta_dir: str) -> Dict[str, dict]:
    path = os.path.join(meta_dir, "ego_pose.json")
    records = json.load(open(path))
    return {r["token"]: r for r in records}


def _load_sample_data_index(meta_dir: str) -> Dict[str, list]:
    """Return a dict mapping frame_token → list of sample_data records."""
    path = os.path.join(meta_dir, "sample_data.json")
    records = json.load(open(path))
    index: Dict[str, list] = {}
    for r in records:
        index.setdefault(r["sample_token"], []).append(r)
    return index


# ---------------------------------------------------------------------------
# LiDAR point cloud loading
# ---------------------------------------------------------------------------

def _load_lidar_points(lidar_path: str) -> np.ndarray:
    """
    Load a nuScenes LIDAR_TOP binary file.
    Format: N × 5 float32 (x, y, z, intensity, ring_index).
    Returns (N, 3) xyz array.
    """
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :3]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _project_lidar_to_camera(
    points_lidar: np.ndarray,
    T_ego_lidar: np.ndarray,
    T_ego_cam: np.ndarray,
    K: np.ndarray,
    img_width: int,
    img_height: int,
) -> tuple:
    """
    Project LiDAR points into camera image coordinates.

    Returns
    -------
    uv   : (M, 2) float array of pixel coordinates for visible points
    depths: (M,) float array of depths in camera frame
    """
    # LiDAR → ego → camera
    T_cam_ego = np.linalg.inv(T_ego_cam)
    T_cam_lidar = T_cam_ego @ T_ego_lidar

    ones = np.ones((len(points_lidar), 1))
    pts_h = np.hstack([points_lidar, ones])          # (N, 4)
    pts_cam = (T_cam_lidar @ pts_h.T).T[:, :3]       # (N, 3)

    # Keep only points in front of camera
    mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[mask]
    if len(pts_cam) == 0:
        return np.empty((0, 2)), np.empty(0)

    depths = pts_cam[:, 2]
    uv_h = (K @ pts_cam.T).T                          # (M, 3)
    uv = uv_h[:, :2] / uv_h[:, 2:3]                  # (M, 2)

    # Keep only points inside image bounds
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_height)
    )
    return uv[in_bounds], depths[in_bounds]


def _depth_to_colour(depths: np.ndarray, d_min: float = 1.0, d_max: float = 50.0) -> np.ndarray:
    """Map depth values to BGR colours (near=red, far=blue) via HSV wheel."""
    t = np.clip((depths - d_min) / (d_max - d_min), 0, 1)
    # Hue 0 (red) → 0.67 (blue)
    hue = (1 - t) * 0          # red at close, via green, to blue
    # Use a simple HSV→RGB approximation: map t to a gradient
    r = np.clip(1.0 - t * 1.5, 0, 1)
    g = np.clip(1.0 - np.abs(t - 0.5) * 2, 0, 1)
    b = np.clip((t - 0.5) * 2, 0, 1)
    return np.stack([r, g, b], axis=1)  # (M, 3) in [0,1]


def lidar_project(
    frame_token: str,
    camera_name: str,
    meta_dir: str,
    nuscenes_root: str,
    calibrations: Dict[str, dict],
    ego_poses: Dict[str, dict],
    sample_data_index: Dict[str, list],
    output_size: int = 224,
) -> Optional[Image.Image]:
    """
    Project LIDAR_TOP into one camera for a given frame_token.

    Returns a PIL.Image (output_size × output_size) with depth-colourised
    LiDAR points on a black background, or None if data is missing.
    """
    frame_records = sample_data_index.get(frame_token, [])

    lidar_record = next(
        (r for r in frame_records if r.get("channel") == "LIDAR_TOP"), None
    )
    cam_record = next(
        (r for r in frame_records if r.get("channel") == camera_name), None
    )
    if lidar_record is None or cam_record is None:
        return None

    lidar_path = os.path.join(nuscenes_root, lidar_record["filename"])
    if not os.path.exists(lidar_path):
        return None

    lidar_cal = calibrations[lidar_record["calibrated_sensor_token"]]
    cam_cal = calibrations[cam_record["calibrated_sensor_token"]]
    lidar_ego = ego_poses[lidar_record["ego_pose_token"]]
    cam_ego = ego_poses[cam_record["ego_pose_token"]]

    T_ego_lidar = (
        _make_transform(lidar_ego["rotation"], lidar_ego["translation"])
        @ _make_transform(lidar_cal["rotation"], lidar_cal["translation"])
    )
    T_ego_cam = (
        _make_transform(cam_ego["rotation"], cam_ego["translation"])
        @ _make_transform(cam_cal["rotation"], cam_cal["translation"])
    )
    K = np.array(cam_cal["camera_intrinsic"], dtype=np.float64)

    # nuScenes camera images are 1600×900
    IMG_W, IMG_H = 1600, 900
    points = _load_lidar_points(lidar_path)
    uv, depths = _project_lidar_to_camera(points, T_ego_lidar, T_ego_cam, K, IMG_W, IMG_H)

    canvas = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    if len(uv) > 0:
        colours = _depth_to_colour(depths)
        u = uv[:, 0].astype(int)
        v = uv[:, 1].astype(int)
        canvas[v, u] = (colours * 255).astype(np.uint8)
        # Thicken points so they're visible at 224×224
        for dv in [-1, 0, 1]:
            for du in [-1, 0, 1]:
                vv = np.clip(v + dv, 0, IMG_H - 1)
                uu = np.clip(u + du, 0, IMG_W - 1)
                canvas[vv, uu] = (colours * 255).astype(np.uint8)

    img = Image.fromarray(canvas, mode="RGB").resize((output_size, output_size))
    return img


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def build_lidar_recovered_dataset(
    meta_dir: str,
    nuscenes_root: str,
    data_dir: str,
    dest: str,
    limit: Optional[int] = None,
) -> tuple:
    """
    Populate dest/<CAM>/<filename> with LiDAR-projected images for every
    (frame, camera) pair found in the DriveBench QA JSON files.

    Pass --corruption Recovered_LiDAR to any inference script to read these.
    """
    calibrations = _load_calibrations(meta_dir)
    ego_poses = _load_ego_poses(meta_dir)
    sample_data_index = _load_sample_data_index(meta_dir)

    # Build frame_token → {cam → original_filename} mapping from QA data
    frame_to_cam_filename: Dict[str, Dict[str, str]] = {}
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
    items = list(frame_to_cam_filename.items())
    if limit:
        items = items[:limit]

    for frame_token, cam_filenames in items:
        for cam, out_filename in cam_filenames.items():
            img = lidar_project(
                frame_token, cam, meta_dir, nuscenes_root,
                calibrations, ego_poses, sample_data_index,
            )
            if img is None:
                skipped.append((frame_token, cam, "LiDAR data not found"))
                continue
            out_dir = os.path.join(dest, cam)
            os.makedirs(out_dir, exist_ok=True)
            img.save(os.path.join(out_dir, out_filename))
            written += 1

    return written, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-dir", required=True,
                        help="nuScenes metadata directory (contains calibrated_sensor.json etc.)")
    parser.add_argument("--nuscenes-root", required=True,
                        help="nuScenes root containing sweeps/LIDAR_TOP/")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dest", default="data/corruption/Recovered_LiDAR")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    written, skipped = build_lidar_recovered_dataset(
        args.meta_dir, args.nuscenes_root, args.data_dir, args.dest, args.limit,
    )
    print(f"Wrote {written} LiDAR-projected images to {args.dest}")
    if skipped:
        for frame_token, cam, reason in skipped[:10]:
            print(f"  {frame_token}/{cam}: {reason}")
        if len(skipped) > 10:
            print(f"  … and {len(skipped) - 10} more")
