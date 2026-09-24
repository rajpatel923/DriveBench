"""
Shared nuScenes helpers for the sensor-text builders: metadata loading,
frame contexts (ego pose + camera models), radar PCD parsing, and projecting
ego-frame points to the DriveBench `<CAM,u,v>` convention (u, v normalized to
[0, 1] over the 1600x900 image, (0, 0) top-left).
"""

import os
import sys
import glob
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from recovery.lidar_recovery import (
    _load_calibrations, _load_ego_poses, _load_sample_data_index,
    _load_lidar_points, _make_transform,
)

META_DIR = "data/nuscenes/v1.0-trainval_meta/v1.0-trainval"
DATA = "data/drivebench-test-final.json"
OUT_DIR = "data/sensor_text"
IMG_W, IMG_H = 1600, 900
MAX_RANGE = 50.0

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
          "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]


def default_roots() -> List[str]:
    return sorted(glob.glob("data/nuscenes/v1.0-trainval*_blobs"))


def load_frames(data_path: str = DATA) -> Dict[str, str]:
    """Return {frame_token: scene_token} for every DriveBench frame."""
    return {e["frame_token"]: e["scene_token"] for e in json.load(open(data_path))}


def find_file(rel_path: str, roots: List[str]) -> str:
    for root in roots:
        path = os.path.join(root, rel_path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{rel_path} not found under any of {roots}")


class Meta:
    """Calibrations, ego poses and the per-sample sample_data index."""

    def __init__(self, meta_dir: str = META_DIR):
        self.meta_dir = meta_dir
        self.calibrations = _load_calibrations(meta_dir)
        self.ego_poses = _load_ego_poses(meta_dir)
        self.sample_data = _load_sample_data_index(meta_dir)

    def keyframe(self, sample_token: str, channel: str) -> Optional[dict]:
        return next((r for r in self.sample_data.get(sample_token, [])
                     if f"/{channel}/" in r["filename"] and r.get("is_key_frame")), None)


class FrameContext:
    """Everything needed to move points between the ego frame (at the
    LIDAR_TOP keyframe time; x forward, y left, z up, metres) and the six
    camera images of one DriveBench frame."""

    def __init__(self, meta: Meta, sample_token: str):
        self.meta = meta
        self.sample_token = sample_token
        self.lidar = meta.keyframe(sample_token, "LIDAR_TOP")
        if self.lidar is None:
            raise KeyError(f"No LIDAR_TOP keyframe for {sample_token}")
        pose = meta.ego_poses[self.lidar["ego_pose_token"]]
        self.T_global_ego = _make_transform(pose["rotation"], pose["translation"])
        self.T_ego_global = np.linalg.inv(self.T_global_ego)

        # camera: (T_cam_from_ego_lidar_time, K)
        self.cameras = {}
        for cam in CAMERAS:
            rec = meta.keyframe(sample_token, cam)
            if rec is None:
                continue
            cal = meta.calibrations[rec["calibrated_sensor_token"]]
            cam_pose = meta.ego_poses[rec["ego_pose_token"]]
            T_global_cam = (_make_transform(cam_pose["rotation"], cam_pose["translation"])
                            @ _make_transform(cal["rotation"], cal["translation"]))
            T_cam_ego = np.linalg.inv(T_global_cam) @ self.T_global_ego
            self.cameras[cam] = (T_cam_ego, np.array(cal["camera_intrinsic"], dtype=np.float64))

    def sensor_to_ego(self, record: dict) -> np.ndarray:
        """4x4 sensor->ego transform (calibration only; the few-ms offset
        between sensor and LIDAR_TOP ego poses is ignored)."""
        cal = self.meta.calibrations[record["calibrated_sensor_token"]]
        return _make_transform(cal["rotation"], cal["translation"])

    def global_to_ego_points(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.atleast_2d(xyz)
        homo = np.hstack([xyz, np.ones((len(xyz), 1))])
        return (self.T_ego_global @ homo.T).T[:, :3]

    def global_to_ego_vectors(self, v: np.ndarray) -> np.ndarray:
        return (self.T_ego_global[:3, :3] @ np.atleast_2d(v).T).T

    def project(self, xyz_ego) -> Optional[Tuple[str, float, float]]:
        """Project one ego-frame point to the camera where it sits closest to
        the image centre. Returns (cam, u_norm, v_norm) or None if no camera
        sees it."""
        p = np.append(np.asarray(xyz_ego, dtype=np.float64), 1.0)
        best = None
        for cam, (T_cam_ego, K) in self.cameras.items():
            pc = (T_cam_ego @ p)[:3]
            if pc[2] < 0.5:
                continue
            uv = K @ pc
            u, v = uv[0] / uv[2], uv[1] / uv[2]
            if not (0 <= u < IMG_W and 0 <= v < IMG_H):
                continue
            off = abs(u / IMG_W - 0.5)
            if best is None or off < best[0]:
                best = (off, cam, u / IMG_W, v / IMG_H)
        return None if best is None else best[1:]

    def lidar_points_ego(self, roots: List[str]) -> np.ndarray:
        path = find_file(self.lidar["filename"], roots)
        pts = _load_lidar_points(path)
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        return (self.sensor_to_ego(self.lidar) @ homo.T).T[:, :3]

    def radar_points_ego(self, roots: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """All five radars' valid returns, as (xyz_ego [N,3], v_ego [N,2]).
        Velocities are the ego-motion-compensated (vx_comp, vy_comp), i.e.
        absolute object velocity, rotated into ego axes."""
        xyz_all, v_all = [], []
        for radar in RADARS:
            rec = self.meta.keyframe(self.sample_token, radar)
            if rec is None:
                continue
            pts = read_radar_pcd(find_file(rec["filename"], roots))
            T = self.sensor_to_ego(rec)
            xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)
            xyz = (T @ np.hstack([xyz, np.ones((len(xyz), 1))]).T).T[:, :3]
            v = np.stack([pts["vx_comp"], pts["vy_comp"], np.zeros(len(pts))], axis=1)
            v = (T[:3, :3] @ v.T).T[:, :2]
            xyz_all.append(xyz)
            v_all.append(v)
        if not xyz_all:
            return np.zeros((0, 3)), np.zeros((0, 2))
        return np.vstack(xyz_all), np.vstack(v_all)


_PCD_TYPES = {("F", 4): "<f4", ("F", 8): "<f8", ("I", 1): "<i1", ("I", 2): "<i2",
              ("I", 4): "<i4", ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4"}


def read_radar_pcd(path: str) -> np.ndarray:
    """Parse a nuScenes radar .pcd (binary) into a structured array, keeping
    returns that pass the devkit's default filters (invalid_state == 0,
    ambig_state == 3)."""
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"DATA binary") + len(b"DATA binary\n")
    header = {}
    for line in raw[:end].decode().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            header[parts[0]] = parts[1:]
    dtype = np.dtype([(name, _PCD_TYPES[(t, int(s))]) for name, t, s in
                      zip(header["FIELDS"], header["TYPE"], header["SIZE"])])
    n = int(header["POINTS"][0])
    pts = np.frombuffer(raw[end:end + n * dtype.itemsize], dtype=dtype)
    return pts[(pts["invalid_state"] == 0) & (pts["ambig_state"] == 3)]
