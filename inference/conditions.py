"""
Shared input construction for Stage 1 ("Pixels or Words?") runs.

Every condition goes through the same two functions so that the only thing
that differs between runs is the content the model sees:

  load_images()      -- resolves the image for a condition and resizes it to
                        IMG_SIZE, so clean, blank, stale and LiDAR frames all
                        reach the model at the same resolution.
  build_user_text()  -- optionally prepends a LiDAR/radar text report and/or a
                        one-line note on how the images are drawn to the
                        question.
"""

import re
import json
import math
from typing import Dict, List, Optional, Tuple

from PIL import Image

# nuScenes 1600x900 scaled by 0.42 (16:9 kept). Qwen2.5-VL's processor snaps
# this to 672x392 (multiples of 28) identically for every condition.
IMG_SIZE = (672, 378)

SENSOR_TEXT_HEADER = "[Sensor report from LiDAR and radar]"
SENSOR_TEXT_FOOTER = "[End of sensor report]"

_LIDAR_NOTE = ("The camera images are replaced by LiDAR renders from the same six cameras: "
               "coloured dots are LiDAR points, coloured by distance (red near, green about "
               "25 m, blue 50 m or more).")
# Legend for image conditions that are not camera photos (--image_note auto).
IMAGE_NOTES = {
    "Recovered_LiDAR": _LIDAR_NOTE,
    "Recovered_LiDAR_Radar": _LIDAR_NOTE + (" White vertical bars mark radar returns from "
                                            "moving objects; a white arrow, where drawn, shows "
                                            "how far that object moves in one second."),
}


def resolve_image_note(image_note: str, condition: str) -> str:
    """'auto' -> IMAGE_NOTES for the condition (or none), 'none' -> no note,
    anything else is used as given."""
    if image_note == "auto":
        return IMAGE_NOTES.get(condition, "")
    return "" if image_note == "none" else image_note


def resolve_image_path(path: str, condition: str) -> Optional[str]:
    """Map an original data/nuscenes/samples/... path to the file for a
    condition. Returns None for NoImage (no file is read)."""
    if condition == "NoImage":
        return None
    if condition:
        return path.replace("nuscenes/samples", f"corruption/{condition}")
    return path


def load_images(image_path_dict: Dict[str, Optional[str]], condition: str,
                size: Tuple[int, int] = IMG_SIZE) -> Tuple[List[str], List[Image.Image]]:
    """Load every camera image for one QA entry under a condition.

    Returns (original_paths, images). original_paths are the untouched
    dataset paths, which replace_system_prompt() uses to name the cameras.
    """
    image_paths = [p for p in image_path_dict.values() if p is not None]
    images = []
    for path in image_paths:
        resolved = resolve_image_path(path, condition)
        if resolved is None:
            img = Image.new("RGB", size, (0, 0, 0))
        else:
            try:
                img = Image.open(resolved).convert("RGB")
            except OSError as e:
                raise FileNotFoundError(f"Cannot load image for condition "
                                        f"'{condition or 'clean'}': {resolved}") from e
            img = img.resize(size, Image.BICUBIC)
        images.append(img)
    return image_paths, images


def load_sensor_text(path: Optional[str]) -> Optional[Dict[str, str]]:
    """Load a {frame_token: text} JSON written by tools/sensor_text/*."""
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def build_user_text(question: str, frame_token: str,
                    sensor_text: Optional[Dict[str, str]], image_note: str = "") -> str:
    """Return exactly the user text the model sees for one question.

    image_note is a fixed legend for LiDAR/radar renders: text routes get a
    header explaining their format, so image routes get one sentence too."""
    if image_note:
        question = f"{image_note}\n\n{question}"
    if sensor_text is None:
        return question
    # KeyError here is intentional: tools/preflight.py checks coverage, so a
    # missing frame means the wrong file was passed.
    report = mark_queried_objects(sensor_text[frame_token], question)
    return f"{SENSOR_TEXT_HEADER}\n{report}\n{SENSOR_TEXT_FOOTER}\n\n{question}"


QUESTION_OBJ = re.compile(r"<(c\d+),(CAM_[A-Z_]+),([\d.]+),([\d.]+)>")
REPORT_POS = re.compile(r"<(CAM_[A-Z_]+),([\d.]+),([\d.]+)>")
MARK_MAX_DIST = 0.15  # normalised image units (width-scaled); farther = no mark


def mark_queried_objects(report: str, question: str) -> str:
    """Tag the report line nearest to each object the question names, e.g.
    "(= c1)". Uses only the report text and the question's own coordinates,
    as any deployed prompt builder could, so every text condition (including
    the shuffled control) gets the same treatment."""
    lines = report.split("\n")
    positions = {}
    for i, line in enumerate(lines):
        m = REPORT_POS.search(line)
        if m:
            positions[i] = (m.group(1), float(m.group(2)), float(m.group(3)))
    tags = {}
    for tag, cam, u, v in QUESTION_OBJ.findall(question):
        u, v = float(u), float(v)
        dist = {i: math.hypot(pu - u, (pv - v) * 9 / 16)
                for i, (pcam, pu, pv) in positions.items() if pcam == cam}
        if dist:
            best = min(dist, key=dist.get)
            if dist[best] <= MARK_MAX_DIST:
                tags.setdefault(best, []).append(tag)
    for i, t in tags.items():
        lines[i] += f" (= {', '.join(t)})"
    return "\n".join(lines)
