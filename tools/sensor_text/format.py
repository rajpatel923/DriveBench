"""
The one shared formatter: every sensor-text source (oracle, cluster, shuffled)
goes through format_report(), so sources differ only in which objects they
found, never in wording.

An object is a dict:
    label        str    e.g. "car" (oracle) or "car-sized object" (cluster)
    x, y         float  ego frame, metres (x forward, y left)
    z            float  height of the object centre above ground, metres
    speed        float | None   m/s (None = no velocity measurement)
    vx, vy       float | None   velocity in ego axes, m/s
    cam, u, v    str, float, float | None   best camera, normalized coords
"""

import json
import math
import os
from typing import Dict, List, Optional

from tools.sensor_text.common import MAX_RANGE, OUT_DIR

DEFAULT_MAX_OBJECTS = 15
STATIONARY_SPEED = 0.5  # m/s

TURN_RATE = 5.0  # deg/s; below this a moving object is "going straight"

# The model otherwise looks for an exact coordinate match and refuses when it
# finds none (22% of W_oracle MCQs in the first full run).
HEADER = ("Ego frame: x forward, y left, in metres. Image positions use the "
          "same <camera,x,y> convention as the question, but are approximate "
          "object centres: an object in the question is the listed object "
          "nearest to its position, not an exact coordinate match.")

_SECTORS = ["front", "front-left", "left", "back-left",
            "back", "back-right", "right", "front-right"]


def direction(x: float, y: float) -> str:
    """Eight-way compass direction in the ego frame."""
    angle = math.degrees(math.atan2(y, x))  # 0 = ahead, +90 = left
    return _SECTORS[int(((angle + 22.5) % 360) // 45)]


def describe_motion(obj: dict) -> str:
    speed = obj.get("speed")
    if speed is None:
        return "speed unknown"
    if speed < STATIONARY_SPEED:
        return "stationary"
    text = f"moving {speed:.1f} m/s toward {direction(obj['vx'], obj['vy'])}"
    yaw_rate = obj.get("yaw_rate")  # deg/s, +left; oracle only, from past boxes
    if yaw_rate is not None:
        if abs(yaw_rate) < TURN_RATE:
            text += ", going straight"
        else:
            text += f", turning {'left' if yaw_rate > 0 else 'right'} ({abs(yaw_rate):.0f} deg/s)"
    return text


def format_object(obj: dict) -> str:
    dist = math.hypot(obj["x"], obj["y"])
    line = (f"- {obj['label']}, {dist:.1f} m {direction(obj['x'], obj['y'])} "
            f"(x={obj['x']:+.1f}, y={obj['y']:+.1f}), {describe_motion(obj)}")
    if obj.get("cam"):
        line += f", at <{obj['cam']},{obj['u']:.4f},{obj['v']:.4f}>"
    return line


# Class labels (oracle) that go ahead of static clutter. Cluster labels are
# size-only, so none match and the cluster list stays nearest-first.
ROAD_USERS = {"car", "truck", "bus", "trailer", "construction vehicle", "emergency vehicle",
              "motorcycle", "bicycle", "pedestrian", "animal"}


def select_objects(objects: List[dict], max_objects: int = DEFAULT_MAX_OBJECTS) -> List[dict]:
    """Within MAX_RANGE, road users first, then nearest-first, capped. This is
    exactly the list the text describes (saved to <source>.objects.json).
    Without the road-user priority, nearby cones/barriers pushed 14 of the 200
    perception-MCQ target objects out of the oracle text."""
    kept = [o for o in objects if math.hypot(o["x"], o["y"]) <= MAX_RANGE]
    kept.sort(key=lambda o: (o["label"] not in ROAD_USERS, math.hypot(o["x"], o["y"])))
    return kept[:max_objects]


def format_report(objects: List[dict]) -> str:
    """Format an already-selected object list (see select_objects)."""
    if not objects:
        return f"{HEADER}\nNo objects detected within {MAX_RANGE:.0f} m."
    return "\n".join([HEADER] + [format_object(o) for o in objects])


def write_source(source: str, objects_by_frame: Dict[str, List[dict]],
                 max_objects: int = DEFAULT_MAX_OBJECTS, out_dir: str = OUT_DIR) -> Dict[str, str]:
    """Select, format and write <source>.json and <source>.objects.json."""
    selected = {tok: select_objects(objs, max_objects) for tok, objs in objects_by_frame.items()}
    texts = {tok: format_report(objs) for tok, objs in selected.items()}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{source}.json"), "w") as f:
        json.dump(texts, f, indent=1)
    with open(os.path.join(out_dir, f"{source}.objects.json"), "w") as f:
        json.dump(selected, f, indent=1)
    print(f"Wrote {len(texts)} frames to {out_dir}/{source}.json")
    log_lengths(source, texts, selected)
    return texts


def log_lengths(source: str, texts: Dict[str, str], selected: Optional[Dict[str, List[dict]]] = None,
                tokenizer: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> dict:
    """Print (and return) mean/max text length in tokens, falling back to a
    whitespace-word count if the Qwen tokenizer isn't available locally."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
        lengths = [len(tok(t)["input_ids"]) for t in texts.values()]
        unit = "tokens"
    except Exception:
        lengths = [len(t.split()) for t in texts.values()]
        unit = "words (tokenizer unavailable)"
    stats = {"mean": sum(lengths) / len(lengths), "max": max(lengths), "unit": unit}
    if selected is not None:
        counts = [len(v) for v in selected.values()]
        stats["objects_mean"] = sum(counts) / len(counts)
        stats["objects_empty_frames"] = sum(c == 0 for c in counts)
    print(f"  [{source}] length: mean {stats['mean']:.0f} / max {stats['max']} {unit}"
          + (f"; objects/frame mean {stats['objects_mean']:.1f}, "
             f"empty frames {stats['objects_empty_frames']}" if selected is not None else ""))
    return stats
