"""
Shuffled-text control: every frame receives another frame's report, always
from a different scene (seeded derangement). Same length and style
distribution as the source, wrong content. If W_shuffled scores like
W_oracle, the model is responding to the presence of text, not its content.

Usage:
    python tools/sensor_text/shuffle.py   # oracle.json -> oracle_shuffled.json
"""

import os
import sys
import json
import random
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.sensor_text.common import DATA, OUT_DIR, load_frames
from tools.sensor_text.format import log_lengths


def cross_scene_derangement(frames: dict, seed: int, max_tries: int = 10000) -> dict:
    """Return {frame: donor_frame} with donor always from a different scene."""
    tokens = sorted(frames)
    rng = random.Random(seed)
    for _ in range(max_tries):
        donors = tokens[:]
        rng.shuffle(donors)
        if all(frames[t] != frames[d] for t, d in zip(tokens, donors)):
            return dict(zip(tokens, donors))
    raise RuntimeError("No cross-scene derangement found; is one scene > half the frames?")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="oracle")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames = load_frames(args.data)
    texts = json.load(open(os.path.join(args.out_dir, f"{args.source}.json")))
    objects = json.load(open(os.path.join(args.out_dir, f"{args.source}.objects.json")))
    mapping = cross_scene_derangement(frames, args.seed)

    name = f"{args.source}_shuffled"
    with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
        json.dump({t: texts[d] for t, d in mapping.items()}, f, indent=1)
    with open(os.path.join(args.out_dir, f"{name}.objects.json"), "w") as f:
        json.dump({t: objects[d] for t, d in mapping.items()}, f, indent=1)
    with open(os.path.join(args.out_dir, f"{name}.mapping.json"), "w") as f:
        json.dump(mapping, f, indent=1)
    print(f"Wrote {len(mapping)} frames to {args.out_dir}/{name}.json (seed {args.seed})")
    log_lengths(name, {t: texts[d] for t, d in mapping.items()})


if __name__ == "__main__":
    main()
