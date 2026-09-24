"""
Visual check: draw each object's <cam,u,v> from <source>.objects.json onto
the real camera images, so you can see whether the text points at the right
things. Writes one 3x2 contact sheet per frame.

Usage:
    python tools/sensor_text/viz.py --source oracle --n 3
    python tools/sensor_text/viz.py --source cluster --frames <token> <token>
"""

import os
import sys
import json
import random
import argparse

from PIL import Image, ImageDraw

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.sensor_text.common import CAMERAS, DATA, OUT_DIR, META_DIR, IMG_W, IMG_H, Meta, find_file, default_roots

TILE = (800, 450)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-dir", default=OUT_DIR)
    parser.add_argument("--out-dir", default="results/stage1/viz")
    args = parser.parse_args()

    objects = json.load(open(os.path.join(args.in_dir, f"{args.source}.objects.json")))
    frames = args.frames or random.Random(args.seed).sample(sorted(objects), args.n)
    meta = Meta(META_DIR)
    roots = default_roots()
    out_dir = os.path.join(args.out_dir, args.source)
    os.makedirs(out_dir, exist_ok=True)

    sx, sy = TILE[0] / IMG_W, TILE[1] / IMG_H
    for tok in frames:
        sheet = Image.new("RGB", (TILE[0] * 3, TILE[1] * 2))
        for i, cam in enumerate(CAMERAS):
            rec = meta.keyframe(tok, cam)
            img = Image.open(find_file(rec["filename"], roots)).convert("RGB").resize(TILE)
            draw = ImageDraw.Draw(img)
            draw.text((8, 8), cam, fill=(255, 255, 0))
            for o in objects[tok]:
                if o.get("cam") != cam:
                    continue
                u, v = o["u"] * IMG_W * sx, o["v"] * IMG_H * sy
                draw.ellipse((u - 6, v - 6, u + 6, v + 6), outline=(255, 0, 0), width=3)
                speed = "?" if o["speed"] is None else f"{o['speed']:.1f}"
                draw.text((u + 8, v - 8), f"{o['label'].split()[0]} {speed}", fill=(255, 64, 64))
            sheet.paste(img, ((i % 3) * TILE[0], (i // 3) * TILE[1]))
        path = os.path.join(out_dir, f"{tok}.jpg")
        sheet.save(path, quality=85)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
