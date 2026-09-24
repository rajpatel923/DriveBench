"""
Write the 400 scored MCQs (perception + behavior, tag 0) to their own data
file. Stage 1 only scores these, and each question is answered independently
with greedy decoding, so running just them gives the same MCQ predictions
~4x faster than running all 1,461 questions.

Usage:
    python tools/make_mcq_data.py   # -> data/drivebench-mcq.json
"""

import os
import sys
import json
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.rescore import is_mcq


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/drivebench-test-final.json")
    parser.add_argument("--out", default="data/drivebench-mcq.json")
    args = parser.parse_args()

    mcq = [q for q in json.load(open(args.data)) if is_mcq(q)]
    with open(args.out, "w") as f:
        json.dump(mcq, f, indent=1)
    print(f"Wrote {len(mcq)} MCQs over {len({q['frame_token'] for q in mcq})} frames to {args.out}")


if __name__ == "__main__":
    main()
