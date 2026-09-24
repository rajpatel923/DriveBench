"""
Re-score MCQ predictions with both the legacy (upstream DriveBench) and the
fixed answer parser, side by side.

The upstream parser lowercases the answer and then matches r"\b(a|option a)\b",
so any prediction containing the English article "a" is scored as option A.
The legacy function is copied verbatim below so this comparison stays
reproducible after evaluate/utils.py is fixed.

Usage:
    python tools/rescore.py res/qwen2.5-vl-7b/pilot_*.json
    python tools/rescore.py --dump 20 res/qwen2.5-vl-7b/pilot_clean.json
"""

import os
import re
import sys
import json
import random
import argparse
from collections import defaultdict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluate.utils import preprocess_answer


def preprocess_answer_legacy(answer: str) -> str:
    """Upstream DriveBench parser, copied verbatim (buggy: matches the article "a")."""
    answer = answer.strip().lower()

    choice_patterns = {
        "A": r"\b(a|option a)\b",
        "B": r"\b(b|option b)\b",
        "C": r"\b(c|option c)\b",
        "D": r"\b(d|option d)\b"
    }

    for choice, pattern in choice_patterns.items():
        if re.search(pattern, answer):
            return choice

    return ""


def is_mcq(item: dict) -> bool:
    return item["question_type"] in ("perception", "behavior") and 0 in item["tag"]


def rescore(path: str) -> dict:
    """Return {task: {n, old, new, unparsed}} for one prediction file."""
    data = [x for x in json.load(open(path)) if is_mcq(x) and x.get("pred")]
    stats = defaultdict(lambda: [0, 0, 0, 0])  # n, old_correct, new_correct, new_unparsed
    for x in data:
        for task in (x["question_type"], "overall"):
            s = stats[task]
            new = preprocess_answer(x["pred"])
            s[0] += 1
            s[1] += preprocess_answer_legacy(x["pred"]) == preprocess_answer_legacy(x["answer"])
            s[2] += new == preprocess_answer(x["answer"])
            s[3] += new == ""
    return {task: {"n": n, "old": o / n, "new": c / n, "unparsed": u / n}
            for task, (n, o, c, u) in stats.items()}


def dump(path: str, k: int, seed: int = 0):
    """Print k random MCQ predictions with both parses, for hand-checking."""
    data = [x for x in json.load(open(path)) if is_mcq(x) and x.get("pred")]
    random.Random(seed).shuffle(data)
    print(f"\n### {path}: {k} random MCQ predictions\n")
    for x in data[:k]:
        pred = " ".join(x["pred"].split())[:160]
        print(f"- gt={x['answer']} old={preprocess_answer_legacy(x['pred']) or '-'} "
              f"new={preprocess_answer(x['pred']) or '-'} | {pred}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Prediction JSON files")
    parser.add_argument("--dump", type=int, default=0,
                        help="Also print N random predictions per file for hand-checking")
    args = parser.parse_args()

    print("| file | task | n | old acc | new acc | unparsed (new) |")
    print("|---|---|---|---|---|---|")
    for path in args.files:
        for task, s in sorted(rescore(path).items()):
            print(f"| {os.path.basename(path)} | {task} | {s['n']} | "
                  f"{s['old']:.3f} | {s['new']:.3f} | {s['unparsed']:.3f} |")

    if args.dump:
        for path in args.files:
            dump(path, args.dump)


if __name__ == "__main__":
    main()
