"""
Paired per-item comparison across DriveBench conditions (e.g.
clean vs. framelost vs. noimage vs. recovered), as called for by
claude/PLAN.md's Finding 2 and claude/TEMPORAL_RECOVERY_RESEARCH.md section 3:
the aggregate gpt_score can mute a real grounding improvement if a
recovered-but-imperfect frame still produces a plausible-sounding answer, so
the paired per-item deltas are the more honest evidence, not just
evaluate/eval.py's printed final_scores.

Reads:
  - res/<model>/<condition>.json                          (raw predictions)
  - res/<model>/gpt_eval_logs/<condition>_eval_log.json    (per-item GPT scores, JSONL,
                                                             written by evaluate/request.py)
  - res/<model>/eval_scores/<condition>_final_scores.json  (aggregate scores, written by
                                                             the evaluate/eval.py patch)

Usage:
    python tools/compare_conditions.py --model llava-1.5-7b \
        --conditions clean framelost noimage recovered
"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict

SCORE_PATTERN = re.compile(r"Total Score:\s*(\d{1,3})\b")


def load_predictions(model_dir: str, condition: str):
    path = os.path.join(model_dir, f"{condition}.json")
    if not os.path.exists(path):
        return {}
    data = json.load(open(path))
    out = {}
    for item in data:
        key = (item["scene_token"], item["frame_token"], item["question"])
        out[key] = item
    return out


def load_gpt_log(model_dir: str, condition: str):
    """gpt_eval_logs/<condition>_eval_log.json is JSONL (one entry per line,
    per evaluate/request.py's GPTEvaluation.log), not a JSON array.
    """
    path = os.path.join(model_dir, "gpt_eval_logs", f"{condition}_eval_log.json")
    scores = {}
    if not os.path.exists(path):
        return scores
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (entry["scene_token"], entry["frame_token"], entry["question"])
            match = SCORE_PATTERN.search(entry.get("gpt_score", ""))
            if match:
                scores[key] = int(match.group(1))
    return scores


def load_final_scores(model_dir: str, condition: str):
    path = os.path.join(model_dir, "eval_scores", f"{condition}_final_scores.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model dir name under res/, e.g. llava-1.5-7b")
    parser.add_argument("--res-dir", default="res", help="Root results directory (default: res)")
    parser.add_argument("--conditions", nargs="+", required=True,
                         help="Condition names to compare, e.g. clean framelost noimage recovered")
    parser.add_argument("--baseline", default=None,
                         help="Condition to use as the paired-comparison baseline (default: first in --conditions)")
    args = parser.parse_args()

    model_dir = os.path.join(args.res_dir, args.model)
    baseline = args.baseline or args.conditions[0]

    preds = {c: load_predictions(model_dir, c) for c in args.conditions}
    gpt_scores = {c: load_gpt_log(model_dir, c) for c in args.conditions}
    final_scores = {c: load_final_scores(model_dir, c) for c in args.conditions}

    print(f"=== Aggregate final_scores (from evaluate/eval.py) ===")
    for c in args.conditions:
        status = "found" if final_scores[c] is not None else "MISSING -- run evaluate/eval.py first"
        print(f"  {c}: {status}")
    print()

    print(f"=== Paired per-item GPT-score deltas vs. baseline='{baseline}' ===")
    baseline_scores = gpt_scores.get(baseline, {})
    if not baseline_scores:
        print(f"  No GPT eval log found for baseline '{baseline}' "
              f"(expected res/{args.model}/gpt_eval_logs/{baseline}_eval_log.json) "
              f"-- run `evaluate/eval.py --eval-gpt` first.")
        return

    for c in args.conditions:
        if c == baseline:
            continue
        cond_scores = gpt_scores.get(c, {})
        common_keys = set(baseline_scores) & set(cond_scores)
        if not common_keys:
            print(f"  {c}: no overlapping scored items with baseline yet")
            continue
        deltas = [cond_scores[k] - baseline_scores[k] for k in common_keys]
        improved = sum(1 for d in deltas if d > 0)
        worsened = sum(1 for d in deltas if d < 0)
        unchanged = sum(1 for d in deltas if d == 0)
        print(f"  {c} vs {baseline} (n={len(deltas)} matched items):")
        print(f"    mean delta: {statistics.mean(deltas):+.2f}   median delta: {statistics.median(deltas):+.2f}")
        print(f"    improved: {improved} ({improved/len(deltas):.1%})   "
              f"worsened: {worsened} ({worsened/len(deltas):.1%})   "
              f"unchanged: {unchanged} ({unchanged/len(deltas):.1%})")

    # Per-question_type breakdown, using whichever condition has predictions
    # loaded (question_type is stable across conditions for the same item).
    qtype_by_key = {}
    for c in args.conditions:
        for key, item in preds[c].items():
            qtype_by_key.setdefault(key, item.get("question_type"))

    print()
    print(f"=== Per-question-type breakdown vs. baseline='{baseline}' ===")
    for c in args.conditions:
        if c == baseline:
            continue
        cond_scores = gpt_scores.get(c, {})
        common_keys = set(baseline_scores) & set(cond_scores)
        if not common_keys:
            continue
        by_qtype = defaultdict(list)
        for k in common_keys:
            qtype = qtype_by_key.get(k, "unknown")
            by_qtype[qtype].append(cond_scores[k] - baseline_scores[k])
        print(f"  {c} vs {baseline}:")
        for qtype, deltas in sorted(by_qtype.items()):
            print(f"    {qtype}: n={len(deltas)}  mean delta={statistics.mean(deltas):+.2f}")


if __name__ == "__main__":
    main()
