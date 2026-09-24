"""
Stage 1 report: MCQ accuracy per run, paired deltas vs the blank-camera
baseline, gap closed, frame-level bootstrap CIs, and the two paper figures.

  gap closed = (acc(run) - acc(C1_blank)) / (acc(C0_clean) - acc(C1_blank))

CIs come from resampling the 200 frames with replacement (2,000 reps,
seed 0). The same resamples are used for every run, so deltas and gap-closed
CIs are paired.

Usage:
    python tools/stage1_report.py
    python tools/stage1_report.py --results-dir results/stage1/qwen2.5-vl-7b_limit20
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluate.utils import preprocess_answer
from tools.rescore import is_mcq

RUNS = ["C0_clean", "C1_blank", "W_oracle", "W_shuffled", "P_lidar", "W_cluster",
        "C0b_clean_rerun", "S05", "S10", "S20", "S10_cluster",
        "P_lidar2", "E_blank_ego", "W_oracle_ego", "W_cluster_ego", "P_lidar2_ego"]
CLEAN, BLANK = "C0_clean", "C1_blank"
EXPECTED_MCQ = 400
TASKS = ("overall", "perception", "behavior")
STALE = [("C0_clean", 0.0), ("S05", 0.5), ("S10", 1.0), ("S20", 2.0)]
N_OPTIONS = {"perception": 3, "behavior": 4}


def majority_baseline(data_path):
    """Accuracy of always answering each task's most common letter."""
    from collections import Counter
    mcq = [q for q in json.load(open(data_path)) if is_mcq(q)]
    out = {}
    for t in ("perception", "behavior"):
        answers = [q["answer"].strip() for q in mcq if q["question_type"] == t]
        letter, hits = Counter(answers).most_common(1)[0]
        out[t] = {"letter": letter, "acc": hits / len(answers), "n": len(answers)}
    n = sum(v["n"] for v in out.values())
    out["overall"] = {"acc": sum(v["acc"] * v["n"] for v in out.values()) / n, "n": n}
    return out


def load_run(path):
    """Return (per-frame {task: [correct, n]}, stats dict, {key: pred})."""
    data = json.load(open(path))
    per_frame = defaultdict(lambda: {t: [0, 0] for t in TASKS})
    counts = {t: [0, 0, 0, 0.0] for t in TASKS}  # correct, n, unparsed, chance credit
    by_answer = defaultdict(lambda: [0, 0])  # (task, gold letter) -> correct, n
    preds = {}
    for item in data:
        key = (item["frame_token"], item["question_type"], item["question"], is_mcq(item))
        preds.setdefault(key, []).append(item.get("pred") or "")
        if not is_mcq(item):
            continue
        letter = preprocess_answer(item.get("pred") or "")
        ok = int(letter == item["answer"].strip())
        for t in ("overall", item["question_type"]):
            per_frame[item["frame_token"]][t][0] += ok
            per_frame[item["frame_token"]][t][1] += 1
            counts[t][0] += ok
            counts[t][1] += 1
            counts[t][2] += int(letter == "")
            # A refusal scored as a uniform guess over the options (A-C or A-D).
            counts[t][3] += (1 / N_OPTIONS[item["question_type"]]) if letter == "" else 0.0
        by_answer[(item["question_type"], item["answer"].strip())][0] += ok
        by_answer[(item["question_type"], item["answer"].strip())][1] += 1
    nan = float("nan")
    stats = {t: {"acc": c / n if n else nan, "n": n, "unparsed": u / n if n else nan,
                 "acc_refusal_as_guess": (c + g) / n if n else nan}
             for t, (c, n, u, g) in counts.items()}
    # Balanced accuracy: mean per-gold-letter recall. Always giving one answer
    # scores 1/N_OPTIONS here, so it is not inflated by the skewed perception
    # answers (A = "going ahead" is 59% of them).
    for t in N_OPTIONS:
        recalls = [c / n for (task, _), (c, n) in by_answer.items() if task == t and n]
        stats[t]["balanced_acc"] = sum(recalls) / len(recalls) if recalls else nan
    return dict(per_frame), stats, preds


def frame_matrix(per_frame, frames, task):
    """[F, 2] array of (correct, n) per frame, zeros for absent frames."""
    return np.array([per_frame.get(f, {task: [0, 0]})[task] for f in frames], dtype=float)


def boot_acc(mat, idx):
    """Accuracy for each bootstrap resample. mat [F,2], idx [B,F] -> [B]."""
    c, n = mat[idx, 0].sum(1), mat[idx, 1].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return c / n


def ci(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def agreement(a, b):
    """MCQ-letter and exact-string agreement between two runs' predictions."""
    keys = sorted(set(a) & set(b))
    exact = letter = n_mcq = 0
    for k in keys:
        pa, pb = a[k], b[k]
        exact += pa == pb
        if k[3]:
            n_mcq += 1
            letter += [preprocess_answer(p) for p in pa] == [preprocess_answer(p) for p in pb]
    return {"n": len(keys), "exact": exact / len(keys) if keys else float("nan"),
            "n_mcq_keys": n_mcq, "mcq_letter": letter / n_mcq if n_mcq else float("nan")}


def fmt(x, lo_hi=None, pct=True):
    s = f"{100 * x:.1f}" if pct else f"{x:.2f}"
    if lo_hi:
        lo, hi = lo_hi
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]" if pct else f" [{lo:.2f}, {hi:.2f}]"
    return s


def make_figures(report, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = report["runs"]
    acc = lambda r: runs[r]["overall"]["acc"]
    err = lambda r: [[acc(r) - runs[r]["overall"]["ci"][0]], [runs[r]["overall"]["ci"][1] - acc(r)]]

    # Fig 1: accuracy vs staleness, with blank and the sensor routes as reference lines.
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    pts = [(dt, r) for r, dt in STALE if r in runs]
    if pts:
        xs = [dt for dt, _ in pts]
        ys = [100 * acc(r) for _, r in pts]
        yerr = np.array([[100 * e[0][0], 100 * e[1][0]] for e in (err(r) for _, r in pts)]).T
        ax.errorbar(xs, ys, yerr=yerr, marker="o", color="black", capsize=2, label="stale camera")
    styles = {BLANK: ("gray", ":", "blank camera"), "P_lidar": ("tab:blue", ":", "LiDAR image (1 sweep)"),
              "P_lidar2": ("tab:blue", "--", "LiDAR+radar image"),
              "W_oracle": ("tab:green", "--", "oracle text"), "W_cluster": ("tab:orange", "--", "cluster text")}
    for r, (color, ls, name) in styles.items():
        if r in runs:
            ax.axhline(100 * acc(r), color=color, ls=ls, lw=1, label=name)
    ax.set_xlabel("camera staleness (s)")
    ax.set_ylabel("MCQ accuracy (%)")
    ax.set_xticks([0, 0.5, 1, 2])
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_staleness.pdf"))
    plt.close(fig)

    # Fig 2: gap closed, pixels vs words vs controls.
    order = [("P_lidar", "LiDAR\n1 sweep", "lightsteelblue"), ("P_lidar2", "LiDAR\n+radar", "tab:blue"),
             ("W_cluster", "cluster\ntext", "tab:orange"), ("W_oracle", "oracle\ntext", "tab:green"),
             ("W_shuffled", "shuffled\ntext", "lightgray"), ("S10", "stale\n1 s", "black"),
             ("S10_cluster", "stale 1 s\n+cluster", "dimgray"), ("E_blank_ego", "ego\nonly", "tab:purple"),
             ("P_lidar2_ego", "LiDAR+radar\n+ego", "navy"), ("W_cluster_ego", "cluster\n+ego", "darkorange"),
             ("W_oracle_ego", "oracle\n+ego", "darkgreen")]
    order = [o for o in order if o[0] in report["gap_closed"]]
    fig, ax = plt.subplots(figsize=(3.4 if len(order) <= 7 else 7.0, 2.5))
    if order:
        g = [report["gap_closed"][r] for r, _, _ in order]
        vals = [x["value"] for x in g]
        yerr = np.array([[x["value"] - x["ci"][0], x["ci"][1] - x["value"]] for x in g]).T
        ax.bar(range(len(order)), vals, yerr=yerr, capsize=2, color=[c for _, _, c in order])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([n for _, n, _ in order], fontsize=6)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("gap closed (blank=0, clean=1)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_gap_closed.pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/stage1/qwen2.5-vl-7b_mcq")
    parser.add_argument("--out-dir", default=None, help="default: parent of --results-dir")
    parser.add_argument("--data", default="data/drivebench-mcq.json",
                        help="question file, for the always-majority baseline row")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.normpath(args.results_dir))

    loaded = {}
    for r in RUNS:
        path = os.path.join(args.results_dir, f"{r}.json")
        if not os.path.exists(path):
            print(f"warning: missing {path}")
            continue
        try:
            loaded[r] = load_run(path)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"warning: skipping unreadable {path} ({type(e).__name__})")
    if not loaded:
        sys.exit("No runs found.")

    frames = sorted(set().union(*(pf.keys() for pf, _, _ in loaded.values())))
    idx = np.random.default_rng(args.seed).integers(0, len(frames), size=(args.reps, len(frames)))

    report = {"results_dir": args.results_dir, "n_frames": len(frames), "reps": args.reps,
              "seed": args.seed, "runs": {}, "delta_vs_blank": {}, "gap_closed": {}}
    boots = {}
    for r, (pf, stats, _) in loaded.items():
        if stats["overall"]["n"] != EXPECTED_MCQ:
            print(f"warning: {r} has {stats['overall']['n']} MCQs (expected {EXPECTED_MCQ})")
        report["runs"][r] = {}
        for t in TASKS:
            b = boot_acc(frame_matrix(pf, frames, t), idx)
            boots[(r, t)] = b
            report["runs"][r][t] = dict(stats[t], ci=ci(b))

    if BLANK in loaded:
        for r in loaded:
            if r == BLANK:
                continue
            d = boots[(r, "overall")] - boots[(BLANK, "overall")]
            report["delta_vs_blank"][r] = {
                "value": report["runs"][r]["overall"]["acc"] - report["runs"][BLANK]["overall"]["acc"],
                "ci": ci(d)}
    if BLANK in loaded and CLEAN in loaded:
        a_clean, a_blank = (report["runs"][x]["overall"]["acc"] for x in (CLEAN, BLANK))
        denom_b = boots[(CLEAN, "overall")] - boots[(BLANK, "overall")]
        for r in loaded:
            if r in (CLEAN, BLANK):
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                g = (boots[(r, "overall")] - boots[(BLANK, "overall")]) / denom_b
            value = (report["runs"][r]["overall"]["acc"] - a_blank) / (a_clean - a_blank) \
                if a_clean != a_blank else float("nan")
            report["gap_closed"][r] = {"value": value, "ci": ci(g)}
    if CLEAN in loaded and "C0b_clean_rerun" in loaded:
        report["c0_c0b_agreement"] = agreement(loaded[CLEAN][2], loaded["C0b_clean_rerun"][2])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [f"# Stage 1 report", "",
             f"Results: `{args.results_dir}`; {len(frames)} frames; frame-level bootstrap, "
             f"{args.reps} reps, seed {args.seed}; 95% CIs in brackets.", "",
             "| run | n | overall | perception | behavior | unparsed | overall, refusal = guess "
             "| perception, balanced | behavior, balanced | Δ vs blank | gap closed |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in RUNS:
        if r not in report["runs"]:
            continue
        s = report["runs"][r]
        d = report["delta_vs_blank"].get(r)
        g = report["gap_closed"].get(r)
        lines.append(" | ".join([
            f"| {r}", str(s["overall"]["n"]),
            *(fmt(s[t]["acc"], s[t]["ci"]) for t in TASKS),
            fmt(s["overall"]["unparsed"]),
            fmt(s["overall"]["acc_refusal_as_guess"]),
            *(fmt(s[t]["balanced_acc"]) for t in N_OPTIONS),
            fmt(d["value"], d["ci"]) if d else "—",
            fmt(g["value"], g["ci"], pct=False) if g else "—"]) + " |")
    if os.path.exists(args.data):
        mb = report["majority_baseline"] = majority_baseline(args.data)
        lines.append(f"| always-majority ({mb['perception']['letter']}/{mb['behavior']['letter']}) | "
                     f"{mb['overall']['n']} | " + " | ".join(fmt(mb[t]["acc"]) for t in TASKS)
                     + " | — | — | " + " | ".join(fmt(1 / N_OPTIONS[t]) for t in N_OPTIONS)
                     + " | — | — |")
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
    lines += ["", "Balanced = mean accuracy over the gold answer letters; a constant answer "
                  "scores 33.3 (perception) / 25.0 (behavior)."]
    if "c0_c0b_agreement" in report:
        a = report["c0_c0b_agreement"]
        lines += ["", f"C0 vs C0b (greedy twice): MCQ letter agreement {fmt(a['mcq_letter'])}% "
                      f"over {a['n_mcq_keys']} MCQs; exact-string agreement {fmt(a['exact'])}% over {a['n']} questions."]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    make_figures(report, out_dir)
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/report.json, report.md, fig1_staleness.pdf, fig2_gap_closed.pdf")


if __name__ == "__main__":
    main()
