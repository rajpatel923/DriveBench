# Pilot Study: Does Temporal Frame Recovery Beat the DriveBench Base Model?

> **SUPERSEDED (2026-09-23).** The MCQ accuracies below were scored with the
> upstream DriveBench parser, which lowercases the prediction and matches
> `\b(a|option a)\b` — so any answer containing the article "a" (e.g. "C. …
> appears to be a white van") is scored as option A. That is why clean and all
> five recovery methods tie at 0.625 on perception MCQ. The pilot also used
> T=0.2 sampling and mixed image resolutions (NoImage and LiDAR at 224², others
> native). Do not cite any number in §3–4. Stage 1 replaces this pilot.
>
> Re-scored with the fixed parser (`python tools/rescore.py res/qwen2.5-vl-7b/pilot_*.json`).
> Only 8 perception and 18 behavior MCQs exist in the 18-frame subset, so one
> answer moves perception by 0.125:
>
> | condition | perception old → new | behavior old → new | overall old → new | unparsed (new) |
> |---|---|---|---|---|
> | clean | 0.625 → 0.375 | 0.444 → 0.444 | 0.500 → 0.423 | 0.000 |
> | noimage | 0.375 → 0.125 | 0.389 → 0.278 | 0.385 → 0.231 | 0.154 |
> | previous | 0.625 → 0.375 | 0.444 → 0.444 | 0.500 → 0.423 | 0.000 |
> | nearest | 0.625 → 0.375 | 0.444 → 0.444 | 0.500 → 0.423 | 0.000 |
> | linear_blend | 0.625 → 0.375 | 0.389 → 0.389 | 0.462 → 0.385 | 0.000 |
> | rife | 0.625 → 0.375 | 0.389 → 0.389 | 0.462 → 0.385 | 0.000 |
> | lidar | 0.625 → 0.375 | 0.167 → 0.222 | 0.308 → 0.269 | 0.000 |
>
> The legacy parser adds +25 points to perception MCQ on every image condition,
> and on NoImage it hides a 50% unparsed rate (the model refuses without an
> image, and the refusal text contains "a"). Automated consistency check: for
> every prediction of the form "X. <option text>" in all 7 files (26/26, 20/26
> on NoImage), the parsed letter matches the option text the model wrote; the
> only unparsed outputs are NoImage refusals. A manual 20-per-condition check
> (`tools/rescore.py --dump 20`) is still on the validity checklist.

**Date:** 2026-09-20
**Model:** Qwen2.5-VL-7B-Instruct (vLLM, `max_model_len=16384`)
**Data:** 18-frame blob-01 subset (`data/blob01_subset.json`), 102 QA pairs, 7 conditions
**Purpose:** Sanity-check temporal recovery before committing GPU time to the full 200-frame / 1461-question DriveBench run.

> This is a **pilot**, not a final result. 18 frames is a small sample — MCQ accuracy values below are quantized in steps of ~1/16–1/18 (i.e. one flipped answer moves a score by ~0.06), so treat exact numbers as directional, not conclusive. The full run (`script/03_run_inference_linux.sh`, no `--pilot` flag) is needed to confirm.

---

## 1. Background

DriveBench evaluates VLMs on nuScenes driving QA under various corruption conditions. `NoImage` — where the camera frames for a scene are simply dropped — is the condition that stands in for "what happens to the base model when it gets no visual input at all." This project adds a preprocessing step: instead of feeding the model nothing, **reconstruct the missing frames** from temporal neighbors (previous keyframe / nearest keyframe / linear blend / RIFE optical-flow interpolation / LiDAR point-cloud projection) and feed *those* in.

The question this pilot answers: **does recovering the frame actually help the model answer better than giving it nothing (`noimage`), and how close does it get to feeding the real, uncorrupted frame (`clean`)?**

## 2. Recovery image quality (pixel-level, vs. ground truth)

Before even running the VLM, each recovery method's *output image* was compared against the real frame it's standing in for (`script/02_validate_recovery.sh`, 1800 image comparisons across all cameras/frames where a method produced output):

| Method | PSNR (dB) ↑ | SSIM ↑ | L1 error ↓ |
|---|---|---|---|
| Previous frame (copy) | 22.98 | 0.635 | — |
| Nearest keyframe | 23.51 | 0.657 | — |
| Linear blend | 24.91 | 0.675 | — |
| **RIFE (optical flow)** | **29.85** | **0.855** | lowest |
| LiDAR projection | *(not directly comparable — sparse point-cloud reprojection, not evaluated on the same pixel-metric scale)* | | |

RIFE is unambiguously the best pixel-level reconstruction — ~5–7 dB better PSNR and noticeably higher structural similarity than the simpler strategies. This is the expected ranking (video-frame-interpolation network vs. naive copy/blend).

## 3. Does it translate into better VLM answers?

Ran all 7 conditions through Qwen2.5-VL-7B and scored with `evaluate/eval.py` (DriveBench's own metric suite: MCQ accuracy for perception/behavior, BLEU/ROUGE-L/CIDEr for open-ended VQA in perception/prediction/planning).

### Headline metrics

| Condition | Perception MCQ Acc | Perception ROUGE-L | Prediction ROUGE-L | Planning ROUGE-L | Behavior MCQ Acc |
|---|---|---|---|---|---|
| **clean** (ceiling — real frames) | **0.625** | 0.175 | 0.301 | 0.098 | 0.444 |
| **noimage** (DriveBench base model — no recovery) | 0.375 | 0.059 | 0.282 | 0.044 | 0.389 |
| previous | 0.625 | 0.186 | 0.321 | 0.075* | 0.444 |
| nearest | 0.625 | 0.182 | 0.316 | 0.078* | 0.444 |
| linear_blend | 0.625 | 0.191 | 0.310 | 0.076* | 0.389 |
| rife | 0.625 | 0.190 | 0.327 | 0.080* | 0.389 |
| lidar | 0.625 | 0.167 | 0.317 | 0.072* | **0.167** |

\* planning scores are within noise of each other at n=18 and shouldn't be over-read.

Full per-condition JSON (all BLEU-1..4/ROUGE-L/CIDEr, all task categories) is in `res/qwen2.5-vl-7b/eval_scores/pilot_<condition>_final_scores.json`.

### Reading the results

- **Perception accuracy — clear win.** `noimage` scores 0.375 vs. `clean`'s 0.625. **Every recovery method, including the simplest (`previous`), recovers all the way to 0.625** — matching the clean ceiling exactly. This is the strongest, most consistent signal in the pilot: giving the model *any* plausible reconstruction of the missing frame beats giving it nothing, and at this sample size even naive strategies close the gap completely.
- **Perception language quality (ROUGE-L) — also a clear win.** `noimage` 0.059 vs. recovery methods 0.167–0.191, all of which land at or above `clean` (0.175). Recovered frames give the model enough grounding to produce fuller free-text descriptions, not just pick the right MCQ letter.
- **Prediction quality — consistent improvement.** `noimage` 0.282 vs. recovery methods 0.310–0.327 (all above `clean`'s 0.301). Temporal continuity from neighbor frames appears to help reasoning about what happens next, slightly more than a single clean frame does.
- **Behavior MCQ — mixed, and this is where RIFE's pixel-quality lead does *not* show up.** `previous`/`nearest` match `clean` (0.444). `linear_blend`/`rife` merely tie `noimage` (0.389) — no improvement despite RIFE's much better PSNR/SSIM. **`lidar` is the outlier: 0.167, worse than doing nothing (`noimage` 0.389).** At n=18 this could be 2–3 flipped answers rather than a real deficiency, but it's the one result in the pilot that doesn't fit the "recovery helps" pattern and is worth confirming on the full run.
- **Planning** is flat/noisy across every condition (0.044–0.098) — the category has the fewest questions in the pilot subset, too small a signal to draw conclusions from yet.

## 4. Bottom line

On this pilot: **yes, temporal frame recovery beats the `noimage` base model**, most clearly and consistently on perception (every recovery strategy matches the clean-frame ceiling exactly) and on prediction quality. This supports proceeding to the full 200-frame run.

One thing to watch in the full run: **LiDAR's behavior-MCQ score underperforming `noimage`**, despite LiDAR being the physically most "real" recovery signal (actual depth-projected geometry rather than interpolated pixels). The full run's larger per-category sample size will show whether that's pilot noise or a genuine weakness of LiDAR-reprojected frames for behavior-reasoning questions specifically.

## 5. Repro / how to reproduce these numbers

```bash
# 1. Build all recovery conditions (already done for this repo)
bash script/01_build_recovery.sh --with-rife --with-lidar

# 2. Validate pixel-level recovery quality (Section 2 table)
bash script/02_validate_recovery.sh

# 3. Run the 18-frame pilot inference sweep across all 7 conditions
bash script/03_run_inference_linux.sh --pilot --with-rife --with-lidar

# 4. Score each condition against DriveBench's metric suite
for c in clean noimage previous nearest linearblen rife lidar; do
    python evaluate/eval.py "res/qwen2.5-vl-7b/pilot_${c}.json"
done
# -> writes res/qwen2.5-vl-7b/eval_scores/pilot_<condition>_final_scores.json
```

## 6. Next steps

- [ ] Run the full dataset (200 frames / 1461 questions): `bash script/03_run_inference_linux.sh --with-rife --with-lidar` (no `--pilot`)
- [ ] Confirm whether LiDAR's behavior-MCQ weakness holds up at full scale
- [ ] Once Qwen2.5-VL-7B's full-scale numbers are in, repeat the sweep with the other two backbones (InternVL2.5-8B, LLaVA-1.5-7B — weights already downloadable via `script/00b_download_vlm_weights.sh`) to see whether the recovery benefit is model-agnostic or specific to Qwen2.5-VL
- [ ] Use `script/04_compare_conditions.sh` to confirm QA-pair coverage parity across all conditions before treating full-run numbers as final
