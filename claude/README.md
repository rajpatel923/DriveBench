# Thesis Research Notes

Planning and feasibility research for a frame-recovery thesis built on top of
this DriveBench repo: detect a missing/corrupted camera frame, reconstruct
it (or substitute a sensor-derived description for it), and measure whether
downstream VQA accuracy improves versus the corrupted/missing frame as-is.
No implementation lives in this folder — these are analysis documents; the
coding scaffolding they point to is elsewhere in the repo (`tools/`,
`recovery/`, `evaluate/eval.py`, `script/llava1.5-7b-recovered.sh`).

Read in this order — each document builds on the one before it:

1. **[PLAN.md](./PLAN.md)** — Is DriveBench a suitable testbed for this
   thesis at all? Verifies the corruption-injection mechanism, per-item
   evaluation logging, and QA density; concludes yes, and lays out what
   would need to happen to build it.
2. **[LITERATURE_REVIEW.md](./LITERATURE_REVIEW.md)** — Nine-paper review
   building the case that frame loss is a real, unsolved, safety-critical
   problem, and that a recovery method is a credible contribution.
   Includes a reproduced result on this repo's own pipeline: clean vs.
   `NoImage` changes 82.8% of LLaVA-1.5-7B's predictions.
3. **[TEMPORAL_RECOVERY_RESEARCH.md](./TEMPORAL_RECOVERY_RESEARCH.md)** —
   Feasibility and expected-impact research for one concrete method:
   reconstructing the missing frame from real temporal neighbors. Scopes
   the problem to 2 of DriveBench's 17 conditions (`NoImage`, `FrameLost`
   only), identifies the local data gap (neighbor image bytes aren't
   downloaded yet), and recommends a nearest-sweep baseline before any
   interpolation model.
4. **[LIDAR_RADAR_TEXT_RECOVERY.md](./LIDAR_RADAR_TEXT_RECOVERY.md)** —
   Same two conditions, alternative method: converting LiDAR/radar sensor
   readings into a text scene description instead of reconstructing an
   image. Compares its odds and engineering cost directly against document
   3, and splits the verdict by sensor (LiDAR: meaningfully likely to help;
   radar alone: closer to a toss-up).

---

## Pilot Experiment Results (blob-01, 18 frames)

Results from the first end-to-end pipeline run. All numbers are on the
**blob-01 pilot subset** (18 keyframes × 6 cameras = 108 recovered images per
strategy; 5 QA questions used for the inference smoke-test). Full 200-frame
results require downloading additional nuScenes blobs.

### Recovery Image Quality (PSNR / SSIM vs. original camera frame)

| Strategy | PSNR (full) | PSNR (crop) | SSIM (full) | L1 |
|---|---|---|---|---|
| Previous sweep | 19.74 dB | 20.21 dB | 0.5095 | 16.62 |
| Nearest sweep | 20.27 dB | 20.74 dB | 0.5352 | 15.51 |
| Linear blend | 21.70 dB | 22.21 dB | 0.5560 | 13.82 |
| **RIFE HDv3** | **27.14 dB** | **28.60 dB** | **0.8189** | **7.30** |
| LiDAR projection | 6.86 dB | — | 0.0203 | 103.66 |

RIFE (learned optical-flow interpolation, ECCV 2022) outperforms all other
strategies by **+5.4 dB** over the next-best method (Linear blend). LiDAR's
pixel metrics are low by design — the output is a depth-colorised point-cloud
projection, not a natural camera image, so PSNR/SSIM do not capture its
semantic content.

The LiDAR row predates the Stage 1 rebuild (native 1600×900, radius-4
depth-sorted points) and will change.

### VLM Answer Quality — removed

The pilot VLM numbers previously listed here are withdrawn: they used T=0.2
sampling, mixed image resolutions, and (for MCQ) the upstream parser that
scores any answer containing the article "a" as option A. See the
SUPERSEDED note in `claude/PILOT_FINDINGS.md`. Stage 1 results will be
reported from `results/stage1/report.md`.

### Notes and Caveats

- n=5 questions is too small for statistical conclusions; variance is high.
  The full pilot (102 questions) and ideally the full 1461-question set are
  needed for publication-quality numbers.
- Additional nuScenes blobs (02, 03, 10) required to cover all 200 DriveBench
  keyframes.
- GPT semantic scoring pending an OpenAI API key.
- LiDAR projection needs a fairness note in the paper: pixel metrics (PSNR,
  SSIM) do not apply to cross-modal outputs.
