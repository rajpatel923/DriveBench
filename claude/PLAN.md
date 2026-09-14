# Frame-Recovery Thesis: Feasibility Assessment on DriveBench

## Context

Thesis topic: improve VLM output accuracy in autonomous-driving settings by detecting a
missing/corrupted camera frame, generating a recovered frame (potentially using other
sensor modalities like lidar/radar), substituting it back into the model's input stack,
and measuring whether downstream VQA accuracy improves versus using the corrupted/missing
frame as-is. Prior reading includes DriveLM, RoboDrive, and other multi-fusion VLM papers
— none of which address lost/missing frames specifically.

Question asked: is the DriveBench repo
(`/home/rajpatel/documents/code/python_codes/DriveBench`) a suitable testbed for this
research? This document is a **feasibility answer only** — no code has been changed as
part of this analysis, and no implementation has been started.

## Findings (verified by reading the actual code and data, not assumed)

### 1. DriveBench already separates "missing frame" from "degraded frame"

Checked actual pixel content with PIL:
- `NoImage` and `data/corruption/FrameLost/` are literal blank/black frames
  (mean=0, std=0) — true missing-data conditions.
- `CameraCrash` and the other 14 corruption types (Fog, Rain, MotionBlur, Snow,
  Brightness, ColorQuant, LensObstacleCorruption, Saturate, ZoomBlur,
  WaterSplashCorruption, H256ABRCompression, LowLight, BitError) are real,
  content-bearing but degraded images — a different research condition (recovering
  from corruption vs. recovering from total loss).

All corruptions are handled by one generic code path in `inference/llava1.5.py`
(`LLMPredictor.__call__`):
- `NoImage`: synthesizes a blank `np.zeros((224,224,3))` array in memory, no file read.
- Everything else: `img_path.replace('nuscenes/samples', f'corruption/{self.corruption}')`,
  then reads the pre-baked JPEG at that swapped path.

This means a "recovered frame" condition could plug into the exact same mechanism —
add a new corruption-name branch (or a preprocessing step) that swaps in a
generated/recovered image instead of the blank or corrupted one.

### 2. Evaluation is granular enough for paired comparisons

`evaluate/eval.py`'s `EvaluationSuit` and `evaluate/request.py`'s `GPTEvaluation`
score and log results per individual `scene_token` + `frame_token` + `question`
(log key format: `f"{scene_token}_{frame_token}_{question}"`), not just aggregate
numbers across the whole dataset. The final `evaluation()` method aggregates these
into task/type-level scores for the printed report, but the underlying per-item
scores/logs already exist and are individually addressable.

This matters because the thesis experiment design is a **paired comparison**: same
frame, same question, only the image input changes (clean vs. missing vs. recovered).
That comparison is directly supported by the existing log structure — it just isn't
surfaced by the default aggregate printout, so a small script would be needed to pull
matched triples out of the logs.

### 3. Density: enough questions per frame to make recovery worthwhile

`data/drivebench-test-final.json` has 1,461 questions across 200 unique `frame_token`s
— about 7.3 questions per frame, spanning all 4 driving tasks (perception=400,
prediction=261, planning=600, behavior=200). Recovering one frame gives signal across
~7 different questions at once.

### 4. Lidar/radar/temporal data: reachable, but not currently present or wired up

- `data/nuscenes/samples/` currently has only the 6 camera folders (1,200 files,
  ~178MB) pulled by `tools/fetch_nuscenes_subset.py` — no `sweeps/`, no lidar, no radar.
- `nuscenes-devkit` is not installed in either conda env (`drivebench`, `drivebench5090`).
- However, the `v1.0-trainval` metadata (`sample.json`, `sample_data.json`) is already
  downloaded to `/mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/` from earlier dataset
  setup work. This metadata has the `prev`/`next` temporal links and per-sensor
  filenames needed to locate the matching LIDAR_TOP/RADAR `sample_data` entries for a
  given camera keyframe (same `sample_token`), and to walk to adjacent frames in time.
  It is just not used by any code yet.
- To actually get lidar/radar sensor data, the same nuscenes.org account used for the
  camera keyframes would need to download the lidar/radar "keyframe blobs" for the
  relevant scenes (same download-page mechanism already used, larger since lidar/radar
  data is a bigger share of each blob than camera-only keyframes).

### 5. No frame-recovery code exists anywhere — expected, this is the novel contribution

Repo-wide search for `diffusers`, `inpaint`, `gan`, `controlnet`, `stable diffusion`,
`nuscenes-devkit` across all `.py`/`.md`/`.txt`/`.yaml` files returned no matches. Package
lists in both conda envs confirmed no diffusion/GAN/nuscenes-devkit libraries installed
(`drivebench5090` does have `opencv-python-headless`, useful for classical
projection/inpainting baselines if desired).

## Conclusion

DriveBench is well-suited as the **evaluation harness** for this thesis:
- it already isolates true missing-frame conditions (`NoImage`, `FrameLost`) from
  degraded-frame conditions (everything else),
- it scores at per-frame/per-question granularity, enabling the paired
  clean-vs-missing-vs-recovered comparison the thesis needs,
- and it sits on top of nuScenes data where the lidar/radar/temporal linkage needed
  for a recovery technique is technically reachable — the metadata is already in hand.

The frame-recovery generation technique itself (however it's designed — point-cloud
projection, GAN, diffusion inpainting, temporal interpolation, etc.) and the work of
wiring it into the inference pipeline (feeding recovered frames into the same
`img_path.replace(...)` slot DriveBench already uses for corruptions) would be the
student's own novel contribution. DriveBench does not, and was never expected to,
provide that part.

## What would need to happen to actually build this (not started — for later review)

1. Download nuScenes lidar/radar sweep data for the 174 scenes already in
   `data/drivebench-test-final.json`, via the same nuscenes.org account.
2. Install `nuscenes-devkit` in `drivebench5090` to parse/project point clouds and
   resolve `prev`/`next`/`sample_token` links from the metadata already downloaded.
3. Design and implement the actual frame-recovery module (the thesis's core
   contribution).
4. Add a new corruption-like branch in `inference/llava1.5.py` (alongside `NoImage`
   and the existing path-substitution logic) that swaps in the recovered frame.
5. Run inference with the recovered condition as a new corruption type (e.g.
   `"Recovered"`) and compare against `FrameLost`/`NoImage`/`clean` baselines using
   `evaluate/eval.py`, plus a small script to extract matched per-item score triples
   from the GPT eval logs for the paired-comparison analysis described in Finding 2.

**Status: no code has been changed. This is a planning document only, for review
before any implementation work begins.**
