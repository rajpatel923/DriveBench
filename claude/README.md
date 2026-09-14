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
