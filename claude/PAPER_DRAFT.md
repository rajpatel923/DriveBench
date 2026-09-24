# Pixels or Words? Feeding LiDAR and Radar to a Frozen Driving VLM After Total Camera Loss

> **Status:** draft for the AAAI-27 Student Abstract (deadline Mon Sep 28 2026 AoE, target Sun Sep 27).
> Numbers come from `results/stage1/report.md` (16 MCQ-only runs, 2026-09-24).
> Items marked **[verify]** need checking against the AAAI-27 call or the cited paper.
> Budget: AAAI student abstracts have historically been 2 pages plus 1 page of references **[verify]**, which is roughly 900–1,100 words of body text.

---

## Abstract

Vision-language models (VLMs) are being tested as driving assistants, but they assume a working camera. We ask what a *frozen* driving VLM should be given when every camera fails and only LiDAR, radar and odometry remain: the point cloud rendered as an **image**, a short **text** report of surrounding objects, or the last camera frame, now stale. On 400 DriveBench multiple-choice questions with Qwen2.5-VL-7B, a 2 s old frame keeps clean-camera accuracy. A text report from ground-truth boxes closes 1.36 of the blank-to-clean gap, but deployable routes do not: a report from unlabelled LiDAR clusters closes -0.64 and a 10-sweep LiDAR+radar image -0.18. A single line of past ego motion closes 2.31, beating the clean camera, almost entirely on behavior questions, which shows that much of DriveBench's behavior track is answerable without seeing the scene. Words beat pixels only when perception is perfect; until then, keep the last frame and the ego state.

---

## 1  Introduction

Driving VLMs answer questions about a scene from camera images (Xie et al. 2025; Sima et al. 2024). Cameras are also the sensor most likely to fail: glare, dirt, water and hardware faults can blank some or all views. Existing robustness work either measures the drop (Xie et al. 2025) or retrains the model to fuse other sensors (RoboDriveVLM 2025) **[verify this characterisation]**. Retraining is not always possible when the VLM is a frozen, off-the-shelf component.

A vehicle that loses its cameras usually still has LiDAR and radar. The question is how to get that information *into* a frozen VLM through the interfaces it already has, images and text. We compare three options:

- **Pixels:** project the LiDAR points (10 sweeps) into each camera view, mark moving radar returns, and give the model the resulting image.
- **Words:** describe the detected objects (class or size, position, motion, and image location) in a short text report.
- **Stale camera:** reuse the last good camera frame, 0.5–2 s old. This is the cheap default a system would fall back on.

**Contributions.** (1) A controlled comparison of image and text routes for LiDAR/radar under total camera loss, with a blank-camera floor, a clean-camera ceiling, and a shuffled-text control. (2) An oracle text ceiling, a detector-free deployable text baseline, and an image route that carries the same sensors (LiDAR + radar motion) as the text. (3) A separate ego-odometry condition that shows how much of the behavior score past ego motion alone explains. (4) Frame-level bootstrap CIs for every comparison, plus a parser audit showing that a common MCQ answer parser inflated pilot accuracy by 25 points.

---

## 2  Setup

**Model and data.** Qwen2.5-VL-7B-Instruct (Bai et al. 2025), frozen, greedy decoding (temperature 0, seed 0). DriveBench (Xie et al. 2025) test set: 200 nuScenes (Caesar et al. 2020) keyframes. We score the 400 multiple-choice questions: 200 perception questions ("what is the moving status of the object at ⟨camera, u, v⟩?": going ahead, turn left or turn right) and 200 behavior questions (ego-vehicle behavior, 4 options). All six camera images are resized to 672×378 in every condition, so image resolution is held constant. The system prompt is identical across conditions. Each question is answered in its own request, so running only the 400 MCQs gives the same answers as running the full test set. Always answering the most common option scores 43.2%. We report this majority baseline because it is close to clean accuracy.

**Conditions.**

| Label | Camera input | Extra text |
|---|---|---|
| Clean (C0) | current frames | — |
| Blank (C1) | black images | — |
| LiDAR image, 1 sweep (P) | LiDAR keyframe projected into each camera, depth-coloured | one-line legend |
| LiDAR + radar image (P2) | 10 LiDAR sweeps + moving radar returns as white pillars/arrows | one-line legend |
| Oracle text (W-oracle) | black images | report from ground-truth boxes |
| Cluster text (W-cluster) | black images | report from LiDAR clusters + radar |
| Shuffled text (W-shuf) | black images | oracle report from a frame in a *different* scene |
| Stale k (S0.5/S1/S2) | frames from 1, 2 or 4 keyframes earlier | — |
| Stale + cluster | S1 frames | cluster report |
| Ego only (E) | black images | ego-motion line from odometry |
| + ego (W-oracle+E, W-cluster+E, P2+E) | as above | ego line + that condition's input |

**LiDAR/radar images.** P uses only the keyframe LiDAR sweep. P2 follows standard nuScenes practice (the devkit's multi-sweep loader; the 10-sweep input of CenterPoint and BEVFusion): it adds the 9 previous sweeps (≈0.45 s), each moved into the current frame with its own ego pose, which gives about 10× more points. Radar measures radial (Doppler) velocity, so for traffic ahead the motion vector lies along the line of sight and is invisible when projected. Following CRF-Net (Nobis et al. 2019), each radar return moving faster than 0.5 m/s is drawn as a 2 m vertical white pillar, and gets an arrow showing one second of motion where that arrow is visible. Both LiDAR conditions put one fixed sentence before the question explaining the colours, the counterpart of the text reports' header. Only past sensor data is used.

**Text reports.** Each report lists up to 15 objects within 50 m, road users first, then nearest first. Each line gives the label, distance and direction, ego-frame position, motion ("stationary", "moving 4.1 m/s toward front", or "speed unknown"), and the object's image location in the same normalised `<camera,u,v>` format the questions use. Example:

```
- car, 11.8 m front-left (x=+11.2, y=+3.4 m), moving 4.1 m/s toward front, going straight, at <CAM_FRONT,0.4886,0.5481> (= c1)
```

A one-line header says that positions are approximate object centres and that a queried object is the nearest listed one. In the first run, the model often refused because it looked for an exact coordinate match. The prompt builder also tags the report line nearest each queried object with "(= cN)". It uses only the report and the question's own coordinates, so every text condition, including the shuffled control, gets the same treatment.

- *Oracle* reports use nuScenes ground-truth boxes with at least one LiDAR or radar return, with class labels. Velocity and turning rate come from the previous annotation only, as a tracker would have them; the next annotation would leak the object's future motion, which is what the perception questions ask. It is an upper bound on what perfect perception could supply. It includes the queried object for 161 of the 188 perception questions whose target matches a box (86%). The rest are beyond 50 m or cut by the 15-object cap.
- *Cluster* reports need no detector and no labels. LiDAR points 0.3–2.5 m above ground are grouped on a 0.3 m bird's-eye grid. Clusters are labelled by size only ("car-sized", "pedestrian-sized", …) and take the median ego-motion-compensated velocity of radar returns within 2 m. Moving radar returns with no cluster are reported separately.
- *Shuffled* reports reassign oracle reports across scenes with a fixed-seed derangement. Length and style are unchanged, but the content is wrong for the scene.

- *Ego* line (separate condition): speed, turning rate and acceleration over the past 0.5 s from the ego poses, e.g. "Ego vehicle (odometry, past 0.5 s): speed 5.2 m/s, turning left (7 deg/s), steady speed." The past turning direction matches the answer's steering in 166 of 200 behavior questions. Ego motion is a strong prior for "what will the car do next", so it is kept as its own condition (E alone, and added to each route) rather than mixed into the object reports.

Reports average 753 (oracle) and 770 (cluster) tokens.

**Stale frames.** Walking back k keyframes (k = 1, 2, 4) gives mean ages of 0.50, 1.00 and 1.89 s. For k = 4, 29 of 200 frames are near the start of their scene and use the earliest available frame (at least 0.95 s old).

**Metrics.** MCQ accuracy, with the answer letter parsed by a strict rule (leading letter, "answer is X", or a standalone capital A–D; otherwise unparsed and counted wrong). We also report accuracy with unparsed answers scored as a random guess (1/3 or 1/4), which separates refusals from wrong answers. *Gap closed* = (acc − acc_blank) / (acc_clean − acc_blank). 95% CIs come from a frame-level bootstrap (2,000 resamples of the 200 frames, shared across conditions so differences are paired).

---

**Pipeline (Figure 0).** Every condition passes through one prompt builder, so conditions differ only in content:

```
 nuScenes keyframe (t)                     offline builders (past data only)
 ├─ 6 cameras ──────┬─ clean / blank ───────────────────────────────┐
 │                  └─ stale: frame from t−0.5/1/2 s ───────────────┤ images
 ├─ LIDAR_TOP (+9 prev. sweeps) ─ project to 6 cams, depth colour ──┤ (672×378)
 ├─ 5 radars ─ moving returns → pillars/arrows (CRF-Net style) ─────┘
 │
 ├─ GT boxes (prev. annotation) ─┐
 ├─ LiDAR clusters + radar ──────┼─ one shared formatter ─ object report ─┐
 │  (cross-scene shuffle = ctrl) ┘   (≤15 objects, 50 m, <CAM,u,v>)      │ text
 └─ ego poses (past 0.5/1 s) ───── odometry line ─────────────────────────┘
                                                        │
      prompt builder: [report] + [queried-object marks "(= cN)"] + [image legend] + question
                                                        │
                       frozen Qwen2.5-VL-7B (vLLM, greedy, seed 0, 1 question/request)
                                                        │
                   letter parser → accuracy, balanced accuracy, refusal = guess
                   → frame-level paired bootstrap (2,000×) → gap closed, Δ vs blank
```

Code: images `tools/build_lidar_radar_images.py`, `recovery/lidar_recovery.py`; text `tools/sensor_text/{oracle,cluster,shuffle,ego,format}.py`; prompt `inference/conditions.py`; runs `script/stage1/run_all.sh` with `tools/preflight.py`; scoring `tools/stage1_report.py`.

## 3  Results

**Table 1.** MCQ accuracy (%) on the 400 scored questions, with 95% frame-level bootstrap CIs. In brackets: balanced accuracy (mean over gold answer letters; a constant answer scores 33.3 on perception, 25.0 on behavior). Gap closed: blank = 0, clean = 1.

| Condition | Overall [CI] | Perception (bal.) | Behavior (bal.) | Gap closed [CI] |
|---|---|---|---|---|
| Clean | 44.5 [39.8, 49.0] | 50.0 (54.5) | 39.0 (38.9) | 1 |
| Blank | 34.8 [30.2, 39.5] | 35.5 (47.8) | 34.0 (34.8) | 0 |
| Majority option | 43.2 | 59.0 (33.3) | 27.5 (25.0) | — |
| Stale 0.5 s | 43.0 [38.2, 47.5] | 48.5 (53.2) | 37.5 (37.3) | 0.85 [0.61, 1.07] |
| Stale 1 s | 44.0 [39.2, 48.8] | 50.0 (54.0) | 38.0 (37.7) | 0.95 [0.74, 1.16] |
| Stale 2 s | 44.0 [39.2, 48.8] | 49.5 (54.3) | 38.5 (38.0) | 0.95 [0.72, 1.20] |
| LiDAR image, 1 sweep | 37.8 [33.2, 42.2] | 43.5 (50.4) | 32.0 (33.3) | 0.31 [-0.12, 0.62] |
| LiDAR + radar image | 33.0 [28.5, 37.2] | 29.5 (30.9) | 36.5 (37.4) | -0.18 [-1.06, 0.37] |
| Oracle text | 48.0 [43.2, 52.8] | 60.0 (62.7) | 36.0 (37.6) | 1.36 [0.77, 2.29] |
| Cluster text | 28.5 [24.5, 32.5] | 25.5 (35.8) | 31.5 (33.3) | -0.64 [-1.65, -0.14] |
| Shuffled text | 30.8 [26.5, 35.0] | 23.5 (28.5) | 38.0 (39.4) | -0.41 [-1.32, 0.06] |
| Stale 1 s + cluster | 38.2 [33.8, 42.8] | 33.5 (39.3) | 43.0 (42.0) | 0.36 [-0.19, 0.78] |
| Ego only | 57.2 [52.2, 62.3] | 50.0 (54.5) | 64.5 (65.2) | 2.31 [1.66, 3.81] |
| LiDAR + radar + ego | 59.8 [55.2, 64.8] | 51.0 (40.4) | 68.5 (69.0) | 2.56 [1.83, 4.21] |
| Cluster text + ego | 46.0 [41.2, 50.7] | 28.0 (33.8) | 64.0 (65.2) | 1.15 [0.63, 1.96] |
| Oracle text + ego | 60.2 [55.5, 65.0] | 53.0 (56.2) | 67.5 (68.2) | 2.62 [1.85, 4.44] |

**Figure 1** (`results/stage1/fig1_staleness.pdf`): accuracy vs camera staleness (0–2 s), with horizontal lines for the blank camera, both LiDAR images, oracle text and cluster text. *Caption:* Accuracy as the last good camera frame ages, compared with the sensor-only routes. Error bars: 95% frame-level bootstrap CIs.

**Figure 2** (`results/stage1/fig2_gap_closed.pdf`): gap closed for each route. *Caption:* Fraction of the blank-to-clean gap closed by each input. 0 = blank camera, 1 = clean camera; values above 1 beat the clean camera.

Differences below are paired (same questions), with 95% frame-level bootstrap CIs.

**A stale camera beats every deployable sensor route.** A 2 s old frame keeps clean accuracy (clean − stale 2 s: +0.5 pts [-1.5, +2.8]). A 1 s old frame is ahead of the 1-sweep LiDAR image (+6.2 pts [+2.8, +10.0]) and of cluster text (+15.5 pts [+11.0, +20.2]).

**Words win only with perfect perception.** Oracle text beats blank by +13.2 pts [+7.2, +19.2], and the gain comes from its content: it beats the same reports shuffled across scenes by +17.2 pts [+12.2, +22.0]. On perception it has the best balanced accuracy of any condition (62.7, clean 54.5). Detector-free cluster text falls below blank (28.5); the model often declines to answer (19.2% unparsed) when it cannot match the queried object to a report line. The oracle–cluster gap (+19.5 pts [+14.5, +24.5]) is what a real detector could buy.

**Pixels do not help, and density does not fix them.** Neither LiDAR image is clearly above blank. Adding 9 past sweeps and radar marks does not raise accuracy (1 sweep − 10 sweeps + radar: +4.8 pts [-0.5, +9.8]); its perception balanced accuracy (30.9) is at chance. With matched sensors, deployable pixels and words are close (P2 − cluster text: +4.5 pts [-0.8, +9.5]).

**Ego motion is the largest single effect.** One line of past odometry (~33 tokens) raises overall accuracy over blank by +22.5 pts [+17.0, +28.0] and beats the clean camera (+12.8 pts [+8.0, +17.8]), almost entirely on behavior (64.5 vs blank 34.0). The behavior questions ask what the ego car does next, and past turning agrees with the answer's steering in 166/200 cases, so this measures how much of DriveBench behavior is answerable from ego state rather than from the scene. It adds little to perception (balanced 54.5 vs blank 47.8). With ego motion given to both, LiDAR+radar pixels lead cluster text (+13.8 pts [+8.8, +18.5]), while oracle text adds no significant gain on top of ego alone (+3.0 pts [-2.0, +8.0]).

**Answer bias.** "Going ahead" is the gold answer for 59% of perception questions, so raw perception accuracy rewards answering A. Always answering the majority option scores 43.2% overall, inside the clean camera's CI, so we read perception results from balanced accuracy.

**Reliability.** Running the clean condition twice gives 100% identical MCQ letters and 100% identical outputs (400 questions), so differences between conditions are not decoding noise.

---

## 4  Limitations and Future Work

One model (Qwen2.5-VL-7B), 200 frames and MCQs only. The oracle report uses ground-truth boxes and is an upper bound, not a deployable system. The cluster report is noisy: size-only labels, frequent poles and vegetation, and radar speed for only 39% of clusters. Reports cover 50 m, so 14% of perception targets are absent even from the oracle. Multi-sweep LiDAR smears fast-moving objects over ≈0.45 s. Radar gives only radial velocity, so crossing traffic is under-marked. The ego line uses odometry, which a real car has but DriveBench's camera-only setting does not assume. Next steps: a learned 3D detector for the text route, more VLMs, open-ended questions, and partial (single-camera) loss.

---

## References [verify all before submission]

- Bai, S. et al. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923.
- Nobis, F. et al. 2019. A Deep Learning-based Radar and Camera Sensor Fusion Architecture for Object Detection (CRF-Net). SDF Workshop. **[verify]**
- Yin, T. et al. 2021. Center-based 3D Object Detection and Tracking (CenterPoint). CVPR.
- Liu, Z. et al. 2023. BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation. ICRA.
- Caesar, H. et al. 2020. nuScenes: A Multimodal Dataset for Autonomous Driving. CVPR.
- [authors — verify] 2025. RoboDriveVLM. arXiv:2512.01300.
- Sima, C. et al. 2024. DriveLM: Driving with Graph Visual Question Answering. ECCV.
- Xie, S. et al. 2025. Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives (DriveBench). ICCV.

---

## Notes for the author (delete before submission)

- **Do not use pilot numbers.** They were produced with the broken parser (`claude/PILOT_FINDINGS.md`).
- The **parser audit** is worth one sentence in §2 or a footnote: the old parser mapped any lower-case "a" to option A, which inflated perception accuracy by 25 points and hid refusals.
- Optional analyses that would strengthen §3 (about 1 h, ask Claude):
  - oracle accuracy on only the 161 questions whose target is in the report
  - refusal rate per condition
  - S2 without the 29 truncated frames
- Before submitting:
  - Check every number against `report.md`.
  - Check the figures are readable at column width.
  - Check anonymity rules and the page limit in the AAAI-27 call.
- Word budget: the draft is now over the limit; cut §2 to the conditions table + one sentence per route in the final version.
- Code and data: `script/stage1/run_all.sh`, `tools/build_lidar_radar_images.py`, `tools/sensor_text/ego.py`, `tools/stage1_report.py`, `data/sensor_text/`, `results/stage1/manifests/`.
