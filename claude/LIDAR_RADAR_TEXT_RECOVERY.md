# LiDAR/Radar-as-Text Recovery: Chances of a Better Reading

## Purpose

`claude/TEMPORAL_RECOVERY_RESEARCH.md` scoped a recovery method for
`NoImage`/`FrameLost` around reconstructing the missing *image*. This
document evaluates an alternative for the same two conditions: instead of
reconstructing a frame, convert LiDAR/radar sensor readings into a *text*
scene description and inject that into the prompt, leaving the frozen VLM's
image input blank/absent as it already is. It answers the question asked
directly: what are the chances this produces a better reading than the
`NoImage`/`FrameLost` baseline, and how does it compare to temporal
recovery?

Sourced from this session's chat analysis, `claude/LITERATURE_REVIEW.md`,
`claude/TEMPORAL_RECOVERY_RESEARCH.md`, and new web research (Radar4D-VLM,
TPCNet/Talk2PC, nuScenes radar limitations) — not recalled from training
data. No code has been changed producing this document.

---

## Verdict

| Signal | Chance of a real improvement | Why |
|---|---|---|
| **LiDAR-derived text** | Meaningfully likely to help | Dense, accurate range/geometry; no motion-artifact problem; plays directly into a mechanism already proven in this repo |
| **Radar-derived text (alone)** | Close to a toss-up | nuScenes radar specifically is sparse, elevation-blind, and prone to false positives — useful mainly for velocity/motion cues, risky elsewhere |
| **Combined (LiDAR-led + radar supplement)** | Similar to LiDAR alone, with an edge on `prediction`-task questions | Radar's genuine strength (radial velocity) supplements LiDAR's geometry without radar carrying general scene description alone |

This is a genuinely open hypothesis, like temporal recovery — no paper found
runs this exact version (literal sensor-derived sentences injected into an
off-the-shelf, frozen VLM's prompt). The reasoning below is why it's a
*better-motivated* bet than temporal recovery, not a proven one.

---

## 1. Why this avoids temporal recovery's core risk

`TEMPORAL_RECOVERY_RESEARCH.md` section 2 flagged nuScenes' 2Hz keyframe
spacing (0.5s, large motion at driving speed) as a real technical risk for
video frame interpolation. A sensor-to-text pipeline never synthesizes an
image at all — there's no VFI quality question to fight. The trade is
different, not free: LiDAR point clouds still need converting into
*correct, well-placed* text, which has its own failure modes (below).

## 2. Why this plays into a mechanism already proven in this repo

The reproduced finding in `LITERATURE_REVIEW.md` — clean vs. `NoImage` on
LLaVA-1.5-7B changed 82.8% of predictions, with `NoImage` answers fabricating
specific, plausible-sounding details — is usually read as evidence VLMs lean
on textual cues *instead of* real visual grounding. For this intervention,
that's leverage rather than a problem: if the model already over-weights
text, handing it text that's actually true (derived from real sensor
measurements, not guessed) should move the needle more directly than
requiring the model to correctly *see* an imperfectly reconstructed image.

## 3. New literature (this session) validates the general direction, with a caveat

- **Radar4D-VLM** (arXiv:2608.04130) — a radar-only VLM reasoning over ten
  consecutive 4D-radar sweeps, with a parameter-efficient projector mapping
  radar tokens into a frozen language backbone; predicts object count,
  spatial distribution, motion state, collision risk, category, and radial
  velocity.
- **TPCNet / Talk2PC** — the first outdoor 3D visual-grounding model using
  prompt-guided combination of LiDAR *and* radar point clouds, for more
  accurate natural-language 3D grounding.

Both confirm "sensor data reasoned about through a language model" is an
active, credible 2025/2026 research direction — not a stretch. **The
caveat**: both use *trained projectors* mapping sensor features into
embedding space, not literal natural-language sentences dropped into an
existing prompt template the way this proposal would work in DriveBench.
The plain-text version is simpler to build (no projector training, fits
DriveBench's existing prompt-based pipeline) but a level cruder than what's
published — plausible, not validated at that fidelity.

## 4. Why LiDAR and radar get different verdicts, not one combined number

nuScenes radar specifically (sourced this session): sparse point returns,
no elevation channel, false positives from mirror effects and large
sidelobes, and reliable mainly for cars — other object classes are barely
distinguishable. Converting that straight to text risks injecting
*specific, confident, wrong* claims ("truck at 12m") into the prompt — which
could be more damaging than the vague hallucinations DriveBench already
documented, since a precise wrong claim reads as more credible than a vague
guess. Radar's genuine strength in this dataset is radial velocity, which
matters specifically for `prediction`-task questions — that's the scope
where it should be trusted, not general scene description.

## 5. The implementation catch: coordinate frame

DriveBench questions reference objects by *image-plane* coordinates (e.g.
`<c1,CAM_FRONT,0.6279,0.5731>` in `data/drivebench-test-final.json`). A
LiDAR/radar-derived paragraph describing 3D world positions doesn't
automatically answer a question phrased that way — the model needs the
recovered text to tie back to the same object references the question uses.
That means 3D detections have to be projected into each camera's 2D frame
using calibration data before being turned into text, not just described in
3D. The calibration metadata for this is reachable the same way
`TEMPORAL_RECOVERY_RESEARCH.md` found the temporal-neighbor metadata: via
`calibrated_sensor.json`/`sensor.json` under the already-downloaded
`v1.0-trainval` metadata directory. This is real engineering, and it adds a
component temporal recovery didn't need at all: an actual 3D object
detector (raw point clouds aren't text) — no such detector is installed in
either conda env currently.

## 6. Side-by-side with temporal recovery

| | Temporal recovery | LiDAR/radar-as-text |
|---|---|---|
| New data needed | nuScenes sweep/neighbor camera frames | nuScenes LiDAR + radar sweeps |
| New ML dependency | None for the baseline (nearest-sweep); FILM only if that baseline shows signal | A 3D object detector (none installed currently) |
| Core technical risk | Large-motion interpolation quality (0.5s keyframe gaps) | Radar noise/false positives; coordinate-frame projection correctness |
| Plays to VLM's known behavior | Neutral — still asks the model to *see* correctly | Directly — hands the model text, which it already over-trusts |
| Estimated odds of a real gain | Genuinely open, unproven either way | LiDAR: meaningfully likely; radar alone: closer to even |

Neither is free of engineering cost. Temporal recovery has a *simpler*
zero-ML baseline to test the hypothesis cheaply first (nearest-sweep
substitution). LiDAR/radar-as-text has no equivalently cheap first
experiment — the 3D detector and camera-frame projection are both needed
before the first result is even measurable — but the underlying mechanism
it leverages (text over-trust) is one this repo has already proven exists,
which the frame-recovery hypothesis has to establish from scratch.

---

## Sources

- `claude/LITERATURE_REVIEW.md`, `claude/TEMPORAL_RECOVERY_RESEARCH.md` (this project)
- Radar4D-VLM: "Proposal-Grounded Temporal 4D Radar Reasoning Across Frozen Language Models." arXiv:2608.04130
- TPCNet / Talk2PC: "Enhance 3D Visual Grounding through LiDAR and Radar Point Clouds Fusion for Autonomous Driving." arXiv:2503.08336
- "Resilient Sensor Fusion under Adverse Sensor Failures via Multi-Modal Expert Fusion." arXiv:2503.19776 (carried over — camera-failure compensation precedent)
- nuScenes radar characteristics: sparsity/false-positive/elevation limitations, sourced from radar-detection literature (Continental ARS 408-21 sensor characteristics as used in nuScenes)
- `data/drivebench-test-final.json` (question coordinate-reference format), `calibrated_sensor.json`/`sensor.json` under the local `v1.0-trainval` metadata
