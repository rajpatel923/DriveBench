# Literature Review: VLM Architectures and the Case for Frame-Recovery in Autonomous Driving

## Purpose

This document analyzes the VLM/VLA architectures used across the nine papers underpinning
the thesis, then builds the argument for why frame loss is a real, unsolved problem in
this literature, and why a frame-recovery method (reconstructing a missing/corrupted
camera frame — via temporal interpolation and/or cross-modal lidar/radar fusion — before
it reaches the VLM) is a credible, well-motivated contribution.

Every factual claim below is sourced from the actual paper (fetched directly, not
recalled from training data — several of these are 2025/2026 papers). Cite the original
paper, not this document, in the thesis itself; use this as a working synthesis to build
from.

---

## 1. VLM/VLA architecture landscape across the nine papers

### (a) General-purpose driving-VLM benchmarks

**DriveBench** (Xie et al., 2025, arXiv:2501.04003) — the benchmark this repo implements.
Evaluates 12 VLMs spanning generalist and specialist, open and closed models:
GPT-4o, LLaVA-1.5 (7B/13B), LLaVA-NeXT (7B), InternVL2 (8B), Phi-3 (4.2B), Phi-3.5 (4.2B),
Oryx (7B), Qwen2-VL (7B/72B), DriveLM-Agent (a LLaMA-Adapter-V2-based specialist
fine-tuned on DriveLM-nuScenes), and Dolphins (7B specialist). Its central finding:
VLMs "generate plausible responses derived from general knowledge or textual cues rather
than true visual grounding, especially under degraded or missing visual inputs" — models
exploit text-based spatial cues (e.g. a camera name mentioned in the question) instead of
actually looking at the image.

**USB — Unstructured Scene Benchmark** (Zheng et al., 2026, *Communications in
Transportation Research*) — architecture-agnostic; benchmarks arbitrary VLMs against 20
realistic input perturbations (illumination, weather, **sensor reliability**, occlusion)
across 6 synchronized camera views augmented with vehicle-state data. Notable for
explicitly naming "sensor reliability" as a first-class perturbation axis in a 2026
benchmark, without proposing any compensation mechanism.

**AutoTrust** (Xing et al., 2024, arXiv:2412.15206) — a 5-axis trustworthiness benchmark
(trustfulness, safety, **robustness**, privacy, fairness) across 6 VLMs including
LLaVA-v1.6, GPT-4o-mini, and DriveLM-Agent. Establishes robustness as a formally
recognized trust dimension for DriveVLMs, but the published abstract does not treat
missing/corrupted visual input as a distinct failure mode from other robustness threats
(e.g. adversarial prompts, privacy leakage).

### (b) Unified VLA architectures

**UniDriveVLA** (Li et al., 2026, arXiv:2604.02190) — a unified Vision-Language-Action
model using a **Mixture-of-Transformers** design: three specialized experts (driving
understanding, scene perception, action planning) coordinated through masked joint
attention, resolving the tension where "directly adopting 2D VLMs yields limited spatial
perception, whereas enhancing them with 3D spatial representations often impairs the
native reasoning capacity of VLMs." Trained via a sparse perception paradigm with a
three-stage progressive strategy. No mention of multi-frame temporal handling or
missing/corrupted sensor frames — a current-generation unified VLA architecture that
still assumes complete, synchronized input at every timestep.

**nuReasoning** (Huang et al., 2026, arXiv:2605.31572) — a nuScenes/nuPlan-lineage
reasoning-centric dataset: 20,000 20-second clips with synchronized multi-camera images,
LiDAR, HD maps, object annotations, and human-verified reasoning labels (spatial,
decision, counterfactual). Evaluates both VLMs and VLAs, showing fine-tuning on it
improves driving QA and planning. No discussion found of temporal sequencing robustness
or missing/corrupted frames — notable because this dataset already contains exactly the
multi-modal ingredients (synchronized camera + LiDAR) a frame-recovery method would
need, simply not used for that purpose.

### (c) Robustness-specific work

**RoboDriveVLM** (Liao et al., 2025, arXiv:2512.01300) — the closest existing work to
this thesis. A benchmark of 11 corruption scenarios (6 environmental sensor-corruption
cases, 5 prompt-corruption cases; 64,559 trajectory-prediction cases total) paired with a
proposed fix: mapping "more multimodal data — e.g., lidar and radar — into a unified
latent space," plus a Test-Time Adaptation method based on cross-modal knowledge
distillation. This is real evidence that lidar/radar can compensate for degraded camera
input — but the compensation happens at the **feature/fusion level inside a retrained
model**, not by reconstructing the missing **input frame itself** for an unmodified,
frozen, off-the-shelf VLM.

**DejaVu / temporal misalignment attack** (Shahriar et al., 2025, arXiv:2507.09095) — not
a benchmark but an attack: manipulates in-vehicle network timing to desynchronize
camera and LiDAR streams feeding a sensor-fusion perception stack. Demonstrated on an
automotive Ethernet testbed with the Autoware stack. Quantified impact: a **single-frame
LiDAR delay reduces car-detection mAP by up to 88.5%**; a **three-frame camera delay
drops multi-object-tracking accuracy (MOTA) by 73%**. No defense is proposed in the
paper. This is the strongest available evidence that frame-level timing/availability
issues are catastrophic, safety-critical, and — crucially — adversarially exploitable,
not merely a benign edge case from weather or sensor faults.

### (d) Enabling techniques borrowed from outside autonomous driving

**Q-Frame** (Zhang et al., 2025, ICCV, arXiv:2506.22139) — a general video-LLM technique,
not AD-specific: training-free, CLIP-based query-aware frame selection combined with
multi-resolution adaptation, using the Gumbel-Max trick for efficient selection. Explicitly
model-agnostic — shown to improve open-source models (VILA-1.5, Qwen2-VL) and API models
(GPT-4o) alike. Its core assumption is that all candidate frames **exist** and the task is
choosing/reweighting among them — it has no mechanism for the case where the specific view
needed is entirely absent. Useful mainly as a design analogy: its query-aware gating shows
the field already accepts lightweight, frame-level, model-agnostic preprocessing as a
legitimate intervention point.

**AMT** (Li et al., 2023, CVPR) — a general-purpose video frame-interpolation network, not
AD-specific. Architecture: builds bidirectional all-pairs correlation volumes, uses
predicted bilateral flow to retrieve correlations and jointly update flow fields and
interpolated content features, then derives multiple fine-grained flow fields for
backward-warping the input frames. Convolutional (not transformer-based), achieving
SOTA accuracy/efficiency trade-offs. This supplies a concrete, proven mechanism for the
*temporal* arm of frame recovery: synthesizing a plausible frame *t* from real frames
*t-1* and *t+1* in the same camera stream.

---

## 2. Argument: frame loss is a real, unsolved problem

**It is empirically catastrophic, not cosmetic.** DejaVu's numbers are the strongest
evidence available: a single delayed LiDAR frame cuts detection mAP by up to 88.5%; a
three-frame camera delay cuts tracking MOTA by 73%. These are not marginal degradations —
they represent near-total loss of the corresponding perception capability from a
disruption measured in single-digit frames.

**VLMs fail silently and confidently, which is worse than failing loudly.** DriveBench's
central finding — that VLMs answer driving questions from language priors and textual
cues rather than actual visual grounding when input is missing or degraded — was directly
reproduced in this project's own pipeline: running the same 1,461 DriveBench questions
under `clean` vs. `NoImage` conditions on LLaVA-1.5-7B showed 82.8% of predictions changed
between the two, with the NoImage answers fabricating specific, plausible-sounding scene
details ("dark environment," "black and white environment") that don't correspond to
anything the model actually saw. In a safety-critical planning loop, a confidently wrong
answer is more dangerous than a system that flags "I cannot see this frame."

**The problem is being measured faster than it's being solved.** USB, AutoTrust, and
RoboDriveVLM all now treat sensing degradation or robustness as an explicit,
first-class benchmark axis — this is a genuinely active, multi-paper concern across
2024-2026, not a niche worry. Yet the newest unified VLA architectures surveyed here
(UniDriveVLA, nuReasoning), both published in 2026, build no explicit compensation for
a missing or desynchronized frame into their design — they assume complete synchronized
input as a precondition, exactly the assumption DejaVu shows is unsafe to make.

---

## 3. Argument: the proposed recovery method is a credible solution

**Precedent exists for cross-modal compensation, but at the wrong layer.**
RoboDriveVLM already demonstrates that lidar/radar data can compensate for degraded
camera input — but only by fusing modalities into a shared latent space *inside a
retrained model*. The method proposed here differs in an important, practically
valuable way: reconstruct the **frame itself** before it ever reaches the VLM, which
means the VLM can remain frozen and off-the-shelf. This mirrors a pattern already
proven to work in this exact codebase — DriveBench's own corruption mechanism swaps the
image file path (`img_path.replace('nuscenes/samples', f'corruption/{self.corruption}')`
in `inference/llava1.5.py`) *before* the frozen LLaVA model ever sees it. A recovery
module fits into that same slot, meaning it could be benchmarked against any of
DriveBench's 12 VLMs, or AutoTrust's LLaVA-v1.6/GPT-4o-mini/DriveLM-Agent, without
retraining a single model.

**A proven mechanism exists for the temporal-recovery arm.** AMT's bidirectional
all-pairs correlation + bilateral-flow architecture is state-of-the-art, efficient, and
purpose-built for exactly the sub-problem this thesis needs solved: synthesizing a
missing frame from real temporal neighbors in the same camera stream. This is not a
speculative technique — it is a CVPR-validated, open-source, convolutional network with
public code (github.com/MCG-NKU/AMT).

**The field already accepts frame-level, model-agnostic preprocessing as legitimate.**
Q-Frame demonstrates that a lightweight, query-aware, model-agnostic intervention at the
frame level is a viable and increasingly standard design pattern for VLMs — validated
across both open-source (VILA, Qwen2-VL) and closed (GPT-4o) backbones. While Q-Frame
solves a different problem (selecting among existing frames, not recovering an absent
one), its design proves the field is receptive to exactly this class of intervention:
lightweight, plug-in, and independent of the underlying VLM's training.

**A second, non-obvious contribution: potential defense against an active attack
class.** DejaVu's attack works specifically by desynchronizing camera and LiDAR timing.
A working recovery/consistency-check module — one that can detect when an expected frame
is missing or inconsistent with its neighbors and reconstruct a plausible substitute —
is structurally positioned to also serve as a *mitigation* for that attack class, not
just for benign weather- or hardware-induced frame loss. This broadens the thesis's
claimed significance from "improves accuracy under corruption" to "provides a
generalizable defense against a demonstrated, safety-critical adversarial vulnerability."

---

## 4. The gap this thesis fills

None of the nine papers combine all three of the following:

1. **Evaluation against frozen, off-the-shelf, publicly available VLMs** (as DriveBench,
   AutoTrust, and USB do), rather than requiring model retraining (as RoboDriveVLM's
   fix requires).
2. **An explicit, isolated missing-frame condition** — distinct from generic corruption —
   as DriveBench's own `NoImage`/`FrameLost` categories already isolate (verified
   directly in this repo: both are literal blank frames, mean=0/std=0, distinct from the
   14 other corruption types which are degraded-but-present images).
3. **An active reconstruction step** (temporal interpolation via an AMT-style mechanism,
   and/or cross-modal reconstruction informed by lidar/radar) applied *before* inference,
   rather than passive robustness training or feature-level fusion.

This is the same gap already verified empirically in `claude/PLAN.md`: DriveBench
isolates missing-frame conditions and scores at per-frame/per-question granularity
(enabling clean-vs-missing-vs-recovered paired comparisons), but has no recovery
mechanism of its own; the raw nuScenes lidar/radar and temporal (`prev`/`next`) metadata
needed to build one already exists and is technically reachable (metadata already
downloaded to this machine), just unused by any current pipeline in this space.

---

## Sources

- Xie, S. et al. "Are VLMs Ready for Autonomous Driving?" arXiv:2501.04003 (2025)
- Zheng, Y. et al. "Unstructured Scene Benchmark (USB)." *Comm. in Transportation Research*, DOI: 10.26599/COMMTR.2026.9640034 (2026)
- Li, Y. et al. "UniDriveVLA." arXiv:2604.02190 (2026)
- Liao, D. et al. "RoboDriveVLM." arXiv:2512.01300 (2025)
- Xing, S. et al. "AutoTrust." arXiv:2412.15206 (2024)
- Huang, Z. et al. "nuReasoning." arXiv:2605.31572 (2026)
- Zhang, S. et al. "Q-Frame." ICCV 2025, arXiv:2506.22139
- Shahriar, M.H. et al. "Temporal Misalignment Attacks (DejaVu)." arXiv:2507.09095 (2025)
- Li, Z. et al. "AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation." CVPR 2023
