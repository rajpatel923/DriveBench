# Codex Agent Brief — DriveBench Frame Recovery Research

## Who you are and what this project is

You are an AI coding agent working on a graduate thesis in autonomous driving. The thesis
contributes a **frame-recovery preprocessing module** that reconstructs missing or
corrupted camera frames before they reach a frozen Vision-Language Model (VLM). The
recovery module sits between the raw sensor data and the model — the VLM is never
retrained.

This repo (`DriveBench`) is a fork/extension of the DriveBench benchmark
(Xie et al., arXiv:2501.04003), which evaluates VLMs under 17 input-corruption
conditions on nuScenes autonomous-driving data.

---

## Where to find everything

### Research context (read these first)
| File | What it contains |
|------|-----------------|
| `claude/IMPLEMENTATION_GUIDE.md` | **Master reference** — architecture diagram, phase-by-phase instructions, full file map, paper citations, troubleshooting |
| `claude/LITERATURE_REVIEW.md` | Nine-paper literature survey. Establishes why frame loss is catastrophic, why VLMs fail silently, and why a preprocessing recovery layer is the right fix |
| `claude/TEMPORAL_RECOVERY_RESEARCH.md` | Deep-dive on the temporal interpolation approach — why 12 Hz sweeps work, what FILM vs. RIFE vs. nearest-neighbour means, measurement strategy |
| `claude/LIDAR_RADAR_TEXT_RECOVERY.md` | Analysis of using LiDAR/radar as cross-modal recovery signal |
| `claude/PLAN.md` | Original feasibility assessment — confirms DriveBench is suitable as the evaluation harness |

### Codebase (what already exists)
| File | Status | What it does |
|------|--------|-------------|
| `inference/llava1.5.py` | Complete | LLaVA-1.5-7B inference, Ray + vLLM, 17 conditions |
| `inference/qwen2vl.py` | Complete | Qwen2-VL inference, Ray + vLLM (GPU) |
| `inference/qwen2vl_mlx.py` | Complete | Qwen2-VL / **Qwen2.5-VL** inference, MLX (Mac) |
| `inference/internvl.py` | Complete | InternVL2.5-8B inference, sequential Mac or GPU |
| `inference/utils.py` | Complete | `replace_system_prompt()`, `load_json_files()`, `save_json()` |
| `recovery/temporal_recovery.py` | Complete | `nearest_sweep_substitution()` ✅; `rife_interpolate()` ✅ (needs RIFE repo + weights) |
| `recovery/lidar_recovery.py` | Complete | LiDAR→camera projection, depth-colourised output |
| `recovery/recovery_module.py` | Complete | `FrameRecoveryModule(nn.Module)` wrapping RIFE — differentiable, for UniDriveVLA fine-tuning |
| `tools/fetch_nuscenes_temporal_neighbors.py` | Complete | Builds `neighbor_manifest.json` from nuScenes metadata |
| `tools/compare_conditions.py` | Complete | Paired per-item score analysis across conditions |
| `evaluate/eval.py` | Complete | GPT + language metric scoring |
| `data/drivebench-test-final.json` | On disk | 1,461 QA entries, 200 frames, 174 scenes |
| `data/nuscenes/samples/` | On disk | 200 keyframe images × 6 cameras |
| `data/corruption/` | On disk | 14 synthetic corruptions + FrameLost; Recovered_* dirs not yet populated |

### Scripts
```
script/
├── llava1.5-7b.sh / llava1.5-7b-recovered.sh
├── qwen2.5vl-7b-mlx.sh                       ← 17-condition sweep, Mac
├── qwen2.5vl-7b-mlx-recovered-nearest.sh     ← Recovered_Nearest condition
├── qwen2.5vl-7b-mlx-recovered-rife.sh        ← Recovered_RIFE condition
├── qwen2.5vl-7b-mlx-recovered-lidar.sh       ← Recovered_LiDAR condition
├── qwen2.5vl-7b.sh / qwen2.5vl-7b-recovered.sh  ← GPU/vLLM variants
└── internvl2.5-8b.sh / internvl2.5-8b-recovered.sh
```

---

## The research problem (plain language)

VLMs used in autonomous driving **fail silently and confidently** when a camera frame is
missing. Instead of saying "I cannot see this frame," they generate plausible-sounding
but fabricated answers based on text cues in the question (e.g., a camera name like
"CAM_FRONT" in the question makes them describe what a front camera typically sees, even
with a blank image).

**Evidence (from this repo's own experiments):**
- Running LLaVA-1.5-7B on DriveBench: 82.8% of predictions change between Clean and
  NoImage conditions. NoImage answers fabricate specific details like "dark environment"
  or "black and white environment."
- DejaVu (2025): a single-frame LiDAR delay cuts car-detection mAP by 88.5%. A
  3-frame camera delay drops multi-object tracking accuracy (MOTA) by 73%.

**The gap in existing work:**
- DriveBench, USB, AutoTrust, RoboDriveVLM all **measure** this problem.
- UniDriveVLA, nuReasoning (both 2026) still **assume complete synchronized input**.
- RoboDriveVLM proposes cross-modal fusion but only **inside a retrained model**.
- No one reconstructs the **input frame itself** before a frozen, off-the-shelf VLM.

**This thesis fills that gap.**

---

## The solution architecture

```
Missing/Corrupted camera frame
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  FrameRecoveryModule  (recovery/recovery_module.py)    │
│                                                        │
│  Strategy A — nearest_sweep_substitution               │
│    • Zero-ML baseline                                  │
│    • Substitute closest 12 Hz sweep frame in time      │
│    • No model needed                                   │
│                                                        │
│  Strategy B — rife_interpolate                         │
│    • RIFE IFNet (ECCV 2022, fully differentiable)      │
│    • Interpolate from bracketing prev/next sweep frames│
│    • t = (missing_ts - prev_ts) / (next_ts - prev_ts) │
│                                                        │
│  Strategy C — lidar_project (fallback)                 │
│    • When no temporal neighbours exist                 │
│    • Project LIDAR_TOP into camera frame               │
│    • Depth-colourised structural prior                 │
└────────────────────────────────────────────────────────┘
         │
         ▼
Recovered frame  (224×224 PIL/tensor, looks like clean input)
         │
         ▼
Frozen VLM  (Qwen2.5-VL-7B or InternVL2.5-8B)
         │
         ▼
DriveBench scores (17 conditions, 1,461 QA questions)
```

---

## The three research phases

### Phase 1 — DriveBench evaluation with Qwen2.5-VL-7B

**Goal:** Quantitatively show that recovering a missing frame improves VQA accuracy
versus using a blank frame, across a state-of-the-art 2025 VLM.

**Tasks for agent:**
1. Verify `recovery/temporal_recovery.py` runs correctly on the nearest strategy
2. Implement any fixes needed to `rife_interpolate()` once RIFE repo is cloned to
   `recovery/rife/` — the function is written but needs the RIFE IFNet to be importable
3. Confirm output images in `data/corruption/Recovered_Nearest/` are non-blank (mean > 0)
4. Confirm inference pipeline reads recovered frames correctly via the path-swap:
   `img_path.replace('nuscenes/samples', 'corruption/Recovered_Nearest')`
5. Run `tools/compare_conditions.py` and verify the delta table is correctly formatted

**Expected result:** `Clean > Recovered_RIFE ≥ Recovered_Nearest > NoImage / FrameLost`

**Key files to read:** `recovery/temporal_recovery.py`, `inference/qwen2vl_mlx.py`,
`tools/compare_conditions.py`

---

### Phase 2 — LiDAR cross-modal fallback

**Goal:** For frames with no temporal neighbours, use nuScenes LiDAR as a structural
prior. Show that even a depth-projection image beats a blank frame for spatial-reasoning
questions.

**Tasks for agent:**
1. Verify `recovery/lidar_recovery.py` can load nuScenes calibration JSON files from
   the metadata path without the nuscenes-devkit (pure numpy/scipy)
2. Check the point projection math:
   - Transform: LiDAR frame → ego frame → camera frame
   - Project: K @ T_cam_lidar @ [x, y, z, 1]^T → [u, v]
   - Filter: only keep points where z > 0.1 (in front of camera) and u,v within image
3. Confirm output image has visible coloured points (not all black)
4. Add a hybrid strategy: use RIFE when neighbours exist, fall back to LiDAR when they
   don't — wire this into `build_recovered_dataset()` as strategy `"hybrid"`

**Key files:** `recovery/lidar_recovery.py`, nuScenes metadata at
`/mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval/calibrated_sensor.json`

---

### Phase 3 — UniDriveVLA integration and fine-tuning

**Goal:** Plug `FrameRecoveryModule` as a trainable pre-layer into UniDriveVLA's
forward pass, then fine-tune. Show that the recovery module, when trained end-to-end
with the driving VLA, further reduces the Clean-vs-Recovered gap.

**Tasks for agent:**
1. Clone UniDriveVLA (`github.com/xiaomi-research/unidrivevla`) and trace the forward
   pass to find where camera tensor(s) enter the scene-perception expert
2. Add `FrameRecoveryModule` as an `nn.Module` attribute of the UniDriveVLA model class
3. In the forward pass, call `recovery_module.recover_batch(frames, prev, next, mask)`
   before passing frames to the vision encoder
4. Verify the gradient check in `recovery/recovery_module.py` passes — gradients must
   flow from the VLA loss through `FrameRecoveryModule` back to its parameters
5. Implement the 2-stage fine-tuning loop:
   - Stage 1: freeze UniDriveVLA, train only `FrameRecoveryModule`
   - Stage 2: unfreeze UniDriveVLA's driving-understanding expert, joint training

**Key files:** `recovery/recovery_module.py`, UniDriveVLA forward pass (find after
cloning)

---

## Key concepts an agent must understand

### How DriveBench's corruption-swap works
Every inference script does one thing to load images:
```python
img_path = img_path.replace('nuscenes/samples', f'corruption/{corruption}')
img = Image.open(img_path).convert('RGB')
```
To add a new condition (`Recovered_RIFE`, `Recovered_LiDAR`, etc.), you only need to
pre-populate `data/corruption/<NAME>/<CAM>/<original_filename>.jpg`. No inference code
changes are needed. This is the key design principle.

### Why Qwen2.5-VL-7B and not LLaVA-1.5-7B
Qwen2.5-VL has much stronger visual grounding. That means it actually uses the image
to answer questions (unlike LLaVA which leans on text priors). Stronger visual grounding
= larger drop from Clean → NoImage = larger improvement when recovery works = more
statistically meaningful results.

### Why RIFE and not FILM or AMT
- FILM (arXiv:2202.04901) is TensorFlow-based — hard to mix into a PyTorch pipeline
- AMT (CVPR 2023) is the literature-review's theoretical choice but more setup overhead
- RIFE (ECCV 2022) is pure PyTorch, ~100MB, 10-50ms/frame, fully differentiable,
  and the most widely deployed VFI model in practice. Use RIFE first, cite AMT as the
  natural upgrade path.

### Why the recovery module must be differentiable
Phase 3 requires fine-tuning. Gradients must flow from UniDriveVLA's loss (VQA cross-
entropy + planning trajectory loss) back through the recovered frame tensor, into
RIFE's IFNet parameters. RIFE is a CNN — it is fully differentiable. This is what
makes `FrameRecoveryModule` a training-time contribution, not just an inference-time
preprocessing step.

### DriveBench vs nuReasoning (common confusion)
- **DriveBench**: benchmarks VLMs under corrupted/missing inputs. Measures visual
  grounding quality. This is your evaluation harness.
- **nuReasoning**: a reasoning-quality dataset (20k clips, human-verified spatial/
  decision/counterfactual labels). Does not test degraded inputs. Different goal.
- They are complementary but serve completely different purposes for this thesis.

### UniDriveVLA vs Qwen2.5-VL-7B
- **Qwen2.5-VL**: VLM — outputs text only. You test it frozen on DriveBench.
- **UniDriveVLA**: VLA — outputs text + driving actions (trajectories). It has 3
  expert transformers (driving understanding, scene perception, action planning).
  You integrate the recovery module into it and fine-tune for Phase 3.

---

## Data the agent can assume is available

| Data | Location | Status |
|------|----------|--------|
| DriveBench QA | `data/drivebench-test-final.json` | On disk |
| nuScenes keyframe images | `data/nuscenes/samples/<CAM>/` | On disk (200 × 6) |
| nuScenes metadata | `/mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval/` | On disk |
| All 14 synthetic corruptions | `data/corruption/<TYPE>/<CAM>/` | On disk |
| FrameLost (blank frames) | `data/corruption/FrameLost/<CAM>/` | On disk |
| nuScenes sweep images (12 Hz) | Must be downloaded from nuscenes.org | **NOT on disk yet** |
| RIFE weights | Must be cloned + downloaded from github.com/hzwer/ECCV2022-RIFE | **NOT on disk yet** |

---

## Invariants the agent must not violate

1. **Never modify inference code to handle a new condition.** Add the condition by
   pre-populating `data/corruption/<NAME>/`. The path-swap in every inference file
   handles it automatically.

2. **Never retrain the VLM backbone.** Qwen2.5-VL-7B, InternVL2.5-8B, and
   LLaVA-1.5-7B must remain frozen throughout Phases 1 and 2. Only the
   `FrameRecoveryModule` parameters are trained in Phase 3.

3. **Never change the evaluate/ directory.** The evaluation logic, GPT rubric prompts,
   and per-item logging format are fixed. New conditions are evaluated the same way.

4. **Output filenames in Recovered_* directories must match the original keyframe
   filenames exactly.** The path-swap only changes the directory prefix. If the
   original file is `n008-2018-...-CAM_FRONT-....jpg`, the recovered version must
   be saved with that exact name.

5. **RIFE must remain differentiable.** Do not use `.detach()` on intermediate tensors
   in `FrameRecoveryModule.forward()`. Phase 3 fine-tuning depends on gradient flow.

---

## How to verify work is correct (test commands)

### Verify recovered images are non-blank
```python
from PIL import Image
import numpy as np, glob

for path in glob.glob('data/corruption/Recovered_RIFE/**/*.jpg', recursive=True)[:5]:
    arr = np.array(Image.open(path))
    print(f"{path}: mean={arr.mean():.1f}, std={arr.std():.1f}")
# Expected: mean > 0, std > 0 (non-blank)
```

### Verify inference reads recovered frames correctly
```bash
python inference/qwen2vl_mlx.py \
    --model mlx-community/Qwen2.5-VL-7B-Instruct-bf16 \
    --data data/drivebench-test-final.json \
    --output /tmp/smoke_test.json \
    --system_prompt prompt.txt \
    --corruption Recovered_RIFE --limit 3
python -c "import json; data=json.load(open('/tmp/smoke_test.json')); print(data[0]['pred'])"
# Expected: a non-empty string, not None
```

### Verify RIFE gradients flow
```python
from recovery.recovery_module import FrameRecoveryModule
import torch

m = FrameRecoveryModule.from_pretrained("recovery/rife/train_log")
prev = torch.rand(1, 3, 224, 224, requires_grad=True)
nxt  = torch.rand(1, 3, 224, 224, requires_grad=True)
out  = m(prev, nxt)
out.mean().backward()
assert prev.grad is not None, "FAIL: gradients not flowing through RIFE"
print(f"PASS: prev.grad.norm() = {prev.grad.norm():.6f}")
```

### Verify LiDAR projection produces visible output
```python
from recovery.lidar_recovery import lidar_project, _load_calibrations, _load_ego_poses, _load_sample_data_index
import numpy as np

meta = "/mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval"
cal = _load_calibrations(meta)
ego = _load_ego_poses(meta)
idx = _load_sample_data_index(meta)
# Use any frame_token from data/drivebench-test-final.json
import json; ft = json.load(open("data/drivebench-test-final.json"))[0]["frame_token"]
img = lidar_project(ft, "CAM_FRONT", meta, "<NUSCENES_ROOT>", cal, ego, idx)
if img:
    arr = np.array(img)
    print(f"Points visible: {(arr > 0).any(axis=2).sum()} pixels non-black")
```

---

## Research narrative (for thesis writing)

The thesis argument is:

1. **The problem is real and measured** (DriveBench Clean→NoImage drop, DejaVu attack
   numbers)
2. **Existing solutions are incomplete** (RoboDriveVLM fuses inside a retrained model;
   no one recovers the input frame itself for frozen VLMs)
3. **Recovery is feasible** (RIFE is CVPR/ECCV-validated; nuScenes already has the
   temporal + LiDAR data needed)
4. **Recovery works** (Recovered_RIFE > NoImage/FrameLost on DriveBench with
   Qwen2.5-VL-7B and InternVL2.5-8B → architecture-agnostic)
5. **It generalises to VLAs** (FrameRecoveryModule fine-tuned inside UniDriveVLA
   further closes the gap)
