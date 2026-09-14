# Frame Recovery Layer — Full Implementation Guide

## What this project does (one paragraph)

This thesis adds a **frame-recovery preprocessing module** that detects a missing or
corrupted camera frame in an autonomous-driving pipeline and reconstructs it before
the VLM ever sees it. The VLM stays frozen and unchanged. The recovery module sits
between the raw sensor data and the model input. DriveBench is the evaluation harness
used to measure how much downstream QA accuracy improves when a recovered frame
replaces a blank or corrupted one.

---

## Conceptual map: how the pieces fit

```
nuScenes raw data (cameras + LiDAR sweeps)
         │
         ▼
[Recovery Module]  ← YOUR CONTRIBUTION
  Strategy A: nearest_sweep_substitution  (zero-ML baseline)
  Strategy B: rife_interpolate            (temporal interpolation, ECCV 2022)
  Strategy C: lidar_project               (cross-modal, depth-coloured image)
         │
         ▼
Recovered frame  (looks like a clean camera image)
         │
         ▼
Frozen VLM  (Qwen2.5-VL-7B  /  InternVL2.5-8B  /  LLaVA-1.5-7B)
         │
         ▼
DriveBench evaluation harness  (1,461 QA questions, 17 corruption conditions)
         │
         ▼
Scores: Clean > Recovered_RIFE ≥ Recovered_Nearest > NoImage / FrameLost
```

---

## Key distinctions (for thesis clarity)

| Concept | What it is | Role |
|---------|-----------|------|
| **DriveBench** | Benchmark that scores VLMs under 17 input conditions | Your evaluation ruler — do not modify it |
| **nuReasoning** | Dataset for spatial/decision reasoning quality | Different goal — not used here |
| **Qwen2.5-VL-7B** | General-purpose VLM (2025, best open-source AD scores) | Frozen backbone; receives recovered frames |
| **UniDriveVLA** | VLA with 3 expert transformers (understanding + perception + action) | Future integration target — Phase 3 |
| **Recovery module** | Preprocessing layer — RIFE / LiDAR projection | Your novel contribution |
| **RIFE** | Frame interpolation network (ECCV 2022, fully differentiable) | Engine inside the recovery module |

---

## Codebase layout

```
DriveBench/
├── inference/
│   ├── llava1.5.py           LLaVA 1.5-7B  (Ray + vLLM, GPU)
│   ├── qwen2vl.py            Qwen2-VL      (Ray + vLLM, GPU)
│   ├── qwen2vl_mlx.py        Qwen2-VL / Qwen2.5-VL  (MLX, Mac)
│   ├── internvl.py           InternVL2.5-8B (sequential, Mac or GPU)
│   └── utils.py              replace_system_prompt(), load_json_files()
│
├── recovery/
│   ├── temporal_recovery.py  nearest + RIFE strategies; build_recovered_dataset()
│   ├── lidar_recovery.py     LiDAR projection → depth-colourised image
│   └── recovery_module.py    nn.Module wrapping RIFE (for UniDriveVLA fine-tuning)
│
├── tools/
│   ├── fetch_nuscenes_temporal_neighbors.py   builds neighbor_manifest.json
│   ├── fetch_nuscenes_subset.py               copies 200 keyframe images
│   └── compare_conditions.py                  paired per-item score analysis
│
├── script/
│   ├── llava1.5-7b.sh / llava1.5-7b-recovered.sh
│   ├── qwen2.5vl-7b-mlx.sh  (17 conditions)
│   ├── qwen2.5vl-7b-mlx-recovered-nearest.sh
│   ├── qwen2.5vl-7b-mlx-recovered-rife.sh
│   ├── qwen2.5vl-7b-mlx-recovered-lidar.sh
│   ├── qwen2.5vl-7b.sh / qwen2.5vl-7b-recovered.sh  (GPU/vLLM)
│   └── internvl2.5-8b.sh / internvl2.5-8b-recovered.sh
│
├── data/
│   ├── drivebench-test-final.json   1,461 QA entries, 200 frames
│   ├── nuscenes/samples/<CAM>/      200 keyframe images per camera
│   ├── nuscenes/temporal_neighbors/ manifest + sweep images (to be populated)
│   └── corruption/
│       ├── FrameLost/  NoImage/  Fog/  Rain/ … (17 conditions)
│       ├── Recovered_Nearest/    ← populated by recovery/temporal_recovery.py
│       ├── Recovered_RIFE/       ← populated by recovery/temporal_recovery.py
│       └── Recovered_LiDAR/      ← populated by recovery/lidar_recovery.py
│
└── evaluate/
    ├── eval.py       EvaluationSuit — routes QA to GPT scorer + language metrics
    ├── request.py    GPTEvaluation — gpt-3.5-turbo scoring, per-item logs
    ├── prompts.py    6 rubric prompts (perception/prediction/planning/behavior)
    └── utils.py      preprocess_answer()
```

---

## Phase 0 — Data setup (manual steps, do these first)

### 0a. nuScenes neighbor manifest (metadata only — no download needed)

The nuScenes metadata is already at:
`/mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval/`

Run this to build the neighbor manifest:
```bash
python tools/fetch_nuscenes_temporal_neighbors.py \
    --meta-dir /mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval \
    --manifest-only \
    --dest data/nuscenes/temporal_neighbors \
    --steps 2

# Verify:
ls data/nuscenes/temporal_neighbors/neighbor_manifest.json
```

`--steps 2` means ±2 hops at 12 Hz → ±2 frames × 83 ms = ±166 ms around each
missing keyframe. This is the temporal window RIFE will interpolate across.

### 0b. nuScenes sweep images (requires download from nuscenes.org)

Sweep images (12 Hz, unannotated) are NOT on disk yet. You need:
- `sweeps/CAM_FRONT/`
- `sweeps/CAM_FRONT_LEFT/` etc.
- `sweeps/LIDAR_TOP/`  (for Phase 2 LiDAR recovery)

Download the v1.0-trainval camera sweeps blob from nuscenes.org. Then re-run
the command above without `--manifest-only` to copy the neighbor images:

```bash
python tools/fetch_nuscenes_temporal_neighbors.py \
    --meta-dir /mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval \
    --nuscenes-root <PATH_TO_NUSCENES_WITH_SWEEPS> \
    --dest data/nuscenes/temporal_neighbors \
    --steps 2
```

### 0c. RIFE weights (needed for Phase 1b)

```bash
git clone https://github.com/hzwer/ECCV2022-RIFE recovery/rife
# Download HDv3 weights from the RIFE release page into recovery/rife/train_log/
# File to confirm: recovery/rife/train_log/IFNet_HDv3.pkl
```

---

## Phase 1 — Temporal recovery + DriveBench evaluation

**Goal:** Show that substituting a real temporal neighbour (or interpolating one) for a
blank frame lifts Qwen2.5-VL-7B's QA accuracy on the NoImage/FrameLost conditions.

### Step 1a — Nearest-sweep baseline (zero-ML, run first)

```bash
python recovery/temporal_recovery.py \
    --manifest data/nuscenes/temporal_neighbors/neighbor_manifest.json \
    --neighbors-dir data/nuscenes/temporal_neighbors \
    --dest data/corruption/Recovered_Nearest \
    --strategy nearest

# Smoke-test (3 frames):
python recovery/temporal_recovery.py \
    --manifest data/nuscenes/temporal_neighbors/neighbor_manifest.json \
    --neighbors-dir data/nuscenes/temporal_neighbors \
    --dest data/corruption/Recovered_Nearest \
    --strategy nearest --limit 3

# Confirm images are non-blank:
python -c "
from PIL import Image; import numpy as np, glob
imgs = glob.glob('data/corruption/Recovered_Nearest/**/*.jpg', recursive=True)[:3]
for p in imgs: img=Image.open(p); print(p, np.array(img).mean())
"
```

### Step 1b — RIFE interpolation (after sweep images are downloaded)

```bash
python recovery/temporal_recovery.py \
    --manifest data/nuscenes/temporal_neighbors/neighbor_manifest.json \
    --neighbors-dir data/nuscenes/temporal_neighbors \
    --dest data/corruption/Recovered_RIFE \
    --strategy rife \
    --rife-weights recovery/rife/train_log
```

### Step 1c — Run Qwen2.5-VL-7B inference (Mac/MLX)

```bash
# All 17 standard conditions (clean, noimage, framelost, fog, etc.)
bash script/qwen2.5vl-7b-mlx.sh

# Recovered conditions
bash script/qwen2.5vl-7b-mlx-recovered-nearest.sh
bash script/qwen2.5vl-7b-mlx-recovered-rife.sh
```

Output lands in `res/qwen2.5-vl-7b-mlx/<condition>.json`

### Step 1d — Evaluate

```bash
# Run GPT eval for each condition:
python evaluate/eval.py \
    --pred res/qwen2.5-vl-7b-mlx/noimage.json \
    --output res/qwen2.5-vl-7b-mlx/eval_scores/noimage_final_scores.json

python evaluate/eval.py \
    --pred res/qwen2.5-vl-7b-mlx/recovered_nearest.json \
    --output res/qwen2.5-vl-7b-mlx/eval_scores/recovered_nearest_final_scores.json

python evaluate/eval.py \
    --pred res/qwen2.5-vl-7b-mlx/recovered_rife.json \
    --output res/qwen2.5-vl-7b-mlx/eval_scores/recovered_rife_final_scores.json

# Paired comparison across conditions:
python tools/compare_conditions.py \
    --model qwen2.5-vl-7b-mlx \
    --conditions clean noimage framelost recovered_nearest recovered_rife
```

### What to look for in results

| Comparison | What it proves |
|------------|---------------|
| Clean → NoImage drops | Qwen2.5-VL actually uses visual input (strong visual grounding) |
| Nearest > NoImage | Any real temporal frame beats blank — recovery has value |
| RIFE ≥ Nearest | Interpolation beats nearest; confirms interpolation adds signal |
| RIFE < Clean gap | How much recovery "closes the gap" — thesis's headline number |

If RIFE does NOT beat Nearest: motion between 12Hz sweeps is small enough that
nearest substitution is already near-optimal. This is still a valid thesis result —
it means the zero-ML baseline is sufficient, and you document why.

---

## Phase 2 — LiDAR cross-modal fallback

**Goal:** For frames with NO temporal neighbours (start/end of clip), use the nuScenes
LiDAR point cloud projected into camera space as a structural scene prior.

### Step 2a — Populate Recovered_LiDAR

Requires nuScenes sweep data including `sweeps/LIDAR_TOP/` to be on disk.

```bash
python recovery/lidar_recovery.py \
    --meta-dir /mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval \
    --nuscenes-root <PATH_TO_NUSCENES_WITH_SWEEPS> \
    --dest data/corruption/Recovered_LiDAR

# Smoke test:
python recovery/lidar_recovery.py \
    --meta-dir /mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval \
    --nuscenes-root <PATH_TO_NUSCENES_WITH_SWEEPS> \
    --dest data/corruption/Recovered_LiDAR \
    --limit 3
```

Output: depth-colourised image with LiDAR points projected onto black background.
Near objects = red, far = blue. 224×224.

### Step 2b — Run inference and compare

```bash
bash script/qwen2.5vl-7b-mlx-recovered-lidar.sh

python tools/compare_conditions.py \
    --model qwen2.5-vl-7b-mlx \
    --conditions clean noimage framelost recovered_nearest recovered_rife recovered_lidar
```

**Expected:** LiDAR recovery helps most on spatial-reasoning / perception questions
(where object positions matter), less on planning/behavior (which rely more on color,
lighting, road state that LiDAR doesn't capture).

---

## Phase 3 — UniDriveVLA integration and fine-tuning

**Goal:** Show the recovery module works inside a VLA (not just as a frozen VLM
preprocessor) — plug it in as a trainable pre-layer before UniDriveVLA's vision
encoder, then fine-tune.

### Step 3a — Clone UniDriveVLA

```bash
git clone https://github.com/xiaomi-research/unidrivevla
cd unidrivevla
pip install -e .
```

### Step 3b — Load FrameRecoveryModule

```python
from recovery.recovery_module import FrameRecoveryModule

recovery = FrameRecoveryModule.from_pretrained("recovery/rife/train_log")
recovery.freeze()  # frozen during evaluation; unfreeze for fine-tuning
```

### Step 3c — Integration point in UniDriveVLA

Find the image tokenization / vision encoder entry point in UniDriveVLA's forward pass
(likely in `unidrivevla/model/` — trace from the model's `forward()` method to where
camera tensors are passed to the scene-perception expert).

Add before the vision encoder call:
```python
# In UniDriveVLA forward():
if self.recovery_module is not None:
    camera_frames = self.recovery_module.recover_batch(
        camera_frames,    # (B, 3, H, W) — current (possibly blank) frames
        prev_frames,      # (B, 3, H, W) — frames at t-1
        next_frames,      # (B, 3, H, W) — frames at t+1
        missing_mask,     # (B,) bool — True where frame is missing/corrupted
    )
vision_tokens = self.scene_perception_expert(camera_frames)
```

### Step 3d — Fine-tuning schedule

**Stage 1 (5–10 epochs): recovery module only, UniDriveVLA frozen**
```python
recovery.unfreeze()
for p in unidrivevla.parameters():
    p.requires_grad_(False)
# Loss: CE on DriveBench VQA answers under FrameLost/NoImage conditions
```

**Stage 2 (joint): recovery module + UniDriveVLA driving-understanding expert**
```python
for p in unidrivevla.driving_understanding_expert.parameters():
    p.requires_grad_(True)
# Loss: VQA CE + nuScenes planning trajectory loss
```

### Step 3e — Gradient check (run before training)

```python
from recovery.recovery_module import FrameRecoveryModule
import torch

m = FrameRecoveryModule.from_pretrained("recovery/rife/train_log")
prev = torch.rand(1, 3, 224, 224, requires_grad=True)
nxt  = torch.rand(1, 3, 224, 224, requires_grad=True)
out  = m(prev, nxt)
out.mean().backward()
assert prev.grad is not None, "Gradients must flow through RIFE"
print("Gradient check passed.")
```

---

## VLM selection rationale

| Model | Why chosen | Where used |
|-------|-----------|-----------|
| **Qwen2.5-VL-7B** (Jan 2025) | Best open-source AD score (DriveLM: 0.6002); strong visual grounding means recovery delta is large and measurable; direct successor to Qwen2-VL already in codebase | Primary backbone for all DriveBench experiments |
| **InternVL2.5-8B** (Dec 2024) | Different architecture family → makes recovery results architecture-agnostic (stronger thesis claim) | Secondary comparison |
| **LLaVA-1.5-7B** | Already integrated; weaker visual grounding (hallucinates from text priors) → shows WHY recovery matters | Baseline / contrast |
| **UniDriveVLA** (2026) | MoT VLA with 3 experts (understanding + perception + action); assumes complete frames; your recovery module is what it's missing | Phase 3 fine-tuning target |

**Do NOT use:** GPT-4o (API-only, can't run 17 conditions locally, costs per call).

---

## Paper citations for each component

| Component | Paper | Venue |
|-----------|-------|-------|
| DriveBench benchmark | Xie et al., "Are VLMs Ready for Autonomous Driving?" | arXiv:2501.04003 (2025) |
| Corruption impact evidence | Shahriar et al., "Temporal Misalignment Attacks (DejaVu)" | arXiv:2507.09095 (2025) |
| VLM hallucination under missing input | Liao et al., "RoboDriveVLM" | arXiv:2512.01300 (2025) |
| RIFE interpolation | Huang et al., "RIFE: Real-Time Intermediate Flow Estimation" | ECCV 2022 |
| AMT (upgrade from RIFE) | Li et al., "AMT: All-Pairs Multi-Field Transforms" | CVPR 2023 |
| SGM-VFI (large motion) | "Sparse Global Matching VFI" | CVPR 2024 |
| UniDriveVLA | Li et al., "UniDriveVLA" | arXiv:2604.02190 (2026) |
| nuReasoning dataset | Huang et al., "nuReasoning" | arXiv:2605.31572 (2026) |
| AutoTrust | Xing et al., "AutoTrust" | arXiv:2412.15206 (2024) |
| Q-Frame (model-agnostic preprocessing precedent) | Zhang et al., "Q-Frame" | ICCV 2025, arXiv:2506.22139 |
| LiGenCam (LiDAR→RGB reference) | "Reconstruction of Color Camera Images from Multimodal LiDAR" | 2025 |

---

## Troubleshooting

**"RIFE repo not found at recovery/rife"**
→ Clone ECCV2022-RIFE there: `git clone https://github.com/hzwer/ECCV2022-RIFE recovery/rife`

**"neighbor image(s) not found locally" in skipped list**
→ Sweep images not downloaded. Run with `--strategy nearest` first (uses keyframes already on disk).

**`replace_system_prompt` raises ValueError "Unable to extract camera name"**
→ Image path doesn't contain `samples/CAM_*/` pattern. Check that you're passing the original (pre-swap) paths, not the corrupted paths.

**InternVL2.5 flash_attn error on Mac**
→ The `inference/internvl.py` already disables flash_attn on MPS. Make sure `device == "mps"` is detected correctly.

**MLX out of memory on Qwen2.5-VL-72B**
→ Use the 7B variant. 72B requires ~144GB RAM; 7B needs ~14GB unified memory (M3 Max has 128GB max but 14GB+ base).

---

## Expected result summary (thesis narrative)

1. **Baseline finding (reproduces DriveBench paper):** Qwen2.5-VL under NoImage/FrameLost
   scores significantly lower than Clean — model can't hallucinate its way past missing
   visual input the way LLaVA 1.5 does (strong visual grounding = more sensitive to
   frame quality).

2. **Recovery finding (novel contribution):** Recovered_Nearest > NoImage/FrameLost
   confirms that any real frame beats blank. Recovered_RIFE ≥ Recovered_Nearest confirms
   interpolation adds signal. Gap to Clean = "remaining recovery challenge."

3. **Architecture-agnostic claim:** Same pattern holds for InternVL2.5-8B → recovery
   is model-agnostic, not tuned to Qwen2.5-VL specifically.

4. **UniDriveVLA claim:** Fine-tuned recovery module inside UniDriveVLA further reduces
   the gap to Clean by adapting to the driving-domain distribution.
