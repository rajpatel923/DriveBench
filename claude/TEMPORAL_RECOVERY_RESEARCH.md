# Temporal-Interpolation Frame Recovery: Feasibility & Expected-Impact Research

## Purpose

`claude/PLAN.md` established that a frame-recovery thesis is architecturally
compatible with this repo, and `claude/LITERATURE_REVIEW.md` built the
literature case for why frame loss is a real, unsolved problem. Both stopped
short of picking one concrete recovery technique and reasoning about whether
it would actually move the benchmark's numbers. This document does that for
**temporal interpolation** specifically (reconstructing a frame from its
real neighbors in time), and answers the two questions asked directly:

1. **Is it possible to implement in this repo, given what's on disk right now?**
2. **Is it plausible that it would produce measurably better readings (scores)?**

Everything below is sourced from reading this repo's actual code/data this
session (file/line citations included) plus new web research — not recalled
from training data. No code or data has been changed as part of producing
this document; the accompanying code scaffolding (`tools/`, `recovery/`,
the `evaluate/eval.py` patch) is described in a companion note but kept
separate from this analysis.

---

## Direct answers

**Q1 — Is it possible here?** Yes, but with one gap `claude/PLAN.md` didn't
fully account for: the neighboring frame *image bytes* needed for
interpolation are not on disk yet, only the *metadata* to find them. See
"Data gap, quantified" below. Everything else — the injection point, the
per-item evaluation logs, the QA density — is exactly as favorable as
`claude/PLAN.md` found.

**Q2 — Will it likely produce better readings?** Conditionally yes, and the
condition matters: the case is strong for the two *true missing-frame*
conditions (`NoImage`, `FrameLost`), where there is no visual signal at all
to start from — literally any real, temporally-plausible frame should beat a
blank image or a confidently-hallucinated answer. The case is **not**
established (and arguably the wrong problem) for the other 14 corruption
types, where DriveBench already keeps the original clean frame on disk — see
"Scope correction" below. No published work was found (this session's web
research) that has actually run this exact experiment — reconstructed frame
→ frozen VLM → measured VQA accuracy — so this is a genuinely open question,
not a safe bet, which is good for thesis novelty but means the report can't
promise a result, only a well-motivated hypothesis and a way to measure it
honestly.

---

## 1. Scope correction: interpolation is the right tool for 2 of 17 conditions, not 17

DriveBench's 17 input settings split into two very different problems:

- **`NoImage` and `FrameLost`** (verified via PIL earlier: both are literal
  blank frames, mean=0/std=0) — there is *no content in that frame* to work
  with. This is exactly the case temporal interpolation is for: synthesize
  the missing content from real neighboring frames in time.
- **The other 14** (Fog, Rain, MotionBlur, Snow, Brightness, ColorQuant,
  LensObstacleCorruption, Saturate, ZoomBlur, WaterSplashCorruption,
  H256ABRCompression, LowLight, BitError, CameraCrash) are synthetic
  degradations applied to a frame that DriveBench still has the clean
  original of, sitting right there in `data/nuscenes/samples/`. Recovering
  those is an **image-restoration** problem (denoise/defog/deblur/dering),
  not a frame-interpolation problem — there's no temporal gap to fill, the
  ground-truth pixels already exist, just under a different directory.

Conflating the two would weaken a thesis's methodology section — a reviewer
would reasonably ask "why interpolate when the clean frame is one directory
away?" for the 14 corruption types. This report, and the coding scaffolding
built alongside it, scope temporal recovery to `NoImage`/`FrameLost` only.

---

## 2. Technical feasibility for `NoImage`/`FrameLost` recovery

### Injection point (confirmed in code)

`inference/llava1.5.py:84-92`, inside `LLMPredictor.__call__`:

```python
if self.corruption and len(self.corruption) > 1 and self.corruption != 'NoImage':
    img_path = img_path.replace('nuscenes/samples', f'corruption/{self.corruption}')
if self.corruption == 'NoImage':
    img = np.zeros((224, 224, 3), dtype=np.uint8)
else:
    img = Image.open(img_path).convert('RGB')
```

This generic branch means **no inference code change is required at all** to
add a recovery condition — passing `--corruption Recovered` and having
`data/corruption/Recovered/<CAM>/<filename>` pre-populated is sufficient; the
existing path-substitution logic reads it automatically. (This is a
simplification versus `claude/PLAN.md`'s original assumption that a new
branch would need to be added to the inference script.)

### Data gap, quantified

- `data/nuscenes/samples/<CAM>/` has **exactly 200 files per camera** — one
  per unique `frame_token` referenced in `data/drivebench-test-final.json`
  (1,461 questions, 200 unique frames), confirmed by both a direct file
  count and by reading `tools/fetch_nuscenes_subset.py`, which only ever
  copies filenames it finds referenced in the QA JSON. **No temporal
  neighbor frames are present locally** — this repo currently holds only the
  exact keyframes the benchmark needs, nothing either side of them in time.
- The metadata to find those neighbors *is* present:
  `/mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval/sample.json`
  and `sample_data.json` (34,149 samples). Verified directly this session:
  `frame_token` in the QA JSON **is** the nuScenes `sample_token` — e.g.
  frame `7e4c3282bc2a4402b5d1d6705f9eb844` resolves in `sample.json` to
  `{"prev": "bf4b53f308314e65bdb35fae61499f6a", "next": "54daaee2a25346cca0e33fb76eeb39cf", "scene_token": "da41ecbc644b4915b84bb732e35ebf8c", ...}`
  — a direct, zero-ambiguity lookup, no fuzzy matching needed.
- What's still missing is the **image bytes** for those `prev`/`next`
  samples (and, ideally, the denser sweep frames between them) — those
  aren't part of the camera-keyframes-only download already used for this
  repo's 200 frames, so getting them means either widening the existing
  keyframe download to include the immediate scene neighbors, or (better,
  see next point) downloading the corresponding sweep blob(s).

### The frame-rate problem, and why sweeps matter

nuScenes keyframes are sampled at 2 Hz (0.5 s apart) — at ordinary driving
speed that is substantial motion between frames, and large-motion gaps are a
documented hard case for video frame interpolation (VFI) methods, which are
mostly designed and evaluated on the much smaller inter-frame motion of
high-fps video. nuScenes camera **sweeps** run at 12 Hz (~83 ms apart) — six
times denser — which is a far more tractable interpolation input, and dense
enough that for many missing keyframes the nearest real sweep frame could
simply be substituted directly, with no interpolation model needed at all,
as a cheap zero-ML baseline. Sweeps are not currently downloaded either.

### Candidate technique

- **Baseline (build/validate first): nearest-sweep substitution.** No model,
  no training, no GPU beyond what inference already needs — just pick the
  temporally closest real frame. Cheapest way to test whether the hypothesis
  ("a real frame beats a blank/hallucinated one") has any legs before
  investing further.
- **If the baseline shows signal: FILM (arXiv:2202.04901, "Frame
  Interpolation for Large Motion")** over the AMT (CVPR 2023) candidate
  originally suggested in `claude/LITERATURE_REVIEW.md` — FILM is
  purpose-built for exactly the large-motion regime nuScenes keyframe gaps
  fall into, which AMT is not specifically optimized for. Both are
  open-source, pretrained-checkpoint-available, convolutional (not
  requiring diffusion-scale compute).

---

## 3. Evidence-based reasoning on "better readings"

**The mechanism a recovered frame would fix is already reproduced in this
project.** `claude/LITERATURE_REVIEW.md` documents running this repo's own
pipeline clean vs. `NoImage` on LLaVA-1.5-7B: 82.8% of predictions changed,
with `NoImage` answers fabricating specific, plausible-sounding scene details
that don't correspond to anything the model saw. A recovered frame with real
(if imperfect) visual content should reduce this — the model has something
genuine to describe instead of falling back on language priors.

**Supporting evidence from adjacent literature (new this session):**
"Resilient Sensor Fusion under Adverse Sensor Failures via Multi-Modal Expert
Fusion" (arXiv:2503.19776) reports that under complete camera failure, a
compensation mechanism retains 87.9% NDS / 85.2% mAP relative to the
full-sensor baseline — i.e. meaningful recovery from total camera loss *is*
achievable in this problem family. The caveat: that recovery happens via
feature-level multi-expert fusion inside a model built for it, not by
reconstructing the missing input frame ahead of a frozen, off-the-shelf VLM
the way this thesis proposes. **No paper found does the frame-level
reconstruction → frozen-VLM version of this experiment** — so the precedent
supports "recovery is possible in this problem family," not "this specific
method will work," which is the honest framing for a thesis motivation
section.

**A real risk to the measurement itself, not just the method.** Because
DriveBench's `gpt_score` judge already showed it can be fooled by a fluent,
confident, wrong answer under `NoImage` (per the 82.8% finding above), a
recovered-but-imperfect frame that still produces a *plausible-sounding*
answer might not show as large a score delta as the underlying grounding
improvement actually warrants — the aggregate GPT rubric score could mute a
real gain. The existing per-item logging
(`res/<model>/gpt_eval_logs/<corruption>_eval_log.json`, keyed by
`scene_token_frame_token_question` per `evaluate/request.py`) already
supports pulling matched triples (clean vs. `NoImage`/`FrameLost` vs.
`Recovered`) for a **paired** comparison plus manual/qualitative spot
checks, rather than trusting the aggregate score alone — this should be the
thesis's actual evidence, not just the printed `final_scores` dict.

---

## 4. Cost / effort estimate for an eventual pilot

| Step | Cost | Notes |
|---|---|---|
| Download nuScenes sweep or neighbor-keyframe blob(s) | Free (nuscenes.org account already used) but bandwidth/time | Only for the ~200 frames' scenes, not the full dataset |
| Nearest-sweep baseline implementation | Dev time only | No new ML dependency |
| FILM/AMT integration (optional, second phase) | Dev time + one pretrained checkpoint download | GPU already needed for LLaVA inference |
| Re-running inference for the `Recovered` condition | GPU time, same as any one of the existing 17 runs | `script/llava1.5-7b.sh` pattern |
| `accuracy` / `language_metrics` (BLEU/ROUGE-L/CIDEr) scoring | Free, local | No API key needed |
| `gpt_score` scoring | Small but nonzero — `gpt-3.5-turbo` calls, ~1,461 questions per condition scored | Needs `OPENAI_API_KEY`; run free metrics first to sanity-check before spending here |

---

## Sources

- This repo: `inference/llava1.5.py`, `evaluate/eval.py`, `evaluate/request.py`,
  `tools/fetch_nuscenes_subset.py`, `data/drivebench-test-final.json`,
  `data/corruption/FrameLost/`, `data/nuscenes/samples/`,
  `/mnt/c/Users/Raj/Downloads/v1.0-trainval_meta/v1.0-trainval/{sample,sample_data}.json`
- `claude/PLAN.md`, `claude/LITERATURE_REVIEW.md` (this project, prior session)
- Li, Z. et al. "FILM: Frame Interpolation for Large Motion." arXiv:2202.04901
- "Resilient Sensor Fusion under Adverse Sensor Failures via Multi-Modal Expert Fusion." arXiv:2503.19776
- Li, Z. et al. "AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation." CVPR 2023 (carried over from `claude/LITERATURE_REVIEW.md`, re-evaluated against FILM above)
- nuScenes sensor rate documentation (nuscenes.org, CVPR 2020 paper) — 2 Hz keyframes, 12 Hz camera sweeps, 20 Hz LIDAR
