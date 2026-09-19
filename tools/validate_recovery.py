"""
Pilot: measure recovery quality against originals WITHOUT running any VLM.

For each (frame, camera, method) triple, computes:

  PSNR   Peak Signal-to-Noise Ratio (dB) — higher is better
  SSIM   Structural Similarity Index (0–1) — higher is better
  L1     Mean absolute pixel error (0–255 scale) — lower is better
  time   Wall-clock recovery time (ms) — not measured here, use --time-build
         to benchmark build_recovered_dataset separately

Central-crop metrics are also reported (inner 50% of each axis) to flag cases
where the model gets the background right but loses small foreground objects
such as pedestrians, signs, or traffic lights.

Requires:  scikit-image  (pip install scikit-image)

Usage:
    python tools/validate_recovery.py \\
        --originals  data/nuscenes/samples \\
        --framelost-dir data/corruption/FrameLost \\
        --recovered  data/corruption/Recovered_Previous \\
                     data/corruption/Recovered_Nearest \\
                     data/corruption/Recovered_LinearBlend \\
                     data/corruption/Recovered_RIFE \\
        --output data/recovery_validation.json \\
        --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
from PIL import Image


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    except ImportError:
        # Lightweight fallback (less accurate but no extra dep)
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        a, b = a.astype(float), b.astype(float)
        mu_a, mu_b = a.mean(), b.mean()
        sig_a = np.sqrt(((a - mu_a) ** 2).mean())
        sig_b = np.sqrt(((b - mu_b) ** 2).mean())
        sig_ab = ((a - mu_a) * (b - mu_b)).mean()
        return float(
            (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
            / ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a ** 2 + sig_b ** 2 + c2))
        )


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(float) - b.astype(float))))


def _mask_metrics(orig: np.ndarray, rec: np.ndarray, threshold: int = 10) -> dict:
    """
    Metrics computed only over pixels where the recovered image actually has
    content (e.g. a projected LiDAR point), instead of the whole frame.

    Full-frame PSNR/SSIM unfairly penalizes a sparse point-overlay recovery
    method (e.g. LiDAR projection) against a near-black background -- it will
    always score badly there even if every point is placed correctly, since
    it's compared against a normally-exposed photo pixel-for-pixel/window-for-
    window. This reports how accurate the recovered pixels are *where they
    exist*, plus how much of the frame they actually cover, so a sparse method
    isn't mistaken for "broken" just because it isn't photorealistic.

    SSIM needs spatial neighborhoods and isn't meaningful over a scattered,
    non-contiguous pixel mask, so only pointwise metrics (PSNR/L1) are
    reported here, alongside the coverage fraction.
    """
    mask = rec.max(axis=2) > threshold
    coverage_pct = 100.0 * mask.sum() / mask.size
    if mask.sum() == 0:
        return {"psnr_masked": None, "l1_masked": None, "coverage_pct": 0.0}
    orig_m = orig[mask].astype(float)
    rec_m = rec[mask].astype(float)
    mse = np.mean((orig_m - rec_m) ** 2)
    psnr_masked = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    l1_masked = float(np.mean(np.abs(orig_m - rec_m)))
    return {
        "psnr_masked": psnr_masked,
        "l1_masked": l1_masked,
        "coverage_pct": coverage_pct,
    }


def _center_crop(arr: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Return the central frac×frac region of an HWC array."""
    h, w = arr.shape[:2]
    h0 = int(h * (1 - frac) / 2)
    w0 = int(w * (1 - frac) / 2)
    return arr[h0:h0 + int(h * frac), w0:w0 + int(w * frac)]


def compare_pair(
    orig_path: str,
    rec_path: str,
    size: int = 224,
) -> Optional[dict]:
    """Return metric dict or None if either image is missing."""
    if not os.path.exists(orig_path) or not os.path.exists(rec_path):
        return None
    orig = np.array(Image.open(orig_path).convert("RGB").resize((size, size)))
    rec = np.array(Image.open(rec_path).convert("RGB").resize((size, size)))

    crop_orig = _center_crop(orig)
    crop_rec = _center_crop(rec)

    result = {
        "psnr": _psnr(orig, rec),
        "ssim": _ssim(orig, rec),
        "l1": _l1(orig, rec),
        "psnr_crop": _psnr(crop_orig, crop_rec),
        "ssim_crop": _ssim(crop_orig, crop_rec),
        "l1_crop": _l1(crop_orig, crop_rec),
    }
    result.update(_mask_metrics(orig, rec))
    return result


def _collect_framelost_pairs(framelost_dir: str) -> list[tuple[str, str]]:
    """
    Return (camera, filename) pairs present in the FrameLost directory.
    These are the frames we want to evaluate recovery for.
    """
    pairs = []
    if not os.path.isdir(framelost_dir):
        return pairs
    for cam in sorted(os.listdir(framelost_dir)):
        cam_dir = os.path.join(framelost_dir, cam)
        if not os.path.isdir(cam_dir):
            continue
        for fname in sorted(os.listdir(cam_dir)):
            pairs.append((cam, fname))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--originals",
        default="data/nuscenes/samples",
        help="Root directory of original nuScenes keyframe images (default: data/nuscenes/samples)",
    )
    parser.add_argument(
        "--framelost-dir",
        default="data/corruption/FrameLost",
        help="FrameLost corruption directory used to select which frames to evaluate",
    )
    parser.add_argument(
        "--recovered",
        nargs="+",
        required=True,
        help="One or more recovered-condition directories, "
             "e.g. data/corruption/Recovered_Previous data/corruption/Recovered_Nearest",
    )
    parser.add_argument(
        "--output",
        default="data/recovery_validation.json",
        help="Output JSON path (default: data/recovery_validation.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N (camera, frame) pairs (pilot mode)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Resize images to size×size before comparing (default: 224)",
    )
    args = parser.parse_args()

    pairs = _collect_framelost_pairs(args.framelost_dir)
    if not pairs:
        print(f"No frames found in {args.framelost_dir}")
        return

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Evaluating {len(pairs)} (camera, frame) pairs across {len(args.recovered)} method(s)...")

    results = []
    method_names = [os.path.basename(d) for d in args.recovered]

    for cam, fname in pairs:
        orig_path = os.path.join(args.originals, cam, fname)
        row: dict = {"camera": cam, "filename": fname, "methods": {}}

        for method_dir, method_name in zip(args.recovered, method_names):
            rec_path = os.path.join(method_dir, cam, fname)
            t0 = time.perf_counter()
            metrics = compare_pair(orig_path, rec_path, size=args.size)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if metrics is None:
                row["methods"][method_name] = {"available": False}
            else:
                metrics["load_compare_ms"] = round(elapsed_ms, 2)
                metrics["available"] = True
                row["methods"][method_name] = metrics

        results.append(row)

    # Aggregate summary per method
    print("\n=== Recovery quality summary ===")
    for method_name in method_names:
        m_results = [
            r["methods"][method_name]
            for r in results
            if r["methods"].get(method_name, {}).get("available")
        ]
        if not m_results:
            print(f"  {method_name}: no results")
            continue
        n = len(m_results)
        avg_psnr = sum(r["psnr"] for r in m_results) / n
        avg_ssim = sum(r["ssim"] for r in m_results) / n
        avg_l1 = sum(r["l1"] for r in m_results) / n
        avg_psnr_c = sum(r["psnr_crop"] for r in m_results) / n
        avg_ssim_c = sum(r["ssim_crop"] for r in m_results) / n
        print(f"  {method_name} (n={n}):")
        print(f"    full  — PSNR: {avg_psnr:.2f} dB   SSIM: {avg_ssim:.4f}   L1: {avg_l1:.2f}")
        print(f"    crop  — PSNR: {avg_psnr_c:.2f} dB   SSIM: {avg_ssim_c:.4f}")

        masked = [r for r in m_results if r.get("psnr_masked") is not None]
        if masked:
            avg_cov = sum(r["coverage_pct"] for r in m_results) / n
            avg_psnr_m = sum(r["psnr_masked"] for r in masked) / len(masked)
            avg_l1_m = sum(r["l1_masked"] for r in masked) / len(masked)
            print(f"    masked — PSNR: {avg_psnr_m:.2f} dB   L1: {avg_l1_m:.2f}   "
                  f"coverage: {avg_cov:.1f}% of frame (n={len(masked)})")
            print(f"    (masked = pixel accuracy only where the recovered image has "
                  f"content, e.g. sparse LiDAR points -- fairer than full-frame "
                  f"PSNR/SSIM for non-photorealistic overlay methods)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
