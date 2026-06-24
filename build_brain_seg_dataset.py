"""
build_brain_seg_dataset.py — SAM2 -> pixel-mask dataset for LWBNA-UNet.

Exports exact binary MASK IMAGES for semantic segmentation (pixel masks, not
polygons — polygons lose detail when they have few points), and applies
geometric QC to DROP failed SAM2 segmentations (whole-head grabs, one-sided
leaks, ragged/leaky masks).

Output:
  out/images/{train,val}/<series>_<z>.png   (8-bit windowed STIR slice)
  out/masks/{train,val}/<series>_<z>.png     (0/255 brain mask)
  out/qc_dropped.png                          (montage of rejected masks)

Run with the MRI_TOM venv and HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import cv2

import sir_analysis as S
from sam2_brain import SAM2BrainSegmenter


def list_series(d):
    names = set()
    for f in glob.glob(str(Path(d) / "*.dcm")):
        m = re.match(r"(.+)_STIR_Coronal_\d+\.dcm$", Path(f).name)
        if m:
            names.add(m.group(1))
    return sorted(names)


def window8(sl):
    v = np.clip(sl, 0, np.percentile(sl, 99.5))
    return (v / max(v.max(), 1e-6) * 255).astype(np.uint8)


def qc_brain_mask(m: np.ndarray):
    """Return (ok, reason, stats) for a candidate brain mask (bool 2D)."""
    H, W = m.shape
    area = int(m.sum())
    frac = area / (H * W)
    if frac < 0.04:
        return False, "too_small", frac
    if frac > 0.40:
        return False, "too_large", frac
    # border touching: brain should not hug the image frame
    border = m[:2].sum() + m[-2:].sum() + m[:, :2].sum() + m[:, -2:].sum()
    if border / max(area, 1) > 0.03:
        return False, "touches_border", frac
    # solidity = area / convex-hull area (leaky/ragged masks score low)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "empty", frac
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = cv2.contourArea(c) / max(hull, 1)
    if solidity < 0.84:
        return False, "ragged", frac
    # left/right balance about the centroid column (reject one-sided leaks)
    xs = np.where(m.any(axis=0))[0]
    cx = int(np.round(np.average(np.arange(W), weights=m.sum(axis=0))))
    left = m[:, :cx].sum()
    right = m[:, cx:].sum()
    if min(left, right) / max(left, right, 1) < 0.25:
        return False, "one_sided", frac
    return True, "ok", frac


def main():
    p = argparse.ArgumentParser(description="SAM2 -> LWBNA-UNet mask-image dataset")
    p.add_argument("--dicom", required=True)
    p.add_argument("--out", default="brain_seg_dataset")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-anterior", type=int, default=6)
    p.add_argument("--review", metavar="DIR",
                   help="also write an overlay preview per KEPT slice here, for "
                        "manual curation (delete bad previews, then run "
                        "curate_brain_dataset.py)")
    args = p.parse_args()

    out = Path(args.out)
    for sub in ("images/train", "images/val", "masks/train", "masks/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    review = Path(args.review) if args.review else None
    if review:
        review.mkdir(parents=True, exist_ok=True)

    series = list_series(args.dicom)
    if args.limit:
        series = series[:args.limit]
    patients = sorted({s.split("_")[0] for s in series})
    n_val = max(1, int(round(len(patients) * args.val_frac)))
    val_patients = set(patients[::max(1, len(patients) // n_val)][:n_val])
    print(f"{len(series)} series / {len(patients)} patients ({len(val_patients)} val)")

    seg = SAM2BrainSegmenter()
    n_keep = n_drop = 0
    drop_reasons: dict[str, int] = {}
    dropped_examples = []   # (img8, mask) for the QC montage
    for si, s in enumerate(series):
        split = "val" if s.split("_")[0] in val_patients else "train"
        try:
            vol, _ = S.load_dicom_series(args.dicom, s)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {s}: {exc}"); continue
        kept = 0
        for z in range(vol.shape[2] - args.max_anterior):
            try:
                bm = seg.brain_mask(vol, z, strict=True)
            except Exception:  # noqa: BLE001
                bm = None
            if bm is None:
                continue
            m = bm[:, :, z]
            ok, reason, _ = qc_brain_mask(m)
            if not ok:
                n_drop += 1
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                if len(dropped_examples) < 24:
                    dropped_examples.append((window8(vol[:, :, z]), m.copy(), f"{s}_{z}:{reason}"))
                continue
            stem = f"{s}_{z:02d}"
            img8 = window8(vol[:, :, z])
            cv2.imwrite(str(out / f"images/{split}/{stem}.png"), img8)
            cv2.imwrite(str(out / f"masks/{split}/{stem}.png"), (m * 255).astype(np.uint8))
            if review is not None:                 # overlay preview for manual curation
                rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 1)
                cv2.imwrite(str(review / f"{split}__{stem}.png"), rgb)
            n_keep += 1
            kept += 1
        print(f"[{si+1}/{len(series)}] {s} -> kept {kept} ({split})")

    # QC montage of dropped masks
    if dropped_examples:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        k = len(dropped_examples); cols = 6; rows = (k + cols - 1) // cols
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
        for a in ax.flat:
            a.axis("off")
        for a, (img, m, lab) in zip(ax.flat, dropped_examples):
            a.imshow(img, cmap="gray")
            a.contour(m.astype(float), levels=[0.5], colors="red", linewidths=0.8)
            a.set_title(lab, fontsize=6)
        fig.suptitle("Dropped SAM2 masks (QC-rejected)", fontsize=11)
        fig.tight_layout(); fig.savefig(out / "qc_dropped.png", dpi=120)
        print(f"Saved QC montage: {out/'qc_dropped.png'}")

    print(f"\nKept {n_keep} slices, dropped {n_drop} ({drop_reasons})")


if __name__ == "__main__":
    main()
