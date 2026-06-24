"""
build_brain_dataset.py — auto-label brain (cerebrum) on STIR with SAM2 and export
a YOLO11-seg dataset (knowledge distillation: SAM2 -> YOLO11n-seg).

For every coronal STIR series it:
  1. loads the DICOM volume,
  2. runs SAM2 (strict) on each slice to get a clean cerebrum mask
     (anterior/orbit slices without cerebrum are skipped),
  3. writes a windowed PNG image and a YOLO-seg polygon label (class 0 = brain).

Split is per-PATIENT (the leading id before the first '_') so the same patient
never appears in both train and val.

Run with the MRI_TOM venv and HF_HUB_OFFLINE=1.

  python build_brain_dataset.py --dicom HANDAI_STIR_Coronal \
      --out brain_dataset --limit 40 --val-frac 0.2
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


def list_series(dicom_dir: str) -> list[str]:
    names = set()
    for f in glob.glob(str(Path(dicom_dir) / "*.dcm")):
        m = re.match(r"(.+)_STIR_Coronal_\d+\.dcm$", Path(f).name)
        if m:
            names.add(m.group(1))
    return sorted(names)


def window_png(sl: np.ndarray) -> np.ndarray:
    """Windowed 8-bit RGB image (same windowing SAM2 saw)."""
    v = np.clip(sl, 0, np.percentile(sl, 99.5))
    v = (v / max(v.max(), 1e-6) * 255).astype(np.uint8)
    return np.stack([v] * 3, axis=-1)


def mask_to_polygons(mask: np.ndarray, eps_frac: float = 0.004, min_area: int = 200):
    """Largest contour(s) of a binary mask -> list of (N,2) int polygons."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = eps_frac * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) >= 3:
            polys.append(ap)
    return polys


def main():
    p = argparse.ArgumentParser(description="SAM2 -> YOLO11-seg brain dataset")
    p.add_argument("--dicom", required=True, help="folder of DICOM series")
    p.add_argument("--out", default="brain_dataset", help="output dataset dir")
    p.add_argument("--limit", type=int, default=0, help="max series (0 = all)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-anterior", type=int, default=6,
                   help="skip this many most-anterior slices (orbit, no cerebrum)")
    args = p.parse_args()

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    series = list_series(args.dicom)
    if args.limit:
        series = series[:args.limit]
    # per-patient split (id before first '_')
    patients = sorted({s.split("_")[0] for s in series})
    n_val = max(1, int(round(len(patients) * args.val_frac)))
    val_patients = set(patients[::max(1, len(patients) // n_val)][:n_val])
    print(f"{len(series)} series / {len(patients)} patients "
          f"({len(val_patients)} val patients)")

    seg = SAM2BrainSegmenter()
    n_img = n_lab = 0
    for si, s in enumerate(series):
        split = "val" if s.split("_")[0] in val_patients else "train"
        try:
            vol, _ = S.load_dicom_series(args.dicom, s)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {s}: {exc}")
            continue
        nz = vol.shape[2]
        H, W = vol.shape[0], vol.shape[1]
        kept = 0
        for z in range(nz - args.max_anterior):  # posterior slices only
            try:
                bm = seg.brain_mask(vol, z, strict=True)
            except Exception:  # noqa: BLE001
                bm = None
            if bm is None:
                continue
            polys = mask_to_polygons(bm[:, :, z])
            if not polys:
                continue
            stem = f"{s}_{z:02d}"
            cv2.imwrite(str(out / f"images/{split}/{stem}.png"), window_png(vol[:, :, z]))
            with open(out / f"labels/{split}/{stem}.txt", "w") as fh:
                for poly in polys:
                    norm = poly.astype(np.float32) / [W, H]
                    coords = " ".join(f"{x:.5f} {y:.5f}" for x, y in norm)
                    fh.write(f"0 {coords}\n")
            n_img += 1
            n_lab += len(polys)
            kept += 1
        print(f"[{si+1}/{len(series)}] {s} -> {kept} slices ({split})")

    yaml = out / "brain.yaml"
    yaml.write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: 1\nnames: [brain]\n", encoding="utf-8")
    print(f"\nDone. {n_img} images, {n_lab} polygons. Dataset YAML: {yaml}")
    print("Train:  yolo segment train model=yolo11n-seg.pt "
          f"data={yaml} epochs=100 imgsz=256")


if __name__ == "__main__":
    main()
