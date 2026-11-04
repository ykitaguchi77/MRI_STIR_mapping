"""
validate_wm_hybrid.py — WM intensity on a chosen slice (histogram peak +
tied-GMM check, wm_gmm.wm_reference_hybrid) against a reader's manual ellipse
ROIs on the same slice.

Annotation JSON: {"dicom": dir, "cases": [prefix, ...], "annotations":
{prefix: {"slice": z, "ellipses": [{"cx","cy","rx","ry"}, ...]}}} (pixels).

For every annotated case it runs on the reader's slice and reports the simple
histogram peak, the tied-GMM mode (MAD outlier rule) and the hybrid value, each
as % difference from the manual ROI mean. It also writes the pixels the
estimate rests on:
  hybrid_masks.npz   per case: used_mask (pixels fed to the fit) and wm_mask
                     (pixels assigned to WM), plus the slice index
  hybrid_masks.png   slice | used pixels (grey) + WM class (blue) + reader ROI
Outputs contain patient images — keep them out of git.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sir_analysis as S
from wm_gmm import wm_reference_hybrid


def ellipse_mask(shape, cx, cy, rx, ry):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ann", required=True)
    p.add_argument("--dicom")
    p.add_argument("--lwbna", default="models/brain_lwbna_best.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", nargs="*", default=[], help="case prefixes to skip, e.g. 182_")
    p.add_argument("--erode", type=int, default=3)
    p.add_argument("--max-disagree", type=float, default=0.15)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lwbna_brain import LWBNABrainSegmenter

    ann = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    dicom = args.dicom or ann["dicom"]
    seg = LWBNABrainSegmenter(args.lwbna)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, panels, masks = [], [], {}
    for case in ann["cases"]:
        if case not in ann["annotations"] or any(case.startswith(e) for e in args.exclude):
            continue
        rec = ann["annotations"][case]
        z = int(rec["slice"])
        vol, _ = S.load_dicom_series(dicom, case)
        sl = vol[:, :, z].astype(float)
        roi = np.zeros(sl.shape, bool)
        for e in rec["ellipses"]:
            roi |= ellipse_mask(sl.shape, e["cx"], e["cy"], e["rx"], e["ry"])
        manual = float(sl[roi].mean())
        brain = seg.brain_mask(vol, z)
        if brain is None:
            print(f"[{case}] no brain mask - skipped")
            continue
        r = wm_reference_hybrid(sl, brain[:, :, z], erode=args.erode, max_disagree=args.max_disagree)
        row = {"case": case, "slice": z, "manual": manual,
               "wm_peak": r["wm_peak"], "wm_gmm": r["wm_gmm"], "wm_hybrid": r["wm"],
               "fallback": int(r["fallback"]),
               "pct_peak": 100 * (r["wm_peak"] / manual - 1),
               "pct_gmm": 100 * (r["wm_gmm"] / manual - 1),
               "pct_hybrid": 100 * (r["wm"] / manual - 1),
               "n_used": int(r["used_mask"].sum()), "n_wm": int(r["wm_mask"].sum()),
               "roi_in_wm_pct": 100 * float(r["wm_mask"][roi].mean())}
        rows.append(row)
        masks[case] = (z, r["used_mask"], r["wm_mask"])
        panels.append((case, sl, roi, r, row))

    with open(out / "hybrid_per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(out / "hybrid_masks.npz",
                        **{f"{c}__used": m[1] for c, m in masks.items()},
                        **{f"{c}__wm": m[2] for c, m in masks.items()},
                        **{f"{c}__slice": np.array(m[0]) for c, m in masks.items()})

    print(f"n = {len(rows)}  (% difference from the manual ROI mean)")
    for key, name in (("pct_peak", "histogram peak"), ("pct_gmm", "tied GMM mode"),
                      ("pct_hybrid", "hybrid")):
        d = np.array([r[key] for r in rows])
        print(f"  {name:15s} {d.mean():+6.1f} ± {d.std(ddof=1):4.1f}   mean|d| {np.abs(d).mean():4.1f}"
              f"   max|d| {np.abs(d).max():5.1f}   >15%: {int((np.abs(d) > 15).sum())}")
    fb = [r["case"].split("_")[0] for r in rows if r["fallback"]]
    print(f"  GMM fallback used on: {', '.join(fb) or 'none'}")

    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.3, nrow * 3.5), squeeze=False)
    for k, (case, sl, roi, r, row) in enumerate(panels):
        a = axes[k // ncol, k % ncol]
        a.imshow(sl, cmap="gray", vmax=np.percentile(sl, 99.5))
        rgba = np.zeros(sl.shape + (4,))
        rgba[r["used_mask"]] = (1.0, 0.9, 0.2, 0.25)
        rgba[r["wm_mask"]] = (0.2, 0.5, 1.0, 0.55)
        a.imshow(rgba)
        a.contour(roi, levels=[0.5], colors="red", linewidths=0.8)
        ys, xs = np.nonzero(r["used_mask"] | roi)
        if len(ys):
            m = 15
            a.set_xlim(max(xs.min() - m, 0), min(xs.max() + m, sl.shape[1]))
            a.set_ylim(min(ys.max() + m, sl.shape[0]), max(ys.min() - m, 0))
        a.set_title(f"{case.split('_')[0]} z={row['slice']}  {'GMM' if r['fallback'] else 'peak'}"
                    f"  {row['pct_hybrid']:+.0f}%", fontsize=8)
        a.axis("off")
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("Pixels used (yellow) and WM class (blue); red = reader ROI", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "hybrid_masks.png", dpi=110)
    print(f"Saved {out / 'hybrid_per_case.csv'}, {out / 'hybrid_masks.npz'}, {out / 'hybrid_masks.png'}")


if __name__ == "__main__":
    main()
