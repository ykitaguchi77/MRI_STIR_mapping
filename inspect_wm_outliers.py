"""
inspect_wm_outliers.py — why does the automatic WM reference deviate from the
reader's ROI on some cases?

For each case whose same-slice automatic value differs from the manual ellipse
mean by more than --thr percent (or the cases given with --cases), draws:

  [1] STIR slice + reader ellipses          [2] brain mask + automatic WM band
  [3] histogram inside the brain mask with the detected peaks, the chosen WM
      peak and the WM band (shaded), plus the histogram of the reader's ROI
      voxels (green) and its mean — so you can see whether the automatic band
      sits on the reader's white matter or has slid onto gray matter.

Usage
-----
  python inspect_wm_outliers.py --ann wm_manual_roi/annotations.json \
      --csv wm_manual_roi/comparison_per_case.csv --thr 20 --out wm_manual_roi/outliers
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import sir_analysis as S
from annotate_wm_roi import ellipse_mask


def peaks_of(vals: np.ndarray, n_bins: int = 128, smooth_sigma: float = 2.0):
    """Reproduce the peak search of sir_analysis.extract_white_matter_reference."""
    lo, hi = np.percentile(vals, (1.0, 99.0))
    core = vals[(vals >= lo) & (vals <= hi)]
    hist, edges = np.histogram(core, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_s = gaussian_filter1d(hist.astype(float), smooth_sigma)
    peaks, props = find_peaks(hist_s, prominence=hist_s.max() * 0.08, distance=max(3, n_bins // 20))
    return centers, hist_s, peaks, props


def main():
    p = argparse.ArgumentParser(description="Inspect outlier cases of manual-vs-auto WM")
    p.add_argument("--ann", default="wm_manual_roi/annotations.json")
    p.add_argument("--csv", default="wm_manual_roi/comparison_per_case.csv")
    p.add_argument("--dicom")
    p.add_argument("--lwbna", default="models/brain_lwbna_best.pt")
    p.add_argument("--thr", type=float, default=20.0, help="|%% difference| threshold")
    p.add_argument("--cases", nargs="*", help="explicit case prefixes instead of --thr")
    p.add_argument("--out", default="wm_manual_roi/outliers")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    ann = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    dicom_dir = args.dicom or ann["dicom"]
    with open(args.csv, encoding="utf-8") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    if args.cases:
        sel = [c for c in rows if any(c.startswith(x) for x in args.cases)]
    else:
        sel = [c for c, r in rows.items() if abs(float(r["pct_auto_slice"])) >= args.thr]
    if not sel:
        raise SystemExit("No cases selected.")
    print("Inspecting:", ", ".join(sel))

    from lwbna_brain import LWBNABrainSegmenter
    seg = LWBNABrainSegmenter(args.lwbna)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for case in sel:
        rec = ann["annotations"][case]
        z = int(rec["slice"])
        vol, _ = S.load_dicom_series(dicom_dir, case)
        sl = vol[:, :, z]
        roi = np.zeros(sl.shape, dtype=bool)
        for e in rec["ellipses"]:
            roi |= ellipse_mask(sl.shape, e["cx"], e["cy"], e["rx"], e["ry"])
        man_vals = sl[roi]

        brain = seg.brain_mask(vol, z)
        wm = S.extract_white_matter_reference(vol, brain)
        b2 = brain[:, :, z]
        vals = sl[b2]
        vals = vals[vals > 0]
        centers, hist_s, peaks, props = peaks_of(vals)
        low, high = wm["low_bound"], wm["high_bound"]
        in_band = np.mean((man_vals >= low) & (man_vals <= high)) * 100
        brain_frac = b2.mean() * 100

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        vmax = np.percentile(sl, 99.5)
        ax[0].imshow(sl, cmap="gray", vmax=vmax)
        ax[1].imshow(sl, cmap="gray", vmax=vmax)
        ax[1].contour(b2, levels=[0.5], colors="magenta", linewidths=0.8)
        rgba = np.zeros(sl.shape + (4,))
        rgba[wm["wm_mask"][:, :, z]] = (0.2, 0.5, 1.0, 0.5)
        ax[1].imshow(rgba)
        for a in ax[:2]:
            for e in rec["ellipses"]:
                a.add_patch(Ellipse((e["cx"], e["cy"]), 2 * e["rx"], 2 * e["ry"],
                                    fill=False, edgecolor="#00ff66", lw=1.5))
            a.axis("off")
        ax[0].set_title(f"{case.replace('_STIR_Coronal', '')}  z={z}   manual ROI mean {man_vals.mean():.0f}")
        ax[1].set_title(f"brain mask ({brain_frac:.1f}% of slice)  auto WM {wm['wm_mean']:.0f} "
                        f"({100 * (wm['wm_mean'] / man_vals.mean() - 1):+.1f}%)")

        a = ax[2]
        a.plot(centers, hist_s, color="k", lw=1.2, label="brain-mask histogram (smoothed)")
        a.plot(centers[peaks], hist_s[peaks], "rv", ms=7, label="prominent peaks")
        a.axvspan(low, high, color="tab:blue", alpha=0.25, label=f"auto WM band [{low:.0f}, {high:.0f}]")
        a.axvline(wm["wm_mean"], color="tab:blue", lw=1.5, ls="--", label=f"auto WM mean {wm['wm_mean']:.0f}")
        hm, em = np.histogram(man_vals, bins=40, range=(centers[0], centers[-1]))
        scale = hist_s.max() / max(hm.max(), 1) * 0.6
        a.bar(0.5 * (em[:-1] + em[1:]), hm * scale, width=em[1] - em[0], color="#00cc55", alpha=0.6,
              label=f"reader ROI voxels (n={man_vals.size}, scaled)")
        a.axvline(man_vals.mean(), color="green", lw=1.5, ls="--", label=f"manual mean {man_vals.mean():.0f}")
        a.set_xlabel("raw STIR signal"); a.set_ylabel("voxels")
        a.set_title(f"{in_band:.0f}% of reader-ROI voxels fall inside the auto band")
        a.legend(fontsize=7)
        fig.tight_layout()
        fp = out / f"{case}.png"
        fig.savefig(fp, dpi=120)
        plt.close(fig)

        peak_str = ", ".join(f"{centers[i]:.0f}" for i in peaks)
        summary.append((case, z, man_vals.mean(), wm["wm_mean"], low, high, brain_frac, in_band, peak_str))
        print(f"{case:28s} z={z:2d} manual {man_vals.mean():6.0f}  auto {wm['wm_mean']:6.0f}  "
              f"band [{low:.0f},{high:.0f}]  brain {brain_frac:4.1f}% of slice  "
              f"reader voxels in band {in_band:3.0f}%  peaks at {peak_str}")
    print(f"\nFigures in {out}")


if __name__ == "__main__":
    main()
