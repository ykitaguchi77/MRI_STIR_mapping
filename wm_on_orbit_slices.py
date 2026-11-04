"""
wm_on_orbit_slices.py — what does the automatic WM extraction do on the
ORBITAL-FAT / EXTRAOCULAR-MUSCLE slices?

The production pipeline reads the white-matter reference on posterior brain
slices and maps SIR on the orbit. This script runs the automatic extraction
(LWBNA brain mask + histogram) on the muscle-belly slices themselves — the
slices `--offsets` (default 2,3,4) behind the eyeball slice, toward the brain —
and compares the result with the reader's manual WM value from the same patient
(annotate_wm_roi.py, deep-brain slice). WM is the same tissue, so the ratio
shows how far the estimate drifts when only the frontal lobes are in the slice.

Outputs: orbit_slices.csv (one row per slice), orbit_slices.png (the middle
offset per case with brain mask / WM band, and the % difference).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sir_analysis as S
from annotate_wm_roi import ellipse_mask
from inspect_wm_outliers import peaks_of
from orbital_sir_map import detect_globes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", default="wm_manual_roi/annotations.json")
    p.add_argument("--lwbna", default="models/brain_lwbna_best.pt")
    p.add_argument("--out", default="wm_manual_roi")
    p.add_argument("--offsets", default="2,3,4",
                   help="slices behind the eyeball slice (toward the brain) to analyse")
    args = p.parse_args()
    offsets = [int(x) for x in args.offsets.split(",")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ann = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    from lwbna_brain import LWBNABrainSegmenter
    seg = LWBNABrainSegmenter(args.lwbna)

    rows, panels = [], []
    for case in ann["cases"]:
        rec = ann["annotations"][case]
        zm = int(rec["slice"])
        vol, _ = S.load_dicom_series(ann["dicom"], case)
        sl = vol[:, :, zm]
        roi = np.zeros(sl.shape, bool)
        for e in rec["ellipses"]:
            roi |= ellipse_mask(sl.shape, e["cx"], e["cy"], e["rx"], e["ry"])
        manual = float(sl[roi].mean())

        _, gz = detect_globes(vol)
        if gz is None:
            print(f"[{case}] no eyeball pair found - skipped")
            continue
        d = 1 if zm > gz else -1                      # from the globe toward the deep brain
        orbit = [gz + d * k for k in offsets if 0 <= gz + d * k < vol.shape[2]]
        zc = orbit[len(orbit) // 2]
        for z in orbit:
            brain = seg.brain_mask(vol, z)
            r = {"case": case, "orbit_slice": z, "offset_from_globe": abs(z - gz),
                 "central": int(z == zc), "manual_slice": zm,
                 "manual": manual, "brain_frac_pct": np.nan, "n_peaks": 0,
                 "wm_peak": np.nan, "band_mean": np.nan, "pct_band": np.nan, "pct_peak": np.nan}
            b2 = wm_mask = None
            if brain is not None:
                b2 = brain[:, :, z]
                r["brain_frac_pct"] = 100 * b2.mean()
                try:
                    wm = S.extract_white_matter_reference(vol, brain)
                    vals = vol[:, :, z][b2]
                    vals = vals[vals > 0]
                    centers, hist_s, peaks, _ = peaks_of(vals)
                    r["n_peaks"] = int(len(peaks))
                    r["wm_peak"] = float(centers[peaks[0]] if len(peaks) else centers[np.argmax(hist_s)])
                    r["band_mean"] = wm["wm_mean"]
                    r["pct_band"] = 100 * (wm["wm_mean"] / manual - 1)
                    r["pct_peak"] = 100 * (r["wm_peak"] / manual - 1)
                    wm_mask = wm["wm_mask"][:, :, z]
                except ValueError:
                    pass
            rows.append(r)
            if z == zc:
                panels.append((case, z, vol[:, :, z], b2, wm_mask, r))

    import csv
    out = Path(args.out)
    with open(out / "orbit_slices.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    pb = np.array([r["pct_band"] for r in rows], float)
    pk = np.array([r["pct_peak"] for r in rows], float)
    bf = np.array([r["brain_frac_pct"] for r in rows], float)
    none = np.isnan(bf).sum()
    ok = np.isfinite(pb)
    print(f"{len(rows)} muscle-level slices (globe+{args.offsets}) from {len(ann['cases'])} cases; "
          f"no brain mask on {none}; "
          f"WM extracted on {ok.sum()}")
    print(f"brain area on orbit slices: median {np.nanmedian(bf):.1f}% of image "
          f"(deep slices used by the reader: ~20-34%)")
    print(f"auto band mean vs manual: mean {np.nanmean(pb):+.1f}%  median {np.nanmedian(pb):+.1f}%  "
          f"range {np.nanmin(pb):+.1f}..{np.nanmax(pb):+.1f}%")
    print(f"WM peak       vs manual: mean {np.nanmean(pk):+.1f}%  median {np.nanmedian(pk):+.1f}%")
    print(f"slices within +-15% of manual (band): {np.mean(np.abs(pb[ok]) <= 15) * 100:.0f}%")
    print(f"\n{'case':14s}{'z':>3}{'brain%':>7}{'peaks':>6}{'manual':>8}{'band':>8}{'%diff':>8}{'peak':>8}{'%diff':>8}")
    for r in rows:
        if r["central"]:
            print(f"{r['case'].replace('_STIR_Coronal', ''):14s}{r['orbit_slice']:>3}{r['brain_frac_pct']:>7.1f}"
                  f"{r['n_peaks']:>6}{r['manual']:>8.0f}{r['band_mean']:>8.0f}{r['pct_band']:>+8.1f}"
                  f"{r['wm_peak']:>8.0f}{r['pct_peak']:>+8.1f}")

    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.3, nrow * 3.5), squeeze=False)
    for k, (case, z, sl, b2, wm_mask, r) in enumerate(panels):
        a = axes[k // ncol, k % ncol]
        a.imshow(sl, cmap="gray", vmax=np.percentile(sl, 99.5))
        if b2 is not None:
            a.contour(b2, levels=[0.5], colors="magenta", linewidths=0.7)
        if wm_mask is not None:
            rgba = np.zeros(sl.shape + (4,))
            rgba[wm_mask] = (0.2, 0.5, 1.0, 0.5)
            a.imshow(rgba)
        t = f"{case.replace('_STIR_Coronal', '')}  z={z} (globe+{r['offset_from_globe']})  brain {r['brain_frac_pct']:.0f}%"
        t += "\n" + (f"auto {r['band_mean']:.0f} vs manual {r['manual']:.0f} ({r['pct_band']:+.0f}%)"
                     if np.isfinite(r["pct_band"]) else "no WM extracted")
        a.set_title(t, fontsize=8)
        a.axis("off")
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("Automatic WM extraction on the extraocular-muscle level slice (blue = WM band, magenta = brain mask)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "orbit_slices.png", dpi=110)
    print(f"\nSaved {out / 'orbit_slices.csv'} and {out / 'orbit_slices.png'}")


if __name__ == "__main__":
    main()
