"""
compare_wm_manual_vs_auto.py — manual elliptical WM ROI vs automatic WM reference.

For every case annotated with annotate_wm_roi.py, computes on the SAME slice:

  manual      mean raw STIR signal inside the reader's ellipse(s)
  auto_slice  LWBNA brain mask -> histogram low-intensity peak -> WM band mean
              (sir_analysis.extract_white_matter_reference on that slice)
  auto_prod   the production value: median of auto_slice over the brain slices
              the pipeline would use (auto-detected, or --brain-slices)
  hist_in_roi histogram WM extraction restricted to the reader's ellipse(s)
              (isolates the histogram step from the brain-mask step)

and reports ratio / % difference, paired tests, Bland-Altman, a scatter plot
and a per-case QC sheet (reader ellipses vs automatic WM band).

Usage
-----
  python compare_wm_manual_vs_auto.py --ann wm_manual_roi/annotations.json \
      --lwbna models/brain_lwbna_best.pt --out wm_manual_roi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sir_analysis as S
from annotate_wm_roi import ellipse_mask, roi_stats


def wm_gmm_same_slice(sl: np.ndarray, brain2d: np.ndarray, erode: int = 3,
                      n_components: int = 3) -> float:
    """Same-slice WM estimate that also works on small frontal-lobe slices:
    erode the brain mask (drop the cortical rim / CSF edge), fit a 3-class
    Gaussian mixture to the log signal and return the mean of the darkest class
    (WM < GM < CSF on STIR)."""
    from scipy import ndimage
    from sklearn.mixture import GaussianMixture
    # NB: scipy treats iterations=0 as "erode until nothing changes" (empty mask)
    core = ndimage.binary_erosion(brain2d, iterations=erode) if erode > 0 else brain2d.astype(bool)
    v = sl[core]
    v = v[v > 0]
    if v.size < 50:
        return float("nan")
    x = np.log(v)[:, None]
    g = GaussianMixture(n_components, random_state=0).fit(x)
    lab = g.predict(x)
    return float(v[lab == int(np.argmin(g.means_[:, 0]))].mean())


def pct_diff(auto: np.ndarray, manual: np.ndarray) -> np.ndarray:
    return 100.0 * (auto - manual) / manual


def summarize(name: str, auto: np.ndarray, manual: np.ndarray) -> dict:
    from scipy import stats

    ok = np.isfinite(auto) & np.isfinite(manual)
    a, m = auto[ok], manual[ok]
    d = pct_diff(a, m)
    ratio = a / m
    res = {
        "comparison": name, "n": int(ok.sum()),
        "manual_mean": float(m.mean()), "auto_mean": float(a.mean()),
        "ratio_mean": float(ratio.mean()), "ratio_sd": float(ratio.std(ddof=1)) if len(a) > 1 else np.nan,
        "pct_diff_mean": float(d.mean()), "pct_diff_sd": float(d.std(ddof=1)) if len(a) > 1 else np.nan,
        "pct_diff_median": float(np.median(d)),
        "pct_diff_min": float(d.min()), "pct_diff_max": float(d.max()),
        "pearson_r": float(stats.pearsonr(a, m)[0]) if len(a) > 2 else np.nan,
    }
    if len(a) > 1:
        res["paired_t_p"] = float(stats.ttest_rel(a, m).pvalue)
        try:
            res["wilcoxon_p"] = float(stats.wilcoxon(a, m).pvalue)
        except ValueError:
            res["wilcoxon_p"] = np.nan
        # Bland-Altman on the raw signal difference (auto - manual)
        diff = a - m
        res["ba_bias"] = float(diff.mean())
        res["ba_loa_low"] = float(diff.mean() - 1.96 * diff.std(ddof=1))
        res["ba_loa_high"] = float(diff.mean() + 1.96 * diff.std(ddof=1))
    return res


def write_csv(path: Path, rows: list[dict]):
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def plot_summary(rows: list[dict], out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man = np.array([r["manual_mean"] for r in rows])
    cols = [("auto_slice_mean", "auto (same slice)", "tab:blue"),
            ("auto_prod_mean", "auto (production, median over slices)", "tab:orange"),
            ("hist_in_roi_mean", "histogram within manual ROI", "tab:green")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    lim = [0, max(man.max(), max(np.nanmax([r[c] for r in rows]) for c, _, _ in cols)) * 1.1]
    ax[0].plot(lim, lim, "k--", lw=0.8, label="identity")
    for c, lab, col in cols:
        v = np.array([r[c] for r in rows])
        ax[0].scatter(man, v, s=28, c=col, label=lab, alpha=0.8)
    ax[0].set_xlabel("manual ellipse ROI mean (raw STIR)")
    ax[0].set_ylabel("automatic WM mean")
    ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_aspect("equal")
    ax[0].legend(fontsize=8); ax[0].set_title("manual vs automatic")

    # Bland-Altman (% difference vs mean) for the same-slice comparison
    for c, lab, col in cols:
        v = np.array([r[c] for r in rows])
        d = pct_diff(v, man)
        ax[1].scatter((man + v) / 2, d, s=28, c=col, alpha=0.8, label=lab)
        ax[1].axhline(np.nanmean(d), color=col, lw=1)
    ax[1].axhline(0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("mean of manual and automatic")
    ax[1].set_ylabel("(auto - manual) / manual  [%]")
    ax[1].set_title("Bland-Altman (% difference)"); ax[1].legend(fontsize=8)

    # per-case bars of % difference (same slice)
    d = pct_diff(np.array([r["auto_slice_mean"] for r in rows]), man)
    order = np.argsort(d)
    ax[2].barh(range(len(rows)), d[order], color="tab:blue")
    ax[2].set_yticks(range(len(rows)))
    ax[2].set_yticklabels([rows[i]["case"].replace("_STIR_Coronal", "") for i in order], fontsize=7)
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].axvline(np.nanmean(d), color="tab:red", lw=1, ls="--", label=f"mean {np.nanmean(d):+.1f}%")
    ax[2].set_xlabel("auto (same slice) vs manual  [%]"); ax[2].legend(fontsize=8)
    ax[2].set_title("per-case % difference")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved plot: {out}")


def plot_qc(qc_items: list[dict], out: Path, ncol: int = 3):
    """Per-case sheet, two panels per case side by side:
    left  = raw STIR slice with the reader's ellipses (green) only,
    right = the same slice with the automatic WM band (blue) inside the LWBNA
            brain mask (magenta contour) plus the reader's ellipses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    n = len(qc_items)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(ncol * 2 * 3.3, nrow * 3.6), squeeze=False)
    for k, it in enumerate(qc_items):
        sl = it["slice"]
        vmax = np.percentile(sl, 99.5)
        a_raw = axes[k // ncol, 2 * (k % ncol)]
        a_ovl = axes[k // ncol, 2 * (k % ncol) + 1]
        for a in (a_raw, a_ovl):
            a.imshow(sl, cmap="gray", vmax=vmax)
            for e in it["ellipses"]:
                a.add_patch(Ellipse((e["cx"], e["cy"]), 2 * e["rx"], 2 * e["ry"],
                                    fill=False, edgecolor="#00ff66", lw=1.5))
            a.axis("off")
        if it["brain"] is not None:
            a_ovl.contour(it["brain"], levels=[0.5], colors="magenta", linewidths=0.7)
        if it["wm"] is not None:
            rgba = np.zeros(it["wm"].shape + (4,), dtype=float)
            rgba[it["wm"]] = (0.2, 0.5, 1.0, 0.45)      # translucent blue on WM band
            a_ovl.imshow(rgba)
        name = it["case"].replace("_STIR_Coronal", "")
        a_raw.set_title(f"{name}  z={it['z']}" + chr(10) + f"manual {it['manual']:.0f}", fontsize=8)
        a_ovl.set_title("auto WM band" + chr(10) + f"auto {it['auto']:.0f} ({it['pct']:+.1f}%)", fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol, 2 * (k % ncol)].axis("off")
        axes[k // ncol, 2 * (k % ncol) + 1].axis("off")
    fig.suptitle("left: STIR + reader ellipses (green)   right: + automatic WM band (blue) "
                 "inside LWBNA brain mask (magenta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved QC sheet: {out}")


def main():
    p = argparse.ArgumentParser(description="Compare manual WM ROI with automatic WM reference")
    p.add_argument("--ann", default="wm_manual_roi/annotations.json")
    p.add_argument("--dicom", help="override the DICOM folder stored in the annotation file")
    p.add_argument("--lwbna", default="models/brain_lwbna_best.pt")
    p.add_argument("--brain-slices", help="fixed brain slices for auto_prod (default: auto-detect)")
    p.add_argument("--bins", type=int, default=128)
    p.add_argument("--out", default="wm_manual_roi")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="case prefixes to drop, e.g. 182_ (brain mask leaked into the orbit)")
    args = p.parse_args()

    ann = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    dicom_dir = args.dicom or ann["dicom"]
    cases = [c for c in ann["cases"] if c in ann["annotations"]]
    excluded = [c for c in cases if any(c.startswith(x) for x in args.exclude)]
    cases = [c for c in cases if c not in excluded]
    if excluded:
        print(f"Excluded: {', '.join(excluded)}")
    if not cases:
        raise SystemExit("No annotated cases in the annotation file.")
    print(f"{len(cases)} annotated cases; DICOM = {dicom_dir}")

    from lwbna_brain import LWBNABrainSegmenter
    from orbital_sir_map import auto_orbit_and_brain_slices, parse_slices
    seg = LWBNABrainSegmenter(args.lwbna)

    rows, qc_items = [], []
    for case in cases:
        rec = ann["annotations"][case]
        z = int(rec["slice"])
        vol, _ = S.load_dicom_series(dicom_dir, case)
        sl = vol[:, :, z]
        ells = rec["ellipses"]
        man = roi_stats(sl, ells)

        # automatic WM on the same slice
        brain = seg.brain_mask(vol, z)
        auto_slice, wm_mask, low, high, brain_n = np.nan, None, np.nan, np.nan, 0
        if brain is not None:
            try:
                wm = S.extract_white_matter_reference(vol, brain, n_bins=args.bins)
                auto_slice, wm_mask = wm["wm_mean"], wm["wm_mask"][:, :, z]
                low, high, brain_n = wm["low_bound"], wm["high_bound"], int(brain.sum())
            except ValueError as exc:
                print(f"[{case}] auto WM failed on slice {z}: {exc}")

        # production value (median over brain slices)
        brain_slices = parse_slices(args.brain_slices) if args.brain_slices \
            else auto_orbit_and_brain_slices(vol)[0]
        try:
            auto_prod, per = S.auto_white_matter_reference(vol, brain_slices, n_bins=args.bins,
                                                           brain_mask_fn=seg.brain_mask)
        except ValueError as exc:
            print(f"[{case}] production WM failed: {exc}")
            auto_prod, per = np.nan, {}

        # histogram step alone, inside the reader's ellipse(s)
        roi3 = np.zeros(vol.shape, dtype=bool)
        for e in ells:
            roi3[:, :, z] |= ellipse_mask(sl.shape, e["cx"], e["cy"], e["rx"], e["ry"])
        try:
            hist_in_roi = S.extract_white_matter_reference(vol, roi3, n_bins=args.bins)["wm_mean"]
        except ValueError:
            hist_in_roi = np.nan

        gmm = wm_gmm_same_slice(sl, brain[:, :, z]) if brain is not None else np.nan

        row = {
            "case": case, "slice": z, "n_ellipses": len(ells),
            "manual_mean": man["mean"], "manual_std": man["std"], "manual_n": man["n"],
            "auto_slice_mean": auto_slice, "auto_slice_low": low, "auto_slice_high": high,
            "brain_mask_n": brain_n,
            "auto_prod_mean": auto_prod, "prod_slices": "-".join(map(str, sorted(per))) if per else "",
            "prod_slice_used": int(z in per),
            "hist_in_roi_mean": hist_in_roi,
            "gmm_slice_mean": gmm,
            "pct_gmm_slice": pct_diff(gmm, man["mean"]),
            "pct_auto_slice": pct_diff(auto_slice, man["mean"]),
            "pct_auto_prod": pct_diff(auto_prod, man["mean"]),
            "pct_hist_in_roi": pct_diff(hist_in_roi, man["mean"]),
        }
        rows.append(row)
        print(f"{case:30s} z={z:2d}  manual {man['mean']:7.1f}  auto_slice {auto_slice:7.1f} "
              f"({row['pct_auto_slice']:+5.1f}%)  auto_prod {auto_prod:7.1f} "
              f"({row['pct_auto_prod']:+5.1f}%)  hist_in_roi {hist_in_roi:7.1f} "
              f"({row['pct_hist_in_roi']:+5.1f}%)")
        qc_items.append({"case": case, "z": z, "slice": sl, "ellipses": ells,
                         "brain": brain[:, :, z] if brain is not None else None,
                         "wm": wm_mask, "manual": man["mean"], "auto": auto_slice,
                         "pct": row["pct_auto_slice"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "comparison_per_case.csv", rows)

    man = np.array([r["manual_mean"] for r in rows])
    summ = [summarize("auto_slice vs manual", np.array([r["auto_slice_mean"] for r in rows]), man),
            summarize("auto_prod vs manual", np.array([r["auto_prod_mean"] for r in rows]), man),
            summarize("hist_in_roi vs manual", np.array([r["hist_in_roi_mean"] for r in rows]), man),
            summarize("gmm_slice vs manual", np.array([r["gmm_slice_mean"] for r in rows]), man)]
    write_csv(out / "comparison_summary.csv", summ)

    print("\n=== Summary (auto relative to manual) ===")
    print(f"{'comparison':24s}{'n':>4}{'manual':>9}{'auto':>9}{'ratio':>8}{'%diff':>9}"
          f"{'sd':>7}{'median':>9}{'r':>7}{'t-test p':>10}{'wilcoxon':>10}")
    for s in summ:
        print(f"{s['comparison']:24s}{s['n']:>4}{s['manual_mean']:>9.1f}{s['auto_mean']:>9.1f}"
              f"{s['ratio_mean']:>8.3f}{s['pct_diff_mean']:>+9.1f}{s['pct_diff_sd']:>7.1f}"
              f"{s['pct_diff_median']:>+9.1f}{s['pearson_r']:>7.3f}"
              f"{s.get('paired_t_p', np.nan):>10.4f}{s.get('wilcoxon_p', np.nan):>10.4f}")
    print(f"\nPer-case CSV: {out / 'comparison_per_case.csv'}\nSummary CSV:  {out / 'comparison_summary.csv'}")

    plot_summary(rows, out / "comparison_plot.png")
    plot_qc(qc_items, out / "qc_sheet.png")


if __name__ == "__main__":
    main()
