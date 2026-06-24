"""
sir_analysis.py — Extraocular muscle Signal Intensity Ratio (SIR) on STIR MRI.

Computes the SIR used to grade orbital inflammation in Thyroid Eye Disease
(Higashiyama et al., Jpn J Ophthalmol 2015):

    SIR(muscle) = mean signal of the muscle on STIR
                  --------------------------------------
                  mean signal of brain WHITE MATTER on STIR

White matter is used as an internal reference because its STIR signal is stable
across patients/scanners. The white-matter reference is extracted automatically
from a brain ROI by reading the intensity HISTOGRAM and taking the band between
the two valleys that bracket the dominant parenchymal (white-matter) peak.

Inputs
------
- A STIR NIfTI volume (RAW intensities — do NOT min-max/z-score it; SIR is a
  ratio of raw signals).
- A muscle label volume in the TOM500 convention (e.g. produced by predict.py):
      1 SR, 2 LR, 3 MR, 4 IR, 5 ON, 6 FAT, 7 LG, 8 SO, 9 EB
  The rectus/oblique muscles (1,2,3,4,8) are scored.

Typical use
-----------
    python sir_analysis.py \
        --stir  scan_stir.nii.gz \
        --label scan_stir_pred.nii.gz \
        --brain-slice 15 --brain-center 256 256 --brain-radius 45 \
        --split-lr \
        --csv results/sir.csv --plot results/sir.png

If --brain-* are omitted a best-effort automatic brain ROI is used; for clinical
work you should place the ROI explicitly and review the histogram plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
except ImportError:  # graceful fallback — boundaries via Otsu instead of valleys
    _HAVE_SCIPY = False


# TOM500 label convention (see categories.json / CLAUDE.md)
MUSCLE_LABELS = {1: "SR", 2: "LR", 3: "MR", 4: "IR", 8: "SO"}
ABNORMAL_SIR = 2.0  # Higashiyama 2015: SIR > 2.0 is outside the normal range


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_volume(path: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI volume as float32 in its native orientation."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img


def load_dicom_series(folder: str, prefix: str | None = None):
    """
    Stack a coronal DICOM series into a (H, W, Z) RAW float32 volume.

    Files are ordered by InstanceNumber (falling back to the trailing _<n> index
    in the filename). Returns (volume, affine) with the affine built from
    PixelSpacing / SliceThickness. Requires pydicom.
    """
    import glob
    import re
    import pydicom

    pat = f"{prefix}*_*.dcm" if prefix else "*.dcm"
    files = glob.glob(str(Path(folder) / pat))
    if not files:
        raise FileNotFoundError(f"No DICOM files matching {pat} in {folder}")

    def tail_idx(f):
        m = re.search(r"_(\d+)\.dcm$", f)
        return int(m.group(1)) if m else 0

    dss = [pydicom.dcmread(f) for f in files]
    order = np.argsort([float(getattr(d, "InstanceNumber", tail_idx(f)))
                        for d, f in zip(dss, files)])
    vol = np.stack([dss[i].pixel_array.astype(np.float32) for i in order], axis=-1)

    d0 = dss[0]
    ps = [float(x) for x in getattr(d0, "PixelSpacing", [1.0, 1.0])]
    st = float(getattr(d0, "SliceThickness", 1.0))
    affine = np.diag([ps[0], ps[1], st, 1.0])
    return vol, affine


# --------------------------------------------------------------------------- #
# Brain ROI
# --------------------------------------------------------------------------- #
def circular_roi(shape, slice_idx: int, center, radius: float) -> np.ndarray:
    """Boolean 3D mask: a filled circle on a single axial slice."""
    mask = np.zeros(shape, dtype=bool)
    ny, nx = shape[0], shape[1]
    cy, cx = center
    yy, xx = np.ogrid[:ny, :nx]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    mask[:, :, slice_idx] = disk
    return mask


def auto_brain_roi(stir: np.ndarray, radius_frac: float = 0.12) -> np.ndarray:
    """
    Best-effort brain ROI when the user does not place one.

    Heuristic: pick the slice with the most bright parenchyma, then place a disk
    at the centroid of that tissue. Review the plot before trusting the result —
    a manual ROI is strongly preferred for real cases.
    """
    fg = stir > np.percentile(stir[stir > 0], 60)
    tissue_per_slice = fg.reshape(-1, stir.shape[2]).sum(axis=0)
    slice_idx = int(np.argmax(tissue_per_slice))
    ys, xs = np.where(fg[:, :, slice_idx])
    center = (int(ys.mean()), int(xs.mean()))
    radius = radius_frac * min(stir.shape[0], stir.shape[1])
    print(f"[auto-ROI] slice={slice_idx} center={center} radius={radius:.1f} "
          f"(heuristic — verify with --plot)")
    return circular_roi(stir.shape, slice_idx, center, radius)


# --------------------------------------------------------------------------- #
# White-matter reference via histogram boundaries
# --------------------------------------------------------------------------- #
def _otsu_threshold(values: np.ndarray, n_bins: int = 128) -> float:
    """Otsu threshold — fallback split between the low (WM) and high populations."""
    hist, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.cumsum().astype(float)
    w_back = w / w[-1]
    mean = (hist * centers).cumsum() / np.maximum(w, 1)
    mean_tot = mean[-1]
    between = (mean_tot * w_back - mean) ** 2 / np.maximum(w_back * (1 - w_back), 1e-12)
    return float(centers[np.argmax(between)])


def extract_white_matter_reference(
    stir: np.ndarray,
    brain_roi: np.ndarray,
    n_bins: int = 128,
    smooth_sigma: float = 2.0,
    clip_pct: tuple[float, float] = (1.0, 99.0),
):
    """
    Extract the white-matter reference signal from a brain ROI.

    Method (matches the "place ROI on brain, read histogram boundaries" approach):
      1. Collect voxel intensities inside the ROI, clip the extreme tails
         (background / CSF / edema) at the given percentiles.
      2. Build a smoothed histogram. On STIR/T2 the brain parenchyma forms the
         dominant peak; white matter sits on its lower-intensity side.
      3. Take the band between the two histogram VALLEYS that bracket that peak;
         voxels inside the band are the white-matter population.
      4. Reference signal = mean of the white-matter voxels.

    Returns a dict with the reference value, the WM voxel mask, the (low, high)
    intensity boundaries and the histogram (for plotting / QC).
    """
    vals = stir[brain_roi]
    vals = vals[vals > 0]
    if vals.size < 50:
        raise ValueError("Brain ROI contains too few voxels — check ROI placement.")

    lo, hi = np.percentile(vals, clip_pct)
    core = vals[(vals >= lo) & (vals <= hi)]
    hist, edges = np.histogram(core, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    if _HAVE_SCIPY:
        hist_s = gaussian_filter1d(hist.astype(float), smooth_sigma)
        # White matter is DARKER than gray matter on STIR/T2, so the WM peak is
        # the LOWEST-intensity prominent peak — not necessarily the tallest one
        # (on cortex-heavy slices gray matter can dominate by volume).
        peaks, _ = find_peaks(hist_s, prominence=hist_s.max() * 0.08,
                              distance=max(3, n_bins // 20))
        if peaks.size:
            peak_idx = int(peaks[0])           # lowest-intensity prominent peak = WM
        else:
            peak_idx = int(np.argmax(hist_s))  # single mode -> use it
        # bound the WM band by the valleys on each side of the WM peak; the right
        # valley is the WM/GM boundary, which keeps gray matter out of the band.
        valleys, _ = find_peaks(-hist_s)
        left = valleys[valleys < peak_idx]
        right = valleys[valleys > peak_idx]
        low_bound = centers[left[-1]] if left.size else centers[0]
        high_bound = centers[right[0]] if right.size else centers[-1]
        method = "histogram-valley(WM-low)"
    else:
        hist_s = hist.astype(float)
        # Fallback: Otsu split; WM = the lower population around its mode.
        thr = _otsu_threshold(core, n_bins)
        low_pop = core[core <= thr]
        peak = low_pop[np.argmax(np.histogram(low_pop, bins=n_bins)[0])] if low_pop.size else thr
        spread = low_pop.std() if low_pop.size else 1.0
        low_bound, high_bound = peak - spread, peak + spread
        method = "otsu-fallback"

    wm_mask = brain_roi & (stir >= low_bound) & (stir <= high_bound)
    wm_vals = stir[wm_mask]
    if wm_vals.size == 0:
        raise ValueError("No white-matter voxels in band — widen ROI or bins.")

    return {
        "wm_mean": float(wm_vals.mean()),
        "wm_median": float(np.median(wm_vals)),
        "wm_std": float(wm_vals.std()),
        "wm_n": int(wm_vals.size),
        "low_bound": float(low_bound),
        "high_bound": float(high_bound),
        "method": method,
        "wm_mask": wm_mask,
        "hist": (centers, hist_s),
    }


# --------------------------------------------------------------------------- #
# Automatic brain ROI + multi-slice white-matter reference
# --------------------------------------------------------------------------- #
def brain_roi_on_slice(stir: np.ndarray, slice_idx: int,
                       lo_pct: float = 40.0, hi_pct: float = 99.0,
                       upper_frac: float = 0.55, side_frac: float = 0.80,
                       erode: int = 2) -> np.ndarray:
    """
    Auto-place a brain ROI on one coronal slice, restricted to the UPPER brain
    (centrum-semiovale level) so it captures cerebral white matter while
    EXCLUDING the paranasal sinuses / skull base below.

    Steps: keep the parenchyma intensity band (drop air and very-bright CSF /
    sinus mucosa) -> opening -> largest tissue blob with an upper centroid (the
    brain) -> keep only its upper `upper_frac` of rows and central `side_frac`
    width -> erode for safety. Returns a 3D boolean mask on this slice.
    """
    from scipy import ndimage

    sl = stir[:, :, slice_idx]
    pos = sl[sl > 0]
    if pos.size == 0:
        raise ValueError(f"Empty slice {slice_idx}.")
    lo, hi = np.percentile(pos, [lo_pct, hi_pct])
    tissue = (sl > lo) & (sl < hi)
    tissue = ndimage.binary_opening(tissue, iterations=3)
    lbl, n = ndimage.label(tissue)
    if n == 0:
        raise ValueError(f"No brain tissue found on slice {slice_idx}.")

    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    H = sl.shape[0]
    order = np.argsort(sizes)[::-1]
    pick = order[0] + 1
    for idx in order:  # prefer the largest blob whose centroid is in the upper head
        if np.where(lbl == idx + 1)[0].mean() < 0.60 * H:
            pick = idx + 1
            break
    blob = ndimage.binary_fill_holes(lbl == pick)

    ys, xs = np.where(blob)
    y0, y1 = ys.min(), ys.max()
    cx = int(xs.mean())
    ycut = int(y0 + upper_frac * (y1 - y0))
    halfw = int(side_frac * 0.5 * (xs.max() - xs.min()))

    roi = np.zeros_like(blob)
    roi[y0:ycut, max(cx - halfw, 0):cx + halfw] = True
    roi &= blob
    roi = ndimage.binary_erosion(roi, iterations=erode)

    mask = np.zeros(stir.shape, dtype=bool)
    mask[:, :, slice_idx] = roi
    return mask


def auto_white_matter_reference(stir: np.ndarray, brain_slices, n_bins: int = 128,
                                brain_mask_fn=None):
    """
    Robust white-matter reference: extract WM on several brain slices and take
    the MEDIAN of the per-slice means (resistant to one bad slice/ROI).

    brain_mask_fn(stir, slice_idx) -> 3D bool brain ROI. Defaults to the
    morphological `brain_roi_on_slice`; pass a SAM2BrainSegmenter.brain_mask for
    a clean cerebrum mask that excludes the sinuses/orbit.

    Returns (wm_ref, per_slice) where per_slice maps slice_idx -> wm result dict.
    """
    brain_mask_fn = brain_mask_fn or brain_roi_on_slice
    per_slice, means = {}, []
    for z in brain_slices:
        try:
            roi = brain_mask_fn(stir, z)
            wm = extract_white_matter_reference(stir, roi, n_bins=n_bins)
            per_slice[z] = wm
            means.append(wm["wm_mean"])
        except Exception as exc:  # noqa: BLE001 — skip unusable slices, keep going
            print(f"[wm] slice {z} skipped: {exc}")
    if not means:
        raise ValueError("White-matter reference failed on all brain slices.")
    return float(np.median(means)), per_slice


# --------------------------------------------------------------------------- #
# Orbital SIR map  (no muscle segmentation needed)
# --------------------------------------------------------------------------- #
def sir_map(stir: np.ndarray, wm_ref: float, slices,
            tissue_pct: float = 35.0):
    """
    Voxel-wise SIR = signal / white-matter reference over the given (orbit)
    slices. Air/bone background (below `tissue_pct` percentile of in-slice
    signal) is set to NaN so only soft tissue is mapped.

    Returns (sir_volume_with_nan, tissue_mask).
    """
    from scipy import ndimage

    sir = np.full(stir.shape, np.nan, dtype=np.float32)
    tissue = np.zeros(stir.shape, dtype=bool)
    for z in slices:
        sl = stir[:, :, z]
        pos = sl[sl > 0]
        if pos.size == 0:
            continue
        m = sl > np.percentile(pos, tissue_pct)
        m = ndimage.binary_opening(m, iterations=1)
        tissue[:, :, z] = m
        sir[:, :, z][m] = sl[m] / wm_ref
    return sir, tissue


# --------------------------------------------------------------------------- #
# Muscle signals + SIR
# --------------------------------------------------------------------------- #
def muscle_signals(stir: np.ndarray, label: np.ndarray, split_lr: bool = False) -> dict:
    """
    Mean STIR signal per muscle from a TOM500 label volume.

    If split_lr, each muscle is split into right/left by the volume's x-midline
    (column index), since orbital muscles are paired.
    """
    if stir.shape != label.shape:
        raise ValueError(f"STIR {stir.shape} and label {label.shape} shapes differ.")

    out = {}
    midline = stir.shape[1] // 2
    for lab, name in MUSCLE_LABELS.items():
        m = label == lab
        if not split_lr:
            if m.any():
                out[name] = float(stir[m].mean())
            continue
        for side, side_mask in (("R", _col_lt(label.shape, midline)),
                                ("L", _col_ge(label.shape, midline))):
            ms = m & side_mask
            if ms.any():
                out[f"{name}_{side}"] = float(stir[ms].mean())
    return out


def _col_lt(shape, midline):
    g = np.zeros(shape, dtype=bool)
    g[:, :midline, :] = True
    return g


def _col_ge(shape, midline):
    g = np.zeros(shape, dtype=bool)
    g[:, midline:, :] = True
    return g


def compute_sir(muscles: dict, wm_mean: float) -> list[dict]:
    """SIR = muscle_mean / white_matter_mean, flagged against the 2.0 threshold."""
    rows = []
    for name, sig in muscles.items():
        sir = sig / wm_mean
        rows.append({
            "muscle": name,
            "muscle_signal": round(sig, 2),
            "wm_reference": round(wm_mean, 2),
            "SIR": round(sir, 3),
            "abnormal(>2.0)": "yes" if sir > ABNORMAL_SIR else "no",
        })
    return rows


# --------------------------------------------------------------------------- #
# Plot (QC)
# --------------------------------------------------------------------------- #
def save_plot(stir, brain_roi, wm, rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slice_idx = int(np.where(brain_roi.any(axis=(0, 1)))[0][0])
    centers, hist_s = wm["hist"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].imshow(stir[:, :, slice_idx], cmap="gray")
    ax[0].imshow(np.ma.masked_where(~wm["wm_mask"][:, :, slice_idx],
                                    wm["wm_mask"][:, :, slice_idx]),
                 cmap="autumn", alpha=0.6)
    ax[0].set_title(f"Brain ROI + white-matter voxels (slice {slice_idx})")
    ax[0].axis("off")

    ax[1].plot(centers, hist_s, color="k")
    ax[1].axvspan(wm["low_bound"], wm["high_bound"], color="orange", alpha=0.3,
                  label="WM band")
    ax[1].axvline(wm["wm_mean"], color="red", ls="--", label=f"WM mean={wm['wm_mean']:.1f}")
    ax[1].set_title(f"ROI histogram ({wm['method']})")
    ax[1].set_xlabel("STIR signal"); ax[1].set_ylabel("count"); ax[1].legend()

    names = [r["muscle"] for r in rows]
    sirs = [r["SIR"] for r in rows]
    colors = ["crimson" if r["abnormal(>2.0)"] == "yes" else "steelblue" for r in rows]
    ax[2].bar(names, sirs, color=colors)
    ax[2].axhline(ABNORMAL_SIR, color="red", ls="--", label="abnormal > 2.0")
    ax[2].set_title("Muscle SIR"); ax[2].set_ylabel("SIR"); ax[2].legend()
    plt.setp(ax[2].get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Extraocular muscle SIR on STIR MRI")
    p.add_argument("--stir", required=True, help="RAW STIR NIfTI (not normalized)")
    p.add_argument("--label", required=True, help="TOM500 muscle label NIfTI")
    p.add_argument("--brain-slice", type=int, help="axial slice index for brain ROI")
    p.add_argument("--brain-center", type=int, nargs=2, metavar=("Y", "X"),
                   help="brain ROI center (row col)")
    p.add_argument("--brain-radius", type=float, help="brain ROI radius (voxels)")
    p.add_argument("--split-lr", action="store_true", help="split muscles into R/L")
    p.add_argument("--bins", type=int, default=128, help="histogram bins")
    p.add_argument("--csv", help="output CSV path")
    p.add_argument("--plot", help="output QC PNG path")
    args = p.parse_args()

    stir, _ = load_volume(args.stir)
    label, _ = load_volume(args.label)
    label = np.rint(label).astype(np.int16)

    if args.brain_slice is not None and args.brain_center and args.brain_radius:
        brain_roi = circular_roi(stir.shape, args.brain_slice,
                                 tuple(args.brain_center), args.brain_radius)
    else:
        brain_roi = auto_brain_roi(stir)

    wm = extract_white_matter_reference(stir, brain_roi, n_bins=args.bins)
    muscles = muscle_signals(stir, label, split_lr=args.split_lr)
    if not muscles:
        raise SystemExit("No muscle labels (1,2,3,4,8) found in --label volume.")
    rows = compute_sir(muscles, wm["wm_mean"])

    print(f"\nWhite-matter reference: mean={wm['wm_mean']:.2f} "
          f"(band {wm['low_bound']:.1f}-{wm['high_bound']:.1f}, "
          f"n={wm['wm_n']} voxels, {wm['method']})\n")
    print(f"{'muscle':<8}{'signal':>10}{'WM ref':>10}{'SIR':>8}  flag")
    for r in rows:
        print(f"{r['muscle']:<8}{r['muscle_signal']:>10}{r['wm_reference']:>10}"
              f"{r['SIR']:>8}  {r['abnormal(>2.0)']}")

    if args.csv:
        import csv
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved CSV: {args.csv}")
        with open(Path(args.csv).with_suffix(".wm.json"), "w", encoding="utf-8") as f:
            json.dump({k: wm[k] for k in
                       ("wm_mean", "wm_median", "wm_std", "wm_n",
                        "low_bound", "high_bound", "method")}, f, indent=2)

    if args.plot:
        save_plot(stir, brain_roi, wm, rows, args.plot)


if __name__ == "__main__":
    main()
