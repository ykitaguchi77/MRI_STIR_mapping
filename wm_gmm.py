"""
wm_gmm.py — white-matter reference by a 1-D Gaussian mixture on log STIR signal.

Three classes on the log signal of the (eroded) brain mask, darkest = white
matter. The WM reference is the MODE of the WM class, exp(mu - sigma^2), which
matches where a reader puts an ROI (pure WM core) better than the class mean.

Width ("spread") constraint, selectable:
  full     each class has its own spread (plain GMM)
  tied     all three classes share one spread (estimated from THIS slice)
  wmgm     WM and GM share a spread, the bright (CSF) class is free — CSF is often
           a much sharper peak on small frontal slices
No fixed numbers: every spread is estimated per slice, and spreads on the log
scale are relative (a global gain change of the scanner does not change them).

Outliers are removed before the fit by a robust rule on the log signal
(|x - median| > 3.5 x 1.4826 x MAD). Unlike a fixed percentile trim it removes
nothing when there are no outliers, so the dark tail is kept on slices where
white matter is only a few % of the brain.

`wm_reference_hybrid` is the recommended estimate for any given slice: the
simple histogram WM peak (most accurate when WM forms its own peak), checked
against the tied-GMM mode; when they disagree by more than `max_disagree` the
peak has landed on grey matter and the GMM mode is used instead (flagged for QC).

CLI — white-matter intensity on one chosen slice:
  python wm_gmm.py --dicom <STIR_DICOM_dir> --series <prefix> --slice 14 --out wm_slice14
writes <out>.png (slice with used pixels / WM class) and <out>_masks.npz.

This is an alternative to the original band-mean method in sir_analysis.py,
which is left unchanged. Validation: Knowledge/manual-vs-auto-white-matter.md.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _inits(x: np.ndarray, n_init: int, seed: int = 0):
    """Starting points: k-means on the log signal (as scikit-learn does) plus a
    few random restarts. Yields (mu, sd, w)."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    for k in range(n_init):
        km = KMeans(3, n_init=1, random_state=seed + k).fit(x[:, None])
        lab = km.labels_
        mu = np.array([x[lab == j].mean() for j in range(3)])
        sd = np.array([max(x[lab == j].std(), 1e-3) for j in range(3)])
        w = np.array([np.mean(lab == j) for j in range(3)])
        yield mu, sd, w
    for _ in range(n_init):
        mu = np.sort(rng.choice(x, 3, replace=False))
        yield mu, np.full(3, x.std() / 3), np.full(3, 1 / 3)


def _em(x, mu, sd, w, mode, n_iter, tol):
    prev = -np.inf
    ll = prev
    for _ in range(n_iter):
        z = (x[:, None] - mu) / sd
        logp = np.log(w) - np.log(sd) - 0.5 * z ** 2
        m = logp.max(1, keepdims=True)
        p = np.exp(logp - m)
        ll = float((m[:, 0] + np.log(p.sum(1))).sum())
        r = p / p.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-12
        w = nk / nk.sum()
        mu = (r * x[:, None]).sum(0) / nk
        ss = (r * (x[:, None] - mu) ** 2).sum(0)           # per-class sum of squares
        if mode == "full":
            var = ss / nk
        elif mode == "tied":
            var = np.full(3, ss.sum() / nk.sum())
        elif mode == "wmgm":
            o = np.argsort(mu)
            var = ss / nk
            shared = (ss[o[0]] + ss[o[1]]) / (nk[o[0]] + nk[o[1]])
            var[o[0]] = var[o[1]] = shared
        elif mode.startswith("cap"):
            # relative cap: no class may be wider than k x the pooled spread of
            # this slice (dimensionless; k from the mode name, e.g. "cap1.5")
            k = float(mode[3:] or 1.5)
            pooled = ss.sum() / nk.sum()
            var = np.minimum(ss / nk, k * k * pooled)
        else:
            raise ValueError(mode)
        sd = np.sqrt(np.maximum(var, 1e-6))
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return mu, sd, w, ll


def fit_gmm1d(x: np.ndarray, mode: str = "wmgm", n_init: int = 4, n_iter: int = 500,
              tol: float = 1e-8):
    """3-class 1-D mixture (EM) with optional tied spreads; best log-likelihood
    over several starts. Returns (mu, sd, w) sorted by mu."""
    best = None
    for mu0, sd0, w0 in _inits(x, n_init):
        mu, sd, w, ll = _em(x, mu0, sd0, w0, mode, n_iter, tol)
        if best is None or ll > best[3]:
            best = (mu, sd, w, ll)
    mu, sd, w, _ = best
    o = np.argsort(mu)
    return mu[o], sd[o], w[o]


def mad_inliers(x: np.ndarray, k: float = 3.5) -> np.ndarray:
    """Boolean mask of values within median +- k robust SD (1.4826 x MAD)."""
    med = np.median(x)
    rsd = 1.4826 * np.median(np.abs(x - med))
    if rsd <= 0:
        return np.ones(x.shape, bool)
    return np.abs(x - med) <= k * rsd


def brain_values(sl: np.ndarray, brain2d: np.ndarray, erode: int = 3) -> np.ndarray:
    """Positive signal inside the brain mask eroded by `erode` pixels."""
    return sl[_core_mask(sl, brain2d, erode)].astype(float)


def _core_mask(sl, brain2d, erode):
    core = ndimage.binary_erosion(brain2d, iterations=erode) if erode > 0 else brain2d.astype(bool)
    return core & (sl > 0)


def wm_reference(sl: np.ndarray, brain2d: np.ndarray, mode: str = "tied",
                 erode: int = 3, mad_k: float | None = 3.5) -> dict:
    """WM reference (mode of the WM class) on one slice inside the brain mask.
    `mad_k=None` disables the outlier rule.

    Masks returned (same shape as `sl`, for QC overlays / saving as NIfTI):
      used_mask  pixels fed to the fit (eroded brain, MAD inliers)
      wm_mask    used pixels assigned to the WM class (highest posterior)
    """
    core = _core_mask(sl, brain2d, erode)
    x = np.log(sl[core].astype(float))
    keep = mad_inliers(x, mad_k) if mad_k is not None else np.ones(x.shape, bool)
    if keep.sum() < 50:
        raise ValueError("too few brain pixels")
    mu, sd, w = fit_gmm1d(x[keep], mode)
    logp = np.log(w) - np.log(sd) - 0.5 * ((x[:, None] - mu) / sd) ** 2
    is_wm = keep & (np.argmax(logp, 1) == 0)
    used = np.zeros(sl.shape, bool)
    used[core] = keep
    wm_mask = np.zeros(sl.shape, bool)
    wm_mask[core] = is_wm
    return {"wm_mode": float(np.exp(mu[0] - sd[0] ** 2)), "mu": mu, "sd": sd, "w": w,
            "used_mask": used, "wm_mask": wm_mask}


def histogram_wm_peak(v: np.ndarray, n_bins: int = 128, smooth_sigma: float = 2.0,
                      return_range: bool = False):
    """Simple mode: lowest prominent peak of the smoothed brain histogram
    (same peak search as sir_analysis.extract_white_matter_reference, but the
    peak position is returned instead of the valley-to-valley band mean).
    With return_range=True also returns (lo, hi): the half-maximum width of that
    peak, i.e. the intensities of the pixels that make up the peak."""
    lo, hi = np.percentile(v, (1.0, 99.0))
    core = v[(v >= lo) & (v <= hi)]
    hist, edges = np.histogram(core, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    hs = gaussian_filter1d(hist.astype(float), smooth_sigma)
    peaks, _ = find_peaks(hs, prominence=hs.max() * 0.08, distance=max(3, n_bins // 20))
    i = int(peaks[0] if len(peaks) else np.argmax(hs))
    if not return_range:
        return float(centers[i])
    half = hs[i] / 2
    a = i
    while a > 0 and hs[a - 1] >= half:
        a -= 1
    b = i
    while b < n_bins - 1 and hs[b + 1] >= half:
        b += 1
    return float(centers[i]), (float(edges[a]), float(edges[b + 1]))


def wm_reference_hybrid(sl: np.ndarray, brain2d: np.ndarray, erode: int = 3,
                        max_disagree: float = 0.15, mad_k: float | None = 3.5) -> dict:
    """Histogram WM peak, falling back to the tied-GMM mode when the two differ
    by more than `max_disagree` (relative). Returns the chosen value, both
    estimates, `fallback` (True = GMM used; worth a look in QC) and masks:
      used_mask  pixels fed to the estimate (eroded brain, MAD inliers)
      wm_mask    pixels the chosen value rests on: the pixels under the
                 histogram peak (half-maximum width) when the peak is used,
                 the GMM WM class when the GMM value is used
      peak_mask / gmm_wm_mask  both, regardless of which was chosen"""
    peak, (lo, hi) = histogram_wm_peak(brain_values(sl, brain2d, erode), return_range=True)
    g = wm_reference(sl, brain2d, mode="tied", erode=erode, mad_k=mad_k)
    fallback = abs(peak / g["wm_mode"] - 1) > max_disagree
    peak_mask = g["used_mask"] & (sl >= lo) & (sl <= hi)
    return {"wm": g["wm_mode"] if fallback else peak, "wm_peak": peak,
            "wm_gmm": g["wm_mode"], "fallback": bool(fallback), "peak_range": (lo, hi),
            "used_mask": g["used_mask"],
            "wm_mask": g["wm_mask"] if fallback else peak_mask,
            "peak_mask": peak_mask, "gmm_wm_mask": g["wm_mask"], "gmm": g}


def main():
    import argparse
    p = argparse.ArgumentParser(description="White-matter intensity on one chosen slice")
    p.add_argument("--dicom", required=True, help="folder with the STIR DICOM files")
    p.add_argument("--series", help="DICOM filename prefix of the series")
    p.add_argument("--slice", type=int, required=True, help="slice index (0-based)")
    p.add_argument("--lwbna", default="models/brain_lwbna_best.pt")
    p.add_argument("--erode", type=int, default=3)
    p.add_argument("--max-disagree", type=float, default=0.15)
    p.add_argument("--out", help="output prefix for the QC image and masks")
    args = p.parse_args()

    import sir_analysis as S
    from lwbna_brain import LWBNABrainSegmenter
    vol, _ = S.load_dicom_series(args.dicom, args.series)
    z = args.slice
    brain = LWBNABrainSegmenter(args.lwbna).brain_mask(vol, z)
    if brain is None:
        raise SystemExit("no brain found on this slice")
    sl = vol[:, :, z].astype(float)
    r = wm_reference_hybrid(sl, brain[:, :, z], erode=args.erode, max_disagree=args.max_disagree)
    print(f"slice {z}: WM = {r['wm']:.1f}  ({'GMM fallback' if r['fallback'] else 'histogram peak'}; "
          f"peak {r['wm_peak']:.1f}, GMM mode {r['wm_gmm']:.1f}; "
          f"{int(r['used_mask'].sum())} pixels used, {int(r['wm_mask'].sum())} behind the value)")
    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        np.savez_compressed(f"{args.out}_masks.npz", used_mask=r["used_mask"],
                            wm_mask=r["wm_mask"], peak_mask=r["peak_mask"],
                            gmm_wm_mask=r["gmm_wm_mask"], slice=z, wm=r["wm"],
                            fallback=r["fallback"])
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for a in ax:
            a.imshow(sl, cmap="gray", vmax=np.percentile(sl, 99.5))
            a.axis("off")
        rgba = np.zeros(sl.shape + (4,))
        rgba[r["used_mask"]] = (1.0, 0.9, 0.2, 0.25)
        rgba[r["wm_mask"]] = (0.2, 0.5, 1.0, 0.55)
        ax[1].imshow(rgba)
        ax[1].set_title(f"WM {r['wm']:.0f} ({'GMM' if r['fallback'] else 'peak'}); "
                        "yellow = used, blue = pixels behind the value", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{args.out}.png", dpi=110)
        print(f"Saved {args.out}.png and {args.out}_masks.npz")


if __name__ == "__main__":
    main()
