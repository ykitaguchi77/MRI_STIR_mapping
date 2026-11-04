"""
orbital_sir_map.py — Whole-orbit Signal Intensity Ratio (SIR) mapping on STIR.

Pipeline (no muscle segmentation required):
  1. Load a coronal STIR series (a folder of DICOMs, or a NIfTI volume).
  2. Auto-extract the brain WHITE-MATTER reference from posterior brain slices
     (brain ROI -> intensity histogram -> white-matter band; median across
     slices for robustness). See sir_analysis.auto_white_matter_reference.
  3. Compute a voxel-wise SIR = signal / white-matter reference over the
     anterior orbit slices and render it as a heat-map overlay, with the
     SIR = 2.0 abnormal-threshold contour (Higashiyama et al. 2015).

Outputs a QC figure (STIR vs SIR map per slice) and, optionally, the SIR volume
as NIfTI.

Examples
--------
  # from a DICOM folder, auto slice ranges:
  python orbital_sir_map.py --dicom HANDAI_STIR_Coronal --series 100_20150904 \
      --brain-slices 8-13 --orbit-slices 18-21 \
      --plot results/sirmap_100.png --save-nifti results/sirmap_100.nii.gz

  # from a NIfTI volume:
  python orbital_sir_map.py --nifti scan_stir.nii.gz \
      --brain-slices 8-13 --orbit-slices 18-21 --plot results/sirmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

import sir_analysis as S

ABNORMAL_SIR = S.ABNORMAL_SIR


def parse_slices(spec: str) -> list[int]:
    """'8-13' -> [8..13]; '8,9,12' -> [8,9,12]; mix allowed."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def detect_globes(stir: np.ndarray, min_area: int = 150, max_area: int = 3000,
                  min_circularity: float = 0.4, bright_pct: float = 96.0):
    """
    Per-slice eyeball score. The globes are the two very bright, ROUND blobs
    that sit symmetrically about the midline at the same height. Requiring
    roundness + a symmetric pair keeps bright CSF / ventricles / sinus mucosa on
    brain slices from being mistaken for eyeballs.

    Returns (globe_score[z], globe_center_slice or None).
    """
    from scipy import ndimage

    nz = stir.shape[2]
    H, W = stir.shape[:2]
    hi_global = np.percentile(stir[stir > 0], bright_pct)
    score = np.zeros(nz)
    for z in range(nz):
        sl = stir[:, :, z]
        pos = sl[sl > 0]
        if pos.size == 0:
            continue
        # a per-slice threshold catches globes that are dim relative to the
        # whole volume (coil sensitivity); the global one guards bright slices
        hi = min(hi_global, np.percentile(pos, bright_pct))
        bright = ndimage.binary_opening(sl > hi, iterations=1)
        bright = ndimage.binary_fill_holes(bright)
        lbl, n = ndimage.label(bright)
        if n < 2:
            continue
        blobs = []
        for i in range(1, n + 1):
            m = lbl == i
            area = int(m.sum())
            if not (min_area <= area <= max_area):
                continue
            # circularity = 4*pi*A / P^2 (perimeter from the eroded boundary)
            perim = int((m & ~ndimage.binary_erosion(m)).sum())
            circ = 4 * np.pi * area / max(perim, 1) ** 2
            if circ < min_circularity:
                continue
            ys, xs = np.where(m)
            blobs.append((area, circ, ys.mean(), xs.mean()))
        # best symmetric pair: similar height, on opposite sides of the midline,
        # similar area
        best = 0.0
        for a in range(len(blobs)):
            for b in range(a + 1, len(blobs)):
                A, cA, yA, xA = blobs[a]
                B, cB, yB, xB = blobs[b]
                if abs(yA - yB) > 0.08 * H:
                    continue
                if (xA - W / 2) * (xB - W / 2) >= 0:          # same side
                    continue
                if min(A, B) / max(A, B) < 0.3:
                    continue
                sep = abs(xA - xB)
                # globes are ~60 mm apart; the lateral ventricles are a bright,
                # round, symmetric pair too but sit close to the midline
                if not (0.22 * W <= sep <= 0.5 * W):
                    continue
                best = max(best, (A + B) * (cA + cB) / 2)
        score[z] = best
    if score.max() == 0:
        return score, None
    return score, int(np.argmax(score))


def auto_orbit_and_brain_slices(stir: np.ndarray):
    """
    Best-effort split into (brain_slices, orbit_slices).

    Orbit slices are detected by the eyeballs (two bright, round, symmetric
    blobs, see detect_globes). The tissue-rich slices furthest from the orbit
    block are used for the brain reference. Heuristic — print the choice and
    verify with the QC figure.
    """
    nz = stir.shape[2]
    tissue_area = np.zeros(nz)                  # how much soft tissue each slice has
    for z in range(nz):
        sl = stir[:, :, z]
        pos = sl[sl > 0]
        tissue_area[z] = (sl > np.percentile(pos, 55)).sum() if pos.size else 0

    globe_score, orbit_center = detect_globes(stir)
    if orbit_center is None:
        # fall back to the old brightness-only rule
        hi = np.percentile(stir[stir > 0], 98)
        orbit_center = int(np.argmax([(stir[:, :, z] > hi).sum() for z in range(nz)]))
        print("[auto] WARNING: no eyeball pair found; orbit slice from brightness only")
    orbit = [z for z in range(nz) if abs(z - orbit_center) <= 2]
    # Brain = tissue-rich slices well away from the orbit block (the cerebrum).
    brain = [z for z in range(nz)
             if min(abs(z - o) for o in orbit) >= 4
             and tissue_area[z] > 0.5 * tissue_area.max()]
    brain = brain or [z for z in range(nz) if z not in orbit]
    print(f"[auto] orbit slices={orbit} (globe peak {orbit_center}), "
          f"brain slices={brain} - verify with --plot")
    return brain, orbit


def save_plot(stir, sir, orbit_slices, wm_ref, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = np.percentile(stir, 99.5)
    n = len(orbit_slices)
    fig, ax = plt.subplots(2, n, figsize=(n * 3.2, 6.6), squeeze=False)
    im = None
    for j, z in enumerate(orbit_slices):
        ax[0, j].imshow(stir[:, :, z], cmap="gray", vmax=vmax)
        ax[0, j].set_title(f"STIR slice {z}", fontsize=9)
        ax[0, j].axis("off")

        ax[1, j].imshow(stir[:, :, z], cmap="gray", vmax=vmax)
        im = ax[1, j].imshow(np.ma.masked_invalid(sir[:, :, z]), cmap="jet",
                             vmin=0, vmax=3, alpha=0.55)
        ax[1, j].contour(np.nan_to_num(sir[:, :, z]), levels=[ABNORMAL_SIR],
                         colors="white", linewidths=0.8)
        ax[1, j].set_title(f"SIR map (slice {z})", fontsize=9)
        ax[1, j].axis("off")

    if im is not None:
        cb = fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.025, pad=0.01)
        cb.set_label("SIR (signal / white matter)")
    fig.suptitle(f"Orbital SIR mapping (WM ref={wm_ref:.0f}, "
                 f"white contour = SIR {ABNORMAL_SIR})", fontsize=12)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Whole-orbit SIR mapping on STIR")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dicom", help="folder of DICOM slices")
    src.add_argument("--nifti", help="STIR NIfTI volume (raw intensities)")
    p.add_argument("--series", help="DICOM filename prefix, e.g. 100_20150904")
    p.add_argument("--brain-slices", help="brain slices for WM ref, e.g. 8-13")
    p.add_argument("--orbit-slices", help="orbit slices to map, e.g. 18-21")
    p.add_argument("--bins", type=int, default=128)
    p.add_argument("--sam2", action="store_true",
                   help="use SAM2 to segment the cerebrum for the WM reference "
                        "(clean, excludes sinuses/orbit; needs transformers + cache)")
    p.add_argument("--lwbna", metavar="WEIGHTS",
                   help="use a distilled LWBNA-UNet model (.pt) for the brain mask "
                        "(fast production inference, no SAM2 needed)")
    p.add_argument("--tissue-pct", type=float, default=35.0,
                   help="percentile below which voxels are treated as background")
    p.add_argument("--plot", help="output QC PNG")
    p.add_argument("--save-nifti", help="output SIR-map NIfTI")
    args = p.parse_args()

    if args.dicom:
        stir, affine = S.load_dicom_series(args.dicom, args.series)
    else:
        stir, img = S.load_volume(args.nifti)
        affine = img.affine
    print(f"Volume: {stir.shape}  signal range {stir.min():.0f}-{stir.max():.0f}")

    if args.brain_slices and args.orbit_slices:
        brain = parse_slices(args.brain_slices)
        orbit = parse_slices(args.orbit_slices)
    else:
        brain, orbit = auto_orbit_and_brain_slices(stir)

    brain_mask_fn, source = None, "morphological ROI"
    if args.lwbna:
        from lwbna_brain import LWBNABrainSegmenter
        print(f"Loading distilled LWBNA-UNet: {args.lwbna}")
        brain_mask_fn = LWBNABrainSegmenter(args.lwbna).brain_mask
        source = "LWBNA-UNet cerebrum"
    elif args.sam2:
        from sam2_brain import SAM2BrainSegmenter
        print("Loading SAM2 (facebook/sam2-hiera-large) for brain segmentation...")
        brain_mask_fn = SAM2BrainSegmenter().brain_mask
        source = "SAM2 cerebrum"
    wm_ref, per_slice = S.auto_white_matter_reference(
        stir, brain, n_bins=args.bins, brain_mask_fn=brain_mask_fn)
    print(f"White-matter reference = {wm_ref:.1f} "
          f"({source}, median over slices {sorted(per_slice)})")

    sir, tissue = S.sir_map(stir, wm_ref, orbit, tissue_pct=args.tissue_pct)

    # quick summary per orbit slice
    print(f"\n{'slice':>6}{'tissue px':>11}{'mean SIR':>10}{'%>2.0':>8}{'max SIR':>9}")
    for z in orbit:
        vals = sir[:, :, z][tissue[:, :, z]]
        if vals.size:
            print(f"{z:>6}{vals.size:>11}{vals.mean():>10.2f}"
                  f"{100 * np.mean(vals > ABNORMAL_SIR):>7.1f}%{vals.max():>9.2f}")

    if args.plot:
        save_plot(stir, sir, orbit, wm_ref, args.plot)
    if args.save_nifti:
        Path(args.save_nifti).parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.nan_to_num(sir).astype(np.float32), affine),
                 args.save_nifti)
        print(f"Saved SIR NIfTI: {args.save_nifti}")


if __name__ == "__main__":
    main()
