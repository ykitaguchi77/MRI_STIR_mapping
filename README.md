# MRI STIR Mapping — Orbital Signal Intensity Ratio (SIR) for Thyroid Eye Disease

Lightweight pipeline to compute the **orbital SIR map** on coronal **STIR** MRI for
grading inflammation in Thyroid Eye Disease (TED / Graves' orbitopathy).

The SIR (Higashiyama et al., *Jpn J Ophthalmol* 2015) normalises the orbital
signal by the **cerebral white matter**:

```
SIR = orbital STIR signal / brain white-matter STIR signal      (abnormal: SIR > 2.0)
```

The hard part is obtaining a clean **brain mask** to read the white-matter
reference from. This repo implements that as a distillation pipeline:

```
SAM2 brain mask  →  human curation  →  LWBNA-UNet training  →  fast brain mask
                                                                  ↓
                        histogram white-matter extraction (low-intensity peak)
                                                                  ↓
                         voxel-wise orbital SIR map (heat-map + SIR 2.0 contour)
```

A trained **LWBNA-UNet** model is included (`models/brain_lwbna_best.pt`,
val Dice ≈ 0.978), so inference needs no SAM2.

## Example results

**Brain segmentation** — LWBNA-UNet cerebrum mask merged onto coronal STIR:

![brain mask merged on STIR](assets/example_brain_mask.png)

**White-matter separation** — white matter (blue) extracted within the brain mask
(magenta) by the low-intensity histogram peak; its mean is the SIR reference:

![white matter separation](assets/example_white_matter.png)

## Why this design
- **Morphological skull-stripping fails** on coronal STIR: the brain stays
  connected to the face/sinuses through the skull base, so ROIs leak into the
  orbit/sinuses. **SAM2** segments the cerebrum cleanly from a single point prompt.
- **SAM2 is heavy**; it is distilled into a 2.95M-param **LWBNA-UNet** (pixel
  masks, smooth contours) for fast production inference.
- **White matter is the low-intensity peak** of the brain-ROI histogram on STIR/T2
  (WM < GM), not the tallest peak — see `Knowledge/white-matter-histogram-extraction.md`.

## Files
| File | Role |
|---|---|
| `sam2_brain.py` | SAM2 cerebrum segmenter (teacher / auto-labeller) |
| `build_brain_seg_dataset.py` | SAM2 → pixel-mask dataset + geometric QC + review previews |
| `curate_brain_dataset.py` | apply manual curation (delete bad previews → drop pairs) |
| `lwbna_unet.py` | LWBNA-UNet (PyTorch, Sharma et al. 2022) |
| `train_lwbna_brain.py` | train LWBNA-UNet (Dice + BCE) |
| `lwbna_brain.py` | distilled LWBNA-UNet inference (brain mask) |
| `sir_analysis.py` | white-matter extraction, SIR, orbital SIR map, DICOM I/O |
| `orbital_sir_map.py` | end-to-end CLI (`--lwbna` for production, `--sam2` for labelling) |
| `annotate_wm_roi.py` | Tkinter UI: manual elliptical white-matter ROI (validation); `--orbit-offset N` starts at the muscle level |
| `web_annotate_wm_roi.py` | same annotation in a smartphone browser (Basic auth; expose with `cloudflared tunnel --url`) |
| `compare_wm_manual_vs_auto.py` | manual ROI vs automatic WM reference: stats, Bland-Altman, QC sheet |
| `inspect_wm_outliers.py` | histogram diagnostics for cases where manual and automatic WM disagree |
| `wm_on_orbit_slices.py` | automatic WM on the extraocular-muscle level slices (shows why the band-mean method fails there) |
| `wm_gmm.py` | alternative same-slice WM reference: histogram WM peak checked by a tied 1-D GMM (MAD outlier rule); returns the used / WM pixel masks |
| `validate_wm_hybrid.py` | validates `wm_gmm.wm_reference_hybrid` against the manual ROIs and saves the pixel masks |
| `models/brain_lwbna_best.pt` | trained LWBNA-UNet brain segmenter |
| `Knowledge/` | distilled notes on the method, pitfalls, anatomy, architecture |

## Install
```bash
pip install -r requirements.txt
# torch with CUDA per your platform: https://pytorch.org
```

## Usage

### Inference — orbital SIR map (uses the bundled LWBNA model, no SAM2)
```bash
python orbital_sir_map.py --dicom <STIR_DICOM_dir> --series <patient_date> \
    --brain-slices 8-13 --orbit-slices 18-21 \
    --lwbna models/brain_lwbna_best.pt \
    --plot out.png --save-nifti out.nii.gz
```
Omit `--brain-slices/--orbit-slices` for automatic slice detection (verify with the plot).

### Re-train the brain segmenter (distillation)
```bash
# 1. auto-label with SAM2 (needs transformers + the facebook/sam2-hiera-large cache)
HF_HUB_OFFLINE=1 python build_brain_seg_dataset.py --dicom <DICOM_dir> \
    --out brain_seg_dataset --review brain_seg_review
# 2. open brain_seg_review/, delete the previews that look wrong
# 3. apply the curation
python curate_brain_dataset.py --data brain_seg_dataset --review brain_seg_review
# 4. (re-split into train/val by patient, then) train
python train_lwbna_brain.py --data brain_seg_dataset --epochs 120 --out runs/lwbna_brain
```

### Validate the automatic white matter against manual ellipse ROIs
The automatic reference (brain mask + histogram band) can read higher than a
reader's hand-placed white-matter ROI. To quantify this on your own data:
```bash
# 1. annotate: 20 cases (one per patient, random, seed 0), ONE slice per case,
#    one or more ellipses on cerebral white matter. Auto-saves; re-run to resume.
python annotate_wm_roi.py --dicom <STIR_DICOM_dir> --out wm_manual_roi/annotations.json --n 20
# 2. compare with the automatic value on the same slice (and the production
#    median-over-slices value), with Bland-Altman plot and per-case QC sheet
python compare_wm_manual_vs_auto.py --ann wm_manual_roi/annotations.json     --lwbna models/brain_lwbna_best.pt --out wm_manual_roi
```
Outputs: `comparison_per_case.csv`, `comparison_summary.csv` (ratio, % difference,
paired t / Wilcoxon, limits of agreement), `comparison_plot.png`, `qc_sheet.png`.
The automatic result is not shown during annotation so the reader is not anchored.
For the cases that disagree most, `inspect_wm_outliers.py --thr 20` draws the
brain-mask histogram with the detected peaks, the chosen WM band and the reader's
ROI voxels. Findings on the HANDAI data: `Knowledge/manual-vs-auto-white-matter.md`.

An alternative reference read on the same slice as the extraocular muscles
(`wm_gmm.wm_reference_hybrid`: histogram WM peak, replaced by the tied-GMM mode
when the two differ by >15%) agreed with the reader within +0.2 ± 4.1% (n=19).
The original band-mean method in `sir_analysis.py` is unchanged.
```bash
python validate_wm_hybrid.py --ann wm_manual_roi_orbit/annotations.json --exclude 182_
```

## Model card — `models/brain_lwbna_best.pt`
- Architecture: LWBNA-UNet (2.95M params), input 1×256×256, output 1 (binary cerebrum).
- Training: SAM2-labelled coronal STIR, geometric QC + manual curation, Dice+BCE.
- Validation Dice ≈ **0.978** (patient-level split, no leakage).
- White-matter reference matches SAM2 within ~1% on held-out patients.
- Input expects a windowed 8-bit STIR slice (see `lwbna_brain.window8`).

## Notes
- Input STIR must be **raw intensities** (SIR is a ratio of raw signals — do not
  min/max or z-score normalise the volume).
- STIR ≠ T2: models trained on T2 do not transfer (domain shift). See
  `Knowledge/stir-domain-shift-and-anatomy.md`.

## License
**Non-commercial / research use only.** See [`LICENSE`](LICENSE)
(PolyForm Noncommercial License 1.0.0).

The **LWBNA-UNet architecture is patented** (Sharma et al. 2022) and its
**commercial use requires permission from the authors** — this is the binding
constraint, so the model and code here are released for non-commercial research
only. (SAM2 itself is Apache-2.0; it imposes no such restriction.)

## Credits
- SIR method: Higashiyama et al., *Jpn J Ophthalmol* 2015.
- LWBNA-UNet: Sharma et al., *Sci Rep* 12:8508 (2022); ref impl
  [parmanandsharma/Lightweight_AI](https://github.com/parmanandsharma/Lightweight_AI).
- SAM2: Meta AI (`facebook/sam2-hiera-large`), Apache-2.0.
