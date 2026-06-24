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
| `yolo_brain.py`, `build_brain_dataset.py` | optional YOLO11-seg distillation variant |
| `sir_analysis.py` | white-matter extraction, SIR, orbital SIR map, DICOM I/O |
| `orbital_sir_map.py` | end-to-end CLI (`--sam2` / `--yolo` / `--lwbna`) |
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

## Credits
- SIR method: Higashiyama et al., *Jpn J Ophthalmol* 2015.
- LWBNA-UNet: Sharma et al., *Sci Rep* 12:8508 (2022); ref impl
  [parmanandsharma/Lightweight_AI](https://github.com/parmanandsharma/Lightweight_AI).
- SAM2: Meta AI (`facebook/sam2-hiera-large`).
