"""
sam2_brain.py — SAM2-based brain (cerebrum) segmentation on coronal STIR slices.

Morphological skull-stripping fails on these images because the brain stays
connected to the face/sinuses through the skull base. SAM2 (Segment Anything 2)
segments the cerebrum cleanly from a single interior point prompt, excluding the
paranasal sinuses and orbit. The resulting brain mask is then used as the ROI for
the white-matter histogram reference (see sir_analysis.extract_white_matter_reference).

Uses the cached HuggingFace model `facebook/sam2-hiera-large` via transformers
(set HF_HUB_OFFLINE=1 to force the local cache). Run with the MRI_TOM venv.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class SAM2BrainSegmenter:
    """Lazy-loaded SAM2 wrapper that returns a clean cerebrum mask per slice."""

    def __init__(self, model_id: str = "facebook/sam2-hiera-large", device: str | None = None):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()
        self._torch = torch

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _to_rgb(sl: np.ndarray):
        from PIL import Image
        v = np.clip(sl, 0, np.percentile(sl, 99.5))
        v = (v / max(v.max(), 1e-6) * 255).astype(np.uint8)
        return Image.fromarray(np.stack([v] * 3, axis=-1))

    @staticmethod
    def _interior_point(sl: np.ndarray) -> tuple[float, float]:
        """A point reliably inside the cerebrum: centroid of strongly-eroded
        upper-head tissue (thin skull-base bridges erode away first)."""
        pos = sl[sl > 0]
        m = (sl > np.percentile(pos, 40)) & (sl < np.percentile(pos, 99))
        core = ndimage.binary_erosion(m, iterations=6)
        lbl, n = ndimage.label(core)
        if n == 0:
            ys, xs = np.where(m)
            return float(xs.mean()), float(ys.mean())
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        H = sl.shape[0]
        for idx in np.argsort(sizes)[::-1]:
            comp = lbl == idx + 1
            if np.where(comp)[0].mean() < 0.55 * H:  # upper head -> cerebrum
                ys, xs = np.where(comp)
                return float(xs.mean()), float(ys.mean())
        ys, xs = np.where(lbl == int(np.argmax(sizes)) + 1)
        return float(xs.mean()), float(ys.mean())

    # -- main ----------------------------------------------------------------
    def brain_mask(self, stir: np.ndarray, slice_idx: int,
                   max_area_frac: float = 0.45, min_area_frac: float = 0.02,
                   strict: bool = False):
        """
        Return a 3D boolean mask (True on `slice_idx`) of the cerebrum.

        SAM2 is prompted with one interior point and returns 3 candidate masks;
        we keep the highest-scoring one that is brain-sized (area fraction in
        [min_area_frac, max_area_frac]) and centred in the upper head — this
        rejects the "whole head" and "tiny blob" candidates.

        strict=True returns None when no candidate passes those criteria (used by
        the dataset builder to skip anterior/orbit slices that have no cerebrum).
        """
        sl = stir[:, :, slice_idx]
        img = self._to_rgb(sl)
        px, py = self._interior_point(sl)
        inputs = self.processor(images=img,
                                input_points=[[[[px, py]]]],
                                input_labels=[[[1]]],
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inputs)
        masks = self.processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy() > 0.5
        scores = out.iou_scores.cpu().numpy().reshape(-1)

        H = sl.shape[0]
        total = sl.size
        best, best_score = None, -1.0
        for k in range(masks.shape[0]):
            area = masks[k].sum()
            frac = area / total
            if not (min_area_frac <= frac <= max_area_frac):
                continue
            cy = np.where(masks[k])[0].mean() if area else H
            if cy >= 0.60 * H:           # brain should sit in the upper head
                continue
            if scores[k] > best_score:
                best, best_score = masks[k], scores[k]
        if best is None:
            if strict:                   # no valid cerebrum on this slice -> skip
                return None
            cand = [(scores[k], k) for k in range(masks.shape[0])
                    if masks[k].sum() / total <= max_area_frac]
            best = masks[max(cand)[1]] if cand else masks[int(np.argmax(scores))]

        # tidy: largest component, fill holes, slight erosion to avoid the CSF rim
        lbl, n = ndimage.label(best)
        if n > 1:
            sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
            best = lbl == int(np.argmax(sizes)) + 1
        best = ndimage.binary_fill_holes(best)
        best = ndimage.binary_erosion(best, iterations=1)

        mask = np.zeros(stir.shape, dtype=bool)
        mask[:, :, slice_idx] = best
        return mask
