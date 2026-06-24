"""
yolo_brain.py — distilled brain segmenter (YOLO11n-seg) + WM-intensity pipeline.

After distilling SAM2 into YOLO11n-seg (see build_brain_dataset.py + training),
this provides the fast inference end of the pipeline:

    STIR slice --(YOLO11n-seg)--> brain mask --> contour --> histogram WM intensity

`YOLOBrainSegmenter.brain_mask` has the SAME signature as
`SAM2BrainSegmenter.brain_mask`, so it drops into
`sir_analysis.auto_white_matter_reference(..., brain_mask_fn=...)` and
`orbital_sir_map.py` interchangeably — SAM2 for labelling, YOLO for production.

Run with the MRI_TOM venv.
"""

from __future__ import annotations

import numpy as np
import cv2

import sir_analysis as S


def window_rgb(sl: np.ndarray) -> np.ndarray:
    """Windowed 8-bit RGB (must match build_brain_dataset.window_png)."""
    v = np.clip(sl, 0, np.percentile(sl, 99.5))
    v = (v / max(v.max(), 1e-6) * 255).astype(np.uint8)
    return np.stack([v] * 3, axis=-1)


class YOLOBrainSegmenter:
    """Lazy-loaded YOLO11-seg wrapper returning a cerebrum mask per slice."""

    def __init__(self, weights: str, device: str | None = None, conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def brain_mask(self, stir: np.ndarray, slice_idx: int):
        """3D boolean mask (True on slice_idx) of the highest-confidence brain.
        Returns None if YOLO finds no brain on this slice."""
        sl = stir[:, :, slice_idx]
        H, W = sl.shape
        res = self.model.predict(window_rgb(sl), conf=self.conf,
                                 device=self.device, verbose=False)[0]
        if res.masks is None or len(res.masks) == 0:
            return None
        confs = res.boxes.conf.cpu().numpy()
        md = res.masks.data.cpu().numpy()          # (n, h, w) at model resolution
        best = md[int(np.argmax(confs))]
        best = cv2.resize(best.astype(np.float32), (W, H),
                          interpolation=cv2.INTER_NEAREST) > 0.5
        mask = np.zeros(stir.shape, dtype=bool)
        mask[:, :, slice_idx] = best
        return mask

    def contour(self, stir: np.ndarray, slice_idx: int):
        """Largest brain contour as an (N,2) int array, or None."""
        m = self.brain_mask(stir, slice_idx)
        if m is None:
            return None
        cnts, _ = cv2.findContours(m[:, :, slice_idx].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea).reshape(-1, 2)


def white_matter_from_yolo(stir: np.ndarray, weights: str, brain_slices,
                           n_bins: int = 128):
    """Full distilled pipeline: YOLO brain mask -> histogram white-matter ref."""
    seg = YOLOBrainSegmenter(weights)
    return S.auto_white_matter_reference(stir, brain_slices, n_bins=n_bins,
                                         brain_mask_fn=seg.brain_mask)
