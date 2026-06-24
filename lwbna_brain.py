"""
lwbna_brain.py — distilled LWBNA-UNet brain segmenter (inference).

Same `brain_mask(stir, slice_idx)` interface as SAM2BrainSegmenter, so it drops
into sir_analysis.auto_white_matter_reference(..., brain_mask_fn=...) and
orbital_sir_map.py. SAM2 labels the data; this distilled model runs production.
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
from scipy import ndimage

import sir_analysis as S
from lwbna_unet import LWBNAUNet


def window8(sl: np.ndarray) -> np.ndarray:
    """8-bit windowing of a STIR slice (clip at the 99.5th percentile, scale to
    0-255). MUST match build_brain_seg_dataset.window8 so the model sees the same
    contrast at inference as it did during training."""
    v = np.clip(sl, 0, np.percentile(sl, 99.5))
    return (v / max(v.max(), 1e-6) * 255).astype(np.uint8)


class LWBNABrainSegmenter:
    """Loads a trained LWBNA-UNet and segments the cerebrum on STIR slices.
    Drop-in replacement for SAM2BrainSegmenter (same `brain_mask` signature)."""

    def __init__(self, weights: str, device: str | None = None,
                 imgsz: int = 256, thr: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location=self.device, weights_only=False)
        self.model = LWBNAUNet(in_channels=ckpt.get("in_channels", 1),
                               num_classes=ckpt.get("num_classes", 1)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.imgsz = imgsz
        self.thr = thr

    def brain_mask(self, stir: np.ndarray, slice_idx: int):
        """Segment the cerebrum on one slice. Returns a 3D boolean mask (True on
        slice_idx), or None if nothing is found. Pipeline: window to 8-bit ->
        resize to the model size -> sigmoid -> threshold -> resize back -> keep
        the largest connected component and fill holes."""
        sl = stir[:, :, slice_idx]
        H, W = sl.shape
        x = window8(sl).astype(np.float32) / 255.0
        if (H, W) != (self.imgsz, self.imgsz):
            x = cv2.resize(x, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(x)[None, None].to(self.device)   # (1,1,H,W)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(t))[0, 0].cpu().numpy()
        m = prob > self.thr
        if (H, W) != (self.imgsz, self.imgsz):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if m.sum() == 0:
            return None
        lbl, n = ndimage.label(m)              # keep only the largest blob
        if n > 1:
            sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
            m = lbl == int(np.argmax(sizes)) + 1
        m = ndimage.binary_fill_holes(m)
        out = np.zeros(stir.shape, dtype=bool)
        out[:, :, slice_idx] = m
        return out


def white_matter_from_lwbna(stir, weights, brain_slices, n_bins: int = 128):
    """Convenience: segment the brain with LWBNA-UNet and return the white-matter
    reference (median over `brain_slices`). See sir_analysis.auto_white_matter_reference."""
    seg = LWBNABrainSegmenter(weights)
    return S.auto_white_matter_reference(stir, brain_slices, n_bins=n_bins,
                                         brain_mask_fn=seg.brain_mask)
