"""
train_lwbna_brain.py — train LWBNA-UNet to segment the cerebrum on STIR,
distilled from the SAM2-labelled mask images (build_brain_seg_dataset.py).

Loss = Dice + BCE. Saves the best model by validation Dice.
Run with the MRI_TOM venv.

  python train_lwbna_brain.py --data brain_seg_dataset --epochs 100 \
      --out runs/lwbna_brain
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lwbna_unet import LWBNAUNet


class BrainSegDS(Dataset):
    """Loads (STIR image PNG, brain mask PNG) pairs from <root>/images/<split>
    and <root>/masks/<split>. Images are scaled to [0, 1]; masks to {0, 1}.
    With augment=True, applies a random horizontal flip and brightness jitter."""

    def __init__(self, root: str, split: str, imgsz: int = 256, augment: bool = False):
        self.imgs = sorted(glob.glob(str(Path(root) / "images" / split / "*.png")))
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        mp = str(self.root / "masks" / self.split / Path(ip).name)
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img.shape[0] != self.imgsz:
            img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        msk = (msk > 127).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:                 # horizontal flip
                img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
            if np.random.rand() < 0.5:                 # brightness/contrast jitter
                img = np.clip(img * np.random.uniform(0.85, 1.15)
                              + np.random.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(img)[None], torch.from_numpy(msk)[None]


def dice_loss(logits, target, eps=1.0):
    """Soft Dice loss (1 - Dice) on the sigmoid probabilities. Complements BCE:
    BCE gets pixels right, Dice handles the foreground/background imbalance."""
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return (1 - num / den).mean()


@torch.no_grad()
def dice_score(logits, target, eps=1.0):
    """Hard Dice (overlap) of the thresholded prediction vs the mask — the
    validation metric, in [0, 1]; higher is better."""
    p = (torch.sigmoid(logits) > 0.5).float()
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return (num / den).mean().item()


def main():
    ap = argparse.ArgumentParser(description="Train LWBNA-UNet to segment the cerebrum")
    ap.add_argument("--data", required=True, help="dataset dir (images/ + masks/)")
    ap.add_argument("--out", default="runs/lwbna_brain", help="where to save best.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imgsz", type=int, default=256, help="must be a multiple of 16")
    ap.add_argument("--patience", type=int, default=25,
                    help="stop if val Dice does not improve for this many epochs")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tr = BrainSegDS(args.data, "train", args.imgsz, augment=True)
    va = BrainSegDS(args.data, "val", args.imgsz, augment=False)
    print(f"train {len(tr)} | val {len(va)} | device {dev}")
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4)

    model = LWBNAUNet(in_channels=1, num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler(dev)
    bce = nn.BCEWithLogitsLoss()

    best, bad = 0.0, 0          # best val Dice so far; epochs since last improvement
    for ep in range(1, args.epochs + 1):
        # --- train one epoch (mixed precision) ---
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.amp.autocast(dev):
                out_logits = model(x)
                loss = bce(out_logits, y) + dice_loss(out_logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        # --- validate ---
        model.eval()
        ds = []
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                with torch.amp.autocast(dev):
                    ds.append(dice_score(model(x), y))
        vdice = float(np.mean(ds)) if ds else 0.0
        print(f"epoch {ep:3d}/{args.epochs}  val_dice={vdice:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        # --- keep the best model; stop early if it plateaus ---
        if vdice > best:
            best, bad = vdice, 0
            torch.save({"model": model.state_dict(), "val_dice": best,
                        "arch": "LWBNAUNet", "in_channels": 1, "num_classes": 1},
                       out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep} (best val_dice={best:.4f})")
                break

    print(f"Done. best val_dice={best:.4f}  weights: {out/'best.pt'}")


if __name__ == "__main__":
    main()
