"""
curate_brain_dataset.py — apply a manual review to the brain-seg dataset.

Workflow:
  1. build_brain_seg_dataset.py ... --review brain_seg_review
       writes one overlay preview per kept slice as <split>__<stem>.png
  2. Open the review folder, DELETE the previews that look wrong.
  3. python curate_brain_dataset.py --data brain_seg_dataset --review brain_seg_review
       removes every image/mask pair whose preview was deleted, so only the
       previews you kept remain in the training set.

Use --dry-run first to see what would be removed.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    review = Path(args.review)

    # stems the user KEPT (preview still present): "<split>__<stem>"
    kept = {Path(f).stem for f in glob.glob(str(review / "*.png"))}

    removed = keptn = 0
    for split in ("train", "val"):
        for img in glob.glob(str(data / "images" / split / "*.png")):
            stem = Path(img).stem
            if f"{split}__{stem}" in kept:
                keptn += 1
                continue
            mask = data / "masks" / split / f"{stem}.png"
            removed += 1
            print(f"[remove] {split}/{stem}")
            if not args.dry_run:
                Path(img).unlink(missing_ok=True)
                mask.unlink(missing_ok=True)

    print(f"\n{'DRY-RUN ' if args.dry_run else ''}kept {keptn}, removed {removed} "
          f"({len(kept)} previews in review folder)")


if __name__ == "__main__":
    main()
