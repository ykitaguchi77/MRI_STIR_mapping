"""
annotate_wm_roi.py — manual elliptical white-matter ROI annotation (Tkinter).

Purpose: validate the automatic white-matter reference (LWBNA brain mask +
histogram low-intensity peak) against the way a human places an ROI. A reader
picks ONE coronal slice per case and draws one or more ellipses on cerebral
white matter; the mean raw STIR signal inside the ellipse(s) is recorded.

The automatic result is deliberately NOT shown while annotating, so the reader
is not anchored to it. Compare afterwards with compare_wm_manual_vs_auto.py.

Usage
-----
  python annotate_wm_roi.py --dicom ../MRI_TOM/HANDAI_STIR_Coronal \
      --out wm_manual_roi/annotations.json --n 20 --seed 0

  # or an explicit case list (one series prefix per line, e.g. <id>_<YYYYMMDD>)
  python annotate_wm_roi.py --dicom <dir> --cases cases.txt --out ...

Controls
--------
  mouse drag (empty area)     draw a new ellipse (bounding box)
  mouse drag (inside ellipse) move it
  wheel over ellipse          scale (Ctrl = width only, Shift = height only)
  right click on ellipse      delete it
  Up / Down  (or PgUp/PgDn)   previous / next slice
  Left / Right                previous / next case  (auto-saves)
  Enter                       save and go to the next case
  z                           undo last ellipse on this slice
  c                           clear all ellipses on this case
  - / =                       darker / brighter display window
  s                           save        q / Esc   save and quit

Only ONE slice per case is kept: drawing on a different slice replaces the
previous slice's ellipses (after confirmation in the status bar).
"""

from __future__ import annotations

import argparse
import getpass
import json
import random
import re
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

import sir_analysis as S

try:  # draw at physical pixels on high-DPI Windows displays (no blurry upscaling)
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:  # noqa: BLE001 — not Windows / old Windows
    pass


# --------------------------------------------------------------------------- #
# Case selection
# --------------------------------------------------------------------------- #
def list_series(dicom_dir: str) -> list[str]:
    """Distinct series prefixes ('<patient>_<date>_STIR_Coronal') in the folder."""
    names = set()
    for f in Path(dicom_dir).glob("*.dcm"):
        m = re.match(r"(.+)_\d+\.dcm$", f.name)
        if m:
            names.add(m.group(1))
    return sorted(names)


def pick_cases(series: list[str], n: int, seed: int) -> list[str]:
    """Random sample of n series, at most one per patient (prefix before '_')."""
    by_patient: dict[str, list[str]] = {}
    for s in series:
        by_patient.setdefault(s.split("_")[0], []).append(s)
    rng = random.Random(seed)
    patients = sorted(by_patient)
    rng.shuffle(patients)
    chosen = [rng.choice(by_patient[p]) for p in patients[:n]]
    return sorted(chosen, key=lambda s: (int(s.split("_")[0]) if s.split("_")[0].isdigit() else 1e9, s))


# --------------------------------------------------------------------------- #
# Ellipse geometry / statistics
# --------------------------------------------------------------------------- #
def ellipse_mask(shape, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Boolean mask of a filled axis-aligned ellipse in image (row=y, col=x) coords."""
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    return ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0


def roi_stats(sl: np.ndarray, ellipses: list[dict]) -> dict:
    """Per-ellipse and pooled (voxel-weighted) mean/std/n of raw signal."""
    per = []
    pooled = np.zeros(sl.shape, dtype=bool)
    for e in ellipses:
        m = ellipse_mask(sl.shape, e["cx"], e["cy"], e["rx"], e["ry"])
        pooled |= m
        v = sl[m]
        per.append({"mean": float(v.mean()) if v.size else float("nan"),
                    "std": float(v.std()) if v.size else float("nan"),
                    "n": int(v.size)})
    v = sl[pooled]
    return {"per_ellipse": per,
            "mean": float(v.mean()) if v.size else float("nan"),
            "std": float(v.std()) if v.size else float("nan"),
            "n": int(v.size)}


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class Annotator:
    MIN_R = 2.0         # minimum ellipse radius (image px)

    def __init__(self, dicom_dir: str, cases: list[str], out: Path, zoom: int | None = None,
                 orbit_offset: int | None = None):
        self.orbit_offset = orbit_offset   # start N slices behind the eyeballs (muscle level)
        self.dicom_dir = dicom_dir
        self.cases = cases
        self.out = out
        self.user = getpass.getuser()
        self.ann: dict = {}
        if out.exists():
            self.ann = json.loads(out.read_text(encoding="utf-8")).get("annotations", {})

        self.idx = 0
        self.vol = None
        self.z = 0
        self.ellipses: list[dict] = []     # on the current slice
        self.win_pct = 99.5                # display window upper percentile
        self.drag = None                   # ("new", x0, y0, item) | ("move", i, x, y)
        self._photo = None

        self.root = tk.Tk()
        self.root.title("White-matter ROI annotation")
        # integer zoom that fits the screen (leave room for the status lines)
        self.ZOOM = zoom or max(1, min(4, (self.root.winfo_screenheight() - 180) // 256))
        self.canvas = tk.Canvas(self.root, width=256 * self.ZOOM, height=256 * self.ZOOM,
                                bg="black", cursor="crosshair")
        self.canvas.pack()
        self.status = tk.Label(self.root, anchor="w", justify="left", font=("Consolas", 10))
        self.status.pack(fill="x")
        self.help = tk.Label(self.root, anchor="w", justify="left", fg="gray40",
                             font=("Consolas", 9), wraplength=256 * self.ZOOM - 10,
                             text="drag: ellipse | drag inside: move | wheel: scale (Ctrl=W, Shift=H) | "
                                  "RClick: delete | Up/Down: slice | Left/Right: case | "
                                  "Enter: save+next | z: undo | c: clear | -/=: window | s: save | q: quit")
        self.help.pack(fill="x")

        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<ButtonPress-3>", self.on_right)
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Control-MouseWheel>", lambda e: self.on_wheel(e, "w"))
        c.bind("<Shift-MouseWheel>", lambda e: self.on_wheel(e, "h"))
        r = self.root
        r.bind("<Up>", lambda e: self.set_slice(self.z - 1))
        r.bind("<Down>", lambda e: self.set_slice(self.z + 1))
        r.bind("<Prior>", lambda e: self.set_slice(self.z - 1))
        r.bind("<Next>", lambda e: self.set_slice(self.z + 1))
        r.bind("<Left>", lambda e: self.go_case(self.idx - 1))
        r.bind("<Right>", lambda e: self.go_case(self.idx + 1))
        r.bind("<Return>", lambda e: self.go_case(self.idx + 1))
        r.bind("z", lambda e: self.undo())
        r.bind("c", lambda e: self.clear())
        r.bind("s", lambda e: self.save(verbose=True))
        r.bind("-", lambda e: self.set_window(self.win_pct - 0.5))
        r.bind("=", lambda e: self.set_window(self.win_pct + 0.5))
        r.bind("q", lambda e: self.quit())
        r.bind("<Escape>", lambda e: self.quit())
        r.protocol("WM_DELETE_WINDOW", self.quit)

        # start on the first case without an annotation
        todo = [i for i, s in enumerate(cases) if s not in self.ann]
        self.load_case(todo[0] if todo else 0)

    # ----- case / slice handling ------------------------------------------- #
    @property
    def series(self) -> str:
        return self.cases[self.idx]

    def load_case(self, i: int):
        self.idx = i
        self.vol, _ = S.load_dicom_series(self.dicom_dir, self.series)
        H, W = self.vol.shape[:2]
        self.canvas.config(width=W * self.ZOOM, height=H * self.ZOOM)
        self.vmax = float(np.percentile(self.vol, self.win_pct))
        rec = self.ann.get(self.series)
        if rec:
            self.z = int(rec["slice"])
            self.ellipses = [dict(e) for e in rec["ellipses"]]
        else:
            # default to the slice the auto pipeline would use (middle of the
            # brain block) so the reader starts near the centrum semiovale
            try:
                import orbital_sir_map as O
                if self.orbit_offset is not None:
                    # muscle level: N slices from the eyeball slice toward the brain
                    _, gz = O.detect_globes(self.vol)
                    d = -1 if gz > self.vol.shape[2] / 2 else 1
                    self.z = int(np.clip(gz + d * self.orbit_offset, 0, self.vol.shape[2] - 1))
                else:
                    brain, _ = O.auto_orbit_and_brain_slices(self.vol)
                    self.z = int(np.median(brain))
            except Exception:  # noqa: BLE001
                self.z = self.vol.shape[2] // 2
            self.ellipses = []
        self.redraw()

    def go_case(self, i: int):
        self.save()
        if 0 <= i < len(self.cases):
            self.load_case(i)
        else:
            self.set_status("(no more cases) " + self.status_text())

    def set_slice(self, z: int):
        z = int(np.clip(z, 0, self.vol.shape[2] - 1))
        if z == self.z:
            return
        self.z = z
        # one slice per case: moving away from an annotated slice drops its ROIs
        if self.ellipses:
            self.ellipses = []
        self.redraw()

    def set_window(self, pct: float):
        self.win_pct = float(np.clip(pct, 90.0, 100.0))
        self.vmax = float(np.percentile(self.vol, self.win_pct))
        self.redraw()

    # ----- persistence ------------------------------------------------------ #
    def save(self, verbose: bool = False):
        if self.vol is None:
            return
        if self.ellipses:
            sl = self.vol[:, :, self.z]
            self.ann[self.series] = {
                "slice": int(self.z),
                "ellipses": [{k: round(float(e[k]), 2) for k in ("cx", "cy", "rx", "ry")}
                             for e in self.ellipses],
                "stats": roi_stats(sl, self.ellipses),
                "annotator": self.user,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            self.ann.pop(self.series, None)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dicom": str(self.dicom_dir), "cases": self.cases,
                   "annotations": self.ann}
        self.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        if verbose:
            self.set_status(f"saved -> {self.out}  ({len(self.ann)}/{len(self.cases)} done)")

    def quit(self):
        self.save()
        self.root.destroy()

    # ----- drawing ---------------------------------------------------------- #
    def img_to_canvas(self, x: float, y: float):
        return (x + 0.5) * self.ZOOM, (y + 0.5) * self.ZOOM

    def canvas_to_img(self, X: float, Y: float):
        return X / self.ZOOM - 0.5, Y / self.ZOOM - 0.5

    def redraw(self):
        sl = self.vol[:, :, self.z]
        img8 = (np.clip(sl, 0, self.vmax) / max(self.vmax, 1e-6) * 255).astype(np.uint8)
        im = Image.fromarray(img8).resize((img8.shape[1] * self.ZOOM, img8.shape[0] * self.ZOOM),
                                          Image.NEAREST)
        self._photo = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        for i, e in enumerate(self.ellipses):
            self.draw_ellipse(e, i)
        self.set_status(self.status_text())

    def draw_ellipse(self, e: dict, i: int):
        x0, y0 = self.img_to_canvas(e["cx"] - e["rx"], e["cy"] - e["ry"])
        x1, y1 = self.img_to_canvas(e["cx"] + e["rx"], e["cy"] + e["ry"])
        self.canvas.create_oval(x0, y0, x1, y1, outline="#00ff66", width=2, tags=f"ell{i}")
        cx, cy = self.img_to_canvas(e["cx"], e["cy"])
        self.canvas.create_text(cx, cy, text=str(i + 1), fill="#00ff66", font=("Arial", 11, "bold"))

    def status_text(self) -> str:
        done = "done" if self.series in self.ann else "todo"
        s = (f"case {self.idx + 1}/{len(self.cases)}  {self.series}  [{done}]   "
             f"slice {self.z}/{self.vol.shape[2] - 1}   window {self.win_pct:.1f}%")
        if self.ellipses:
            st = roi_stats(self.vol[:, :, self.z], self.ellipses)
            parts = [f"#{i + 1}: {p['mean']:.0f}±{p['std']:.0f} (n={p['n']})"
                     for i, p in enumerate(st["per_ellipse"])]
            s += f"\nROI mean {st['mean']:.1f} ± {st['std']:.1f}  n={st['n']}   |  " + "  ".join(parts)
        else:
            s += "\n(no ROI on this slice — drag to draw an ellipse on white matter)"
        return s

    def set_status(self, text: str):
        self.status.config(text=text)

    # ----- mouse ------------------------------------------------------------ #
    def hit(self, X: float, Y: float) -> int | None:
        x, y = self.canvas_to_img(X, Y)
        for i in range(len(self.ellipses) - 1, -1, -1):
            e = self.ellipses[i]
            if ((x - e["cx"]) / e["rx"]) ** 2 + ((y - e["cy"]) / e["ry"]) ** 2 <= 1.0:
                return i
        return None

    def on_press(self, ev):
        i = self.hit(ev.x, ev.y)
        if i is not None:
            self.drag = ("move", i, ev.x, ev.y)
        else:
            item = self.canvas.create_oval(ev.x, ev.y, ev.x, ev.y, outline="yellow", width=2)
            self.drag = ("new", ev.x, ev.y, item)

    def on_drag(self, ev):
        if not self.drag:
            return
        if self.drag[0] == "new":
            _, x0, y0, item = self.drag
            self.canvas.coords(item, x0, y0, ev.x, ev.y)
        else:
            _, i, px, py = self.drag
            e = self.ellipses[i]
            e["cx"] += (ev.x - px) / self.ZOOM
            e["cy"] += (ev.y - py) / self.ZOOM
            self.drag = ("move", i, ev.x, ev.y)
            self.redraw()

    def on_release(self, ev):
        if not self.drag:
            return
        if self.drag[0] == "new":
            _, x0, y0, item = self.drag
            self.canvas.delete(item)
            ax, ay = self.canvas_to_img(min(x0, ev.x), min(y0, ev.y))
            bx, by = self.canvas_to_img(max(x0, ev.x), max(y0, ev.y))
            rx, ry = (bx - ax) / 2, (by - ay) / 2
            if rx >= self.MIN_R and ry >= self.MIN_R:
                self.ellipses.append({"cx": (ax + bx) / 2, "cy": (ay + by) / 2, "rx": rx, "ry": ry})
        self.drag = None
        self.redraw()

    def on_right(self, ev):
        i = self.hit(ev.x, ev.y)
        if i is not None:
            del self.ellipses[i]
            self.redraw()

    def on_wheel(self, ev, axis: str = "both"):
        i = self.hit(ev.x, ev.y)
        if i is None:
            return
        f = 1.08 if ev.delta > 0 else 1 / 1.08
        e = self.ellipses[i]
        if axis in ("both", "w"):
            e["rx"] = max(self.MIN_R, e["rx"] * f)
        if axis in ("both", "h"):
            e["ry"] = max(self.MIN_R, e["ry"] * f)
        self.redraw()

    # ----- edits ------------------------------------------------------------ #
    def undo(self):
        if self.ellipses:
            self.ellipses.pop()
            self.redraw()

    def clear(self):
        self.ellipses = []
        self.redraw()

    def run(self):
        self.root.mainloop()


def main():
    p = argparse.ArgumentParser(description="Manual elliptical white-matter ROI annotation")
    p.add_argument("--dicom", required=True, help="folder of coronal STIR DICOM slices")
    p.add_argument("--out", default="wm_manual_roi/annotations.json")
    p.add_argument("--n", type=int, default=20, help="number of cases (one per patient)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cases", help="text file: one series prefix per line (overrides --n/--seed)")
    p.add_argument("--zoom", type=int, help="display zoom factor (default: fit the screen)")
    p.add_argument("--orbit-offset", type=int,
                   help="start each case N slices behind the eyeball slice (extraocular-muscle "
                        "level, e.g. 3) to annotate WM on the same slice as the muscles")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        cases = json.loads(out.read_text(encoding="utf-8"))["cases"]
        print(f"Resuming {out} ({len(cases)} cases)")
    elif args.cases:
        cases = [ln.strip() for ln in Path(args.cases).read_text().splitlines() if ln.strip()]
    else:
        series = list_series(args.dicom)
        cases = pick_cases(series, args.n, args.seed)
        print(f"{len(series)} series found; sampled {len(cases)} (seed {args.seed}):")
        print("  " + "\n  ".join(cases))
    Annotator(args.dicom, cases, out, zoom=args.zoom, orbit_offset=args.orbit_offset).run()


if __name__ == "__main__":
    main()
