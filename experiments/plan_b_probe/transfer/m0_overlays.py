"""M0 -- sanity overlays: do the stored LS polygons sit on the right pixels?

Two things have to be proven before a single number is computed, because both
would silently poison every later measurement:

1. **Coordinate frame.** ``tda.core.ls_import`` rasterises the percent geometry
   into :data:`tda.core.ls_export.NATIVE_HW`, and refuses the ``_Align_``
   uploads, so an OAK mask should land on the 4032x3040 still from
   ``frame.path``. If some geometry had really been drawn on the aligned
   1280x800 stream, the polygons would sit on the wrong scene content here.
2. **Step alignment.** The Label Studio step numbering was reported to be off by
   one in places; if it is, a polygon drawn round a part that has already been
   removed will outline bare board instead.

Writes one two-panel JPEG per (desktop, view) into
``experiments_out/plan_b_probe/transfer/m0/``; they are meant to be *looked at*,
not parsed. ``m4_diffmap.py`` re-tests the step alignment numerically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.plan_b_probe.transfer.common import (  # noqa: E402
    DESKTOPS, OUT, VIEWS, banner, connect, draw_shapes, frame_path,
    hstack_pad, index_shapes, load_geometry, read_rgb, save_jpg,
)

PANEL = 900


def pick_steps(steps: list[int]) -> list[int]:
    """Two steps far enough apart to show the scene in two different states."""
    if len(steps) < 2:
        return steps
    return [steps[len(steps) // 4], steps[(3 * len(steps)) // 4]]


def panel(con, desktop: int, view: str, step: int, by_frame) -> np.ndarray | None:
    path = frame_path(con, desktop, view, step)
    if path is None:
        return None
    img = read_rgb(path)
    if img is None:
        return None
    shapes = list(by_frame.get((desktop, view, step), {}).values())
    # Thickness and text scale in *native* pixels, so both survive the resize.
    scale = max(img.shape[:2]) / float(PANEL)
    drawn = draw_shapes(img, shapes, thickness=max(2, int(round(2 * scale))),
                        font_scale=max(0.6, 0.55 * scale))
    h, w = drawn.shape[:2]
    factor = PANEL / float(max(h, w))
    small = cv2.resize(drawn, (int(round(w * factor)), int(round(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return banner(small, f"D{desktop} {view} step {step}  ({len(shapes)} LS shapes)")


def main() -> int:
    shapes = load_geometry()
    by_frame = index_shapes(shapes)
    con = connect()
    outdir = OUT / "m0"
    made = []
    for desktop in DESKTOPS:
        for view in VIEWS:
            steps = sorted({s.step for s in shapes
                            if s.desktop == desktop and s.view == view})
            if not steps:
                print(f"D{desktop} {view}: no LS shapes, skipped")
                continue
            panels = [p for p in (panel(con, desktop, view, st, by_frame)
                                  for st in pick_steps(steps)) if p is not None]
            if not panels:
                continue
            path = save_jpg(outdir / f"m0_d{desktop}_{view}.jpg",
                            hstack_pad(panels), max_side=1900, quality=80)
            made.append(path)
            print(f"{path}  {path.stat().st_size // 1024} KiB  steps={pick_steps(steps)}")
    con.close()
    print(f"\n{len(made)} overlays under {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
