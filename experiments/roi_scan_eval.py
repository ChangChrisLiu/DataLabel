"""Evaluate the scanner ROI strategies over the 66 real cached scan frames.

Read-only on the cache: it opens ``<cache>/scan/D<nn>/s001.png`` and writes
nothing except the overlays it is asked for. The numbers it prints are what
calibrated :data:`tda.core.cache_roi_detect.ROI_MIN_RECTANGULARITY` and the
other plausibility bounds.

::

    python experiments/roi_scan_eval.py                     # the table
    python experiments/roi_scan_eval.py --draw 64 55        # overlays too
    python experiments/roi_scan_eval.py --draw changed      # ... for every change

Each overlay is a downscaled copy of the frame with the dark-object box in red,
the scan-bed box in green and the chosen box in white.

The last column of the table, ``bed?``, is what the bed strategy *would* have
proposed for the desktops that fall back to the central box. It is printed for
information only: nine machines cover the tape square entirely, their chassis
fills the whole frame, and "no crop" - which the ROI cannot express - is the
right answer for them. Deciding that is not this script's job.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import cv2
import numpy as np

from tda.core.cache import _central_box, suggest_roi
from tda.core.cache_roi_detect import (
    BED_MIN_RECTANGULARITY,
    box_plausibility,
    scan_bed_box,
    scan_bed_candidates,
    scan_chassis_box,
    scan_chassis_candidates,
)

CACHE = "D:/DataSet/cache/scan"
OUT = "D:/DataSet/experiments_out/roi_scan"
DESKTOPS = range(1, 67)


def read(path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def frac(box, width: int, height: int) -> float:
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0) / float(width * height)


def draw(bgr, boxes: dict[str, tuple], dest: str, side: int = 900) -> None:
    colours = {"dark": (0, 0, 255), "bed": (0, 200, 0), "chosen": (255, 255, 255)}
    canvas = bgr.copy()
    for name, box in boxes.items():
        if box is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colours[name],
                      10 if name == "chosen" else 6)
    height, width = canvas.shape[:2]
    scale = side / float(max(height, width))
    small = cv2.resize(canvas, (int(width * scale), int(height * scale)),
                       interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cv2.imencode(".png", small)[1].tofile(dest)


def cell(value: Optional[float]) -> str:
    return "   -  " if value is None else f"{value:6.3f}"


def measure(bgr) -> dict:
    """Everything the table shows about one frame."""
    height, width = bgr.shape[:2]
    dark_all = scan_chassis_candidates(bgr)
    bed_all = scan_bed_candidates(bgr)
    dark = scan_chassis_box(bgr)
    bed = scan_bed_box(bgr)
    chosen = suggest_roi(bgr, "scan")
    return {
        "w": width, "h": height,
        "dark": dark, "bed": bed,
        "dark_raw": dark_all[0] if dark_all else None,
        "bed_raw": bed_all[0] if bed_all else None,
        "chosen": chosen,
        "pick": ("dark" if dark is not None and chosen == dark[0]
                 else "bed" if bed is not None and chosen == bed[0] else "central"),
    }


def old_box(bgr, m: dict) -> tuple:
    """What the *previous* build returned: the largest dark object, or central."""
    raw = m["dark_raw"]
    return _central_box(m["w"], m["h"]) if raw is None else raw[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--draw", nargs="*", default=[],
                    help="desktop numbers, or 'changed' for every changed one")
    args = ap.parse_args(argv)

    wanted = {int(d) for d in args.draw if str(d).isdigit()}
    draw_changed = "changed" in [str(d) for d in args.draw]

    frames, changed, rows = {}, [], []
    print(f"{'D':>4} {'old%':>6} {'new%':>6} {'dark%':>6} {'bed%':>6} "
          f"{'darkR':>6} {'bedR':>6}  {'pick':<8} bed?")
    for desktop in DESKTOPS:
        bgr = read(f"{args.cache}/D{desktop:02d}/s001.png")
        if bgr is None:
            print(f"D{desktop:02d}: no cached s001.png")
            continue
        m = measure(bgr)
        frames[desktop] = (bgr, m)
        old, new = old_box(bgr, m), m["chosen"]
        if tuple(old) != tuple(new):
            changed.append(desktop)
        rows.append((desktop, m["pick"]))
        # what the bed strategy would have said where nothing convinced
        would = ""
        if m["pick"] == "central" and m["bed_raw"] is not None:
            box, rect = m["bed_raw"]
            would = (f"largest not-bed region {frac(box, m['w'], m['h']):.3f} "
                     f"of the frame, fill {rect:.2f} "
                     f"({'plausible' if box_plausibility(box, rect, m['w'], m['h'], BED_MIN_RECTANGULARITY) else 'vetoed'})")
        print(f"D{desktop:02d} {frac(old, m['w'], m['h']):6.3f} "
              f"{frac(new, m['w'], m['h']):6.3f} "
              f"{cell(None if m['dark'] is None else frac(m['dark'][0], m['w'], m['h']))} "
              f"{cell(None if m['bed'] is None else frac(m['bed'][0], m['w'], m['h']))} "
              f"{cell(None if m['dark'] is None else m['dark'][1])} "
              f"{cell(None if m['bed'] is None else m['bed'][1])}  {m['pick']:<8} {would}")

    for desktop, (bgr, m) in frames.items():
        if desktop in wanted or (draw_changed and desktop in changed):
            draw(bgr, {"dark": None if m["dark"] is None else m["dark"][0],
                       "bed": None if m["bed"] is None else m["bed"][0],
                       "chosen": m["chosen"]},
                 f"{args.out}/roi_D{desktop:02d}.png")

    picks: dict[str, int] = {}
    for _d, pick in rows:
        picks[pick] = picks.get(pick, 0) + 1
    print(f"\ndesktops: {len(rows)}   picks: {dict(sorted(picks.items()))}")
    print(f"changed ({len(changed)}): {changed}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
