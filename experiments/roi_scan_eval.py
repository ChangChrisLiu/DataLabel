"""Evaluate the scanner ROI strategies over the 66 real cached scan frames.

Read-only: it opens ``<cache>/scan/D<nn>/s001.png`` and writes nothing except
the overlays it is asked for. The numbers it prints are what decided the
plausibility bounds in :mod:`tda.core.cache`.

::

    python experiments/roi_scan_eval.py                     # the table
    python experiments/roi_scan_eval.py --draw 64 13 41     # overlays too

Each overlay is a downscaled copy of the frame with the dark-object box in red,
the scan-bed box in green and the chosen box in white.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import cv2
import numpy as np

from tda.core.cache import (
    _box_plausibility,
    _central_box,
    _scan_bed_box,
    _scan_chassis_box,
    suggest_roi,
)

CACHE = "D:/DataSet/cache/scan"
OUT = "D:/DataSet/.cache/tmp/roi_probe"
DESKTOPS = range(1, 67)


def read(path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def frac(box, width: int, height: int) -> float:
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0) / float(width * height)


def old_suggestion(bgr) -> tuple[int, int, int, int]:
    """What ``suggest_roi`` returned before the second strategy existed."""
    height, width = bgr.shape[:2]
    found = _scan_chassis_box(bgr)
    return _central_box(width, height) if found is None else found[0]


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--draw", nargs="*", type=int, default=[])
    args = ap.parse_args(argv)

    wanted = set(args.draw)
    changed, rows = [], []
    print(f"{'D':>4} {'old%':>6} {'new%':>6} {'dark%':>6} {'bed%':>6} "
          f"{'darkR':>6} {'bedR':>6}  pick")
    for desktop in DESKTOPS:
        bgr = read(f"{args.cache}/D{desktop:02d}/s001.png")
        if bgr is None:
            print(f"D{desktop:02d}: no cached s001.png")
            continue
        height, width = bgr.shape[:2]
        dark, bed = _scan_chassis_box(bgr), _scan_bed_box(bgr)
        dark_score = None if dark is None else _box_plausibility(*dark, width, height)
        bed_score = None if bed is None else _box_plausibility(*bed, width, height)
        old = old_suggestion(bgr)
        new = suggest_roi(bgr, "scan")
        pick = ("dark" if dark is not None and new == dark[0]
                else "bed" if bed is not None and new == bed[0] else "central")
        rows.append((desktop, frac(old, width, height), frac(new, width, height), pick))
        if tuple(old) != tuple(new):
            changed.append(desktop)
        print(f"D{desktop:02d} {frac(old, width, height):6.3f} "
              f"{frac(new, width, height):6.3f} "
              f"{'   -  ' if dark is None else f'{frac(dark[0], width, height):6.3f}'} "
              f"{'   -  ' if bed is None else f'{frac(bed[0], width, height):6.3f}'} "
              f"{'   -  ' if dark_score is None else f'{dark_score:6.3f}'} "
              f"{'   -  ' if bed_score is None else f'{bed_score:6.3f}'}  {pick}")
        if desktop in wanted:
            draw(bgr, {"dark": None if dark is None else dark[0],
                       "bed": None if bed is None else bed[0], "chosen": new},
                 f"{args.out}/roi_D{desktop:02d}.png")

    picks: dict[str, int] = {}
    for _d, _o, _n, pick in rows:
        picks[pick] = picks.get(pick, 0) + 1
    print(f"\ndesktops: {len(rows)}   picks: {dict(sorted(picks.items()))}")
    print(f"changed ({len(changed)}): {changed}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
