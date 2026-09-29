"""Which events have a nearly black (or blown) frame j or k?

``frame.image_quality`` is NULL everywhere, and B2 counted at least one black
scan frame as a method failure.  This records the mean grey level inside the
ROI of both frames of every event, so the report can say how many misses are
data rather than method.  Writes ``frame_quality.csv``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l1_localise import base  # noqa: E402

DARK = 40.0     # mean grey inside the ROI; a normal frame is 90-180


def main() -> int:
    events, _shapes, _by_frame, con = base.load_events()
    rows = []
    frames, cur = None, None
    for ev in events:
        if cur != (ev.desktop, ev.view):
            cur = (ev.desktop, ev.view)
            frames = base.Frames(con, *cur, keep=4)
        x0, y0, x1, y1 = ev.roi
        m = {}
        for name, step in (("j", ev.j), ("k", ev.step)):
            img = frames.get(step)
            g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
            m[name] = float(g[::4, ::4].mean())
        rows.append({"desktop": ev.desktop, "view": ev.view, "step": ev.step,
                     "mean_j": round(m["j"], 1), "mean_k": round(m["k"], 1),
                     "dark": int(min(m.values()) < DARK)})
    con.close()
    path = base.OUT / "frame_quality.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dark = [r for r in rows if r["dark"]]
    print(f"{len(dark)} of {len(rows)} events have a frame with mean grey < {DARK}:")
    for r in dark:
        print("  ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
