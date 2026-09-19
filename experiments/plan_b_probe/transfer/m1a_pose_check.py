"""Did the camera or the chassis move inside a capture sequence?

A homography clicked once per (desktop, view) is only usable if the chassis and
the camera hold still for the whole sequence. The dataset owner reports the
cameras were bumped over the ten capture days, so this asks the frames
themselves before any corner is clicked: if a view has to be split into pose
segments, M1 has to be clicked per segment.

Method: every frame is reduced to a 512 px grey image, Hann-windowed, and
compared to the *first* frame of its sequence with :func:`cv2.phaseCorrelate`,
which returns the sub-pixel translation that best aligns the two. The result is
scaled back to native pixels. Translation is the only motion phase correlation
sees, but a bumped tripod shows up as translation first, and a real bump is a
**step change** between consecutive frames rather than the slow drift that parts
being removed produce in the correlation peak.

Reports, per (desktop, view): the largest frame-to-frame jump, where it happened,
and the largest displacement from the first frame. Prints a table; writes
``m1a_pose_check.csv``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.plan_b_probe.transfer.common import (  # noqa: E402
    DESKTOPS, OUT, VIEWS, connect, frame_path,
)

SMALL = 512


def reduced(path: str) -> tuple[np.ndarray, float] | None:
    """Grey float32 image at most ``SMALL`` px on its long edge, plus the scale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    factor = SMALL / float(max(h, w))
    small = cv2.resize(img, (max(8, int(round(w * factor))),
                             max(8, int(round(h * factor)))),
                       interpolation=cv2.INTER_AREA)
    return small.astype(np.float32), factor


def main() -> int:
    con = connect()
    rows = []
    print(f"{'view':14s} {'n':>3s} {'max jump px':>11s} {'at step':>8s} "
          f"{'max drift px':>12s}")
    for desktop in DESKTOPS:
        for view in VIEWS:
            steps = [int(r["step"]) for r in con.execute(
                "SELECT step FROM frame WHERE desktop=? AND view=? ORDER BY step",
                (desktop, view))]
            ref = None
            win = None
            prev = None
            drifts: list[tuple[int, float, float]] = []
            for st in steps:
                path = frame_path(con, desktop, view, st)
                if path is None:
                    continue
                got = reduced(path)
                if got is None:
                    continue
                small, factor = got
                if ref is None:
                    ref, win = small, cv2.createHanningWindow(
                        (small.shape[1], small.shape[0]), cv2.CV_32F)
                    prev = small
                    drifts.append((st, 0.0, 0.0))
                    continue
                (dx, dy), _ = cv2.phaseCorrelate(ref, small, win)
                drift = float(np.hypot(dx, dy)) / factor
                (jx, jy), _ = cv2.phaseCorrelate(prev, small, win)
                jump = float(np.hypot(jx, jy)) / factor
                drifts.append((st, drift, jump))
                prev = small
            if not drifts:
                continue
            worst = max(drifts[1:], key=lambda t: t[2], default=(0, 0.0, 0.0))
            max_drift = max(d for _s, d, _j in drifts)
            print(f"D{desktop} {view:8s} {len(drifts):3d} {worst[2]:11.1f} "
                  f"{worst[0]:8d} {max_drift:12.1f}")
            for st, drift, jump in drifts:
                rows.append({"desktop": desktop, "view": view, "step": st,
                             "drift_from_first_px": round(drift, 2),
                             "jump_from_prev_px": round(jump, 2)})
    con.close()
    path = OUT / "m1a_pose_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
