"""Look at the segment-wide ROI proposal on real frames (plan B task B3, round 1).

    D:\\Anaconda\\envs\\tda\\python.exe experiments/roi_segment_eval.py \\
        --db D:/DataSet/.cache/tmp/planb_ro.sqlite --out D:/DataSet/.cache/tmp/b3/seg

One proposal per ``(desktop, view, pose segment)`` -- the union of what the
segment's first, middle and last frames say, gated as a whole, and grown by
``OAK_ROI_PAD`` on an OAK view -- drawn on the segment's **first and last**
frame so the same box can be judged against a full machine and an empty one.

``--before`` measures the way it was measured before the review instead: one
reference frame, no union, no pad. Running both writes two sets of overlays
with the same names under different directories, which is what makes the two
countable side by side.

The database is opened **read-only** and nothing is written to it; ``F:`` is
only read.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tda.core.cache import (  # noqa: E402
    ROI_SAMPLE_FRAMES,
    full_frame,
    measure_roi,
    suggest_roi,
    suggest_roi_over,
)

#: The twelve scanner machines the reviewer looked at.
SCAN_DESKTOPS = (1, 13, 21, 24, 29, 33, 34, 36, 45, 61, 63, 64)
#: The twelve OAK machines, spread over the rig epochs.
OAK_DESKTOPS = (1, 13, 21, 24, 29, 34, 36, 45, 52, 58, 63, 64)
SHOT_WIDTH = 1000


def segments(conn, desktop: int, view: str) -> list[tuple[int, list[int]]]:
    """``[(seg, steps)]`` for one view, from the pose segment table."""
    rows = conn.execute(
        "SELECT step FROM frame WHERE desktop=? AND view=? "
        "AND COALESCE(missing, 0) = 0 AND path IS NOT NULL ORDER BY step",
        (desktop, view)).fetchall()
    steps = [int(r["step"]) for r in rows]
    if not steps:
        return []
    found = conn.execute(
        "SELECT seg, start_step, end_step FROM pose_segment WHERE desktop=? AND view=? "
        "AND start_step IS NOT NULL ORDER BY seg", (desktop, view)).fetchall()
    if not found:
        return [(1, steps)]
    out = []
    for row in found:
        inside = [s for s in steps
                  if int(row["start_step"]) <= s <= int(row["end_step"])]
        if inside:
            out.append((int(row["seg"]), inside))
    return out


def sample(steps: list[int]) -> list[int]:
    if len(steps) <= ROI_SAMPLE_FRAMES:
        return list(steps)
    return [steps[0], steps[len(steps) // 2], steps[-1]]


def path_of(conn, desktop: int, view: str, step: int):
    row = conn.execute(
        "SELECT path FROM frame WHERE desktop=? AND view=? AND step=?",
        (desktop, view, step)).fetchone()
    return None if row is None else row["path"]


def overlay(img, box, label: str, out: Path) -> None:
    height, width = img.shape[:2]
    factor = SHOT_WIDTH / float(width)
    small = cv2.resize(img, (SHOT_WIDTH, int(round(height * factor))),
                       interpolation=cv2.INTER_AREA)
    if box is not None and tuple(box) != full_frame(width, height):
        x0, y0, x1, y1 = (int(round(v * factor)) for v in box)
        cv2.rectangle(small, (x0, y0), (x1, y1), (0, 0, 255), 3)
        frac = (box[2] - box[0]) * (box[3] - box[1]) / float(width * height)
        text = f"{label}  {frac:.0%} of the frame"
    else:
        text = f"{label}  NO PROPOSAL"
    cv2.rectangle(small, (0, 0), (small.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(small, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="D:/DataSet/.cache/tmp/planb_ro.sqlite")
    ap.add_argument("--out", default="D:/DataSet/.cache/tmp/b3/seg")
    ap.add_argument("--views", default="scan,oak1,oak2")
    ap.add_argument("--desktops", default="")
    ap.add_argument("--before", action="store_true",
                    help="one reference frame, no union and no pad -- the way "
                         "it was measured before the review")
    args = ap.parse_args(argv)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out_dir = Path(args.out)
    views = [v for v in args.views.split(",") if v]
    print(f"{'segment':26s} {'steps':14s} {'proposed':9s} {'frac':6s} box")
    made = {v: [0, 0] for v in views}
    for view in views:
        chosen = ([int(d) for d in args.desktops.split(",") if d]
                  or (SCAN_DESKTOPS if view == "scan" else OAK_DESKTOPS))
        for desktop in chosen:
            for seg, steps in segments(conn, desktop, view):
                wanted = sample(steps)
                if args.before:
                    wanted = wanted[:1]      # the old single reference frame
                images = []
                for step in wanted:
                    path = path_of(conn, desktop, view, step)
                    found = None if path is None else cv2.imread(path)
                    if found is not None:
                        images.append(found)
                if not images:
                    continue
                height, width = images[0].shape[:2]
                if args.before:
                    box = suggest_roi(images[0], view)
                else:
                    box = suggest_roi_over(images, view)
                proposed = tuple(box) != full_frame(width, height)
                made[view][1] += 1
                made[view][0] += int(proposed)
                label = f"D{desktop:02d} {view} seg{seg}"
                frac = ((box[2] - box[0]) * (box[3] - box[1]) / float(width * height)
                        if proposed else None)
                print(f"{label:26s} {str(wanted):14s} {str(proposed):9s} "
                      f"{'-' if frac is None else f'{frac:.3f}':6s} "
                      f"{tuple(box) if proposed else ''}")
                # drawn on the two frames a human has to judge it against
                for step in (steps[0], steps[-1]):
                    path = path_of(conn, desktop, view, step)
                    img = None if path is None else cv2.imread(path)
                    if img is None:
                        continue
                    overlay(img, box if proposed else None,
                            f"{label} s{step:03d}",
                            out_dir / f"D{desktop:02d}_{view}_seg{seg}_s{step:03d}.jpg")
    conn.close()
    for view, (yes, total) in made.items():
        print(f"{view}: a proposal on {yes}/{total} segments")
    print(f"overlays in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
