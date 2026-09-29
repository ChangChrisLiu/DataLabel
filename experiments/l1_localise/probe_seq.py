"""Is the part removed at step k near the one removed at step k+1?

In the reverse walk the annotator draws the part removed at ``k+1`` on frame
``k`` and then moves to frame ``k-1``, so that part's location is known when
the guess for step ``k`` is made.  Screws come out in a tour, so the next one
may well be close.  Geometry only (LS drafts, GT-B): for each event whose
``k+1`` event is of the same class in the same view, the distance between the
two parts, in ROI diagonals, and whether the removed part is the nearest of
the same-class parts at ``k-1`` to the previous one.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l1_localise import base  # noqa: E402


def main() -> int:
    events, _shapes, by_frame, con = base.load_events()
    con.close()
    loc = {(int(r["desktop"]), r["view"], int(r["step"])): r["loc_instance"]
           for r in csv.DictReader((base.OUT / "gt_check.csv").open(encoding="utf-8"))}
    ev_by = {(e.desktop, e.view, e.step): e for e in events}
    rows = []
    for e in events:
        nxt = ev_by.get((e.desktop, e.view, e.step + 1))
        lk, lk2 = loc.get((e.desktop, e.view, e.step)), None
        if nxt is not None:
            lk2 = loc.get((nxt.desktop, nxt.view, nxt.step))
        if not lk or nxt is None or not lk2 or nxt.cls != e.cls:
            continue
        cur = by_frame[(e.desktop, e.view, e.j)][lk]
        prev = by_frame[(e.desktop, e.view, e.step)][lk2]   # drawn on frame k
        diag = math.hypot(e.roi[2] - e.roi[0], e.roi[3] - e.roi[1])
        d = math.hypot(cur.centroid[0] - prev.centroid[0],
                       cur.centroid[1] - prev.centroid[1]) / diag
        # ordinals are per frame, so the part removed next is found at k-1 by
        # its location, not by its key
        same = [q for q in by_frame[(e.desktop, e.view, e.j)].values()
                if q.cls == e.cls and base.box_iou(q.box, prev.box) < 0.3]
        ds = sorted((math.hypot(q.centroid[0] - prev.centroid[0],
                                q.centroid[1] - prev.centroid[1]), q.instance)
                    for q in same)
        keys = [inst for _, inst in ds]
        if lk not in keys:
            continue
        rank = keys.index(lk)
        rows.append({"desktop": e.desktop, "view": e.view, "step": e.step,
                     "label": e.label, "bucket": e.bucket,
                     "dist_diag": round(d, 3), "n_same": len(same),
                     "nearest_rank": rank})
    d = np.array([r["dist_diag"] for r in rows])
    rk = np.array([r["nearest_rank"] for r in rows])
    print(f"{len(rows)} events whose k+1 event is the same class (GT-B both)")
    print(f"distance to the part removed next, in ROI diagonals: "
          f"p25 {np.percentile(d, 25):.3f} median {np.median(d):.3f} "
          f"p75 {np.percentile(d, 75):.3f}")
    print(f"the removed part is the nearest same-class part: {100 * np.mean(rk == 0):.1f}%;"
          f" among the 2 nearest: {100 * np.mean(rk <= 1):.1f}%; mean candidates "
          f"{np.mean([r['n_same'] for r in rows]):.1f}")
    path = base.OUT / "seq_probe.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
