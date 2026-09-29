"""Runtime of the L2 guess on this GPU: detector on the ROI + dE ranking.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.timing

For up to 30 events per view (frame ``j`` decoded once, outside the clock),
each measured 3 times and the median kept -- L1's protocol:

* ``det``    -- tiled YOLO26n on the ROI crop (crop + tiling + one batched
  forward + merge), fp16, the fold that held the desktop out;
* ``rank``   -- native dE in every candidate box (conf >= 0.10, after
  suppression) and the sort;
* ``guess``  -- det + rank, what would run on a frame change;
* ``diff``   -- today's ``diff_delta_e(k, j)`` + ``diff_blobs`` on the same
  pair, for scale;
* ``decode`` -- reading the frame from the raw drive (not in ``guess``: the
  app has the frame decoded when it shows it).

Also: the same detector on the full 12 MP OAK frame (no ROI) and the
throughput of a background pass over a whole view's frames (precompute).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402
from experiments.l2_detector.guess_eval import (  # noqa: E402
    CONF, FOLD_OF, setup_l1, suppressed,
)
from experiments.l2_detector.infer import Detector, weights_of  # noqa: E402
from experiments.l1_localise import base  # noqa: E402
from experiments.l1_localise import methods as M  # noqa: E402
from experiments.l1_localise.siblings import known_boxes_at  # noqa: E402
from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

PER_VIEW = 30
REPS = 3


def main() -> int:
    events, shapes, by_frame, con = setup_l1()
    dets_by_fold = {f: Detector(weights_of(f)) for f in FOLD_OF.values()}
    rng = np.random.default_rng(0)
    rows = []
    for view in D.VIEWS:
        evs = [e for e in events if e.view == view]
        if len(evs) > PER_VIEW:
            evs = [evs[i] for i in sorted(rng.choice(len(evs), PER_VIEW, replace=False))]
        for ev in evs:
            det = dets_by_fold[FOLD_OF[ev.desktop]]
            t0 = time.perf_counter()
            pj = base.frame_path(con, ev.desktop, view, ev.j)
            pk = base.frame_path(con, ev.desktop, view, ev.step)
            img_j = D.read_rgb(pj) if pj else None
            decode = time.perf_counter() - t0
            img_k = D.read_rgb(pk) if pk else None
            if img_j is None or img_k is None:
                continue
            known = known_boxes_at(ev, by_frame)
            rec = defaultdict(list)
            for _ in range(REPS):
                tim = {}
                t0 = time.perf_counter()
                found = det.detect(img_j, ev.roi, view, timings=tim)
                t1 = time.perf_counter()
                cands = [d for d in found if d.cls == ev.cls and d.conf >= CONF]
                cands = [d for d in cands
                         if not suppressed(tuple(int(round(v)) for v in d.box), known)]
                ch = [M.box_change(img_j, img_k, (0, 0),
                                   tuple(int(round(v)) for v in d.box)) for d in cands]
                _order = np.argsort(-np.asarray(ch)) if ch else []
                t2 = time.perf_counter()
                delta = diff_delta_e(img_k, img_j, roi=ev.roi, max_side=M.MAX_SIDE)
                diff_blobs(delta)
                t3 = time.perf_counter()
                rec["det"].append(t1 - t0)
                rec["net"].append(tim["net_s"])
                rec["prep"].append(tim["prep_s"])
                rec["rank"].append(t2 - t1)
                rec["guess"].append(t2 - t0)
                rec["diff"].append(t3 - t2)
                rec["n_tiles"].append(tim["n_tiles"])
                rec["n_cands"].append(len(cands))
            row = {"view": view, "ev": f"{ev.desktop}:{view}:{ev.step}",
                   "roi_w": ev.roi[2] - ev.roi[0], "roi_h": ev.roi[3] - ev.roi[1],
                   "decode": round(decode, 4)}
            row.update({k: round(float(np.median(v)), 4) for k, v in rec.items()})
            rows.append(row)
        print(f"  {view}: {sum(1 for r in rows if r['view'] == view)} events", flush=True)

    # full 12 MP OAK frame, no ROI (reference)
    full = []
    for ev in [e for e in events if e.view == "oak1"][:10]:
        det = dets_by_fold[FOLD_OF[ev.desktop]]
        img = D.read_rgb(base.frame_path(con, ev.desktop, "oak1", ev.j))
        h, w = img.shape[:2]
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            det.detect(img, (0, 0, w, h), "oak1")
            ts.append(time.perf_counter() - t0)
        full.append(float(np.median(ts)))

    # background precompute throughput over one view (decode + detect)
    tp = {}
    for view in ("scan", "oak1"):
        steps = sorted({s for (d, v, s) in by_frame if d == 24 and v == view})[:20]
        det = dets_by_fold["hold24"]
        roi = next(e.roi for e in events if e.desktop == 24 and e.view == view)
        t0 = time.perf_counter()
        n = 0
        for s in steps:
            p = base.frame_path(con, 24, view, s)
            img = D.read_rgb(p) if p else None
            if img is None:
                continue
            det.detect(img, roi, view)
            n += 1
        tp[view] = (time.perf_counter() - t0) / max(1, n)
    con.close()

    path = env.OUT / "timing_events.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| view | n | ROI px (median) | tiles | det | net only | rank (dE) | "
             "guess = det + rank | today's diff | decode (not counted) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for view in D.VIEWS:
        s = [r for r in rows if r["view"] == view]
        if not s:
            continue

        def mp(k):
            v = np.asarray([r[k] for r in s], float)
            return f"{np.median(v):.3f} / {np.percentile(v, 90):.3f}"
        roi = f"{int(np.median([r['roi_w'] for r in s]))} x {int(np.median([r['roi_h'] for r in s]))}"
        lines.append(f"| {view} | {len(s)} | {roi} | {int(np.median([r['n_tiles'] for r in s]))} | "
                     f"{mp('det')} | {mp('net')} | {mp('rank')} | {mp('guess')} | "
                     f"{mp('diff')} | {mp('decode')} |")
    lines.append("")
    lines.append(f"Full 12 MP OAK frame, no ROI (10 frames): median {np.median(full):.3f} s, "
                 f"p90 {np.percentile(full, 90):.3f} s")
    lines.append("Background precompute, decode + detect per frame (20 frames of D24): "
                 + ", ".join(f"{v} {t:.3f} s" for v, t in tp.items()))
    txt = "\n".join(lines)
    (env.OUT / "timing.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
