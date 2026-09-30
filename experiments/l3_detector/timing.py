"""Runtime of the guess per model, with L2's protocol (``l2_detector.timing``).

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.timing --model yolo26s

Same events (up to 30 per view, ``default_rng(0)``), frame ``j`` decoded
outside the clock, 3 repetitions, median:

* ``det``   -- tiled detector on the ROI crop (crop + tiling + one batched
  fp16 forward + merge), the fold that held the desktop out;
* ``rank``  -- native dE in every candidate box (conf >= 0.10, after the skip);
* ``guess`` -- det + rank, what a frame change would cost without precompute;
* ``full``  -- the detector on the whole 12 MP OAK1 frame (no ROI);
* ``bg``    -- background precompute: decode + detect per frame over 20 frames
  of D24 scan / oak1 (after which a frame change pays only ``rank``);
* ``gpu_mb`` -- peak CUDA memory allocated by the process while detecting.

Every model runs through this one script in one session, YOLO26n included, so
the numbers compare under the same machine load.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402

PER_VIEW = 30
REPS = 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--views", default="scan,oak1,oak2,rs")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="score floor before the merge (0.001 = L2's protocol)")
    args = ap.parse_args(argv)
    from experiments.l3_detector.infer import tag_of

    out = env.bind_l2(tag_of(args.model, args.conf))
    from experiments.l2_detector import data as D
    from experiments.l2_detector.guess_eval import CONF, FOLD_OF, setup_l1, suppressed
    from experiments.l1_localise import base
    from experiments.l1_localise import methods as M
    from experiments.l1_localise.siblings import known_boxes_at
    from experiments.l3_detector.infer import make_detector

    events, shapes, by_frame, con = setup_l1()
    import torch

    dets_by_fold = {f: make_detector(args.model, f, conf=args.conf) for f in FOLD_OF.values()}
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(0)
    rows = []
    views = args.views.split(",")
    for view in D.VIEWS:
        evs = [e for e in events if e.view == view]
        if len(evs) > PER_VIEW:
            # drawn for every view, as L2 did, so the same events are timed
            evs = [evs[i] for i in sorted(rng.choice(len(evs), PER_VIEW, replace=False))]
        if view not in views:
            continue
        for ev in evs:
            det = dets_by_fold[FOLD_OF[ev.desktop]]
            pj = base.frame_path(con, ev.desktop, view, ev.j)
            pk = base.frame_path(con, ev.desktop, view, ev.step)
            img_j = D.read_rgb(pj) if pj else None
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
                rec["det"].append(t1 - t0)
                rec["net"].append(tim["net_s"])
                rec["rank"].append(t2 - t1)
                rec["guess"].append(t2 - t0)
                rec["n_tiles"].append(tim["n_tiles"])
            row = {"view": view, "ev": f"{ev.desktop}:{view}:{ev.step}",
                   "roi_w": ev.roi[2] - ev.roi[0], "roi_h": ev.roi[3] - ev.roi[1]}
            row.update({k: round(float(np.median(v)), 4) for k, v in rec.items()})
            rows.append(row)
        print(f"  {view}: {sum(1 for r in rows if r['view'] == view)} events", flush=True)

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

    bg = {}
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
        bg[view] = (time.perf_counter() - t0) / max(1, n)
    con.close()
    gpu_mb = torch.cuda.max_memory_allocated() / 2 ** 20

    summ = {"model": args.model, "conf_floor": args.conf, "views": {},
            "full_oak1_median": float(np.median(full)),
            "full_oak1_p90": float(np.percentile(full, 90)), "bg_per_frame": bg,
            "gpu_peak_mb_3_folds_loaded": round(gpu_mb)}
    for view in D.VIEWS:
        s = [r for r in rows if r["view"] == view]
        if not s:
            continue
        d = {"n": len(s), "roi": f"{int(np.median([r['roi_w'] for r in s]))} x "
                                  f"{int(np.median([r['roi_h'] for r in s]))}",
             "tiles": int(np.median([r["n_tiles"] for r in s]))}
        for k in ("det", "net", "rank", "guess"):
            v = np.asarray([r[k] for r in s], float)
            d[k] = [round(float(np.median(v)), 4), round(float(np.percentile(v, 90)), 4)]
        summ["views"][view] = d
    (out / "timing_l3.json").write_text(json.dumps({"summary": summ, "events": rows}, indent=1),
                                        encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
