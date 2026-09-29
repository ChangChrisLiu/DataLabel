"""Runtime per event on 12 MP OAK frames, measured on its own.

The main run times every method while ~25 variants and a neighbouring test
suite share the CPU, so its ``secs`` column is an upper bound.  This replays a
sample of OAK events (frames decoded beforehand, as they are in the app), runs
each stage ``--repeat`` times and keeps the median:

* ``m0_full``  -- today's call: ``diff_delta_e`` on the full frame + ``diff_blobs``
* ``m0_crop``  -- the same map computed on the ROI crop (same scale)
* ``reg_ecc``  -- M2's ECC estimate + warp of the crop
* ``m2e``      -- ``reg_ecc`` + dE + blobs (what M4 builds on)
* ``search``   -- the M3/M4 template search on an existing dE map
* ``m5``       -- the size-matched filter on an existing dE map

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l1_localise.timing --variant M4_b1
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

from experiments.l1_localise import base  # noqa: E402
from experiments.l1_localise import methods as M  # noqa: E402
from experiments.l1_localise.run_l1 import VARIANTS  # noqa: E402
from experiments.l1_localise.siblings import (  # noqa: E402
    Priors, known_boxes_at, sibling_pool,
)


def timed(fn, repeat: int) -> float:
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="M4_b1")
    ap.add_argument("--views", default="oak1,oak2")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)
    views = tuple(args.views.split(","))
    events, shapes, by_frame, con = base.load_events()
    priors = Priors(shapes)
    events = [e for e in events if e.view in views]
    # spread the sample over desktops and sizes
    events = events[:: max(1, len(events) // args.n)][: args.n]
    names = args.variants.split(",")
    rows = {k: [] for k in ["m0_full", "m0_crop", "reg_ecc", "m2e", "m5"]
            + [f"search:{n}" for n in names]}
    frames = None
    cur = None
    for ev in events:
        if cur != (ev.desktop, ev.view):
            cur = (ev.desktop, ev.view)
            frames = base.Frames(con, ev.desktop, ev.view, keep=8)
        ij, ik = frames.get(ev.j), frames.get(ev.step)
        if ij is None or ik is None:
            continue
        # the app's call: neighbour first, default min_area
        rows["m0_full"].append(timed(lambda: diff_blobs(diff_delta_e(
            ik, ij, roi=ev.roi, max_side=1600)), args.repeat))
        rows["m0_crop"].append(timed(lambda: M.m2(ij, ik, ev.roi, ev.min_area,
                                                  "none"), args.repeat))
        margin = int(0.05 * max(ev.roi[2] - ev.roi[0], ev.roi[3] - ev.roi[1]))
        x0, y0, x1, y1 = M._roi_window(ev.roi, ij.shape, margin)
        cj, ck = ij[y0:y1, x0:x1], ik[y0:y1, x0:x1]
        rows["reg_ecc"].append(timed(lambda: M.estimate_shift(cj, ck, "ecc"),
                                     args.repeat))
        rows["m2e"].append(timed(lambda: M.m2(ij, ik, ev.roi, ev.min_area, "ecc"),
                                 args.repeat))
        _p, d2, org, _i, ck = M.m2(ij, ik, ev.roi, ev.min_area, "ecc")
        side = priors.median_side(ev)
        if side:
            rows["m5"].append(timed(lambda: M.m5(d2, org, ev.roi, side),
                                    args.repeat))
        s, pool = sibling_pool(ev, by_frame)
        src = frames.get(s) if pool else None
        if src is None:
            continue
        med = max(1.0, float(np.median([q.area for q in pool])))
        ordered = sorted(pool, key=lambda q: abs(math.log(max(1, q.area) / med)))
        for n in names:
            cfg, src_kind = VARIANTS[n]
            temps = [t for t in (M.cut_template(src, tuple(q.box), s, q.instance,
                                                pad_frac=cfg.pad_frac)
                                 for q in ordered) if t is not None]
            known = known_boxes_at(ev, by_frame)
            kr = ck if src_kind in ("reg", "reg_app") else None
            rows[f"search:{n}"].append(timed(lambda: M.m3(
                ij, ik, ev.roi, temps, cfg, known_boxes=known, delta=d2,
                delta_origin=org, k_reg=kr, k_origin=org), args.repeat))
    con.close()
    print(f"views {views}, {len(events)} events sampled, median of "
          f"{args.repeat} runs each; seconds per event")
    print("| stage | n | median | p90 | max |")
    print("|---|---|---|---|---|")
    for k, v in rows.items():
        if v:
            print(f"| {k} | {len(v)} | {np.median(v):.3f} | "
                  f"{np.percentile(v, 90):.3f} | {max(v):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
