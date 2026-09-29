"""L1's M1 gate replayed with the bands in pixels and in ROI fractions (task U2h).

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l1_localise.replay_m1_units \\
        --db D:\\DataSet\\.cache\\tmp\\<copy>.sqlite

Why ``fit_bands.py`` fits pixels.  Reads L1's per-event rows
(``experiments_out/l1_localise/events_v*.csv``; ``M0app`` is the app's own
rank-1 blob) and the drafts of a database copy, and prints, for ``k = 3`` and
GT-B, how many boxes each band withholds, how many of the wrong ones it cuts
and how many good ones it withholds, per view:

* ``frac-lodo (L1)`` -- L1's own ``band_lodo`` column: reproduces REPORT.md;
* ``frac-lodo, stored ROI`` -- the same bands converted with the ROI the app
  would divide by on D13/scan (its stored one; no other segment of the copy
  has one);
* ``px-lodo`` -- pixel bands, leave-one-desktop-out over every desktop;
* ``px-all`` -- the production fit (in-sample for these events).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.l1_localise import fit_bands as F  # noqa: E402

K = 3.0
EVENTS = "D:/DataSet/experiments_out/l1_localise/events_v*.csv"
#: The one stored ROI of the copy L1 ran on: D13/scan, [627, 294, 1385, 1144].
STORED_ROI_AREA = {(13, "scan"): 758.0 * 850.0}


def load_m0app() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(EVENTS)):
        d = pd.read_csv(path)
        if "method" in d.columns:
            frames.append(d[d["method"] == "M0app"])
    return pd.concat(frames).drop_duplicates(subset=["desktop", "view", "step"])


def outside(area: float, band) -> float:
    """L1's ``m1_outside``: how far outside ``band`` ``area`` is, as a factor."""
    if band is None or not (area == area) or area <= 0:
        return 1.0
    lo, hi = band
    return lo / area if area < lo else (area / hi if area > hi else 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, help="a COPY of the annotation database")
    args = ap.parse_args(argv)
    m0 = load_m0app()
    con = F._connect(args.db)
    try:
        drafts = F.load_drafts(con)
    finally:
        con.close()
    px = defaultdict(list)
    for d in drafts:
        px[(d["view"], d["cls"])].append((d["area"], d["desktop"]))

    def band_px(r, exclude=True):
        vals = [a for a, desk in px.get((r.view, r.cls), [])
                if not exclude or desk != int(r.desktop)]
        return (tuple(np.percentile(vals, F.PERCENTILES))
                if len(vals) >= F.MIN_SAMPLES else None)

    def frac_l1(r):
        raw = r.band_lodo
        return tuple(json.loads(raw)) if isinstance(raw, str) and raw else None

    def frac_stored(r):
        band = frac_l1(r)
        area = STORED_ROI_AREA.get((int(r.desktop), r.view))
        if band is None or area is None:
            return band
        scale = area / float(r.roi_area)
        return (band[0] * scale, band[1] * scale)

    def table(name, band_of):
        print(name)
        for view in F.VIEWS:
            sub = m0[(m0["view"] == view) & m0["loc_instance"].notna()
                     & (m0["loc_instance"] != "")]
            has = (sub["n_props"] > 0).to_numpy()
            good = (sub["B_in1"] == 1).to_numpy() & has
            wrong = ~good & has
            held = has & np.array([outside(r.p1_area, band_of(r)) > K
                                   for r in sub.itertuples()], dtype=bool)

            def pct(base):
                return f"{100 * held[base].mean():5.1f}" if base.any() else "    -"

            print(f"  {view:5s} boxes {int(has.sum()):3d}  withheld {pct(has)}  "
                  f"wrong cut {pct(wrong)}  good withheld {pct(good)}")

    table("frac-lodo (L1's band_lodo)", frac_l1)
    table("frac-lodo, D13/scan converted by its stored ROI", frac_stored)
    table("px-lodo (every desktop)", band_px)
    table("px-all (the production fit, in-sample)", lambda r: band_px(r, False))
    sub = m0[(m0.view == "scan") & (m0.desktop == 13) & m0.loc_instance.notna()]
    for name, fn in (("frac-lodo (L1)", frac_l1), ("frac-lodo, stored ROI", frac_stored),
                     ("px-lodo", band_px)):
        held = [(int(r.step) - 1, "good" if int(r.B_in1) else "wrong")
                for r in sub.itertuples()
                if r.n_props > 0 and outside(r.p1_area, fn(r)) > K]
        print(f"D13/scan frames withheld under {name}: {held}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
