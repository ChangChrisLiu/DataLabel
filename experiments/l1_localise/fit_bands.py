"""Fit the area bands the app's "no box when unsure" gate reads (task U2h).

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l1_localise.fit_bands \\
        --db D:\\DataSet\\.cache\\tmp\\<copy>.sqlite

writes ``configs/prompt_box_bands.yaml``.  L1 (``REPORT.md`` sections 4 and 8)
measured the gate -- **M1**: withhold the difference map's rank-1 box when its
blob's area is outside the part class's area band by more than a factor
``k = 3`` -- with bands fitted leave-one-desktop-out; production fits on every
desktop.  Where the gate is switched on, and ``k``, are not decided here but in
``configs/prompt_gate.yaml``.

What is fitted, and why so:

* **samples** -- every ``ls:`` keyframe (an old Label Studio draft) of a view,
  one per ``(desktop, view, step, draft)`` as L1 counted them
  (:func:`experiments.plan_b_probe.transfer.common.load_geometry`).  Its class
  is the instance table's ``cls`` for the ``ls:`` key -- the mapping
  ``siblings.py`` and the L1 harness use.
* **band** -- the 5th to 95th percentile of the drafts' areas, as L1's
  ``siblings.Priors._lodo``, but over all desktops.
* **unit: image pixels**, not L1's fraction of the ROI.  The gate compares the
  blob's pixel count, and a pixel band makes that comparison the one that was
  fitted: no ROI enters it.  A fraction does not survive the trip -- L1's
  fractions were taken against a stand-in ROI (the drafts' union box padded
  10 %), and the ROI the app divides by is the annotator's stored one, which on
  D13/scan is 1.75x smaller (644,300 px against 1,129,752).  Replaying L1's own
  leave-one-desktop-out fraction bands with D13's stored ROI withholds two more
  D13 boxes (one of them good: scan's good boxes withheld 5.9 % -> 11.8 %);
  pixel bands fitted leave-one-desktop-out cut the same 72.5 % of scan's wrong
  boxes at 11.8 % good withheld in the replay *and* in the app, and do as well
  or better on the three OAK/RealSense views (``replay_m1_units.py``).  The
  scanner's geometry is fixed, so on scan a screw is the same number of
  pixels on every machine.
* a class with fewer than :data:`MIN_SAMPLES` samples in a view gets **no
  band** (listed under ``unbanded``): the gate never withholds a box on its
  account.

Read-only: the database is opened ``mode=ro``, and a path under
``annotations/`` is refused -- a copy is what this is for.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tda.core import masks  # noqa: E402

#: The views bands are fitted for.  All four, so that switching the gate on
#: for another view is a change to ``configs/prompt_gate.yaml`` alone.
VIEWS = ("scan", "oak1", "oak2", "rs")
#: Fewer samples than this and a class has no band.
MIN_SAMPLES = 5
PERCENTILES = (5.0, 95.0)
OUT = REPO / "configs" / "prompt_box_bands.yaml"
#: The live database lives under ``annotations/``; never read it from here.
DB_GUARD = "annotations"


def _connect(db: str) -> sqlite3.Connection:
    if DB_GUARD in Path(db).resolve().parts:
        raise SystemExit(f"refusing to read {db}: fit on a copy of the database")
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def load_drafts(con: sqlite3.Connection) -> list[dict]:
    """``{desktop, view, step, cls, area, box}`` per ``ls:`` keyframe."""
    rows = con.execute(
        """SELECT k.id, k.desktop, k.view, k.anchor_step AS step, k.instance,
                  i.cls AS cls, p.rle_json
           FROM shape_keyframe k
           JOIN shape_part p ON p.keyframe_id = k.id
           LEFT JOIN instance i ON i.desktop = k.desktop AND i."key" = k.instance
           WHERE k.instance LIKE 'ls:%'
           ORDER BY k.id"""
    ).fetchall()
    by_id: dict[int, dict] = {}
    for r in rows:
        rle = json.loads(r["rle_json"])
        area = masks.rle_area(rle)
        box = masks.rle_bbox(rle)
        if area <= 0 or box is None:
            continue
        found = by_id.get(int(r["id"]))
        if found is None:
            by_id[int(r["id"])] = {
                "desktop": int(r["desktop"]), "view": str(r["view"]),
                "step": int(r["step"]), "cls": str(r["cls"] or ""),
                "area": int(area), "box": tuple(int(v) for v in box)}
        else:                          # a draft in several parts: one sample
            found["area"] += int(area)
            b = found["box"]
            found["box"] = (min(b[0], box[0]), min(b[1], box[1]),
                            max(b[2], box[2]), max(b[3], box[3]))
    return list(by_id.values())


def fit(drafts: list[dict], views=VIEWS) -> dict:
    """``{view: {"classes": {cls: band}, "unbanded": {cls: n}, "desktops": [...]}}``."""
    out: dict = {}
    for view in views:
        areas: dict = defaultdict(list)
        for d in drafts:
            if d["view"] == view and d["cls"]:
                areas[d["cls"]].append((d["area"], d["desktop"]))
        classes, unbanded = {}, {}
        for cls, pairs in sorted(areas.items()):
            values = [a for a, _d in pairs]
            if len(values) < MIN_SAMPLES:
                unbanded[cls] = len(values)
                continue
            lo, hi = (float(np.percentile(values, p)) for p in PERCENTILES)
            classes[cls] = {"lo_px": round(lo, 1), "hi_px": round(hi, 1),
                            "median_px": round(float(np.median(values)), 1),
                            "samples": len(values),
                            # how many machines the samples come from: a band
                            # from one desktop is one machine's parts
                            "desktops": len({d for _a, d in pairs})}
        out[view] = {
            "desktops": sorted({d["desktop"] for d in drafts if d["view"] == view}),
            "classes": classes,
            "unbanded": unbanded,
        }
    return out


HEADER = """\
# Area bands of the parts a difference-map box prompt may be for (task U2h).
# GENERATED by experiments/l1_localise/fit_bands.py -- do not edit by hand;
# re-run it on a copy of the database instead.  Where the gate uses these
# (which views, which factor k) is configs/prompt_gate.yaml.
#
# Per view and taxonomy class: the {lo:g}th-{hi:g}th percentile of the areas of
# the old Label Studio drafts (ls: keyframes, every desktop), in image pixels at
# the view's native resolution -- the unit of the blob it is compared with (the
# script's docstring says why not a fraction of the ROI).  A class with fewer
# than {min_samples} samples has no band (unbanded) and is never gated.
"""


def write(bands: dict, source: str, out: Path) -> None:
    doc = {
        "fitted_from": source,
        "unit": "image px",
        "percentiles": list(PERCENTILES),
        "min_samples": MIN_SAMPLES,
        "views": bands,
    }
    text = HEADER.format(lo=PERCENTILES[0], hi=PERCENTILES[1], min_samples=MIN_SAMPLES)
    # one line per class: the leaves inline, the structure in blocks
    text += yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100,
                           default_flow_style=None)
    out.write_text(text, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, help="a COPY of the annotation database")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--source", default="",
                    help="what the copy is a copy of, for the file's fitted_from")
    args = ap.parse_args(argv)
    con = _connect(args.db)
    try:
        drafts = load_drafts(con)
    finally:
        con.close()
    bands = fit(drafts, views=tuple(args.views.split(",")))
    write(bands, args.source or Path(args.db).name, Path(args.out))
    for view, entry in bands.items():
        print(f"{view}: {len(entry['classes'])} classes banded, unbanded "
              f"{entry['unbanded']}, desktops {entry['desktops']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
