"""Fit the area bands the app's "no box when unsure" gate reads (tasks U2h, U2i).

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

* **parts, not keyframes** (task U2i).  The old Label Studio drafts are
  ``ls:`` keyframes, one per ``(desktop, view, step, draft)``, and a part
  that stays in the machine for thirty steps is drawn thirty times.  Counting
  keyframes gave scan's ``expansion_card`` -- one GPU on one desktop, drawn on
  32 steps -- a band of 1,359-1,359 px, and ``cooler_bracket`` -- three real
  brackets, four keyframes -- none.  A **part** is ``(desktop, ls: key)`` in
  a view; its value is the **median** of its keyframes' areas, so one
  long-lived part counts once.  (An ``ls:`` ordinal is the part's rank from
  the left on its frame, not a physical identity, so a key can pass from one
  screw to the next when a neighbour is removed; the parts it names are of one
  label and one size, and the number of keys of a label on a desktop is the
  number of those parts in the machine.)  The class is the instance table's
  ``cls`` for the key -- the mapping ``siblings.py`` and the L1 harness use.
* **band** -- the 5th to 95th percentile of the parts' values, and only for a
  class with at least :data:`MIN_PARTS` parts from at least
  :data:`MIN_DESKTOPS` desktops in the view: fewer is one machine's parts, not
  the class.  Any other class is listed under ``unbanded`` and the gate never
  withholds a box on its account.
* **unit: image pixels**, not L1's fraction of the ROI.  The gate compares the
  blob's pixel count, and a pixel band makes that comparison the one that was
  fitted: no ROI enters it.  A fraction does not survive the trip -- L1's
  fractions were taken against a stand-in ROI (the drafts' union box padded
  10 %), and the ROI the app divides by is the annotator's stored one, which on
  D13/scan is 1.75x smaller (644,300 px against 1,129,752).  Replaying L1's own
  leave-one-desktop-out fraction bands with D13's stored ROI withholds two more
  D13 boxes, one of them good.  The scanner's geometry is fixed, so on scan a
  screw is the same number of pixels on every machine.
  ``replay_m1_units.py`` gives the numbers, **leave-one-desktop-out** (the
  event's desktop left out of the fit, as L1 measured) and **in-sample** (the
  production fit, which has seen every event's desktop) side by side; only
  the first is a measurement of how the gate will do on a new machine.  On
  scan's 57 boxes of L1's events (k = 3, GT-B) these bands cut 72.5 % (29/40)
  of the wrong boxes and withhold 5.9 % (1/17) of the good ones
  leave-one-desktop-out, and the same in-sample -- L1's own fraction bands,
  leave-one-desktop-out, did exactly that too.  U2h's keyframe-counted pixel
  bands withheld 11.8 % (2/17) leave-one-desktop-out.

Read-only: the database is opened ``mode=ro``, and a path under
``annotations/`` is refused -- a copy is what this is for.  The output is a
function of the database alone: two runs on two copies are byte-identical.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tda.core import masks  # noqa: E402

#: The views bands are fitted for.  All four, so that switching the gate on
#: for another view is a change to ``configs/prompt_gate.yaml`` alone.
VIEWS = ("scan", "oak1", "oak2", "rs")
#: A class needs this many distinct parts ...
MIN_PARTS = 5
#: ... from this many desktops, or it has no band.
MIN_DESKTOPS = 2
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
    """``{desktop, view, step, instance, cls, area, box}`` per ``ls:`` keyframe."""
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
                "step": int(r["step"]), "instance": str(r["instance"]),
                "cls": str(r["cls"] or ""), "area": int(area),
                "box": tuple(int(v) for v in box)}
        else:                          # a draft in several parts: one sample
            found["area"] += int(area)
            b = found["box"]
            found["box"] = (min(b[0], box[0]), min(b[1], box[1]),
                            max(b[2], box[2]), max(b[3], box[3]))
    return list(by_id.values())


def part_values(drafts: Iterable[dict], view: str, cls: str,
                exclude_desktop: Optional[int] = None) -> list[tuple[float, int, int]]:
    """``(median area, desktop, keyframes)`` per part of ``cls`` in ``view``.

    A part is ``(desktop, ls: key)``; ``exclude_desktop`` leaves one machine
    out (the leave-one-desktop-out replay).
    """
    areas: dict[tuple[int, str], list[int]] = defaultdict(list)
    for d in drafts:
        if (d["view"] == view and d["cls"] == cls
                and (exclude_desktop is None or d["desktop"] != exclude_desktop)):
            areas[(d["desktop"], d["instance"])].append(d["area"])
    return [(float(np.median(values)), desktop, len(values))
            for (desktop, _key), values in sorted(areas.items())]


def band_of(parts: list[tuple[float, int, int]]) -> Optional[dict]:
    """The band of a class from its parts, or ``None`` below the bar."""
    desktops = {desktop for _v, desktop, _n in parts}
    if len(parts) < MIN_PARTS or len(desktops) < MIN_DESKTOPS:
        return None
    values = [v for v, _d, _n in parts]
    lo, hi = (float(np.percentile(values, p)) for p in PERCENTILES)
    return {"lo_px": round(lo, 1), "hi_px": round(hi, 1),
            "median_px": round(float(np.median(values)), 1),
            "parts": len(parts), "desktops": len(desktops),
            "keyframes": int(sum(n for _v, _d, n in parts))}


def fit(drafts: list[dict], views=VIEWS) -> dict:
    """``{view: {"classes": {cls: band}, "unbanded": {cls: counts}, "desktops": [...]}}``."""
    out: dict = {}
    for view in views:
        classes, unbanded = {}, {}
        for cls in sorted({d["cls"] for d in drafts if d["view"] == view and d["cls"]}):
            parts = part_values(drafts, view, cls)
            band = band_of(parts)
            if band is None:
                unbanded[cls] = {"parts": len(parts),
                                 "desktops": len({d for _v, d, _n in parts})}
            else:
                classes[cls] = band
        out[view] = {
            "desktops": sorted({d["desktop"] for d in drafts if d["view"] == view}),
            "classes": classes,
            "unbanded": unbanded,
        }
    return out


HEADER = """\
# Area bands of the parts a difference-map box prompt may be for (tasks U2h, U2i).
# GENERATED by experiments/l1_localise/fit_bands.py -- do not edit by hand;
# re-run it on a copy of the database instead.  Where the gate uses these
# (which views, which factor k) is configs/prompt_gate.yaml.
#
# Per view and taxonomy class: the {lo:g}th-{hi:g}th percentile over the class's
# PARTS -- (desktop, ls: key), each the median area of its old Label Studio
# drafts, so a part drawn on thirty steps counts once -- in image pixels at the
# view's native resolution, the unit of the blob it is compared with (the
# script's docstring says why not a fraction of the ROI).  A class with fewer
# than {min_parts} parts, or with parts from fewer than {min_desktops} desktops,
# has no band (unbanded) and is never gated.
"""


def render(bands: dict, source: str) -> str:
    doc = {
        "fitted_from": source,
        "unit": "image px",
        "per_part": "median area over the part's keyframes",
        "percentiles": list(PERCENTILES),
        "min_parts": MIN_PARTS,
        "min_desktops": MIN_DESKTOPS,
        "views": bands,
    }
    text = HEADER.format(lo=PERCENTILES[0], hi=PERCENTILES[1], min_parts=MIN_PARTS,
                         min_desktops=MIN_DESKTOPS)
    # one line per class: the leaves inline, the structure in blocks
    return text + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100,
                                 default_flow_style=None)


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
    text = render(bands, args.source or Path(args.db).name)
    Path(args.out).write_bytes(text.encode("utf-8"))
    for view, entry in bands.items():
        print(f"{view}: {len(entry['classes'])} classes banded, unbanded "
              f"{ {c: (u['parts'], u['desktops']) for c, u in entry['unbanded'].items()} }, "
              f"desktops {entry['desktops']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
