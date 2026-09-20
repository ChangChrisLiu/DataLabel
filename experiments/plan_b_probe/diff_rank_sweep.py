"""Which ranking cue actually picks the part?  Dump the features once, sweep.

The splitter's candidate set holds the right region far more often than its
top-1 does (the ``--diag`` columns of ``diff_eval.py`` say 33-57 % against
16-38 %), so the open question is the *ranking*, not the generation.  Re-running
the whole harness per weight combination costs four minutes of 12 MP JPEG
decoding each time, so this dumps every candidate's features once
(``features.json``) and then scores them offline.

Honesty note: the weights are swept on the **same** 201 noisy Label Studio
events the result is reported on, so a combination that wins here by a couple of
points has not been validated on anything.  Only a large, structural difference
is worth acting on, and the per-view spread is printed so a win that comes from
one view can be seen for what it is.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tda.core import diff_split as DS  # noqa: E402
from tda.core.diffmap import diff_delta_e  # noqa: E402

from experiments.plan_b_probe import diff_eval as E  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402

FEATURES = E.OUT / "features.json"


def dump(args) -> None:
    from tda.ui.app_priors import load_priors

    db = str(args.db)
    common.DB_URI = db if db.startswith("file:") else f"file:{db}?mode=ro"
    common.TMP = E.TMP
    shapes = common.load_geometry()
    by_frame = common.index_shapes(shapes)
    priors = load_priors()
    con = common.connect()
    classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}
    out: list[dict] = []

    for desktop in common.DESKTOPS:
        for view in [v for v in args.views.split(",") if v]:
            if not any(s.desktop == desktop and s.view == view for s in shapes):
                continue
            roi = E.roi_of(shapes, desktop, view)
            roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
            h, w = common.NATIVE_HW[view]
            min_area = max(30, int(round(80 * (h * w) / (1600 * 1600))))
            events, _ = E.resolve_events(con, shapes, by_frame, classes,
                                         desktop, view)
            cache: dict[int, Optional[np.ndarray]] = {}

            def frame(step: int) -> Optional[np.ndarray]:
                if step not in cache:
                    if len(cache) > 3:
                        cache.clear()
                    path = common.frame_path(con, desktop, view, step)
                    cache[step] = None if path is None else common.read_rgb(path)
                return cache[step]

            for ev in events:
                prev_img, cur_img = frame(ev.step - 1), frame(ev.step)
                if prev_img is None or cur_img is None:
                    continue
                gt = common.frame_masks(con, desktop, view, ev.step - 1
                                        ).get(ev.shape.instance)
                if gt is None or not gt.any():
                    continue
                delta = diff_delta_e(prev_img, cur_img, roi=roi,
                                     max_side=E.MAX_SIDE)
                blobs = DS.diff_blobs(delta, min_area=min_area,
                                      max_blobs=DS.MAX_PARENTS)
                cands = DS._tightest(DS._candidates(delta, blobs, min_area,
                                                    roi_area))
                if not cands:
                    continue
                window = DS._clip_roi(roi, *prev_img.shape[:2])
                gprev, gcur = DS._gradient(prev_img, window), DS._gradient(cur_img, window)
                origin = (window[0], window[1])
                gt_box = tuple(int(v) for v in ev.shape.box)
                rows = []
                for box, mask, parent in cands:
                    area = int(mask.sum())
                    strength = float(delta[box[1]:box[3], box[0]:box[2]][mask].mean())
                    point = DS._dt_point(box, mask)
                    rows.append({
                        "box": list(box), "area": area,
                        "strength": round(strength, 3),
                        "frac": area / roi_area,
                        "solidity": area / float(max(1, DS._box_area(box))),
                        "appear": round(DS._appearance(gprev, gcur, box, mask,
                                                       origin), 4),
                        "parent": parent,
                        "in_gt": int(bool(
                            0 <= point[1] < gt.shape[0]
                            and 0 <= point[0] < gt.shape[1]
                            and gt[point[1], point[0]])),
                        "box_iou": round(E.box_iou(box, gt_box), 4),
                    })
                out.append({
                    "desktop": desktop, "view": view, "step": ev.step,
                    "cls": ev.cls, "group": ev.group,
                    "label": ev.shape.label,
                    "part_size": round(ev.shape.size, 1),
                    "band": E.priors_for(ev.cls, priors),
                    "cands": rows,
                })
                print(f"D{desktop} {view} s{ev.step}: {len(rows)} candidates",
                      flush=True)
            cache.clear()
    con.close()
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    FEATURES.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {FEATURES} ({len(out)} events)")


def score(cand: dict, band, weights: dict) -> float:
    base = cand["strength"] * (max(1, cand["area"]) ** weights["area_exp"])
    prior = DS._prior_fit(cand["frac"], band) ** weights["prior"]
    appear = 1.0 + weights["appear"] * cand["appear"]
    solid = cand["solidity"] ** weights["solid"]
    return base * prior * appear * solid


def evaluate(events: list[dict], weights: dict) -> dict:
    """Hit rate *and* top-1 box IoU.

    The hit rate is the plan's headline, but on this population it is a
    coin-flip statistic: 54 % of the events are parts under 40 px across, where
    no candidate is ever right, so the whole metric rests on ~90 events and
    moves by a point per two of them.  Median **box IoU on the parts that are
    resolvable at all** is the signal that actually drives what SAM returns, so
    it is what the weights are chosen on.
    """
    per_view: dict[str, list[int]] = {}
    per_band: dict[str, list[int]] = {}
    ious: dict[str, list[float]] = {}
    big: list[float] = []
    huge: list[float] = []
    for ev in events:
        band = ev["band"]
        band = tuple(band) if band else None
        ranked = sorted(ev["cands"],
                        key=lambda c: (-score(c, band, weights), c["box"]))
        hit = int(ranked[0]["in_gt"])
        per_view.setdefault(ev["view"], []).append(hit)
        key = "<40" if ev["part_size"] < 40 else ">=40"
        per_band.setdefault(key, []).append(hit)
        ious.setdefault(ev["view"], []).append(ranked[0]["box_iou"])
        if ev["part_size"] >= 40:
            big.append(ranked[0]["box_iou"])
        if ev["part_size"] >= 100:
            huge.append(ranked[0]["box_iou"])
    out = {"per_view": {v: 100 * sum(h) / len(h) for v, h in per_view.items()},
           "per_band": {b: 100 * sum(h) / len(h) for b, h in per_band.items()},
           "box_iou": {v: statistics.median(i) for v, i in ious.items()},
           "iou40": statistics.median(big) if big else 0.0,
           "iou100": statistics.median(huge) if huge else 0.0}
    total = [h for hs in per_view.values() for h in hs]
    out["all"] = 100 * sum(total) / len(total)
    return out


def sweep(args) -> None:
    events = json.loads(FEATURES.read_text(encoding="utf-8"))
    grids = {
        "area_exp": [0.0, 0.25, 0.5, 0.75, 1.0],
        "prior": [0.0, 0.5, 1.0, 2.0],
        "appear": [0.0, 0.5, 1.0],
        "solid": [0.0, 0.5, 1.0],
    }
    results = []
    for area_exp in grids["area_exp"]:
        for prior in grids["prior"]:
            for appear in grids["appear"]:
                for solid in grids["solid"]:
                    w = {"area_exp": area_exp, "prior": prior,
                         "appear": appear, "solid": solid}
                    results.append((evaluate(events, w), w))
    results.sort(key=lambda r: -r[0]["iou40"])
    print(f"{len(results)} combinations, best by median box IoU on parts "
          f">= 40 px\n")
    header = (f"{'area^':>6} {'prior':>6} {'appr':>5} {'solid':>6} | "
              f"{'IoU>=40':>8} {'IoU>=100':>9} | "
              f"{'all':>6} {'scan':>6} {'oak1':>6} {'oak2':>6} {'rs':>6} | "
              f"{'<40':>6} {'>=40':>6}")
    print(header)
    print("-" * len(header))
    for res, w in results[: args.top]:
        pv, pb = res["per_view"], res["per_band"]
        print(f"{w['area_exp']:6.2f} {w['prior']:6.1f} {w['appear']:5.1f} "
              f"{w['solid']:6.1f} | {res['iou40']:8.3f} {res['iou100']:9.3f} | "
              f"{res['all']:6.1f} "
              f"{pv.get('scan', 0):6.1f} {pv.get('oak1', 0):6.1f} "
              f"{pv.get('oak2', 0):6.1f} {pv.get('rs', 0):6.1f} | "
              f"{pb.get('<40', 0):6.1f} {pb.get('>=40', 0):6.1f}")
    print("\nshipped defaults for comparison:")
    shipped = {"area_exp": 0.5, "prior": DS.PRIOR_WEIGHT,
               "appear": DS.APPEAR_WEIGHT, "solid": DS.SOLIDITY_WEIGHT}
    res = evaluate(events, shipped)
    print(f"  {shipped} -> all={res['all']:.1f} {res['per_view']}")
    print(f"  box IoU medians: "
          + ", ".join(f"{v}={i:.3f}" for v, i in res['box_iou'].items()))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=E.DEFAULT_DB)
    ap.add_argument("--views", default="scan,oak1,oak2,rs")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)
    if args.dump or not FEATURES.exists():
        dump(args)
    sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
