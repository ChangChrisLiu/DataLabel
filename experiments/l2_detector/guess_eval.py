"""The reverse-walk guess with detector candidates, on L1's events (GT-B).

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.guess_eval

Population: B2's pinned 201 events (``experiments.l1_localise.base``), scored
against L1's position-resolved ground truth GT-B (``probe_gt.check``; 184
events resolve).  Every event is on D13, D24 or D33, so every event is scored
by the fold that held its desktop out -- nothing is scored by a model that saw
its desktop.

The guess on frame ``j`` (step ``k = j + 1`` removed the part):

1. candidates = the fold's detections in frame ``j`` of the step's class
   (``conf >= CONF``);
2. minus parts already drawn: a candidate is dropped when its centre lies in,
   or its box overlaps (IoU >= 0.3), a same-class draft on frame ``k`` -- the
   walk has drawn those (L1's ``known_boxes_at``);
3. ranked by native-resolution mean dE between ``j`` and ``k`` inside the box
   (L1's ``box_change``, 1 px shift tolerance);
4. sequence prior as tie-breaker only: among candidates within ``EPS`` dE of
   the best, the one nearest the part removed at ``k + 1`` (drawn on frame
   ``k`` just before; L1's ``PrevParts``) goes first;
5. the armed point is the box centre; no candidate left -> today's guess
   (``M0app``: the app's own ``diff_delta_e(k, j)`` + ``diff_blobs``).

Policies: ``L2small`` (pre-declared primary) uses the detector for the six
small classes only and M0app for every other class; ``L2all`` also for
``ram_module`` / ``cpu``.  Variants (``nop`` no prior, ``nosup`` no
suppression, ``c05`` / ``c25`` other thresholds) are sensitivity checks.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402

from experiments.l1_localise import base  # noqa: E402
from experiments.l1_localise import methods as M  # noqa: E402
from experiments.l1_localise.probe_gt import check as gt_check  # noqa: E402
from experiments.l1_localise.siblings import PrevParts, known_boxes_at  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402
from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

DETS = env.OUT / "detections"
CONF = 0.10
EPS = 0.5
SUP_IOU = 0.3
FOLD_OF = {13: "hold13", 24: "hold24", 33: "hold33"}
L1_OUT = Path("D:/DataSet/experiments_out/l1_localise")


def setup_l1():
    """Point L1's harness at the L2 copy of the database (read-only)."""
    base.DB = D.DB
    base.TMP = env.TMP
    return base.load_events()


def load_dets() -> dict[str, list]:
    out = {}
    for fold in FOLD_OF.values():
        p = DETS / f"{fold}.json"
        if p.exists():
            out.update(json.loads(p.read_text(encoding="utf-8")))
    return out


def suppressed(box, known, iou_thr: float = SUP_IOU) -> bool:
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    for kb in known:
        if kb[0] - 2 <= cx < kb[2] + 2 and kb[1] - 2 <= cy < kb[3] + 2:
            return True
        if base.box_iou(box, kb) >= iou_thr:
            return True
    return False


def rank(cands: list[dict], prev_pt, eps: float = EPS, prior: bool = True) -> list[dict]:
    order = sorted(cands, key=lambda c: -c["change"])
    if prior and prev_pt is not None and len(order) >= 2:
        best = order[0]["change"]
        tie = [c for c in order if c["change"] >= best - eps]
        rest = [c for c in order if c["change"] < best - eps]
        tie.sort(key=lambda c: math.hypot(c["pt"][0] - prev_pt[0], c["pt"][1] - prev_pt[1]))
        order = tie + rest
    return order


def l1_reference() -> dict:
    """L1's per-event M0app / M3js_rr10 (GT-B) top-1 and top-3."""
    ref: dict = {}
    import pandas as pd

    for v, meths in (("v3", ("M3js_rr10", "M0")), ("v4", ("M0app",))):
        d = pd.read_csv(L1_OUT / f"events_{v}.csv",
                        usecols=["desktop", "view", "step", "method", "B_in1", "B_in3"])
        for _, r in d[d.method.isin(meths)].iterrows():
            # r["view"], not r.view: Series.view is a method
            k = (int(r["desktop"]), str(r["view"]), int(r["step"]))
            ref.setdefault(k, {})[r["method"]] = (r["B_in1"], r["B_in3"])
    return ref


def main() -> int:
    t_all = time.perf_counter()
    events, shapes, by_frame, con = setup_l1()
    gtb = {(r["desktop"], r["view"], r["step"]): r["loc_instance"]
           for r in gt_check(events, by_frame)}
    # the same GT-B L1 used
    with (L1_OUT / "gt_check.csv").open(encoding="utf-8") as fh:
        l1_loc = {(int(r["desktop"]), r["view"], int(r["step"])): r["loc_instance"]
                  for r in csv.DictReader(fh)}
    diff = [k for k in gtb if gtb[k] != l1_loc.get(k)]
    if diff:
        raise SystemExit(f"GT-B differs from L1's on {len(diff)} events: {diff[:5]}")
    print(f"[l2] {len(events)} events; GT-B resolves {sum(1 for v in gtb.values() if v)}"
          " (identical to L1's gt_check.csv)")
    dets = load_dets()
    prev_parts = PrevParts(con, by_frame)
    ref = l1_reference()

    variants = {
        "L2small": dict(classes=D.SMALL_CLASSES, conf=CONF, prior=True, sup=True),
        "L2all": dict(classes=D.DET_CLASSES, conf=CONF, prior=True, sup=True),
        "L2small_nop": dict(classes=D.SMALL_CLASSES, conf=CONF, prior=False, sup=True),
        "L2small_nosup": dict(classes=D.SMALL_CLASSES, conf=CONF, prior=True, sup=False),
        "L2small_c05": dict(classes=D.SMALL_CLASSES, conf=0.05, prior=True, sup=True),
        "L2small_c25": dict(classes=D.SMALL_CLASSES, conf=0.25, prior=True, sup=True),
    }
    rows = []
    frames = None
    cur = None
    for ev in events:
        key3 = (ev.desktop, ev.view, ev.step)
        loc = gtb.get(key3) or ""
        if not loc:
            continue
        if cur != (ev.desktop, ev.view):
            cur = (ev.desktop, ev.view)
            frames = base.Frames(con, ev.desktop, ev.view, keep=6)
        img_j = frames.get(ev.j)
        img_k = frames.get(ev.step)
        if img_j is None or img_k is None:
            print(f"  skip {key3}: frame missing")
            continue
        masks = common.frame_masks(con, ev.desktop, ev.view, ev.j)
        gmask = masks[loc]
        geo = base.GtGeom(gmask)
        sB = by_frame[(ev.desktop, ev.view, ev.j)][loc]
        # today's guess, the app's own call
        t0 = time.perf_counter()
        delta_app = diff_delta_e(img_k, img_j, roi=ev.roi, max_side=M.MAX_SIDE)
        m0 = M.blobs_to_props(diff_blobs(delta_app))
        m0_s = time.perf_counter() - t0
        m0_hits = [base.point_in(gmask, p.point) for p in m0]
        known = known_boxes_at(ev, by_frame)
        prev = prev_parts.get(ev)
        prev_pt = None
        if prev is not None and prev[1] == ev.cls:
            pb = prev[0]
            prev_pt = (0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3]))
        drow = dets.get(f"{ev.desktop}:{ev.view}:{ev.j}")
        row = {
            "ev": f"{ev.desktop}:{ev.view}:{ev.step}", "desktop": ev.desktop,
            "view": ev.view, "step": ev.step, "frame_j": ev.j, "cls": ev.cls,
            "target": ev.target, "loc_instance": loc,
            "partB_width": round(geo.size, 1), "bucketB": base.bucket_of(geo.size),
            "gtB_box": json.dumps(list(map(int, sB.box))),
            "n_known": len(known), "prev_same": int(prev_pt is not None),
            "dets_available": int(drow is not None),
            "M0app_in1": int(bool(m0_hits[:1]) and m0_hits[0]),
            "M0app_in3": int(any(m0_hits[:3])),
            "M0app_s": round(m0_s, 3),
            "M0app_p1": json.dumps([round(v, 1) for v in m0[0].point]) if m0 else "",
        }
        r1 = ref.get(key3, {})
        row["L1_M0app_in1"] = r1.get("M0app", (np.nan, np.nan))[0]
        row["L1_M0app_in3"] = r1.get("M0app", (np.nan, np.nan))[1]
        m3 = r1.get("M3js_rr10")
        m0b = r1.get("M0", (np.nan, np.nan))
        row["L1_M3js_in1"] = m3[0] if m3 is not None else m0b[0]
        row["L1_M3js_in3"] = m3[1] if m3 is not None else m0b[1]

        # every same-class detection, once, with its change score
        allc = []
        t0 = time.perf_counter()
        for r in (drow or []):
            if r[4] != ev.cls or r[5] < 0.01:
                continue
            box = tuple(int(round(v)) for v in r[:4])
            allc.append({"box": box, "conf": float(r[5]),
                         "pt": (0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])),
                         "sup": suppressed(box, known)})
        # ceilings (no ranking): is the removed part among the candidates?
        for thr, name in ((0.01, "c01"), (0.05, "c05"), (CONF, "c10"), (0.25, "c25")):
            pool = [c for c in allc if c["conf"] >= thr]
            row[f"ceil_{name}"] = int(any(base.point_in(gmask, c["pt"]) for c in pool))
            row[f"ceilsup_{name}"] = int(any(base.point_in(gmask, c["pt"])
                                             for c in pool if not c["sup"]))
            row[f"ncand_{name}"] = len(pool)
            row[f"nkept_{name}"] = sum(1 for c in pool if not c["sup"])
        for c in allc:
            if c["conf"] >= 0.05:
                c["change"] = M.box_change(img_j, img_k, (0, 0), c["box"])
        rank_s = time.perf_counter() - t0
        row["rank_s"] = round(rank_s, 4)

        for name, cfg in variants.items():
            use_det = ev.cls in cfg["classes"] and drow is not None
            order = []
            if use_det:
                pool = [c for c in allc if c["conf"] >= cfg["conf"]
                        and (not cfg["sup"] or not c["sup"])]
                for c in pool:
                    if "change" not in c:
                        c["change"] = M.box_change(img_j, img_k, (0, 0), c["box"])
                order = rank(pool, prev_pt, prior=cfg["prior"])
            if order:
                hits = [base.point_in(gmask, c["pt"]) for c in order]
                src = "det"
                p1 = order[0]["pt"]
            else:
                hits = m0_hits
                src = "M0app"
                p1 = m0[0].point if m0 else None
            row[f"{name}_in1"] = int(bool(hits[:1]) and hits[0])
            row[f"{name}_in3"] = int(any(hits[:3]))
            row[f"{name}_src"] = src
            row[f"{name}_n"] = len(order)
            row[f"{name}_p1"] = json.dumps([round(p1[0], 1), round(p1[1], 1)]) if p1 else ""
            row[f"{name}_dist1"] = round(geo.dist(p1), 1) if p1 else ""
            if name == "L2small":
                true_rank = next((i + 1 for i, h in enumerate(hits) if h), 0)
                row["L2small_true_rank"] = true_rank if src == "det" else ""
                row["L2small_top"] = json.dumps([[list(c["box"]), round(c["conf"], 3),
                                                  round(c["change"], 2)]
                                                 for c in order[:5]])
        rows.append(row)
        print(f"  {row['ev']:>14} {ev.cls:>16} {row['bucketB']:>7} "
              f"M0app={row['M0app_in1']} L2={row['L2small_in1']}/{row['L2small_in3']} "
              f"ceil={row['ceilsup_c10']} n={row['L2small_n']}", flush=True)
    con.close()
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path = env.OUT / "guess_events.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path} in {time.perf_counter() - t_all:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
