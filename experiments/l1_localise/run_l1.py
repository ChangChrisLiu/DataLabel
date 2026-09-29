"""L1 runner: every method on every one of the 201 events, one CSV row each.

::

    set TMP=D:\\DataSet\\.cache\\tmp\\l1 & set TEMP=D:\\DataSet\\.cache\\tmp\\l1
    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l1_localise.run_l1 --tag all

Two ground truths are scored side by side, and they must never be mixed up:

* ``A`` -- B2's, **verbatim**: the draft whose *key* disappears between ``k-1``
  and ``k``.  This is the ground truth the task asks for.
* ``B`` -- the draft at ``k-1`` whose *location* has no same-class partner at
  ``k`` (``probe_gt.py``).  ``ls:`` ordinals are renumbered per frame, so for a
  class with siblings the key that disappears is always the last ordinal, not
  the part that left; ``B`` fixes that for 184 of the 201 events and is blank
  for the other 17.

Nothing is tuned here: the runner records every candidate's box and point, and
``analyse.py`` derives M1 (a gate on M0) and the M4 policies from these rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

from experiments.l1_localise import base  # noqa: E402
from experiments.l1_localise import methods as M  # noqa: E402
from experiments.l1_localise.siblings import (  # noqa: E402
    PrevParts, Priors, known_boxes_at, sibling_pool,
)
from experiments.plan_b_probe import diff_eval as E  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402

C = M.M3Config
#: name -> (config, dE source).  dE source: None (M3, appearance only),
#: "reg" (M2e's registered map, M4), "raw" (M0's unregistered map).
VARIANTS: dict[str, tuple[M.M3Config, Optional[str]]] = {
    # ---- M3 as specified: NCC in j minus NCC in k ------------------------
    "M3": (C(), None),
    "M3_pad0": (C(pad_frac=0.0), None),
    "M3_grad": (C(feats=("grad",)), None),
    "M3_color": (C(feats=("color",)), None),
    "M3_gg": (C(feats=("gray", "grad")), None),
    "M3_1t": (C(n_templates=1), None),
    "M3_sup": (C(suppress_known=True), None),
    "M3_t16": (C(target_side=16.0), None),
    # ---- M3 with an appearance-change term instead of / on top of NCC_k --
    "M3z": (C(k_weight=0.0, zncc_weight=1.0), None),
    "M3kz": (C(k_weight=1.0, zncc_weight=1.0), None),
    "M3j": (C(k_weight=0.0), None),                # appearance in j only
    # ---- M4: boosted by / restricted to the registered dE map -------------
    "M4_b05": (C(k_weight=0.0, de_weight=0.5), "reg"),
    "M4_b1": (C(k_weight=0.0, de_weight=1.0), "reg"),
    "M4_b2": (C(k_weight=0.0, de_weight=2.0), "reg"),
    "M4k_b1": (C(k_weight=1.0, de_weight=1.0), "reg"),
    "M4z_b1": (C(k_weight=0.0, zncc_weight=1.0, de_weight=1.0), "reg"),
    "M4_gate": (C(k_weight=1.0, de_gate=6.0), "reg"),
    "M4_b1_t16": (C(k_weight=0.0, de_weight=1.0, target_side=16.0), "reg"),
    "M4_b1_pad0": (C(k_weight=0.0, de_weight=1.0, pad_frac=0.0), "reg"),
    "M4_b1_sup": (C(k_weight=0.0, de_weight=1.0, suppress_known=True), "reg"),
    "M4_b1_raw": (C(k_weight=0.0, de_weight=1.0), "raw"),
    # ---- second round: a size-matched (centre-surround) dE term ------------
    "M4_cs_b1": (C(k_weight=0.0, de_weight=1.0, de_mode="cs"), "reg"),
    "M4_cs_b2": (C(k_weight=0.0, de_weight=2.0, de_mode="cs"), "reg"),
    "M4k_cs_b1": (C(k_weight=1.0, de_weight=1.0, de_mode="cs"), "reg"),
    # ablation: the dE term alone, at the sibling's size (no appearance)
    "M4_cs_only": (C(ncc_weight=0.0, k_weight=0.0, de_weight=1.0,
                     de_mode="cs"), "reg"),
    # ---- third round: siblings match *themselves* in j (they are still
    # there), so appearance must skip what is already drawn, and "gone" is
    # judged per candidate at native resolution (two-stage) ----------------
    "M3js": (C(k_weight=0.0, suppress_known=True), None),
    "M3js_rr10": (C(k_weight=0.0, suppress_known=True, rerank=10), None),
    "M3js_rr20": (C(k_weight=0.0, suppress_known=True, rerank=20), None),
    "M4j_rr10": (C(k_weight=0.0, rerank=10), "reg"),
    "M4js_rr10": (C(k_weight=0.0, suppress_known=True, rerank=10), "reg"),
    "M4js_rr20": (C(k_weight=0.0, suppress_known=True, rerank=20), "reg"),
    "M4js_rr10_a05": (C(k_weight=0.0, suppress_known=True, rerank=10,
                        rerank_app=0.5), "reg"),
    "M4js_b1": (C(k_weight=0.0, suppress_known=True, de_weight=1.0), "reg"),
    "M4js_b1_rr10": (C(k_weight=0.0, suppress_known=True, de_weight=1.0,
                       rerank=10), "reg"),
    # ---- fourth round: + the sequence prior (near the part removed at k+1,
    # which the walk drew one frame ago; same class only) -------------------
    "M3js_p": (C(k_weight=0.0, suppress_known=True, prior_weight=1.0), None),
    "M4js_b1_p": (C(k_weight=0.0, suppress_known=True, de_weight=1.0,
                    prior_weight=1.0), "reg"),
    "M4js_b1_p05": (C(k_weight=0.0, suppress_known=True, de_weight=1.0,
                      prior_weight=0.5), "reg"),
    "M4js_b1_p_s10": (C(k_weight=0.0, suppress_known=True, de_weight=1.0,
                        prior_weight=1.0, prior_sigma=0.10), "reg"),
    "M4js_rr10_p": (C(k_weight=0.0, suppress_known=True, rerank=10,
                      prior_weight=1.0), "reg"),
    "M4js_b1_rr10_p": (C(k_weight=0.0, suppress_known=True, de_weight=1.0,
                         rerank=10, prior_weight=1.0), "reg"),
    # ---- fifth round: the dE map in the app's argument order (k, j) --------
    "M4_b1_app": (C(k_weight=0.0, de_weight=1.0), "reg_app"),
    "M4js_b1_app": (C(k_weight=0.0, suppress_known=True, de_weight=1.0),
                    "reg_app"),
    "M4js_b1_p_app": (C(k_weight=0.0, suppress_known=True, de_weight=1.0,
                        prior_weight=1.0), "reg_app"),
}
#: M5: no sibling, the class's median size (other desktops):
#: name -> (dE source, filter, prior weight, prior on any class?)
M5_VARIANTS = {
    "M5cs": ("reg", "cs", 0.0, False),
    "M5cs_raw": ("raw", "cs", 0.0, False),
    "M5cs_p": ("reg", "cs", 1.0, False),
    "M5cs_pany": ("reg", "cs", 1.0, True),
    "M5cs_p_app": ("reg_app", "cs", 1.0, False),
}


def score_props(props, gt: Optional[base.GtGeom], gt_box, prefix: str) -> dict:
    """Point-in-part / distance / box IoU of a ranked list against one GT."""
    out: dict = {}
    if gt is None:
        return out
    hits = [base.point_in(gt.mask, p.point) for p in props]
    out[f"{prefix}_in1"] = int(bool(hits[:1]) and hits[0])
    out[f"{prefix}_in3"] = int(any(hits[:3]))
    out[f"{prefix}_inall"] = int(any(hits))
    if props:
        p = props[0]
        d = gt.dist(p.point)
        out[f"{prefix}_dist1"] = round(d, 1)
        out[f"{prefix}_dist1n"] = round(d / gt.size, 3)
        out[f"{prefix}_cdist1n"] = round(gt.centre_dist(p.point) / gt.size, 3)
        out[f"{prefix}_iou1"] = round(base.box_iou(p.box, gt_box), 4)
        out[f"{prefix}_touch1"] = int(gt.box_touches(p.box))
        out[f"{prefix}_iou3"] = round(max(base.box_iou(q.box, gt_box)
                                          for q in props[:3]), 4)
        out[f"{prefix}_dist3n"] = round(min(gt.dist(q.point) for q in props[:3])
                                        / gt.size, 3)
    return out


def props_json(props, n: int = 5) -> str:
    return json.dumps([[list(map(int, p.box)), [round(p.point[0], 1),
                                                 round(p.point[1], 1)],
                        round(float(p.score), 4), int(p.area)]
                       for p in props[:n]])


def gt_change(img_j, img_k, box, pad: int = 10) -> float:
    """Median native dE inside a GT box between j and k (1 px tolerance)."""
    h, w = img_j.shape[:2]
    x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
    x1, y1 = min(w, box[2] + pad), min(h, box[3] + pad)
    d = diff_delta_e(img_j[y0:y1, x0:x1], img_k[y0:y1, x0:x1], blur=3)
    inner = d[box[1] - y0:box[3] - y0, box[0] - x0:box[2] - x0]
    return float(np.median(inner)) if inner.size else float("nan")


def run(args) -> list[dict]:
    events, shapes, by_frame, con = base.load_events()
    loc = {}
    with (base.OUT / "gt_check.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            loc[(int(r["desktop"]), r["view"], int(r["step"]))] = r["loc_instance"]
    priors = Priors(shapes)
    prev_parts = PrevParts(con, by_frame)
    want_views = set(args.views.split(","))
    want_desk = {int(d) for d in args.desktops.split(",")}
    events = [e for e in events if e.view in want_views and e.desktop in want_desk]
    if args.only:
        keys = {tuple(x.split(":")) for x in args.only.split(",")}
        events = [e for e in events
                  if (str(e.desktop), e.view, str(e.step)) in keys]
    if args.limit:
        events = events[: args.limit]
    wanted = set(args.methods.split(","))
    var_names = [n for n in VARIANTS if n in wanted or "all" in wanted]
    want_app = ("all" in wanted or "M2e_app" in wanted
                or any(VARIANTS[n][1] == "reg_app" for n in var_names)
                or any(M5_VARIANTS[n][0] == "reg_app" for n in M5_VARIANTS
                       if n in wanted))

    rows: list[dict] = []
    frames: Optional[base.Frames] = None
    cur = None
    t_start = time.perf_counter()
    for n, ev in enumerate(events):
        if cur != (ev.desktop, ev.view):
            cur = (ev.desktop, ev.view)
            frames = base.Frames(con, ev.desktop, ev.view, keep=8)
        t0 = time.perf_counter()
        img_j = frames.get(ev.j)
        img_k = frames.get(ev.step)
        load_s = time.perf_counter() - t0
        if img_j is None or img_k is None:
            print(f"  skip {ev.key}: frame missing")
            continue
        masks = common.frame_masks(con, ev.desktop, ev.view, ev.j)
        gA = masks.get(ev.instance)
        loc_key = loc.get((ev.desktop, ev.view, ev.step)) or ""
        gB = masks.get(loc_key) if loc_key else None
        geoA = base.GtGeom(gA)
        geoB = base.GtGeom(gB) if gB is not None and gB.any() else None
        boxA = ev.gt_box
        boxB = None
        if loc_key:
            sB = by_frame[(ev.desktop, ev.view, ev.j)][loc_key]
            boxB = tuple(int(v) for v in sB.box)

        bands = priors.bands(ev)
        pool_step, pool = sibling_pool(ev, by_frame)
        known = known_boxes_at(ev, by_frame)
        prev = prev_parts.get(ev)
        prev_any = prev_same = None
        if prev is not None:
            pb, pcls = prev
            prev_any = (0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3]))
            prev_same = prev_any if pcls == ev.cls else None
        common_row = {
            "desktop": ev.desktop, "view": ev.view, "step": ev.step,
            "frame_j": ev.j, "target": ev.target, "verb": ev.verb,
            "cls": ev.cls, "group": ev.group, "label": ev.label,
            "instance": ev.instance, "loc_instance": loc_key,
            "gtB_same": int(loc_key == ev.instance) if loc_key else "",
            "part_area": ev.part_area, "part_width": round(ev.part_width, 1),
            "bucket": ev.bucket,
            "partB_width": round(geoB.size, 1) if geoB else "",
            "bucketB": base.bucket_of(geoB.size) if geoB else "",
            "roi": json.dumps(list(ev.roi)),
            "roi_area": (ev.roi[2] - ev.roi[0]) * (ev.roi[3] - ev.roi[1]),
            "band_ship": json.dumps(bands["ship"]) if bands["ship"] else "",
            "band_lodo": json.dumps(bands["lodo"]) if bands["lodo"] else "",
            "sib_step": pool_step if pool_step is not None else "",
            "sib_gap": (pool_step - ev.step) if pool_step is not None else "",
            "sib_n": len(pool),
            "sib_labels": "|".join(sorted({s.label for s in pool})),
            "prev_known": int(prev is not None),
            "prev_same_cls": int(prev_same is not None),
            # diagnostic only (uses the GT): a stale LS draft on frame k can
            # make the "part removed at k+1" be this event's own part
            "prev_is_gtA": int(prev is not None
                               and base.box_iou(prev[0], boxA) >= 0.3),
            "prev_is_gtB": int(prev is not None and boxB is not None
                               and base.box_iou(prev[0], boxB) >= 0.3),
            "load_s": round(load_s, 3),
        }
        if args.gt_change:
            common_row["gtA_dE"] = round(gt_change(img_j, img_k, boxA), 2)
            common_row["gtB_dE"] = (round(gt_change(img_j, img_k, boxB), 2)
                                    if boxB else "")

        def emit(method: str, props, secs: float, info: Optional[dict] = None):
            row = dict(common_row)
            row["method"] = method
            row["n_props"] = len(props)
            row["secs"] = round(secs, 4)
            if props:
                row["p1_area"] = props[0].area
                row["p1_box"] = json.dumps(list(map(int, props[0].box)))
                row["p1_boxarea"] = ((props[0].box[2] - props[0].box[0])
                                     * (props[0].box[3] - props[0].box[1]))
                row["p1_score"] = round(float(props[0].score), 4)
            row.update(score_props(props, geoA, boxA, "A"))
            row.update(score_props(props, geoB, boxB, "B"))
            row["props"] = props_json(props)
            for k, v in (info or {}).items():
                row[k] = round(v, 4) if isinstance(v, float) else v
            rows.append(row)

        # ---- M0 (B2 baseline) and relatives --------------------------------
        t0 = time.perf_counter()
        p0, delta, blobs = M.m0(img_j, img_k, ev.roi, ev.min_area)
        m0_s = time.perf_counter() - t0
        emit("M0", p0, m0_s)
        t0 = time.perf_counter()
        emit("M0p", M.blobs_to_props(diff_blobs(delta)),
             m0_s + time.perf_counter() - t0 - 0.0)
        # The app's own call: AssistController.compute passes the neighbour
        # (k) first and the frame on screen (j) second, and diff_blobs keeps
        # its default min_area=80.  diff_delta_e is *not* symmetric (the
        # shift tolerance and the blur act on the second frame only), so this
        # is not B2's M0.
        t0 = time.perf_counter()
        delta_app = diff_delta_e(img_k, img_j, roi=ev.roi, max_side=M.MAX_SIDE)
        emit("M0app", M.blobs_to_props(diff_blobs(delta_app)),
             time.perf_counter() - t0)
        if "split" in args.methods or "all" in args.methods:
            t0 = time.perf_counter()
            sp = E.split_proposals(img_j, img_k, ev.roi, delta=delta,
                                   min_area=ev.min_area, max_proposals=5,
                                   expect_area=bands["ship_frac"])
            emit("split", [M.Prop(box=p.box, point=p.point, score=p.score,
                                  area=p.area) for p in sp],
                 m0_s + time.perf_counter() - t0)

        # ---- M2: register, then the same dE map -----------------------------
        reg = {}
        kreg = {}
        for mode, name, app in (("none", "M0c", False), ("phase", "M2p", False),
                                ("ecc", "M2e", False), ("ecc", "M2e_app", True)):
            if app and not want_app:
                continue
            t0 = time.perf_counter()
            props, d2, origin, info, kreg[name] = M.m2(
                img_j, img_k, ev.roi, 80 if app else ev.min_area, mode,
                app_order=app)
            secs = time.perf_counter() - t0
            info = {f"reg_{k}": v for k, v in info.items()}
            info["n_blobs"] = len(diff_blobs(d2, min_area=ev.min_area,
                                             max_blobs=400))
            emit(name, props, secs, info)
            reg[name] = (d2, origin, secs)

        # ---- M5: compact change of the class's size (every event) -----------
        side = priors.median_side(ev)
        for name, (dsrc, mode, pw, any_cls) in M5_VARIANTS.items():
            if not (name in wanted or "all" in wanted) or side is None:
                continue
            d_map, d_org, extra_s = (reg["M2e"] if dsrc == "reg" else
                                     reg["M2e_app"] if dsrc == "reg_app" else
                                     (delta, (0, 0), m0_s))
            pp = prev_any if any_cls else prev_same
            props, info = M.m5(d_map, d_org, ev.roi, side, mode=mode,
                               prior_point=pp, prior_weight=pw)
            info["m5_side"] = round(side, 1)
            emit(name, props, info["search_s"] + extra_s, info)

        # ---- M3 / M4: sibling appearance ------------------------------------
        src = frames.get(pool_step) if pool else None
        if src is not None and var_names:
            med = max(1.0, float(np.median([q.area for q in pool])))
            ordered = sorted(pool, key=lambda s: abs(math.log(max(1, s.area) / med)))
            temps_by_pad: dict[float, list] = {}
            for name in var_names:
                cfg, dsrc = VARIANTS[name]
                if cfg.pad_frac not in temps_by_pad:
                    temps_by_pad[cfg.pad_frac] = [
                        t for t in (M.cut_template(src, tuple(s.box), pool_step,
                                                   s.instance, pad_frac=cfg.pad_frac)
                                    for s in ordered) if t is not None]
                temps = temps_by_pad[cfg.pad_frac]
                if not temps:
                    continue
                d_map, d_org, extra_s = None, (0, 0), 0.0
                k_reg = None
                if dsrc in ("reg", "reg_app"):
                    rname = "M2e" if dsrc == "reg" else "M2e_app"
                    d_map, d_org, extra_s = reg[rname]
                    k_reg = kreg[rname]
                elif dsrc == "raw":
                    d_map, d_org, extra_s = delta, (0, 0), m0_s
                props, info = M.m3(img_j, img_k, ev.roi, temps, cfg,
                                   known_boxes=known, delta=d_map,
                                   delta_origin=d_org, k_reg=k_reg,
                                   k_origin=d_org, prior_point=prev_same)
                info["search_s"] = info.pop("m3_s")
                emit(name, props, info["search_s"] + extra_s, info)
        if (n + 1) % 10 == 0:
            el = time.perf_counter() - t_start
            print(f"  {n + 1}/{len(events)} events, {el:.0f} s", flush=True)
    con.close()
    return rows


def write(rows: list[dict], tag: str) -> Path:
    base.OUT.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path = base.OUT / f"events_{tag}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", default="scan,oak1,oak2,rs")
    ap.add_argument("--desktops", default="13,24,33")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--only", default="", help="d:view:step,... events")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt-change", action="store_true")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args(argv)
    rows = run(args)
    path = write(rows, args.tag)
    print(f"wrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
