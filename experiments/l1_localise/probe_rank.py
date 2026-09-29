"""Why does the sibling search miss?  Where the true part ranks under each cue.

For every sibling-covered event with a location-resolved GT (GT-B), this asks
each cue for its top 200 peaks and records the rank of the first peak that
lands on the part (point inside the GT-B box grown by 50 %):

* ``app``  -- NCC with the sibling templates in frame j, already-drawn
  same-class parts suppressed (M3js);
* ``de``   -- the registered dE map at the sibling's size (centre-surround);
* ``both`` -- NCC + dE (M4js_b1).

And one oracle test of the two-stage idea: if the candidate set were exactly
the true same-class parts in frame j (the removed one plus the ones still
there), would "the one whose patch changed most" pick the removed one?
Writes ``rank_probe.csv`` and prints a summary per view and width.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l1_localise import base  # noqa: E402
from experiments.l1_localise import methods as M  # noqa: E402
from experiments.l1_localise.siblings import known_boxes_at, sibling_pool  # noqa: E402

CUES = {
    "app": M.M3Config(k_weight=0.0, suppress_known=True, top=200),
    "de": M.M3Config(ncc_weight=0.0, k_weight=0.0, de_weight=1.0, de_mode="cs",
                     top=200),
    "both": M.M3Config(k_weight=0.0, suppress_known=True, de_weight=1.0, top=200),
}


def rank_of(props, box) -> int:
    x0, y0, x1, y1 = box
    gx, gy = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    for i, p in enumerate(props):
        if x0 - gx <= p.point[0] <= x1 + gx and y0 - gy <= p.point[1] <= y1 + gy:
            return i
    return -1


def main() -> int:
    events, _shapes, by_frame, con = base.load_events()
    loc = {(int(r["desktop"]), r["view"], int(r["step"])): r["loc_instance"]
           for r in csv.DictReader((base.OUT / "gt_check.csv").open(encoding="utf-8"))}
    rows = []
    frames, cur = None, None
    for ev in events:
        lk = loc.get((ev.desktop, ev.view, ev.step))
        if not lk:
            continue
        s, pool = sibling_pool(ev, by_frame)
        if not pool:
            continue
        if cur != (ev.desktop, ev.view):
            cur = (ev.desktop, ev.view)
            frames = base.Frames(con, *cur, keep=6)
        ij, ik, src = frames.get(ev.j), frames.get(ev.step), frames.get(s)
        gt = by_frame[(ev.desktop, ev.view, ev.j)][lk]
        _p, d2, org, _i, ck = M.m2(ij, ik, ev.roi, ev.min_area, "ecc")
        med = max(1.0, float(np.median([q.area for q in pool])))
        ordered = sorted(pool, key=lambda q: abs(math.log(max(1, q.area) / med)))
        temps = [t for t in (M.cut_template(src, tuple(q.box), s, q.instance)
                             for q in ordered) if t is not None]
        known = known_boxes_at(ev, by_frame)
        row = {"desktop": ev.desktop, "view": ev.view, "step": ev.step,
               "label": ev.label, "widthB": round(math.sqrt(gt.area), 1),
               "bucketB": base.bucket_of(math.sqrt(gt.area)), "n_temps": len(temps)}
        for name, cfg in CUES.items():
            props, _ = M.m3(ij, ik, ev.roi, temps, cfg, known_boxes=known,
                            delta=d2, delta_origin=org)
            row[f"rank_{name}"] = rank_of(props, gt.box)
            row[f"npk_{name}"] = len(props)
        # oracle two-stage: the true same-class parts of frame j
        same_j = [q for q in by_frame[(ev.desktop, ev.view, ev.j)].values()
                  if q.cls == ev.cls]
        ch = {q.instance: M.box_change(ij, ck, org, tuple(q.box)) for q in same_j}
        row["n_same_j"] = len(same_j)
        row["change_gt"] = round(ch[lk], 2)
        others = [v for k, v in ch.items() if k != lk]
        row["change_others_max"] = round(max(others), 2) if others else ""
        row["oracle_2stage_hit"] = int(not others or ch[lk] > max(others))
        rows.append(row)
    con.close()
    path = base.OUT / "rank_probe.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} sibling-covered events with GT-B")
    for b in [x[0] for x in base.BUCKETS]:
        sel = [r for r in rows if r["bucketB"] == b]
        if not sel:
            continue
        line = [f"{b:>8} n={len(sel):3d}"]
        for name in CUES:
            rk = np.array([r[f"rank_{name}"] for r in sel])
            found = rk >= 0
            line.append(f"{name}: top1 {100 * np.mean(rk == 0):4.1f}% top10 "
                        f"{100 * np.mean(found & (rk < 10)):4.1f}% top200 "
                        f"{100 * np.mean(found):4.1f}% med rank "
                        f"{np.median(np.where(found, rk, 999)):.0f}")
        multi = [r for r in sel if r["n_same_j"] > 1]
        line.append(f"oracle 2-stage (>1 same-class part): "
                    f"{100 * np.mean([r['oracle_2stage_hit'] for r in multi]):.1f}% "
                    f"of {len(multi)}")
        print("\n    ".join(line))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
