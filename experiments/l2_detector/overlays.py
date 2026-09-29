"""Overlay sheets of chosen events: frame j | frame k, around the removed part.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.overlays 13:scan:38 24:oak1:12 ...

Drawn on frame ``j`` (left) and ``k`` (right), both cropped around the
removed part and the top candidate and upscaled for reading:

* green outline -- the removed part (GT-B draft mask);
* yellow boxes  -- the step class's detections (conf >= 0.10) the guess keeps,
  with their dE; grey boxes -- those skipped as already drawn;
* red box       -- the L2 top-1 (cyan when it is on the part);
* magenta cross -- today's guess (M0app top-1 point);
* blue cross    -- the part removed at k + 1 (sequence prior), when known.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402
from experiments.l2_detector.guess_eval import (  # noqa: E402
    CONF, load_dets, setup_l1, suppressed,
)
from experiments.l1_localise import base  # noqa: E402
from experiments.l1_localise import methods as M  # noqa: E402
from experiments.l1_localise.siblings import PrevParts, known_boxes_at  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402

OUTDIR = env.OUT / "overlays"
CELL = 560


def _label(img, text, org, col, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def sheet(ev, row, img_j, img_k, gmask, dets, known, prev_pt) -> np.ndarray:
    p_l2 = json.loads(row["L2small_p1"]) if isinstance(row["L2small_p1"], str) else None
    p_m0 = json.loads(row["M0app_p1"]) if isinstance(row["M0app_p1"], str) else None
    ys, xs = np.nonzero(gmask)
    gc = (xs.mean(), ys.mean())
    pts = [gc]
    if p_l2:
        pts.append(tuple(p_l2))
    w = max(xs.max() - xs.min(), ys.max() - ys.min())
    # today's guess is shown when it is not too far off
    if p_m0 and np.hypot(p_m0[0] - gc[0], p_m0[1] - gc[1]) < 25 * max(w, 20):
        pts.append(tuple(p_m0))
    xs_all = [p[0] for p in pts]
    ys_all = [p[1] for p in pts]
    margin = max(60, 3 * w)
    side = int(max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all)) + 2 * margin)
    H, W = img_j.shape[:2]
    side = min(side, H, W)
    cx, cy = 0.5 * (max(xs_all) + min(xs_all)), 0.5 * (max(ys_all) + min(ys_all))
    x0 = int(min(max(0, cx - side / 2), W - side))
    y0 = int(min(max(0, cy - side / 2), H - side))
    x1, y1 = x0 + side, y0 + side
    s = CELL / float(side)

    def crop(img):
        c = img[y0:y1, x0:x1]
        return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)

    def P(x, y):
        return int(round((x - x0) * s)), int(round((y - y0) * s))

    L = cv2.cvtColor(crop(img_j), cv2.COLOR_RGB2BGR)
    R = cv2.cvtColor(crop(img_k), cv2.COLOR_RGB2BGR)
    m = cv2.resize(gmask[y0:y1, x0:x1].astype(np.uint8), (L.shape[1], L.shape[0]),
                   interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for img in (L, R):
        cv2.drawContours(img, cnts, -1, (0, 0, 0), 4)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    for d in dets:
        b = d["box"]
        col = (140, 140, 140) if d["sup"] else (0, 230, 255)
        cv2.rectangle(L, P(b[0], b[1]), P(b[2], b[3]), col, 1)
        if not d["sup"]:
            _label(L, f"{d['change']:.1f}", P(b[0], b[1] - 2), col, 0.4)
    if p_l2 and row["L2small_src"] == "det":
        top = json.loads(row["L2small_top"])[0][0]
        col = (255, 255, 0) if row["L2small_in1"] else (0, 0, 255)
        for img in (L, R):
            cv2.rectangle(img, P(top[0], top[1]), P(top[2], top[3]), (0, 0, 0), 5)
            cv2.rectangle(img, P(top[0], top[1]), P(top[2], top[3]), col, 3)
    if p_m0:
        for img in (L, R):
            cv2.drawMarker(img, P(*p_m0), (255, 0, 255), cv2.MARKER_CROSS, 22, 3)
    if prev_pt:
        cv2.drawMarker(L, P(*prev_pt), (255, 128, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
    gap = np.full((L.shape[0], 8, 3), 255, np.uint8)
    body = np.hstack([L, gap, R])
    res = "HIT" if row["L2small_in1"] else "miss"
    d_m0 = (np.hypot(p_m0[0] - gc[0], p_m0[1] - gc[1]) if p_m0 else float("nan"))
    lines = [
        (f"D{ev.desktop} {ev.view}  frame j={ev.j} -> k={ev.step}  {ev.target}  "
         f"part width {row['partB_width']:.0f} px", 0.55, (0, 0, 0)),
        (f"L2 top-1: {res}  (source {row['L2small_src']}, {int(row['L2small_n'])} candidates "
         f"after skip)    today's guess M0app: {'HIT' if row['M0app_in1'] else 'miss'}"
         f" ({d_m0:.0f} px from the part)", 0.5, (0, 0, 0)),
        ("left: frame j (part present)   right: frame k (part gone)", 0.45, (60, 60, 60)),
        ("green = removed part   yellow = kept candidates (mean dE j->k)   grey = skipped "
         "(already drawn)", 0.45, (60, 60, 60)),
        ("red / cyan = L2 top-1 (miss / hit)   magenta + = M0app top-1   blue x = part "
         "removed at k+1", 0.45, (60, 60, 60)),
    ]
    strip = np.full((22 * len(lines) + 8, body.shape[1], 3), 255, np.uint8)
    for i, (t, sc, col) in enumerate(lines):
        cv2.putText(strip, t, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1,
                    cv2.LINE_AA)
    return np.vstack([strip, body])


def main(argv=None) -> int:
    want = (argv if argv is not None else sys.argv[1:])
    d = pd.read_csv(env.OUT / "guess_events.csv")
    events, shapes, by_frame, con = setup_l1()
    dets = load_dets()
    prev_parts = PrevParts(con, by_frame)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for key in want:
        dk, view, step = key.split(":")
        ev = next(e for e in events if (e.desktop, e.view, e.step) == (int(dk), view, int(step)))
        row = d[d.ev == key].iloc[0]
        frames = base.Frames(con, ev.desktop, ev.view)
        img_j, img_k = frames.get(ev.j), frames.get(ev.step)
        gmask = common.frame_masks(con, ev.desktop, ev.view, ev.j)[row["loc_instance"]]
        known = known_boxes_at(ev, by_frame)
        cands = []
        for r in dets.get(f"{ev.desktop}:{ev.view}:{ev.j}", []):
            if r[4] != ev.cls or r[5] < CONF:
                continue
            box = tuple(int(round(v)) for v in r[:4])
            c = {"box": box, "sup": suppressed(box, known)}
            c["change"] = M.box_change(img_j, img_k, (0, 0), box)
            cands.append(c)
        prev = prev_parts.get(ev)
        prev_pt = None
        if prev is not None and prev[1] == ev.cls:
            pb = prev[0]
            prev_pt = (0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3]))
        img = sheet(ev, row, img_j, img_k, gmask, cands, known, prev_pt)
        tag = "hit" if row["L2small_in1"] else "miss"
        path = OUTDIR / f"D{ev.desktop}_{ev.view}_k{ev.step:02d}_{tag}.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(path)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
