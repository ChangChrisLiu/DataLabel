"""Overlay sheets: where each method's armed box/point landed, vs both GTs.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l1_localise.overlays \
        --tag v1 --methods M0,M2e,M3,M4_b1 --events 13:scan:38,13:scan:39 ...

Left: frame ``j`` (the part is present), ROI crop.  GT-A (B2, verbatim) green,
GT-B (location-resolved) magenta when it differs; each method's rank-1 box and
point in its own colour.  Right: native-resolution zooms around GT-B and
around each method's point, in frame ``j`` and frame ``k``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l1_localise import base  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402

COLOURS = {  # RGB; the first prefix that matches wins
    "M0": (255, 140, 0), "M2e": (255, 215, 0), "split": (255, 215, 0),
    "M3": (30, 144, 255), "M4js": (170, 60, 255), "M4": (0, 255, 255),
    "M5": (255, 255, 255), "P": (0, 255, 255),
}
GREEN, MAGENTA = (0, 220, 0), (255, 0, 255)


def colour_of(method: str):
    for k, c in COLOURS.items():
        if method.startswith(k):
            return c
    return (255, 255, 255)


def text(img, s, org, scale=0.6, colour=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                cv2.LINE_AA)


def small_banner(img, s, height=30):
    strip = np.full((height, img.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, s, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def zoom(img, centre, side, out=240, boxes=()):
    """Native crop of ``side`` px around ``centre``, upscaled to ``out``."""
    h, w = img.shape[:2]
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    crop = img[y0:y1, x0:x1].copy()
    s = out / float(max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    for (bx, col) in boxes:
        p0 = (int((bx[0] - x0) * s), int((bx[1] - y0) * s))
        p1 = (int((bx[2] - x0) * s), int((bx[3] - y0) * s))
        cv2.rectangle(crop, p0, p1, col, 2)
    return crop


def sheet(con, frames, by_frame, rows: pd.DataFrame, methods, out_path: Path,
          note: str = ""):
    r0 = rows.iloc[0]
    d, v, k = int(r0["desktop"]), r0["view"], int(r0["step"])
    img_j, img_k = frames.get(k - 1), frames.get(k)
    roi = json.loads(r0["roi"])
    x0, y0, x1, y1 = roi
    shapes_j = by_frame[(d, v, k - 1)]
    gtA = shapes_j[r0["instance"]]
    locB = r0["loc_instance"] if isinstance(r0["loc_instance"], str) else ""
    gtB = shapes_j.get(locB) if locB else None

    canvas = img_j.copy()
    thick = max(2, int(round(max(img_j.shape[:2]) / 700)))
    for poly in gtA.polys:
        pts = np.asarray(poly).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], True, GREEN, thick + 1, cv2.LINE_AA)
    if gtB is not None and gtB.instance != gtA.instance:
        for poly in gtB.polys:
            pts = np.asarray(poly).reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(canvas, [pts], True, MAGENTA, thick + 1, cv2.LINE_AA)
    legend = []
    guesses = []
    for m in methods:
        sel = rows[rows["method"] == m]
        if sel.empty:
            legend.append((f"{m}: n/a (no sibling)", colour_of(m)))
            continue
        r = sel.iloc[0]
        props = json.loads(r["props"]) if isinstance(r["props"], str) else []
        col = colour_of(m)
        held = r.get("withheld", 0)
        if not props or (held == held and int(held or 0)):
            legend.append((f"{m}: no box", col))
            continue
        box, pt = props[0][0], props[0][1]
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), col, thick)
        cv2.drawMarker(canvas, (int(pt[0]), int(pt[1])), col, cv2.MARKER_CROSS,
                       6 * thick, thick)
        a = "hit" if int(r["A_in1"]) else "miss"
        b = ("hit" if r.get("B_in1") == 1 else "miss") if gtB is not None else "-"
        legend.append((f"{m}: A {a} / B {b}  ({box[2]-box[0]}x{box[3]-box[1]})",
                       col))
        guesses.append((m, pt, box, col))
    crop = canvas[y0:y1, x0:x1]
    s = 820.0 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    box_h = 12 + 22 * len(legend)
    shade = crop[:box_h, :430].astype(np.float32) * 0.35
    crop[:box_h, :430] = shade.astype(np.uint8)
    y = 24
    for t, c in legend:
        cv2.putText(crop, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1,
                    cv2.LINE_AA)
        y += 22

    # zooms
    part = gtB if gtB is not None else gtA
    side = int(max(60, 4 * max(part.box[2] - part.box[0], part.box[3] - part.box[1])))
    zrows = []
    cen = ((part.box[0] + part.box[2]) / 2, (part.box[1] + part.box[3]) / 2)
    zb = [(tuple(part.box), MAGENTA if gtB is not None else GREEN)]
    zj = zoom(img_j, cen, side, boxes=zb)
    zk = zoom(img_k, cen, side, boxes=zb)
    text(zj, f"part @ j={k - 1}", (6, 18))
    text(zk, f"part @ k={k}", (6, 18))
    zrows.append(common.hstack_pad([zj, zk], 6))
    for m, pt, box, col in guesses[:3]:
        zj = zoom(img_j, pt, side, boxes=[(box, col)])
        zk = zoom(img_k, pt, side, boxes=[(box, col)])
        text(zj, f"{m} @ j", (6, 18), 0.55, col)
        text(zk, f"{m} @ k", (6, 18), 0.55, col)
        zrows.append(common.hstack_pad([zj, zk], 6))
    wmax = max(z.shape[1] for z in zrows)
    col_img = np.full((sum(z.shape[0] + 6 for z in zrows), wmax, 3), 255, np.uint8)
    yy = 0
    for z in zrows:
        col_img[yy:yy + z.shape[0], :z.shape[1]] = z
        yy += z.shape[0] + 6
    im = common.hstack_pad([crop, col_img], 10)
    gtnote = ("GT-A green = GT-B" if gtB is not None and gtB.instance == gtA.instance
              else "GT-A green (B2), GT-B magenta (the part that left)"
              if gtB is not None else "GT-A green; GT-B unresolved")
    im = small_banner(im, f"{gtnote}. {note}")
    im = small_banner(im, f"D{d} {v} step {k} (annotate frame {k - 1}): "
                          f"{r0['target']} [{r0['label']}], {r0['part_width']} px")
    base.save_rgb(out_path, im)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--csv", default="")
    ap.add_argument("--methods", default="M0,M3,M4_b1")
    ap.add_argument("--events", required=True, help="d:view:step[=name],...")
    ap.add_argument("--subdir", default="overlays")
    args = ap.parse_args(argv)
    path = Path(args.csv) if args.csv else base.OUT / f"events_{args.tag}.csv"
    d = pd.read_csv(path)
    methods = args.methods.split(",")
    _events, _shapes, by_frame, con = base.load_events()
    frames_cache = {}
    for item in args.events.split(","):
        key, _, name = item.partition("=")
        ds, v, st = key.split(":")
        rows = d[(d["desktop"] == int(ds)) & (d["view"] == v) & (d["step"] == int(st))]
        if rows.empty:
            print(f"no rows for {key}")
            continue
        fk = (int(ds), v)
        if fk not in frames_cache:
            frames_cache = {fk: base.Frames(con, *fk)}
        name = name or f"d{ds}_{v}_s{st}"
        note = ""
        m1 = rows[rows["method"] == "M1app (k=3)"]
        if not m1.empty and "withheld" in m1.columns:
            held = m1.iloc[0]["withheld"]
            note = ("M1(k=3) withholds M0app's box." if held == held and int(held)
                    else "M1(k=3) keeps M0app's box.")
        p = sheet(con, frames_cache[fk], by_frame, rows, methods,
                  base.OUT / args.subdir / f"{name}.jpg", note)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
