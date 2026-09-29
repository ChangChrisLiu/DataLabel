"""Look before believing ``probe_gt.py``: frame k-1 and k side by side.

For every event where the B2 GT draft still has a same-class box at ``k``, draw
the same-class drafts of both frames on a crop around the GT: the B2 GT in
green, the location-resolved part in magenta, every other same-class draft in
yellow.  If the B2 GT is visibly still in frame ``k`` the ordinal explanation
holds.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l1_localise import base  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402

GREEN, MAGENTA, YELLOW = (0, 220, 0), (255, 0, 255), (255, 220, 0)


def draw(img, shapes, colour, thick):
    for s in shapes:
        for poly in s.polys:
            pts = np.asarray(poly).reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, colour, thick, cv2.LINE_AA)


def sheet(con, frames, by_frame, row, out_dir: Path) -> Path:
    d, v, k = int(row["desktop"]), row["view"], int(row["step"])
    before = by_frame.get((d, v, k - 1), {})
    after = by_frame.get((d, v, k), {})
    gt = before[row["instance"]]
    loc = before.get(row["loc_instance"]) if row["loc_instance"] else None
    cls = gt.cls
    # crop: union of same-class boxes at k-1, padded
    same = [s for s in before.values() if s.cls == cls]
    x0 = min(s.box[0] for s in same)
    y0 = min(s.box[1] for s in same)
    x1 = max(s.box[2] for s in same)
    y1 = max(s.box[3] for s in same)
    pad = max(40, int(0.25 * max(x1 - x0, y1 - y0)))
    panels = []
    for step, pool in ((k - 1, before), (k, after)):
        img = frames.get(step)
        if img is None:
            continue
        h, w = img.shape[:2]
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
        canvas = img.copy()
        others = [s for s in pool.values() if s.cls == cls
                  and s.instance not in (gt.instance,
                                         loc.instance if loc else None)]
        thick = 1 if v == "scan" or v == "rs" else 3
        draw(canvas, others, YELLOW, thick)
        if step == k - 1:
            draw(canvas, [gt], GREEN, thick + 1)
            if loc is not None and loc.instance != gt.instance:
                draw(canvas, [loc], MAGENTA, thick + 1)
        crop = canvas[cy0:cy1, cx0:cx1]
        scale = 700.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LINEAR if scale > 1
                          else cv2.INTER_AREA)
        panels.append(common.banner(crop, f"frame {step}", 36))
    # zooms: the B2 GT and the location-resolved part, in both frames
    zooms = []
    for shape, colour, name in ((gt, GREEN, "B2 GT"), (loc, MAGENTA, "loc GT")):
        if shape is None:
            continue
        row_panels = []
        for step in (k - 1, k):
            img = frames.get(step)
            if img is None:
                continue
            bx0, by0, bx1, by1 = shape.box
            side = max(bx1 - bx0, by1 - by0)
            p = max(12, int(1.5 * side))
            h, w = img.shape[:2]
            cx0, cy0 = max(0, bx0 - p), max(0, by0 - p)
            cx1, cy1 = min(w, bx1 + p), min(h, by1 + p)
            crop = img[cy0:cy1, cx0:cx1].copy()
            scale = 300.0 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
            r = [(int((bx0 - cx0) * scale), int((by0 - cy0) * scale)),
                 (int((bx1 - cx0) * scale), int((by1 - cy0) * scale))]
            cv2.rectangle(crop, r[0], r[1], colour, 1)
            row_panels.append(common.banner(crop, f"{name} @ {step}", 30))
        if row_panels:
            zooms.append(common.hstack_pad(row_panels))
    if zooms:
        wmax = max(z.shape[1] for z in zooms)
        col = np.full((sum(z.shape[0] + 10 for z in zooms), wmax, 3), 255, np.uint8)
        y = 0
        for z in zooms:
            col[y:y + z.shape[0], :z.shape[1]] = z
            y += z.shape[0] + 10
        zooms = [col]
    im = common.hstack_pad(panels + zooms)
    im = common.banner(im, f"D{d} {v} k={k} {row['label']}  GT(B2) {row['instance']}"
                       f" green, loc {row['loc_instance'] or '-'} magenta", 40)
    path = out_dir / f"d{d}_{v}_k{k}.jpg"
    base.save_rgb(path, im)
    return path


def main() -> int:
    events, _shapes, by_frame, con = base.load_events()
    rows = list(csv.DictReader((base.OUT / "gt_check.csv").open(encoding="utf-8")))
    want = [r for r in rows if r["gt_still_present"] == "1"]
    if len(sys.argv) > 1:
        want = want[: int(sys.argv[1])]
    out = base.OUT / "gt_check"
    frames = {}
    for r in want:
        key = (int(r["desktop"]), r["view"])
        if key not in frames:
            frames = {key: base.Frames(con, *key)}
        print(sheet(con, frames[key], by_frame, r, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
