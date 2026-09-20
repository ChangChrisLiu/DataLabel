"""Failure sheets for Task B2: what the two methods actually propose.

A metric that moves by three points means nothing until the pictures behind it
have been opened.  This renders one sheet per event, sorted into
``fix`` / ``break`` / ``both_fail`` / ``both_ok`` so the interesting ones are
easy to find.

Each sheet has three panels:

* **ROI context** -- the frame being annotated (``step - 1``), the whole ROI,
  with the draft polygon in green, the baseline's box and centroid in orange and
  the split method's top proposal in cyan (its other two in thin cyan);
* **zoom, this frame** -- the draft's own box padded, same overlays, so a 19 px
  screw is actually visible;
* **zoom, next frame** -- the same window at ``step``, i.e. what the part's
  removal left behind.

Usage::

    python -m experiments.plan_b_probe.diff_overlays --views scan,oak1,oak2,rs \
        --per-class 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tda.core.diffmap import diff_delta_e  # noqa: E402

from experiments.plan_b_probe import diff_eval as E  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402

GREEN = (60, 220, 90)
ORANGE = (255, 150, 0)
CYAN = (0, 210, 255)
WHITE = (255, 255, 255)


def _draw_box(img, box, colour, thickness=2) -> None:
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]) - 1,
                  int(box[3]) - 1), colour, thickness, cv2.LINE_AA)


def _draw_point(img, point, colour, marker=cv2.MARKER_CROSS, size=26,
                thickness=3) -> None:
    cv2.drawMarker(img, (int(round(point[0])), int(round(point[1]))), colour,
                   marker, size, thickness)


def _panel(image, window, gt_mask, base, split, scale_to=640) -> np.ndarray:
    """One panel: a window of ``image`` with every overlay drawn on it."""
    x0, y0, x1, y1 = window
    out = np.ascontiguousarray(image[y0:y1, x0:x1]).copy()
    if gt_mask is not None:
        sub = gt_mask[y0:y1, x0:x1].astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, GREEN, 3, cv2.LINE_AA)
    shift = np.array([x0, y0, x0, y0])
    for rank, p in enumerate(split[:3]):
        _draw_box(out, np.asarray(p.box) - shift, CYAN, 3 if rank == 0 else 1)
    if split:
        _draw_point(out, (split[0].point[0] - x0, split[0].point[1] - y0),
                    CYAN, cv2.MARKER_TILTED_CROSS, 34, 3)
    if base:
        _draw_box(out, np.asarray(base[0].box) - shift, ORANGE, 2)
        _draw_point(out, (base[0].point[0] - x0, base[0].point[1] - y0), ORANGE)
    h, w = out.shape[:2]
    if max(h, w) > scale_to:
        s = scale_to / float(max(h, w))
        out = cv2.resize(out, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    return out


def _sheet(caption: str, panels: list[np.ndarray]) -> np.ndarray:
    height = max(p.shape[0] for p in panels)
    padded = [np.pad(p, ((0, height - p.shape[0]), (0, 0), (0, 0)),
                     constant_values=255) for p in panels]
    strip = np.full((52, sum(p.shape[1] for p in padded) + 16 * (len(padded) - 1),
                     3), 255, np.uint8)
    cv2.putText(strip, caption, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 0, 0), 2, cv2.LINE_AA)
    row = padded[0]
    for p in padded[1:]:
        row = np.hstack([row, np.full((height, 16, 3), 255, np.uint8), p])
    if row.shape[1] < strip.shape[1]:
        row = np.pad(row, ((0, 0), (0, strip.shape[1] - row.shape[1]), (0, 0)),
                     constant_values=255)
    return np.vstack([strip, row[:, : strip.shape[1]]])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=E.DEFAULT_DB)
    ap.add_argument("--views", default="scan,oak1,oak2,rs")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--out", default=str(E.OUT / "overlays"))
    args = ap.parse_args(argv)

    from tda.ui.app_priors import load_priors

    db = str(args.db)
    common.DB_URI = db if db.startswith("file:") else f"file:{db}?mode=ro"
    common.TMP = E.TMP
    shapes = common.load_geometry()
    by_frame = common.index_shapes(shapes)
    priors = load_priors()
    con = common.connect()
    classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}
    out_root = Path(args.out)
    written: dict[str, int] = {}

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
                base = E.baseline_proposals(delta, min_area)
                split = E.split_proposals(prev_img, cur_img, roi, delta=delta,
                                          min_area=min_area,
                                          expect_area=E.priors_for(ev.cls, priors))

                def inside(props) -> bool:
                    if not props:
                        return False
                    x, y = (int(round(props[0].point[0])),
                            int(round(props[0].point[1])))
                    return bool(0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]
                                and gt[y, x])

                b_ok, s_ok = inside(base), inside(split)
                kind = ("fix" if s_ok and not b_ok else
                        "break" if b_ok and not s_ok else
                        "both_ok" if b_ok else "both_fail")
                key = f"{kind}/{ev.group}"
                if written.get(key, 0) >= args.per_class:
                    continue
                written[key] = written.get(key, 0) + 1

                gb = ev.shape.box
                pad = max(60, int(0.9 * max(gb[2] - gb[0], gb[3] - gb[1])))
                zoom = (max(0, gb[0] - pad), max(0, gb[1] - pad),
                        min(prev_img.shape[1], gb[2] + pad),
                        min(prev_img.shape[0], gb[3] + pad))
                caption = (f"D{ev.desktop} {ev.view} s{ev.step} {ev.verb} "
                           f"{ev.cls} [{ev.group}] {ev.shape.label} "
                           f"{ev.shape.size:.0f}px  base={'HIT' if b_ok else 'miss'} "
                           f"split={'HIT' if s_ok else 'miss'}  "
                           f"cands={len(split)}")
                sheet = _sheet(caption, [
                    _panel(prev_img, roi, gt, base, split, 620),
                    _panel(prev_img, zoom, gt, base, split, 430),
                    _panel(cur_img, zoom, None, base, split, 430),
                ])
                path = (out_root / kind /
                        f"{ev.group}_d{ev.desktop}_{ev.view}_s{ev.step}_"
                        f"{ev.cls}.jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                print("wrote", path, flush=True)
            cache.clear()
    con.close()
    print({k: v for k, v in sorted(written.items())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
