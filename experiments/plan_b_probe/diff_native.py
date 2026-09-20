"""Task B2b -- is the sub-40 px half unreachable because of *resolution*?

Task B2 measured, on the same 201 removal events, that splitting the diff blob
is a large win on parts wider than 100 px (median SAM IoU 0.320 -> 0.762) and
changes nothing at all on the 54 % of events whose part is narrower than 40 px:
over the **whole** candidate set the best box IoU there has a median of 0.000
(``experiments_out/plan_b_probe/diff_eval/report.md`` section 5). Its
explanation was arithmetic rather than algorithmic -- the pair is compared at
``max_side=1600``, so a 19 px screw on a 4032 px OAK frame is 7 px on the dE
map and ``blur=5`` with ``shift_px=1`` erases it.

This harness tests that explanation directly, and it is the only thing it
tests. Around a **seed box** -- the top split proposal, or the top diff blob --
it pads a window, re-runs :func:`tda.core.diffmap.diff_delta_e` on that window
at **native resolution**, and re-ranks with the same
:func:`tda.core.diff_split.propose_parts`. Everything else is inherited from
``diff_eval.py``: the events, the ground truth, the ROI, the SAM protocol and
the metric definitions, so the columns are comparable line for line with the
B2 report.

Held out, because there are knobs
---------------------------------
The window source, its padding and the native noise floor are tuned on **D24
only** (``--grid --desktops 24``, no SAM, so the tuning never sees the number
it is judged on) and reported on **D13 + D33**. The gate the task sets: the
held-out SAM IoU median for parts under 100 px must improve by >= 0.10 with no
regression >= 0.10 on parts at or above 100 px.

Two SAM columns, and the difference between them is the point
-------------------------------------------------------------
``sam_iou`` prompts exactly as the shipped tool does -- the ROI window scaled
to at most 1024 px -- which on a 12 MP OAK frame is another 3-4x downscale, so
a 19 px screw is 5 px in the crop whatever the proposal says. ``sam_iou_local``
prompts on a crop of the *proposal's own window*, i.e. at native resolution.
The first is what an annotator feels today; the second says whether the prompt
or the crop is the remaining blocker.

Usage::

    python -m experiments.plan_b_probe.diff_native --grid --desktops 24
    python -m experiments.plan_b_probe.diff_native --desktops 13,33 --sam \
        --source split --pad 2.0 --min-area 40

Writes ``events_<tag>.csv`` and ``table_<tag>.md`` under
``experiments_out/plan_b_probe/diff_eval_b2b/``.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tda.core.diff_split import propose_parts  # noqa: E402
from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

from experiments.plan_b_probe import diff_eval as E  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402

OUT = Path("D:/DataSet/experiments_out/plan_b_probe/diff_eval_b2b")
TMP = Path("D:/DataSet/.cache/tmp/b2b")

#: Size bands, on ``sqrt(area)`` of the draft polygon -- the "part width" of the
#: B2 report, so the rows line up with its section 5.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("<40", 0.0, 40.0),
    ("40-100", 40.0, 100.0),
    ("<100", 0.0, 100.0),
    (">=100", 100.0, 1e9),
)

Box = tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# the native pass
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Native:
    """Everything the second pass can be tuned by (on D24, and only there)."""

    #: Where the window comes from: the top split proposal, or the top blob.
    #: The blob is the wider net -- it *overlaps* the removed part 47-64 % of
    #: the time -- and the split proposal is the tighter one.
    source: str = "split"
    #: Window = the seed box grown by ``pad x its longer side`` on every side.
    #: It has to be generous for two reasons: the seed is regularly off the
    #: part, and ``propose_parts`` drops any candidate covering more than 60 %
    #: of the window it is given, which a tight window would trigger.
    pad: float = 2.0
    #: Absolute floor for that growth, in native pixels.
    min_pad: int = 32
    #: Hard cap on the window's long side, in native pixels. The whole claim of
    #: this experiment is that a *local* pass is cheap -- "a few hundred
    #: thousand pixels, not 12 MP" -- and a generous ``pad`` around a big seed
    #: on a 4032 px frame reaches the whole frame, which would be the expensive
    #: thing the ROI-wide pass already avoids.
    max_side: int = 800
    #: Noise floor at native resolution. 80 (the production default) is a
    #: 9x9 patch; a 19 px screw is ~280 px, so this is not the binding limit.
    min_area: int = 40
    blur: int = 5
    shift_px: int = 1
    #: Use the class's area prior at all. It has to be **rescaled** when it is:
    #: ``propose_parts`` divides a candidate's area by the area of the region it
    #: was given, and here that region is the padded window, not the ROI the
    #: bands in ``configs/area_priors.yaml`` are fractions of -- a factor of 10
    #: to 1000. Feeding the raw band in would not be a weak cue, it would be an
    #: actively wrong one.
    prior: bool = True

    @property
    def tag(self) -> str:
        return (f"{self.source}_pad{self.pad:g}_min{self.min_area}"
                f"_blur{self.blur}_shift{self.shift_px}"
                f"_prior{int(self.prior)}")


def _window(seed: Box, cfg: Native, hw: tuple[int, int]) -> Box:
    """The padded native-resolution window around a seed box, capped and clipped.

    The cap is applied around the seed's **centre**, so a box that is already
    wider than :attr:`Native.max_side` keeps its middle rather than its left
    edge -- the part, if the seed found one at all, is nearer the middle.
    """
    h, w = int(hw[0]), int(hw[1])
    x0, y0, x1, y1 = (int(v) for v in seed)
    grow = max(int(round(cfg.pad * max(x1 - x0, y1 - y0))), int(cfg.min_pad))
    box = [x0 - grow, y0 - grow, x1 + grow, y1 + grow]
    for axis, limit in ((0, w), (1, h)):
        low, high = box[axis], box[axis + 2]
        if high - low > int(cfg.max_side):
            mid = 0.5 * (low + high)
            low = int(round(mid - 0.5 * cfg.max_side))
            high = low + int(cfg.max_side)
        box[axis], box[axis + 2] = max(0, low), min(limit, high)
    return (box[0], box[1], box[2], box[3])


def native_proposals(prev_rgb: np.ndarray, cur_rgb: np.ndarray, seed: Box,
                     cfg: Native, *, expect_area=None, roi_area: float = 0.0,
                     max_proposals: int = 3) -> tuple[list[E.Proposal], Box, float]:
    """Re-diff one window at native resolution and rank what is in it.

    Returns ``(proposals, window, seconds)``; the boxes and points are back in
    full-frame coordinates. The window is cropped out of both frames *before*
    the dE map is computed, so the cost is the window's own pixels -- a few
    hundred thousand at most -- and not the 12 MP the ROI-wide pass touches.

    ``expect_area`` is a band of ROI fractions and is converted to a band of
    *window* fractions with ``roi_area``; see :attr:`Native.prior`.
    """
    t0 = time.perf_counter()
    win = _window(seed, cfg, prev_rgb.shape[:2])
    wx0, wy0, wx1, wy1 = win
    if wx1 - wx0 < 8 or wy1 - wy0 < 8:
        return [], win, time.perf_counter() - t0
    sub_prev = np.ascontiguousarray(prev_rgb[wy0:wy1, wx0:wx1])
    sub_cur = np.ascontiguousarray(cur_rgb[wy0:wy1, wx0:wx1])
    delta = diff_delta_e(sub_prev, sub_cur, roi=None, blur=cfg.blur,
                         shift_px=cfg.shift_px, max_side=None)
    band = None
    if cfg.prior and expect_area is not None and roi_area > 0.0:
        window_area = float((wx1 - wx0) * (wy1 - wy0))
        if window_area > 0.0:
            scale = roi_area / window_area
            band = (float(expect_area[0]) * scale, float(expect_area[1]) * scale)
    parts = propose_parts(sub_prev, sub_cur, None, delta_e=delta,
                          expect_area=band, min_area=cfg.min_area,
                          max_proposals=max_proposals)
    out = [E.Proposal(box=(int(p.box[0]) + wx0, int(p.box[1]) + wy0,
                           int(p.box[2]) + wx0, int(p.box[3]) + wy0),
                      point=(float(p.point[0]) + wx0, float(p.point[1]) + wy0),
                      score=float(p.score), area=int(p.area))
           for p in parts]
    return out, win, time.perf_counter() - t0


def seed_box(cfg: Native, split: list[E.Proposal], delta: np.ndarray,
             min_area: int) -> Optional[Box]:
    """Where the native pass looks: the top split proposal, or the top blob."""
    if cfg.source == "blob":
        blobs = diff_blobs(delta, min_area=min_area, max_blobs=8)
        return tuple(int(v) for v in blobs[0].box) if blobs else None
    return tuple(int(v) for v in split[0].box) if split else None


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def _point_in(point, gt: np.ndarray) -> bool:
    x, y = int(round(point[0])), int(round(point[1]))
    return bool(0 <= y < gt.shape[0] and 0 <= x < gt.shape[1] and gt[y, x])


def _local_crop(image: np.ndarray, window: Box) -> tuple[np.ndarray, Box, float]:
    """A SAM crop of one window, scaled like ``sam_crop`` would scale it."""
    x0, y0, x1, y1 = window
    crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = 1.0
    if max(ch, cw) > E.MAX_SAM_SIDE:
        scale = E.MAX_SAM_SIDE / float(max(ch, cw))
        crop = cv2.resize(crop, (max(1, int(round(cw * scale))),
                                 max(1, int(round(ch * scale)))),
                          interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(crop), (x0, y0, x1, y1), scale


def run(args) -> list[dict]:
    from tda.ui.app_priors import load_priors

    common.DB_URI = args.db_uri
    common.TMP = TMP
    TMP.mkdir(parents=True, exist_ok=True)

    cfg = Native(source=args.source, pad=args.pad, min_pad=args.min_pad,
                 min_area=args.min_area, blur=args.blur,
                 shift_px=args.shift_px, prior=args.prior)
    desktops = [int(d) for d in str(args.desktops).split(",") if d]
    shapes = common.load_geometry()
    by_frame = common.index_shapes(shapes)
    priors = load_priors()
    con = common.connect()
    classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}
    views = [v for v in args.views.split(",") if v]
    sam = E.Sam() if args.sam else None

    rows: list[dict] = []
    for desktop in desktops:
        for view in views:
            if not any(s.desktop == desktop and s.view == view for s in shapes):
                continue
            roi = E.roi_of(shapes, desktop, view)
            roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
            h, w = common.NATIVE_HW[view]
            wide_min_area = max(30, int(round(80 * (h * w) / (1600 * 1600))))
            events, unresolved = E.resolve_events(con, shapes, by_frame, classes,
                                                  desktop, view)
            if args.limit:
                events = events[: args.limit]
            print(f"D{desktop} {view}: {len(events)} events "
                  f"({unresolved} unresolved), roi={roi}", flush=True)

            cache: dict[int, Optional[np.ndarray]] = {}

            def frame(step: int) -> Optional[np.ndarray]:
                if step not in cache:
                    if len(cache) > 3:
                        cache.clear()
                    path = common.frame_path(con, desktop, view, step)
                    cache[step] = None if path is None else common.read_rgb(path)
                return cache[step]

            for ev in events:
                prev_img = frame(ev.step - 1)     # the annotated frame
                cur_img = frame(ev.step)          # the one the card diffs against
                if prev_img is None or cur_img is None:
                    continue
                gt = common.frame_masks(con, desktop, view, ev.step - 1
                                        ).get(ev.shape.instance)
                if gt is None or not gt.any():
                    continue
                expect = E.priors_for(ev.cls, priors)
                gt_box = tuple(int(v) for v in ev.shape.box)

                delta = diff_delta_e(prev_img, cur_img, roi=roi,
                                     max_side=E.MAX_SIDE)
                base = E.baseline_proposals(delta, wide_min_area)
                split = E.split_proposals(prev_img, cur_img, roi, delta=delta,
                                          min_area=wide_min_area,
                                          expect_area=expect)
                seed = seed_box(cfg, split, delta, wide_min_area)
                if seed is None:
                    native, window, native_s = [], None, 0.0
                else:
                    native, window, native_s = native_proposals(
                        prev_img, cur_img, seed, cfg, expect_area=expect,
                        roi_area=roi_area)

                crop = rect = scale = gt_crop = None
                if sam is not None:
                    crop, rect, scale = sam.crop_for(prev_img, roi)
                    gt_crop = E.gt_in_crop(gt, rect, crop.shape[:2])

                for method, props in (("baseline", base), ("split", split),
                                      ("native", native)):
                    row = {
                        "desktop": desktop, "view": view, "step": ev.step,
                        "method": method, "cls": ev.cls, "group": ev.group,
                        "label": ev.shape.label, "instance": ev.shape.instance,
                        "part_area": ev.shape.area,
                        "part_size_px": round(ev.shape.size, 1),
                        "part_frac_roi": round(ev.shape.area / roi_area, 6),
                        "n_proposals": len(props),
                        "seed_box": None if seed is None else list(seed),
                        "window": None if window is None else list(window),
                        "native_s": round(native_s, 4),
                    }
                    for rank, p in enumerate(props[:3]):
                        row[f"p{rank}_point_in"] = int(_point_in(p.point, gt))
                        row[f"p{rank}_box_iou"] = round(
                            E.box_iou(p.box, gt_box), 4)
                        row[f"p{rank}_armed"] = int(
                            not E.covers_most(p.box, roi))
                        row[f"p{rank}_box"] = list(p.box)
                    if props and sam is not None:
                        top = props[0]
                        armed = not E.covers_most(top.box, roi)
                        mask = sam.predict(crop, rect, scale, top, with_box=armed)
                        row["sam_with_box"] = int(armed)
                        row["sam_iou"] = (round(E.mask_iou(mask, gt_crop), 4)
                                          if mask is not None else "")
                        if args.sam_local:
                            # the same prompt on a crop of the proposal's own
                            # window: native pixels, no 12 MP -> 1024 downscale
                            lwin = _window(tuple(int(v) for v in top.box), cfg,
                                           prev_img.shape[:2])
                            lcrop, lrect, lscale = _local_crop(prev_img, lwin)
                            lmask = sam.predict(lcrop, lrect, lscale, top,
                                                with_box=armed)
                            lgt = E.gt_in_crop(gt, lrect, lcrop.shape[:2])
                            row["sam_iou_local"] = (
                                round(E.mask_iou(lmask, lgt), 4)
                                if lmask is not None else "")
                    rows.append(row)
            cache.clear()
    con.close()
    if sam is not None:
        print(f"SAM: {sam.calls} calls, {sam.seconds:.1f} s "
              f"({1000 * sam.seconds / max(1, sam.calls):.0f} ms each)")
    return rows


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _median(values) -> float:
    vals = [float(v) for v in values if v not in (None, "")]
    return statistics.median(vals) if vals else float("nan")


def band_table(rows: list[dict], methods: list[str], column: str = "sam_iou"
               ) -> str:
    """Point-in-part, box IoU and SAM IoU per part-size band -- the B2 section 5."""
    lines = [f"| part width | n | method | point-in-part | box IoU med | "
             f"{column} med | {column} p75 |", "|---|---|---|---|---|---|---|"]
    for name, low, high in BANDS:
        for method in methods:
            sel = [r for r in rows if r["method"] == method
                   and low <= float(r["part_size_px"]) < high]
            if not sel:
                continue
            n = len(sel)
            hits = sum(int(r.get("p0_point_in") or 0) for r in sel)
            bious = [r.get("p0_box_iou") for r in sel]
            sious = [float(v) for v in (r.get(column) for r in sel)
                     if v not in (None, "")]
            p75 = (statistics.median(sorted(sious)[(len(sious) + 1) // 2:])
                   if len(sious) >= 2 else (sious[0] if sious else float("nan")))
            lines.append(
                f"| {name} | {n} | {method} | {100 * hits / n:.1f} % | "
                f"{_median(bious):.3f} | "
                + (f"{_median(sious):.3f} | {p75:.3f} |" if sious else "- | - |"))
    return "\n".join(lines)


def view_table(rows: list[dict], views: list[str], methods: list[str]) -> str:
    lines = ["| view | method | n | point-in-part | box IoU med | SAM IoU med |",
             "|---|---|---|---|---|---|"]
    for view in views:
        for method in methods:
            sel = [r for r in rows if r["view"] == view and r["method"] == method]
            if not sel:
                continue
            hits = sum(int(r.get("p0_point_in") or 0) for r in sel)
            lines.append(
                f"| {view} | {method} | {len(sel)} | "
                f"{100 * hits / len(sel):.1f} % | "
                f"{_median(r.get('p0_box_iou') for r in sel):.3f} | "
                f"{_median(r.get('sam_iou') for r in sel):.3f} |")
    return "\n".join(lines)


def write_outputs(rows: list[dict], views: list[str], out: Path, tag: str,
                  header: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    methods = ["baseline", "split", "native"]
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    csv_path = out / f"events_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    parts = [f"# diff_native ({tag})", header,
             "## per part-size band (SAM prompted as the tool does)",
             band_table(rows, methods)]
    if any("sam_iou_local" in r for r in rows):
        parts += ["## per part-size band (SAM on a native-resolution local crop)",
                  band_table(rows, methods, column="sam_iou_local")]
    parts += ["## per view", view_table(rows, views, methods)]
    text = "\n\n".join(parts)
    (out / f"table_{tag}.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {csv_path}")


# --------------------------------------------------------------------------- #
# tuning grid (D24 only, no SAM)
# --------------------------------------------------------------------------- #
#: 16 configs, not 48. ``min_area`` 40 vs 120 moved nothing in a first pass and
#: ``shift_px=3`` costs 49 shifted Lab comparisons per window (against 9 at 1),
#: which is most of an hour of tuning for a knob whose whole job is to say
#: whether the two 12 MP frames are registered to a pixel or to three.
GRID = [
    Native(source=s, pad=p, shift_px=x, prior=pr)
    for s in ("split", "blob")
    for p in (1.0, 3.0)
    for x in (1, 2)
    for pr in (True, False)
]


def grid(args) -> int:
    """Sweep the knobs on the tuning desktops; point-in-part and box IoU only.

    Deliberately blind to SAM: the held-out gate is a SAM number, and a knob
    chosen by looking at it would not be held out at all.

    Every config shares one pass over the events. The wide dE map, the frame
    decode and the two reference methods cost ~0.3 s per event and do not
    depend on any knob; the native pass costs ~0.02 s and is the only thing
    swept, so re-running the whole harness per config would have spent 95 % of
    the time recomputing identical numbers.
    """
    from tda.ui.app_priors import load_priors

    common.DB_URI = args.db_uri
    common.TMP = TMP
    TMP.mkdir(parents=True, exist_ok=True)
    desktops = [int(d) for d in str(args.desktops).split(",") if d]
    shapes = common.load_geometry()
    by_frame = common.index_shapes(shapes)
    priors = load_priors()
    con = common.connect()
    classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}

    #: cfg -> list of (part width, point-in, box IoU) ; plus the two references
    scores: dict[str, list[tuple[float, int, float]]] = {}

    def note(name: str, width: float, props) -> None:
        hit = int(bool(props) and _point_in(props[0].point, gt))
        iou = E.box_iou(props[0].box, gt_box) if props else 0.0
        scores.setdefault(name, []).append((width, hit, iou))

    for desktop in desktops:
        for view in [v for v in args.views.split(",") if v]:
            if not any(s.desktop == desktop and s.view == view for s in shapes):
                continue
            roi = E.roi_of(shapes, desktop, view)
            roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
            h, w = common.NATIVE_HW[view]
            wide_min_area = max(30, int(round(80 * (h * w) / (1600 * 1600))))
            events, _ = E.resolve_events(con, shapes, by_frame, classes,
                                         desktop, view)
            if args.limit:
                events = events[: args.limit]
            print(f"D{desktop} {view}: {len(events)} events", flush=True)
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
                gt_box = tuple(int(v) for v in ev.shape.box)
                width = float(ev.shape.size)
                expect = E.priors_for(ev.cls, priors)

                delta = diff_delta_e(prev_img, cur_img, roi=roi, max_side=E.MAX_SIDE)
                base = E.baseline_proposals(delta, wide_min_area)
                split = E.split_proposals(prev_img, cur_img, roi, delta=delta,
                                          min_area=wide_min_area,
                                          expect_area=expect)
                note("baseline", width, base)
                note("split", width, split)
                seeds = {"split": (tuple(int(v) for v in split[0].box)
                                   if split else None),
                         "blob": (tuple(int(v) for v in base[0].box)
                                  if base else None)}
                for cfg in GRID:
                    seed = seeds[cfg.source]
                    props = ([] if seed is None else
                             native_proposals(prev_img, cur_img, seed, cfg,
                                              expect_area=expect,
                                              roi_area=roi_area)[0])
                    note(cfg.tag, width, props)
            cache.clear()
    con.close()

    def band(name: str, low: float, high: float) -> tuple[str, str]:
        sel = [s for s in scores.get(name, []) if low <= s[0] < high]
        if not sel:
            return "-", "-"
        return (f"{100 * sum(s[1] for s in sel) / len(sel):.1f} %",
                f"{statistics.median(s[2] for s in sel):.3f}")

    lines = ["| config | <40 point-in | <40 box IoU | 40-100 point-in | "
             "40-100 box IoU | >=100 point-in | >=100 box IoU |",
             "|---|---|---|---|---|---|---|"]
    for name in ["baseline", "split"] + [c.tag for c in GRID]:
        cells = [c for low, high in ((0.0, 40.0), (40.0, 100.0), (100.0, 1e9))
                 for c in band(name, low, high)]
        lines.append("| " + " | ".join([name] + cells) + " |")
        print(lines[-1], flush=True)
    counts = {name: len(v) for name, v in scores.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    text = (f"# diff_native tuning grid (desktops {args.desktops}, no SAM)\n\n"
            f"{counts.get('baseline', 0)} events. `baseline` and `split` are the "
            f"B2 methods on the same events, for reference; every other row is "
            f"the native second pass with those knobs. A method that proposes "
            f"nothing scores 0 here rather than being dropped from the median.\n\n"
            + "\n".join(lines) + "\n")
    (OUT / "table_grid.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


# --------------------------------------------------------------------------- #
# overlays -- nothing here is believed until the pictures have been opened
# --------------------------------------------------------------------------- #
MAGENTA = (255, 80, 220)


def _overlay_panel(image, window, gt_mask, rank1, alts, native, native_win,
                   scale_to=620):
    """One panel: a window of ``image`` with every rank drawn on it."""
    from experiments.plan_b_probe import diff_overlays as O

    x0, y0, x1, y1 = window
    out = np.ascontiguousarray(image[y0:y1, x0:x1]).copy()
    if gt_mask is not None:
        sub = gt_mask[y0:y1, x0:x1].astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, O.GREEN, 3, cv2.LINE_AA)
    shift = np.array([x0, y0, x0, y0])
    if native_win is not None:
        O._draw_box(out, np.asarray(native_win) - shift, O.WHITE, 1)
    for rank, p in enumerate(native[:3]):
        O._draw_box(out, np.asarray(p.box) - shift, MAGENTA, 3 if rank == 0 else 1)
    if native:
        O._draw_point(out, (native[0].point[0] - x0, native[0].point[1] - y0),
                      MAGENTA, cv2.MARKER_DIAMOND, 30, 3)
    for rank, p in enumerate(alts[:3]):
        O._draw_box(out, np.asarray(p.box) - shift, O.CYAN, 3 if rank == 0 else 1)
    if alts:
        O._draw_point(out, (alts[0].point[0] - x0, alts[0].point[1] - y0),
                      O.CYAN, cv2.MARKER_TILTED_CROSS, 34, 3)
    if rank1 is not None:
        O._draw_box(out, np.asarray(rank1.box) - shift, O.ORANGE, 2)
        O._draw_point(out, (rank1.point[0] - x0, rank1.point[1] - y0), O.ORANGE)
    h, w = out.shape[:2]
    if max(h, w) > scale_to:
        s = scale_to / float(max(h, w))
        out = cv2.resize(out, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    return out


def overlays(args) -> int:
    """One sheet per event: rank 1 orange, the ``Shift+C`` alternates cyan, the
    native-resolution proposals magenta inside their white window, ground truth
    green. Sorted into ``alt_fixes`` / ``native_fixes`` / ``rank1_ok`` /
    ``all_miss`` so the interesting ones can be found without opening 200.
    """
    from experiments.plan_b_probe import diff_overlays as O
    from tda.ui.app_diff import alternate_parts
    from tda.ui.app_priors import load_priors

    common.DB_URI = args.db_uri
    common.TMP = TMP
    cfg = Native(source=args.source, pad=args.pad, min_pad=args.min_pad,
                 min_area=args.min_area, blur=args.blur,
                 shift_px=args.shift_px, prior=args.prior)
    shapes = common.load_geometry()
    by_frame = common.index_shapes(shapes)
    priors = load_priors()
    con = common.connect()
    classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}
    out_root = Path(args.out) / "overlays"
    written: dict[str, int] = {}

    for desktop in [int(d) for d in str(args.desktops).split(",") if d]:
        for view in [v for v in args.views.split(",") if v]:
            if not any(s.desktop == desktop and s.view == view for s in shapes):
                continue
            roi = E.roi_of(shapes, desktop, view)
            roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
            h, w = common.NATIVE_HW[view]
            wide_min_area = max(30, int(round(80 * (h * w) / (1600 * 1600))))
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
                expect = E.priors_for(ev.cls, priors)
                delta = diff_delta_e(prev_img, cur_img, roi=roi, max_side=E.MAX_SIDE)
                base = E.baseline_proposals(delta, wide_min_area)
                split = E.split_proposals(prev_img, cur_img, roi, delta=delta,
                                          min_area=wide_min_area,
                                          expect_area=expect)
                rank1 = base[0] if base else None
                armed = None if rank1 is None else tuple(float(v) for v in rank1.box)
                alts = alternate_parts(
                    [p for p in split if not E.covers_most(p.box, roi)], armed)
                seed = seed_box(cfg, split, delta, wide_min_area)
                native, window = ([], None) if seed is None else \
                    native_proposals(prev_img, cur_img, seed, cfg,
                                     expect_area=expect, roi_area=roi_area)[:2]

                def hit(props) -> bool:
                    return bool(props) and _point_in(props[0].point, gt)

                r1_ok = rank1 is not None and _point_in(rank1.point, gt)
                alt_ok = any(_point_in(p.point, gt) for p in alts)
                nat_ok = hit(native)
                kind = ("rank1_ok" if r1_ok else
                        "alt_fixes" if alt_ok else
                        "native_fixes" if nat_ok else "all_miss")
                key = f"{kind}/{ev.group}"
                if written.get(key, 0) >= args.per_class:
                    continue
                written[key] = written.get(key, 0) + 1

                gb = ev.shape.box
                pad = max(80, int(1.1 * max(gb[2] - gb[0], gb[3] - gb[1])))
                zoom = (max(0, gb[0] - pad), max(0, gb[1] - pad),
                        min(prev_img.shape[1], gb[2] + pad),
                        min(prev_img.shape[0], gb[3] + pad))
                caption = (f"D{ev.desktop} {ev.view} s{ev.step} {ev.cls} "
                           f"[{ev.group}] {ev.shape.label} {ev.shape.size:.0f}px  "
                           f"rank1={'HIT' if r1_ok else 'miss'} "
                           f"alts={len(alts)}{'/HIT' if alt_ok else ''} "
                           f"native={len(native)}{'/HIT' if nat_ok else ''}")
                sheet = O._sheet(caption, [
                    _overlay_panel(prev_img, roi, gt, rank1, alts, native,
                                   window, 620),
                    _overlay_panel(prev_img, zoom, gt, rank1, alts, native,
                                   window, 430),
                    _overlay_panel(cur_img, zoom, None, rank1, alts, native,
                                   window, 430),
                ])
                path = (out_root / kind /
                        f"{ev.group}_d{ev.desktop}_{ev.view}_s{ev.step}_"
                        f"{ev.cls}.jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                print("wrote", path, flush=True)
            cache.clear()
    con.close()
    print({k: v for k, v in sorted(written.items())})
    return 0


# --------------------------------------------------------------------------- #
# what the alternates cost the diff worker
# --------------------------------------------------------------------------- #
def cost(args) -> int:
    """Time the *production* comparison with and without the Shift+C alternates.

    :meth:`tda.ui.app_diff.AssistController.compute` is called on real frame
    pairs, once as it is and once with the splitter stubbed out, so the number
    reported is the added cost of the feature on the diff worker thread and not
    a benchmark of :func:`~tda.core.diff_split.propose_parts` in isolation.
    """
    from tda.ui import app_diff

    common.DB_URI = args.db_uri
    common.TMP = TMP
    TMP.mkdir(parents=True, exist_ok=True)
    shapes = common.load_geometry()
    con = common.connect()
    real = app_diff.split_proposals
    controller = app_diff.AssistController()
    lines = ["| view | pairs | blobs only (median s) | with alternates (median s) "
             "| added (median s) | proposals |", "|---|---|---|---|---|---|"]
    for view in [v for v in args.views.split(",") if v]:
        roi = E.roi_of(shapes, 13, view)
        steps = sorted({s.step for s in shapes if s.desktop == 13 and s.view == view})
        without: list[float] = []
        with_alt: list[float] = []
        counts: list[int] = []
        for step in steps[: max(1, args.cost_pairs)]:
            prev_path = common.frame_path(con, 13, view, step - 1)
            cur_path = common.frame_path(con, 13, view, step)
            if not prev_path or not cur_path:
                continue
            prev_img, cur_img = common.read_rgb(prev_path), common.read_rgb(cur_path)
            if prev_img is None or cur_img is None:
                continue
            try:
                app_diff.split_proposals = lambda *a, **k: []
                t0 = time.perf_counter()
                controller.compute(("k", step), prev_img, cur_img, roi, [])
                without.append(time.perf_counter() - t0)
                app_diff.split_proposals = real
                t0 = time.perf_counter()
                payload = controller.compute(("k", step), prev_img, cur_img, roi, [])
                with_alt.append(time.perf_counter() - t0)
                counts.append(len(payload["proposals"]))
            finally:
                app_diff.split_proposals = real
        if not without:
            continue
        a, b = statistics.median(without), statistics.median(with_alt)
        lines.append(f"| {view} | {len(without)} | {a:.3f} | {b:.3f} | "
                     f"{b - a:.3f} | {statistics.median(counts):.1f} |")
        print(lines[-1], flush=True)
    con.close()
    controller.shutdown()
    OUT.mkdir(parents=True, exist_ok=True)
    text = ("# Added cost of the Shift+C alternates on the diff worker\n\n"
            "D13, real frames, `AssistController.compute` end to end "
            "(dE map + blobs + explain, then the same plus the splitter).\n\n"
            + "\n".join(lines) + "\n")
    (OUT / "table_cost.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=E.DEFAULT_DB,
                    help="path to a COPY of the annotation database (read-only)")
    ap.add_argument("--desktops", default="13,33")
    ap.add_argument("--views", default="scan,oak1,oak2,rs")
    ap.add_argument("--source", default="split", choices=("split", "blob"))
    ap.add_argument("--pad", type=float, default=2.0)
    ap.add_argument("--min-pad", dest="min_pad", type=int, default=32)
    ap.add_argument("--min-area", dest="min_area", type=int, default=40)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--no-prior", dest="prior", action="store_false",
                    help="rank the native candidates without the class's "
                         "area band (which is rescaled to the window when on)")
    ap.add_argument("--shift-px", dest="shift_px", type=int, default=1,
                    help="misregistration tolerated by the local dE map, in "
                         "NATIVE pixels (1 at 1600 px is 2.5 on a 12 MP frame)")
    ap.add_argument("--sam", action="store_true")
    ap.add_argument("--sam-local", dest="sam_local", action="store_true",
                    help="also prompt SAM on a native-resolution local crop")
    ap.add_argument("--grid", action="store_true",
                    help="sweep the knobs (tuning only; never with --sam)")
    ap.add_argument("--cost", action="store_true",
                    help="time the production comparison with/without alternates")
    ap.add_argument("--cost-pairs", dest="cost_pairs", type=int, default=12)
    ap.add_argument("--overlays", action="store_true",
                    help="draw one sheet per event instead of measuring")
    ap.add_argument("--per-class", dest="per_class", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    db = str(args.db)
    args.db_uri = db if db.startswith("file:") else f"file:{db}?mode=ro"
    if args.cost:
        return cost(args)
    if args.overlays:
        return overlays(args)
    if args.grid:
        return grid(args)

    rows = run(args)
    cfg = Native(source=args.source, pad=args.pad, min_pad=args.min_pad,
                 min_area=args.min_area, blur=args.blur,
                 shift_px=args.shift_px, prior=args.prior)
    tag = args.tag or f"d{args.desktops.replace(',', '_')}_{cfg.tag}"
    header = (f"Desktops {args.desktops}, views {args.views}, "
              f"config `{cfg}`. Ground truth, ROI, event resolution and the SAM "
              f"protocol are `diff_eval.py`'s, unchanged.")
    write_outputs(rows, [v for v in args.views.split(",") if v],
                  Path(args.out), tag, header)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
