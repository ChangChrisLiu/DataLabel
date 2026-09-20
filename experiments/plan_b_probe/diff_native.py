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
from dataclasses import dataclass, replace
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
    #: Noise floor at native resolution. 80 (the production default) is a
    #: 9x9 patch; a 19 px screw is ~280 px, so this is not the binding limit.
    min_area: int = 40
    blur: int = 5
    shift_px: int = 1

    @property
    def tag(self) -> str:
        return (f"{self.source}_pad{self.pad:g}_min{self.min_area}"
                f"_blur{self.blur}")


def _window(seed: Box, cfg: Native, hw: tuple[int, int]) -> Box:
    """The padded native-resolution window around a seed box, clipped."""
    h, w = int(hw[0]), int(hw[1])
    x0, y0, x1, y1 = (int(v) for v in seed)
    grow = max(int(round(cfg.pad * max(x1 - x0, y1 - y0))), int(cfg.min_pad))
    return (max(0, x0 - grow), max(0, y0 - grow),
            min(w, x1 + grow), min(h, y1 + grow))


def native_proposals(prev_rgb: np.ndarray, cur_rgb: np.ndarray, seed: Box,
                     cfg: Native, *, expect_area=None,
                     max_proposals: int = 3) -> tuple[list[E.Proposal], Box, float]:
    """Re-diff one window at native resolution and rank what is in it.

    Returns ``(proposals, window, seconds)``; the boxes and points are back in
    full-frame coordinates. The window is cropped out of both frames *before*
    the dE map is computed, so the cost is the window's own pixels -- a few
    hundred thousand at most -- and not the 12 MP the ROI-wide pass touches.
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
    parts = propose_parts(sub_prev, sub_cur, None, delta_e=delta,
                          expect_area=expect_area, min_area=cfg.min_area,
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
                 min_area=args.min_area, blur=args.blur)
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
                        prev_img, cur_img, seed, cfg, expect_area=expect)

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
GRID = [
    Native(source=s, pad=p, min_area=m)
    for s in ("split", "blob")
    for p in (1.0, 2.0, 4.0)
    for m in (40, 120)
]


def grid(args) -> int:
    """Sweep the knobs on the tuning desktops; point-in-part and box IoU only.

    Deliberately blind to SAM: the held-out gate is a SAM number, and a knob
    chosen by looking at it would not be held out at all.
    """
    lines = ["| source | pad | min_area | <40 point-in | <40 box IoU | "
             "40-100 point-in | 40-100 box IoU | >=100 box IoU |",
             "|---|---|---|---|---|---|---|---|"]
    for cfg in GRID:
        rows = run(replace(args, source=cfg.source, pad=cfg.pad,
                           min_area=cfg.min_area, sam=False, sam_local=False))
        nat = [r for r in rows if r["method"] == "native"]

        def band(low, high):
            sel = [r for r in nat if low <= float(r["part_size_px"]) < high]
            if not sel:
                return "-", "-"
            hits = sum(int(r.get("p0_point_in") or 0) for r in sel)
            return (f"{100 * hits / len(sel):.1f} %",
                    f"{_median(r.get('p0_box_iou') for r in sel):.3f}")

        small, small_iou = band(0.0, 40.0)
        mid, mid_iou = band(40.0, 100.0)
        _big, big_iou = band(100.0, 1e9)
        lines.append(f"| {cfg.source} | {cfg.pad:g} | {cfg.min_area} | {small} | "
                     f"{small_iou} | {mid} | {mid_iou} | {big_iou} |")
        print(lines[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    text = ("# diff_native tuning grid (desktops "
            f"{args.desktops}, no SAM)\n\n" + "\n".join(lines) + "\n")
    (OUT / "table_grid.md").write_text(text, encoding="utf-8")
    print("\n" + text)
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
            controller = app_diff.AssistController()
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
                controller.shutdown()
        if not without:
            continue
        a, b = statistics.median(without), statistics.median(with_alt)
        lines.append(f"| {view} | {len(without)} | {a:.3f} | {b:.3f} | "
                     f"{b - a:.3f} | {statistics.median(counts):.1f} |")
        print(lines[-1], flush=True)
    con.close()
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
    ap.add_argument("--sam", action="store_true")
    ap.add_argument("--sam-local", dest="sam_local", action="store_true",
                    help="also prompt SAM on a native-resolution local crop")
    ap.add_argument("--grid", action="store_true",
                    help="sweep the knobs (tuning only; never with --sam)")
    ap.add_argument("--cost", action="store_true",
                    help="time the production comparison with/without alternates")
    ap.add_argument("--cost-pairs", dest="cost_pairs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    db = str(args.db)
    args.db_uri = db if db.startswith("file:") else f"file:{db}?mode=ro"
    if args.cost:
        return cost(args)
    if args.grid:
        return grid(args)

    rows = run(args)
    cfg = Native(source=args.source, pad=args.pad, min_pad=args.min_pad,
                 min_area=args.min_area, blur=args.blur)
    tag = args.tag or f"d{args.desktops.replace(',', '_')}_{cfg.tag}"
    header = (f"Desktops {args.desktops}, views {args.views}, "
              f"config `{cfg}`. Ground truth, ROI, event resolution and the SAM "
              f"protocol are `diff_eval.py`'s, unchanged.")
    write_outputs(rows, [v for v in args.views.split(",") if v],
                  Path(args.out), tag, header)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
