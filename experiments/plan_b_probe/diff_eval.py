"""Task B2 -- how good is the per-view diff as a SAM prompt, before and after?

The E2 probe (``experiments_out/plan_b_probe/transfer/report.md`` section 7)
measured one number on the per-view difference map: the **top-1 blob centroid
lands inside the removed part 21-41 % of the time**, while the blob *overlaps*
it 47-64 % of the time.  That gap is the whole of Task B2: the region is right
far more often than the point, because removing a part reveals a hole, a socket
and a patch of board and :func:`tda.core.diffmap.diff_blobs` merges all of it.

This harness is the measuring stick for any change to that.  It replays the same
201 removal events on the same Label Studio ground truth and reports, per view:

* **point in part** -- does the armed *point* land inside the removed part?
  (the baseline's point is the top-1 blob's centroid, which is what E2 M4
  measured, so this column reproduces the E2 table);
* **box IoU** -- IoU of the armed *box* with the part's own bounding box;
* **SAM IoU** -- IoU of the mask SAM returns for that ``box + point`` prompt
  with the part's mask, which is the only column the annotator actually feels.

Ground truth
------------
The old Label Studio drafts (``ls:*`` shape keyframes) on desktops 13, 24 and
33.  They are **noisy and visible-only** (``amodal_complete=False``), so every
number here is a lower bound on the method and an upper bound on the data;
medians and counts, never means, and look at the overlays before believing a
delta.  Event resolution and geometry loading are reused verbatim from
``experiments/plan_b_probe/transfer`` rather than rewritten.

Direction of the diff (this is the part that is easy to get backwards)
----------------------------------------------------------------------
Annotation runs in **reverse** order.  Standing on frame *j* the task card asks
for the part that is still present in *j* and already gone in *j+1*, and the
canvas compares *j* with *j+1*.  The action table calls that step *k = j+1*, so
here ``step - 1`` is the frame being annotated and the proposal must land on the
part **as it appears in ``step - 1``**.  Ground truth is therefore always the
draft polygon at ``step - 1``.

SAM protocol
------------
Mirrors :mod:`tda.ui.canvas.sam_tools`: the crop is the ROI window scaled to at
most :data:`tda.ui.canvas.sam_crop.MAX_SAM_SIDE`, the prompt is the proposal's
box plus its point as one positive click, ``multimask=True`` (no prior mask), and
the mask taken is ``candidates[0]`` -- the one the tool applies.  Both methods of
one event share the crop, so the embedding is computed once.

Usage::

    python -m experiments.plan_b_probe.diff_eval \
        --db D:/DataSet/.cache/tmp/planb_ro.sqlite \
        --views scan,oak1,oak2,rs --methods baseline --sam

Writes ``events.csv``, ``summary.txt`` and ``table.md`` under
``experiments_out/plan_b_probe/diff_eval/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

from experiments.plan_b_probe.transfer import common  # noqa: E402

OUT = Path("D:/DataSet/experiments_out/plan_b_probe/diff_eval")
TMP = Path("D:/DataSet/.cache/tmp/b2")
DEFAULT_DB = "D:/DataSet/.cache/tmp/planb_ro.sqlite"

#: Long side the pair is compared at -- the production value
#: (``tda.ui.app_diff.MAX_DIFF_SIDE``).
MAX_SIDE = 1600
#: Long side of the SAM crop -- ``tda.ui.canvas.sam_crop.MAX_SAM_SIDE``.
MAX_SAM_SIDE = 1024
#: A box bigger than this share of the ROI is not armed
#: (``tda.ui.app_assist.MAX_PROMPT_BOX_FRAC``).
MAX_PROMPT_BOX_FRAC = 0.6

Box = tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """One removal the drafts resolve to exactly one part."""

    desktop: int
    view: str
    step: int            # the action's step; the annotated frame is step - 1
    verb: str
    target: str
    cls: str
    shape: Any           # common.Shape of the part at step - 1
    group: str


def resolve_events(con, shapes, by_frame, classes, desktop: int,
                   view: str) -> tuple[list[Event], int]:
    """Removal events of one (desktop, view); the second value is unresolved.

    Straight from ``transfer/m4_m5_diff.py``: an event resolves when exactly one
    draft **of the action target's taxonomy class** is annotated at ``step - 1``
    and absent at ``step``.
    """
    from experiments.plan_b_probe.transfer.m4_m5_diff import (
        REMOVAL_VERBS, action_class,
    )

    events: list[Event] = []
    unresolved = 0
    rows = con.execute(
        "SELECT step, target, verb FROM action WHERE desktop=? ORDER BY step, idx",
        (desktop,)).fetchall()
    for act in rows:
        step = int(act["step"])
        if act["verb"] not in REMOVAL_VERBS or step < 2:
            continue
        cls = action_class(act["target"], classes)
        if cls is None:
            continue
        before = by_frame.get((desktop, view, step - 1), {})
        after = by_frame.get((desktop, view, step), {})
        cands = [s for k, s in before.items() if s.cls == cls and k not in after]
        if len(cands) != 1:
            unresolved += 1
            continue
        events.append(Event(desktop=desktop, view=view, step=step,
                            verb=str(act["verb"]), target=str(act["target"]),
                            cls=cls, shape=cands[0], group=cands[0].group))
    return events, unresolved


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def box_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(inter) / float(union) if union else 0.0


def blob_centroid(blob) -> tuple[float, float]:
    """Centroid of the blob's region -- the baseline's point."""
    x0, y0, _x1, _y1 = blob.box
    ys, xs = np.nonzero(blob.mask)
    if xs.size == 0:
        return (0.5 * (blob.box[0] + blob.box[2]), 0.5 * (blob.box[1] + blob.box[3]))
    return (float(x0 + xs.mean()), float(y0 + ys.mean()))


def covers_most(box, roi, limit: float = MAX_PROMPT_BOX_FRAC) -> bool:
    """``tda.ui.app_assist._covers_most`` -- the shipped "do not arm" rule."""
    bw, bh = max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])
    rw, rh = max(1.0, float(roi[2] - roi[0])), max(1.0, float(roi[3] - roi[1]))
    return (bw * bh) > limit * (rw * rh)


def roi_of(shapes, desktop: int, view: str) -> Box:
    """Union box of the view's own drafts, padded 10 % -- the stand-in ROI."""
    from experiments.plan_b_probe.transfer.m4_m5_diff import roi_of as _roi_of

    return _roi_of(shapes, desktop, view)


# --------------------------------------------------------------------------- #
# proposals
# --------------------------------------------------------------------------- #
@dataclass
class Proposal:
    """What a method arms: a box, a point and a rank."""

    box: Box
    point: tuple[float, float]
    score: float
    area: int


def baseline_proposals(delta: np.ndarray, min_area: int,
                       max_proposals: int = 3) -> list[Proposal]:
    """Today's behaviour: ``diff_blobs``, box = blob box, point = blob centroid."""
    blobs = diff_blobs(delta, min_area=min_area, max_blobs=8)
    return [Proposal(box=tuple(int(v) for v in b.box), point=blob_centroid(b),
                     score=float(b.score), area=int(b.area))
            for b in blobs[:max_proposals]]


def split_proposals(prev_rgb, cur_rgb, roi: Box, *, delta: np.ndarray,
                    min_area: int, expect_area, max_proposals: int = 3
                    ) -> list[Proposal]:
    """The candidate replacement: :func:`tda.core.diff_split.propose_parts`."""
    from tda.core.diff_split import propose_parts

    parts = propose_parts(prev_rgb, cur_rgb, roi, expect_area=expect_area,
                          max_proposals=max_proposals, min_area=min_area,
                          delta_e=delta)
    return [Proposal(box=tuple(int(v) for v in p.box),
                     point=(float(p.point[0]), float(p.point[1])),
                     score=float(p.score), area=int(p.area)) for p in parts]


# --------------------------------------------------------------------------- #
# SAM
# --------------------------------------------------------------------------- #
class Sam:
    """One :class:`tda.models.sam_service.SamService`, prompted like the tool."""

    def __init__(self) -> None:
        from tda.models import sam_service

        self._mod = sam_service
        self.service = sam_service.SamService()
        self.calls = 0
        self.seconds = 0.0

    def crop_for(self, image: np.ndarray, roi: Box
                 ) -> tuple[np.ndarray, Box, float]:
        """The crop a zoomed-to-ROI viewport would hand ``viewport_crop``."""
        h, w = image.shape[:2]
        x0, y0, x1, y1 = (max(0, int(roi[0])), max(0, int(roi[1])),
                          min(w, int(roi[2])), min(h, int(roi[3])))
        crop = image[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        scale = 1.0
        if max(ch, cw) > MAX_SAM_SIDE:
            scale = MAX_SAM_SIDE / float(max(ch, cw))
            crop = cv2.resize(crop, (max(1, int(round(cw * scale))),
                                     max(1, int(round(ch * scale)))),
                              interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(crop), (x0, y0, x1, y1), scale

    def predict(self, crop: np.ndarray, rect: Box, scale: float,
                proposal: Proposal, *, with_box: bool) -> Optional[np.ndarray]:
        """SAM's best candidate for ``box + point``, in crop coordinates."""
        x0, y0, x1, y1 = rect
        px, py = proposal.point
        if not (x0 <= px < x1 and y0 <= py < y1):
            return None
        points = [((px - x0) * scale, (py - y0) * scale, 1)]
        box = None
        if with_box:
            bx0 = (min(max(proposal.box[0], x0), x1) - x0) * scale
            by0 = (min(max(proposal.box[1], y0), y1) - y0) * scale
            bx1 = (min(max(proposal.box[2], x0), x1) - x0) * scale
            by1 = (min(max(proposal.box[3], y0), y1) - y0) * scale
            if bx1 - bx0 >= 1.0 and by1 - by0 >= 1.0:
                box = (bx0, by0, bx1, by1)
        req = self._mod.SamRequest(image_crop=crop, points=points, box=box,
                                   mask_input=None, multimask=True)
        t0 = time.perf_counter()
        result = self.service.predict(req)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return np.asarray(result.candidates[0], dtype=bool)


def gt_in_crop(mask_full: np.ndarray, rect: Box, crop_hw: tuple[int, int]
               ) -> np.ndarray:
    """The ground-truth mask cropped to ``rect`` and scaled to the crop grid."""
    x0, y0, x1, y1 = rect
    sub = mask_full[y0:y1, x0:x1]
    if sub.shape != tuple(crop_hw):
        sub = cv2.resize(sub.astype(np.uint8), (crop_hw[1], crop_hw[0]),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    return np.ascontiguousarray(sub, dtype=bool)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def priors_for(cls: str, priors) -> Optional[tuple[float, float]]:
    """``(p05, p95)`` area fractions of the ROI for a class, or ``None``."""
    row = priors.classes.get(str(cls))
    if not isinstance(row, dict):
        return None
    try:
        return float(row["p05_frac"]), float(row["p95_frac"])
    except (KeyError, TypeError, ValueError):
        return None


def run(args) -> list[dict]:
    from tda.ui.app_priors import load_priors

    common.DB_URI = args.db_uri
    common.TMP = TMP
    TMP.mkdir(parents=True, exist_ok=True)

    shapes = common.load_geometry()
    by_frame = common.index_shapes(shapes)
    priors = load_priors()
    con = common.connect()
    classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}

    views = [v for v in args.views.split(",") if v]
    methods = [m for m in args.methods.split(",") if m]
    sam = Sam() if args.sam else None

    rows: list[dict] = []
    for desktop in common.DESKTOPS:
        for view in views:
            present = {s.step for s in shapes
                       if s.desktop == desktop and s.view == view}
            if not present:
                continue
            roi = roi_of(shapes, desktop, view)
            roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
            h, w = common.NATIVE_HW[view]
            min_area = max(30, int(round(80 * (h * w) / (1600 * 1600))))
            events, unresolved = resolve_events(con, shapes, by_frame, classes,
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
                t_load = time.perf_counter()
                prev_img = frame(ev.step - 1)     # the annotated frame
                cur_img = frame(ev.step)          # the one the card diffs against
                load_s = time.perf_counter() - t_load
                if prev_img is None or cur_img is None:
                    continue
                gt = common.frame_masks(con, desktop, view, ev.step - 1
                                        ).get(ev.shape.instance)
                if gt is None or not gt.any():
                    continue

                t_delta = time.perf_counter()
                delta = diff_delta_e(prev_img, cur_img, roi=roi, max_side=MAX_SIDE)
                delta_s = time.perf_counter() - t_delta

                crop = rect = scale = gt_crop = None
                if sam is not None:
                    crop, rect, scale = sam.crop_for(prev_img, roi)
                    gt_crop = gt_in_crop(gt, rect, crop.shape[:2])

                want = 40 if args.diag else 3
                for method in methods:
                    t0 = time.perf_counter()
                    if method == "baseline":
                        props = baseline_proposals(delta, min_area,
                                                   max_proposals=want)
                    else:
                        props = split_proposals(
                            prev_img, cur_img, roi, delta=delta,
                            min_area=min_area, max_proposals=want,
                            expect_area=priors_for(ev.cls, priors))
                    propose_s = time.perf_counter() - t0

                    row = {
                        "desktop": desktop, "view": view, "step": ev.step,
                        "method": method, "target": ev.target, "verb": ev.verb,
                        "cls": ev.cls, "group": ev.group,
                        "instance": ev.shape.instance, "label": ev.shape.label,
                        "part_area": ev.shape.area,
                        "part_size_px": round(ev.shape.size, 1),
                        "part_frac_roi": round(ev.shape.area / roi_area, 6),
                        "n_proposals": len(props),
                        "load_s": round(load_s, 3),
                        "delta_s": round(delta_s, 3),
                        "propose_s": round(propose_s, 3),
                    }
                    gt_box = tuple(int(v) for v in ev.shape.box)
                    for rank, p in enumerate(props[:3]):
                        inside = bool(
                            0 <= int(round(p.point[1])) < gt.shape[0]
                            and 0 <= int(round(p.point[0])) < gt.shape[1]
                            and gt[int(round(p.point[1])), int(round(p.point[0]))])
                        row[f"p{rank}_point_in"] = int(inside)
                        row[f"p{rank}_box_iou"] = round(box_iou(p.box, gt_box), 4)
                        row[f"p{rank}_armed"] = int(not covers_most(p.box, roi))
                        row[f"p{rank}_area"] = p.area
                        row[f"p{rank}_box"] = json.dumps(list(p.box))
                        row[f"p{rank}_point"] = json.dumps(
                            [round(p.point[0], 1), round(p.point[1], 1)])
                    if args.diag:
                        # Does the *candidate set* hold the answer at all, and
                        # at what rank?  Separates "cannot see the part" from
                        # "sees it and ranks it third".
                        ranks = [i for i, p in enumerate(props)
                                 if 0 <= int(round(p.point[1])) < gt.shape[0]
                                 and 0 <= int(round(p.point[0])) < gt.shape[1]
                                 and gt[int(round(p.point[1])),
                                        int(round(p.point[0]))]]
                        row["oracle_rank"] = ranks[0] if ranks else -1
                        ious = [box_iou(p.box, gt_box) for p in props]
                        row["oracle_box_iou"] = round(max(ious), 4) if ious else 0.0
                        row["oracle_box_rank"] = (
                            int(np.argmax(ious)) if ious else -1)
                        row["n_cands"] = len(props)
                    if props and sam is not None:
                        top = props[0]
                        armed = not covers_most(top.box, roi)
                        t1 = time.perf_counter()
                        mask = sam.predict(crop, rect, scale, top,
                                           with_box=armed)
                        row["sam_s"] = round(time.perf_counter() - t1, 3)
                        row["sam_with_box"] = int(armed)
                        row["sam_iou"] = (round(mask_iou(mask, gt_crop), 4)
                                          if mask is not None else "")
                    rows.append(row)
            cache.clear()
    con.close()
    if sam is not None:
        print(f"SAM: {sam.calls} calls, {sam.seconds:.1f} s "
              f"({1000 * sam.seconds / max(1, sam.calls):.0f} ms each)")
    return rows


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def quartiles(values: Iterable[float]) -> tuple[float, float, float, int]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return (float("nan"), float("nan"), float("nan"), 0)
    med = statistics.median(vals)
    if len(vals) >= 2:
        low = statistics.median(vals[: len(vals) // 2])
        high = statistics.median(vals[(len(vals) + 1) // 2:])
    else:
        low = high = med
    return (low, med, high, len(vals))


def summarise(rows: list[dict], views: list[str], methods: list[str]) -> str:
    lines: list[str] = []
    lines.append("| view | method | n | point-in-part top-1 | top-3 | "
                 "box IoU p25/med/p75 | SAM IoU p25/med/p75 |")
    lines.append("|---|---|---|---|---|---|---|")
    for view in views:
        for method in methods:
            sel = [r for r in rows if r["view"] == view and r["method"] == method]
            if not sel:
                continue
            n = len(sel)
            top1 = sum(int(r.get("p0_point_in") or 0) for r in sel)
            top3 = sum(int(any(int(r.get(f"p{i}_point_in") or 0) for i in range(3)))
                       for r in sel)
            bious = [float(r["p0_box_iou"]) for r in sel
                     if r.get("p0_box_iou") not in (None, "")]
            sious = [float(r["sam_iou"]) for r in sel
                     if r.get("sam_iou") not in (None, "")]
            bq, sq = quartiles(bious), quartiles(sious)
            lines.append(
                f"| {view} | {method} | {n} | {100 * top1 / n:.1f}% | "
                f"{100 * top3 / n:.1f}% | "
                f"{bq[0]:.3f} / **{bq[1]:.3f}** / {bq[2]:.3f} | "
                + (f"{sq[0]:.3f} / **{sq[1]:.3f}** / {sq[2]:.3f} (n={sq[3]})"
                   if sq[3] else "-") + " |")
    return "\n".join(lines)


def group_table(rows: list[dict], methods: list[str]) -> str:
    lines = ["| group | method | n | point-in-part | box IoU med | SAM IoU med |",
             "|---|---|---|---|---|---|"]
    for group in ("tall", "flat", "rim", "other"):
        for method in methods:
            sel = [r for r in rows
                   if r["group"] == group and r["method"] == method]
            if not sel:
                continue
            n = len(sel)
            top1 = sum(int(r.get("p0_point_in") or 0) for r in sel)
            bious = [float(r["p0_box_iou"]) for r in sel
                     if r.get("p0_box_iou") not in (None, "")]
            sious = [float(r["sam_iou"]) for r in sel
                     if r.get("sam_iou") not in (None, "")]
            lines.append(
                f"| {group} | {method} | {n} | {100 * top1 / n:.1f}% | "
                f"{quartiles(bious)[1]:.3f} | "
                + (f"{quartiles(sious)[1]:.3f}" if sious else "-") + " |")
    return "\n".join(lines)


def runtime_table(rows: list[dict], views: list[str], methods: list[str]) -> str:
    lines = ["| view | method | n | frame load s | delta s | propose s | SAM s |",
             "|---|---|---|---|---|---|---|"]
    for view in views:
        for method in methods:
            sel = [r for r in rows if r["view"] == view and r["method"] == method]
            if not sel:
                continue

            def med(key: str) -> str:
                vals = [float(r[key]) for r in sel if r.get(key) not in (None, "")]
                return f"{statistics.median(vals):.3f}" if vals else "-"

            lines.append(f"| {view} | {method} | {len(sel)} | {med('load_s')} | "
                         f"{med('delta_s')} | {med('propose_s')} | {med('sam_s')} |")
    return "\n".join(lines)


def write_outputs(rows: list[dict], views: list[str], methods: list[str],
                  out: Path, tag: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
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

    text = "\n\n".join([
        f"# diff_eval ({tag})",
        "## per view", summarise(rows, views, methods),
        "## per class group", group_table(rows, methods),
        "## runtime (median seconds per event)",
        runtime_table(rows, views, methods),
    ])
    (out / f"table_{tag}.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {csv_path}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB,
                    help="path to a COPY of the annotation database (read-only)")
    ap.add_argument("--views", default="scan,oak1,oak2,rs")
    ap.add_argument("--methods", default="baseline",
                    help="baseline and/or split, comma separated")
    ap.add_argument("--sam", action="store_true", help="also measure SAM IoU")
    ap.add_argument("--limit", type=int, default=0,
                    help="at most this many events per (desktop, view)")
    ap.add_argument("--diag", action="store_true",
                    help="also record how deep in the ranking the answer sits")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    db = str(args.db)
    args.db_uri = db if db.startswith("file:") else f"file:{db}?mode=ro"
    rows = run(args)
    views = [v for v in args.views.split(",") if v]
    methods = [m for m in args.methods.split(",") if m]
    tag = args.tag or ("_".join(methods) + ("_sam" if args.sam else ""))
    write_outputs(rows, views, methods, Path(args.out), tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
