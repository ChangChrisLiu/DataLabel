"""M4/M5 -- the per-view difference map, and the homography gated by it.

M4 is the competitor that needs no cross-view geometry at all. Frame *k* shows
the state **after** action *k*, so a part removed at step *k* is in frame *k-1*
and gone from frame *k*: the change between those two frames, computed **inside
the target view itself**, should already point at the part. If it does, Plan B
does not need a homography to place a SAM prompt in ``oak1``/``oak2``/``rs`` --
it only needs the two frames that view already has.

M5 combines the two: take the homography's projected point, then snap it to the
nearest of the top-3 diff blobs. The question is whether the diff repairs the
parallax error, or whether a wrong projection just picks a wrong blob.

Which instance an action removed
--------------------------------
``action.target`` carries real instance keys (``screw.motherboard.03``) while
the drafts carry provisional ones (``ls:Motherboard Screw#3``), and nothing
links them. The bridge is ``instance.cls``: the action's target key starts with
its taxonomy class, the drafts carry the same class, so the removed instance is
the draft **of that class that is annotated at step k-1 and not at step k**. An
event with no such draft, or with more than one, is not counted -- ``n`` in the
report is the number of events that resolved to exactly one.

Step alignment
--------------
Label Studio's step numbering was reported to be off by one in places. This
tests it rather than assuming: the located instance's polygon is taken at
anchor step ``k-1+off`` for ``off`` in -1, 0, +1 and the hit rate is reported
per offset. If the numbering were off, a non-zero offset would win.

ROI
---
``diff_delta_e`` wants the chassis box, and no ROI is stored in this database
(``pose_segment.roi_json`` is NULL everywhere). The union box of the view's own
Label Studio shapes, padded 10%, stands in: data-derived, not hand-tuned.

Writes ``m4_per_event.csv`` and a printed summary.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

from experiments.plan_b_probe.transfer.common import (  # noqa: E402
    DESKTOPS, HERE, NATIVE_HW, OUT, VIEWS, connect, frame_path, index_shapes,
    load_geometry, read_rgb,
)
from experiments.plan_b_probe.transfer.m1_m2_m3_homography import (  # noqa: E402
    crop_masks, homography, load_quads, point_in, project,
)

#: Verbs whose step takes a part out of the chassis or unplugs it.
REMOVAL_VERBS = {"remove", "disconnect", "unscrew", "displace"}
#: Longest match wins, so ``screw.cpu_cooler.01`` resolves to ``screw``.
MAX_SIDE = 1600
OFFSETS = (-1, 0, 1)


def action_class(target: str, classes: set[str]) -> str | None:
    """The taxonomy class an ``action.target`` key names, longest prefix first."""
    parts = str(target).split(".")
    for n in range(len(parts), 0, -1):
        candidate = ".".join(parts[:n])
        if candidate in classes:
            return candidate
    return None


def roi_of(shapes, desktop: int, view: str) -> tuple[int, int, int, int]:
    boxes = [s.box for s in shapes if s.desktop == desktop and s.view == view]
    h, w = NATIVE_HW[view]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    px, py = int(0.10 * (x1 - x0)), int(0.10 * (y1 - y0))
    return (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))


def blob_centre(blob) -> tuple[float, float]:
    """Centroid of a blob's actual region, in image coordinates."""
    x0, y0, _x1, _y1 = blob.box
    ys, xs = np.nonzero(blob.mask)
    return (float(x0 + xs.mean()), float(y0 + ys.mean()))


def blob_overlaps(blob, crop) -> bool:
    """Does any pixel of the blob lie inside the instance mask?"""
    (mx0, my0, mx1, my1), mask = crop
    bx0, by0, bx1, by1 = blob.box
    ix0, iy0 = max(mx0, bx0), max(my0, by0)
    ix1, iy1 = min(mx1, bx1), min(my1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    a = mask[iy0 - my0 : iy1 - my0, ix0 - mx0 : ix1 - mx0]
    b = blob.mask[iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0]
    return bool((a & b).any())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", default=",".join(VIEWS))
    args = ap.parse_args()
    views = [v for v in args.views.split(",") if v]

    shapes = load_geometry()
    by_frame = index_shapes(shapes)
    quads = load_quads()
    con = connect()
    classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}

    rows: list[dict] = []
    for desktop in DESKTOPS:
        actions = con.execute(
            "SELECT step, target, verb FROM action WHERE desktop=? ORDER BY step, idx",
            (desktop,)).fetchall()
        for view in views:
            steps_with_ls = {s.step for s in shapes
                             if s.desktop == desktop and s.view == view}
            if not steps_with_ls:
                continue
            roi = roi_of(shapes, desktop, view)
            h, w = NATIVE_HW[view]
            min_area = max(30, int(round(80 * (h * w) / (1600 * 1600))))
            hom = None
            if view != "scan" and (desktop, view, "rim") in quads:
                hom = homography(quads[(desktop, "scan", "rim")],
                                 quads[(desktop, view, "rim")])

            cache: dict[int, np.ndarray] = {}

            def frame(step: int) -> np.ndarray | None:
                if step not in cache:
                    path = frame_path(con, desktop, view, step)
                    cache.clear()
                    cache[step] = None if path is None else read_rgb(path)
                return cache[step]

            for act in actions:
                step = int(act["step"])
                if act["verb"] not in REMOVAL_VERBS or step < 2:
                    continue
                cls = action_class(act["target"], classes)
                if cls is None:
                    continue
                before = by_frame.get((desktop, view, step - 1), {})
                after = by_frame.get((desktop, view, step), {})
                cands = [s for k, s in before.items()
                         if s.cls == cls and k not in after]
                if len(cands) != 1:
                    rows.append({"desktop": desktop, "view": view, "step": step,
                                 "target": act["target"], "verb": act["verb"],
                                 "cls": cls or "", "resolved": 0,
                                 "n_candidates": len(cands)})
                    continue
                inst = cands[0]

                prev_img, cur_img = frame(step - 1), frame(step)
                if prev_img is None or cur_img is None:
                    continue
                delta = diff_delta_e(prev_img, cur_img, roi=roi, max_side=MAX_SIDE)
                blobs = diff_blobs(delta, min_area=min_area, max_blobs=8)
                crops = crop_masks(con, desktop, view, step - 1)
                crop = crops.get(inst.instance)
                if crop is None:
                    continue

                centres = [blob_centre(b) for b in blobs]
                hit_rank = next((i for i, c in enumerate(centres)
                                 if point_in(crop, c)), -1)
                ov_rank = next((i for i, b in enumerate(blobs)
                                if blob_overlaps(b, crop)), -1)

                row = {
                    "desktop": desktop, "view": view, "step": step,
                    "target": act["target"], "verb": act["verb"], "cls": cls,
                    "resolved": 1, "n_candidates": 1,
                    "instance": inst.instance, "label": inst.label,
                    "group": inst.group, "tgt_area": inst.area,
                    "tgt_size_px": round(inst.size, 1),
                    "n_blobs": len(blobs),
                    "top1_hit": int(hit_rank == 0),
                    "top3_hit": int(0 <= hit_rank < 3),
                    "any_hit": int(hit_rank >= 0),
                    "hit_rank": hit_rank,
                    "top1_overlap": int(ov_rank == 0),
                    "top3_overlap": int(0 <= ov_rank < 3),
                    "overlap_rank": ov_rank,
                }
                # step-alignment probe: same blobs, polygon taken one step either way
                for off in OFFSETS:
                    other = by_frame.get((desktop, view, step - 1 + off), {}
                                         ).get(inst.instance)
                    if other is None:
                        row[f"top3_hit_off{off:+d}"] = ""
                        continue
                    oc = crop_masks(con, desktop, view, step - 1 + off).get(
                        inst.instance)
                    if oc is None:
                        row[f"top3_hit_off{off:+d}"] = ""
                        continue
                    r = next((i for i, c in enumerate(centres[:3])
                              if point_in(oc, c)), -1)
                    row[f"top3_hit_off{off:+d}"] = int(r >= 0)

                # M5: homography point, snapped to the nearest of the top-3 blobs
                src = by_frame.get((desktop, "scan", step - 1), {}).get(inst.instance)
                if hom is not None and src is not None:
                    p = project(hom, [src.centroid])[0]
                    row["m1_hit"] = int(point_in(crop, p))
                    row["m1_dist_norm"] = round(
                        float(np.hypot(p[0] - inst.centroid[0],
                                       p[1] - inst.centroid[1])) / inst.size, 3)
                    if centres:
                        top3 = centres[:3]
                        d = [float(np.hypot(c[0] - p[0], c[1] - p[1])) for c in top3]
                        pick = int(np.argmin(d))
                        row["m5_hit"] = int(point_in(crop, top3[pick]))
                        row["m5_pick_rank"] = pick
                    else:
                        row["m5_hit"] = 0
                        row["m5_pick_rank"] = -1
                rows.append(row)
            print(f"D{desktop} {view}: {len(rows)} rows", flush=True)
    con.close()

    OUT.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path = OUT / "m4_per_event.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
