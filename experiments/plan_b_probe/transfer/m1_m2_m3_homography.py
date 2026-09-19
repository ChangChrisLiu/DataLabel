"""M1/M2/M3 -- how good is a 4-corner homography as a cross-view hint?

The Plan B proposal is: a human clicks four semantic chassis corners per
(desktop, view), that gives a ``scan -> view`` homography, and the homography
carries an instance's **centre point and box** (never its mask) into the other
views as a SAM prompt and an identity candidate. This measures whether the
carried point actually lands on the part.

What is measured, per instance that the Label Studio drafts place in both the
scan and the target view **at the same step**:

* **HIT** -- the projected scan centroid falls inside the target polygon of the
  same identity. This is the only thing a point prompt has to get right.
* **centre error** -- distance from the projected point to the target polygon's
  centroid, in pixels and divided by ``sqrt(target area)``, which is the part's
  own length scale: 1.0 means "off by the size of the part".
* **box IoU** -- IoU between the target box and the axis-aligned box of the
  projected scan box.

M1 uses the **rim** plane, M2 repeats everything on the **floor** (motherboard)
plane and compares per class group, M3 jitters the four clicks.

Three things keep the number honest:

* Identity. ``ls:<Label>#<n>`` numbers a frame's shapes left to right *within
  that frame* (:func:`tda.core.ls_export.geometry_of` sorts on the percent
  anchor), so for a label with several instances the ``#n`` of one view is not
  the ``#n`` of another. Labels that are unique on the desktop are reported
  separately as the clean number; for the rest the ordinal pairing is compared
  against a Hungarian assignment on projected-centre distance.
* Parts on the bench. Once a part is lifted out it is annotated where it lies on
  the table, which no chassis-plane homography can reach. Rows whose scan
  centroid falls outside the clicked scan quad are flagged ``outside_chassis``
  and excluded from the headline, and counted.
* A floor. ``centre_baseline`` is the HIT rate of prompting with the centre of
  the target's own chassis quad and no cross-view geometry at all. A homography
  that cannot beat it is buying nothing.

Writes ``m1_per_instance.csv``, ``m3_jitter.csv`` and a printed summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.plan_b_probe.transfer.common import (  # noqa: E402
    DESKTOPS, HERE, OUT, TARGETS, Shape, connect, frame_masks, index_shapes,
    load_geometry,
)

CORNER_ORDER = ("s_tl", "s_tr", "s_br", "s_bl")
PLANES = ("rim", "floor")
SIGMAS = (3.0, 6.0, 12.0)
N_JITTER = 200
RNG_SEED = 20260919


# --------------------------------------------------------------------------- #
# homography
# --------------------------------------------------------------------------- #
def load_quads() -> dict[tuple[int, str, str], np.ndarray]:
    """``(desktop, view, plane) -> (4, 2) float32`` of clicked corners."""
    data = yaml.safe_load((HERE / "corners.yaml").read_text(encoding="utf-8"))
    out = {}
    for key, entry in (data.get("views") or {}).items():
        desktop, view = key.split("/")
        for plane in PLANES:
            pts = entry.get(plane)
            if not pts:
                continue
            out[(int(desktop), view, plane)] = np.asarray(
                [pts[c] for c in CORNER_ORDER], dtype=np.float32)
    return out


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to an ``(n, 2)`` array of points."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, h).reshape(-1, 2)


def inside_quad(quad: np.ndarray, point) -> bool:
    return cv2.pointPolygonTest(quad.astype(np.float32), (float(point[0]),
                                                          float(point[1])), False) >= 0


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


# --------------------------------------------------------------------------- #
# masks, cropped
# --------------------------------------------------------------------------- #
def crop_masks(con, desktop: int, view: str, step: int
               ) -> dict[str, tuple[tuple[int, int, int, int], np.ndarray]]:
    """Per instance, its bbox and the mask cropped to it.

    A 12 MP boolean mask is 12 MB; a frame has up to forty instances and the
    jitter loop revisits each of them 1 200 times. Cropping to the box keeps the
    whole frame in a few MB and makes "is this point inside?" two comparisons
    and one array lookup.
    """
    out = {}
    for key, mask in frame_masks(con, desktop, view, step).items():
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        out[key] = (box, np.ascontiguousarray(mask[box[1]:box[3], box[0]:box[2]]))
    return out


def point_in(cropped, point) -> bool:
    (x0, y0, x1, y1), mask = cropped
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if not (x0 <= x < x1 and y0 <= y < y1):
        return False
    return bool(mask[y - y0, x - x0])


def points_in(cropped, pts: np.ndarray) -> np.ndarray:
    """Vectorised :func:`point_in` over an ``(n, 2)`` array."""
    (x0, y0, x1, y1), mask = cropped
    xy = np.rint(pts).astype(np.int64)
    ok = ((xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1))
    out = np.zeros(len(pts), dtype=bool)
    if ok.any():
        sel = xy[ok]
        out[ok] = mask[sel[:, 1] - y0, sel[:, 0] - x0]
    return out


# --------------------------------------------------------------------------- #
# main measurement
# --------------------------------------------------------------------------- #
def unique_labels(shapes: list[Shape]) -> dict[int, set[str]]:
    """Labels that never carry an ordinal above 1 on that desktop."""
    top: dict[tuple[int, str], int] = {}
    for s in shapes:
        key = (s.desktop, s.label)
        top[key] = max(top.get(key, 0), s.ordinal)
    out: dict[int, set[str]] = defaultdict(set)
    for (desktop, label), n in top.items():
        if n == 1:
            out[desktop].add(label)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-jitter", action="store_true", help="skip M3")
    args = ap.parse_args()

    shapes = load_geometry()
    by_frame = index_shapes(shapes)
    uniq = unique_labels(shapes)
    quads = load_quads()
    con = connect()
    rng = np.random.default_rng(RNG_SEED)

    rows: list[dict] = []
    jitter_rows: list[dict] = []

    for desktop in DESKTOPS:
        for view in TARGETS:
            if (desktop, view, "rim") not in quads:
                continue
            steps = sorted({s.step for s in shapes
                            if s.desktop == desktop and s.view == view})
            homs, jit_homs = {}, {}
            for plane in PLANES:
                src = quads[(desktop, "scan", plane)]
                dst = quads[(desktop, view, plane)]
                homs[plane] = homography(src, dst)
                if not args.no_jitter:
                    jit = {}
                    for sigma in SIGMAS:
                        mats = []
                        for _ in range(N_JITTER):
                            noise = rng.normal(0.0, sigma, size=dst.shape)
                            mats.append(homography(src, dst + noise))
                        jit[sigma] = mats
                    jit_homs[plane] = jit
            tgt_quad = quads[(desktop, view, "rim")]
            scan_quad = quads[(desktop, "scan", "rim")]
            centre_prompt = tgt_quad.mean(axis=0)

            for step in steps:
                scan_shapes = by_frame.get((desktop, "scan", step), {})
                tgt_shapes = by_frame.get((desktop, view, step), {})
                common = sorted(set(scan_shapes) & set(tgt_shapes))
                if not common:
                    continue
                cropped = crop_masks(con, desktop, view, step)
                for key in common:
                    src_s, dst_s = scan_shapes[key], tgt_shapes[key]
                    crop = cropped.get(key)
                    if crop is None:
                        continue
                    outside = not inside_quad(scan_quad, src_s.centroid)
                    base = {
                        "desktop": desktop, "view": view, "step": step,
                        "instance": key, "label": src_s.label,
                        "ordinal": src_s.ordinal, "cls": src_s.cls,
                        "group": src_s.group,
                        "unique_label": int(src_s.label in uniq[desktop]),
                        "outside_chassis": int(outside),
                        "scan_area": src_s.area, "tgt_area": dst_s.area,
                        "tgt_size_px": round(dst_s.size, 1),
                        "centre_baseline_hit": int(point_in(crop, centre_prompt)),
                    }
                    for plane in PLANES:
                        h = homs[plane]
                        p = project(h, [src_s.centroid])[0]
                        x0, y0, x1, y1 = src_s.box
                        corners = project(h, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                        pbox = (corners[:, 0].min(), corners[:, 1].min(),
                                corners[:, 0].max(), corners[:, 1].max())
                        dist = float(np.hypot(p[0] - dst_s.centroid[0],
                                              p[1] - dst_s.centroid[1]))
                        base[f"{plane}_hit"] = int(point_in(crop, p))
                        base[f"{plane}_dist_px"] = round(dist, 1)
                        base[f"{plane}_dist_norm"] = round(dist / dst_s.size, 3)
                        base[f"{plane}_iou"] = round(box_iou(pbox, dst_s.box), 3)
                        base[f"{plane}_px"] = round(float(p[0]), 1)
                        base[f"{plane}_py"] = round(float(p[1]), 1)
                    rows.append(base)

                    if args.no_jitter or outside:
                        continue
                    for plane in PLANES:
                        for sigma in SIGMAS:
                            mats = jit_homs[plane][sigma]
                            pts = np.vstack([project(m, [src_s.centroid])
                                             for m in mats])
                            hits = points_in(crop, pts)
                            jitter_rows.append({
                                "desktop": desktop, "view": view, "step": step,
                                "instance": key, "plane": plane, "sigma": sigma,
                                "unique_label": base["unique_label"],
                                "group": base["group"],
                                "n": int(hits.size), "hits": int(hits.sum()),
                            })
            print(f"D{desktop} {view}: {len(rows)} rows so far", flush=True)

    # ---- identity: Hungarian vs the ordinal pairing ------------------------ #
    ident_rows = []
    for desktop in DESKTOPS:
        for view in TARGETS:
            if (desktop, view, "rim") not in quads:
                continue
            h = homography(quads[(desktop, "scan", "rim")], quads[(desktop, view, "rim")])
            steps = sorted({s.step for s in shapes
                            if s.desktop == desktop and s.view == view})
            for step in steps:
                scan_shapes = by_frame.get((desktop, "scan", step), {})
                tgt_shapes = by_frame.get((desktop, view, step), {})
                groups: dict[str, tuple[list[Shape], list[Shape]]] = defaultdict(
                    lambda: ([], []))
                for s in scan_shapes.values():
                    groups[s.label][0].append(s)
                for s in tgt_shapes.values():
                    groups[s.label][1].append(s)
                for label, (src_list, dst_list) in groups.items():
                    if len(src_list) < 2 or len(dst_list) < 2:
                        continue
                    proj = project(h, [s.centroid for s in src_list])
                    cost = np.linalg.norm(
                        proj[:, None, :]
                        - np.asarray([s.centroid for s in dst_list])[None, :, :],
                        axis=2)
                    ri, ci = linear_sum_assignment(cost)
                    for i, j in zip(ri, ci):
                        ident_rows.append({
                            "desktop": desktop, "view": view, "step": step,
                            "label": label,
                            "scan_ordinal": src_list[i].ordinal,
                            "assigned_ordinal": dst_list[j].ordinal,
                            "agrees": int(src_list[i].ordinal == dst_list[j].ordinal),
                            "n_src": len(src_list), "n_dst": len(dst_list),
                        })
    con.close()

    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "m1_per_instance.csv", rows)
    _write(OUT / "m1_identity.csv", ident_rows)
    if jitter_rows:
        _write(OUT / "m3_jitter.csv", jitter_rows)
    print(f"\nwrote {len(rows)} instance rows, {len(ident_rows)} identity rows, "
          f"{len(jitter_rows)} jitter rows")
    return 0


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
