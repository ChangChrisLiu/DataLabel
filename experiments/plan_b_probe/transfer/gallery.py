"""Render the failure gallery: what a missed cross-view hint actually looks like.

Each panel pair is one (desktop, view, step, instance):

* left  -- the **scan** frame with the source polygon (green) and the centroid
  the hint is built from (green cross), plus the clicked rim quad (red) and
  floor quad (blue) so the reader can see which plane the part sits on;
* right -- the **target** frame with that instance's own polygon (green), the
  rim-plane projection (red cross) and the floor-plane projection (blue cross),
  and, for a diff case, the top-3 change blobs (orange boxes, thickest first).

Cases are chosen from the measurement CSVs by rule, so the gallery is
reproducible rather than curated by hand; ``--list`` prints the candidates and
their numbers without rendering.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

from experiments.plan_b_probe.transfer.common import (  # noqa: E402
    NATIVE_HW, OUT, banner, connect, draw_shapes, frame_path, hstack_pad,
    index_shapes, load_geometry, read_rgb, save_jpg,
)
from experiments.plan_b_probe.transfer.m1_m2_m3_homography import (  # noqa: E402
    CORNER_ORDER, homography, load_quads, project,
)
from experiments.plan_b_probe.transfer.m4_m5_diff import roi_of  # noqa: E402

PANEL = 820


def read_csv(name: str) -> list[dict]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict, key: str, default=float("nan")) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else default


def pick_cases() -> list[dict]:
    """Rule-chosen cases, one per failure mode, spread over views and desktops."""
    rows = read_csv("m1_per_instance.csv")
    events = [r for r in read_csv("m4_per_event.csv") if r.get("resolved") == "1"]
    cases: list[dict] = []

    def add(row, why, extra=None):
        if row is None:
            return
        case = {"desktop": int(row["desktop"]), "view": row["view"],
                "step": int(row["step"]), "instance": row["instance"], "why": why}
        if extra:
            case.update(extra)
        cases.append(case)

    uniq = [r for r in rows if r["unique_label"] == "1" and r["outside_chassis"] == "0"]
    # 1-3: the biggest parallax miss per view, on a part whose two planes disagree
    for view in ("oak1", "oak2", "rs"):
        sel = [r for r in uniq if r["view"] == view
               and abs(fnum(r, "rim_dist_px") - fnum(r, "floor_dist_px")) > 40]
        if sel:
            add(max(sel, key=lambda r: min(fnum(r, "rim_dist_px"),
                                           fnum(r, "floor_dist_px"))),
                f"parallax: the two planes disagree and both miss ({view})")
    # 4: a small flat part that both planes miss although the error is modest
    small = [r for r in uniq if fnum(r, "tgt_size_px") < 60
             and r["rim_hit"] == "0" and r["floor_hit"] == "0"]
    if small:
        add(min(small, key=lambda r: fnum(r, "rim_dist_px")),
            "small part: the error is ordinary but the part is smaller than it")
    # 5: a part annotated outside the chassis quad
    out_rows = [r for r in rows if r["outside_chassis"] == "1"]
    if out_rows:
        add(max(out_rows, key=lambda r: fnum(r, "rim_dist_px")),
            "on the bench: the part has left the plane the homography knows")
    # 6-7: ordinal identity mismatch -- same key, wildly different places
    multi = [r for r in rows if r["unique_label"] == "0"
             and r["outside_chassis"] == "0" and fnum(r, "rim_dist_norm") > 8]
    for view in ("oak1", "rs"):
        sel = [r for r in multi if r["view"] == view]
        if sel:
            add(max(sel, key=lambda r: fnum(r, "rim_dist_norm")),
                f"identity: the #ordinal pairs different physical parts ({view})")
    # 8: a case the rim plane gets right and the floor plane gets wrong
    swap = [r for r in uniq if r["rim_hit"] == "1" and r["floor_hit"] == "0"
            and fnum(r, "floor_dist_px") > 150]
    if swap:
        add(max(swap, key=lambda r: fnum(r, "floor_dist_px")),
            "wrong plane: a part on the rim projected with the floor homography")
    # 9: and the reverse
    swap = [r for r in uniq if r["floor_hit"] == "1" and r["rim_hit"] == "0"
            and fnum(r, "rim_dist_px") > 150]
    if swap:
        add(max(swap, key=lambda r: fnum(r, "rim_dist_px")),
            "wrong plane: a part on the board projected with the rim homography")
    # 10-12: diff-map failures -- the strongest blob is not the part
    for view in ("oak2", "oak1", "rs", "scan"):
        sel = [r for r in events if r["view"] == view and r["top3_hit"] == "0"]
        if sel:
            row = max(sel, key=lambda r: fnum(r, "tgt_area"))
            add({"desktop": row["desktop"], "view": row["view"],
                 "step": str(int(row["step"]) - 1), "instance": row["instance"]},
                f"diff map: no top-3 blob lands on the removed part ({view}, "
                f"{row['label']} at step {row['step']})",
                {"diff_step": int(row["step"])})
    return cases[:12]


def draw_quad(img, quad, colour, thickness):
    cv2.polylines(img, [np.asarray(quad, np.int32).reshape(-1, 1, 2)], True,
                  colour, thickness, cv2.LINE_AA)


def shrink(img, out_px=PANEL):
    h, w = img.shape[:2]
    f = out_px / float(max(h, w))
    return cv2.resize(img, (int(round(w * f)), int(round(h * f))),
                      interpolation=cv2.INTER_AREA)


def render(con, case, by_frame, quads) -> np.ndarray | None:
    desktop, view, step = case["desktop"], case["view"], int(case["step"])
    key = case["instance"]
    src = by_frame.get((desktop, "scan", step), {}).get(key)
    dst = by_frame.get((desktop, view, step), {}).get(key)
    if dst is None:
        return None

    panels = []
    if src is not None:
        img = read_rgb(frame_path(con, desktop, "scan", step))
        t = 3
        draw_quad(img, [quads[(desktop, "scan", "rim")][i] for i in range(4)],
                  (255, 40, 40), t)
        draw_quad(img, [quads[(desktop, "scan", "floor")][i] for i in range(4)],
                  (40, 120, 255), t)
        img = draw_shapes(img, [src], thickness=4, label=False)
        cv2.drawMarker(img, (int(src.centroid[0]), int(src.centroid[1])),
                       (0, 230, 0), cv2.MARKER_CROSS, 60, 5)
        panels.append(banner(shrink(img), f"scan D{desktop} step {step}  {key}",
                             height=34))

    img = read_rgb(frame_path(con, desktop, view, step))
    scale = max(img.shape[:2]) / float(PANEL)
    t = max(3, int(round(3 * scale)))
    draw_quad(img, quads[(desktop, view, "rim")], (255, 40, 40), t)
    draw_quad(img, quads[(desktop, view, "floor")], (40, 120, 255), t)
    img = draw_shapes(img, [dst], thickness=max(4, int(round(4 * scale))), label=False)
    if src is not None:
        for plane, colour in (("rim", (255, 40, 40)), ("floor", (40, 120, 255))):
            h = homography(quads[(desktop, "scan", plane)], quads[(desktop, view, plane)])
            p = project(h, [src.centroid])[0]
            cv2.drawMarker(img, (int(p[0]), int(p[1])), colour,
                           cv2.MARKER_TILTED_CROSS, int(70 * scale), t + 2)
    if case.get("diff_step"):
        k = int(case["diff_step"])
        prev_img = read_rgb(frame_path(con, desktop, view, k - 1))
        cur_img = read_rgb(frame_path(con, desktop, view, k))
        if prev_img is not None and cur_img is not None:
            shapes_all = [s for s in by_frame.values() for s in s.values()]
            roi = roi_of(shapes_all, desktop, view)
            hh, ww = NATIVE_HW[view]
            delta = diff_delta_e(prev_img, cur_img, roi=roi, max_side=1600)
            blobs = diff_blobs(delta, min_area=max(30, int(80 * hh * ww / 1600 ** 2)),
                               max_blobs=8)[:3]
            for i, b in enumerate(blobs):
                cv2.rectangle(img, (b.box[0], b.box[1]), (b.box[2], b.box[3]),
                              (255, 150, 0), t + 3 - i)
                cv2.putText(img, f"#{i + 1}", (b.box[0], max(20, b.box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 150, 0),
                            t + 1, cv2.LINE_AA)
    panels.append(banner(shrink(img), f"{view} D{desktop} step {step}  "
                                      f"(red=rim, blue=floor, green=LS truth)",
                         height=34))
    sheet = hstack_pad(panels)
    return banner(sheet, case["why"][:110], height=40)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cases = pick_cases()
    if args.list:
        for i, c in enumerate(cases, 1):
            print(i, c)
        return 0
    shapes = load_geometry()
    by_frame = index_shapes(shapes)
    quads = load_quads()
    con = connect()
    outdir = OUT / "gallery"
    for i, case in enumerate(cases, 1):
        sheet = render(con, case, by_frame, quads)
        if sheet is None:
            print(f"{i}: could not render {case}")
            continue
        path = save_jpg(outdir / f"fail_{i:02d}_{case['view']}_d{case['desktop']}.jpg",
                        sheet, max_side=1700, quality=80)
        print(f"{path}  {path.stat().st_size // 1024} KiB  -- {case['why']}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
