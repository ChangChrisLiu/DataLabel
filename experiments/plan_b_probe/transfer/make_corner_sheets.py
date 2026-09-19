"""Render the images the four chassis corners are clicked on (M1/M2).

No fiducials and no extrinsics exist, so the only way to a view<->scan
homography is a human picking the same four physical points in both views. This
script renders what that human looks at, in two stages:

``--stage 1``
    the whole reference frame with a labelled pixel grid, so an approximate
    coordinate can be read straight off the image;
``--stage 2``
    a montage of 200-600 px crops centred on the coordinates already written
    into ``corners.yaml``, each with a crosshair and a fine grid, so the click
    can be *checked* against the pixel it claims to be on and corrected.

Stage 2 is the one that makes the clicks honest: at stage-1 display scale one
screen pixel is ~2.7 native pixels on a 4032 px OAK frame, which is already
coarser than the 3 px jitter M3 tests.

Two planes are clicked per view:

``rim``
    the four outer corners of the **chassis opening rim** -- the lip the side
    cover seats on. This is the plane the homography is exact on in M1.
``floor``
    four corners on the **chassis floor / motherboard plane**, roughly a
    chassis height below the rim. This is M2's parallax probe.

Order is fixed and semantic, running clockwise seen from above in the scan
view: ``rear_left, rear_right, front_right, front_left``, where *rear* is the
side carrying the motherboard I/O panel and *front* the drive-bay side.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.plan_b_probe.transfer.common import (  # noqa: E402
    HERE, OUT, banner, connect, frame_path, hstack_pad, read_rgb, save_jpg,
)

#: Corner names are fixed by where the corner appears in **that desktop's scan
#: frame** -- ``s_tl`` is the chassis corner at the top left of the scan. The
#: other views are clicked on the *same physical* corner, located through the
#: rotation between the views, which the M0 overlays pin down from where the
#: annotated parts sit: ``rs`` is in the scan's orientation, ``oak1`` is the
#: scan turned 90 degrees anticlockwise, ``oak2`` the scan turned 180 degrees.
#: Naming them after the scan rather than after "rear"/"front" avoids having to
#: decide which chassis side is the rear in every oblique view.
CORNER_ORDER = ("s_tl", "s_tr", "s_br", "s_bl")
PLANES = ("rim", "floor")

#: The step the corners are clicked on: the **last step on which the motherboard
#: is still in the chassis** in every view of that desktop. Later than that the
#: board is gone and the floor plane has no landmark left to click; earlier than
#: that the chassis is full of parts that hide the rim. The pose check
#: (``m1a_pose_check.py``) found no bump inside any sequence, so one step's
#: clicks are valid for the whole sequence.
REF_STEP = {13: 41, 24: 41, 33: 41}

#: Which (desktop, view) pairs carry LS drafts and therefore need corners.
COMBOS = [(13, v) for v in ("scan", "oak1", "oak2", "rs")] \
    + [(24, v) for v in ("scan", "oak1", "oak2", "rs")] \
    + [(33, v) for v in ("scan", "oak1", "rs")]

GRID = {"scan": 100, "oak1": 200, "oak2": 200, "rs": 50}
CROP = {"scan": 220, "oak1": 520, "oak2": 520, "rs": 180}


def corners_path() -> Path:
    return HERE / "corners.yaml"


def load_corners() -> dict:
    path = corners_path()
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def ref_step(desktop: int, view: str, override: dict) -> int:
    return int(override.get(f"{desktop}/{view}", REF_STEP[desktop]))


def ls_extent(shapes, desktop: int, view: str) -> tuple[int, int, int, int]:
    """Union bbox of every LS shape of a (desktop, view) -- a chassis proxy.

    The corners have to be clicked *somewhere*, and the region the annotators
    drew in is the chassis, so this is a data-derived crop rather than a
    hand-tuned one. Parts that were carried to the bench stretch it outwards,
    which is harmless: it only decides what the sheet shows.
    """
    boxes = [s.box for s in shapes if s.desktop == desktop and s.view == view]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


def tile(img: np.ndarray, box: tuple[int, int, int, int], spacing: int,
         name: str, out_px: int = 700) -> np.ndarray:
    """One quadrant of the chassis region, with every grid line labelled."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    patch = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    factor = out_px / float(max(patch.shape[:2]))
    patch = cv2.resize(patch, (int(round(patch.shape[1] * factor)),
                               int(round(patch.shape[0] * factor))),
                       interpolation=cv2.INTER_CUBIC)
    ph, pw = patch.shape[:2]
    first_x = ((x0 + spacing - 1) // spacing) * spacing
    for gx in range(first_x, x1, spacing):
        px = int(round((gx - x0) * factor))
        cv2.line(patch, (px, 0), (px, ph), (0, 255, 255), 1)
        cv2.putText(patch, str(gx), (px + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(patch, str(gx), (px + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
    first_y = ((y0 + spacing - 1) // spacing) * spacing
    for gy in range(first_y, y1, spacing):
        py = int(round((gy - y0) * factor))
        cv2.line(patch, (0, py), (pw, py), (0, 255, 255), 1)
        cv2.putText(patch, str(gy), (2, py - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(patch, str(gy), (2, py - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return banner(patch, f"{name}  x{x0}-{x1} y{y0}-{y1}  1px={1/factor:.2f}nat",
                  height=26)


def stage1(con, overrides: dict, shapes) -> None:
    """A 2x2 montage of the chassis region's quadrants, per (desktop, view).

    One gridded full frame would be read at ~0.4 native pixels per screen pixel
    on a 12 MP OAK still, which is already coarser than the click error M3
    tests. Quadrants of the chassis region land near 1:1 instead.
    """
    outdir = OUT / "corners" / "stage1"
    for desktop, view in COMBOS:
        step = ref_step(desktop, view, overrides)
        path = frame_path(con, desktop, view, step)
        if path is None:
            print(f"D{desktop} {view} step {step}: no frame")
            continue
        img = read_rgb(path)
        h, w = img.shape[:2]
        x0, y0, x1, y1 = ls_extent(shapes, desktop, view)
        padx, pady = int(0.10 * (x1 - x0)), int(0.10 * (y1 - y0))
        x0, y0 = max(0, x0 - padx), max(0, y0 - pady)
        x1, y1 = min(w, x1 + padx), min(h, y1 + pady)
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        spacing = max(10, int(round((x1 - x0) / 14 / 10)) * 10)
        quads = [
            tile(img, (x0, y0, mx, my), spacing, "TL"),
            tile(img, (mx, y0, x1, my), spacing, "TR"),
            tile(img, (x0, my, mx, y1), spacing, "BL"),
            tile(img, (mx, my, x1, y1), spacing, "BR"),
        ]
        top = hstack_pad(quads[:2], gap=8)
        bottom = hstack_pad(quads[2:], gap=8)
        width = max(top.shape[1], bottom.shape[1])
        canvas = np.full((top.shape[0] + bottom.shape[0] + 8, width, 3), 255, np.uint8)
        canvas[: top.shape[0], : top.shape[1]] = top
        canvas[top.shape[0] + 8 :, : bottom.shape[1]] = bottom
        out = save_jpg(outdir / f"g_d{desktop}_{view}.jpg",
                       banner(canvas, f"D{desktop} {view} step {step} "
                                      f"(frame {w}x{h}, grid {spacing} native px)"),
                       max_side=1560, quality=86)
        print(f"{out}  {out.stat().st_size // 1024} KiB  region "
              f"x{x0}-{x1} y{y0}-{y1} grid {spacing}")


def zoom(img: np.ndarray, cx: int, cy: int, half: int, name: str,
         out_px: int = 420) -> np.ndarray:
    """A crop centred on ``(cx, cy)`` with a crosshair and a fine grid."""
    h, w = img.shape[:2]
    x0, y0 = cx - half, cy - half
    patch = np.zeros((2 * half, 2 * half, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + 2 * half), min(h, y0 + 2 * half)
    if sx1 > sx0 and sy1 > sy0:
        patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    factor = out_px / float(2 * half)
    patch = cv2.resize(patch, (out_px, out_px), interpolation=cv2.INTER_CUBIC)
    # Grid every 1/8 of the crop, labelled in *native* coordinates.
    tick = out_px // 8
    for i in range(1, 8):
        cv2.line(patch, (i * tick, 0), (i * tick, out_px), (0, 255, 255), 1)
        cv2.line(patch, (0, i * tick), (out_px, i * tick), (0, 255, 255), 1)
    step_native = int(round(tick / factor))
    for i in (2, 4, 6):
        label = str(x0 + int(round(i * tick / factor)))
        cv2.putText(patch, label, (i * tick + 3, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(patch, label, (i * tick + 3, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
        label = str(y0 + int(round(i * tick / factor)))
        cv2.putText(patch, label, (3, i * tick - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(patch, label, (3, i * tick - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
    mid = out_px // 2
    cv2.drawMarker(patch, (mid, mid), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
    cv2.circle(patch, (mid, mid), 3, (255, 0, 0), -1)
    return banner(patch, f"{name} ({cx},{cy}) 1px={1/factor:.2f}nat "
                         f"grid={step_native}px", height=30)


def stage2(con, overrides: dict) -> None:
    data = load_corners()
    outdir = OUT / "corners" / "stage2"
    for desktop, view in COMBOS:
        key = f"{desktop}/{view}"
        entry = (data.get("views") or {}).get(key)
        if not entry:
            print(f"{key}: not in corners.yaml yet")
            continue
        step = ref_step(desktop, view, overrides)
        img = read_rgb(frame_path(con, desktop, view, step))
        half = CROP[view] // 2
        rows = []
        for plane in PLANES:
            pts = entry.get(plane) or {}
            tiles = [zoom(img, int(pts[c][0]), int(pts[c][1]), half,
                          f"{plane}.{c}") for c in CORNER_ORDER if c in pts]
            if tiles:
                rows.append(hstack_pad(tiles, gap=6))
        if not rows:
            continue
        width = max(r.shape[1] for r in rows)
        canvas = np.full((sum(r.shape[0] for r in rows) + 8 * len(rows), width, 3),
                         255, np.uint8)
        y = 0
        for r in rows:
            canvas[y : y + r.shape[0], : r.shape[1]] = r
            y += r.shape[0] + 8
        out = save_jpg(outdir / f"z_d{desktop}_{view}.jpg",
                       banner(canvas, f"D{desktop} {view} step {step} - clicked corners"),
                       max_side=1800, quality=86)
        print(f"{out}  {out.stat().st_size // 1024} KiB")


def stage3(con, overrides: dict) -> None:
    """Both clicked quads drawn on the whole frame -- the gross-error check.

    A zoom tile only helps once the click is already within a crop of the truth;
    when it is 200 px out on a 12 MP frame the tile shows bare table and says
    nothing about where the corner actually is. Drawing the quad over the frame
    shows the shape of the mistake instead.
    """
    data = load_corners()
    outdir = OUT / "corners" / "stage3"
    for desktop, view in COMBOS:
        entry = (data.get("views") or {}).get(f"{desktop}/{view}")
        if not entry:
            continue
        step = ref_step(desktop, view, overrides)
        img = read_rgb(frame_path(con, desktop, view, step))
        scale = max(img.shape[:2]) / 1500.0
        for plane, colour in (("rim", (255, 40, 40)), ("floor", (40, 120, 255))):
            pts = entry.get(plane) or {}
            quad = [pts[c] for c in CORNER_ORDER if c in pts]
            if len(quad) != 4:
                continue
            arr = np.asarray(quad, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [arr], True, colour, max(2, int(3 * scale)), cv2.LINE_AA)
            for name, (px, py) in zip(CORNER_ORDER, quad):
                cv2.drawMarker(img, (int(px), int(py)), colour, cv2.MARKER_TILTED_CROSS,
                               int(40 * scale), max(2, int(3 * scale)))
                cv2.putText(img, f"{plane[0]}.{name}", (int(px) + int(12 * scale),
                                                        int(py) - int(12 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (0, 0, 0),
                            int(5 * scale), cv2.LINE_AA)
                cv2.putText(img, f"{plane[0]}.{name}", (int(px) + int(12 * scale),
                                                        int(py) - int(12 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, colour,
                            max(2, int(2 * scale)), cv2.LINE_AA)
        from experiments.plan_b_probe.transfer.common import draw_grid
        img = draw_grid(img, int(200 * scale) if scale > 1 else 100,
                        colour=(0, 255, 0), thickness=1,
                        font_scale=max(0.7, 0.9 * scale))
        out = save_jpg(outdir / f"q_d{desktop}_{view}.jpg",
                       banner(img, f"D{desktop} {view} step {step} - clicked quads "
                                   f"(red rim, blue floor)", height=int(40 * scale)),
                       max_side=1500, quality=84)
        print(f"{out}  {out.stat().st_size // 1024} KiB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--step", action="append", default=[],
                    help="override the reference step, e.g. --step 13/oak2=30")
    args = ap.parse_args()
    overrides = dict(kv.split("=", 1) for kv in args.step)
    con = connect()
    if args.stage == 1:
        from experiments.plan_b_probe.transfer.common import load_geometry
        stage1(con, overrides, load_geometry())
    elif args.stage == 2:
        stage2(con, overrides)
    else:
        stage3(con, overrides)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
