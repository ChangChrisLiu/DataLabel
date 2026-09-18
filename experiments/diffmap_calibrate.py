"""Calibrate ``tda.core.diffmap`` on real scanner frames and draw the overlays.

The unit tests only pin the *contract*; the absolute dE threshold
(:data:`tda.core.diffmap.BLOB_DELTA_E`) is a number about the scanner, and this
script is where it comes from. It compares four kinds of frame pair on one
machine -- a large part leaving, a screw leaving, a frame against itself, and a
frame against a 1 px shift of itself -- and prints the dE percentiles and the
blobs each one produces, so the threshold can be placed between the quiet cases
and the real ones.

It also writes the overlays a human eyeballs. Nothing else writes outside the
repo: the tests deliberately do not, so a test run never litters ``D:``.

Usage::

    source D:/DataSet/envs/activate_tda.sh
    D:/Anaconda/envs/tda/python.exe -m experiments.diffmap_calibrate
    D:/Anaconda/envs/tda/python.exe -m experiments.diffmap_calibrate --desktop 13
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import cv2
import numpy as np
import yaml

from tda.core.cache import cache_path, suggest_roi
from tda.core.diffmap import (
    BLOB_DELTA_E,
    DiffBlob,
    diff_blobs,
    diff_delta_e,
    diff_heat,
    explain_blobs,
    heat_to_rgba,
)
from tda.core.model import FrameKey

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = "D:/DataSet/experiments_out/diffmap"
#: Where the CPU cooler sits in D13's 1600x1600 scanner frame.
COOLER_BOX = (1030, 440, 1290, 700)


def _cache_dir() -> str:
    with open(os.path.join(REPO_ROOT, "configs", "paths.yaml"), encoding="utf-8") as fh:
        return str((yaml.safe_load(fh) or {}).get("cache_dir", "D:/DataSet/cache"))


def _load(desktop: int, step: int) -> Optional[np.ndarray]:
    path = cache_path(_cache_dir(), FrameKey(desktop, step, "scan"), "png")
    bgr = cv2.imread(path)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stats(delta: np.ndarray, roi: tuple[int, int, int, int]) -> dict:
    x0, y0, x1, y1 = roi
    inside = delta[y0:y1, x0:x1]
    return {
        "p50": float(np.percentile(inside, 50)),
        "p99": float(np.percentile(inside, 99)),
        "p99.5": float(np.percentile(inside, 99.5)),
        "max": float(inside.max()),
    }


def _report(name: str, blobs: list[DiffBlob], stats: dict, ms: float) -> None:
    head = (
        f"{name:<28} dE p50={stats['p50']:5.2f} p99={stats['p99']:6.2f} "
        f"p99.5={stats['p99.5']:6.2f} max={stats['max']:6.2f}  {ms:5.0f} ms"
    )
    print(head)
    if not blobs:
        print(f"{'':<28} no blobs")
        return
    for i, blob in enumerate(blobs):
        w, h = blob.box[2] - blob.box[0], blob.box[3] - blob.box[1]
        print(
            f"{'':<28} #{i} box={blob.box} {w}x{h} area={blob.area:6d} "
            f"score={blob.score:8.1f}"
        )


def _overlay(
    frame: np.ndarray,
    heat: np.ndarray,
    blobs: list[DiffBlob],
    expected: Optional[tuple[int, int, int, int]],
    path: str,
) -> None:
    """Blend the heat map over ``frame`` and outline every blob."""
    rgba = heat_to_rgba(heat, alpha_max=200)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    color = rgba[..., [2, 1, 0]].astype(np.float32)
    base = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32)
    out = (base * (1.0 - alpha) + color * alpha).astype(np.uint8)
    for i, blob in enumerate(blobs):
        x0, y0, x1, y1 = blob.box
        shade = (0, 255, 0) if i == 0 else (255, 255, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), shade, 3)
        cv2.putText(
            out, f"{i}:{blob.score:.0f}", (x0, max(14, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, shade, 2,
        )
    if expected is not None:
        cv2.rectangle(out, expected[:2], expected[2:], (255, 0, 255), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out)
    print(f"{'':<28} -> {path}")


def _case(
    name: str,
    before: np.ndarray,
    after: np.ndarray,
    roi: tuple[int, int, int, int],
    out_dir: Optional[str],
    slug: str,
    expected: Optional[tuple[int, int, int, int]] = None,
    min_area: int = 80,
) -> list[DiffBlob]:
    t0 = time.perf_counter()
    delta = diff_delta_e(before, after, roi=roi)
    ms = (time.perf_counter() - t0) * 1000.0
    blobs = diff_blobs(delta, min_area=min_area)
    _report(name, blobs, _stats(delta, roi), ms)
    if expected is not None and blobs:
        explained, unexplained = explain_blobs(blobs, [expected])
        print(
            f"{'':<28} explained={len(explained)} unexplained={len(unexplained)}"
        )
    if out_dir:
        heat = diff_heat(before, after, roi=roi)
        _overlay(after, heat, blobs, expected, os.path.join(out_dir, f"{slug}.png"))
    return blobs


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desktop", type=int, default=13)
    ap.add_argument("--out", default=DEFAULT_OUT, help="overlay dir ('' to skip)")
    ap.add_argument("--cooler-step", type=int, default=13)
    ap.add_argument("--screw-step", type=int, default=3)
    args = ap.parse_args(argv)

    desktop, out_dir = args.desktop, args.out or None
    print(f"diffmap calibration on D{desktop:02d} (BLOB_DELTA_E = {BLOB_DELTA_E})\n")

    cooler_a = _load(desktop, args.cooler_step - 1)
    cooler_b = _load(desktop, args.cooler_step)
    screw_a = _load(desktop, args.screw_step - 1)
    screw_b = _load(desktop, args.screw_step)
    if cooler_a is None or cooler_b is None or screw_a is None or screw_b is None:
        print("scanner cache missing; nothing to calibrate")
        return 1

    roi = suggest_roi(cooler_a, "scan")
    full = (0, 0, cooler_a.shape[1], cooler_a.shape[0])
    print(f"chassis ROI {roi}   full frame {full}\n")

    print("-- real changes --")
    _case("cooler s%03d->s%03d ROI" % (args.cooler_step - 1, args.cooler_step),
          cooler_a, cooler_b, roi, out_dir, "d13_cooler_roi", COOLER_BOX)
    _case("cooler       no ROI", cooler_a, cooler_b, full, out_dir,
          "d13_cooler_full", COOLER_BOX)
    _case("screw  s%03d->s%03d ROI" % (args.screw_step - 1, args.screw_step),
          screw_a, screw_b, roi, out_dir, "d13_screw_roi", min_area=100)
    _case("screw        no ROI", screw_a, screw_b, full, out_dir,
          "d13_screw_full", min_area=100)

    print("\n-- quiet pairs (must stay empty) --")
    _case("same frame twice ROI", cooler_a, cooler_a.copy(), roi, None, "")
    _case("1 px shift ROI", cooler_a, np.roll(cooler_a, 1, axis=1), roi,
          out_dir, "d13_shift_roi")
    _case("2 px shift ROI", cooler_a, np.roll(cooler_a, 2, axis=1), roi, None, "")

    print("\n-- nearby steps (small real changes) --")
    for step in (25, 26, 27):
        a, b = _load(desktop, step - 1), _load(desktop, step)
        if a is None or b is None:
            continue
        _case(f"connector s{step - 1:03d}->s{step:03d} ROI", a, b, roi, None, "")

    _sweep(cooler_a, cooler_b, screw_a, screw_b, roi)
    return 0


def _sweep(
    cooler_a: np.ndarray,
    cooler_b: np.ndarray,
    screw_a: np.ndarray,
    screw_b: np.ndarray,
    roi: tuple[int, int, int, int],
) -> None:
    """Blob counts per candidate threshold: where to put ``BLOB_DELTA_E``."""
    print("\n-- threshold sweep (blobs @ min_area=80) --")
    maps = {
        "cooler": diff_delta_e(cooler_a, cooler_b, roi=roi),
        "screw": diff_delta_e(screw_a, screw_b, roi=roi),
        "shift1": diff_delta_e(cooler_a, np.roll(cooler_a, 1, axis=1), roi=roi),
        "shift2": diff_delta_e(cooler_a, np.roll(cooler_a, 2, axis=1), roi=roi),
        "same": diff_delta_e(cooler_a, cooler_a.copy(), roi=roi),
    }
    print(f"{'dE':>5}  " + "  ".join(f"{k:>16}" for k in maps))
    for thresh in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0):
        cells = []
        for delta in maps.values():
            blobs = diff_blobs(delta, min_delta_e=thresh)
            top = blobs[0] if blobs else None
            cells.append(
                f"{len(blobs)} blob {top.area:6d}px" if top else f"{0} blob      -"
            )
        print(f"{thresh:>5.0f}  " + "  ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
