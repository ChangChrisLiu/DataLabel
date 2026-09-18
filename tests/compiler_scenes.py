"""Builders shared by the layer-compiler tests (not a test module itself).

Everything here is synthetic and 64x64: no fixtures, no I/O. Imported by
``test_compiler.py`` (the compiler end to end) and ``test_compiler_layers.py``
(the z-order, its overrides and their cycles).
"""
from __future__ import annotations

import numpy as np

from tda.core import masks
from tda.core.compiler import CompiledFrame, compile_frame
from tda.core.model import (
    FrameKey,
    PairOverride,
    Placement,
    ShapeKeyframe,
    ShapePart,
    Similarity,
    ZOrderRec,
)

HW = (64, 64)
KEY = FrameKey(desktop=1, step=5, view="scan")
IDENTITY = Similarity()
BENCH = Placement.ON_BENCH.value
CHASSIS = Placement.IN_CHASSIS.value

#: A at (8..40), B at (20..52); B is on top in the global order.
A_BOX = (8, 8, 40, 40)
B_BOX = (20, 20, 52, 52)


def rect(x0: int, y0: int, x1: int, y1: int, hw=HW) -> np.ndarray:
    """A filled rectangle ``[x0, x1) x [y0, y1)`` as a bool mask."""
    m = np.zeros(hw, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def kf(
    instance: str,
    rects: dict[str, tuple[int, int, int, int]],
    *,
    anchor_step: int = 10,
    kf_id: int | None = 1,
    version: int = 1,
    placement: str = CHASSIS,
    geom_type: str = "mask",
    view: str = "scan",
    pose_segment: int = 1,
    hw=HW,
) -> ShapeKeyframe:
    """A keyframe whose parts are the given rectangles (mask or box geometry)."""
    parts = []
    for name, box in rects.items():
        if geom_type == "box":
            parts.append(ShapePart(name, None, tuple(float(v) for v in box)))
        else:
            parts.append(ShapePart(name, masks.encode_rle(rect(*box, hw=hw)), box))
    return ShapeKeyframe(
        id=kf_id,
        instance=instance,
        desktop=1,
        view=view,
        pose_segment=pose_segment,
        anchor_step=anchor_step,
        placement=placement,
        geom_type=geom_type,
        parts=parts,
        version=version,
    )


def zo(order: list[tuple[str, str]], version: int = 1) -> ZOrderRec:
    return ZOrderRec(desktop=1, view="scan", pose_segment=1, order=order, version=version)


def po(above_key: str, below_key: str) -> PairOverride:
    return PairOverride(1, "scan", 1, above=above_key, below=below_key)


def run(
    keyframes: dict[str, list[ShapeKeyframe]],
    zorder: ZOrderRec,
    *,
    needs: dict[str, str] | None = None,
    overrides=(),
    occluders=(),
    frame_overrides=None,
    transform: Similarity = IDENTITY,
    placements: dict[str, str] | None = None,
    pose_segment: int | None = None,
    compiler_version: str = "1",
    key: FrameKey = KEY,
    bench_roi=(0, 0, HW[1], HW[0]),
) -> CompiledFrame:
    """compile_frame with the boilerplate filled in (needs defaults to masks).

    ``bench_roi`` defaults to the whole canvas, i.e. a view that *can* see the
    staging area, because that is what the bench cases here are about; pass
    ``None`` for a view like the scanner, which sees none and is therefore never
    asked for a bench box (spec 4.2 item 1).
    """
    if needs is None:
        needs = {inst: "mask" for inst in keyframes}
    return compile_frame(
        key,
        HW,
        needs,
        keyframes,
        zorder,
        list(overrides),
        list(occluders),
        dict(frame_overrides or {}),
        transform,
        compiler_version,
        placements=placements,
        pose_segment=pose_segment,
        bench_roi=None if bench_roi is None else list(bench_roi),
    )


def two_squares() -> dict:
    """A (8..40) under B (20..52), both masks in the chassis."""
    return {
        "keyframes": {
            "A": [kf("A", {"main": A_BOX}, kf_id=11)],
            "B": [kf("B", {"main": B_BOX}, kf_id=22)],
        },
        "zorder": zo([("A", "main"), ("B", "main")]),
    }


def three_squares() -> dict:
    """A < B < C in the global order, all three overlapping."""
    return {
        "keyframes": {
            "A": [kf("A", {"main": (8, 8, 40, 40)}, kf_id=1)],
            "B": [kf("B", {"main": (16, 16, 48, 48)}, kf_id=2)],
            "C": [kf("C", {"main": (24, 24, 56, 56)}, kf_id=3)],
        },
        "zorder": zo([("A", "main"), ("B", "main"), ("C", "main")]),
    }


def covered() -> dict:
    """A small instance completely hidden under a big cover."""
    return {
        "keyframes": {
            "small": [kf("small", {"main": (20, 20, 30, 30)}, kf_id=1)],
            "cover": [kf("cover", {"main": (10, 10, 50, 50)}, kf_id=2)],
        },
        "zorder": zo([("small", "main"), ("cover", "main")]),
    }
