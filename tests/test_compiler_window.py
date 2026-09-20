"""The compiler's per-shape windows must never be smaller than the shape.

:mod:`tda.core.compiler` composites inside a window per layer instead of over
the canvas, and the whole thing stands on one property: **a window contains
every pixel the unwindowed computation would have produced**. Too large only
costs time; too small silently truncates stored geometry, and nothing
downstream would notice -- the RLE simply comes out missing a corner.

:mod:`tests.test_compiler_golden` pins the answers for a fixed set of scenes,
which catches a window that is wrong *there*. This module attacks the bound
itself, by brute force: for every transform it builds the mask both ways --
windowed, and full-frame with no window at all -- and checks the window really
holds the second one. The warp is the case that matters, because
``cv2.warpAffine`` with nearest-neighbour sampling spreads one source pixel
over a ``scale x scale`` block of the output, so a fixed slack is only correct
while the scale is near one.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tda.core import masks
from tda.core.compiler import (
    WARP_SLACK,
    WARP_SPREAD,
    _clip_window,
    _part_mask,
    _part_window,
    _transform_box,
    warp_slack,
)
from tda.core.model import ShapePart, Similarity

HW = (96, 128)

#: The sweep: 6 scales x 5 angles x 4 sub-pixel offsets x 6 shapes = 720 warps.
SCALES = (1.0, 1.7, 3.0, 5.0, 8.0, 12.0)
THETAS = (0.0, 0.3, 0.7, 1.0, 1.3)
OFFSETS = ((0.0, 0.0), (0.37, -0.62), (-11.5, 7.25), (23.9, 31.1))
SHAPES = (
    (4, 4, 10, 10),          # a small square near the origin
    (0, 0, 3, 3),            # touching the corner
    (40, 30, 90, 70),        # a big block in the middle
    (120, 88, 128, 96),      # touching the far corner
    (60, 10, 63, 80),        # a tall thin sliver
    (10, 44, 110, 47),       # a wide thin sliver
)


def _part(box) -> ShapePart:
    mask = np.zeros(HW, dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return ShapePart("main", masks.encode_rle(mask), box)


def _cases():
    for box in SHAPES:
        part = _part(box)
        for scale in SCALES:
            for theta in THETAS:
                for tx, ty in OFFSETS:
                    yield part, Similarity(scale=scale, theta=theta, tx=tx, ty=ty)


def test_the_sweep_is_the_size_it_claims():
    assert len(list(_cases())) == 720


@pytest.mark.parametrize("scale", SCALES)
def test_a_warped_shape_never_reaches_outside_its_window(scale: float):
    """Both ways round, for every angle, offset and shape at this scale."""
    for part, transform in _cases():
        if transform.scale != scale:
            continue
        window = _part_window(part, transform, HW)
        assert window is not None

        full = _part_mask(part, transform, HW)          # no window at all
        reached = masks.bbox(full)
        if reached is None:
            continue                                     # warped off the canvas
        assert (window[0] <= reached[0] and window[1] <= reached[1]
                and window[2] >= reached[2] and window[3] >= reached[3]), (
            f"window {window} misses {reached} at scale={transform.scale} "
            f"theta={transform.theta} t=({transform.tx}, {transform.ty}) "
            f"box={part.box}"
        )

        # ... and the windowed array really is what the window says it holds
        cropped = _part_mask(part, transform, HW, window)
        assert cropped.shape == (window[3] - window[1], window[2] - window[0])
        assert np.array_equal(cropped, full[window[1]:window[3], window[0]:window[2]])


def test_the_slack_grows_with_the_scale_and_never_drops_below_the_floor():
    assert warp_slack(Similarity()) == WARP_SLACK + math.ceil(WARP_SPREAD)
    assert warp_slack(Similarity(scale=12.0)) == WARP_SLACK + math.ceil(12 * WARP_SPREAD)
    assert warp_slack(Similarity(scale=0.1)) >= WARP_SLACK
    # a negative scale is not something the model produces, but the slack must
    # not come out negative if one ever arrives
    assert warp_slack(Similarity(scale=-4.0)) >= WARP_SLACK


def _holds(window, reached) -> bool:
    return (window[0] <= reached[0] and window[1] <= reached[1]
            and window[2] >= reached[2] and window[3] >= reached[3])


def test_a_fixed_slack_of_two_would_have_truncated_real_warps():
    """The regression this file exists for, stated as a measurement.

    Before :func:`warp_slack` the margin was the constant 2 whatever the
    transform was. Over this sweep that window is too small for a large number
    of the warps -- the shape is simply cut off at the window's edge -- and the
    scale-dependent one holds every single one of them.
    """
    missed_by_two = 0
    checked = 0
    for part, transform in _cases():
        reached = masks.bbox(_part_mask(part, transform, HW))
        if reached is None:
            continue                     # warped off the canvas entirely
        checked += 1
        assert _holds(_part_window(part, transform, HW), reached)
        warped = _transform_box(part.box, transform)
        fixed = _clip_window((warped[0] - 2, warped[1] - 2,
                              warped[2] + 2, warped[3] + 2), HW)
        missed_by_two += int(not _holds(fixed, reached))

    assert checked > 200, f"only {checked} warps stayed on the canvas"
    assert missed_by_two > 0, (
        "the sweep no longer contains a warp the old fixed slack of 2 missed"
    )


def test_a_box_only_part_is_covered_too():
    """A part with no RLE is rasterised by fillPoly, which rounds its corners."""
    for scale in SCALES:
        for theta in THETAS:
            part = ShapePart("main", None, (12.4, 8.6, 57.2, 41.9))
            transform = Similarity(scale=scale, theta=theta, tx=1.5, ty=-2.5)
            window = _part_window(part, transform, HW)
            full = _part_mask(part, transform, HW)
            reached = masks.bbox(full)
            if reached is None:
                continue
            assert (window[0] <= reached[0] and window[1] <= reached[1]
                    and window[2] >= reached[2] and window[3] >= reached[3]), (
                f"window {window} misses {reached} at scale={scale} theta={theta}"
            )
