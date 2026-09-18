"""The scanner ROI on a chassis that is not the darkest thing on the bed.

``suggest_roi``'s dark-object stage assumes the chassis is darker than
everything around it. On D64 -- a light, silver machine -- it is not: the stage
locks onto the motherboard's PCB and returns 13% of the frame, with every
sampled step agreeing, so no median over steps can catch it either.

The second strategy asks the opposite question: the bed is a bright, low
saturation board, so the chassis is whatever *differs* from it. The two answers
are chosen between by shape alone (:func:`tda.core.cache._box_plausibility`),
and the dark stage keeps precedence whenever its answer is shaped like a
chassis -- it is tighter on the 55 machines it gets right.

The synthetic scenes here are the contract; the real-data test at the bottom
runs on the cached scanner frame of D64 and is skipped when that cache is not
on this machine. ``experiments/roi_scan_eval.py`` produced the numbers.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tda.core.cache import (
    CENTRAL_FRAC,
    _box_plausibility,
    _central_box,
    _scan_bed_box,
    _scan_chassis_box,
    suggest_roi,
)

SIZE = 1000
BED = 240
TAPE = (120, 880)
REAL_D64 = Path("D:/DataSet/cache/scan/D64/s001.png")


def _bed(size: int = SIZE, band: int = 30) -> np.ndarray:
    """A white scan bed framed by the orange tape square."""
    img = np.full((size, size, 3), BED, np.uint8)
    lo, hi = TAPE
    for part in ((slice(lo, lo + band), slice(lo, hi)),
                 (slice(hi - band, hi), slice(lo, hi)),
                 (slice(lo, hi), slice(lo, lo + band)),
                 (slice(lo, hi), slice(hi - band, hi))):
        img[part] = (0, 200, 255)  # BGR: the yellow/orange tape
    return img


def _light_machine() -> np.ndarray:
    """A silver chassis with a dark PCB inside it -- the D64 shape."""
    img = _bed()
    img[200:800, 250:760] = (175, 175, 175)  # the chassis: far above DARK_MAX
    img[330:530, 400:600] = (40, 45, 40)     # the motherboard: the old answer
    return img


def _dark_machine() -> np.ndarray:
    """A black chassis on the same bed -- what the dark stage was written for."""
    img = _bed()
    img[210:790, 250:760] = (45, 45, 45)
    return img


def _frac(box, size: int = SIZE) -> float:
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0) / float(size * size)


def _contains(box, inner) -> bool:
    return (box[0] <= inner[0] and box[1] <= inner[1]
            and box[2] >= inner[2] and box[3] >= inner[3])


# --------------------------------------------------------------------------- #
# the light chassis
# --------------------------------------------------------------------------- #
def test_the_dark_stage_still_finds_only_the_board():
    """The premise: this is what makes the second strategy necessary."""
    found = _scan_chassis_box(_light_machine())
    assert found is not None
    assert _frac(found[0]) < 0.2  # the PCB, not the machine


def test_the_bed_stage_finds_the_light_chassis():
    found = _scan_bed_box(_light_machine())
    assert found is not None
    assert _contains(found[0], (250, 200, 760, 800))
    assert 0.2 <= _frac(found[0]) <= 0.95


def test_suggest_roi_prefers_the_bed_box_when_the_dark_one_is_a_speck():
    box = suggest_roi(_light_machine(), "scan")
    assert _contains(box, (250, 200, 760, 800))
    assert box != _central_box(SIZE, SIZE)


def test_a_dark_chassis_still_wins_with_the_dark_stage():
    """The tighter, calibrated answer keeps precedence where it is shaped right."""
    img = _dark_machine()
    dark, bed = _scan_chassis_box(img), _scan_bed_box(img)
    assert dark is not None and bed is not None
    # both see the same machine here; the dark one is the answer that is kept
    assert suggest_roi(img, "scan") == dark[0]
    assert _contains(suggest_roi(img, "scan"), (250, 210, 760, 790))


def test_a_bare_bed_suggests_nothing_and_falls_back():
    assert suggest_roi(_bed(), "scan") == _central_box(SIZE, SIZE)


def test_a_frame_that_is_all_chassis_falls_back_rather_than_crop_nothing():
    """A box covering 100% of the frame is not a crop; the central box is."""
    img = np.full((SIZE, SIZE, 3), 60, np.uint8)
    assert suggest_roi(img, "scan") == _central_box(SIZE, SIZE)


# --------------------------------------------------------------------------- #
# the shape test both strategies are judged by
# --------------------------------------------------------------------------- #
def test_plausibility_rejects_a_speck_and_the_whole_frame():
    assert _box_plausibility((0, 0, 200, 200), 40000, SIZE, SIZE) is None  # 4%
    assert _box_plausibility((0, 0, SIZE, SIZE), SIZE * SIZE, SIZE, SIZE) is None


def test_plausibility_rejects_a_sliver():
    assert _box_plausibility((0, 0, 900, 300), 270000, SIZE, SIZE) is None  # 3:1


def test_plausibility_rejects_a_box_nothing_fills():
    assert _box_plausibility((0, 0, 700, 700), 10_000, SIZE, SIZE) is None


def test_plausibility_scores_a_chassis_shaped_box():
    score = _box_plausibility((100, 100, 700, 700), 200_000, SIZE, SIZE)
    assert score is not None and 0 < score <= 1.0


# --------------------------------------------------------------------------- #
# the real D64
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_D64.is_file(), reason="the real scanner cache is not here")
def test_the_real_d64_no_longer_returns_the_motherboard():
    img = cv2.imdecode(np.fromfile(str(REAL_D64), dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    old = _scan_chassis_box(img)
    assert old is not None and _frac(old[0], width) < 0.2  # the PCB: 13%

    box = suggest_roi(img, "scan")
    assert box != old[0]
    assert box != _central_box(width, height)
    assert 0.5 <= (box[2] - box[0]) * (box[3] - box[1]) / float(width * height) <= 0.95
