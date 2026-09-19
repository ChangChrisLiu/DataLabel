"""The scanner ROI on a chassis that is not the darkest thing on the bed.

``suggest_roi``'s dark-object stage assumes the chassis is darker than
everything around it. On D64 -- a light, silver machine -- it is not: the stage
locks onto the motherboard's PCB and returns 13% of the frame, with every
sampled step agreeing, so no median over steps can catch it either.

:mod:`tda.core.cache_roi_detect` asks the opposite question as well -- the bed
is a bright board, so the chassis is whatever differs from it -- and both
strategies are judged on shape alone by ``box_plausibility``. Three things that
matters for are pinned here: the box is the component's **axis-aligned bounding
box** (a ``minAreaRect`` hull around an irregular blob is far bigger than the
blob); a hand or a tool lying against the machine must not be measured as part
of it; and the dark stage keeps precedence wherever its answer is chassis-shaped,
because it is tighter on the 55 machines it gets right.

The synthetic scenes here are the contract; the real-data tests at the bottom
run on the cached scanner frames of D64 and D55 and are skipped when that cache
is not on this machine. ``experiments/roi_scan_eval.py`` produced the numbers.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tda.core.cache import full_frame, suggest_roi
from tda.core.cache_roi_detect import (
    BED_MIN_RECTANGULARITY,
    DARK_MIN_RECTANGULARITY,
    TOP_COMPONENTS,
    best_candidate,
    box_plausibility,
    scan_bed_box,
    scan_bed_candidates,
    scan_chassis_box,
    scan_chassis_candidates,
)

SIZE = 1000
BED = 240
TAPE = (120, 880)
CACHE = Path("D:/DataSet/cache/scan")
CHASSIS = (250, 200, 760, 800)  # x0, y0, x1, y1 of the synthetic machine


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
    x0, y0, x1, y1 = CHASSIS
    img[y0:y1, x0:x1] = (175, 175, 175)  # far above DARK_MAX
    img[330:530, 400:600] = (40, 45, 40)  # the motherboard: the old answer
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


def _real(desktop: int):
    path = CACHE / f"D{desktop:02d}" / "s001.png"
    if not path.is_file():
        pytest.skip("the real scanner cache is not on this machine")
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# the light chassis
# --------------------------------------------------------------------------- #
def test_the_dark_stage_still_finds_only_the_board():
    """The premise: this is what makes the second strategy necessary."""
    found = scan_chassis_candidates(_light_machine())
    assert found and _frac(found[0][0]) < 0.2  # the PCB, not the machine


def test_the_bed_stage_finds_the_light_chassis():
    found = scan_bed_box(_light_machine())
    assert found is not None
    assert _contains(found[0], CHASSIS)
    assert 0.2 <= _frac(found[0]) <= 0.95


def test_suggest_roi_prefers_the_bed_box_when_the_dark_one_is_a_speck():
    box = suggest_roi(_light_machine(), "scan")
    assert _contains(box, CHASSIS)
    assert box != full_frame(SIZE, SIZE)


def test_a_dark_chassis_still_wins_with_the_dark_stage():
    """The tighter, calibrated answer keeps precedence where it is shaped right."""
    img = _dark_machine()
    dark, bed = scan_chassis_box(img), scan_bed_box(img)
    assert dark is not None and bed is not None
    assert suggest_roi(img, "scan") == dark[0]
    assert _contains(suggest_roi(img, "scan"), (250, 210, 760, 790))


def test_a_bare_bed_suggests_nothing_and_falls_back_to_the_whole_frame():
    """A crop that cuts the machine is worse than no crop: the nine tape-less
    real desktops all had their chassis clipped by the old central-70% box."""
    assert suggest_roi(_bed(), "scan") == (0, 0, SIZE, SIZE)


def test_a_frame_that_is_all_chassis_is_not_cropped_at_all():
    """A box covering 100% of the frame is vetoed, and the answer is the frame."""
    img = np.full((SIZE, SIZE, 3), 60, np.uint8)
    assert suggest_roi(img, "scan") == (0, 0, SIZE, SIZE)


# --------------------------------------------------------------------------- #
# the box is the component's bounding box, not a rotated rect's hull
# --------------------------------------------------------------------------- #
def test_the_box_hugs_a_diagonal_machine_rather_than_its_rotated_hull():
    """A ``minAreaRect`` around a tilted blob spans far more than the blob."""
    img = _bed()
    points = np.array([[300, 250], [740, 400], [700, 760], [280, 600]], np.int32)
    cv2.fillPoly(img, [points], (170, 170, 170))
    found = scan_bed_box(img)
    assert found is not None
    x0, y0, x1, y1 = found[0]
    assert x0 <= 300 and y0 <= 250 and x1 >= 740 and y1 >= 760
    assert _frac(found[0]) < 0.45  # the hull of its minAreaRect would be far bigger


# --------------------------------------------------------------------------- #
# a hand against the chassis is not part of the chassis
# --------------------------------------------------------------------------- #
def _machine_with_a_hand() -> np.ndarray:
    """The light machine with a skin-toned arm lying across its left edge."""
    img = _light_machine()
    cv2.ellipse(img, (250, 500), (190, 55), 0, 0, 360, (120, 150, 200), -1)
    return img


def test_a_hand_touching_the_chassis_does_not_join_it():
    img = _machine_with_a_hand()
    found = scan_bed_box(img)
    assert found is not None
    x0, _y0, x1, _y1 = found[0]
    # the arm reaches to x=60; a merged region would start there
    assert x0 > 120
    assert x1 >= CHASSIS[2]


def test_each_strategy_offers_several_components_to_choose_from():
    """"The biggest region" is the wrong answer exactly when something merged."""
    candidates = scan_bed_candidates(_machine_with_a_hand())
    assert 1 <= len(candidates) <= TOP_COMPONENTS
    assert candidates == sorted(
        candidates, key=lambda c: -(c[0][2] - c[0][0]) * (c[0][3] - c[0][1])
    )


def test_the_most_convincing_candidate_wins_not_the_biggest():
    big = ((0, 0, 900, 900), 0.30)   # a merged sprawl
    tight = ((200, 200, 700, 700), 0.90)  # the machine
    assert best_candidate([big, tight], SIZE, SIZE, 0.2) == tight


# --------------------------------------------------------------------------- #
# the shape test both strategies are judged by
# --------------------------------------------------------------------------- #
def test_plausibility_rejects_a_speck_and_the_whole_frame():
    assert box_plausibility((0, 0, 200, 200), 0.9, SIZE, SIZE) is None  # 4%
    assert box_plausibility((0, 0, SIZE, SIZE), 0.9, SIZE, SIZE) is None


def test_plausibility_rejects_a_sliver():
    assert box_plausibility((0, 0, 900, 300), 0.9, SIZE, SIZE) is None  # 3:1


def test_plausibility_rejects_a_box_nothing_fills():
    assert box_plausibility((0, 0, 700, 700), 0.05, SIZE, SIZE) is None


def test_the_two_strategies_are_judged_by_their_own_floor():
    """A dark region is dark only where the machine is; a bed region is all of it."""
    assert DARK_MIN_RECTANGULARITY < BED_MIN_RECTANGULARITY
    box, fill = (100, 100, 700, 700), 0.30
    assert box_plausibility(box, fill, SIZE, SIZE, DARK_MIN_RECTANGULARITY) is not None
    assert box_plausibility(box, fill, SIZE, SIZE, BED_MIN_RECTANGULARITY) is None


# --------------------------------------------------------------------------- #
# the real machines
# --------------------------------------------------------------------------- #
def test_the_real_d64_no_longer_returns_the_motherboard():
    img = _real(64)
    height, width = img.shape[:2]
    old = scan_chassis_candidates(img)
    assert old and _frac(old[0][0], width) < 0.2  # the PCB: 13%

    box = suggest_roi(img, "scan")
    assert box != old[0][0]
    assert box != full_frame(width, height)
    assert 0.40 <= _frac(box, width) <= 0.55  # the chassis: ~46%


def test_the_real_d55_is_cropped_at_all():
    """No tape square, so the dark stage says nothing; the bed stage sees the machine."""
    img = _real(55)
    height, width = img.shape[:2]
    assert scan_chassis_candidates(img) == []
    box = suggest_roi(img, "scan")
    assert box != full_frame(width, height)
    assert _frac(box, width) > 0.8


@pytest.mark.parametrize("desktop", [13, 30, 42])
def test_a_dark_machine_is_measured_by_the_dark_stage(desktop: int):
    img = _real(desktop)
    height, width = img.shape[:2]
    dark = scan_chassis_box(img)
    assert dark is not None
    assert suggest_roi(img, "scan") == dark[0]
