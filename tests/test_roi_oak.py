"""The chassis ROI on a 12 MP OAK frame (spec 2.4, plan B task B3 step 3).

An OAK frame is not a scanner frame and the scanner's two strategies return
nothing on one: the machine sits on a white workbench with a yellow tape square
that covers a *quarter* of the picture rather than framing it, and the rest of
the frame is floor, an operator in dark clothes and the robot rig.  The
scanner's tape gate (the square has to span half the frame) rejects the square,
so ``suggest_roi`` fell through to "no proposal" on every OAK frame -- which is
what the controller measured as ``roi: []``.

What is pinned here:

* the tape square is found at OAK spans and the chassis is the largest thing on
  the board that is **not the board**, judged on shape alone
  (:func:`~tda.core.cache_roi_detect.box_plausibility`);
* an operator standing beside the bench is outside the board and can therefore
  never be proposed, however large and dark they are;
* no tape square, a machine that fills the frame, or a candidate outside the
  area band -> **no proposal at all**.  A wrong box costs the annotator more
  than no box: they have to notice it is wrong first.

The real-frame checks at the bottom run on ``F:`` and are skipped when the raw
data is not mounted; ``experiments/roi_oak_eval.py`` is what produced the
by-eye numbers in the task report.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from tda.core.cache import full_frame, suggest_roi
from tda.core.cache_roi_detect import (
    OAK_MAX_AREA_FRAC,
    OAK_MIN_AREA_FRAC,
    OAK_TAPE_MIN_SPAN_FRAC,
    oak_board_mask,
    oak_chassis_box,
    oak_chassis_candidates,
)

HW = (760, 1008)          # a 12 MP OAK frame, scaled down by four
TABLE = (0.10, 0.06, 0.78, 0.97)   # x0, y0, x1, y1 as fractions of the frame
TAPE = (0.22, 0.16, 0.66, 0.74)
CHASSIS = (0.30, 0.26, 0.56, 0.60)


def _box(frac, hw=HW) -> tuple[int, int, int, int]:
    height, width = hw
    return (int(frac[0] * width), int(frac[1] * height),
            int(frac[2] * width), int(frac[3] * height))


def bench(hw=HW, *, chassis=CHASSIS, tape=TAPE, operator: bool = True,
          band: int = 10) -> np.ndarray:
    """A white bench with a yellow tape square, a dark machine and an operator."""
    height, width = hw
    img = np.full((height, width, 3), (170, 172, 168), np.uint8)   # the floor
    x0, y0, x1, y1 = _box(TABLE, hw)
    img[y0:y1, x0:x1] = (238, 240, 240)                            # the bench
    tx0, ty0, tx1, ty1 = _box(tape, hw)
    for part in ((slice(ty0, ty0 + band), slice(tx0, tx1)),
                 (slice(ty1 - band, ty1), slice(tx0, tx1)),
                 (slice(ty0, ty1), slice(tx0, tx0 + band)),
                 (slice(ty0, ty1), slice(tx1 - band, tx1))):
        img[part] = (40, 190, 230)                                 # BGR: the tape
    cx0, cy0, cx1, cy1 = _box(chassis, hw)
    img[cy0:cy1, cx0:cx1] = (52, 54, 56)                           # the machine
    inner = 22
    img[cy0 + inner:cy1 - inner, cx0 + inner:cx1 - inner] = (150, 152, 150)  # its guts
    if operator:
        img[int(0.25 * height):height, 0:int(0.09 * width)] = (28, 26, 30)
    return img


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #
def test_the_tape_square_is_found_at_oak_spans():
    """It covers well under half the frame, which the scanner gate rejects."""
    tx0, ty0, tx1, ty1 = _box(TAPE)
    assert (tx1 - tx0) / HW[1] < 0.5 and (ty1 - ty0) / HW[0] < 0.6
    assert (tx1 - tx0) / HW[1] > OAK_TAPE_MIN_SPAN_FRAC

    board = oak_board_mask(bench())
    assert board is not None
    ys, xs = np.nonzero(board)
    assert xs.min() >= tx0 - 2 and xs.max() <= tx1 + 2
    assert ys.min() >= ty0 - 2 and ys.max() <= ty1 + 2


def test_no_tape_square_means_no_board():
    plain = np.full((*HW, 3), 240, np.uint8)
    assert oak_board_mask(plain) is None


# --------------------------------------------------------------------------- #
# the chassis
# --------------------------------------------------------------------------- #
def test_the_proposal_contains_the_machine_and_little_else():
    x0, y0, x1, y1 = suggest_roi(bench(), "oak1")
    cx0, cy0, cx1, cy1 = _box(CHASSIS)

    assert all(isinstance(v, int) for v in (x0, y0, x1, y1))
    assert x0 <= cx0 and y0 <= cy0 and x1 >= cx1 and y1 >= cy1
    assert (x1 - x0) * (y1 - y0) <= 1.6 * (cx1 - cx0) * (cy1 - cy0)


def test_the_operator_beside_the_bench_is_never_the_proposal():
    """The largest dark region in an oak2 frame is a person, off the board."""
    img = bench(operator=True)
    x0, _y0, _x1, _y1 = suggest_roi(img, "oak2")
    assert x0 > int(0.09 * HW[1]), "the proposal starts inside the operator"
    for box, _fill in oak_chassis_candidates(img):
        assert box[0] >= _box(TAPE)[0] - 4


def test_a_machine_that_fills_the_frame_gets_no_proposal():
    """D63/D64 are towers photographed from 40 cm: there is no bench to see."""
    img = np.full((*HW, 3), 60, np.uint8)
    assert suggest_roi(img, "oak1") == full_frame(HW[1], HW[0])


def test_a_candidate_outside_the_area_band_is_refused():
    tiny = bench(chassis=(0.40, 0.40, 0.44, 0.45))
    assert suggest_roi(tiny, "oak1") == full_frame(HW[1], HW[0])
    assert OAK_MIN_AREA_FRAC < 0.10 < OAK_MAX_AREA_FRAC


def test_the_proposal_is_in_original_pixels_on_a_full_size_frame():
    """The detector measures on a downscale; the box comes back full size."""
    big = cv2.resize(bench(), (4032, 3040), interpolation=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = suggest_roi(big, "oak1")
    cx0, cy0, cx1, cy1 = _box(CHASSIS, (3040, 4032))

    assert (x0, y0, x1, y1) != full_frame(4032, 3040)
    assert x0 <= cx0 and y0 <= cy0 and x1 >= cx1 and y1 >= cy1
    assert x1 <= 4032 and y1 <= 3040


def test_the_scanner_is_not_measured_the_oak_way():
    """The two detectors are separate: a scanner frame still takes its own path."""
    assert oak_chassis_box(np.full((*HW, 3), 240, np.uint8)) is None


# --------------------------------------------------------------------------- #
# real frames (skipped without F:)
# --------------------------------------------------------------------------- #
OAK_ROOT = ("F:/PHD Data Backup/Desktop Dataset/OAKD Capture/"
            "Desktop_Datacollection/DesktopData")


def _real(desktop: int, camera: int, step: int):
    from pathlib import Path

    folder = Path(OAK_ROOT) / f"Desktop {desktop}" / "Disassemble" / f"Camera_{camera}" / f"{step:03d}"
    if not folder.is_dir():
        return None
    found = sorted(folder.glob("*_rgb_12mp.jpg"))
    return cv2.imread(str(found[0])) if found else None


@pytest.mark.slow
@pytest.mark.parametrize("desktop,camera,step", [(13, 1, 1), (13, 2, 1), (24, 1, 1)])
def test_a_real_oak_frame_gets_a_chassis_sized_proposal(desktop, camera, step):
    img = _real(desktop, camera, step)
    if img is None:
        pytest.skip("the raw OAK data is not mounted on this machine")
    height, width = img.shape[:2]
    box = suggest_roi(img, f"oak{camera}")
    assert box != full_frame(width, height), "no proposal on a frame that has one"
    frac = (box[2] - box[0]) * (box[3] - box[1]) / float(width * height)
    assert OAK_MIN_AREA_FRAC <= frac <= OAK_MAX_AREA_FRAC
