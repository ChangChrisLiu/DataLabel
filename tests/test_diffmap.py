"""Tests for the frame-difference map (spec 4.2).

Synthetic pairs cover the contract; the last test runs on two real cached
scanner frames of machine D13 (step 13 removed the CPU cooler) and is skipped
when that cache is not on this machine.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tda.core.diffmap import (
    DiffBlob,
    diff_blobs,
    diff_heat,
    explain_blobs,
    heat_to_rgba,
)

HW = (240, 320)
RECT = (80, 60, 160, 120)  # x0, y0, x1, y1 -- the "removed part"
RECT2 = (220, 150, 280, 200)  # a second, unexplained change

#: Cached scanner frames of D13 and where the overlay is written for eyeballing.
D13 = Path("D:/DataSet/cache/scan/D13")
DIFF_OUT = Path("D:/DataSet/experiments_out/diffmap/d13_s12_s13.png")
#: Where the CPU cooler sits in the 1600x1600 scanner frame (read off s012.png).
COOLER_BOX = (1030, 440, 1290, 700)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _base() -> np.ndarray:
    """Textured mid-grey background with a couple of stripes (deterministic)."""
    rng = np.random.default_rng(11)
    img = rng.integers(64, 96, size=(HW[0], HW[1], 3)).astype(np.uint8)
    img[:, 40:48] = 120
    img[100:108, :] = 110
    return img


def _with_rect(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = np.asarray((205, 195, 185), dtype=np.uint8)
    return out


# ---------------------------------------------------------------------------
# diff_heat
# ---------------------------------------------------------------------------
def test_removed_rectangle_becomes_one_blob_covering_it():
    after = _base()
    before = _with_rect(after, RECT)

    heat = diff_heat(before, after)
    assert heat.shape == HW
    assert heat.dtype == np.float32
    assert heat.min() >= 0.0 and heat.max() <= 1.0
    x0, y0, x1, y1 = RECT
    assert heat[(y0 + y1) // 2, (x0 + x1) // 2] > 0.5, "the removed part must be hot"
    assert heat[10, 10] < 0.1, "untouched background must stay cold"

    blobs = diff_blobs(heat)
    assert len(blobs) == 1
    blob = blobs[0]
    assert isinstance(blob, DiffBlob)
    assert blob.box[0] <= x0 and blob.box[1] <= y0
    assert blob.box[2] >= x1 and blob.box[3] >= y1
    assert _iou(blob.box, RECT) > 0.7, f"blob {blob.box} is not the rectangle"
    assert blob.area >= (x1 - x0) * (y1 - y0) * 0.9
    assert blob.score > 0.0


def test_global_brightness_shift_is_not_a_change():
    """+8 grey levels everywhere is exposure drift, not a scene change."""
    a = _base()
    b = np.clip(a.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    assert a.max() + 8 <= 255, "the fixture must not clip"

    heat = diff_heat(a, b)
    assert heat.max() < 0.25
    assert diff_blobs(heat) == []


def test_roi_limits_the_heat_and_the_blobs():
    after = _base()
    before = _with_rect(_with_rect(after, RECT), RECT2)

    heat = diff_heat(before, after, roi=(0, 0, 200, 240))  # contains RECT only
    assert heat[:, 200:].max() == 0.0, "outside the ROI the heat must be 0"
    blobs = diff_blobs(heat)
    assert len(blobs) == 1
    assert _iou(blobs[0].box, RECT) > 0.7


def test_an_empty_roi_yields_a_zero_heat_map():
    after = _base()
    before = _with_rect(after, RECT)
    heat = diff_heat(before, after, roi=(10, 10, 10, 10))
    assert heat.shape == HW
    assert not heat.any()


def test_identical_frames_are_cold_everywhere():
    a = _base()
    heat = diff_heat(a, a.copy())
    assert not heat.any()
    assert diff_blobs(heat) == []


def test_diff_heat_validates_its_inputs():
    a = _base()
    with pytest.raises(ValueError):
        diff_heat(a, a[:, :100])  # shape mismatch
    with pytest.raises(ValueError):
        diff_heat(a[..., 0], a[..., 0])  # not RGB
    with pytest.raises(ValueError):
        diff_heat(a.astype(np.float32), a.astype(np.float32))  # not uint8


# ---------------------------------------------------------------------------
# diff_blobs
# ---------------------------------------------------------------------------
def test_diff_blobs_honours_min_area_max_blobs_and_sorts_by_score():
    heat = np.zeros((100, 200), dtype=np.float32)
    for i in range(5):
        heat[10:30, 10 + i * 35 : 30 + i * 35] = 0.5 + 0.1 * i  # 400 px each
    heat[70:73, 70:73] = 0.9  # 9 px: below min_area

    blobs = diff_blobs(heat)
    assert len(blobs) == 5
    scores = [b.score for b in blobs]
    assert scores == sorted(scores, reverse=True)
    assert blobs[0].box == (150, 10, 170, 30), "the hottest patch must come first"
    assert all(b.area >= 80 for b in blobs)

    assert len(diff_blobs(heat, max_blobs=2)) == 2
    assert diff_blobs(heat, min_area=1000) == []
    assert diff_blobs(heat, thresh=0.95) == []


# ---------------------------------------------------------------------------
# explain_blobs
# ---------------------------------------------------------------------------
def test_explain_blobs_splits_expected_from_unexplained():
    after = _base()
    before = _with_rect(_with_rect(after, RECT), RECT2)
    blobs = diff_blobs(diff_heat(before, after))
    assert len(blobs) == 2

    explained, unexplained = explain_blobs(blobs, [RECT])
    assert len(explained) == 1
    assert len(unexplained) == 1
    assert _iou(explained[0].box, RECT) > 0.7
    assert _iou(unexplained[0].box, RECT2) > 0.7


def test_explain_blobs_without_expected_boxes_explains_nothing():
    blobs = [DiffBlob(box=(0, 0, 10, 10), area=100, score=1.0)]
    explained, unexplained = explain_blobs(blobs, [])
    assert explained == []
    assert unexplained == blobs


def test_explain_blobs_needs_more_than_a_corner_touch():
    blob = DiffBlob(box=(0, 0, 100, 100), area=10_000, score=1.0)
    far = (95, 95, 195, 195)  # IoU ~0.0013
    assert explain_blobs([blob], [far]) == ([], [blob])
    assert explain_blobs([blob], [far], iou_min=0.001) == ([blob], [])


# ---------------------------------------------------------------------------
# heat_to_rgba
# ---------------------------------------------------------------------------
def test_heat_to_rgba_is_a_transparent_red_to_yellow_ramp():
    heat = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(1, 256)
    rgba = heat_to_rgba(heat)
    assert rgba.shape == (1, 256, 4)
    assert rgba.dtype == np.uint8

    assert rgba[0, 0, 3] == 0, "cold pixels must be fully transparent"
    assert rgba[0, -1, 3] == 140
    assert tuple(rgba[0, -1, :3]) == (255, 255, 0), "hottest pixel is yellow"
    assert rgba[0, 128, 0] > rgba[0, 128, 1], "mid heat stays red"
    assert (np.diff(rgba[0, :, 3].astype(int)) >= 0).all(), "alpha is monotone"
    assert heat_to_rgba(heat, alpha_max=255)[0, -1, 3] == 255


def test_heat_to_rgba_clamps_out_of_range_input():
    rgba = heat_to_rgba(np.array([[-2.0, 3.0]], dtype=np.float32))
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 140


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (D13 / "s012.png").is_file() or not (D13 / "s013.png").is_file(),
    reason="D13 scanner cache is not on this machine",
)
def test_real_pair_points_at_the_removed_cpu_cooler():
    """D13 step 13 removes the CPU cooler; it must be the top diff blob."""
    before = cv2.cvtColor(cv2.imread(str(D13 / "s012.png")), cv2.COLOR_BGR2RGB)
    after = cv2.cvtColor(cv2.imread(str(D13 / "s013.png")), cv2.COLOR_BGR2RGB)

    heat = diff_heat(before, after)
    blobs = diff_blobs(heat)
    assert blobs, "no change detected between s012 and s013"
    top = blobs[0]
    assert _iou(top.box, COOLER_BOX) > 0.25, (
        f"top blob {top.box} does not cover the cooler {COOLER_BOX}"
    )

    # The blob's box is exactly what SamPointTool.set_prompt_box wants.
    explained, unexplained = explain_blobs(blobs, [COOLER_BOX])
    assert explained and explained[0].box == top.box

    # Save an overlay so a human can eyeball the result.
    DIFF_OUT.parent.mkdir(parents=True, exist_ok=True)
    rgba = heat_to_rgba(heat, alpha_max=200)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    color_bgr = rgba[..., [2, 1, 0]].astype(np.float32)
    base = cv2.cvtColor(after, cv2.COLOR_RGB2BGR).astype(np.float32)
    blend = (base * (1.0 - alpha) + color_bgr * alpha).astype(np.uint8)
    for i, blob in enumerate(blobs):
        x0, y0, x1, y1 = blob.box
        col = (0, 255, 0) if blob is top else (255, 255, 255)
        cv2.rectangle(blend, (x0, y0), (x1, y1), col, 3)
        cv2.putText(
            blend, f"{i}:{blob.score:.0f}", (x0, max(0, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2,
        )
    cv2.rectangle(blend, COOLER_BOX[:2], COOLER_BOX[2:], (255, 0, 255), 2)
    assert cv2.imwrite(str(DIFF_OUT), blend)
