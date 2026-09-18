"""Tests for the frame-difference map (spec 4.2).

Synthetic pairs cover the contract; the real-data tests at the bottom run on the
cached scanner frames of machine D13 and are skipped when that cache is not on
this machine. Nothing here writes outside ``tmp_path`` -- the overlays a human
eyeballs are produced by ``experiments/diffmap_calibrate.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
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

HW = (240, 320)
RECT = (80, 60, 160, 120)  # x0, y0, x1, y1 -- the "removed part"
RECT2 = (220, 150, 280, 200)  # a second, unexplained change

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Where the CPU cooler sits in D13's 1600x1600 scanner frame (read off s012.png).
COOLER_BOX = (1030, 440, 1290, 700)


def _cache_dir() -> Optional[str]:
    """``cache_dir`` from configs/paths.yaml; never raises."""
    try:
        with open(REPO_ROOT / "configs" / "paths.yaml", "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        value = cfg.get("cache_dir")
        return str(value) if value else None
    except Exception:
        return None


def _scan(step: int, desktop: int = 13) -> Optional[Path]:
    root = _cache_dir()
    if root is None:
        return None
    path = Path(cache_path(root, FrameKey(desktop, step, "scan"), "png"))
    return path if path.is_file() else None


def _pair(before: int, after: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The two cached scanner frames as RGB, or ``None`` when absent."""
    import cv2

    pa, pb = _scan(before), _scan(after)
    if pa is None or pb is None:
        return None
    a, b = cv2.imread(str(pa)), cv2.imread(str(pb))
    if a is None or b is None:
        return None
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


needs_d13 = pytest.mark.skipif(
    _scan(12) is None or _scan(13) is None or _scan(2) is None or _scan(3) is None,
    reason="D13 scanner cache is not on this machine",
)


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


def _textured() -> np.ndarray:
    """Busy image whose every pixel has a gradient, so a 1 px shift is visible."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, size=(HW[0], HW[1], 3)).astype(np.uint8)
    import cv2

    return cv2.GaussianBlur(noise, (5, 5), 0)


def _with_rect(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = np.asarray((205, 195, 185), dtype=np.uint8)
    return out


# ---------------------------------------------------------------------------
# diff_delta_e / diff_heat
# ---------------------------------------------------------------------------
def test_removed_rectangle_becomes_one_blob_covering_it():
    after = _base()
    before = _with_rect(after, RECT)

    delta = diff_delta_e(before, after)
    assert delta.shape == HW
    assert delta.dtype == np.float32
    x0, y0, x1, y1 = RECT
    assert delta[(y0 + y1) // 2, (x0 + x1) // 2] > 20.0, "a removed part is a big dE"
    assert delta[10, 10] < BLOB_DELTA_E, "untouched background must stay cold"

    heat = diff_heat(before, after)
    assert heat.shape == HW
    assert heat.dtype == np.float32
    assert heat.min() >= 0.0 and heat.max() <= 1.0
    assert heat[(y0 + y1) // 2, (x0 + x1) // 2] > 0.5

    blobs = diff_blobs(delta)
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

    assert diff_delta_e(a, b).max() < BLOB_DELTA_E
    assert diff_heat(a, b).max() < 0.25
    assert diff_blobs(diff_delta_e(a, b)) == []


def test_a_one_pixel_shift_is_not_a_change():
    """Scanner re-registration moves the frame by a pixel; that is not a change."""
    img = _textured()
    shifted = np.roll(img, 1, axis=1)
    inner = (2, 2, HW[1] - 2, HW[0] - 2)  # ignore the wrapped column

    assert diff_blobs(diff_delta_e(img, shifted, roi=inner)) == []

    # ... and the tolerance is what does it, not a blunt threshold:
    strict = diff_delta_e(img, shifted, roi=inner, shift_px=0)
    assert diff_blobs(strict), "shift_px=0 must still see the misregistration"
    assert strict.max() > diff_delta_e(img, shifted, roi=inner).max() * 3


def test_a_shifted_frame_still_shows_a_real_change():
    """The shift tolerance must not swallow a part that really disappeared."""
    after = _textured()
    before = _with_rect(after, RECT)
    shifted_before = np.roll(before, 1, axis=1)
    inner = (2, 2, HW[1] - 2, HW[0] - 2)

    blobs = diff_blobs(diff_delta_e(shifted_before, after, roi=inner))
    assert len(blobs) == 1
    assert _iou(blobs[0].box, RECT) > 0.6


def test_roi_limits_the_heat_and_the_blobs():
    after = _base()
    before = _with_rect(_with_rect(after, RECT), RECT2)

    delta = diff_delta_e(before, after, roi=(0, 0, 200, 240))  # contains RECT only
    assert delta[:, 200:].max() == 0.0, "outside the ROI there is no difference"
    assert diff_heat(before, after, roi=(0, 0, 200, 240))[:, 200:].max() == 0.0
    blobs = diff_blobs(delta)
    assert len(blobs) == 1
    assert _iou(blobs[0].box, RECT) > 0.7


def test_blob_detection_does_not_depend_on_the_roi():
    """Absolute dE, not the in-ROI percentile, decides what counts as a blob."""
    after = _base()
    before = _with_rect(after, RECT)
    wide = diff_blobs(diff_delta_e(before, after))
    tight = diff_blobs(diff_delta_e(before, after, roi=(40, 20, 220, 180)))
    assert len(wide) == len(tight) == 1
    assert wide[0].box == tight[0].box
    # Not bit-identical: the photometric equalisation is estimated inside the
    # ROI, so a different ROI moves the dE field by a hair. What must not move
    # is the decomposition -- same blob, same size to within a pixel or two.
    assert abs(wide[0].area - tight[0].area) <= 0.01 * wide[0].area
    # The *display* map is relative, so it does differ -- that is the point of
    # separating the two.
    assert diff_heat(before, after).max() == pytest.approx(1.0)


def test_an_empty_roi_yields_a_zero_map():
    after = _base()
    before = _with_rect(after, RECT)
    delta = diff_delta_e(before, after, roi=(10, 10, 10, 10))
    assert delta.shape == HW
    assert not delta.any()
    assert not diff_heat(before, after, roi=(10, 10, 10, 10)).any()


def test_identical_frames_are_cold_everywhere():
    a = _base()
    assert not diff_delta_e(a, a.copy()).any()
    assert not diff_heat(a, a.copy()).any()
    assert diff_blobs(diff_delta_e(a, a.copy())) == []


def test_diff_delta_e_validates_its_inputs():
    a = _base()
    with pytest.raises(ValueError):
        diff_delta_e(a, a[:, :100])  # shape mismatch
    with pytest.raises(ValueError):
        diff_delta_e(a[..., 0], a[..., 0])  # not RGB
    with pytest.raises(ValueError):
        diff_delta_e(a.astype(np.float32), a.astype(np.float32))  # not uint8
    with pytest.raises(ValueError):
        diff_heat(a, a[:, :100])


def test_max_side_downscales_and_still_finds_the_part():
    """A 4032x3040 OAK frame is computed small and mapped back to full res."""
    after = _base()
    before = _with_rect(after, RECT)
    delta = diff_delta_e(before, after, max_side=160)
    assert delta.shape == HW, "the map must come back at full resolution"
    blobs = diff_blobs(delta)
    assert len(blobs) == 1
    assert _iou(blobs[0].box, RECT) > 0.6, "boxes must be in full-res coordinates"


def test_diff_heat_scale_parameters_are_overridable():
    after = _base()
    before = _with_rect(after, RECT)
    delta = diff_delta_e(before, after)
    # A floor above the actual maximum flattens the map towards 0.
    flat = diff_heat(before, after, min_delta_e=float(delta.max()) * 4.0)
    assert flat.max() < 0.3
    # A lower percentile saturates more of the blob.
    assert (diff_heat(before, after, robust_pct=50.0) >= 1.0).sum() > (
        diff_heat(before, after, robust_pct=99.5) >= 1.0
    ).sum()


# ---------------------------------------------------------------------------
# diff_blobs
# ---------------------------------------------------------------------------
def _fake_delta(hw=(100, 200)) -> np.ndarray:
    return np.zeros(hw, dtype=np.float32)


def test_diff_blobs_honours_min_area_max_blobs_and_sorts_by_score():
    delta = _fake_delta()
    for i in range(5):
        delta[10:30, 10 + i * 35 : 30 + i * 35] = 20.0 + 4.0 * i  # 400 px each
    delta[70:73, 70:73] = 60.0  # 9 px: below min_area

    blobs = diff_blobs(delta, merge_gap_px=0)
    assert len(blobs) == 5
    scores = [b.score for b in blobs]
    assert scores == sorted(scores, reverse=True)
    assert blobs[0].box == (150, 10, 170, 30), "the hottest patch must come first"
    assert all(b.area >= 80 for b in blobs)

    assert len(diff_blobs(delta, merge_gap_px=0, max_blobs=2)) == 2
    assert diff_blobs(delta, merge_gap_px=0, min_area=1000) == []
    assert diff_blobs(delta, merge_gap_px=0, min_delta_e=100.0) == []


def test_nearby_fragments_merge_into_one_blob():
    """One part that breaks into satellites must not eat every blob slot."""
    delta = _fake_delta()
    delta[20:40, 20:40] = 30.0  # the part
    delta[20:40, 48:60] = 30.0  # a satellite 8 px away
    delta[20:40, 150:170] = 30.0  # an unrelated change far away

    assert len(diff_blobs(delta, merge_gap_px=0)) == 3
    merged = diff_blobs(delta, merge_gap_px=12)
    assert len(merged) == 2
    top = merged[0]
    assert top.box == (20, 20, 60, 40), "the union box must cover both fragments"
    assert top.area == 20 * 20 + 20 * 12
    assert merged[1].box == (150, 20, 170, 40)


def test_a_blob_carries_its_own_mask():
    delta = _fake_delta()
    delta[20:40, 20:40] = 30.0
    blob = diff_blobs(delta)[0]
    assert blob.mask is not None
    assert blob.mask.dtype == np.bool_
    x0, y0, x1, y1 = blob.box
    assert blob.mask.shape == (y1 - y0, x1 - x0), "the mask is box-local"
    assert blob.mask.all(), "a solid square is solid"
    assert int(blob.mask.sum()) == blob.area


def test_diff_blobs_validates_its_input():
    with pytest.raises(ValueError):
        diff_blobs(np.zeros((4, 4, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# explain_blobs
# ---------------------------------------------------------------------------
def test_explain_blobs_splits_expected_from_unexplained():
    after = _base()
    before = _with_rect(_with_rect(after, RECT), RECT2)
    blobs = diff_blobs(diff_delta_e(before, after))
    assert len(blobs) == 2

    explained, unexplained = explain_blobs(blobs, [RECT])
    assert len(explained) == 1
    assert len(unexplained) == 1
    assert _iou(explained[0].box, RECT) > 0.7
    assert _iou(unexplained[0].box, RECT2) > 0.7


def test_a_blob_inside_an_expected_box_is_explained():
    """A screw hole revealed under a removed cooler is not an unexplained change."""
    inside = DiffBlob(box=(120, 100, 140, 120), area=400, score=10.0)
    expected = (80, 60, 200, 180)
    assert _iou(inside.box, expected) < 0.1, "IoU alone would miss it"

    explained, unexplained = explain_blobs([inside], [expected])
    assert explained == [inside]
    assert unexplained == []

    # contain_min=None restores the strict IoU rule.
    assert explain_blobs([inside], [expected], contain_min=None) == ([], [inside])


def test_containment_must_be_substantial():
    half_in = DiffBlob(box=(150, 100, 250, 120), area=2000, score=10.0)
    expected = (80, 60, 200, 180)  # covers half of the blob's box
    assert explain_blobs([half_in], [expected]) == ([], [half_in])
    assert explain_blobs([half_in], [expected], contain_min=0.4) == ([half_in], [])


def test_explain_blobs_without_expected_boxes_explains_nothing():
    blobs = [DiffBlob(box=(0, 0, 10, 10), area=100, score=1.0)]
    explained, unexplained = explain_blobs(blobs, [])
    assert explained == []
    assert unexplained == blobs


def test_explain_blobs_needs_more_than_a_corner_touch():
    blob = DiffBlob(box=(0, 0, 100, 100), area=10_000, score=1.0)
    far = (95, 95, 195, 195)  # IoU ~0.0013, containment 0.0025
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


def test_heat_to_rgba_validates_its_input():
    with pytest.raises(ValueError):
        heat_to_rgba(np.zeros((4, 4, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# real data (D13 scanner cache)
# ---------------------------------------------------------------------------
@needs_d13
def test_real_pair_points_at_the_removed_cpu_cooler():
    """D13 step 13 removes the CPU cooler; it must be the top diff blob."""
    pair = _pair(12, 13)
    assert pair is not None
    before, after = pair
    roi = suggest_roi(before, "scan")

    blobs = diff_blobs(diff_delta_e(before, after, roi=roi))
    assert blobs, "no change detected between s012 and s013"
    top = blobs[0]
    assert _iou(top.box, COOLER_BOX) > 0.3, (
        f"top blob {top.box} does not cover the cooler {COOLER_BOX}"
    )
    # Merging must keep the cooler in one piece: a box smaller than the part is
    # a worse SAM prompt than no box at all.
    assert len(blobs) == 1, f"the cooler fragmented into {len(blobs)} blobs"
    assert top.area > 10_000

    # Every satellite of the cooler removal must count as explained, or the
    # unexplained-change queue drowns in them.
    _, unexplained = explain_blobs(blobs, [COOLER_BOX])
    for blob in unexplained:
        assert _iou(blob.box, COOLER_BOX) == 0.0, (
            f"blob {blob.box} overlaps the cooler yet reads as unexplained"
        )


@needs_d13
def test_real_pair_flags_the_neighbourhood_of_a_removed_screw():
    """D13 step 3 removes CPU fan screw 1 (~15 px across, ~170 px of area).

    At scanner resolution the screw itself is barely resolved: what the map
    actually catches is the cooler settling once the screw is out, so the blob
    lands *on the cooler*, not on the screw. That is still the answer the
    annotator needs (spec 4.2 already expects captive screws to need a manual
    click), but it is why this asserts a neighbourhood and not a silhouette.
    """
    pair = _pair(2, 3)
    assert pair is not None
    before, after = pair
    roi = suggest_roi(before, "scan")

    blobs = diff_blobs(diff_delta_e(before, after, roi=roi), min_area=100)
    assert blobs, "the screw step produced no blob at all: the threshold is too high"
    on_cooler = [b for b in blobs if _iou(b.box, COOLER_BOX) > 0 or
                 (b.box[0] >= COOLER_BOX[0] and b.box[2] <= COOLER_BOX[2]
                  and b.box[1] >= COOLER_BOX[1] and b.box[3] <= COOLER_BOX[3])]
    assert on_cooler, f"nothing flagged near the screw; got {[b.box for b in blobs]}"

    # A screw step must not look like a part removal, or the annotator cannot
    # tell the two apart from the overlay.
    assert all(b.area < 2_000 for b in blobs), "a screw step produced a part-sized blob"
    cooler_pair = _pair(12, 13)
    assert cooler_pair is not None
    cooler_blobs = diff_blobs(
        diff_delta_e(cooler_pair[0], cooler_pair[1], roi=roi)
    )
    assert blobs[0].score < cooler_blobs[0].score / 5.0


@needs_d13
def test_a_frame_against_itself_is_quiet():
    """The scanner's own re-metering must not manufacture work."""
    pair = _pair(12, 12)
    assert pair is not None
    before, _ = pair
    roi = suggest_roi(before, "scan")
    assert diff_blobs(diff_delta_e(before, before.copy(), roi=roi)) == []


@needs_d13
def test_a_one_pixel_shift_of_a_real_frame_is_quiet():
    """Misregistration must score far below a real removal."""
    pair = _pair(12, 13)
    assert pair is not None
    before, after = pair
    roi = suggest_roi(before, "scan")
    shifted = np.roll(before, 1, axis=1)

    real = diff_blobs(diff_delta_e(before, after, roi=roi))
    bogus = diff_blobs(diff_delta_e(before, shifted, roi=roi))
    assert real, "the real removal must still be found"
    top_bogus = bogus[0].score if bogus else 0.0
    assert top_bogus < real[0].score / 5.0, (
        f"a 1 px shift scores {top_bogus:.0f} against a real change {real[0].score:.0f}"
    )
