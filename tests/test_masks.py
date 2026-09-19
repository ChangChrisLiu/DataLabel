"""Tests for tda.core.masks -- the pure numpy/cv2 mask toolbox."""
from __future__ import annotations

import numpy as np
import pytest

from tda.core import masks as M
from tda.core.model import Similarity


# --------------------------------------------------------------------------
# helpers (test-local, deliberately independent of the implementation)
# --------------------------------------------------------------------------


def _square(hw=(64, 64), y0=10, x0=10, side=30) -> np.ndarray:
    m = np.zeros(hw, dtype=bool)
    m[y0 : y0 + side, x0 : x0 + side] = True
    return m


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a mask by (dy, dx) without wrap-around."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = mask[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
    return out


def _disk(hw, cy, cx, r) -> np.ndarray:
    yy, xx = np.ogrid[: hw[0], : hw[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


# --------------------------------------------------------------------------
# RLE
# --------------------------------------------------------------------------


def test_rle_roundtrip_random_mask():
    rng = np.random.default_rng(1234)
    mask = rng.random((37, 53)) > 0.5
    rle = M.encode_rle(mask)
    assert rle["size"] == [37, 53]
    assert isinstance(rle["counts"], str)
    back = M.decode_rle(rle)
    assert back.dtype == bool
    assert back.shape == mask.shape
    assert np.array_equal(back, mask)


def test_rle_roundtrip_empty_and_full():
    empty = np.zeros((9, 11), dtype=bool)
    full = np.ones((9, 11), dtype=bool)
    assert np.array_equal(M.decode_rle(M.encode_rle(empty)), empty)
    assert np.array_equal(M.decode_rle(M.encode_rle(full)), full)


def test_decode_rle_accepts_bytes_counts():
    mask = _square((20, 20), 3, 4, 6)
    rle = M.encode_rle(mask)
    as_bytes = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    assert np.array_equal(M.decode_rle(as_bytes), mask)


# --------------------------------------------------------------------------
# bbox / min_side / area
# --------------------------------------------------------------------------


def test_bbox_min_side_area_on_rectangle():
    mask = np.zeros((40, 40), dtype=bool)
    mask[7:12, 20:29] = True  # 5 rows tall, 9 cols wide
    assert M.bbox(mask) == (20, 7, 29, 12)  # x0, y0, x1, y1 exclusive
    assert M.min_side(mask) == 5
    assert M.area(mask) == 45


def test_bbox_min_side_area_on_empty_mask():
    empty = np.zeros((12, 15), dtype=bool)
    assert M.bbox(empty) is None
    assert M.min_side(empty) == 0
    assert M.area(empty) == 0


def test_bbox_covers_single_pixel():
    m = np.zeros((10, 10), dtype=bool)
    m[4, 6] = True
    assert M.bbox(m) == (6, 4, 7, 5)
    assert M.min_side(m) == 1


# --------------------------------------------------------------------------
# morphology
# --------------------------------------------------------------------------


def test_fill_holes_fills_a_ring():
    outer = _disk((64, 64), 32, 32, 20)
    inner = _disk((64, 64), 32, 32, 8)
    ring = outer & ~inner
    filled = M.fill_holes(ring)
    assert filled.dtype == bool
    assert np.array_equal(filled, outer)


def test_fill_holes_does_not_fill_a_notch_open_to_the_border():
    m = np.zeros((32, 32), dtype=bool)
    m[5:25, 5:25] = True
    m[10:15, 5:15] = False  # notch cut in from the left edge of the square
    m[10:15, 0:5] = False  # ... and connected to the image border
    filled = M.fill_holes(m)
    assert not filled[12, 7]  # still background
    assert np.array_equal(filled, m)


def test_remove_small_components_removes_a_three_pixel_speck():
    m = np.zeros((40, 40), dtype=bool)
    m[5:25, 5:25] = True  # 400 px component
    m[35, 10:13] = True  # 3 px speck
    out = M.remove_small_components(m, min_px=10)
    assert out.dtype == bool
    assert not out[35, 10:13].any()
    assert np.array_equal(out[5:25, 5:25], m[5:25, 5:25])
    assert M.area(out) == 400


def test_remove_small_components_keeps_components_at_the_threshold():
    m = np.zeros((20, 20), dtype=bool)
    m[2, 2:6] = True  # exactly 4 px
    m[10, 10:12] = True  # 2 px
    out = M.remove_small_components(m, min_px=4)
    assert out[2, 2:6].all()
    assert not out[10, 10:12].any()


def test_remove_small_components_with_min_px_zero_is_identity():
    m = _square((30, 30), 1, 1, 5)
    assert np.array_equal(M.remove_small_components(m, 0), m)


# --------------------------------------------------------------------------
# tolerant symmetric difference / conflicts
# --------------------------------------------------------------------------


def test_tolerant_sym_diff_of_identical_masks_is_zero():
    a = _square()
    assert M.tolerant_sym_diff(a, a) == 0


def test_tolerant_sym_diff_ignores_a_one_pixel_shift():
    a = _square()
    assert M.tolerant_sym_diff(a, _shift(a, 0, 1)) == 0
    assert M.tolerant_sym_diff(a, _shift(a, 1, 1)) == 0


def test_tolerant_sym_diff_reports_a_five_pixel_shift():
    a = _square()
    assert M.tolerant_sym_diff(a, _shift(a, 0, 5)) > 0


def test_tolerant_sym_diff_band_absorbs_shifts_up_to_tol_px():
    """The band is grown around the reference boundary only, so a translation
    registers as soon as it exceeds tol_px pixels."""
    a = _square()
    assert M.tolerant_sym_diff(a, _shift(a, 0, 2), tol_px=2) == 0
    assert M.tolerant_sym_diff(a, _shift(a, 0, 3), tol_px=2) > 0
    assert M.tolerant_sym_diff(a, _shift(a, 0, 3), tol_px=4) == 0


def test_tolerant_sym_diff_grows_with_the_shift():
    a = _square()
    d6 = M.tolerant_sym_diff(a, _shift(a, 0, 6))
    d10 = M.tolerant_sym_diff(a, _shift(a, 0, 10))
    assert 0 < d6 < d10


def test_tolerant_sym_diff_is_asymmetric_in_its_reference():
    """The first argument is the reference: its boundary defines the band."""
    big = _square((64, 64), 10, 10, 30)  # covers 10..39
    small = _square((64, 64), 14, 14, 22)  # covers 14..35, eroded by 4
    ref_big = M.tolerant_sym_diff(big, small)
    ref_small = M.tolerant_sym_diff(small, big)
    assert ref_big > 0 and ref_small > 0
    assert ref_big != ref_small


def test_is_conflict_false_for_a_one_pixel_shift():
    a = _square(side=30)
    assert M.is_conflict(a, _shift(a, 0, 1)) is False


def test_is_conflict_true_for_a_forty_percent_erosion():
    """30x30 square eroded to 23x23 -- a ~41% area loss."""
    old = _square((64, 64), 10, 10, 30)
    new = _square((64, 64), 14, 14, 23)
    assert abs(M.area(new) / M.area(old) - 0.6) < 0.02
    assert M.is_conflict(old, new) is True


def test_is_conflict_true_for_a_forty_percent_erosion_at_a_larger_scale():
    old = _square((128, 128), 20, 20, 80)
    new = _square((128, 128), 29, 29, 62)
    assert abs(M.area(new) / M.area(old) - 0.6) < 0.02
    assert M.is_conflict(old, new) is True


def test_is_conflict_uses_the_min_px_floor_for_tiny_masks():
    old = _square((64, 64), 10, 10, 10)  # area 100 -> 2% is only 2 px
    new = _shift(old, 0, 20)  # disjoint
    assert M.is_conflict(old, new) is True
    assert M.is_conflict(old, new, min_px=10_000) is False


def test_is_conflict_false_for_identical_masks():
    a = _square()
    assert M.is_conflict(a, a) is False


# --------------------------------------------------------------------------
# polygons
# --------------------------------------------------------------------------


def test_polygon_roundtrip_iou_on_a_blob():
    hw = (96, 96)
    blob = _disk(hw, 48, 48, 26)
    blob[20:40, 45:75] = True  # a rectangular bump, so it is not just a disk
    polys = M.mask_to_polygons(blob, tol=1.0)
    assert polys and all(len(p) >= 6 and len(p) % 2 == 0 for p in polys)
    assert all(isinstance(v, float) for v in polys[0])
    back = M.polygons_to_mask(polys, hw)
    assert back.dtype == bool
    assert back.shape == hw
    iou = _iou(blob, back)
    assert iou >= 0.97, f"polygon roundtrip IoU too low: {iou:.4f}"


def test_mask_to_polygons_on_empty_mask_is_empty():
    assert M.mask_to_polygons(np.zeros((16, 16), dtype=bool)) == []


def test_mask_to_polygons_returns_one_polygon_per_component():
    m = np.zeros((64, 64), dtype=bool)
    m[5:20, 5:20] = True
    m[40:60, 40:60] = True
    assert len(M.mask_to_polygons(m)) == 2


def test_polygons_to_mask_unions_overlapping_polygons():
    polys = [
        [2.0, 2.0, 12.0, 2.0, 12.0, 12.0, 2.0, 12.0],
        [8.0, 8.0, 18.0, 8.0, 18.0, 18.0, 8.0, 18.0],
    ]
    out = M.polygons_to_mask(polys, (24, 24))
    assert out[3, 3] and out[16, 16] and out[10, 10]  # overlap stays filled


def test_polygons_to_mask_with_no_polygons_is_empty():
    out = M.polygons_to_mask([], (8, 9))
    assert out.shape == (8, 9) and not out.any()


# --------------------------------------------------------------------------
# warp
# --------------------------------------------------------------------------


def test_warp_mask_identity_is_unchanged():
    m = _square((64, 64), 12, 20, 15)
    out = M.warp_mask(m, Similarity(), (64, 64))
    assert out.dtype == bool
    assert np.array_equal(out, m)


def test_warp_mask_pure_translation():
    m = _square((64, 64), 10, 10, 12)
    out = M.warp_mask(m, Similarity(tx=5, ty=-3), (64, 64))
    assert np.array_equal(out, _shift(m, -3, 5))


def test_warp_mask_quarter_turn_matches_rot90():
    n = 48
    m = np.zeros((n, n), dtype=bool)
    m[5:20, 8:30] = True
    # x' = R(pi/2) x + (n-1, 0) maps (row y, col x) -> (row x, col n-1-y)
    out = M.warp_mask(m, Similarity(theta=np.pi / 2, tx=n - 1, ty=0), (n, n))
    assert np.array_equal(out, np.rot90(m, -1))


def test_warp_mask_scale_doubles_the_area():
    m = _square((80, 80), 4, 4, 20)  # rows/cols 4..23
    out = M.warp_mask(m, Similarity(scale=2.0), (80, 80))
    assert M.area(out) == 4 * M.area(m)
    # Nearest-neighbour inverse-maps output x' to input x'/2 and rounds, so
    # the scaled box straddles the ideal (8, 8, 48, 48) by half a pixel:
    # x'=7 samples 3.5 -> 4 (inside), x'=47 samples 23.5 -> 24 (outside).
    assert M.bbox(out) == (7, 7, 47, 47)


def test_warp_mask_honours_the_output_size():
    m = _square((64, 64), 10, 10, 20)
    out = M.warp_mask(m, Similarity(), (32, 100))
    assert out.shape == (32, 100)
    assert np.array_equal(out[:32, :64], m[:32, :64])


# --------------------------------------------------------------------------
# crop / paste
# --------------------------------------------------------------------------


def test_crop_returns_the_requested_window():
    m = np.zeros((20, 20), dtype=bool)
    m[5:10, 6:12] = True
    out = M.crop(m, (6, 5, 12, 10))
    assert out.shape == (5, 6)
    assert out.all()


def test_crop_zero_pads_outside_the_image():
    m = np.ones((10, 10), dtype=bool)
    out = M.crop(m, (-2, -2, 4, 4))
    assert out.shape == (6, 6)
    assert not out[:2, :].any() and not out[:, :2].any()
    assert out[2:, 2:].all()


def test_crop_paste_roundtrip():
    src = np.zeros((32, 32), dtype=bool)
    src[8:20, 4:18] = True
    window = (4, 8, 18, 20)
    patch = M.crop(src, window)
    dst = np.zeros((32, 32), dtype=bool)
    assert M.paste(dst, patch, (4, 8)) is None
    assert np.array_equal(dst, src)


def test_paste_overwrites_the_destination_region():
    dst = np.ones((10, 10), dtype=bool)
    M.paste(dst, np.zeros((3, 4), dtype=bool), (2, 1))
    assert not dst[1:4, 2:6].any()
    assert dst.sum() == 100 - 12


def test_paste_clips_at_the_border():
    dst = np.zeros((8, 8), dtype=bool)
    M.paste(dst, np.ones((4, 4), dtype=bool), (6, 6))
    assert dst[6:8, 6:8].all()
    assert M.area(dst) == 4


def test_paste_fully_outside_is_a_no_op():
    dst = np.zeros((8, 8), dtype=bool)
    M.paste(dst, np.ones((3, 3), dtype=bool), (20, 20))
    assert not dst.any()


# --------------------------------------------------------------------------
# labelmap
# --------------------------------------------------------------------------


def test_labelmap_from_masks_gives_later_instances_higher_ids():
    hw = (32, 32)
    a = _square(hw, 4, 4, 16)
    b = _square(hw, 12, 12, 16)
    lm, id2key = M.labelmap_from_masks({"a": a, "b": b}, ["a", "b"], hw)
    assert lm.dtype == np.uint16
    assert lm.shape == hw
    assert id2key == {1: "a", 2: "b"}
    assert lm[5, 5] == 1  # only a
    assert lm[25, 25] == 2  # only b
    assert lm[15, 15] == 2  # overlap -> the later (top) instance wins
    assert lm[0, 0] == 0  # background


def test_labelmap_order_controls_who_is_on_top():
    hw = (32, 32)
    a = _square(hw, 4, 4, 16)
    b = _square(hw, 12, 12, 16)
    lm, id2key = M.labelmap_from_masks({"a": a, "b": b}, ["b", "a"], hw)
    assert id2key == {1: "b", 2: "a"}
    assert lm[15, 15] == 2 and id2key[2] == "a"


def test_labelmap_skips_instances_without_a_mask_and_masks_without_an_order():
    hw = (16, 16)
    a = _square(hw, 1, 1, 4)
    c = _square(hw, 10, 10, 4)
    lm, id2key = M.labelmap_from_masks({"a": a, "c": c}, ["a", "b", "c"], hw)
    assert id2key == {1: "a", 2: "c"}
    lm2, id2key2 = M.labelmap_from_masks({"a": a, "c": c}, ["a"], hw)
    assert id2key2 == {1: "a"}
    assert not (lm2 == 2).any()
    assert lm2[11, 11] == 0


def test_labelmap_with_empty_order_is_all_background():
    lm, id2key = M.labelmap_from_masks({}, [], (5, 6))
    assert lm.shape == (5, 6) and not lm.any() and id2key == {}


def test_labelmap_rejects_more_than_uint16_ids():
    hw = (4, 4)
    with pytest.raises(ValueError):
        M.labelmap_from_masks(
            {str(i): np.zeros(hw, dtype=bool) for i in range(65_536)},
            [str(i) for i in range(65_536)],
            hw,
        )


# --------------------------------------------------------------------------
# counts of an RLE (one helper, three callers)
# --------------------------------------------------------------------------
def test_rle_counts_normalises_bytes_to_str():
    rle = M.encode_rle(_square((16, 16), 2, 2, 4))
    as_bytes = {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")}
    assert M.rle_counts(as_bytes) == rle["counts"]
    assert M.rle_counts(rle) == rle["counts"]


def test_rle_counts_of_nothing_is_none():
    assert M.rle_counts(None) is None
    assert M.rle_counts({}) is None
    assert M.rle_counts({"size": [4, 4]}) is None
