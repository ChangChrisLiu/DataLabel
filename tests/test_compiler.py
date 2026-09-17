"""Tests for the pure layer compiler (spec 3.3, task 7).

Every mask is built synthetically on a 64x64 canvas and encoded with
:func:`tda.core.masks.encode_rle`, so the whole file is free of fixtures and
of I/O.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from tda.core import masks
from tda.core.compiler import (
    CompiledFrame,
    CompiledInstance,
    above,
    compile_frame,
    derive_visibility,
    select_keyframe,
)
from tda.core.model import (
    FrameKey,
    FrameOverride,
    OccluderMask,
    PairOverride,
    Placement,
    ShapeKeyframe,
    ShapePart,
    Similarity,
    Visibility,
    ZOrderRec,
)

HW = (64, 64)
KEY = FrameKey(desktop=1, step=5, view="scan")
IDENTITY = Similarity()
BENCH = Placement.ON_BENCH.value
CHASSIS = Placement.IN_CHASSIS.value


# --- builders ---
def rect(x0: int, y0: int, x1: int, y1: int, hw=HW) -> np.ndarray:
    """A filled rectangle ``[x0, x1) x [y0, y1)`` as a bool mask."""
    m = np.zeros(hw, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _kf(
    instance: str,
    rects: dict[str, tuple[int, int, int, int]],
    *,
    anchor_step: int = 10,
    kf_id: int | None = 1,
    version: int = 1,
    placement: str = CHASSIS,
    geom_type: str = "mask",
    view: str = "scan",
) -> ShapeKeyframe:
    """A keyframe whose parts are the given rectangles (mask or box geometry)."""
    parts = []
    for name, box in rects.items():
        if geom_type == "box":
            parts.append(ShapePart(name, None, tuple(float(v) for v in box)))
        else:
            parts.append(ShapePart(name, masks.encode_rle(rect(*box)), box))
    return ShapeKeyframe(
        id=kf_id,
        instance=instance,
        desktop=1,
        view=view,
        pose_segment=1,
        anchor_step=anchor_step,
        placement=placement,
        geom_type=geom_type,
        parts=parts,
        version=version,
    )


def _zo(order: list[tuple[str, str]], version: int = 1) -> ZOrderRec:
    return ZOrderRec(desktop=1, view="scan", pose_segment=1, order=order, version=version)


def _po(above_key: str, below_key: str) -> PairOverride:
    return PairOverride(1, "scan", 1, above=above_key, below=below_key)


def _compile(
    keyframes: dict[str, list[ShapeKeyframe]],
    zorder: ZOrderRec,
    *,
    needs: dict[str, str] | None = None,
    overrides=(),
    occluders=(),
    frame_overrides=None,
    transform: Similarity = IDENTITY,
    placements: dict[str, str] | None = None,
    compiler_version: str = "1",
    key: FrameKey = KEY,
) -> CompiledFrame:
    """compile_frame with the boilerplate filled in (needs defaults to masks)."""
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
    )


#: A at (8..40), B at (20..52); B is on top in the global order.
A_BOX = (8, 8, 40, 40)
B_BOX = (20, 20, 52, 52)


def _two_squares() -> dict:
    return {
        "keyframes": {
            "A": [_kf("A", {"main": A_BOX}, kf_id=11)],
            "B": [_kf("B", {"main": B_BOX}, kf_id=22)],
        },
        "zorder": _zo([("A", "main"), ("B", "main")]),
    }


def _covered() -> dict:
    """A small instance completely hidden under a big cover."""
    return {
        "keyframes": {
            "small": [_kf("small", {"main": (20, 20, 30, 30)}, kf_id=1)],
            "cover": [_kf("cover", {"main": (10, 10, 50, 50)}, kf_id=2)],
        },
        "zorder": _zo([("small", "main"), ("cover", "main")]),
    }


# --- select_keyframe ---
def test_select_keyframe_takes_the_smallest_anchor_at_or_after_the_step():
    kfs = [_kf("A", {"main": A_BOX}, anchor_step=a, kf_id=a) for a in (10, 20, 40)]
    assert select_keyframe(kfs, 15).anchor_step == 20
    assert select_keyframe(kfs, 10).anchor_step == 10
    assert select_keyframe(kfs, 20).anchor_step == 20
    assert select_keyframe(kfs, 21).anchor_step == 40
    assert select_keyframe(kfs, 45) is None
    assert select_keyframe([], 1) is None


def test_select_keyframe_breaks_anchor_ties_on_the_newest_version():
    old = _kf("A", {"main": A_BOX}, anchor_step=20, kf_id=1, version=1)
    new = _kf("A", {"main": B_BOX}, anchor_step=20, kf_id=2, version=3)
    assert select_keyframe([old, new], 5) is new
    assert select_keyframe([new, old], 5) is new


# --- above ---
def test_above_uses_the_global_order_and_treats_unlisted_layers_as_topmost():
    zorder = _zo([("A", "main"), ("B", "main")])
    assert above(("B", "main"), ("A", "main"), zorder, []) is True
    assert above(("A", "main"), ("B", "main"), zorder, []) is False
    assert above(("A", "main"), ("A", "main"), zorder, []) is False
    # unlisted layers float to the top, two unlisted ones are unordered
    assert above(("C", "main"), ("B", "main"), zorder, []) is True
    assert above(("B", "main"), ("C", "main"), zorder, []) is False
    assert above(("C", "main"), ("D", "main"), zorder, []) is False


def test_above_override_matches_on_instance_keys_regardless_of_part():
    zorder = _zo([("A", "main"), ("B", "main")])
    ov = [_po("A", "B")]
    assert above(("A", "main"), ("B", "main"), zorder, ov) is True
    assert above(("B", "main"), ("A", "main"), zorder, ov) is False
    # the override also decides for parts that are not in the global order
    assert above(("A", "wall"), ("B", "floor"), zorder, ov) is True


# --- derive_visibility ---
def test_derive_visibility_ratio_thresholds():
    big = rect(10, 10, 50, 50)
    assert derive_visibility(big, big, 0.0) == Visibility.VISIBLE.value
    assert derive_visibility(big, big, 0.29) == Visibility.VISIBLE.value
    assert derive_visibility(big, big, 0.30) == Visibility.OCCLUDED_PARTIAL.value
    assert derive_visibility(big, big, 0.94) == Visibility.OCCLUDED_PARTIAL.value
    assert derive_visibility(big, big, 0.95) == Visibility.OCCLUDED_FULL.value
    assert derive_visibility(big, big, 1.0) == Visibility.OCCLUDED_FULL.value
    # full occlusion wins over a handful of surviving pixels
    assert derive_visibility(rect(30, 30, 34, 34), big, 0.97) == Visibility.OCCLUDED_FULL.value
    assert derive_visibility(None, None, 0.0) == Visibility.OUT_OF_VIEW.value


def test_derive_visibility_size_thresholds():
    amodal = rect(0, 0, 60, 60)
    small, tiny = Visibility.TOO_SMALL.value, Visibility.VISIBLE_TINY.value
    assert derive_visibility(rect(30, 30, 34, 34), amodal, 0.0) == small
    assert derive_visibility(rect(30, 30, 35, 35), amodal, 0.0) == small
    assert derive_visibility(rect(30, 30, 36, 36), amodal, 0.0) == tiny
    assert derive_visibility(rect(30, 30, 41, 41), amodal, 0.0) == tiny
    assert derive_visibility(rect(30, 30, 42, 42), amodal, 0.0) == Visibility.VISIBLE.value
    # a thin sliver is measured on the bounding box's shorter side
    assert derive_visibility(rect(2, 30, 60, 35), amodal, 0.0) == small


# --- compile_frame: two overlapping squares ---
def test_compile_frame_subtracts_the_layer_above():
    out = _compile(**_two_squares())
    a_mask, b_mask = rect(*A_BOX), rect(*B_BOX)

    assert isinstance(out, CompiledFrame)
    assert out.key == KEY
    assert set(out.instances) == {"A", "B"}
    assert out.problems == []

    a = out.instances["A"]
    assert isinstance(a, CompiledInstance)
    assert np.array_equal(a.visible, a_mask & ~b_mask)
    assert np.array_equal(a.amodal, a_mask)
    assert a.occlusion_ratio > 0
    assert a.occlusion_ratio == pytest.approx(400 / 1024)
    assert a.visibility == Visibility.OCCLUDED_PARTIAL.value
    assert a.placement == CHASSIS
    assert a.keyframe_id == 11
    assert a.box == (8, 8, 40, 40)

    b = out.instances["B"]
    assert np.array_equal(b.visible, b_mask)
    assert b.occlusion_ratio == 0.0
    assert b.visibility == Visibility.VISIBLE.value
    assert b.keyframe_id == 22


def test_compile_frame_pair_override_flips_the_pair():
    out = _compile(**_two_squares(), overrides=[_po("A", "B")])
    a_mask, b_mask = rect(*A_BOX), rect(*B_BOX)
    a, b = out.instances["A"], out.instances["B"]
    assert np.array_equal(a.visible, a_mask)
    assert a.occlusion_ratio == 0.0
    assert a.visibility == Visibility.VISIBLE.value
    assert np.array_equal(b.visible, b_mask & ~a_mask)
    assert b.occlusion_ratio == pytest.approx(400 / 1024)
    assert out.problems == []


def test_compile_frame_occluder_removes_pixels_from_every_instance():
    band = rect(0, 0, 64, 24)
    occ = OccluderMask(frame=KEY, occluder_type="hand", rle=masks.encode_rle(band))
    plain = _compile(**_two_squares())
    out = _compile(**_two_squares(), occluders=[occ])
    a_mask, b_mask = rect(*A_BOX), rect(*B_BOX)

    assert np.array_equal(out.instances["A"].visible, a_mask & ~b_mask & ~band)
    assert np.array_equal(out.instances["B"].visible, b_mask & ~band)
    assert out.instances["A"].occlusion_ratio > plain.instances["A"].occlusion_ratio
    assert out.instances["B"].occlusion_ratio > 0.0

    # an occluder of another frame is not an input of this one
    elsewhere = OccluderMask(
        frame=FrameKey(desktop=1, step=6, view="scan"),
        occluder_type="hand",
        rle=masks.encode_rle(band),
    )
    other = _compile(**_two_squares(), occluders=[elsewhere])
    assert np.array_equal(other.instances["B"].visible, b_mask)
    assert other.input_hash == plain.input_hash


# --- frame overrides ---
def test_frame_override_visibility_wins_over_the_derived_label():
    blurred = FrameOverride(KEY, "A", visibility=Visibility.MOTION_BLUR.value)
    out = _compile(**_two_squares(), frame_overrides={"A": blurred})
    a = out.instances["A"]
    assert a.visibility == Visibility.MOTION_BLUR.value
    assert np.array_equal(a.visible, rect(*A_BOX) & ~rect(*B_BOX))

    # out_of_view leaves no geometry behind at all
    gone = FrameOverride(KEY, "A", visibility=Visibility.OUT_OF_VIEW.value)
    out = _compile(**_two_squares(), frame_overrides={"A": gone})
    a = out.instances["A"]
    assert a.visible is None
    assert a.box is None
    assert a.visibility == Visibility.OUT_OF_VIEW.value
    assert a.occlusion_ratio == 1.0
    assert np.array_equal(out.instances["B"].visible, rect(*B_BOX))  # B untouched


def test_frame_override_visible_rle_replaces_the_mask_before_deriving():
    fo = FrameOverride(KEY, "A", visible_rle=masks.encode_rle(rect(*A_BOX)))
    out = _compile(**_two_squares(), frame_overrides={"A": fo})
    a = out.instances["A"]
    assert np.array_equal(a.visible, rect(*A_BOX))
    assert a.occlusion_ratio == 0.0
    assert a.visibility == Visibility.VISIBLE.value


# --- problems ---
def test_missing_keyframe_for_a_chassis_instance_is_a_problem():
    scene = _two_squares()
    scene["keyframes"]["ghost"] = []
    out = _compile(**scene, needs={"A": "mask", "B": "mask", "ghost": "mask"})
    assert "missing_shape:ghost" in out.problems
    ghost = out.instances["ghost"]
    assert ghost.visible is None and ghost.amodal is None and ghost.box is None
    assert ghost.keyframe_id is None
    assert ghost.occlusion_ratio == 0.0
    assert ghost.visibility == Visibility.OUT_OF_VIEW.value

    # a keyframe anchored before this step does not apply either
    early = _compile({"A": [_kf("A", {"main": A_BOX}, anchor_step=3)]}, _zo([("A", "main")]))
    assert "missing_shape:A" in early.problems
    assert early.instances["A"].visible is None


def test_a_missing_bench_shape_does_not_block_the_frame():
    out = _compile({}, _zo([]), needs={"screw.01": "box"}, placements={"screw.01": BENCH})
    assert "bench_missing:screw.01" in out.problems
    assert "missing_shape:screw.01" not in out.problems
    assert out.instances["screw.01"].placement == BENCH
    # without explicit placements, a "box" need is a bench instance as well
    assert _compile({}, _zo([]), needs={"screw.01": "box"}).problems == [
        "bench_missing:screw.01"
    ]


def test_a_part_missing_from_the_zorder_floats_to_the_top():
    scene = _two_squares()
    scene["zorder"] = _zo([("A", "main")])
    out = _compile(**scene)
    assert "zorder_missing:B/main" in out.problems
    assert np.array_equal(out.instances["A"].visible, rect(*A_BOX) & ~rect(*B_BOX))
    assert np.array_equal(out.instances["B"].visible, rect(*B_BOX))


def test_an_instance_with_no_visible_pixels_is_reported_unless_overridden():
    out = _compile(**_covered())
    small = out.instances["small"]
    assert "empty_visible:small" in out.problems
    assert masks.area(small.visible) == 0
    assert small.occlusion_ratio == 1.0
    assert small.visibility == Visibility.OCCLUDED_FULL.value
    assert small.box is None

    fo = FrameOverride(KEY, "small", visible_rle=masks.encode_rle(rect(20, 20, 30, 30)))
    fixed = _compile(**_covered(), frame_overrides={"small": fo})
    assert "empty_visible:small" not in fixed.problems
    assert fixed.instances["small"].occlusion_ratio == 0.0


# --- zorder overrides: topological sort and cycles ---
def _three_squares() -> dict:
    return {
        "keyframes": {
            "A": [_kf("A", {"main": (8, 8, 40, 40)}, kf_id=1)],
            "B": [_kf("B", {"main": (16, 16, 48, 48)}, kf_id=2)],
            "C": [_kf("C", {"main": (24, 24, 56, 56)}, kf_id=3)],
        },
        "zorder": _zo([("A", "main"), ("B", "main"), ("C", "main")]),
    }


def test_an_override_against_the_global_order_is_resolved_topologically():
    a, b, c = rect(8, 8, 40, 40), rect(16, 16, 48, 48), rect(24, 24, 56, 56)
    plain = _compile(**_three_squares())
    assert np.array_equal(plain.instances["C"].visible, c)
    assert np.array_equal(plain.instances["B"].visible, b & ~c)
    assert np.array_equal(plain.instances["A"].visible, a & ~b & ~c)

    # "A above C" contradicts the global order A < B < C; B must stay at the
    # bottom (the global order is the tie-breaker), giving B < C < A.
    out = _compile(**_three_squares(), overrides=[_po("A", "C")])
    assert out.problems == []
    assert np.array_equal(out.instances["A"].visible, a)
    assert np.array_equal(out.instances["C"].visible, c & ~a)
    assert np.array_equal(out.instances["B"].visible, b & ~c & ~a)


def test_a_cycle_of_overrides_is_reported_and_falls_back_to_the_global_order():
    out = _compile(**_two_squares(), overrides=[_po("A", "B"), _po("B", "A")])
    assert "zorder_cycle:A,B" in out.problems
    a_mask, b_mask = rect(*A_BOX), rect(*B_BOX)
    assert np.array_equal(out.instances["A"].visible, a_mask & ~b_mask)
    assert np.array_equal(out.instances["B"].visible, b_mask)


# --- groups: bench vs chassis ---
def test_bench_and_chassis_instances_never_occlude_each_other():
    keyframes = {
        "board": [_kf("board", {"main": (10, 10, 40, 40)}, kf_id=1)],
        "screw.01": [_kf("screw.01", {"main": (20, 20, 45, 45)}, kf_id=2, placement=BENCH)],
    }
    out = _compile(
        keyframes,
        _zo([("board", "main"), ("screw.01", "main")]),
        placements={"board": CHASSIS, "screw.01": BENCH},
    )
    assert out.problems == []
    assert np.array_equal(out.instances["board"].visible, rect(10, 10, 40, 40))
    assert out.instances["board"].occlusion_ratio == 0.0
    assert np.array_equal(out.instances["screw.01"].visible, rect(20, 20, 45, 45))
    assert out.instances["screw.01"].occlusion_ratio == 0.0


def test_placement_selects_one_of_the_two_keyframe_chains():
    keyframes = {
        "cover": [
            _kf("cover", {"main": (8, 8, 40, 40)}, kf_id=1),
            _kf("cover", {"main": (44, 44, 60, 60)}, kf_id=2, placement=BENCH),
        ]
    }
    zorder = _zo([("cover", "main")])
    installed = _compile(keyframes, zorder)
    assert installed.instances["cover"].keyframe_id == 1
    assert np.array_equal(installed.instances["cover"].amodal, rect(8, 8, 40, 40))

    detached = _compile(keyframes, zorder, placements={"cover": BENCH})
    assert detached.instances["cover"].keyframe_id == 2
    assert np.array_equal(detached.instances["cover"].amodal, rect(44, 44, 60, 60))

    # a keyframe of another view is not part of either chain
    elsewhere = _compile({"A": [_kf("A", {"main": A_BOX}, kf_id=7, view="oak1")]}, zorder)
    assert "missing_shape:A" in elsewhere.problems


# --- box-only geometry ---
def test_a_box_instance_carries_only_a_box():
    keyframes = {
        "screw.01": [
            _kf("screw.01", {"main": (20, 20, 40, 40)}, kf_id=5, geom_type="box", placement=BENCH)
        ]
    }
    out = _compile(keyframes, _zo([]), needs={"screw.01": "box"})
    rec = out.instances["screw.01"]
    assert rec.visible is None
    assert rec.amodal is None
    assert rec.box == (20.0, 20.0, 40.0, 40.0)
    assert rec.occlusion_ratio == 0.0
    assert rec.visibility == Visibility.VISIBLE.value
    assert rec.placement == BENCH
    assert out.problems == []


def test_a_box_instance_neither_occludes_nor_is_occluded():
    keyframes = {
        "board": [_kf("board", {"main": (10, 10, 40, 40)}, kf_id=1)],
        "tray": [_kf("tray", {"main": (20, 20, 50, 50)}, kf_id=2, geom_type="box")],
    }
    out = _compile(keyframes, _zo([("board", "main"), ("tray", "main")]))
    assert np.array_equal(out.instances["board"].visible, rect(10, 10, 40, 40))
    assert out.instances["board"].occlusion_ratio == 0.0
    assert out.instances["tray"].visible is None
    assert out.problems == []


def test_a_box_is_transformed_by_mapping_its_corners():
    keyframes = {
        "screw.01": [
            _kf("screw.01", {"main": (10, 0, 20, 10)}, kf_id=5, geom_type="box", placement=BENCH)
        ]
    }
    needs = {"screw.01": "box"}
    shifted = _compile(keyframes, _zo([]), needs=needs, transform=Similarity(tx=5, ty=3))
    assert shifted.instances["screw.01"].box == pytest.approx((15.0, 3.0, 25.0, 13.0))

    turned = _compile(
        keyframes, _zo([]), needs=needs, transform=Similarity(theta=math.pi / 2)
    )
    assert turned.instances["screw.01"].box == pytest.approx((-10.0, 10.0, 0.0, 20.0))


# --- multi-part shapes ---
def test_multi_part_shapes_are_layered_part_by_part():
    floor, wall = (8, 40, 24, 56), (8, 8, 24, 24)
    keyframes = {
        "cage": [_kf("cage", {"floor": floor, "wall": wall}, kf_id=1)],
        "board": [_kf("board", {"main": (8, 20, 56, 44)}, kf_id=2)],
    }
    zorder = _zo([("cage", "floor"), ("board", "main"), ("cage", "wall")])
    out = _compile(keyframes, zorder)
    floor_m, wall_m, board_m = rect(*floor), rect(*wall), rect(8, 20, 56, 44)

    cage = out.instances["cage"]
    assert np.array_equal(cage.amodal, floor_m | wall_m)
    assert np.array_equal(cage.visible, wall_m | (floor_m & ~board_m))
    assert cage.occlusion_ratio == pytest.approx(64 / 512)

    board = out.instances["board"]
    assert np.array_equal(board.visible, board_m & ~wall_m)
    assert board.occlusion_ratio == pytest.approx(64 / 1152)
    assert out.problems == []


# --- transform ---
def test_a_similarity_transform_moves_the_shapes():
    keyframes = {"A": [_kf("A", {"main": (10, 10, 40, 40)}, kf_id=1)]}
    out = _compile(keyframes, _zo([("A", "main")]), transform=Similarity(tx=5, ty=0))
    assert np.array_equal(out.instances["A"].amodal, rect(15, 10, 45, 40))
    assert np.array_equal(out.instances["A"].visible, rect(15, 10, 45, 40))
    assert out.instances["A"].box == (15, 10, 45, 40)


def test_a_tiny_shape_is_labelled_too_small():
    out = _compile({"S": [_kf("S", {"main": (30, 30, 34, 34)}, kf_id=1)]}, _zo([("S", "main")]))
    assert out.instances["S"].visibility == Visibility.TOO_SMALL.value
    assert out.instances["S"].occlusion_ratio == 0.0


# --- input hash ---
def test_identical_inputs_give_an_identical_hash():
    first = _compile(**_two_squares())
    second = _compile(**_two_squares())
    assert first.input_hash == second.input_hash
    assert len(first.input_hash) == 40


@pytest.mark.parametrize(
    "what",
    [
        "keyframe_version",
        "compiler_version",
        "zorder_version",
        "keyframe_id",
        "override",
        "occluder",
        "frame_override_visibility",
        "frame_override_rle",
        "transform",
        "needs",
        "placements",
        "step",
    ],
)
def test_every_input_change_changes_the_hash(what: str):
    reference = _compile(**_two_squares()).input_hash
    scene = _two_squares()
    kwargs: dict = {}
    if what == "keyframe_version":
        scene["keyframes"]["A"][0].version = 2
    elif what == "compiler_version":
        kwargs["compiler_version"] = "2"
    elif what == "zorder_version":
        scene["zorder"].version = 9
    elif what == "keyframe_id":
        scene["keyframes"]["A"][0].id = 99
    elif what == "override":
        kwargs["overrides"] = [_po("A", "B")]
    elif what == "occluder":
        kwargs["occluders"] = [
            OccluderMask(KEY, "hand", masks.encode_rle(rect(0, 0, 64, 24)))
        ]
    elif what == "frame_override_visibility":
        kwargs["frame_overrides"] = {"A": FrameOverride(KEY, "A", visibility="motion_blur")}
    elif what == "frame_override_rle":
        kwargs["frame_overrides"] = {
            "A": FrameOverride(KEY, "A", visible_rle=masks.encode_rle(rect(8, 8, 20, 20)))
        }
    elif what == "transform":
        kwargs["transform"] = Similarity(tx=1.0)
    elif what == "needs":
        kwargs["needs"] = {"A": "mask", "B": "box"}
    elif what == "placements":
        kwargs["placements"] = {"A": BENCH, "B": CHASSIS}
    else:
        kwargs["key"] = FrameKey(desktop=1, step=6, view="scan")
    assert _compile(**scene, **kwargs).input_hash != reference


def test_an_unselected_keyframe_does_not_affect_the_hash():
    scene = _two_squares()
    scene["keyframes"]["A"].append(_kf("A", {"main": B_BOX}, anchor_step=40, kf_id=77))
    reference = _compile(**copy.deepcopy(scene)).input_hash
    scene["keyframes"]["A"][1].version = 5
    assert _compile(**scene).input_hash == reference


def test_a_draft_keyframe_without_an_id_still_hashes_its_shape():
    zorder = _zo([("A", "main")])
    first = _compile({"A": [_kf("A", {"main": A_BOX}, kf_id=None)]}, zorder)
    second = _compile({"A": [_kf("A", {"main": (9, 9, 41, 41)}, kf_id=None)]}, zorder)
    assert first.input_hash != second.input_hash


# --- purity ---
def test_compile_frame_is_deterministic_and_leaves_its_inputs_alone():
    scene = _two_squares()
    occluders = [OccluderMask(KEY, "tool", masks.encode_rle(rect(0, 0, 64, 12)))]
    before_scene, before_occ = copy.deepcopy(scene), copy.deepcopy(occluders)
    first = _compile(**scene, occluders=occluders, overrides=[_po("A", "B")])
    second = _compile(**scene, occluders=occluders, overrides=[_po("A", "B")])

    assert scene["zorder"] == before_scene["zorder"]
    assert scene["keyframes"] == before_scene["keyframes"]
    assert occluders == before_occ
    assert first.input_hash == second.input_hash
    assert first.problems == second.problems
    for inst in first.instances:
        assert np.array_equal(first.instances[inst].visible, second.instances[inst].visible)
        assert np.array_equal(first.instances[inst].amodal, second.instances[inst].amodal)
