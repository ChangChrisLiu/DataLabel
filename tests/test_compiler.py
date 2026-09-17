"""Tests for the pure layer compiler (spec 3.3, task 7).

Keyframe selection, the geometry, the occluders, the frame overrides, the
problems and the input hash. The layer ordering has its own module,
``test_compiler_layers.py``; the builders are shared through
``compiler_scenes.py``.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from compiler_scenes import (
    A_BOX,
    B_BOX,
    BENCH,
    CHASSIS,
    KEY,
    covered,
    kf,
    po,
    rect,
    run,
    two_squares,
    zo,
)
from tda.core import masks
from tda.core.compiler import (
    CompiledFrame,
    CompiledInstance,
    compile_frame,
    derive_visibility,
    select_keyframe,
)
from tda.core.model import (
    FrameKey,
    FrameOverride,
    OccluderMask,
    Similarity,
    Visibility,
)


# --- select_keyframe ---
def test_select_keyframe_takes_the_smallest_anchor_at_or_after_the_step():
    kfs = [kf("A", {"main": A_BOX}, anchor_step=a, kf_id=a) for a in (10, 20, 40)]
    assert select_keyframe(kfs, 15).anchor_step == 20
    assert select_keyframe(kfs, 10).anchor_step == 10
    assert select_keyframe(kfs, 20).anchor_step == 20
    assert select_keyframe(kfs, 21).anchor_step == 40
    assert select_keyframe(kfs, 45) is None
    assert select_keyframe([], 1) is None


def test_select_keyframe_breaks_anchor_ties_on_the_newest_version():
    old = kf("A", {"main": A_BOX}, anchor_step=20, kf_id=1, version=1)
    new = kf("A", {"main": B_BOX}, anchor_step=20, kf_id=2, version=3)
    assert select_keyframe([old, new], 5) is new
    assert select_keyframe([new, old], 5) is new


def test_pose_segment_selects_the_chain_and_an_unset_one_is_reported():
    keyframes = {
        "A": [
            kf("A", {"main": A_BOX}, anchor_step=6, kf_id=1, pose_segment=1),
            kf("A", {"main": (44, 44, 60, 60)}, anchor_step=10, kf_id=2, pose_segment=2),
        ]
    }
    zorder = zo([("A", "main")])

    first = run(keyframes, zorder, pose_segment=1)
    assert first.problems == []
    assert first.instances["A"].keyframe_id == 1
    assert np.array_equal(first.instances["A"].amodal, rect(*A_BOX))

    second = run(keyframes, zorder, pose_segment=2)
    assert second.problems == []
    assert second.instances["A"].keyframe_id == 2
    assert np.array_equal(second.instances["A"].amodal, rect(44, 44, 60, 60))

    # left unset, the two chains are ambiguous: reported, highest segment wins
    # (the plain anchor rule would have picked segment 1's shape here)
    both = run(keyframes, zorder)
    assert "pose_segment_ambiguous:A" in both.problems
    assert both.instances["A"].keyframe_id == 2
    assert len({first.input_hash, second.input_hash, both.input_hash}) == 3


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
    out = run(**two_squares())
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
    assert a.occlusion_ratio == 400 / 1024
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
    out = run(**two_squares(), overrides=[po("A", "B")])
    a_mask, b_mask = rect(*A_BOX), rect(*B_BOX)
    a, b = out.instances["A"], out.instances["B"]
    assert np.array_equal(a.visible, a_mask)
    assert a.occlusion_ratio == 0.0
    assert a.visibility == Visibility.VISIBLE.value
    assert np.array_equal(b.visible, b_mask & ~a_mask)
    assert b.occlusion_ratio == 400 / 1024
    assert out.problems == []


def test_compile_frame_occluder_removes_pixels_from_every_instance():
    band = rect(0, 0, 64, 24)
    occ = OccluderMask(frame=KEY, occluder_type="hand", rle=masks.encode_rle(band))
    plain = run(**two_squares())
    out = run(**two_squares(), occluders=[occ])
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
    other = run(**two_squares(), occluders=[elsewhere])
    assert np.array_equal(other.instances["B"].visible, b_mask)
    assert other.input_hash == plain.input_hash


def test_occluders_subtract_from_bench_instances_too():
    band = rect(0, 0, 64, 20)
    keyframes = {"s1": [kf("s1", {"main": (10, 10, 30, 30)}, kf_id=1, placement=BENCH)]}
    out = run(
        keyframes,
        zo([("s1", "main")]),
        placements={"s1": BENCH},
        occluders=[OccluderMask(KEY, "hand", masks.encode_rle(band))],
    )
    assert np.array_equal(out.instances["s1"].visible, rect(10, 10, 30, 30) & ~band)
    assert out.instances["s1"].occlusion_ratio > 0


# --- frame overrides ---
def test_frame_override_visibility_wins_over_the_derived_label():
    blurred = FrameOverride(KEY, "A", visibility=Visibility.MOTION_BLUR.value)
    out = run(**two_squares(), frame_overrides={"A": blurred})
    a = out.instances["A"]
    assert a.visibility == Visibility.MOTION_BLUR.value
    assert np.array_equal(a.visible, rect(*A_BOX) & ~rect(*B_BOX))

    # out_of_view leaves no geometry behind at all
    gone = FrameOverride(KEY, "A", visibility=Visibility.OUT_OF_VIEW.value)
    out = run(**two_squares(), frame_overrides={"A": gone})
    a = out.instances["A"]
    assert a.visible is None
    assert a.box is None
    assert a.visibility == Visibility.OUT_OF_VIEW.value
    assert a.occlusion_ratio == 1.0
    assert np.array_equal(out.instances["B"].visible, rect(*B_BOX))  # B untouched


def test_frame_override_visible_rle_replaces_the_mask_before_deriving():
    fo = FrameOverride(KEY, "A", visible_rle=masks.encode_rle(rect(*A_BOX)))
    out = run(**two_squares(), frame_overrides={"A": fo})
    a = out.instances["A"]
    assert np.array_equal(a.visible, rect(*A_BOX))
    assert a.occlusion_ratio == 0.0
    assert a.visibility == Visibility.VISIBLE.value


def test_a_missing_shape_still_honours_a_visible_mask_override():
    fo = FrameOverride(KEY, "ghost", visible_rle=masks.encode_rle(rect(30, 30, 50, 50)))
    out = run(
        **two_squares(),
        needs={"A": "mask", "B": "mask", "ghost": "mask"},
        frame_overrides={"ghost": fo},
    )
    ghost = out.instances["ghost"]
    assert "missing_shape:ghost" in out.problems  # still missing a keyframe
    assert np.array_equal(ghost.visible, rect(30, 30, 50, 50))
    assert ghost.box == (30, 30, 50, 50)
    assert ghost.amodal is None
    assert ghost.occlusion_ratio == 0.0
    assert ghost.visibility == Visibility.VISIBLE.value


# --- problems ---
def test_missing_keyframe_for_a_chassis_instance_is_a_problem():
    scene = two_squares()
    scene["keyframes"]["ghost"] = []
    out = run(**scene, needs={"A": "mask", "B": "mask", "ghost": "mask"})
    assert "missing_shape:ghost" in out.problems
    ghost = out.instances["ghost"]
    assert ghost.visible is None and ghost.amodal is None and ghost.box is None
    assert ghost.keyframe_id is None
    assert ghost.occlusion_ratio == 0.0
    assert ghost.visibility == Visibility.OUT_OF_VIEW.value

    # a keyframe anchored before this step does not apply either
    early = run({"A": [kf("A", {"main": A_BOX}, anchor_step=3)]}, zo([("A", "main")]))
    assert "missing_shape:A" in early.problems
    assert early.instances["A"].visible is None


def test_a_missing_bench_shape_does_not_block_the_frame():
    out = run({}, zo([]), needs={"screw.01": "box"}, placements={"screw.01": BENCH})
    assert "bench_missing:screw.01" in out.problems
    assert "missing_shape:screw.01" not in out.problems
    assert out.instances["screw.01"].placement == BENCH
    # without explicit placements, a "box" need is a bench instance as well
    assert run({}, zo([]), needs={"screw.01": "box"}).problems == [
        "bench_missing:screw.01"
    ]


def test_an_instance_with_no_visible_pixels_is_reported_unless_overridden():
    out = run(**covered())
    small = out.instances["small"]
    assert "empty_visible:small" in out.problems
    assert masks.area(small.visible) == 0
    assert small.occlusion_ratio == 1.0
    assert small.visibility == Visibility.OCCLUDED_FULL.value
    assert small.box is None

    fo = FrameOverride(KEY, "small", visible_rle=masks.encode_rle(rect(20, 20, 30, 30)))
    fixed = run(**covered(), frame_overrides={"small": fo})
    assert "empty_visible:small" not in fixed.problems
    assert fixed.instances["small"].occlusion_ratio == 0.0


def test_a_shape_of_the_wrong_size_is_reported_and_skipped():
    # a 32x32 RLE cannot be placed on a 64x64 frame without guessing
    keyframes = {"A": [kf("A", {"main": (4, 4, 20, 20)}, kf_id=1, hw=(32, 32))]}
    out = run(keyframes, zo([("A", "main")]))
    assert "shape_size_mismatch:A/main" in out.problems
    assert masks.area(out.instances["A"].amodal) == 0
    assert masks.area(out.instances["A"].visible) == 0
    assert "empty_visible:A" in out.problems

    # so is an override mask of the wrong size, and it is not applied
    fo = FrameOverride(KEY, "A", visible_rle=masks.encode_rle(rect(4, 4, 20, 20, hw=(32, 32))))
    out = run(**two_squares(), frame_overrides={"A": fo})
    assert "shape_size_mismatch:A/override" in out.problems
    assert np.array_equal(out.instances["A"].visible, rect(*A_BOX) & ~rect(*B_BOX))

    # and an occluder of the wrong size
    occ = OccluderMask(KEY, "hand", masks.encode_rle(rect(0, 0, 32, 10, hw=(32, 32))))
    out = run(**two_squares(), occluders=[occ])
    assert "shape_size_mismatch:occluder/hand" in out.problems
    assert np.array_equal(out.instances["B"].visible, rect(*B_BOX))


# --- groups: bench vs chassis ---
def test_bench_and_chassis_instances_never_occlude_each_other():
    keyframes = {
        "board": [kf("board", {"main": (10, 10, 40, 40)}, kf_id=1)],
        "screw.01": [kf("screw.01", {"main": (20, 20, 45, 45)}, kf_id=2, placement=BENCH)],
    }
    out = run(
        keyframes,
        zo([("board", "main"), ("screw.01", "main")]),
        placements={"board": CHASSIS, "screw.01": BENCH},
    )
    assert out.problems == []
    assert np.array_equal(out.instances["board"].visible, rect(10, 10, 40, 40))
    assert out.instances["board"].occlusion_ratio == 0.0
    assert np.array_equal(out.instances["screw.01"].visible, rect(20, 20, 45, 45))
    assert out.instances["screw.01"].occlusion_ratio == 0.0


def test_two_bench_instances_do_occlude_each_other():
    keyframes = {
        "s1": [kf("s1", {"main": (10, 10, 30, 30)}, kf_id=1, placement=BENCH)],
        "s2": [kf("s2", {"main": (20, 20, 40, 40)}, kf_id=2, placement=BENCH)],
    }
    out = run(
        keyframes,
        zo([("s1", "main"), ("s2", "main")]),
        placements={"s1": BENCH, "s2": BENCH},
    )
    assert out.problems == []
    assert np.array_equal(
        out.instances["s1"].visible, rect(10, 10, 30, 30) & ~rect(20, 20, 40, 40)
    )
    assert out.instances["s1"].occlusion_ratio > 0
    assert np.array_equal(out.instances["s2"].visible, rect(20, 20, 40, 40))
    assert out.instances["s2"].occlusion_ratio == 0.0


def test_placement_selects_one_of_the_two_keyframe_chains():
    keyframes = {
        "cover": [
            kf("cover", {"main": (8, 8, 40, 40)}, kf_id=1),
            kf("cover", {"main": (44, 44, 60, 60)}, kf_id=2, placement=BENCH),
        ]
    }
    zorder = zo([("cover", "main")])
    installed = run(keyframes, zorder)
    assert installed.instances["cover"].keyframe_id == 1
    assert np.array_equal(installed.instances["cover"].amodal, rect(8, 8, 40, 40))

    detached = run(keyframes, zorder, placements={"cover": BENCH})
    assert detached.instances["cover"].keyframe_id == 2
    assert np.array_equal(detached.instances["cover"].amodal, rect(44, 44, 60, 60))

    # a keyframe of another view is not part of either chain
    elsewhere = run({"A": [kf("A", {"main": A_BOX}, kf_id=7, view="oak1")]}, zorder)
    assert "missing_shape:A" in elsewhere.problems


# --- box-only geometry ---
def test_a_box_instance_carries_only_a_box():
    keyframes = {
        "screw.01": [
            kf("screw.01", {"main": (20, 20, 40, 40)}, kf_id=5, geom_type="box", placement=BENCH)
        ]
    }
    out = run(keyframes, zo([]), needs={"screw.01": "box"})
    rec = out.instances["screw.01"]
    assert rec.visible is None
    assert rec.amodal is None
    assert rec.box == (20.0, 20.0, 40.0, 40.0)
    assert rec.occlusion_ratio == 0.0
    assert rec.visibility == Visibility.VISIBLE.value
    assert rec.placement == BENCH
    assert out.problems == []

    # an override may still hand it a visible mask for this one frame
    fo = FrameOverride(KEY, "screw.01", visible_rle=masks.encode_rle(rect(22, 22, 38, 38)))
    out = run(keyframes, zo([]), needs={"screw.01": "box"}, frame_overrides={"screw.01": fo})
    rec = out.instances["screw.01"]
    assert np.array_equal(rec.visible, rect(22, 22, 38, 38))
    assert rec.box == (22, 22, 38, 38)
    assert rec.amodal is None
    assert rec.occlusion_ratio == 0.0
    assert rec.visibility == Visibility.VISIBLE.value


def test_a_box_instance_neither_occludes_nor_is_occluded():
    keyframes = {
        "board": [kf("board", {"main": (10, 10, 40, 40)}, kf_id=1)],
        "tray": [kf("tray", {"main": (20, 20, 50, 50)}, kf_id=2, geom_type="box")],
    }
    out = run(keyframes, zo([("board", "main"), ("tray", "main")]))
    assert np.array_equal(out.instances["board"].visible, rect(10, 10, 40, 40))
    assert out.instances["board"].occlusion_ratio == 0.0
    assert out.instances["tray"].visible is None
    assert out.problems == []


def test_a_box_is_transformed_by_mapping_its_corners():
    keyframes = {
        "screw.01": [
            kf("screw.01", {"main": (10, 0, 20, 10)}, kf_id=5, geom_type="box", placement=BENCH)
        ]
    }
    needs = {"screw.01": "box"}
    shifted = run(keyframes, zo([]), needs=needs, transform=Similarity(tx=5, ty=3))
    assert shifted.instances["screw.01"].box == pytest.approx((15.0, 3.0, 25.0, 13.0))

    turned = run(keyframes, zo([]), needs=needs, transform=Similarity(theta=math.pi / 2))
    assert turned.instances["screw.01"].box == pytest.approx((-10.0, 10.0, 0.0, 20.0))


# --- transform ---
def test_a_similarity_transform_moves_the_shapes():
    keyframes = {"A": [kf("A", {"main": (10, 10, 40, 40)}, kf_id=1)]}
    out = run(keyframes, zo([("A", "main")]), transform=Similarity(tx=5, ty=0))
    assert np.array_equal(out.instances["A"].amodal, rect(15, 10, 45, 40))
    assert np.array_equal(out.instances["A"].visible, rect(15, 10, 45, 40))
    assert out.instances["A"].box == (15, 10, 45, 40)


def test_a_tiny_shape_is_labelled_too_small():
    out = run({"S": [kf("S", {"main": (30, 30, 34, 34)}, kf_id=1)]}, zo([("S", "main")]))
    assert out.instances["S"].visibility == Visibility.TOO_SMALL.value
    assert out.instances["S"].occlusion_ratio == 0.0


# --- input hash ---
def test_identical_inputs_give_an_identical_hash():
    first = run(**two_squares())
    second = run(**two_squares())
    assert first.input_hash == second.input_hash
    assert len(first.input_hash) == 40


@pytest.mark.parametrize(
    "what",
    [
        "keyframe_version",
        "compiler_version",
        "zorder_version",
        "zorder_order",
        "keyframe_shape",
        "keyframe_id",
        "override",
        "occluder",
        "frame_override_visibility",
        "frame_override_rle",
        "transform",
        "needs",
        "placements",
        "pose_segment",
        "step",
    ],
)
def test_every_input_change_changes_the_hash(what: str):
    reference = run(**two_squares()).input_hash
    scene = two_squares()
    kwargs: dict = {}
    if what == "keyframe_version":
        scene["keyframes"]["A"][0].version = 2
    elif what == "compiler_version":
        kwargs["compiler_version"] = "2"
    elif what == "zorder_version":
        scene["zorder"].version = 9
    elif what == "zorder_order":  # reordered in place, version untouched
        scene["zorder"] = zo([("B", "main"), ("A", "main")])
    elif what == "keyframe_shape":  # edited in place, version untouched
        scene["keyframes"]["A"][0].parts[0].rle = masks.encode_rle(rect(9, 9, 41, 41))
    elif what == "keyframe_id":
        scene["keyframes"]["A"][0].id = 99
    elif what == "override":
        kwargs["overrides"] = [po("A", "B")]
    elif what == "occluder":
        kwargs["occluders"] = [OccluderMask(KEY, "hand", masks.encode_rle(rect(0, 0, 64, 24)))]
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
    elif what == "pose_segment":
        kwargs["pose_segment"] = 1
    else:
        kwargs["key"] = FrameKey(desktop=1, step=6, view="scan")
    assert run(**scene, **kwargs).input_hash != reference


def test_the_hash_does_not_depend_on_dict_or_list_order():
    band1 = OccluderMask(KEY, "hand", masks.encode_rle(rect(0, 0, 64, 10)))
    band2 = OccluderMask(KEY, "tool", masks.encode_rle(rect(0, 54, 64, 64)))
    blur = FrameOverride(KEY, "A", visibility="motion_blur")
    patch = FrameOverride(KEY, "B", visible_rle=masks.encode_rle(rect(*B_BOX)))
    scene, other = two_squares(), two_squares()

    first = run(
        {"A": scene["keyframes"]["A"], "B": scene["keyframes"]["B"]},
        scene["zorder"],
        needs={"A": "mask", "B": "mask"},
        occluders=[band1, band2],
        frame_overrides={"A": blur, "B": patch},
    )
    second = run(
        {"B": other["keyframes"]["B"], "A": other["keyframes"]["A"]},
        other["zorder"],
        needs={"B": "mask", "A": "mask"},
        occluders=[band2, band1],
        frame_overrides={"B": patch, "A": blur},
    )
    assert first.input_hash == second.input_hash


def test_an_unselected_keyframe_does_not_affect_the_hash():
    scene = two_squares()
    scene["keyframes"]["A"].append(kf("A", {"main": B_BOX}, anchor_step=40, kf_id=77))
    reference = run(**copy.deepcopy(scene)).input_hash
    scene["keyframes"]["A"][1].version = 5
    assert run(**scene).input_hash == reference


def test_a_draft_keyframe_without_an_id_still_hashes_its_shape():
    zorder = zo([("A", "main")])
    first = run({"A": [kf("A", {"main": A_BOX}, kf_id=None)]}, zorder)
    second = run({"A": [kf("A", {"main": (9, 9, 41, 41)}, kf_id=None)]}, zorder)
    assert first.input_hash != second.input_hash


# --- purity ---
def test_compile_frame_is_deterministic_and_leaves_its_inputs_alone():
    scene = two_squares()
    occluders = [OccluderMask(KEY, "tool", masks.encode_rle(rect(0, 0, 64, 12)))]
    before_scene, before_occ = copy.deepcopy(scene), copy.deepcopy(occluders)
    first = run(**scene, occluders=occluders, overrides=[po("A", "B")])
    second = run(**scene, occluders=occluders, overrides=[po("A", "B")])

    assert scene["zorder"] == before_scene["zorder"]
    assert scene["keyframes"] == before_scene["keyframes"]
    assert occluders == before_occ
    assert first.input_hash == second.input_hash
    assert first.problems == second.problems
    for inst in first.instances:
        assert np.array_equal(first.instances[inst].visible, second.instances[inst].visible)
        assert np.array_equal(first.instances[inst].amodal, second.instances[inst].amodal)


def test_compile_frame_keeps_the_brief_signature():
    # positional order of the brief, placements/pose_segment keyword-only
    out = compile_frame(
        KEY, (64, 64), {"A": "mask"}, {"A": [kf("A", {"main": A_BOX})]},
        zo([("A", "main")]), [], [], {}, Similarity(), "1",
    )
    assert isinstance(out, CompiledFrame)
    assert np.array_equal(out.instances["A"].visible, rect(*A_BOX))
