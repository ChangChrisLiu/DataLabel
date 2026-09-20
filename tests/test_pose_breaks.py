"""The pure re-cut arithmetic of :mod:`tda.core.pose_breaks` (Plan B, task B1).

Nothing here touches a database: a re-cut is decided as a plan first -- which
ranges the view ends up with, which old segment number becomes which new one,
and which old segment was split into several -- and only then applied. Every
case the annotator can reach is a case here: a break inside a segment, a break
that duplicates a ``reorient``, a break outside the machine, and the removal of
a break (which merges two segments back and renumbers everything after them
down).
"""
from __future__ import annotations

import sqlite3

import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.model import (
    FrameKey,
    FrameOverride,
    InstanceRec,
    OccluderMask,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    StepRec,
    ZOrderRec,
)
from tda.core.pose_breaks import boundaries, recut_plan, straddling
from truth_scenes import DESKTOP, PSU, SCREW, VIEW, rect


# --------------------------------------------------------------------------- #
# boundaries: reorient steps union the accepted breaks of one view
# --------------------------------------------------------------------------- #
def test_a_break_at_a_reorient_step_is_one_boundary_not_two():
    assert boundaries(42, [16], [16, 19]) == [16, 19]


def test_boundaries_outside_the_machine_are_dropped():
    """Step 1 starts segment 1 already, and there is no step 43."""
    assert boundaries(42, [], [1, 43]) == []
    assert boundaries(42, [], [0, -5, 42]) == [42]


def test_boundaries_are_sorted_and_deduplicated():
    assert boundaries(42, [30, 16], [19, 30]) == [16, 19, 30]


def test_no_steps_means_no_boundaries():
    assert boundaries(0, [3], [4]) == []


# --------------------------------------------------------------------------- #
# recut_plan: ranges, renumbering, splits
# --------------------------------------------------------------------------- #
def test_one_segment_cut_in_two():
    plan = recut_plan([(1, 1, 42)], [19], 42)
    assert plan.ranges == [(1, 1, 18), (2, 19, 42)]
    assert plan.renumber == {1: 1}
    assert plan.split == {1: [1, 2]}
    assert plan.changed is True


def test_a_break_inside_the_first_of_two_segments_renumbers_the_second_up():
    plan = recut_plan([(1, 1, 15), (2, 16, 42)], [10, 16], 42)
    assert plan.ranges == [(1, 1, 9), (2, 10, 15), (3, 16, 42)]
    assert plan.renumber == {1: 1, 2: 3}
    assert plan.split == {1: [1, 2]}


def test_removing_a_boundary_merges_and_renumbers_down():
    plan = recut_plan([(1, 1, 9), (2, 10, 15), (3, 16, 42)], [16], 42)
    assert plan.ranges == [(1, 1, 15), (2, 16, 42)]
    assert plan.renumber == {1: 1, 2: 1, 3: 2}
    assert plan.split == {}
    assert plan.merged == {1: [1, 2], 2: [3]}


def test_a_plan_that_changes_nothing_says_so():
    plan = recut_plan([(1, 1, 9), (2, 10, 42)], [10], 42)
    assert plan.ranges == [(1, 1, 9), (2, 10, 42)]
    assert plan.renumber == {1: 1, 2: 2}
    assert plan.split == {} and plan.changed is False


def test_the_segment_of_a_step_is_readable_from_the_plan():
    plan = recut_plan([(1, 1, 42)], [19], 42)
    assert plan.segment_of(1) == 1
    assert plan.segment_of(18) == 1
    assert plan.segment_of(19) == 2
    assert plan.segment_of(99) is None


def test_an_old_segment_whose_range_moved_is_reported_as_affected():
    """Both sides of a split lose or gain keyframes, so both are affected."""
    plan = recut_plan([(1, 1, 42)], [19], 42)
    assert plan.affected_steps() == set(range(1, 43))
    # a second, untouched view-wide re-cut touches nothing
    assert recut_plan([(1, 1, 42)], [], 42).affected_steps() == set()


def test_a_view_with_no_segments_yet_plans_nothing():
    plan = recut_plan([], [19], 42)
    assert plan.ranges == [] and plan.renumber == {} and plan.changed is False


# --------------------------------------------------------------------------- #
# which keyframes a new boundary cuts through
# --------------------------------------------------------------------------- #
def test_straddling_finds_the_keyframe_whose_coverage_crosses_the_boundary():
    """Anchors 6 and 12 in [1, 15]: 12 covers (6, 12] and 6 covers [1, 6]."""
    assert straddling([6, 12], start=1, boundary=10) == [12]
    assert straddling([6, 12], start=1, boundary=8) == [12]
    # a boundary right after an anchor cuts nothing: (6, 12] starts at 7 already
    assert straddling([6, 12], start=1, boundary=7) == []
    assert straddling([6, 12], start=1, boundary=13) == []


def test_the_first_keyframe_of_a_chain_covers_from_the_segment_start():
    assert straddling([12], start=1, boundary=10) == [12]
    assert straddling([12], start=12, boundary=12) == []


def test_a_keyframe_entirely_on_one_side_never_straddles():
    assert straddling([4, 9], start=1, boundary=10) == []      # both below
    # 12 covers [1, 12] and 16 covers [13, 16]: the boundary falls between them
    assert straddling([12, 16], start=1, boundary=13) == []


# --------------------------------------------------------------------------- #
# the transactional re-cut (task B1 step 3)
# --------------------------------------------------------------------------- #
#: The scene these tests re-cut: eight steps of one view, with something in
#: every table a pose segment keys -- two shape chains, a layer order, a pair
#: override, a frame-level occluder and frame override, and two frozen frames.
STEPS = 8
BOUNDARY = 5


def _keyframe(instance: str, box, anchor: int, seg: int = 1) -> ShapeKeyframe:
    return ShapeKeyframe(
        id=None, instance=instance, desktop=DESKTOP, view=VIEW, pose_segment=seg,
        anchor_step=anchor, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(*box)))],
    )


@pytest.fixture
def scene(tmp_db_path: str):
    """One view, one segment [1, 8], and a row in every table it keys."""
    db = Db(tmp_db_path)
    db.upsert_instance(InstanceRec(key=PSU, desktop=DESKTOP, cls="psu"))
    db.upsert_instance(InstanceRec(key=SCREW, desktop=DESKTOP, cls="screw",
                                   parent=PSU, attached=True, fastens=PSU))
    db.replace_steps(
        DESKTOP,
        [StepRec(DESKTOP, k, "normal", f"row {k}") for k in range(1, STEPS + 1)],
        [],
    )
    for step in range(1, STEPS + 1):
        db.upsert_frame(FrameKey(DESKTOP, step, VIEW), f"s{step:03d}.jpg",
                        {"hw": [64, 64]}, None)
    # two chains: the PSU is redrawn at step 4, the screw has one shape for the
    # whole view -- which is the one a boundary at step 5 cuts through
    for kf in (_keyframe(PSU, (10, 10, 50, 50), 4),
               _keyframe(PSU, (12, 12, 52, 52), STEPS),
               _keyframe(SCREW, (14, 14, 22, 22), STEPS)):
        db.add_keyframe(kf)
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(PSU, "main"), (SCREW, "main")]))
    db.set_pair_override(PairOverride(DESKTOP, VIEW, 1, PSU, SCREW))
    db.set_occluder(OccluderMask(FrameKey(DESKTOP, 6, VIEW), "hand",
                                 masks.encode_rle(rect(0, 0, 8, 8))))
    db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 7, VIEW), PSU, None,
                                        "occluded_partial"))
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, STEPS, STEPS,
                        [[0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0]], None)
    db.set_pose_segment_roi(DESKTOP, VIEW, 1, [4, 4, 60, 60])
    db.set_pose_segment_bench_roi(DESKTOP, VIEW, 1, [0, 0, 32, 32])
    # one frozen frame of each flavour the re-check queue knows
    db.set_frame_flags(FrameKey(DESKTOP, 6, VIEW), review_status="verified")
    db.put_compiled(FrameKey(DESKTOP, 2, VIEW), PSU,
                    {"size": [64, 64], "counts": "0 8 4088"},
                    0.0, "visible", "in_chassis", "verified", "hash-2", verified_by="anna")
    db.clear_recheck(DESKTOP, VIEW, 2)
    db.clear_recheck(DESKTOP, VIEW, 6)
    yield db
    db.close()


def _segments(db: Db) -> list[tuple]:
    return [(r["seg"], r["start_step"], r["end_step"], r["ref_step"])
            for r in db.pose_segments(DESKTOP, VIEW)]


def _anchors(db: Db) -> dict[tuple[str, int], int]:
    """``{(instance, anchor_step): pose_segment}`` of every keyframe of the view."""
    return {(kf.instance, kf.anchor_step): kf.pose_segment
            for kf in db.keyframes(DESKTOP, VIEW)}


def _raw(db: Db, table: str) -> list[tuple]:
    return [tuple(r) for r in db.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def test_a_recut_moves_every_keyframe_to_the_segment_of_its_own_anchor(scene: Db):
    out = scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS)

    assert out["changed"] is True
    assert _segments(scene) == [(1, 1, 4, 4), (2, 5, 8, 8)]
    assert _anchors(scene) == {(PSU, 4): 1, (PSU, 8): 2, (SCREW, 8): 2}
    # nothing was duplicated: the screw's shape went with its anchor and the
    # frames before the boundary are now missing_shape on purpose
    assert len(scene.keyframes(DESKTOP, VIEW)) == 3


def test_per_segment_rows_are_copied_into_both_halves(scene: Db):
    scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS)

    for seg in (1, 2):
        assert scene.zorder(DESKTOP, VIEW, seg).order == [(PSU, "main"), (SCREW, "main")]
        assert [(p.above, p.below) for p in scene.pair_overrides(DESKTOP, VIEW, seg)] \
            == [(PSU, SCREW)]
    rows = {r["seg"]: r for r in scene.pose_segments(DESKTOP, VIEW)}
    assert rows[1]["roi"] == [4, 4, 60, 60] and rows[2]["roi"] == [4, 4, 60, 60]
    assert rows[1]["bench_roi"] == [0, 0, 32, 32]
    assert rows[2]["bench_roi"] == [0, 0, 32, 32]
    # the corners are drawn against ref_step 8, which is in the second half only
    assert rows[1]["corners"] is None
    assert rows[2]["corners"] == [[0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0]]


def test_frame_level_rows_are_not_touched_by_a_recut(scene: Db):
    """Occluders and frame overrides are keyed by the frame, not by a segment."""
    before = (_raw(scene, "occluder_mask"), _raw(scene, "frame_override"))

    scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS)

    assert (_raw(scene, "occluder_mask"), _raw(scene, "frame_override")) == before


def test_verified_rows_are_queued_and_left_byte_identical(scene: Db):
    before = _raw(scene, "compiled_mask")

    out = scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS)

    assert out["rechecked"] == [2, 6]             # both frozen frames, both affected
    assert scene.rechecks(DESKTOP, VIEW) == [2, 6]
    assert _raw(scene, "compiled_mask") == before  # not one byte of frozen truth moved
    assert scene.conflicts(DESKTOP, VIEW) == []


def test_carry_duplicates_only_the_shapes_the_boundary_cuts_through(scene: Db):
    out = scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS, carry_at=[BOUNDARY])

    assert len(out["carried"]) == 1
    carried = [kf for kf in scene.keyframes(DESKTOP, VIEW) if kf.source == "carried"]
    assert len(carried) == 1
    assert (carried[0].instance, carried[0].anchor_step, carried[0].pose_segment) \
        == (SCREW, BOUNDARY - 1, 1)
    # the copy is the same shape, not an empty stub
    original = next(kf for kf in scene.keyframes(DESKTOP, VIEW, SCREW)
                    if kf.source != "carried")
    assert carried[0].parts[0].rle == original.parts[0].rle
    # and the PSU, whose own keyframe already starts at step 5, is not touched
    assert _anchors(scene)[(PSU, 8)] == 2


def test_rejecting_the_break_merges_the_segments_back_and_drops_the_duplicate(scene: Db):
    scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS, carry_at=[BOUNDARY])

    out = scene.apply_recut(DESKTOP, VIEW, [], STEPS)

    assert _segments(scene) == [(1, 1, 8, 8)]
    assert _anchors(scene) == {(PSU, 4): 1, (PSU, 8): 1, (SCREW, 8): 1}
    assert [kf for kf in scene.keyframes(DESKTOP, VIEW) if kf.source == "carried"] == []
    assert out["uncarried"] == []                  # nothing had to be kept
    # the second segment's layer order could not survive the merge and says so
    assert [d["table"] for d in out["discarded"]] == ["zorder"]
    assert scene.zorder(DESKTOP, VIEW, 2).order == []


def test_a_carried_keyframe_the_annotator_edited_survives_the_merge(scene: Db):
    scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS, carry_at=[BOUNDARY])
    carried = next(kf for kf in scene.keyframes(DESKTOP, VIEW) if kf.source == "carried")
    carried.parts = [ShapePart("main", masks.encode_rle(rect(20, 20, 30, 30)))]
    scene.update_keyframe(carried)                 # a redraw bumps the version

    out = scene.apply_recut(DESKTOP, VIEW, [], STEPS)

    assert out["uncarried"] == [carried.id]
    assert any(kf.id == carried.id for kf in scene.keyframes(DESKTOP, VIEW))


def test_a_recut_that_changes_nothing_writes_nothing(scene: Db):
    scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS)
    scene.clear_recheck(DESKTOP, VIEW, 2)
    scene.clear_recheck(DESKTOP, VIEW, 6)
    ops = len(scene.ops(DESKTOP, VIEW))

    out = scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS)

    assert out["changed"] is False
    assert scene.rechecks(DESKTOP, VIEW) == []
    assert len(scene.ops(DESKTOP, VIEW)) == ops


def test_a_failure_in_the_middle_leaves_every_table_unchanged(scene: Db, monkeypatch):
    tables = ("pose_segment", "shape_keyframe", "shape_part", "zorder", "pair_override",
              "compiled_mask", "recheck_queue", "op_log", "frame")
    before = {name: _raw(scene, name) for name in tables}

    def explode(*_args, **_kwargs):
        """Fail after the segments, the layer rows and the keyframes are written."""
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(type(scene), "_recut_frame_overrides", explode)
    with pytest.raises(sqlite3.OperationalError):
        scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS, carry_at=[BOUNDARY])
    monkeypatch.undo()

    assert {name: _raw(scene, name) for name in tables} == before


def test_the_recut_is_logged_with_the_ranges_it_came_from(scene: Db):
    scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS, annotator="anna")

    op = scene.ops(DESKTOP, VIEW)[0]
    assert op["kind"] == "pose_recut" and op["annotator"] == "anna"
    assert op["payload"]["ranges"] == [[1, 1, 4], [2, 5, 8]]
    assert op["inverse"]["ranges"] == [[1, 1, 8]]


def test_a_view_with_no_segments_is_a_no_op(scene: Db):
    out = scene.apply_recut(DESKTOP, "oak1", [BOUNDARY], STEPS)

    assert out["changed"] is False and out["ranges"] == []
    assert scene.pose_segments(DESKTOP, "oak1") == []


def test_label_studio_drafts_in_segment_zero_come_through_untouched(scene: Db):
    draft = _keyframe("ls:PSU#1", (30, 30, 40, 40), anchor=7, seg=0)
    draft.source = "labelstudio"
    scene.add_keyframe(draft)

    scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS)

    kept = next(kf for kf in scene.keyframes(DESKTOP, VIEW) if kf.instance == "ls:PSU#1")
    assert kept.pose_segment == 0


def test_a_frame_that_names_its_own_segment_follows_the_recut(scene: Db):
    scene.set_frame_flags(FrameKey(DESKTOP, 7, VIEW), pose_segment=1)

    scene.apply_recut(DESKTOP, VIEW, [BOUNDARY], STEPS)

    assert scene.get_frame(FrameKey(DESKTOP, 7, VIEW))["pose_segment"] == 2
    assert scene.pose_segment_for(FrameKey(DESKTOP, 7, VIEW))["seg"] == 2


# --------------------------------------------------------------------------- #
# the break table itself
# --------------------------------------------------------------------------- #
def test_a_break_is_never_overwritten_by_a_second_import(scene: Db):
    scene.add_pose_break(DESKTOP, VIEW, 5, status="proposed", kind="camera",
                         magnitude_px=190.45, source="audit:events.csv", note="looked at")
    scene.set_pose_break_status(DESKTOP, VIEW, 5, "rejected")

    scene.add_pose_break(DESKTOP, VIEW, 5, status="proposed", kind="camera",
                         magnitude_px=190.45, source="audit:events.csv")

    assert scene.pose_break(DESKTOP, VIEW, 5)["status"] == "rejected"
    assert scene.pose_breaks(DESKTOP, VIEW, status="accepted") == []


def test_an_unknown_status_is_refused(scene: Db):
    with pytest.raises(ValueError):
        scene.add_pose_break(DESKTOP, VIEW, 5, status="maybe", source="manual:anna")
    with pytest.raises(ValueError):
        scene.set_pose_break_status(DESKTOP, VIEW, 5, "maybe")
    assert scene.set_pose_break_status(DESKTOP, VIEW, 99, "accepted") is False
