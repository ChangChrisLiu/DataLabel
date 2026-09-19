"""Offscreen tests for :class:`tda.ui.session.AnnotationSession` (spec 4.2-4.4).

Navigation, images, the four edit scopes and the review queues.  The task card
lives in ``test_session_card.py``, undo/redo in ``test_session_undo.py``, the
scope suggestion and the main-window interface in ``test_session_scope.py``,
the sweeper in ``test_session_sweep.py`` and the lazy-truth behaviour and time
budgets in ``test_session_perf.py``.  The scene all of them drive is
``tests/session_scene.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tda.core import masks
from tda.core.model import FrameKey
from tda.ui import session_api as api
from tda.ui.commands import edit_editing_mask_op
from tda.ui.session import AnnotationSession
from tda.ui.session_api import SessionRefusal
from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
    HW,
    LAST_STEP,
    VIEW,
    cell,
    draw,
    make_session,
    rect,
    seed_shapes,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = make_session(tmp_path)
    yield made
    made.close(force=True)  # the worker thread must never outlive the object it signals


# --------------------------------------------------------------------------- #
# the protocol
# --------------------------------------------------------------------------- #
def test_session_satisfies_the_panel_protocol(session):
    assert isinstance(session, api.SessionLike)


# --------------------------------------------------------------------------- #
# open / navigation
# --------------------------------------------------------------------------- #
def test_open_starts_at_the_last_step(session):
    assert session.steps() == list(range(1, LAST_STEP + 1))
    assert session.current() == FrameKey(DESKTOP, LAST_STEP, VIEW)


def test_open_skips_missing_frames_but_keeps_the_logical_steps(qapp, tmp_path):
    session = make_session(tmp_path, missing=(LAST_STEP,))
    assert session.steps() == list(range(1, LAST_STEP + 1))  # the logical step stays
    assert session.current().step == LAST_STEP - 1  # ... but it is not annotated
    assert session.frame_status(LAST_STEP) == api.STATUS_MISSING


def test_prev_and_next_walk_the_available_steps(qapp, tmp_path):
    session = make_session(tmp_path, missing=(12,))
    session.goto(13)
    session.prev()
    assert session.current().step == 11  # 12 has no image
    session.next()
    assert session.current().step == 13
    session.goto(1)
    session.prev()
    assert session.current().step == 1  # nothing before the first step


def test_goto_emits_the_frame_changed_signal(session):
    seen = []
    session.sigFrameChanged.connect(seen.append)
    session.goto(10)
    assert seen == [FrameKey(DESKTOP, 10, VIEW)]


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def test_image_is_rgb_from_the_cache_and_is_reused(session):
    session.goto(10)
    img = session.image()
    assert img.shape == (*HW, 3)
    # cv2 wrote BGR with a constant colour, so R and B differ only by the fill
    assert img[0, 0, 0] == 30 and img[0, 0, 2] == 30
    assert session.image() is img  # the LRU hands back the very same array


def test_flash_compare_shows_the_frame_the_annotator_came_from(session):
    session.goto(10)  # in reverse order that is k+1, which is already annotated
    assert np.array_equal(session.flash_compare(), session.image_at(11))


def test_image_cache_is_bounded(session):
    for step in range(1, LAST_STEP + 1):
        session.goto(step)
        session.image()
    assert len(session.image_cache) <= 8


def test_thumb_path_points_at_the_cached_image(session):
    path = session.thumb_path(7)
    assert path is not None and Path(path).exists()


# --------------------------------------------------------------------------- #
# commit_edit (spec 3.3 / 4.3)
# --------------------------------------------------------------------------- #
def test_keyframe_commit_anchors_at_the_last_step_that_needs_geometry(session):
    session.goto(10)
    result = draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)

    kfs = session.db.keyframes(DESKTOP, VIEW, COOLER)
    assert len(kfs) == 1
    # the cooler is removed at step 13, so its chassis chain ends at step 12
    assert kfs[0].anchor_step == 12
    assert kfs[0].placement == "in_chassis"
    assert result["affected"] == list(range(1, 13))
    # the truth rows of the other frames are a cache: they are written when the
    # frame is visited, not by the commit (see test_session_perf.py)
    for step in result["affected"]:
        session.goto(step)
        assert COOLER in session.db.compiled(FrameKey(DESKTOP, step, VIEW))


def test_keyframe_commit_updates_the_selected_keyframe_and_bumps_the_version(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(5), api.SCOPE_KEYFRAME)

    kfs = session.db.keyframes(DESKTOP, VIEW, COOLER)
    assert len(kfs) == 1  # still one keyframe: the existing one was re-traced
    assert kfs[0].version == 2
    assert np.array_equal(masks.decode_rle(kfs[0].parts[0].rle), cell(5))


def test_a_new_instance_goes_on_top_of_the_zorder(session):
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    order = session.db.zorder(DESKTOP, VIEW, 1).order
    assert order[-1] == (COOLER, "main")  # the order runs bottom -> top


def test_split_in_reverse_anchors_the_new_keyframe_at_the_current_step(session):
    session.goto(12)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    session.goto(8)
    draw(session, COOLER, cell(3), api.SCOPE_SPLIT)

    anchors = sorted(kf.anchor_step for kf in session.db.keyframes(DESKTOP, VIEW, COOLER))
    assert anchors == [8, 12]
    at_8 = session.compiled().instances[COOLER].visible
    session.goto(9)
    at_9 = session.compiled().instances[COOLER].visible
    assert np.array_equal(at_8, cell(3))
    assert not np.array_equal(at_8, at_9)  # the split really did cut the chain


def test_split_forward_moves_the_old_anchor_back(session):
    session.goto(12)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    session.goto(8)
    session.begin_edit(COOLER)
    session.set_editing_mask(cell(3))
    session.commit_edit(api.SCOPE_SPLIT, direction="forward")

    anchors = sorted(kf.anchor_step for kf in session.db.keyframes(DESKTOP, VIEW, COOLER))
    assert anchors == [7, 12]  # the old shape now ends at k-1, the new one inherits


def test_frame_override_touches_only_this_frame(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    result = draw(session, COOLER, cell(6), api.SCOPE_FRAME_OVERRIDE)

    assert result["affected"] == [10]
    assert COOLER in session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW))
    assert session.db.frame_overrides(FrameKey(DESKTOP, 9, VIEW)) == {}
    assert np.array_equal(session.compiled().instances[COOLER].visible, cell(6))
    session.goto(9)
    assert np.array_equal(session.compiled().instances[COOLER].visible, cell(0))


def test_commit_edit_carries_the_windows_own_note_into_the_op_log(session):
    """``extra`` is what the window records about *why* it committed.

    "The annotator was warned the shape is implausibly large and went ahead" is
    not something the session can derive, and it belongs in the audit trail of
    that commit rather than in a log nobody joins back to it.
    """
    session.goto(10)
    session.begin_edit(COOLER)
    session.set_editing_mask(cell(0))
    session.commit_edit(api.SCOPE_KEYFRAME, extra={"area_warning_overridden": True})

    op = session.db.ops(DESKTOP, VIEW)[0]
    assert op["kind"] == "commit_keyframe"
    assert op["payload"]["area_warning_overridden"] is True
    assert op["payload"]["instance"] == COOLER  # and everything else is still there


def test_commit_edit_refuses_a_note_that_cannot_be_logged(session):
    session.goto(10)
    session.begin_edit(COOLER)
    session.set_editing_mask(cell(0))

    with pytest.raises(ValueError, match="JSON"):
        session.commit_edit(api.SCOPE_KEYFRAME, extra={"when": object()})
    # NaN and the infinities are Python's JSON extension, not JSON: a reader
    # anywhere else refuses the file the op log would have become
    with pytest.raises(ValueError, match="JSON"):
        session.commit_edit(api.SCOPE_KEYFRAME, extra={"ratio": float("nan")})

    assert session.db.ops(DESKTOP, VIEW) == []  # nothing was written
    assert session.editing_instance == COOLER   # and the layer is still there


def test_commit_logs_a_source_level_operation(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    ops = session.db.ops(DESKTOP, VIEW)
    assert ops[0]["kind"] == "commit_keyframe"
    assert ops[0]["annotator"] == "tester"
    assert ops[0]["payload"]["instance"] == COOLER


def test_commit_box_writes_a_bench_rectangle(session):
    # the scene's view sees no staging area by default, and a bench box only
    # reaches the frames the compiler would put the part on the bench in
    session.db.set_pose_segment(DESKTOP, VIEW, 1, 1, LAST_STEP, LAST_STEP, None, None)
    session.db.set_pose_segment_bench_roi(DESKTOP, VIEW, 1, (0, 0, 32, 32))
    session.goto(14)  # the cooler is on the bench from step 13 on
    result = session.commit_box(COOLER, (2.0, 2.0, 12.0, 12.0))
    kfs = [kf for kf in session.db.keyframes(DESKTOP, VIEW, COOLER) if kf.geom_type == "box"]
    assert len(kfs) == 1
    assert kfs[0].placement == "on_bench"
    assert kfs[0].anchor_step == LAST_STEP
    assert result["affected"] == [13, 14]


def test_a_bench_box_on_a_view_without_a_staging_area_is_refused(session):
    """The scanner cannot see the bench, so there is nothing to box there.

    It used to be written and then reported as "affects 1 frame" -- a keyframe
    the compiler will never select, on a frame the part is not in.
    """
    session.goto(14)
    with pytest.raises(SessionRefusal, match="堆放区"):
        session.commit_box(COOLER, (2.0, 2.0, 12.0, 12.0))
    assert not [kf for kf in session.db.keyframes(DESKTOP, VIEW, COOLER)
                if kf.geom_type == "box"]


# --------------------------------------------------------------------------- #
# instance list, visibility, z-order
# --------------------------------------------------------------------------- #
def test_instance_rows_are_top_first_with_the_fields_the_panel_reads(session):
    session.goto(10)
    seed_shapes(session, 10)
    rows = session.instance_rows()
    assert rows
    assert set(rows[0]) == {"key", "cls", "state", "placement", "visibility", "z", "hidden"}
    assert [r["z"] for r in rows] == sorted((r["z"] for r in rows), reverse=True)


def test_set_visibility_writes_a_frame_override(session):
    session.goto(10)
    seed_shapes(session, 10)
    session.set_visibility(COOLER, "motion_blur")
    override = session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert override.visibility == "motion_blur"
    row = next(r for r in session.instance_rows() if r["key"] == COOLER)
    assert row["visibility"] == "motion_blur"


def test_set_visibility_keeps_an_existing_mask_override(session):
    session.goto(10)
    seed_shapes(session, 10)
    draw(session, COOLER, cell(20), api.SCOPE_FRAME_OVERRIDE)
    session.set_visibility(COOLER, "motion_blur")
    override = session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert override.visibility == "motion_blur"
    assert np.array_equal(masks.decode_rle(override.visible_rle), cell(20))


def test_set_hidden_is_a_view_setting_and_is_not_persisted(session):
    session.goto(10)
    seed_shapes(session, 10)
    session.set_hidden(COOLER, True)
    assert next(r for r in session.instance_rows() if r["key"] == COOLER)["hidden"] is True
    assert session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW)) == {}
    assert session.db.ops(DESKTOP, VIEW) == []


def test_set_zorder_move_puts_one_instance_above_another(session):
    session.goto(10)
    seed_shapes(session, 10)
    rows = session.instance_rows()
    bottom, above_of = rows[-1]["key"], rows[-2]["key"]
    session.set_zorder_move(bottom, above_of)
    order = [inst for inst, _ in session.db.zorder(DESKTOP, VIEW, 1).order]
    assert order.index(bottom) == order.index(above_of) + 1


# --------------------------------------------------------------------------- #
# confirming a frame (spec 4.2 step 5)
# --------------------------------------------------------------------------- #
def test_confirm_frame_refuses_a_frame_with_a_missing_shape(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    problems: list[list[str]] = []
    session.sigProblems.connect(problems.append)

    assert session.confirm_frame() is False
    assert problems and f"missing_shape:{CHASSIS}" in problems[0]
    assert session.current().step == 12  # it did not move on
    assert session.frame_status(12) != api.STATUS_VERIFIED


def test_confirm_frame_accepts_a_complete_frame_and_steps_back(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    assert session.confirm_frame() is False
    draw(session, CHASSIS, cell(60), api.SCOPE_KEYFRAME)

    assert session.confirm_frame() is True
    assert session.frame_status(12) == api.STATUS_VERIFIED
    assert session.current().step == 11  # reverse order: the next frame is k-1


# --------------------------------------------------------------------------- #
# frame status and queues (spec 3.4 / 4.4)
# --------------------------------------------------------------------------- #
def test_frame_status_reports_unlabeled_then_auto(session):
    assert session.frame_status(10) == api.STATUS_UNLABELED
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert session.frame_status(10) == api.STATUS_AUTO


def test_a_conflicting_edit_on_a_verified_frame_is_queued(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    draw(session, CHASSIS, cell(60), api.SCOPE_KEYFRAME)
    assert session.confirm_frame() is True
    assert session.frame_status(12) == api.STATUS_VERIFIED

    session.goto(12)
    result = draw(session, CHASSIS, rect(30, 30, 60, 60), api.SCOPE_KEYFRAME)

    assert result["conflicts"] >= 1
    assert session.frame_status(12) == api.STATUS_CONFLICT
    queued = session.queues()[api.QUEUE_CONFLICTS]
    assert [entry["step"] for entry in queued] == [12]
    assert queued[0]["instance"] == CHASSIS
    assert queued[0]["sym_diff_px"] > 0


def test_resolve_conflict_closes_the_queue_entry(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    draw(session, CHASSIS, cell(60), api.SCOPE_KEYFRAME)
    session.confirm_frame()
    session.goto(12)
    draw(session, CHASSIS, rect(30, 30, 60, 60), api.SCOPE_KEYFRAME)

    cid = session.queues()[api.QUEUE_CONFLICTS][0]["id"]
    session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW)
    assert session.queues()[api.QUEUE_CONFLICTS] == []
    assert session.frame_status(12) == api.STATUS_VERIFIED


def test_queues_report_missing_shapes_and_needs_review(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    missing = session.queues()[api.QUEUE_MISSING_SHAPE]
    assert {entry["instance"] for entry in missing if entry["step"] == 10} >= {CHASSIS}
    assert session.queues()[api.QUEUE_UNEXPLAINED] == []

    session.truth.demote_frame(FrameKey(DESKTOP, 9, VIEW), "test")
    assert [e["step"] for e in session.queues()[api.QUEUE_NEEDS_REVIEW]] == [9]
    assert session.frame_status(9) == api.STATUS_NEEDS_REVIEW


# --------------------------------------------------------------------------- #
# undo (spec 4.6)
# --------------------------------------------------------------------------- #
def test_undo_of_a_commit_restores_the_previous_shape(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(5), api.SCOPE_KEYFRAME)

    assert session.undo() is True
    kf = session.db.keyframes(DESKTOP, VIEW, COOLER)[0]
    assert np.array_equal(masks.decode_rle(kf.parts[0].rle), cell(0))
    row = session.db.compiled(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert np.array_equal(masks.decode_rle(row["visible_rle"]), cell(0))

    assert session.redo() is True
    kf = session.db.keyframes(DESKTOP, VIEW, COOLER)[0]
    assert np.array_equal(masks.decode_rle(kf.parts[0].rle), cell(5))


def test_undo_of_the_first_commit_takes_the_shape_out_of_every_frame(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert session.undo() is True

    assert session.db.keyframes(DESKTOP, VIEW, COOLER) == []
    # the instance still needs geometry here, so the truth row stays -- but with
    # nothing in it, and the frame is back to reporting a missing shape
    row = session.db.compiled(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert row["visible_rle"] is None
    assert f"missing_shape:{COOLER}" in session.compiled().problems


def test_undo_restores_a_frame_override_and_a_zorder(session):
    session.goto(10)
    seed_shapes(session, 10)
    before = list(session.db.zorder(DESKTOP, VIEW, 1).order)
    rows = session.instance_rows()
    session.set_zorder_move(rows[-1]["key"], rows[-2]["key"])
    assert session.undo() is True
    assert session.db.zorder(DESKTOP, VIEW, 1).order == before

    draw(session, COOLER, cell(20), api.SCOPE_FRAME_OVERRIDE)
    assert session.undo() is True
    assert session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW)) == {}


def test_undo_of_a_brush_stroke_restores_the_editing_layer(session):
    session.goto(10)
    session.begin_edit(COOLER)
    before = np.zeros(HW, dtype=bool)
    session.set_editing_mask(before)
    session.undo_stack.push(edit_editing_mask_op(COOLER, before, cell(2)), apply=True)

    assert np.array_equal(session.editing_mask(), cell(2))
    assert session.undo() is True
    assert not session.editing_mask().any()  # the stroke never touched the database
    assert session.db.keyframes(DESKTOP, VIEW, COOLER) == []


def test_undo_restores_an_occluder(session):
    session.goto(10)
    seed_shapes(session, 10)
    session.commit_occluder(rect(0, 0, 20, 20), "hand")
    assert len(session.db.occluders(FrameKey(DESKTOP, 10, VIEW))) == 1
    assert session.undo() is True
    assert session.db.occluders(FrameKey(DESKTOP, 10, VIEW)) == []


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def test_save_and_close_clear_the_dirty_flag(session):
    dirty: list[bool] = []
    session.sigDirty.connect(dirty.append)
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert dirty[-1] is True
    session.save()
    assert dirty[-1] is False
    session.close()
    assert session.steps() == []
