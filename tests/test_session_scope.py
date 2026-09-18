"""Edit scope, layer order, conflict outcomes, signals and the window interface.

Everything here is about the session telling the truth to whoever is going to
act on it: which scope an edit *means* (spec 4.3), which order the compiler
actually paints in, what came of a conflict resolution, and the additions the
main window needs in order to show "affects N frames" before anything is
written.
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
from tda.ui.session import AnnotationSession
from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
    LAST_STEP,
    VIEW,
    cell,
    draw,
    make_session,
    rect,
    seed_shapes,
)

BIG = rect(0, 0, 40, 40)
SMALL = rect(44, 44, 56, 56)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = make_session(tmp_path)
    yield made
    made.close()  # the worker thread must never outlive the object it signals


@pytest.fixture
def two_shapes(session) -> AnnotationSession:
    """Chassis over the top-left quarter and *above* the cooler, which is clear of it.

    Drawn cooler-first, because a new instance goes on top of the layer order:
    so the chassis is the one painting over, which is what makes "add pixels
    inside it" a question about layering at all.
    """
    session.goto(10)
    draw(session, COOLER, SMALL, api.SCOPE_KEYFRAME)
    draw(session, CHASSIS, BIG, api.SCOPE_KEYFRAME)
    return session


# --------------------------------------------------------------------------- #
# scope suggestion (spec 4.3 默认触发)
# --------------------------------------------------------------------------- #
def test_suggest_scope_is_keyframe_when_nothing_was_edited(two_shapes):
    two_shapes.begin_edit(COOLER)
    assert two_shapes.suggest_scope() == api.SCOPE_KEYFRAME


def test_adding_pixels_inside_another_shape_suggests_going_above_it(two_shapes):
    two_shapes.begin_edit(COOLER)
    two_shapes.set_editing_mask(SMALL | rect(4, 4, 16, 16))
    assert two_shapes.suggest_scope() == f"zorder:above:{CHASSIS}"


def test_adding_pixels_in_free_space_stays_a_keyframe_edit(two_shapes):
    two_shapes.begin_edit(COOLER)
    two_shapes.set_editing_mask(SMALL | rect(44, 20, 56, 32))
    assert two_shapes.suggest_scope() == api.SCOPE_KEYFRAME


def test_erasing_the_overlap_suggests_putting_the_other_one_above(session):
    session.goto(10)
    draw(session, CHASSIS, BIG, api.SCOPE_KEYFRAME)
    draw(session, COOLER, rect(20, 20, 50, 50), api.SCOPE_KEYFRAME)  # overlaps, on top
    session.begin_edit(COOLER)
    session.set_editing_mask(rect(20, 20, 50, 50) & ~BIG)  # rub out exactly the overlap
    assert session.suggest_scope() == f"zorder:below:{CHASSIS}"


def test_commit_with_nothing_edited_is_a_no_op(two_shapes):
    two_shapes.begin_edit(COOLER)
    versions = [kf.version for kf in two_shapes.db.keyframes(DESKTOP, VIEW, COOLER)]
    ops_before = len(two_shapes.db.ops(DESKTOP, VIEW))
    history = len(two_shapes.undo_stack)

    result = two_shapes.commit_edit(api.SCOPE_KEYFRAME)
    assert result["changed"] is False
    assert [kf.version for kf in two_shapes.db.keyframes(DESKTOP, VIEW, COOLER)] == versions
    assert len(two_shapes.db.ops(DESKTOP, VIEW)) == ops_before
    assert len(two_shapes.undo_stack) == history  # nothing to take back


def test_both_zorder_directions_write_the_pair_the_right_way_round(session):
    session.goto(10)
    draw(session, CHASSIS, BIG, api.SCOPE_KEYFRAME)
    draw(session, COOLER, SMALL, api.SCOPE_KEYFRAME)

    session.begin_edit(COOLER)
    session.set_editing_mask(SMALL | rect(4, 4, 16, 16))
    session.commit_edit(f"zorder:above:{CHASSIS}")
    assert [(p.above, p.below) for p in session.db.pair_overrides(DESKTOP, VIEW, 1)] == [
        (COOLER, CHASSIS)
    ]

    session.undo()
    session.begin_edit(COOLER)
    session.set_editing_mask(SMALL)
    session.commit_edit(f"zorder:below:{CHASSIS}")
    assert [(p.above, p.below) for p in session.db.pair_overrides(DESKTOP, VIEW, 1)] == [
        (CHASSIS, COOLER)
    ]


def test_a_zorder_scope_naming_an_absent_instance_is_refused(two_shapes):
    two_shapes.begin_edit(COOLER)
    two_shapes.set_editing_mask(SMALL | rect(4, 4, 16, 16))
    with pytest.raises(ValueError, match="no such instance"):
        two_shapes.commit_edit("zorder:above:not.an.instance")


def test_set_zorder_move_refuses_an_unknown_neighbour(session):
    session.goto(10)
    seed_shapes(session, 10)
    before = list(session.db.zorder(DESKTOP, VIEW, 1).order)
    with pytest.raises(ValueError, match="no such instance"):
        session.set_zorder_move(COOLER, "not.an.instance")
    assert session.db.zorder(DESKTOP, VIEW, 1).order == before


# --------------------------------------------------------------------------- #
# splitting a keyframe that already ends here (spec 3.3)
# --------------------------------------------------------------------------- #
def test_split_on_the_keyframes_own_anchor_retraces_it_instead(session):
    session.goto(12)  # the cooler's chassis chain ends at 12
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(1), api.SCOPE_KEYFRAME)  # version 2

    session.begin_edit(COOLER)
    session.set_editing_mask(cell(2))
    result = session.commit_edit(api.SCOPE_SPLIT)

    kfs = session.db.keyframes(DESKTOP, VIEW, COOLER)
    assert len(kfs) == 1  # a split that cannot split is a re-trace
    assert result["scope"] == api.SCOPE_KEYFRAME
    assert np.array_equal(masks.decode_rle(kfs[0].parts[0].rle), cell(2))
    row = session.db.compiled(FrameKey(DESKTOP, 12, VIEW))[COOLER]
    assert np.array_equal(masks.decode_rle(row["visible_rle"]), cell(2))


def test_a_real_split_outranks_the_keyframe_it_was_cut_from(session):
    session.goto(12)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(1), api.SCOPE_KEYFRAME)  # version 2
    session.goto(8)
    draw(session, COOLER, cell(3), api.SCOPE_SPLIT)

    new = next(kf for kf in session.db.keyframes(DESKTOP, VIEW, COOLER) if kf.anchor_step == 8)
    assert new.version > 2  # or select_keyframe would keep preferring the old one
    row = session.db.compiled(FrameKey(DESKTOP, 8, VIEW))[COOLER]
    assert np.array_equal(masks.decode_rle(row["visible_rle"]), cell(3))


# --------------------------------------------------------------------------- #
# the layer order the compiler actually paints (spec 3.3 step 5)
# --------------------------------------------------------------------------- #
def test_instance_rows_follow_the_compiler_not_the_stored_order(session):
    session.goto(10)
    draw(session, CHASSIS, BIG, api.SCOPE_KEYFRAME)
    draw(session, COOLER, rect(20, 20, 50, 50), api.SCOPE_KEYFRAME)
    # the cooler was drawn last, so it is on top; a pair override reverses that
    session.begin_edit(COOLER)
    session.set_editing_mask(rect(40, 40, 50, 50))
    session.commit_edit(f"zorder:below:{CHASSIS}")

    painted = session.compiled().painted["in_chassis"]
    assert painted.index(CHASSIS) > painted.index(COOLER)  # bottom-up: chassis on top
    rows = [row["key"] for row in session.instance_rows()]
    assert rows.index(CHASSIS) < rows.index(COOLER)  # rows are top-first
    # ... and the pixels agree: the chassis now claims the overlap
    row = session.db.compiled(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    visible = masks.decode_rle(row["visible_rle"])
    assert not (visible & BIG).any()


def test_overlay_layers_hands_the_window_masks_and_the_paint_order(session):
    session.goto(10)
    seed_shapes(session, 10)
    layers, order = session.overlay_layers()
    assert order and set(order) <= set(layers)
    assert order == session.compiled().painted["in_chassis"] + session.compiled().painted.get(
        "on_bench", []
    )
    for mask in layers.values():
        assert mask.dtype == bool

    hidden = order[0]
    session.set_hidden(hidden, True)
    layers, order = session.overlay_layers()
    assert hidden not in layers and hidden not in order


# --------------------------------------------------------------------------- #
# conflict resolution outcomes (spec 3.4)
# --------------------------------------------------------------------------- #
def _conflicting_frame(session) -> int:
    """Verify step 12, then edit the chassis so the frozen row disagrees."""
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    draw(session, CHASSIS, cell(60), api.SCOPE_KEYFRAME)
    assert session.confirm_frame() is True
    session.goto(12)
    draw(session, CHASSIS, rect(30, 30, 60, 60), api.SCOPE_KEYFRAME)
    return session.queues()[api.QUEUE_CONFLICTS][0]["id"]


def test_resolve_conflict_reports_resolved(session):
    cid = _conflicting_frame(session)
    assert session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW) == "resolved"
    assert session.queues()[api.QUEUE_CONFLICTS] == []


def test_resolve_conflict_reports_resolved_for_keep_old(session):
    cid = _conflicting_frame(session)
    assert session.resolve_conflict(cid, api.RESOLVE_KEEP_OLD) == "resolved"
    assert session.queues()[api.QUEUE_CONFLICTS] == []
    assert CHASSIS in session.db.frame_overrides(FrameKey(DESKTOP, 12, VIEW))


def test_resolve_conflict_reports_a_refusal_without_raising(session):
    cid = _conflicting_frame(session)
    session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW)
    problems: list[list[str]] = []
    session.sigProblems.connect(problems.append)

    assert session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW) == "refused"
    assert problems and problems[-1]
    assert session.resolve_conflict(9999, api.RESOLVE_ACCEPT_NEW) == "refused"


def test_resolve_conflict_reports_a_superseded_conflict(session):
    cid = _conflicting_frame(session)
    # the inputs move on before anybody gets to the queued conflict
    session.goto(12)
    draw(session, CHASSIS, rect(2, 34, 30, 62), api.SCOPE_KEYFRAME)
    problems: list[list[str]] = []
    session.sigProblems.connect(problems.append)

    assert session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW) == "superseded"
    assert problems and any("superseded" in text for text in problems[-1])
    # the current disagreement took its place in the queue
    queued = session.queues()[api.QUEUE_CONFLICTS]
    assert [entry["id"] for entry in queued] != [cid] and queued


# --------------------------------------------------------------------------- #
# signals (the panels have no other way to know)
# --------------------------------------------------------------------------- #
def test_problems_are_announced_after_a_commit_and_on_goto(session):
    problems: list[list[str]] = []
    session.sigProblems.connect(problems.append)
    session.goto(10)
    assert problems and all(isinstance(text, str) for text in problems[-1])

    seen = len(problems)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert len(problems) > seen
    assert f"missing_shape:{COOLER}" not in problems[-1]


def test_problems_are_announced_after_undo_and_redo(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    problems: list[list[str]] = []
    session.sigProblems.connect(problems.append)

    session.undo()
    assert problems and f"missing_shape:{COOLER}" in problems[-1]
    session.redo()
    assert f"missing_shape:{COOLER}" not in problems[-1]


def test_confirm_frame_announces_the_frame_change_once(session):
    session.goto(12)
    seed_shapes(session, 12)
    seen: list = []
    session.sigFrameChanged.connect(seen.append)

    assert session.confirm_frame() is True
    assert len(seen) == 1
    assert seen[0].step == 11


# --------------------------------------------------------------------------- #
# what the main window needs (spec 4.3 "影响 N 帧")
# --------------------------------------------------------------------------- #
def test_preview_reports_the_reach_of_an_edit_before_it_is_written(session):
    session.goto(10)
    session.begin_edit(COOLER)
    session.set_editing_mask(cell(0))
    preview = session.preview(api.SCOPE_KEYFRAME)

    assert preview["steps"] == list(range(1, 13))  # the chassis chain ends at 12
    assert preview["verified_steps"] == []
    assert session.db.keyframes(DESKTOP, VIEW, COOLER) == []  # nothing was written

    assert session.preview(api.SCOPE_FRAME_OVERRIDE)["steps"] == [10]
    assert session.preview(api.SCOPE_SPLIT)["steps"] == list(range(1, 11))


def test_preview_counts_the_verified_frames_an_edit_would_disturb(session):
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True

    session.goto(10)
    session.begin_edit(COOLER)
    session.set_editing_mask(cell(0))
    assert 12 in session.preview(api.SCOPE_KEYFRAME)["verified_steps"]


def test_is_open_and_sig_closed_let_the_window_detach(session):
    assert session.is_open is True
    closed: list = []
    session.sigClosed.connect(lambda: closed.append(True))
    session.close()
    assert session.is_open is False
    assert closed == [True]
    with pytest.raises(RuntimeError):
        session.current()


def test_a_closed_session_answers_instead_of_raising(session):
    session.close()
    # the panels are detached, but a queued click may still arrive
    assert session.resolve_conflict(1, api.RESOLVE_ACCEPT_NEW) == "refused"
    assert session.queues()[api.QUEUE_CONFLICTS] == []
    assert session.frame_status(3) == api.STATUS_UNLABELED
    assert session.review.desktop is None  # nothing is queried for a shut view


def test_a_shape_of_the_wrong_size_does_not_count_as_drawn(session):
    """The compiler reports `shape_size_mismatch`; coverage must agree it is missing."""
    from tda.core import masks as _masks
    from tda.core.model import ShapeKeyframe, ShapePart

    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=CHASSIS, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", _masks.encode_rle(np.zeros((32, 32), dtype=bool)))],
    ))
    session.review.invalidate()
    found = session.review.drawn()[10]
    assert CHASSIS in found.missing
    assert {e["instance"] for e in session.queues()[api.QUEUE_MISSING_SHAPE]
            if e["step"] == 10} >= {CHASSIS}


def test_set_unexplained_feeds_the_fourth_queue(session):
    assert session.queues()[api.QUEUE_UNEXPLAINED] == []
    session.set_unexplained(11, [(1.0, 2.0, 3.0, 4.0)])
    entries = session.queues()[api.QUEUE_UNEXPLAINED]
    assert entries == [{"step": 11, "boxes": [(1.0, 2.0, 3.0, 4.0)]}]
    session.set_unexplained(11, [])
    assert session.queues()[api.QUEUE_UNEXPLAINED] == []


def test_thumb_path_prefers_the_offline_thumbnail(session, tmp_path):
    full = session.thumb_path(10)
    assert full is not None and "thumbs" not in full

    thumb = Path(session.cache_dir) / "thumbs" / VIEW / f"D{DESKTOP:02d}" / "s010.jpg"
    thumb.parent.mkdir(parents=True, exist_ok=True)
    thumb.write_bytes(b"not really a jpeg, but it exists")
    assert Path(session.thumb_path(10)) == thumb


# --------------------------------------------------------------------------- #
# refusals (findings 16 and 17)
# --------------------------------------------------------------------------- #
def test_editing_a_frame_without_an_image_is_refused(qapp, tmp_path):
    session = make_session(tmp_path, missing=(LAST_STEP,))
    session.goto(LAST_STEP)
    with pytest.raises(ValueError, match="no image"):
        session.begin_edit(CHASSIS)
    with pytest.raises(ValueError, match="no image"):
        session.commit_box(COOLER, (1.0, 1.0, 5.0, 5.0))


def test_a_mask_commit_on_a_bench_part_is_refused(session):
    session.goto(14)  # the cooler is on the bench from step 13
    with pytest.raises(ValueError, match="bench box"):
        draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert session.db.keyframes(DESKTOP, VIEW, COOLER) == []


# --------------------------------------------------------------------------- #
# the image cache is bounded by memory, not by frame count
# --------------------------------------------------------------------------- #
def test_the_image_cache_respects_a_byte_budget(qapp, tmp_path, monkeypatch):
    session = make_session(tmp_path)
    monkeypatch.setattr(session, "image_budget_bytes", 64 * 64 * 3 * 2)
    for step in range(1, 8):
        session.goto(step)
        session.image()
    assert 1 <= len(session.image_cache) <= 2
