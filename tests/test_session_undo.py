"""Undo/redo of every op kind the session logs (spec 4.6).

The property under test throughout is **idempotence of the history**: replaying
undo and redo any number of times must leave the database in exactly the state
the corresponding point of the history describes -- not an equivalent-looking
one.  A create that is undone and redone has to come back as the *same row*, or
a later op that names it falls through to an insert and leaves a duplicate
behind that nothing points at and everything exports.
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
    HW,
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


def keyframes(session, instance=COOLER):
    return session.db.keyframes(DESKTOP, VIEW, instance)


# --------------------------------------------------------------------------- #
# the row a create/undo/redo cycle comes back as
# --------------------------------------------------------------------------- #
def test_undo_redo_of_two_commits_never_duplicates_the_keyframe(session):
    """commit, re-trace, undo, undo, redo, redo -- one row at every point."""
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    first_id = keyframes(session)[0].id
    draw(session, COOLER, cell(5), api.SCOPE_KEYFRAME)
    assert [kf.id for kf in keyframes(session)] == [first_id]

    assert session.undo() is True  # back to cell(0)
    assert len(keyframes(session)) == 1
    assert session.undo() is True  # back to no shape at all
    assert keyframes(session) == []

    assert session.redo() is True
    assert [kf.id for kf in keyframes(session)] == [first_id]
    assert np.array_equal(masks.decode_rle(keyframes(session)[0].parts[0].rle), cell(0))

    assert session.redo() is True
    assert [kf.id for kf in keyframes(session)] == [first_id]
    assert np.array_equal(masks.decode_rle(keyframes(session)[0].parts[0].rle), cell(5))

    assert session.undo() is True and session.undo() is True
    assert keyframes(session) == []  # no ghost row survives


def test_undo_redo_of_a_split_never_duplicates_the_keyframes(session):
    session.goto(12)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    session.goto(8)
    draw(session, COOLER, cell(3), api.SCOPE_SPLIT)
    ids = sorted(kf.id for kf in keyframes(session))
    assert len(ids) == 2

    for _ in range(3):
        assert session.undo() is True
        assert len(keyframes(session)) == 1
        assert session.redo() is True
        assert sorted(kf.id for kf in keyframes(session)) == ids

    assert session.undo() is True and session.undo() is True
    assert keyframes(session) == []


def test_undo_redo_of_a_frame_override_is_idempotent(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    key = FrameKey(DESKTOP, 10, VIEW)
    draw(session, COOLER, cell(6), api.SCOPE_FRAME_OVERRIDE)

    for _ in range(3):
        assert session.undo() is True
        assert session.db.frame_overrides(key) == {}
        assert session.redo() is True
        stored = session.db.frame_overrides(key)[COOLER]
        assert np.array_equal(masks.decode_rle(stored.visible_rle), cell(6))


def test_undo_redo_of_a_pair_override_is_idempotent(session):
    session.goto(10)
    draw(session, CHASSIS, rect(0, 0, 40, 40), api.SCOPE_KEYFRAME)
    draw(session, COOLER, rect(50, 50, 60, 60), api.SCOPE_KEYFRAME)
    session.begin_edit(COOLER)
    session.set_editing_mask(rect(50, 50, 60, 60) | rect(4, 4, 12, 12))
    session.commit_edit(f"zorder:above:{CHASSIS}")

    def pairs():
        return [(p.above, p.below) for p in session.db.pair_overrides(DESKTOP, VIEW, 1)]

    assert pairs() == [(COOLER, CHASSIS)]
    for _ in range(3):
        assert session.undo() is True
        assert pairs() == []
        assert session.redo() is True
        assert pairs() == [(COOLER, CHASSIS)]


def test_undo_redo_of_an_occluder_is_idempotent(session):
    session.goto(10)
    seed_shapes(session, 10)
    key = FrameKey(DESKTOP, 10, VIEW)
    session.commit_occluder(rect(0, 0, 20, 20), "hand")

    for _ in range(3):
        assert session.undo() is True
        assert session.db.occluders(key) == []
        assert session.redo() is True
        assert [o.occluder_type for o in session.db.occluders(key)] == ["hand"]


def test_undo_redo_of_a_zorder_move_is_idempotent(session):
    session.goto(10)
    seed_shapes(session, 10)
    before = list(session.db.zorder(DESKTOP, VIEW, 1).order)
    rows = session.instance_rows()
    session.set_zorder_move(rows[-1]["key"], rows[-2]["key"])
    after = list(session.db.zorder(DESKTOP, VIEW, 1).order)
    assert after != before

    for _ in range(3):
        assert session.undo() is True
        assert session.db.zorder(DESKTOP, VIEW, 1).order == before
        assert session.redo() is True
        assert session.db.zorder(DESKTOP, VIEW, 1).order == after


# --------------------------------------------------------------------------- #
# what an undo restores
# --------------------------------------------------------------------------- #
def test_undo_of_a_commit_restores_the_previous_shape(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(5), api.SCOPE_KEYFRAME)

    assert session.undo() is True
    assert np.array_equal(masks.decode_rle(keyframes(session)[0].parts[0].rle), cell(0))
    row = session.db.compiled(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert np.array_equal(masks.decode_rle(row["visible_rle"]), cell(0))


def test_undo_of_the_first_commit_takes_the_shape_out_of_every_frame(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert session.undo() is True

    assert keyframes(session) == []
    # the instance still needs geometry here, so the truth row stays -- but with
    # nothing in it, and the frame is back to reporting a missing shape
    row = session.db.compiled(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert row["visible_rle"] is None
    assert f"missing_shape:{COOLER}" in session.compiled().problems


def test_undo_of_a_brush_stroke_restores_the_editing_layer(session):
    session.goto(10)
    session.begin_edit(COOLER)
    before = np.zeros(HW, dtype=bool)
    session.set_editing_mask(before)
    session.push_stroke(before, cell(2))

    assert np.array_equal(session.editing_mask(), cell(2))
    assert session.undo() is True
    assert not session.editing_mask().any()  # the stroke never touched the database
    assert keyframes(session) == []


def test_a_stroke_undo_announces_the_new_editing_layer(session):
    """The window owns the overlay, so it has to be told (spec 10.2)."""
    session.goto(10)
    session.begin_edit(COOLER)
    seen: list = []
    session.sigEditingChanged.connect(seen.append)
    before = np.zeros(HW, dtype=bool)
    session.set_editing_mask(before)
    session.push_stroke(before, cell(2))
    seen.clear()

    assert session.undo() is True
    assert len(seen) == 1 and not seen[0].any()
    assert session.redo() is True
    assert len(seen) == 2 and np.array_equal(seen[1], cell(2))


def test_set_editing_mask_does_not_alias_the_callers_array(session):
    session.goto(10)
    session.begin_edit(COOLER)
    painted = cell(2)
    session.set_editing_mask(painted)
    painted[:] = False  # the window keeps painting into its own overlay buffer
    assert session.editing_mask().any()


# --------------------------------------------------------------------------- #
# the dirty flag follows the history (spec 4.6)
# --------------------------------------------------------------------------- #
def test_dirty_clears_again_when_undo_returns_to_the_saved_point(session):
    session.goto(10)
    assert session.dirty is False
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert session.dirty is True

    assert session.undo() is True
    assert session.dirty is False  # back where the last save left it

    assert session.redo() is True
    assert session.dirty is True
    session.save()
    assert session.dirty is False
    assert session.undo() is True
    assert session.dirty is True  # now it differs from the saved point again
