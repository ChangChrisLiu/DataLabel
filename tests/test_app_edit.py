"""Editing flow, edit scopes, ROI, review handling and crash safety (task 13b).

The brush strokes below are real mouse events delivered to the canvas viewport,
so the whole chain -- ``ImageCanvas`` signals, ``BrushTool``, the overlay's edit
layer, the session's undo stack -- is exercised rather than simulated.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import (
    DESKTOP,
    LAST_STEP,
    VIEW,
    StubSamQueue,
    make_db,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.core.cache import suggest_roi
from tda.core.model import FrameKey
from tda.core.truth import StaleConflictError, TruthService
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.session import AnnotationSession


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, **kwargs) -> MainWindow:
    session = make_session(tmp_path, **kwargs)
    win = MainWindow(session, make_paths(tmp_path), "tester",
                     sam_queue=StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    return win


@pytest.fixture
def window(qapp, tmp_path):
    win = open_window(tmp_path)
    yield win
    win.shutdown()


def paint(win: MainWindow, dx: int = 8) -> None:
    """One left-drag across the middle of the canvas viewport."""
    viewport = win.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(dx, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(dx, 0))
    QApplication.processEvents()


def first_task_instance(win: MainWindow) -> str:
    card = [row for row in win.session.task_card() if row.get("instance")]
    assert card, "the start frame should ask for shapes"
    return str(card[0]["instance"])


# --------------------------------------------------------------------------- #
# begin edit -> stroke -> commit -> undo
# --------------------------------------------------------------------------- #
def test_request_edit_loads_the_instance_into_the_editing_layer(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    assert window.session.editing_instance == instance
    assert window.overlay.editing_instance == instance
    assert window.session.editing_mask() is not None


def test_a_brush_stroke_is_one_undoable_op_on_the_editing_layer(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    window.act_tool("brush")
    paint(window)
    assert window.overlay.editing.any()
    assert np.array_equal(window.session.editing_mask(), window.overlay.editing)
    assert len(window.session.undo_stack) == 1
    assert window.session.undo_stack.ops[-1].kind == "edit_editing_mask"


def test_enter_commits_a_keyframe_and_ctrl_z_takes_it_back(window):
    session = window.session
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    window.act_tool("brush")
    paint(window)
    window.act_commit()
    assert session.db.keyframes(DESKTOP, VIEW, instance)
    window.act_undo()
    assert not session.db.keyframes(DESKTOP, VIEW, instance)
    window.act_redo()
    assert session.db.keyframes(DESKTOP, VIEW, instance)


def test_escape_drops_the_editing_layer(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    window.act_clear_edit()
    assert window.session.editing_mask() is None
    assert window.overlay.editing_instance is None
    assert not window.overlay.editing.any()


def test_fill_holes_and_despeckle_are_single_undoable_ops(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    mask[18:22, 18:22] = False      # the hole
    mask[50, 50] = True             # the speck
    window.set_editing_mask(mask)
    before = len(window.session.undo_stack)
    window.act_fill_holes()
    assert window.session.editing_mask()[20, 20]
    window.act_despeckle()
    assert not window.session.editing_mask()[50, 50]
    assert len(window.session.undo_stack) == before + 2


def test_space_on_an_incomplete_frame_shows_problems_and_does_not_advance(window):
    session = window.session
    step = session.current().step
    assert window.act_confirm() is False
    assert session.current().step == step
    assert window.task_card.problems_visible() is True
    assert window.status_message()


def test_space_advances_once_the_frame_is_complete(window):
    session = window.session
    seed_shapes(session, LAST_STEP)
    session.goto(LAST_STEP)
    assert window.act_confirm() is True
    assert session.current().step < LAST_STEP


# --------------------------------------------------------------------------- #
# edit scopes
# --------------------------------------------------------------------------- #
def test_a_non_keyframe_suggestion_opens_the_scope_bar_instead_of_a_dialog(window,
                                                                          monkeypatch):
    session = window.session
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:above:other")
    committed: list[str] = []
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})

    window.act_commit()
    assert window.scope_bar.isVisibleTo(window)
    assert "zorder:above:other" in window.scope_bar_text()
    assert committed == []          # nothing written yet

    window.act_commit()             # Enter accepts the suggestion
    assert committed == ["zorder:above:other"]
    assert not window.scope_bar.isVisibleTo(window)


def test_the_scope_bar_alternatives_are_frame_override_and_split(window, monkeypatch):
    session = window.session
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:above:other")
    committed: list[str] = []
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})
    window.act_commit()
    window.act_commit_override()
    assert committed == [api.SCOPE_FRAME_OVERRIDE]
    assert not window.scope_bar.isVisibleTo(window)

    committed.clear()
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    window.act_commit()
    window.act_commit_split()
    assert committed == [api.SCOPE_SPLIT]


def test_a_refused_commit_keeps_the_edit_and_shows_the_reason(window, monkeypatch):
    """A part on the bench needs the bench box; that is an answer, not a crash."""
    session = window.session
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)

    def refuse(scope, *a, **k):
        raise ValueError(f"{instance} is on the bench at step 3: use the bench box tool")

    monkeypatch.setattr(session, "commit_edit", refuse)
    window.act_commit()
    assert "bench box" in window.last_error_message()
    assert session.editing_instance == instance          # the edit is still there
    assert not window.scope_bar.isVisibleTo(window)


def test_a_plain_keyframe_suggestion_commits_straight_away(window, monkeypatch):
    session = window.session
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    committed: list[str] = []
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: api.SCOPE_KEYFRAME)
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})
    window.act_commit()
    assert committed == [api.SCOPE_KEYFRAME]
    assert not window.scope_bar.isVisibleTo(window)


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #
def test_first_open_proposes_an_roi_and_enter_stores_it(qapp, tmp_path):
    win = open_window(tmp_path)
    try:
        session = win.session
        key = session.current()
        assert win.roi_editing is True
        expected = suggest_roi(session.image(), VIEW)
        assert tuple(win.roi_draft) == tuple(expected)
        win.act_commit()                      # Enter accepts the rectangle
        assert win.roi_editing is False
        stored = session.db.pose_segment_for(key)["roi"]
        assert tuple(stored) == tuple(expected)
    finally:
        win.shutdown()


def test_a_second_open_does_not_ask_for_the_roi_again(qapp, tmp_path):
    win = open_window(tmp_path)
    win.act_commit()
    win.shutdown()

    session = make_session(tmp_path)          # same tmp database on disk
    again = MainWindow(session, make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        assert again.roi_editing is False
        assert again.roi() is not None
    finally:
        again.shutdown()


def test_shift_r_re_edits_the_roi_and_escape_keeps_the_old_one(qapp, tmp_path):
    win = open_window(tmp_path)
    try:
        win.act_commit()
        stored = tuple(win.roi())
        win.act_edit_roi()
        assert win.roi_editing is True
        win.roi_draft = (1, 2, 30, 40)
        win.act_clear_edit()                  # Esc
        assert win.roi_editing is False
        assert tuple(win.roi()) == stored
    finally:
        win.shutdown()


def test_fit_roi_zooms_to_the_stored_rectangle(window):
    window.act_commit()                       # store the proposed ROI
    window.canvas.set_zoom(1.0)
    window.act_fit_roi()
    assert window.canvas.zoom_factor() > 1.0


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #
def test_a_stale_conflict_is_reported_as_superseded_and_refreshes_the_queue(window,
                                                                           monkeypatch):
    def stale(cid, resolution):
        raise StaleConflictError("the frame moved on")

    monkeypatch.setattr(window.session, "resolve_conflict", stale)
    window.set_mode(A.MODE_REVIEW)
    window.resolve_conflict(1, api.RESOLVE_ACCEPT_NEW)
    message = window.status_message()
    assert "superseded" in message.lower() or "重新入队" in message
    assert window.review_refreshes >= 1


def test_a_refused_conflict_shows_the_refusal_text(window, monkeypatch):
    def refused(cid, resolution):
        raise ValueError("keep_old is not allowed here")

    monkeypatch.setattr(window.session, "resolve_conflict", refused)
    window.resolve_conflict(1, api.RESOLVE_KEEP_OLD)
    assert "keep_old is not allowed here" in window.status_message()


def test_a_session_that_returns_a_string_is_handled_without_raising(window,
                                                                    monkeypatch):
    """The fix round replaces the exceptions with a returned verdict."""
    monkeypatch.setattr(window.session, "resolve_conflict",
                        lambda cid, resolution: "superseded")
    window.resolve_conflict(7, api.RESOLVE_ACCEPT_NEW)
    assert "superseded" in window.status_message().lower()


# --------------------------------------------------------------------------- #
# crash safety
# --------------------------------------------------------------------------- #
def test_an_uncommitted_stroke_is_written_to_a_sidecar_and_offered_back(qapp,
                                                                       tmp_path):
    win = open_window(tmp_path)
    try:
        instance = first_task_instance(win)
        win.task_card.sigRequestEdit.emit(instance)
        paint(win)
        painted = win.session.editing_mask().copy()
        key = win.session.current()
        win.flush_sidecar()
        assert Path(win.sidecar.path_for(key, instance)).exists()
    finally:
        win.shutdown()                        # the process "dies" here

    session = make_session(tmp_path)
    again = MainWindow(session, make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        pending = again.pending_restore()
        assert pending is not None
        assert pending["instance"] == instance
        assert again.restore_bar.isVisibleTo(again)
        again.restore_pending()
        assert np.array_equal(again.session.editing_mask(), painted)
        assert again.pending_restore() is None
    finally:
        again.shutdown()


def test_committing_clears_the_sidecar(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    key = window.session.current()
    window.flush_sidecar()
    assert Path(window.sidecar.path_for(key, instance)).exists()
    window.act_commit()
    assert window.sidecar.pending_for(key, instance) is None
    assert not Path(window.sidecar.path_for(key, instance)).exists()


def test_a_sidecar_for_another_frame_is_not_offered(qapp, tmp_path):
    win = open_window(tmp_path)
    try:
        instance = first_task_instance(win)
        win.task_card.sigRequestEdit.emit(instance)
        paint(win)
        win.flush_sidecar()
        win.sidecar.clear(win.session.current(), instance)
        win.sidecar.save(FrameKey(DESKTOP, 2, VIEW), instance,
                         np.ones((64, 64), dtype=bool))
    finally:
        win.shutdown()

    session = make_session(tmp_path)
    again = MainWindow(session, make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        assert again.session.current().step == LAST_STEP
        assert again.pending_restore() is None
    finally:
        again.shutdown()
