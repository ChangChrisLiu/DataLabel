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
    close_window,
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
    close_window(win)


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
    """The two keys beside ``Enter``; the split keeps the suggestion it answers."""
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
    assert committed == ["split+zorder:above:other"]

    committed.clear()
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    window.act_commit_split()          # no suggestion on the bar: a plain split
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
        close_window(win)


def test_a_second_open_does_not_ask_for_the_roi_again(qapp, tmp_path):
    win = open_window(tmp_path)
    win.act_commit()
    close_window(win)

    session = make_session(tmp_path)          # same tmp database on disk
    again = MainWindow(session, make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        assert again.roi_editing is False
        assert again.roi() is not None
    finally:
        close_window(again)


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
        close_window(win)


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
        close_window(win)                        # the process "dies" here

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
        close_window(again)


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
        close_window(win)

    session = make_session(tmp_path)
    again = MainWindow(session, make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        assert again.session.current().step == LAST_STEP
        assert again.pending_restore() is None
    finally:
        close_window(again)


def test_the_scope_bar_says_whether_the_shape_is_written_too(window, monkeypatch):
    """A layering commit that carries painted pixels writes two things, not one.

    ``zorder:above:B`` with pixels added inside ``B`` re-traces the keyframe as
    well; ``zorder:below:B`` only reverses the pair.  The bar has to say which
    of the two the annotator is about to accept.
    """
    session = window.session
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:above:other")
    window.act_commit()
    assert "层级 + 形状" in window.scope_bar_text()

    window.act_clear_edit()
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:below:other")
    window.act_commit()
    assert "仅层级" in window.scope_bar_text()


def test_the_scope_bar_says_what_each_of_the_three_keys_writes(window, monkeypatch):
    """Three buttons, three different writes: the bar names all three."""
    session = window.session
    window.task_card.sigRequestEdit.emit(first_task_instance(window))
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:above:other")
    window.act_commit()

    text = window.scope_bar_text()
    assert "Enter" in text and "Alt+Enter" in text and "Ctrl+K" in text
    assert "层级 + 形状" in text and "仅本帧" in text and "拆分" in text


@pytest.mark.parametrize("key,expected", [
    ("act_commit", "zorder:above:other"),
    ("act_commit_split", "split+zorder:above:other"),
    ("act_commit_override", api.SCOPE_FRAME_OVERRIDE),
])
def test_the_three_answers_to_a_layering_suggestion(window, monkeypatch, key, expected):
    """``Ctrl+K`` wrote the split pixels and no pair, so the screen did not change.

    The pixels were saved and stayed hidden under ``B``.  A split accepts the
    suggestion as much as ``Enter`` does -- it only asks for a new version of the
    shape rather than a re-trace -- so it carries the pair too.  ``Alt+Enter``
    stays a pure frame override: it makes A visible here by itself.
    """
    session = window.session
    window.task_card.sigRequestEdit.emit(first_task_instance(window))
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:above:other")
    window.act_commit()                     # the bar is up with the suggestion
    committed: list[str] = []
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})

    getattr(window, key)()

    assert committed == [expected]
    assert not window.scope_bar.isVisibleTo(window)


# --------------------------------------------------------------------------- #
# a blocked list click does not leave the list pointing somewhere else
# --------------------------------------------------------------------------- #
def test_a_blocked_timeline_click_snaps_the_selection_back(window):
    """The row the annotator clicked stayed highlighted over a refused move.

    Qt selects the row before the click is delivered, so a refusal left the
    timeline pointing at a frame that is not the one on the canvas.
    """
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    here = window.session.current().step
    lw = window.timeline.list_widget()
    row = next(r for r in range(lw.count())
               if int(lw.item(r).data(int(Qt.ItemDataRole.UserRole))) != here)

    lw.setCurrentRow(row)                 # what the mouse does before the click
    lw.itemClicked.emit(lw.item(row))
    QApplication.processEvents()

    assert window.session.current().step == here
    assert window.timeline.current_step() == here


def test_a_blocked_card_activation_snaps_the_selection_back(window):
    """... and so did the task-card row of the instance that was refused."""
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    lw = window.task_card.list_widget()
    rows = [r for r in range(lw.count())
            if str(lw.item(r).data(int(Qt.ItemDataRole.UserRole))) != instance]
    if not rows:
        pytest.skip("need a second card item")

    lw.setCurrentRow(rows[0])
    lw.itemActivated.emit(lw.item(rows[0]))
    QApplication.processEvents()

    assert window.session.editing_instance == instance
    assert window.task_card.current_instance() == instance


# --------------------------------------------------------------------------- #
# Review mode: Enter accepts the frame the queue points at (item 9)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("queue", list(api.QUEUE_NAMES))
def test_review_enter_opens_the_selected_entry_then_confirms(window, monkeypatch, queue):
    """The label said "the frame the queue points at"; it confirmed the open one.

    The annotator clicks an entry, presses Enter, and a *different* frame --
    whichever one happened to be on the canvas -- is marked verified.
    """
    here = window.session.current().step
    target = min(s for s in window.session.steps() if s != here)
    monkeypatch.setattr(window.session, "queues", lambda: {
        q: ([{"step": target, "instance": "chassis", "id": 1}] if q == queue else [])
        for q in api.QUEUE_NAMES
    })
    window.set_mode(A.MODE_REVIEW)
    window.review.refresh()
    window.review.tabs().setCurrentIndex(list(api.QUEUE_NAMES).index(queue))
    window.review.list_for(queue).setCurrentRow(0)
    confirmed: list[int] = []
    monkeypatch.setattr(window.task_card, "confirm",
                        lambda: confirmed.append(window.session.current().step) or True)

    window.act_confirm()

    assert confirmed == [target], "it confirmed the frame that was on screen"
    assert window.session.current().step == target


def test_review_enter_on_the_open_frame_confirms_it(window, monkeypatch):
    """Nothing selected anywhere: Enter still means "accept this one"."""
    monkeypatch.setattr(window.session, "queues",
                        lambda: {q: [] for q in api.QUEUE_NAMES})
    window.set_mode(A.MODE_REVIEW)
    window.review.refresh()
    here = window.session.current().step
    confirmed: list[int] = []
    monkeypatch.setattr(window.task_card, "confirm",
                        lambda: confirmed.append(window.session.current().step) or True)

    window.act_confirm()

    assert confirmed == [here]


def test_review_enter_is_refused_with_an_uncommitted_edit(window, monkeypatch):
    """Opening another frame is a way out of the edit, wherever it is asked for."""
    window.set_mode(A.MODE_ANNOTATE)
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    here = window.session.current().step

    window.act_confirm()

    assert window.session.current().step == here
    assert window.session.editing_instance == instance


# --------------------------------------------------------------------------- #
# a refused Space on a real start frame (item 11)
# --------------------------------------------------------------------------- #
def test_a_refused_confirm_is_one_short_line(window):
    """60 missing shapes made a 2,550-character status line 30,612 px wide."""
    window.act_confirm()

    line = window.status_message()
    assert len(line) < 120, f"{len(line)} characters in the status bar"
    assert "problem" in line or "个问题" in line
    assert "task card" in line or "任务卡" in line


def test_the_hint_label_never_grows_the_window(window):
    from PySide6.QtWidgets import QSizePolicy

    window.report("x" * 4000)
    hint = window.hint_label
    assert hint.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
    assert hint.minimumSizeHint().width() <= 200
    assert hint.toolTip() == "x" * 4000       # the whole line is still readable


def test_the_hint_is_cleared_on_a_frame_change(window):
    """A stale line from two actions ago stayed on screen, still being read."""
    window.report("something about the frame we are leaving")
    window.session.goto(min(window.session.steps()), force=True)
    QApplication.processEvents()
    assert "the frame we are leaving" not in window.status_message()
    assert len(window.status_message()) < 120


def test_each_problem_appears_once_with_its_code_in_the_tooltip(window):
    """The pane listed `missing_shape:x` AND "Draw x on this frame"."""
    window.act_confirm()

    rows = window.task_card.problem_rows()
    assert rows, "no problems were shown"
    assert len(rows) == len({r["instance"] for r in rows}), rows
    for row in rows:
        assert not row["text"].startswith("missing_shape:")
        assert row["code"].startswith("missing_shape:")
        assert row["instance"] and row["instance"] in row["code"]


def test_clicking_a_problem_selects_that_instances_card_item(window):
    window.act_confirm()
    rows = window.task_card.problem_rows()
    card = [str(x["instance"]) for x in window.session.task_card()]
    wanted = next(r for r in rows if r["instance"] in card)

    window.task_card.activate_problem(wanted["instance"])

    assert window.task_card.current_instance() == wanted["instance"]


# --------------------------------------------------------------------------- #
# the ROI is proposed from the segment's reference frame (item 14)
# --------------------------------------------------------------------------- #
def test_the_roi_is_proposed_from_the_segments_first_frame(qapp, tmp_path, monkeypatch):
    """It was proposed from whatever frame happened to be open -- the LAST one.

    On the real D13 that is the empty chassis with a bright interior, where both
    scanner strategies fail and the proposal is the whole frame; the strongest
    diff blob then sat on a scan-bed artefact at the right edge and *that*
    became the SAM prompt box on 7 of 13 frames.
    """
    from tda.ui import app_roi

    asked: list = []
    win = open_window(tmp_path)
    try:
        def remember(img, view):
            asked.append(np.array(img, copy=True))
            return (4, 4, 40, 40)

        monkeypatch.setattr(app_roi, "suggest_roi", remember)
        win.start_roi_edit()

        assert asked, "no proposal was made"
        first = min(s for s in win.session.steps()
                    if win.session.image_at(s) is not None)
        assert np.array_equal(asked[-1], win.session.image_at(first)), (
            "the proposal was measured on a frame other than the segment's first"
        )
        assert not np.array_equal(asked[-1], win.session.image())   # not the open one
    finally:
        close_window(win)


def test_a_full_frame_proposal_is_not_stored_without_a_drag(qapp, tmp_path, monkeypatch):
    """A full-frame ROI is "I could not find the chassis", not an answer."""
    from tda.ui import app_roi

    win = open_window(tmp_path)
    try:
        monkeypatch.setattr(app_roi, "suggest_roi", lambda img, view: (0, 0, 64, 64))
        win.start_roi_edit()
        assert "未能自动找到机箱" in win.status_message()

        win.act_commit()                       # Enter, with nothing dragged

        assert win.roi() is None, "the full frame was stored"
        assert win.roi_editing is True         # still waiting for a rectangle

        win.on_roi_box((8.0, 8.0, 40.0, 40.0))  # the annotator drags one
        win.act_commit()

        assert win.roi() == (8, 8, 40, 40)
        assert win.roi_editing is False
    finally:
        close_window(win)


def test_no_prompt_box_while_the_segment_has_no_roi(qapp, tmp_path, monkeypatch):
    """A diff blob outside a known chassis is as likely to be a scan artefact."""
    from tda.core.diffmap import DiffBlob

    win = open_window(tmp_path)
    try:
        assert win.roi() is None
        win.begin_add_shape(DiffBlob(box=(10.0, 10.0, 20.0, 20.0), area=100,
                                     score=9.0))
        assert win.sam_point.prompt_box is None
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# a commit with nothing to commit (addendum, item 16a)
# --------------------------------------------------------------------------- #
def test_enter_on_an_unchanged_layer_says_so_and_changes_nothing(window):
    """21 of 21 dry-run Enters reported "committed (keyframe): False".

    It also cleared the editing layer, so the annotator lost the instance they
    had just loaded and had to double-click it again.
    """
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    before = window.session.editing_mask().copy()
    ops = len(window.session.undo_stack)

    window.act_commit()

    assert "没有可提交的修改" in window.status_message()
    assert window.session.editing_instance == instance
    assert np.array_equal(window.session.editing_mask(), before)
    assert len(window.session.undo_stack) == ops
    assert not window.session.db.keyframes(DESKTOP, VIEW, instance)


def test_enter_after_a_stroke_still_commits(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    window.act_commit()
    assert window.session.db.keyframes(DESKTOP, VIEW, instance)


# --------------------------------------------------------------------------- #
# a stroke that changes nothing writes no sidecar (addendum, item 18)
# --------------------------------------------------------------------------- #
def test_a_stroke_that_changes_nothing_writes_no_sidecar(window):
    """Then it offered to "restore" a layer identical to the committed shape."""
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    window.flush_sidecar()
    before = window.sidecar_writes

    window.queue_sidecar(window.session.current(), instance,
                         window.session.editing_mask())
    window.flush_sidecar()

    assert window.sidecar_writes == before
    assert window.sidecar.pending_for(window.session.current(), instance) is None


def test_a_real_stroke_still_writes_a_sidecar(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    window.flush_sidecar()
    assert window.sidecar.pending_for(window.session.current(), instance) is not None


# --------------------------------------------------------------------------- #
# the area warning bar (addendum, item 16b)
# --------------------------------------------------------------------------- #
def tiny_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=bool)
    mask[30, 30:33] = True          # 3 px, one pixel tall
    return mask


def test_a_tiny_mask_warns_once_and_commits_on_the_second_enter(window):
    """A 9-px "part" and a 1-px-wide sliver both went in without a word."""
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    window.set_editing_mask(tiny_mask(), undoable=True)

    window.act_commit()

    assert window.warn_bar.isVisibleTo(window)
    assert "掩码过小" in window.warn_bar_text()
    assert not window.session.db.keyframes(DESKTOP, VIEW, instance), "it was written"
    assert window.session.editing_instance == instance

    window.act_commit()             # the annotator says they meant it

    assert window.session.db.keyframes(DESKTOP, VIEW, instance)
    assert not window.warn_bar.isVisibleTo(window)


def test_escape_on_the_warning_returns_to_editing(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    window.set_editing_mask(tiny_mask(), undoable=True)
    window.act_commit()
    assert window.warn_bar.isVisibleTo(window)

    window.act_clear_edit()

    assert not window.warn_bar.isVisibleTo(window)
    assert window.session.editing_instance == instance, "the layer was discarded"
    assert window.session.editing_mask().any()


def test_a_mask_far_outside_its_class_prior_warns(window, monkeypatch):
    """1,502,386 px committed as a screw, with nothing said."""
    from tda.ui import app_priors

    instance = first_task_instance(window)
    monkeypatch.setattr(type(window), "_class_of", lambda _s, _i: "screw")
    window.priors = app_priors.AreaPriors(
        {"screw": {"min_frac": 0.00003, "max_frac": 0.0002}})
    window.task_card.sigRequestEdit.emit(instance)
    big = np.zeros((64, 64), dtype=bool)
    big[8:56, 8:56] = True                      # most of the ROI
    window.set_editing_mask(big, undoable=True)

    window.act_commit()

    assert window.warn_bar.isVisibleTo(window)
    assert "screw" in window.warn_bar_text()
    assert not window.session.db.keyframes(DESKTOP, VIEW, instance)


def test_a_plausible_mask_is_not_warned_about(window):
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:32, 20:32] = True
    window.set_editing_mask(mask, undoable=True)

    window.act_commit()

    assert not window.warn_bar.isVisibleTo(window)
    assert window.session.db.keyframes(DESKTOP, VIEW, instance)


def test_an_overridden_warning_is_logged(window):
    from tda.ui import app_support as S

    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    window.set_editing_mask(tiny_mask(), undoable=True)
    window.act_commit()
    window.act_commit()

    for handler in window.logger.handlers:
        handler.flush()
    text = Path(S.log_path(window.paths)).read_text(encoding="utf-8")
    assert "area_warning_overridden" in text and instance in text


# --------------------------------------------------------------------------- #
# the debounce must not outlive what it was protecting (F3 round 2, item 2)
# --------------------------------------------------------------------------- #
def test_an_undo_inside_the_debounce_window_cancels_the_write(window):
    """Brush, Ctrl+Z within 300 ms: the stale 197-px mask was written anyway.

    The next open then offered to "restore" pixels the annotator had already
    taken back.
    """
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    assert window._sidecar_pending is not None
    before = window.sidecar_writes

    window.act_undo()             # inside the debounce window

    assert window._sidecar_pending is None
    assert not window._sidecar_timer.isActive()
    window.flush_sidecar()
    assert window.sidecar_writes == before
    assert window.sidecar.pending_for(window.session.current(), instance) is None


def test_an_undo_after_the_write_deletes_the_sidecar(window):
    """... and after the 300 ms, where the file is already on disk."""
    instance = first_task_instance(window)
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    window.flush_sidecar()        # the debounce elapsed
    assert window.sidecar.pending_for(window.session.current(), instance) is not None

    window.act_undo()

    assert window.sidecar.pending_for(window.session.current(), instance) is None
    assert window.pending_restore() is None


# --------------------------------------------------------------------------- #
# a net-zero gesture may only delete its own crash copy (F3 round 3, item 1)
# --------------------------------------------------------------------------- #
def test_a_net_zero_gesture_never_deletes_another_sessions_sidecar(qapp, tmp_path):
    """The A/B the reviewer ran: a previous run's crash copy was deleted.

    Window 1 paints and flushes, the process dies, window 2 opens the same
    frame and offers to restore it -- and the annotator's first gesture on that
    instance is a net-zero one (a stroke then Ctrl+Z).  That gesture used to
    take the *previous session's* work with it, which is the one thing the
    sidecar exists to prevent.
    """
    first = open_window(tmp_path)
    instance = first_task_instance(first)
    try:
        first.task_card.sigRequestEdit.emit(instance)
        paint(first)
        first.flush_sidecar()
        key = first.session.current()
        assert first.sidecar.pending_for(key, instance) is not None
    finally:
        close_window(first)                      # a crash-style exit: no commit

    second = open_window(tmp_path)
    try:
        assert second.pending_restore() is not None, "nothing was offered"
        second.task_card.sigRequestEdit.emit(instance)
        paint(second)
        second.act_undo()                        # net zero: back to begin_edit

        assert second.sidecar.pending_for(key, instance) is not None, (
            "the earlier session's crash copy was deleted by a net-zero gesture"
        )
        second.session.goto(min(second.session.steps()), force=True)
        second.session.goto(key.step, force=True)
        QApplication.processEvents()
        assert second.pending_restore() is not None, "the offer is gone"
    finally:
        close_window(second)


def test_answering_the_restore_offer_still_removes_the_file(qapp, tmp_path):
    """The offer is the *only* thing that may drop somebody else's copy."""
    first = open_window(tmp_path)
    instance = first_task_instance(first)
    try:
        first.task_card.sigRequestEdit.emit(instance)
        paint(first)
        first.flush_sidecar()
        key = first.session.current()
    finally:
        close_window(first)

    second = open_window(tmp_path)
    try:
        assert second.pending_restore() is not None
        second.discard_pending()
        assert second.sidecar.pending_for(key, instance) is None
    finally:
        close_window(second)


# --------------------------------------------------------------------------- #
# Esc is not one of the three ways a foreign copy may go (F3 round 4, item 1)
# --------------------------------------------------------------------------- #
def leave_a_crash_copy(tmp_path: Path) -> tuple[str, object]:
    """Window 1 paints, flushes and dies; returns ``(instance, key)``."""
    first = open_window(tmp_path)
    instance = first_task_instance(first)
    try:
        first.task_card.sigRequestEdit.emit(instance)
        paint(first)
        first.flush_sidecar()
        key = first.session.current()
        assert first.sidecar.pending_for(key, instance) is not None
        return instance, key
    finally:
        close_window(first)


@pytest.mark.parametrize("paint_first", [False, True],
                         ids=["no-paint", "unflushed-stroke"])
def test_escape_never_deletes_another_sessions_offer(qapp, tmp_path, paint_first):
    """Walk (f): ``Esc`` deleted a crash copy that was still being offered.

    Both variants of the reviewer's walk: straight after ``begin_edit`` with
    nothing painted, and after an own stroke that has not been flushed yet.
    The offer disappeared with the file, so the previous session's work was
    gone with no answer from anybody.
    """
    instance, key = leave_a_crash_copy(tmp_path)
    second = open_window(tmp_path)
    try:
        assert second.pending_restore() is not None, "nothing was offered"
        second.task_card.sigRequestEdit.emit(instance)
        if paint_first:
            paint(second)

        second.act_clear_edit()          # Esc

        assert second.sidecar.pending_for(key, instance) is not None, (
            "Esc deleted a crash copy that was still being offered"
        )
        assert second.pending_restore() is not None, "the offer is gone"
        assert second.restore_bar.isVisibleTo(second)
    finally:
        close_window(second)


def test_escape_still_deletes_this_windows_own_copy(qapp, tmp_path):
    """Nothing changes for the layer the annotator was actually working on."""
    win = open_window(tmp_path)
    try:
        instance = first_task_instance(win)
        win.task_card.sigRequestEdit.emit(instance)
        paint(win)
        win.flush_sidecar()
        key = win.session.current()
        assert win.sidecar.pending_for(key, instance) is not None

        win.act_clear_edit()

        assert win.sidecar.pending_for(key, instance) is None
    finally:
        close_window(win)


def test_committing_an_instance_clears_even_a_foreign_copy(qapp, tmp_path):
    """One of the three ways: the instance is written, so the copy is stale."""
    instance, key = leave_a_crash_copy(tmp_path)
    second = open_window(tmp_path)
    try:
        second.task_card.sigRequestEdit.emit(instance)
        paint(second)
        second.act_commit()
        if second.warn_bar.isVisibleTo(second):
            second.act_commit()

        assert second.session.db.keyframes(DESKTOP, VIEW, instance)
        assert second.sidecar.pending_for(key, instance) is None
    finally:
        close_window(second)
