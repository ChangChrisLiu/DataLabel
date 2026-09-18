"""Never lose a manual edit, and never leave a stroke nobody asked for.

Fix round 1 of task 13b: the reviewers found three ways an uncommitted editing
layer could disappear without the annotator being told -- navigating away,
activating another instance, and painting while nothing was being edited -- plus
a crash sidecar that only came back on a frame change.  Each rule gets a test
here; the sequences at the end walk a whole edit the way an annotator would.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

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
from tda.core.db import acquire_lock_file, lock_path_for, read_lock_file
from tda.core.db_pose import clean_roi
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui.app import MainWindow


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, roi: bool = True, **kwargs) -> MainWindow:
    """A window standing on the last frame with the proposed ROI accepted.

    Accepting it matters here: while the ROI rectangle is being dragged the
    canvas belongs to the ROI tool, and these tests are about what the *brush*
    does.
    """
    session = make_session(tmp_path, **kwargs)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    if roi and win.roi_editing:
        win.act_commit()
    return win


@pytest.fixture
def window(qapp, tmp_path):
    win = open_window(tmp_path)
    yield win
    win.shutdown()


def paint(win: MainWindow, dx: int = 8) -> None:
    viewport = win.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(dx, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(dx, 0))
    QApplication.processEvents()


def card_instances(win: MainWindow) -> list[str]:
    return [str(r["instance"]) for r in win.session.task_card() if r.get("instance")]


def drawable_instances(win: MainWindow) -> list[str]:
    """Instances that are painted rather than boxed: in the chassis, on the card.

    Falls back to the frame's chassis instances when the card happens to ask
    only for bench boxes, which is what the start frame of a torn-down machine
    looks like.
    """
    in_chassis = [str(r.get("key")) for r in win.session.instance_rows()
                  if r.get("placement") == "in_chassis"]
    wanted = [i for i in card_instances(win) if i in in_chassis]
    return wanted or in_chassis


def start_edit(win: MainWindow, index: int = 0) -> str:
    instance = card_instances(win)[index]
    win.task_card.sigRequestEdit.emit(instance)
    return instance


# --------------------------------------------------------------------------- #
# the panels no longer begin an edit behind the window's back
# --------------------------------------------------------------------------- #
def test_the_panels_only_report_the_request(window):
    """``begin_edit`` belongs to the window: it is the one that can refuse."""
    seen: list[str] = []
    window.task_card.sigRequestEdit.disconnect()
    window.task_card.sigRequestEdit.connect(seen.append)
    window.task_card._on_item_activated(window.task_card.list_widget().item(0))
    assert seen == [card_instances(window)[0]]
    assert window.session.editing_instance is None      # the panel started nothing


# --------------------------------------------------------------------------- #
# navigating away from an uncommitted edit
# --------------------------------------------------------------------------- #
#: Every gesture that could take the annotator off the frame.  The parametrised
#: test below is the audit: adding a way out without routing it through the gate
#: fails here rather than in six months' worth of lost strokes.
WAYS_OUT = {
    "step_back": lambda w: w.act_step(-1),
    "step_forward": lambda w: w.act_step(+1),
    "first": lambda w: w.act_step_edge("first"),
    "last": lambda w: w.act_step_edge("last"),
    "timeline": lambda w: w.timeline_goto(min(w.session.steps())),
    "view": lambda w: w.act_set_view("oak1"),
    "desktop": lambda w: w.act_set_desktop(DESKTOP),
    "mode_review": lambda w: w.set_mode(A.MODE_REVIEW),
    "mode_steps": lambda w: w.set_mode(A.MODE_STEPS),
    "refresh_all": lambda w: w.act_refresh_all(),
    "rework": lambda w: w.on_rework(min(w.session.steps())),
    "steps_saved": lambda w: w.on_steps_saved(DESKTOP),
    "restore_sidecar": lambda w: w.restore_pending(),
    "confirm": lambda w: w.act_confirm(),
}


def _offer_a_sidecar(win: MainWindow) -> None:
    """A recovered layer waiting to be restored, for the ``restore_sidecar`` case."""
    other = [i for i in card_instances(win) if i != win.session.editing_instance][0]
    win.sidecar.save(win.session.current(), other, np.ones((64, 64), dtype=bool))
    win._restore_offer = {"instance": other, "key": win.session.current(),
                          "mask": np.ones((64, 64), dtype=bool)}


@pytest.mark.parametrize("name", sorted(WAYS_OUT), ids=sorted(WAYS_OUT))
def test_an_uncommitted_edit_blocks_every_way_out_of_the_frame(window, name):
    instance = start_edit(window)
    paint(window)
    if name == "restore_sidecar":
        _offer_a_sidecar(window)
    painted = window.session.editing_mask().copy()
    overlay = window.overlay.editing.copy()
    step, view, mode = window.session.current().step, window.session.view, window.mode

    WAYS_OUT[name](window)

    assert window.session.current().step == step
    assert window.session.view == view
    assert window.mode == mode
    assert window.session.editing_instance == instance
    assert np.array_equal(window.session.editing_mask(), painted)
    assert np.array_equal(window.overlay.editing, overlay)
    assert "Enter" in window.status_message()
    # a block flushes the debounce: inside those 300 ms the layer was only in
    # memory, which is exactly the window a crash would have taken it in
    assert window.sidecar.pending_for(window.session.current(), instance) is not None


def test_the_session_itself_refuses_to_move_with_an_uncommitted_layer(window):
    """Defence in depth: the gate is in the session too, not only in the window."""
    from tda.ui.session_api import SessionRefusal

    session = window.session
    start_edit(window)
    paint(window)
    with pytest.raises(SessionRefusal):
        session.goto(min(session.steps()))
    with pytest.raises(SessionRefusal):
        session.prev()
    with pytest.raises(SessionRefusal):
        session.open(DESKTOP, "oak1")
    with pytest.raises(SessionRefusal):
        session.close()
    step = session.current().step
    session.goto(min(session.steps()), force=True)      # what the window uses
    assert session.current().step < step


def test_navigation_is_allowed_again_once_the_edit_is_committed(window):
    start_edit(window)
    paint(window)
    window.act_commit()
    step = window.session.current().step
    window.act_step(-1)
    assert window.session.current().step < step


def test_navigation_is_allowed_again_once_the_edit_is_discarded(window):
    start_edit(window)
    paint(window)
    window.act_clear_edit()
    step = window.session.current().step
    window.act_step(-1)
    assert window.session.current().step < step


def test_an_untouched_edit_does_not_block(window):
    """Opening an instance and touching nothing is not a change."""
    start_edit(window)
    step = window.session.current().step
    window.act_step(-1)
    assert window.session.current().step < step


def test_activating_another_instance_mid_edit_is_refused(window):
    first = start_edit(window)
    paint(window)
    others = [i for i in card_instances(window) if i != first]
    window.task_card.sigRequestEdit.emit(others[0])
    assert window.session.editing_instance == first
    assert "Enter" in window.status_message()


# --------------------------------------------------------------------------- #
# painting with nothing selected
# --------------------------------------------------------------------------- #
def test_painting_with_no_instance_adopts_the_current_task_item(window):
    assert window.session.editing_instance is None
    paint(window)
    assert window.session.editing_instance == card_instances(window)[0]
    assert window.overlay.editing.any()
    assert "editing" in window.status_message()


def test_painting_with_nothing_to_adopt_leaves_no_pixels(window, monkeypatch):
    monkeypatch.setattr(window.session, "task_card", lambda: [])
    window.task_card.refresh()
    paint(window)
    assert window.session.editing_instance is None
    assert not window.overlay.editing.any()
    assert "任务卡" in window.status_message()


# --------------------------------------------------------------------------- #
# closing with work in flight
# --------------------------------------------------------------------------- #
def test_closing_with_an_uncommitted_edit_asks_and_can_be_cancelled(qapp, tmp_path,
                                                                    monkeypatch):
    win = open_window(tmp_path)
    start_edit(win)
    paint(win)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))
    win.close()
    assert win.closed is False
    assert win.session.editing_instance is not None

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard))
    win.close()
    assert win.closed is True


def test_closing_and_committing_writes_the_edit(qapp, tmp_path, monkeypatch):
    win = open_window(tmp_path)
    instance = start_edit(win)
    paint(win)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save))
    db = win.session.db
    win.close()
    assert win.closed is True
    assert db.keyframes(DESKTOP, VIEW, instance)


# --------------------------------------------------------------------------- #
# the crash sidecar
# --------------------------------------------------------------------------- #
def test_the_sidecar_is_keyed_by_frame_and_instance(window):
    instance = start_edit(window)
    paint(window)
    window.flush_sidecar()
    key = window.session.current()
    stored = sorted(Path(window.sidecar.directory()).glob("*.json"))
    assert len(stored) == 1
    assert window.sidecar.pending_for(key, instance) is not None
    assert window.sidecar.pending_for(key, "someone.else") is None
    assert window.sidecar.pending_for(FrameKey(DESKTOP, 2, VIEW), instance) is None


def test_the_sidecar_write_is_debounced(window):
    start_edit(window)
    paint(window)
    paint(window, dx=12)
    paint(window, dx=16)
    assert window.sidecar_writes == 0          # nothing on the stroke path itself
    window.flush_sidecar()
    assert window.sidecar_writes == 1


def test_beginning_the_same_edit_again_offers_the_sidecar_back(qapp, tmp_path):
    win = open_window(tmp_path)
    instance = start_edit(win)
    paint(win)
    win.flush_sidecar()
    painted = win.session.editing_mask().copy()
    win.shutdown()

    again = MainWindow(make_session(tmp_path), make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        assert again.pending_restore() is not None      # offered on the frame
        again._restore_offer = None                     # the annotator ignores it
        again.restore_bar.hide()
        again.on_request_edit(instance)                 # ... and on begin_edit
        assert again.pending_restore() is not None
        again.restore_pending()
        assert np.array_equal(again.session.editing_mask(), painted)
    finally:
        again.shutdown()


def test_a_sidecar_for_an_instance_that_is_gone_is_dropped(qapp, tmp_path):
    win = open_window(tmp_path)
    start_edit(win)
    paint(win)
    win.flush_sidecar()
    win.sidecar.save(win.session.current(), "ghost.99", np.ones((64, 64), dtype=bool))
    win.shutdown()

    again = MainWindow(make_session(tmp_path), make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        offer = again.pending_restore()
        assert offer is None or offer["instance"] != "ghost.99"
        assert again.sidecar.pending_for(again.session.current(), "ghost.99") is None
    finally:
        again.shutdown()


# --------------------------------------------------------------------------- #
# the lock is taken before the database is opened
# --------------------------------------------------------------------------- #
def test_a_held_lock_leaves_the_database_file_untouched(qapp, tmp_path):
    from app_scene import write_paths_yaml
    from tda.ui import app as app_module

    db, paths, _tax = make_db(tmp_path)
    db.close()
    acquire_lock_file(paths["db_path"], "someone_else")
    before = Path(paths["db_path"]).stat()
    code = app_module.main(paths=write_paths_yaml(tmp_path), desktop=DESKTOP,
                           view=VIEW, annotator="tester", exec_=False)
    after = Path(paths["db_path"]).stat()
    assert code == 3
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)
    assert read_lock_file(paths["db_path"])["annotator"] == "someone_else"


def test_the_lock_is_released_when_the_window_cannot_be_built(qapp, tmp_path,
                                                              monkeypatch):
    from app_scene import write_paths_yaml
    from tda.ui import app as app_module

    db, paths, _tax = make_db(tmp_path)
    db.close()
    monkeypatch.setattr(app_module, "MainWindow",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        app_module.main(paths=write_paths_yaml(tmp_path), desktop=DESKTOP, view=VIEW,
                        annotator="tester", exec_=False)
    assert not lock_path_for(paths["db_path"]).exists()


# --------------------------------------------------------------------------- #
# the ROI accessor
# --------------------------------------------------------------------------- #
def test_clean_roi_validates_and_clamps():
    assert clean_roi([1.4, 2.6, 30, 40]) == [1, 3, 30, 40]
    assert clean_roi([-5, -5, 10_000, 10_000], hw=(64, 64)) == [0, 0, 64, 64]
    for bad in ([1, 2, 3], [10, 2, 3, 40], [10, 40, 30, 20], ["a", 1, 2, 3]):
        with pytest.raises(ValueError):
            clean_roi(bad)


def test_set_pose_segment_roi_writes_validates_and_logs(qapp, tmp_path):
    db, _paths, _tax = make_db(tmp_path)
    try:
        stored = db.set_pose_segment_roi(DESKTOP, VIEW, 1, [10, 12, 40, 44], "tester")
        assert stored == [10, 12, 40, 44]
        assert db.pose_segment_for(FrameKey(DESKTOP, LAST_STEP, VIEW))["roi"] == stored
        logged = [op for op in db.ops(DESKTOP, VIEW) if op["kind"] == "set_pose_roi"]
        assert logged and logged[0]["payload"]["roi"] == stored
        assert logged[0]["inverse"]["roi"] is None
        with pytest.raises(ValueError):
            db.set_pose_segment_roi(DESKTOP, VIEW, 1, [40, 12, 10, 44], "tester")
        with pytest.raises(ValueError):
            db.set_pose_segment_roi(DESKTOP, VIEW, 99, [1, 2, 3, 4], "tester")
        assert db.set_pose_segment_roi(DESKTOP, VIEW, 1, None, "tester") is None
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the keyboard
# --------------------------------------------------------------------------- #
def test_an_auto_repeated_bound_key_is_consumed_not_passed_on(window):
    from PySide6.QtGui import QKeyEvent

    seen: list[bool] = []
    window.act_flash_compare = lambda on: seen.append(on)
    held = QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_Tab),
                     Qt.KeyboardModifier.NoModifier, "", True)
    assert window.handle_key(held) is True      # consumed: Tab must not move focus
    assert seen == []                           # ... but not fired again


def test_an_auto_repeated_unbound_key_is_left_alone(window):
    from PySide6.QtGui import QKeyEvent

    held = QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_M),
                     Qt.KeyboardModifier.NoModifier, "", True)
    assert window.handle_key(held) is False


def test_shortcuts_are_off_while_a_modal_dialog_is_up(window, monkeypatch):
    from PySide6.QtWidgets import QDialog

    dialog = QDialog(window)
    dialog.setModal(True)
    dialog.show()
    QApplication.processEvents()
    monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: dialog))
    calls: list = []
    window.act_tool = lambda *a: calls.append(a)
    from test_app_actions import key_event

    assert window.handle_key(key_event("B")) is False
    assert calls == []
    dialog.close()


# --------------------------------------------------------------------------- #
# window state round trip
# --------------------------------------------------------------------------- #
def test_the_window_state_survives_a_restart(qapp, tmp_path):
    from tda.ui.app_shell import DEFAULT_WINDOW_SIZE

    win = open_window(tmp_path)
    win.resize(1101, 717)
    QApplication.processEvents()
    win.session.goto(LAST_STEP - 3)
    win.save_window_state()
    geometry = bytes(win.saveGeometry())
    win.shutdown()

    again = MainWindow(make_session(tmp_path), make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        assert bytes(again.settings.value("geometry")) == geometry
        assert (again.width(), again.height()) != DEFAULT_WINDOW_SIZE
        assert again.height() == win.height()
        assert again.last_frame_for("tester") == {
            "desktop": DESKTOP, "view": VIEW, "step": LAST_STEP - 3
        }
    finally:
        again.shutdown()


# --------------------------------------------------------------------------- #
# whole sequences
# --------------------------------------------------------------------------- #
def test_sequence_a_draw_commit_undo_redo_restores_the_same_pixels(window):
    """(a) what redo puts back is exactly what was committed."""
    session = window.session
    instance = start_edit(window)
    paint(window)
    painted = window.session.editing_mask().copy()
    window.act_commit()
    stored = session.db.keyframes(DESKTOP, VIEW, instance)
    assert stored
    window.act_undo()
    assert not session.db.keyframes(DESKTOP, VIEW, instance)
    window.act_redo()
    again = session.db.keyframes(DESKTOP, VIEW, instance)
    assert len(again) == len(stored)
    from tda.core import masks as _masks

    restored = _masks.decode_rle(again[-1].parts[0].rle)
    assert np.array_equal(restored, painted)


def test_sequence_b_sam_then_brush_then_commit_then_confirm(window):
    """(b) the normal loop: propose, fix by hand, commit, confirm."""
    session = window.session
    seed_shapes(session, LAST_STEP)
    session.goto(LAST_STEP)
    instance = drawable_instances(window)[0]
    window.task_card.sigRequestEdit.emit(instance)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush()
    QApplication.processEvents()
    assert window.overlay.editing.any()
    window.act_tool("brush")
    paint(window)
    window.act_commit()
    assert session.db.keyframes(DESKTOP, VIEW, instance)
    assert window.act_confirm() in (True, False)   # whatever the compiler says
    assert window.session.editing_instance is None


def test_sequence_c_the_scope_bar_is_read_then_accepted(window, monkeypatch):
    """(c) a suggestion that is not a plain keyframe is shown before it is written."""
    session = window.session
    start_edit(window)
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:above:other")
    committed: list[str] = []
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})
    window.act_commit()
    text = window.scope_bar_text()
    assert "zorder:above:other" in text
    assert "影响" not in text          # a layering statement has no frame reach
    window.act_commit()
    assert committed == ["zorder:above:other"]


# --------------------------------------------------------------------------- #
# review mode is read-only on the canvas
# --------------------------------------------------------------------------- #
def test_review_mode_arms_no_tool_and_a_press_does_nothing(window):
    """An edit begun in Review mode could never be settled: so it cannot begin."""
    window.set_mode(A.MODE_REVIEW)
    assert window.active_tool is None
    paint(window)
    assert window.session.editing_instance is None
    assert not window.overlay.editing.any()
    assert "R" in window.status_message()


def test_rework_moves_the_frame_into_annotate_mode(window):
    window.set_mode(A.MODE_REVIEW)
    target = min(window.session.steps())
    window.on_rework(target)
    assert window.mode == A.MODE_ANNOTATE
    assert window.session.current().step == target
    assert window.active_tool is window.brush


def test_review_enter_goes_through_the_window_action(window, monkeypatch):
    """The panel's own Enter bypassed the unexplained hand-over."""
    seen: list[int] = []
    monkeypatch.setattr(window, "act_confirm", lambda: seen.append(1) or False)
    window.set_mode(A.MODE_REVIEW)
    from test_app_actions import key_event

    assert window.handle_key(key_event("Return")) is True
    assert seen == [1]


def test_the_brush_ignores_the_right_button(window):
    """The right button is the SAM negative point and the context menu."""
    start_edit(window)
    window.act_tool("brush")
    viewport = window.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.RightButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseRelease(viewport, Qt.MouseButton.RightButton,
                       Qt.KeyboardModifier.NoModifier, centre)
    QApplication.processEvents()
    assert not window.overlay.editing.any()


def test_alt_click_is_a_negative_sam_point(window):
    """A trackpad has no comfortable right button; ``Alt`` + click is the same."""
    start_edit(window)
    window.act_tool("sam_point")

    class _Alt:
        def button(self):
            return Qt.MouseButton.LeftButton

        def modifiers(self):
            return Qt.KeyboardModifier.AltModifier

    window.sam_point.on_press(20.0, 20.0, _Alt())
    assert window.sam_point.points[-1][2] == 0
    window.sam_point.on_press(21.0, 21.0, None)
    assert window.sam_point.points[-1][2] == 1


def test_middle_drag_pans_the_canvas_in_annotate_mode(window):
    """``Space`` is confirm, so panning has to live on the middle button."""
    start_edit(window)
    window.act_tool("brush")
    window.canvas.set_zoom(32.0)   # far enough in that the viewport is a window
    QApplication.processEvents()
    before = window.canvas.viewport_image_rect()
    assert before[2] - before[0] < 64
    viewport = window.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.MiddleButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(60, 40))
    QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(60, 40))
    QApplication.processEvents()
    assert window.canvas.viewport_image_rect() != before
    assert not window.overlay.editing.any()      # a pan is not a stroke


def test_sequence_d_blocked_navigation_then_commit_then_navigation(window):
    """(d) the block is a hint, not a wall: commit and the step change happens."""
    start_edit(window)
    paint(window)
    step = window.session.current().step
    window.act_step(-1)
    assert window.session.current().step == step
    window.act_commit()
    window.act_step(-1)
    assert window.session.current().step < step
