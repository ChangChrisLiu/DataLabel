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
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

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
from tda.core.db import acquire_lock_file, lock_path_for, read_lock_file
from tda.core.db_pose import clean_roi
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_edit import FLASH_HINT as A_FLASH_HINT


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
    close_window(win)


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
    # Shift+R armed the ROI tool over an uncommitted layer: Esc then discarded
    # the *pixels* and Enter committed them, while the bar said "Enter 确认"
    # about a rectangle.
    "edit_roi": lambda w: w.act_edit_roi(),
}


def click(button) -> None:
    """A real mouse click, not ``button.click()``: the reviewer's probe."""
    QTest.mouseClick(button, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, button.rect().center())
    QApplication.processEvents()


#: "Confirm" steps the frame back, so the button is a way out like the key is.
#: The reviewer's probe clicked it with 250 uncommitted pixels and watched the
#: frame move from 14 to 13 with nothing written and nothing said.
WAYS_OUT["button_confirm"] = lambda w: click(w.task_card.confirm_button)


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


# --------------------------------------------------------------------------- #
# the panel buttons are the window's actions, not their own
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("button,key", [
    ("commit_button", "act_commit"),
    ("override_button", "act_commit_override"),
    ("split_button", "act_commit_split"),
    ("confirm_button", "act_confirm"),
])
def test_each_task_card_button_is_its_key(window, monkeypatch, button, key):
    """One label, one meaning: the button calls exactly what the key calls."""
    seen: list[str] = []
    monkeypatch.setattr(window, key, lambda *a: seen.append(key) or False)
    click(getattr(window.task_card, button))
    assert seen == [key]


def test_the_commit_button_honours_the_suggested_scope(window, monkeypatch):
    """It used to hard-code ``keyframe`` while ``Enter`` asked the session."""
    session = window.session
    start_edit(window)
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: api.SCOPE_SPLIT)
    committed: list[str] = []
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})
    click(window.task_card.commit_button)
    # exactly what Enter does: a suggestion that is not a plain keyframe is
    # shown first, and the next press accepts it
    assert committed == [] and api.SCOPE_SPLIT in window.scope_bar_text()
    click(window.task_card.commit_button)
    assert committed == [api.SCOPE_SPLIT]


def test_a_button_commit_clears_the_layer_and_the_sidecar(window):
    """The panel's own call skipped every bit of the window's post-processing."""
    instance = drawable_instances(window)[0]
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    window.flush_sidecar()
    key = window.session.current()
    assert Path(window.sidecar.path_for(key, instance)).exists()
    click(window.task_card.commit_button)
    assert window.session.editing_instance is None
    assert not window.overlay.editing.any()
    assert not Path(window.sidecar.path_for(key, instance)).exists()
    assert window.pending_restore() is None


def test_the_instance_move_buttons_go_through_the_window(window, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(window, "move_instance", lambda d: seen.append(d))
    click(window.instances.up_button)
    click(window.instances.down_button)
    assert seen == [-1, +1]


def test_the_hidden_checkbox_goes_through_the_window(window):
    session = window.session
    rows = session.instance_rows()
    if not rows:
        pytest.skip("no instances on this frame")
    key = str(rows[0]["key"])
    table = window.instances.table()
    column = window.instances.COLUMNS.index("Hidden")
    table.item(0, column).setCheckState(Qt.CheckState.Checked)
    QApplication.processEvents()
    assert key in session._hidden
    assert window.overlay is not None        # the overlay was rebuilt without it


def test_activating_a_task_item_outside_annotate_mode_does_nothing(window):
    """Qt's own itemActivated fires in Steps mode, where the canvas is hidden."""
    window.set_mode(A.MODE_STEPS)
    instance = card_instances(window)[0]
    window.task_card.sigRequestEdit.emit(instance)
    assert window.session.editing_instance is None
    assert "Annotate" in window.status_message() or "标注" in window.status_message()


def test_a_bench_box_item_arms_the_box_tool_not_the_brush(window, monkeypatch):
    """A part on the bench is boxed, so no mask layer is opened for it at all.

    Going through ``begin_edit`` only ended in a refusal at commit time, with
    the annotator's strokes already on the canvas.
    """
    kind = getattr(api, "KIND_ADD_BENCH_BOX", None)
    if kind is None:
        pytest.skip("the session has no add_bench_box kind yet")
    instance = "cpu_cooler.fan.01"
    monkeypatch.setattr(window.session, "task_card", lambda: [
        {"instance": instance, "kind": kind, "text": "box it", "done": False},
    ])
    window.task_card.refresh()
    window.task_card.sigRequestEdit.emit(instance)

    assert window.session.editing_instance is None      # no mask layer
    assert window.active_tool is window.bench_tool
    assert window.bench_instance == instance
    assert "台面框" in window.status_message() or "staging" in window.status_message()

    boxed: list[tuple] = []
    monkeypatch.setattr(window.session, "commit_box",
                        lambda key, box, *a, **k: boxed.append((key, tuple(box))) or {})
    window.on_bench_box((10.0, 12.0, 40.0, 44.0))
    assert boxed == [(instance, (10.0, 12.0, 40.0, 44.0))]
    assert window.bench_instance is None


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


def test_the_session_refuses_to_confirm_a_frame_with_an_uncommitted_layer(window):
    """``confirm_frame`` stepped back with ``force``, trusting a caller that may not check."""
    from tda.ui.session_api import SessionRefusal

    session = window.session
    start_edit(window)
    paint(window)
    step = session.current().step
    with pytest.raises(SessionRefusal):
        session.confirm_frame()
    assert session.current().step == step
    assert session.frame_status(step) != "verified"


def test_a_layering_commit_the_other_way_round_does_not_swallow_pixels(window):
    """``zorder:below:`` writes no pixels, so it must not mark any as written.

    Erasing where ``B`` lies under ``A`` says "B should have been on top": the
    pixels stay in ``A``'s amodal shape, they are merely hidden.  A direct API
    user is therefore left holding the mask edit rather than having it silently
    settled; the *window* clears it, because there the erasure was only ever the
    gesture that made the session suggest layering.
    """
    session = window.session
    instance = drawable_instances(window)[0]
    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    other = [i for i in drawable_instances(window) if i != instance]
    if not other:
        pytest.skip("need a second instance to layer against")
    session.commit_edit(f"zorder:below:{other[0]}")
    assert session.layer.changed()          # still the annotator's to settle

    window.task_card.sigRequestEdit.emit(instance)
    paint(window)
    window._commit(f"zorder:below:{other[0]}")
    assert session.editing_instance is None  # the window settles it by clearing


# --------------------------------------------------------------------------- #
# the bench arm is not allowed to outlive the gesture that set it
# --------------------------------------------------------------------------- #
def arm_bench(window, monkeypatch, instance: str) -> None:
    """Activate an ``add_bench_box`` card item for ``instance`` (what R + a card does)."""
    kind = getattr(api, "KIND_ADD_BENCH_BOX", None)
    if kind is None:  # pragma: no cover - the session always has it now
        pytest.skip("the session has no add_bench_box kind yet")
    monkeypatch.setattr(window.session, "task_card", lambda: [
        {"instance": instance, "kind": kind, "text": "box it", "done": False},
    ])
    window.task_card.refresh()
    window.task_card.sigRequestEdit.emit(instance)
    assert window.bench_instance == instance


LOSES_THE_ARM = {
    "frame": lambda w: w.act_step(-1),
    "view": lambda w: w.act_set_view("oak1"),
    "desktop": lambda w: w.act_set_desktop(DESKTOP),
    "mode": lambda w: w.set_mode(A.MODE_REVIEW),
    "tool": lambda w: w.act_tool("brush"),
    "escape": lambda w: w.act_clear_edit(),
}


@pytest.mark.parametrize("name", sorted(LOSES_THE_ARM), ids=sorted(LOSES_THE_ARM))
def test_the_bench_arm_does_not_outlive_the_frame_it_was_set_on(window, monkeypatch, name):
    """Armed on step 14, dragged on step 13: the box was stored on the wrong frame.

    ``bench_instance`` was only ever cleared by a *successful* drag, so it rode
    along through frame, view, desktop and mode changes and through every tool
    switch, waiting to claim somebody else's rectangle.
    """
    instance = "cpu_cooler.fan.01"
    arm_bench(window, monkeypatch, instance)
    LOSES_THE_ARM[name](window)
    assert window.bench_instance is None, f"{name} left the bench tool armed"


def test_a_bench_drag_with_nothing_armed_needs_a_part_on_the_bench(window, monkeypatch):
    """``R`` + a drag on a chassis part is a mistake, not a box for it."""
    rows = [{"key": "cpu_cooler.fan.01", "placement": "in_chassis", "state": "",
             "visibility": "visible", "z": 0, "cls": "", "hidden": False}]
    monkeypatch.setattr(window.session, "instance_rows", lambda: rows)
    window.instances.refresh()
    window.instances.select_instance("cpu_cooler.fan.01")
    boxed: list = []
    monkeypatch.setattr(window.session, "commit_box",
                        lambda *a, **k: boxed.append(a) or {})

    window.on_bench_box((10.0, 12.0, 40.0, 44.0))

    assert boxed == []
    assert "bench" in window.status_message() or "台面" in window.status_message()


def test_a_bench_drag_with_nothing_armed_boxes_the_selected_bench_part(window, monkeypatch):
    """A part the frame says is on the bench is exactly what ``R`` is for."""
    rows = [{"key": "cpu_cooler.fan.01", "placement": "on_bench", "state": "",
             "visibility": "visible", "z": 0, "cls": "", "hidden": False}]
    monkeypatch.setattr(window.session, "instance_rows", lambda: rows)
    window.instances.refresh()
    window.instances.select_instance("cpu_cooler.fan.01")
    boxed: list = []
    monkeypatch.setattr(window.session, "commit_box",
                        lambda key, box, *a, **k: boxed.append((key, tuple(box))) or {})

    window.on_bench_box((10.0, 12.0, 40.0, 44.0))

    assert boxed == [("cpu_cooler.fan.01", (10.0, 12.0, 40.0, 44.0))]


# --------------------------------------------------------------------------- #
# one owner of Enter and Esc at a time
# --------------------------------------------------------------------------- #
def test_enter_stores_the_roi_while_the_roi_bar_is_up(window):
    """The bar says "Enter 确认" about a rectangle, so Enter must mean that."""
    start_edit(window)                       # a clean layer, nothing painted
    window.act_edit_roi()
    assert window.roi_editing
    window.roi_draft = (4, 4, 40, 40)

    window.act_commit()

    assert window.roi_editing is False
    assert window.roi() == (4, 4, 40, 40)


def test_esc_drops_the_roi_rectangle_while_the_roi_bar_is_up(window):
    """... and Esc means the rectangle too, not the instance loaded behind it."""
    instance = start_edit(window)
    stored = window.roi()
    window.act_edit_roi()
    window.roi_draft = (4, 4, 40, 40)

    window.act_clear_edit()

    assert window.roi_editing is False
    assert window.roi() == stored
    assert window.session.editing_instance == instance


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
    close_window(win)

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
        close_window(again)


def test_a_sidecar_for_an_instance_that_is_gone_is_dropped(qapp, tmp_path):
    win = open_window(tmp_path)
    start_edit(win)
    paint(win)
    win.flush_sidecar()
    win.sidecar.save(win.session.current(), "ghost.99", np.ones((64, 64), dtype=bool))
    close_window(win)

    again = MainWindow(make_session(tmp_path), make_paths(tmp_path), "tester",
                       sam_queue=StubSamQueue())
    try:
        offer = again.pending_restore()
        assert offer is None or offer["instance"] != "ghost.99"
        assert again.sidecar.pending_for(again.session.current(), "ghost.99") is None
    finally:
        close_window(again)


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
    close_window(win)

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
        close_window(again)


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
    if window.warn_bar.isVisibleTo(window):
        window.act_commit()      # a synthetic 64x64 part trips the area prior
    assert session.db.keyframes(DESKTOP, VIEW, instance)
    assert window.act_confirm() in (True, False)   # whatever the compiler says
    assert window.session.editing_instance is None


def test_sequence_c_the_scope_bar_is_read_then_accepted(window, monkeypatch):
    """(c) a suggestion that is not a plain keyframe is shown before it is written."""
    session = window.session
    start_edit(window)
    paint(window)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:above:other")
    monkeypatch.setattr(session, "preview",
                        lambda scope, *a, **k: {"steps": [11, 12, 13],
                                                "verified_steps": [12]})
    committed: list[str] = []
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})
    window.act_commit()
    text = window.scope_bar_text()
    assert "zorder:above:other" in text
    # a layering statement reaches frames too, and the strip says how many
    assert "影响 3 帧" in text and "1 个冲突" in text
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


# --------------------------------------------------------------------------- #
# while Tab is held the canvas is showing another frame (final review, item 2)
# --------------------------------------------------------------------------- #
def flashable(win: MainWindow) -> MainWindow:
    """Stand on a frame that *has* a neighbour to flash (the last one has none)."""
    win.session.goto(LAST_STEP - 1, force=True)
    QApplication.processEvents()
    return win


def test_a_press_while_flashing_is_swallowed_with_a_hint(window):
    """The canvas is showing the neighbour; a stroke there would be a lie.

    In reverse order the part the card asks for is ABSENT in j+1, so a SAM
    click on the flashed image cropped pixels that do not contain it and the
    mask landed in the *current* frame's layer, confidently wrong.
    """
    flashable(window)
    instance = start_edit(window)
    window.act_flash_compare(True)
    assert window.is_flashing() is True
    before = window.overlay.editing.copy()

    paint(window)

    assert np.array_equal(window.overlay.editing, before), "a stroke landed while flashed"
    assert "Tab" in window.status_message()


def test_sam_is_refused_while_flashing(window):
    flashable(window)
    start_edit(window)
    window.act_tool("sam_point")
    window.act_flash_compare(True)
    pending = window.sam_queue.pending()

    window.sam_point.on_press(32.0, 32.0, None)

    assert window.sam_queue.pending() == pending, "a prompt went out on the wrong image"


def test_the_status_bar_says_which_frame_is_being_compared(window):
    """What is on the canvas is not what the rest of the window is about."""
    flashable(window)
    window.act_flash_compare(True)
    assert "对照" in window.frame_label.text()
    window.act_flash_compare(False)
    assert "对照" not in window.frame_label.text()
    assert str(window.session.current().step) in window.frame_label.text()


def test_a_lost_key_release_does_not_leave_the_canvas_on_the_neighbour(window):
    """Alt+Tab while holding Tab: the release never arrives.

    The canvas stayed on the neighbour's image with the current frame's overlay
    and status -- and every tool stayed live over it.
    """
    flashable(window)
    window.act_flash_compare(True)
    assert window.is_flashing() is True

    QApplication.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    QApplication.processEvents()

    assert window.is_flashing() is False
    assert np.array_equal(window.canvas.image_rgb(), window.session.image())


def test_any_other_key_ends_the_flash(window):
    flashable(window)
    window.act_flash_compare(True)
    window.handle_key(_key_event(Qt.Key.Key_B))
    assert window.is_flashing() is False


def test_a_frame_change_ends_the_flash(window):
    flashable(window)
    window.act_flash_compare(True)
    window.session.goto(min(window.session.steps()), force=True)
    QApplication.processEvents()
    assert window.is_flashing() is False


def _key_event(key, press: bool = True):
    from PySide6.QtGui import QKeyEvent

    return QKeyEvent(
        QEvent.Type.KeyPress if press else QEvent.Type.KeyRelease,
        key, Qt.KeyboardModifier.NoModifier,
    )


# --------------------------------------------------------------------------- #
# every failure has to reach the annotator (item 12)
# --------------------------------------------------------------------------- #
def test_a_tool_that_raises_does_not_escape_into_qt(window, monkeypatch):
    """``queue.submit`` raising left the label saying "SAM ready" for ever.

    The three tool slots were unguarded, so the exception went into the Qt
    event loop, the status bar said nothing, and every further click repeated
    it.
    """
    start_edit(window)
    window.act_tool("sam_point")

    def explode(*_a, **_k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(window.sam_queue, "submit", explode)
    viewport = window.canvas.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, viewport.rect().center())
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, viewport.rect().center())
    QApplication.processEvents()

    assert "CUDA out of memory" in window.last_error_message()
    assert "error" in window.sam_label.text().lower()
    assert "S" in window.sam_label.text()


def test_an_exception_in_a_slot_is_reported_as_an_error(window):
    """``report_exception`` wrote to the hint line only, so nothing recorded it."""
    window.report_exception(RuntimeError("boom"), "act_commit")
    assert "boom" in window.last_error_message()


def test_a_locked_database_is_explained_in_words(window):
    import sqlite3

    window.report_exception(sqlite3.OperationalError("database is locked"),
                            "act_commit")
    assert "数据库正忙" in window.last_error_message()


def test_a_failed_sidecar_write_is_said_once_and_survives_the_block(window, monkeypatch):
    """The crash protection failing is worse news than the edit being blocked."""
    instance = start_edit(window)

    def explode(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(window.sidecar, "save", explode)
    paint(window)
    window.flush_sidecar()
    assert "崩溃保护已失效" in window.status_message()

    window.act_step(-1)          # blocked: BLOCK_HINT used to overwrite it
    assert "崩溃保护已失效" in window.status_message()

    said = window.status_message()
    window.flush_sidecar()
    assert window.status_message() == said     # not repeated every stroke


def test_a_step_with_no_image_says_so_on_arrival(qapp, tmp_path):
    """The placeholder is silent; the status bar has to say which step."""
    win = open_window(tmp_path, missing=(3,))
    try:
        win.session.goto(3, force=True)
        QApplication.processEvents()
        assert win.session.image() is None
        assert "没有图像" in win.status_message() or "no image" in win.status_message()
    finally:
        close_window(win)


def test_editing_keys_on_a_removed_row_say_why_nothing_happened(window):
    """A removed part is listed read-only; H / V / 1-7 must not be silent."""
    window.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    table = window.instances.table()
    if table.rowCount() <= len(window.instances.rows()):
        pytest.skip("this frame has no removed parts")
    table.setCurrentCell(table.rowCount() - 1, 1)

    window.act_toggle_hidden()

    assert "已移除" in window.status_message()


# --------------------------------------------------------------------------- #
# the small ones (F3 round 2, item 6)
# --------------------------------------------------------------------------- #
def test_the_flash_hint_is_not_overwritten_by_a_late_assist_line(window):
    """The diff map answers a few ms later and took the hint off the screen."""
    flashable(window)
    start_edit(window)
    window.act_flash_compare(True)
    paint(window)                      # the hint appears
    assert "Tab" in window.status_message()

    window.report("prompt box from the difference map: (1, 2, 3, 4)")

    assert "Tab" in window.status_message(), "the hint was overwritten"


def test_a_held_hint_still_gives_way_to_a_frame_change(window):
    flashable(window)
    window.act_flash_compare(True)
    window.report(A_FLASH_HINT, hold_ms=5000)
    window.session.goto(min(window.session.steps()), force=True)
    QApplication.processEvents()
    assert "Tab" not in window.status_message()


def test_switching_to_the_brush_says_the_candidates_go(window):
    """``detach()`` drops them, which the SAM label's docstring did not admit."""
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush(multimask=True)
    QApplication.processEvents()
    assert window.sam_point.candidate_count == 3

    window.act_tool("brush")

    assert "候选" in window.status_message()
    assert window.sam_point.candidate_count == 0


def test_the_zoom_slot_cannot_escape_into_qt(window, monkeypatch):
    """Every slot the window connects goes through the guard, this one too."""
    def explode() -> None:
        raise RuntimeError("status went wrong")

    monkeypatch.setattr(window, "update_status", explode)
    window.canvas.set_zoom(2.0)        # emits sigZoomChanged
    QApplication.processEvents()
    assert "status went wrong" in window.last_error_message()


def test_an_error_is_never_held_back_by_a_hint(window):
    """The hold protects a hint from a background line, not from bad news."""
    window.report(A_FLASH_HINT, hold_ms=5000)
    window.report_error("SAM failed: out of memory")
    assert "out of memory" in window.status_message()
    assert "out of memory" in window.last_error_message()


def test_one_candidate_is_not_worth_a_goodbye(window):
    """``C`` offers nothing with a single proposal, so losing it says nothing."""
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush(multimask=False)       # one candidate
    QApplication.processEvents()
    assert window.sam_point.candidate_count <= 1

    window.act_tool("brush")

    assert "候选" not in window.status_message()
