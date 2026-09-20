"""Main window: layout, lifecycle, frame change and the status bar (task 13b).

Everything runs offscreen against the temporary D13 scene of
:mod:`tests.app_scene`; the SAM queue is the stub, so no checkpoint is loaded
and no GPU is touched.  The editing, ROI, review and crash-safety flows are in
:mod:`tests.test_app_edit`.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from app_scene import (
    close_window,
    DESKTOP,
    LAST_STEP,
    SEGMENT_CUT,
    VIEW,
    StubSamQueue,
    make_db,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.core.model import FrameKey
from tda.core.truth import TruthService
from tda.ui import app_actions as A
from tda.ui import app_support as S
from tda.ui.app import MainWindow, take_lock
from tda.ui.session import AnnotationSession


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, **kwargs) -> MainWindow:
    session = make_session(tmp_path, **kwargs)
    return MainWindow(session, make_paths(tmp_path), "tester",
                      sam_queue=StubSamQueue())


@pytest.fixture
def window(qapp, tmp_path):
    win = open_window(tmp_path)
    yield win
    close_window(win)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_window_shows_the_first_frame_and_its_overlay(window):
    key = window.session.current()
    assert key == FrameKey(DESKTOP, LAST_STEP, VIEW)
    assert window.canvas.image_rgb() is not None
    assert window.canvas.image_rgb().shape == (64, 64, 3)
    assert window.overlay is not None and window.overlay.hw == (64, 64)
    assert window.canvas.overlay() is window.overlay


def test_top_bar_lists_the_desktops_and_the_four_views(window):
    assert window.desktop_combo.count() >= 1
    assert window.desktop_combo.currentData() == DESKTOP
    assert f"D{DESKTOP}" in window.desktop_combo.currentText()
    assert set(window.view_buttons) == {"scan", "oak1", "oak2", "rs"}
    assert window.view_buttons[VIEW].isChecked() is True
    assert [window.mode_tabs.tabText(i) for i in range(window.mode_tabs.count())] == [
        "Steps", "Annotate", "Review"
    ]


def test_status_bar_reports_zoom_frame_tool_and_sam(window):
    assert window.zoom_label.text().endswith("%")
    text = window.frame_label.text()
    assert f"D{DESKTOP}" in text and VIEW in text
    assert f"step {LAST_STEP}/{LAST_STEP}" in text
    # Bilingual and keyed, not ``brush``: the annotator reads Chinese and the
    # key is how they get back to the tool (task U1, ruling R1).  On a fresh
    # segment the ROI rectangle owns the canvas, and *that* is what the badge
    # has to say -- a "brush" badge over a rectangle is the confusion itself.
    assert "ROI" in window.tool_label.text()
    window.act_clear_edit()            # Esc: skip the rectangle for now
    assert window.tool_label.text().startswith("工具：")
    assert "画笔" in window.tool_label.text()
    assert " B" in window.tool_label.text(), window.tool_label.text()
    assert "Brush" in window.tool_label.toolTip()
    assert window.sam_label.text()
    assert window.roi_label.text(), "the ROI state has to be on the status bar"


def test_mode_switch_moves_the_central_widget(window):
    window.set_mode(A.MODE_ANNOTATE)
    assert window.stack.currentWidget() is window.canvas
    window.set_mode(A.MODE_STEPS)
    assert window.stack.currentWidget() is window.steps_panel
    window.set_mode(A.MODE_REVIEW)
    assert window.stack.currentWidget() is window.canvas
    assert window.review_dock.isVisibleTo(window) or window.review_dock.isVisible()


@pytest.mark.parametrize("width,share", [(1920, 0.65), (1600, 0.60)])
def test_the_canvas_gets_most_of_the_window_by_default(qapp, tmp_path, width, share):
    """With no saved state the docks must not crowd out the frame."""
    win = open_window(tmp_path)
    try:
        win.resize(width, 1080)
        win.show()
        QApplication.processEvents()
        canvas = win.canvas.width()
        assert canvas >= share * win.width(), (
            f"canvas {canvas} px of {win.width()}; "
            f"timeline {win.timeline_dock.width()}, right {win.right_dock.width()}"
        )
    finally:
        close_window(win)


def test_every_dock_panel_can_be_narrow(qapp, tmp_path):
    """Their minimum width is what clamped ``resizeDocks``; the review panel's
    ``Enter: confirm frame   R: rework`` label alone was 384 px."""
    win = open_window(tmp_path)
    try:
        for panel in (win.task_card, win.instances, win.review, win.timeline):
            assert panel.minimumSizeHint().width() <= 320, type(panel).__name__
    finally:
        close_window(win)


@pytest.mark.parametrize("width,share", [(1920, 0.65), (1600, 0.60)])
def test_a_trip_through_review_mode_gives_the_canvas_back(qapp, tmp_path, width, share):
    """Entering Review once pushed the right dock to 642 px and it never returned."""
    win = open_window(tmp_path)
    try:
        win.resize(width, 1080)
        win.show()
        QApplication.processEvents()
        before = win.canvas.width()
        win.set_mode(A.MODE_REVIEW)
        QApplication.processEvents()
        win.set_mode(A.MODE_ANNOTATE)
        QApplication.processEvents()
        assert win.canvas.width() >= share * win.width(), (
            f"canvas {win.canvas.width()} px of {win.width()} after a Review trip "
            f"(was {before}); right dock {win.right_dock.width()}"
        )
    finally:
        close_window(win)


def _save_a_layout(tmp_path: Path, right: int) -> None:
    """Leave a dock layout with the right dock ``right`` px wide in the INI."""
    win = open_window(tmp_path)
    try:
        win.resize(1920, 1080)
        win.show()
        QApplication.processEvents()
        win.resizeDocks([win.right_dock], [right], Qt.Orientation.Horizontal)
        QApplication.processEvents()
        win.save_window_state()
    finally:
        close_window(win)


def _reopen_at(tmp_path: Path, width: int = 1920) -> MainWindow:
    """Re-open the window with the saved layout restored at a real desktop width.

    ``restoreGeometry`` is clamped to the offscreen screen (798 px), which would
    restore the docks at a size no desktop ever has; the state is therefore
    restored once more at the width the annotator really runs at.
    """
    win = MainWindow(make_session(tmp_path), make_paths(tmp_path), "tester",
                     sam_queue=StubSamQueue())
    win.resize(width, 1080)
    win.restoreState(win.settings.value("state"))
    win.show()
    QApplication.processEvents()
    return win


def test_the_layout_guard_waits_until_the_canvas_has_a_real_width(qapp, tmp_path):
    """It must be asked once, after the layout, and never from ``__init__``.

    ``restore_window_state`` runs before ``show()``, where the canvas still
    reports the ``QWidget`` default of 640 px.  Measured against a restored
    1920 px window that is under half, so on every real desktop the guard fired
    at every launch: the annotator's own dock layout was discarded each time and
    the status bar blamed it.  Offscreen, ``restoreGeometry`` is clamped to the
    798 px screen and hides the symptom, so what is checked here is that the
    early call is gone and the ``showEvent`` one is the only one left.
    """
    _save_a_layout(tmp_path, right=500)
    seen: list[tuple[int, int]] = []
    original = MainWindow._reject_a_starved_canvas

    def spy(self) -> None:
        seen.append((self.width(), self.canvas.width()))
        original(self)

    MainWindow._reject_a_starved_canvas = spy
    try:
        again = _reopen_at(tmp_path, 1920)
    finally:
        MainWindow._reject_a_starved_canvas = original
    try:
        assert len(seen) == 1, f"the guard ran {len(seen)} times: {seen}"
        width, canvas = seen[0]
        assert canvas * 2 > width, f"judged an unlaid-out canvas: {canvas} of {width}"
    finally:
        close_window(again)


@pytest.mark.parametrize("right,kept", [(500, True), (1500, False)])
def test_only_a_layout_that_starves_the_canvas_is_reset(qapp, tmp_path, right, kept):
    """A poisoned INI must not follow the annotator around -- and a healthy one must.

    500 px of right dock at 1920 leaves the canvas 62 %: the annotator widened
    it on purpose and it has to survive the restart.  1500 px leaves 12 %, which
    nobody chose.
    """
    _save_a_layout(tmp_path, right=right)
    again = _reopen_at(tmp_path, 1920)
    try:
        canvas = again.canvas.width()
        if kept:
            assert abs(again.right_dock.width() - right) <= 40, "the layout was reset"
            assert canvas >= 0.55 * again.width()
        else:
            assert canvas >= 0.5 * again.width(), (
                f"canvas {canvas} of {again.width()}; right {again.right_dock.width()}"
            )
            assert "too small" in again.status_message()
    finally:
        close_window(again)


def test_settings_live_in_an_ini_file_under_the_cache_parent(window, tmp_path):
    settings = window.settings
    assert settings.format() == QSettings.Format.IniFormat
    expected = tmp_path / ".cache" / "tda_app.ini"
    assert Path(settings.fileName()) == expected
    window.save_window_state()
    assert expected.exists()
    assert settings.value(f"last/{window.annotator}/desktop") is not None


def test_log_file_is_written_under_the_cache_parent(window, tmp_path):
    assert Path(S.log_path(window.paths)) == tmp_path / ".cache" / "logs" / "tda_app.log"
    window.report_error("something went wrong")
    assert "something went wrong" in window.status_message()
    assert Path(S.log_path(window.paths)).exists()


# --------------------------------------------------------------------------- #
# the lock
# --------------------------------------------------------------------------- #
def test_take_lock_reports_the_holder_and_the_age(tmp_path):
    db, _paths, _tax = make_db(tmp_path)
    db.acquire_lock("someone_else")
    try:
        message = take_lock(db, "tester")
        assert message is not None
        assert "someone_else" in message
    finally:
        db.release_lock()
        db.close()


def test_main_exits_with_three_when_the_lock_is_held(qapp, tmp_path):
    from app_scene import write_paths_yaml
    from tda.ui import app as app_module

    db, _paths, _tax = make_db(tmp_path)
    db.acquire_lock("someone_else")
    db.close()
    paths_yaml = write_paths_yaml(tmp_path)
    code = app_module.main(paths=paths_yaml, desktop=DESKTOP, view=VIEW,
                           annotator="tester", exec_=False)
    assert code == 3


def test_close_backs_the_database_up_and_releases_the_lock(qapp, tmp_path):
    win = open_window(tmp_path)
    win.session.db.acquire_lock("tester")
    lock_file = Path(win.session.db.path + ".lock")
    assert lock_file.exists()
    win.close()
    backups = sorted(Path(win.paths["backup_dir"]).glob("*.sqlite"))
    assert backups and backups[0].stat().st_size > 0
    assert not lock_file.exists()


def test_the_exit_backup_is_pruned_like_the_command_lines(qapp, tmp_path):
    win = open_window(tmp_path)
    folder = Path(win.paths["backup_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    old = [f"tda_2025{month:02d}01_120000.sqlite" for month in (1, 2, 3)]
    for name in old:
        (folder / name).write_bytes(b"old backup")
    (folder / "tda_pre_v3_20260918.sqlite").write_bytes(b"kept by hand")
    win.paths["backup_keep"] = 2
    win.close()
    left = {p.name for p in folder.iterdir()}
    assert "tda_pre_v3_20260918.sqlite" in left        # never a candidate
    assert old[0] not in left and old[1] not in left   # the oldest went
    assert old[2] in left                              # newest two: this one + the new copy
    assert len([n for n in left if n.startswith("tda_2") and n not in old]) == 1


def test_a_meaningless_backup_keep_does_not_block_the_exit(qapp, tmp_path):
    win = open_window(tmp_path)
    folder = Path(win.paths["backup_dir"])
    win.paths["backup_keep"] = "forty"
    win.close()
    assert win.closed is True
    assert list(folder.glob("tda_*.sqlite"))           # the copy was still made
    assert "backup_keep" in win.status_message() or "backup" in win.status_message().lower()


def test_a_failing_backup_warns_but_still_closes(qapp, tmp_path, monkeypatch):
    win = open_window(tmp_path)
    monkeypatch.setattr(type(win.session.db), "backup",
                        lambda self, dest, keep=None: (_ for _ in ()).throw(OSError("disk full")))
    win.close()
    assert "backup" in win.status_message().lower()
    assert win.closed is True


# --------------------------------------------------------------------------- #
# frame change
# --------------------------------------------------------------------------- #
def test_frame_change_updates_canvas_overlay_and_sam_token(window):
    seed_shapes(window.session, LAST_STEP)
    before = window.canvas.image_rgb().copy()
    window.session.goto(LAST_STEP - 1)
    assert not np.array_equal(window.canvas.image_rgb(), before)
    assert window.sam_point.frame_token == FrameKey(DESKTOP, LAST_STEP - 1, VIEW)
    assert window.sam_point.candidate_count == 0
    assert window.overlay.labelmap.max() > 0  # the compiled masks are painted


def test_hidden_instances_are_left_out_of_the_overlay(window):
    session = window.session
    seed_shapes(session, LAST_STEP)
    session.goto(LAST_STEP)
    painted = int(window.overlay.labelmap.max())
    assert painted > 0
    for row in session.instance_rows():
        session.set_hidden(row["key"], True)
    window.refresh_overlay()
    assert int(window.overlay.labelmap.max()) == 0


def test_zoom_is_kept_inside_a_pose_segment_and_reset_across_one(qapp, tmp_path):
    win = open_window(tmp_path, two_segments=True)
    try:
        win.session.goto(LAST_STEP)
        win.canvas.set_zoom(4.0)
        win.session.goto(LAST_STEP - 1)  # same segment
        assert win.canvas.zoom_factor() == pytest.approx(4.0, rel=1e-3)
        win.canvas.set_zoom(4.0)
        win.session.goto(SEGMENT_CUT - 1)  # the other segment
        assert win.canvas.zoom_factor() != pytest.approx(4.0, rel=1e-3)
    finally:
        close_window(win)


def test_a_missing_frame_shows_a_placeholder_and_disables_the_tools(qapp, tmp_path):
    win = open_window(tmp_path, missing=(LAST_STEP,))
    try:
        win.session.goto(LAST_STEP)
        assert win.tools_enabled is False
        assert "no image" in win.placeholder_label.text().lower()
        assert win.placeholder_label.isVisibleTo(win.stack)
        win.session.goto(LAST_STEP - 1)
        assert win.tools_enabled is True
    finally:
        close_window(win)


def test_paging_walks_backwards_in_annotation_order(window):
    window.set_mode(A.MODE_ANNOTATE)
    start = window.session.current().step
    window.act_step(-1)
    assert window.session.current().step < start
    window.act_step(+1)
    assert window.session.current().step == start
    window.act_step_edge("first")
    assert window.session.current().step == min(window.session.steps())


# --------------------------------------------------------------------------- #
# tools and overlay display
# --------------------------------------------------------------------------- #
def test_tool_switching_and_radius_reach_the_status_bar(window):
    window.act_tool("eraser")
    assert window.active_tool is window.eraser
    assert "橡皮擦" in window.tool_label.text()
    assert "Eraser" in window.tool_label.toolTip()
    radius = window.brush.radius
    window.act_tool("brush")
    window.act_radius(+1)
    assert window.brush.radius == radius + 1
    assert str(window.brush.radius) in window.tool_label.text()


def test_overlay_display_toggles_do_not_touch_the_data(window):
    seed_shapes(window.session, LAST_STEP)
    window.act_toggle_overlays()
    assert window.overlay.visible is False
    window.act_toggle_overlays()
    assert window.overlay.visible is True
    outline = window.canvas.overlay_outline
    window.act_toggle_outline()
    assert window.canvas.overlay_outline is not outline
    alpha = window.canvas.overlay_alpha
    window.act_opacity(+1)
    assert window.canvas.overlay_alpha > alpha
    window.act_toggle_grid()  # pixel grid is a zoom threshold, never an error


def test_no_exception_escapes_a_guarded_slot(window, monkeypatch):
    def boom(*_args, **_kwargs) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(window.session.truth, "refresh", boom)
    window.act_refresh_all()  # must not raise
    assert "kaboom" in window.status_message()
    assert window.session.db.conn.in_transaction is False


def test_f5_recompiles_this_frame_only(window, monkeypatch):
    """The batch belongs to ``cli check``; F5 has to stay instant."""
    swept: list[int] = []
    refreshed: list[object] = []
    truth = window.session.truth
    real = truth.refresh
    monkeypatch.setattr(window.session, "refresh_all", lambda: swept.append(1) or {})
    monkeypatch.setattr(truth, "refresh",
                        lambda key, *a, **k: (refreshed.append(key), real(key, *a, **k))[1])
    key = window.session.current()
    window.act_refresh_all()
    assert swept == []
    assert refreshed and set(refreshed) == {key}    # this frame, nothing else
    assert "cli check" in window.status_message()


def test_f5_reports_the_conflicts_still_open_not_only_the_new_ones(window):
    """"0 conflicts" on a frame somebody is still arguing about.

    The counter says what *this* recompile queued, and queueing the same
    disagreement twice is exactly what the truth table refuses to do -- so on
    the second F5 the line read as though the frame were settled.
    """
    from tda.core.masks import encode_rle
    import numpy as np

    key = window.session.current()
    window.session.db.add_conflict(key, "chassis",
                                   encode_rle(np.zeros((64, 64), bool)), None, 7)

    window.act_refresh_all()

    line = window.status_message()
    assert "0 new conflicts" in line
    assert "1 open" in line


# --------------------------------------------------------------------------- #
# steps mode
# --------------------------------------------------------------------------- #
def test_saving_the_step_table_reopens_the_session_on_the_same_frame(window):
    window.set_mode(A.MODE_STEPS)
    step = window.session.current().step
    window.steps_panel.sigSaved.emit(DESKTOP)
    assert window.session.current().step == step
    assert window.session.steps()


def test_leaving_steps_mode_with_unsaved_edits_asks_first(window, monkeypatch):
    """The question goes through a real ``QMessageBox``, which used to raise.

    ``confirm_discard`` referred to a name its module did not import, so the
    annotator was trapped in Steps mode by a ``NameError`` inside a guarded
    slot; patching the dialog instead of the method is what catches that.
    """
    from PySide6.QtWidgets import QMessageBox

    window.set_mode(A.MODE_STEPS)
    window._steps_dirty = True
    answers = iter([QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes])
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: next(answers)))
    window.set_mode(A.MODE_ANNOTATE)
    assert window.mode == A.MODE_STEPS  # refused
    window.set_mode(A.MODE_ANNOTATE)
    assert window.mode == A.MODE_ANNOTATE


def test_a_staged_constraint_edge_is_an_unsaved_step_table_edit(window, monkeypatch):
    """B5: the Relations tab goes through the one gate, not around it."""
    from PySide6.QtWidgets import QMessageBox

    window.set_mode(A.MODE_STEPS)
    tab = window.steps_panel.relations_tab
    tab.target_box.setCurrentText("storage_drive.ssd.01")
    tab.kind_box.setCurrentText("blocked_by")
    tab.blocker_box.setCurrentText("psu.01")

    tab.add_edge()

    assert window._steps_dirty
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    window.set_mode(A.MODE_ANNOTATE)
    assert window.mode == A.MODE_STEPS  # refused, like any other unsaved edit


def test_a_missing_frame_does_not_replace_the_step_table(qapp, tmp_path):
    """Steps mode is about the log, not about the image: it stays put."""
    win = open_window(tmp_path, missing=(LAST_STEP,))
    try:
        win.set_mode(A.MODE_STEPS)
        assert win.stack.currentWidget() is win.steps_panel
        win.session.goto(LAST_STEP)
        assert win.tools_enabled is False
        assert win.stack.currentWidget() is win.steps_panel
    finally:
        close_window(win)


def test_the_desktop_combo_names_the_brand_and_the_model(window):
    text = window.desktop_combo.currentText()
    meta = window.session.db.get_desktop(DESKTOP) or {}
    for part in str(meta.get("brand") or "").split()[:1]:
        assert part in text
    assert f"D{DESKTOP}" in text and "/" in text     # the done/total counter


# --------------------------------------------------------------------------- #
# review mode
# --------------------------------------------------------------------------- #
def test_review_mode_does_not_sweep_implicitly(window, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(window.session, "refresh_all", lambda: calls.append(1) or {})
    window.set_mode(A.MODE_REVIEW)
    assert calls == []
    window.act_refresh_all()
    assert calls == []      # F5 is this frame only; the sweep is cli check's job


def test_review_item_activation_opens_that_frame(window):
    """Activating a queue entry really moves the canvas -- through the window."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QListWidgetItem

    session = window.session
    seed_shapes(session, LAST_STEP)
    window.set_mode(A.MODE_REVIEW)
    target = LAST_STEP - 2
    queue = window.review.list_for("needs_review")
    item = QListWidgetItem(f"Step {target}")
    item.setData(int(Qt.ItemDataRole.UserRole), target)
    queue.addItem(item)
    queue.setCurrentItem(item)
    queue.itemActivated.emit(item)
    assert window.session.current().step == target


def test_an_uncommitted_edit_blocks_a_review_activation_too(window):
    """The queue is a navigation gesture like any other."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QListWidgetItem

    window.act_commit()                      # accept the ROI, arm the brush
    instance = [r["instance"] for r in window.session.task_card() if r.get("instance")][0]
    window.task_card.sigRequestEdit.emit(instance)
    window.set_editing_mask(np.ones((64, 64), dtype=bool))
    step = window.session.current().step
    window.set_mode(A.MODE_REVIEW)
    queue = window.review.list_for("needs_review")
    item = QListWidgetItem("Step 3")
    item.setData(int(Qt.ItemDataRole.UserRole), 3)
    queue.addItem(item)
    queue.itemActivated.emit(item)
    assert window.session.current().step == step


def test_a_second_window_on_the_same_database_shares_its_session(qapp, tmp_path):
    """Two windows in one process: the second one opens on the same frame."""
    db, paths, tax = make_db(tmp_path)
    session = AnnotationSession(db, tax, TruthService(db, tax), paths["cache_dir"],
                                "tester")
    session.open(DESKTOP, VIEW)
    win = MainWindow(session, paths, "tester", sam_queue=StubSamQueue())
    try:
        assert win.session.current() == FrameKey(DESKTOP, LAST_STEP, VIEW)
        assert win.canvas.image_rgb() is not None
        assert win.frame_label.text().endswith(win.session.frame_status(LAST_STEP))
    finally:
        close_window(win)
        db.close()


# --------------------------------------------------------------------------- #
# round 4 minors
# --------------------------------------------------------------------------- #
def test_the_instance_table_can_show_the_parts_that_are_gone(window):
    """A removed part has no compiled row, so it dropped off the list entirely.

    The annotator loses sight of everything already taken out of the machine,
    which in reverse order is most of it.  The toggle adds them back as greyed,
    un-editable rows read from the frame's state.
    """
    panel = window.instances
    live = {row["key"] for row in panel.rows()}
    assert panel.show_removed.isChecked() is False

    panel.show_removed.setChecked(True)
    QApplication.processEvents()

    shown = [panel.table().item(r, 1).text()
             for r in range(panel.table().rowCount())]
    extra = [key for key in shown if key not in live]
    assert extra, "no removed part was added"
    # they are not editable: the selection cannot land on one
    panel.table().setCurrentCell(panel.table().rowCount() - 1, 1)
    assert panel.selected_instance() is None


def test_a_click_on_the_minimap_pans_instead_of_painting(window):
    """It was transparent to the mouse, so the click fell through as a stroke."""
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest

    window.resize(900, 700)
    window.show()
    QApplication.processEvents()
    mini = window.canvas.minimap()
    assert not mini.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    window.canvas.set_zoom(32.0)      # far enough in that panning has somewhere to go
    window.canvas.center_on((32.0, 32.0))
    QApplication.processEvents()
    before = window.canvas.viewport_image_rect()

    QTest.mouseClick(mini, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(4, 4))
    QApplication.processEvents()

    assert window.canvas.viewport_image_rect() != before


def test_no_panel_reaches_the_session_behind_the_window(window):
    """The dead ones were still there to be called, and still un-forced."""
    assert not hasattr(window.task_card, "commit")
    assert not hasattr(window.review, "confirm")
    assert not hasattr(window.review, "open_selected")
    assert not hasattr(window.review, "_goto")


# --------------------------------------------------------------------------- #
# the timeline shows the status of every frame, not only the open one (item 4)
# --------------------------------------------------------------------------- #
def test_the_timeline_repaints_other_rows_when_the_queues_change(window, monkeypatch):
    """`_refresh_statuses` only ran on a frame change.

    After a commit that touched a verified frame its row stayed amber
    ``recheck`` although ``frame_status`` already said ``conflict`` -- the
    annotator had no way of knowing where the work was.
    """
    from tda.ui import session_api as api

    other = min(window.session.steps())
    before = window.timeline.step_brush(other).color().name()
    monkeypatch.setattr(window.session, "frame_status",
                        lambda step: (api.STATUS_CONFLICT if step == other
                                      else api.STATUS_AUTO))

    window._on_queues_changed()

    assert window.timeline.step_brush(other).color().name() != before


def test_the_sweeper_reporting_progress_repaints_the_timeline(window, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(window.timeline, "refresh_statuses", lambda: seen.append(1))
    window._on_sweep_progress(1, 4, 0)
    assert seen == [1]


def test_a_commit_repaints_the_timeline(window, monkeypatch):
    """A commit is exactly when another frame's status changes."""
    seen: list[int] = []
    card = [r for r in window.session.task_card() if r.get("instance")]
    window.task_card.sigRequestEdit.emit(str(card[0]["instance"]))
    window.set_editing_mask(np.ones((64, 64), dtype=bool))
    monkeypatch.setattr(window.timeline, "refresh_statuses", lambda: seen.append(1))

    window.act_commit()

    assert seen, "the timeline was not asked to repaint"


# --------------------------------------------------------------------------- #
# the machine chooser's [done/total] (item 5)
# --------------------------------------------------------------------------- #
def chooser_text(win: MainWindow) -> str:
    return win.desktop_combo.itemText(win.desktop_combo.currentIndex())


def test_the_chooser_count_follows_a_confirmation(window):
    """Computed once at launch and never again -- and the guide points at it."""
    before = chooser_text(window)
    assert "[0/" in before, before
    seed_shapes(window.session, window.session.current().step)
    window.task_card.refresh()

    assert window.act_confirm() is True

    assert chooser_text(window) != before
    assert "[1/" in chooser_text(window), chooser_text(window)


def test_the_chooser_count_follows_a_view_switch(window):
    """The count is per view, so switching view has to re-read it."""
    seed_shapes(window.session, window.session.current().step)
    window.task_card.refresh()
    window.act_confirm()
    assert "[1/" in chooser_text(window)

    window.act_set_view("oak1")

    assert window.session.view == "oak1"
    assert "[0/2]" in chooser_text(window), chooser_text(window)


def test_the_chooser_count_follows_the_sweeper(window, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(window, "refresh_desktop_counts", lambda: seen.append(1))
    window._on_sweep_progress(4, 4, 0)
    assert seen == [1]


def test_the_zoom_percentage_follows_the_wheel(window):
    """The status bar's zoom was only rewritten by actions that called it."""
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent

    window.resize(900, 700)
    window.show()
    QApplication.processEvents()
    before = window.zoom_label.text()

    viewport = window.canvas.viewport()
    centre = QPointF(viewport.rect().center())
    QApplication.sendEvent(viewport, QWheelEvent(
        centre, viewport.mapToGlobal(QPoint(*map(int, (centre.x(), centre.y())))),
        QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
    ))
    QApplication.processEvents()

    assert window.canvas.zoom_factor() != pytest.approx(float(before.rstrip("%")) / 100)
    assert window.zoom_label.text() != before


# --------------------------------------------------------------------------- #
# a view with no frames at all (item 7)
# --------------------------------------------------------------------------- #
EMPTY_VIEW = "rs"


def test_switching_to_a_view_with_no_frames_is_refused(window):
    """It left the PREVIOUS view's image, masks and timeline on screen.

    `render_frame` returned early on a closed session, so everything the
    annotator could see still belonged to the view they had left -- and the
    brush was still armed over it.
    """
    before = window.session.view
    step = window.session.current().step

    window.act_set_view(EMPTY_VIEW)

    assert window.session.view == before
    assert window.session.current().step == step
    assert window.session.is_open is True
    assert "没有" in window.status_message() or "no frames" in window.status_message()
    assert window.view_buttons[before].isChecked() is True


def test_switching_to_a_desktop_with_no_frames_is_refused(window):
    before = int(window.session.desktop)
    missing = max(window.db.desktop_ids()) + 7

    window.act_set_desktop(missing)

    assert int(window.session.desktop) == before
    assert window.session.is_open is True


def test_a_closed_session_leaves_nothing_of_the_last_frame_on_screen(window):
    """Whatever put the session in this state, the window must not lie."""
    window.session.open(int(window.session.desktop), EMPTY_VIEW, force=True)
    window.render_frame()
    QApplication.processEvents()

    assert window.session.is_open is False
    assert window.stack.currentWidget() is window.placeholder_label
    assert window.active_tool is None or window.tools_enabled is False
    assert window.timeline.list_widget().count() == 0
    assert window.instances.table().rowCount() == 0
    assert window.task_card.list_widget().count() == 0


def test_a_poisoned_last_frame_in_the_ini_cannot_stop_the_launch(qapp, tmp_path):
    """The next launch raised out of __init__ and the app would not start.

    A view with no frames was saved as "last view"; on the next launch the
    timeline asked the session for a frame that does not exist.
    """
    from tda.ui.app_shell import resume_target

    paths = make_paths(tmp_path)
    make_session(tmp_path).close(force=True)          # build the scene's database
    settings = S.make_settings(paths)
    settings.setValue("last/tester/desktop", 999)
    settings.setValue("last/tester/view", "nonsense")
    settings.setValue("last/tester/step", 4242)
    settings.sync()

    target = resume_target(paths, "tester", None, None, None, paths["db_path"])

    assert int(target["desktop"]) == DESKTOP
    assert target["view"] == VIEW            # the only view the scene has frames for
    assert target["step"] in (None, LAST_STEP)


def test_ctrl_s_in_steps_mode_applies_the_step_table(window, monkeypatch):
    """It said "saved" while the step table's edits were still unsaved.

    ``Ctrl+S`` saved the *session*, which in Steps mode is not what is on
    screen: the annotator's row edits sat in the panel's model, and the word
    "saved" is exactly the thing that stops somebody pressing Apply.
    """
    window.set_mode(A.MODE_STEPS)
    applied: list[int] = []
    monkeypatch.setattr(window.steps_panel, "apply", lambda: applied.append(1))
    window.steps_panel.status.setText("Saved D13: 0 open question(s).")

    window.act_save()

    assert applied == [1]
    assert "Saved D13" in window.status_message()


def test_ctrl_s_outside_steps_mode_still_saves_the_session(window, monkeypatch):
    saved: list[int] = []
    monkeypatch.setattr(window.session, "save", lambda: saved.append(1))
    window.set_mode(A.MODE_ANNOTATE)
    window.act_save()
    assert saved == [1]


# --------------------------------------------------------------------------- #
# the toolbar must not eat the keyboard (item 10)
# --------------------------------------------------------------------------- #
def test_the_toolbar_widgets_never_take_the_focus(window):
    """A click on the chooser or a view button killed every shortcut, silently.

    ``blocks_shortcuts`` is right about a combo box that has the focus -- the
    annotator may be typing in it -- so the answer is that these never take it.
    """
    assert window.desktop_combo.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert window.mode_tabs.focusPolicy() == Qt.FocusPolicy.NoFocus
    for button in window.view_buttons.values():
        assert button.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_choosing_a_view_puts_the_focus_back_on_the_canvas(window):
    """After a top-bar choice the next key belongs to the canvas again."""
    window.resize(900, 700)
    window.show()
    QApplication.processEvents()
    window.task_card.list_widget().setFocus()
    QApplication.processEvents()

    window.act_set_view("oak1")          # frame rows, no images: the placeholder
    QApplication.processEvents()
    assert window.stack.currentWidget() is window.placeholder_label

    window.act_set_view(VIEW)            # back to the view that has pictures
    QApplication.processEvents()
    assert window.canvas.hasFocus() or QApplication.focusWidget() is window.canvas


def test_the_cheat_sheet_does_not_switch_the_keyboard_off(window):
    """It is read-only, and it is exactly what somebody has open while learning."""
    window.resize(900, 700)
    window.show()
    QApplication.processEvents()
    window.act_cheat_sheet()
    QApplication.processEvents()
    try:
        assert window._cheat_sheet.isVisible()
        assert window._shortcut_context_ok() is True
    finally:
        window._cheat_sheet.close()


# --------------------------------------------------------------------------- #
# the log file is worth reading (addendum, item 17)
# --------------------------------------------------------------------------- #
def log_text(win: MainWindow) -> str:
    for handler in win.logger.handlers:
        handler.flush()
    return Path(S.log_path(win.paths)).read_text(encoding="utf-8")


def test_the_log_records_the_session_the_frames_and_the_commits(qapp, tmp_path):
    """30 minutes of driving left a 0-byte tda_app.log."""
    win = open_window(tmp_path)
    try:
        win.resize(900, 700)
        win.show()
        QApplication.processEvents()
        assert "window open" in log_text(win)
        assert f"D{DESKTOP}" in log_text(win) and VIEW in log_text(win)

        card = [r for r in win.session.task_card() if r.get("instance")]
        instance = str(card[0]["instance"])
        win.task_card.sigRequestEdit.emit(instance)
        win.set_editing_mask(np.ones((64, 64), dtype=bool))
        win.act_commit()
        text = log_text(win)
        assert "commit" in text and instance in text
        assert "px" in text and "ms" in text

        win.act_step(-1)
        assert "frame" in log_text(win)

        win.act_confirm()
        assert "confirm" in log_text(win)
    finally:
        close_window(win)
    assert "window closed" in log_text(win)


def test_the_log_records_every_sam_prompt(window):
    """SAM timings existed only in the smoke report; the log said nothing.

    And ``tda.models.sam_service``'s own lines never reached the file at all,
    because the handler was on ``tda.app`` rather than on ``tda``.
    """
    import logging

    card = [r for r in window.session.task_card() if r.get("instance")]
    window.task_card.sigRequestEdit.emit(str(card[0]["instance"]))
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush(multimask=True)
    QApplication.processEvents()

    logging.getLogger("tda.models.sam_service").info("hello from the service")
    text = log_text(window)
    assert "sam prompt" in text
    assert "points=1" in text and "candidates=3" in text and "ms=" in text
    assert "hello from the service" in text, "the service's logger is not attached"
    # What the trial's log could not say: how the mask was composed, and how
    # much of what was already there survived it (task U1, ruling R3).
    assert "sam apply" in text and "compose=" in text and "owned" in text

    window.act_cycle_candidate()
    assert "sam cycle" in log_text(window)


# --------------------------------------------------------------------------- #
# a held line delays the next one, it does not swallow it (round 2, Minor 2)
# --------------------------------------------------------------------------- #
def _quiet(win) -> None:
    """Let the ROI measurement land and settle, so nothing else writes a line."""
    win.wait_for_roi_proposal()
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()


def _wait_for_status(win, wanted: str, timeout: float = 5.0) -> str:
    import time as _time

    deadline = _time.perf_counter() + timeout
    while _time.perf_counter() < deadline and win.status_message() != wanted:
        QApplication.processEvents()
        _time.sleep(0.01)
    return win.status_message()


def test_a_line_held_out_is_shown_when_the_hold_expires(window):
    """The layout warning is held for four seconds and the ROI answers inside it.

    Before this, "could not find the chassis, drag a box" -- the one line that
    explains why no rectangle appeared -- was simply dropped.
    """
    _quiet(window)
    window.report("the saved dock layout was reset", hold_ms=60)
    window.report("could not find the chassis")

    assert window.status_message() == "the saved dock layout was reset"
    assert _wait_for_status(window, "could not find the chassis") == \
        "could not find the chassis"


def test_only_the_last_line_the_hold_turned_away_comes_back(window):
    _quiet(window)
    window.report("held", hold_ms=60)
    window.report("first dropped")
    window.report("second dropped")

    assert _wait_for_status(window, "second dropped") == "second dropped"


def test_a_line_shown_after_the_hold_is_not_overwritten_by_the_replay(window):
    """Anything newer that got through clears what was waiting."""
    import time as _time

    _quiet(window)
    window.report("held", hold_ms=40)
    window.report("dropped")
    _time.sleep(0.09)
    window.report("newer, and it got through")
    assert window.status_message() == "newer, and it got through"

    QApplication.processEvents()
    _time.sleep(0.05)
    QApplication.processEvents()
    assert window.status_message() == "newer, and it got through"


def test_clearing_the_status_bar_drops_what_was_waiting(window):
    """An empty text is a frame change, which every hint is about."""
    import time as _time

    _quiet(window)
    window.report("held", hold_ms=40)
    window.report("dropped")
    window.report("")
    assert window.status_message() == ""

    _time.sleep(0.09)
    QApplication.processEvents()
    assert window.status_message() == ""


def test_closing_the_window_joins_the_timeline_thumbnail_reader(qapp, tmp_path):
    """No thread may outlive the window it belongs to (the smoke checks this).

    The timeline reads a row's picture on a thread of its own -- without the
    offline thumbnail pass that picture is the 12 MP frame itself -- so the
    window's shutdown has to stop it, like the diff worker and the SAM queue.
    """
    import threading
    import time

    win = open_window(tmp_path)
    try:
        win.resize(1200, 900)
        win.show()
        QApplication.processEvents()
        win.timeline.ensure_visible_thumbs()
        deadline = time.perf_counter() + 5.0
        while win.timeline._asked and time.perf_counter() < deadline:
            QApplication.processEvents()
        assert win.timeline._thumbs, "no row picture was ever read"
    finally:
        close_window(win)
    assert win.timeline._reader.running() is False
    assert "tda-thumbs" not in {t.name for t in threading.enumerate() if t.is_alive()}


def test_a_reader_that_will_not_stop_does_not_skip_the_rest_of_the_teardown(
    qapp, tmp_path, monkeypatch
):
    """The signal disconnects outrank the thumbnail reader.

    A sweeper still delivering into a window that has let go of its session is
    an exception out of a Qt slot with nothing left to catch it; a reader that
    outlives its panel is a idle thread. So the one that can fail goes inside
    a guard, and the log says it happened.
    """
    win = open_window(tmp_path)
    monkeypatch.setattr(win.timeline, "shutdown",
                        lambda: (_ for _ in ()).throw(RuntimeError("stuck")))
    try:
        win.shutdown()
        assert win.closed is True
        # the disconnects ran: the session can announce into nothing now
        win.session.sigFrameChanged.emit(win.session.current())
        QApplication.processEvents()
    finally:
        win.hide()
        win.setParent(None)
        win.deleteLater()
        QApplication.processEvents()


# --------------------------------------------------------------------------- #
# the armed tool is unmistakable (task U1, report 1, ruling R1)
# --------------------------------------------------------------------------- #
def _answer_roi(win: MainWindow) -> None:
    """Skip the chassis rectangle, the way ``Esc`` does."""
    if win.roi_editing:
        win.act_clear_edit()


@pytest.mark.parametrize("tool,kind", [
    ("brush", "circle"), ("eraser", "circle"), ("occluder", "circle"),
    ("sam_point", "cross"), ("sam_box", "cross"),
])
def test_every_tool_says_what_it_is_under_the_mouse(window, tool, kind):
    _answer_roi(window)
    window.act_tool(tool)
    spec = window.canvas.tool_cursor()
    assert spec is not None and spec.kind == kind, tool
    if kind == "circle":
        assert spec.radius == window._tool_for(tool).radius


def test_a_brush_too_big_for_a_ring_still_shows_its_size(window):
    """The crosshair fallback must not take the size off the screen too."""
    from tda.ui.canvas.view import CURSOR_MAX_PX

    _answer_roi(window)
    window.act_tool("brush")
    window.brush.set_radius(96)
    window.canvas.set_zoom(1.0)
    window.sync_tool_cursor()
    window.update_status()

    assert window.canvas.cursor_diameter() > CURSOR_MAX_PX
    assert window.canvas.cursor_is_ring() is False
    assert window.canvas.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    assert "r=96" in window.tool_label.text(), window.tool_label.text()
    assert "十字" in window.tool_label.toolTip()


def test_the_brush_and_the_eraser_do_not_look_alike(window):
    _answer_roi(window)
    window.act_tool("brush")
    brush = window.canvas.tool_cursor()
    window.act_tool("eraser")
    assert window.canvas.tool_cursor() != brush


def test_the_roi_rectangle_and_review_mode_have_their_own_cursor(window):
    assert window.roi_editing, "a fresh segment offers the rectangle"
    assert window.canvas.tool_cursor().kind == "cross"
    _answer_roi(window)
    window.set_mode(A.MODE_REVIEW)
    assert window.canvas.tool_cursor().kind == "forbidden"


def test_a_flashed_neighbour_forbids_the_canvas(window):
    _answer_roi(window)
    window.act_flash_compare(True)
    if window.is_flashing():
        assert window.canvas.tool_cursor().kind == "forbidden"
    window.act_flash_compare(False)
    assert window.canvas.tool_cursor().kind == "circle"


def test_a_click_on_the_canvas_takes_the_keyboard_back(window):
    """The next letter key has to be a shortcut, whatever had the focus."""
    _answer_roi(window)
    window.task_card._list.setFocus()
    window._on_canvas_press(10.0, 10.0, None)
    assert window.canvas.hasFocus() or window.focusWidget() is window.canvas


def test_activating_a_card_item_hands_the_focus_back_to_the_canvas(window):
    """The double-click leaves the list focused; the canvas gets it back."""
    _answer_roi(window)
    card = [r for r in window.session.task_card() if r.get("instance")]
    if not card:
        pytest.skip("this frame has nothing to draw")
    window.task_card._list.setFocus()
    window.task_card.sigRequestEdit.emit(str(card[0]["instance"]))
    assert window.focusWidget() is window.canvas


def test_a_shortcut_that_went_into_a_text_field_says_so(window):
    """A key that silently does nothing is a key pressed again, harder."""
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QLineEdit

    from tda.ui.app_keys import KEY_SWALLOWED

    _answer_roi(window)
    window.act_tool("eraser")
    field = QLineEdit(window)
    field.setFocus()
    assert window._focus_widget() is field
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B,
                      Qt.KeyboardModifier.NoModifier, "b")
    assert window.handle_key(event) is False, "the field must keep its letter"
    assert window.status_message() == KEY_SWALLOWED
    assert window._tool_name == "eraser", "the key must not also switch the tool"

    # A key that is *not* bound in this mode is an ordinary letter: no line.
    window.report("")
    assert window.handle_key(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Z,
                                       Qt.KeyboardModifier.NoModifier, "z")) is False
    assert window.status_message() == ""
    field.deleteLater()


# --------------------------------------------------------------------------- #
# the log can tell the story next time (task U1, ruling R3)
# --------------------------------------------------------------------------- #
def test_the_log_records_tool_switches_and_strokes(qapp, tmp_path):
    win = open_window(tmp_path)
    try:
        _answer_roi(win)
        card = [r for r in win.session.task_card() if r.get("instance")]
        win.task_card.sigRequestEdit.emit(str(card[0]["instance"]))
        win.act_tool("brush")
        win.act_tool("eraser")
        win.act_tool("brush")
        win.brush.on_press(20.0, 20.0, None)
        win.brush.on_release(24.0, 24.0, None)
        QApplication.processEvents()

        text = log_text(win)
        assert "tool brush -> eraser via key" in text
        assert "tool eraser -> brush via key" in text
        stroke = [ln for ln in text.splitlines() if " stroke tool=" in ln]
        assert stroke, text
        assert "instance=" in stroke[-1] and "layer 0 -> " in stroke[-1]
        assert "+" in stroke[-1] and "px" in stroke[-1]
    finally:
        close_window(win)


def test_the_log_records_the_roi_question_and_its_answer(qapp, tmp_path):
    win = open_window(tmp_path)
    try:
        assert win.roi_editing
        win.wait_for_roi_proposal()
        win.act_clear_edit()                  # Esc: skipped, not answered
        assert "roi proposal opened" in log_text(win)
        assert "roi rectangle dismissed" in log_text(win)
        win.act_no_roi()
        assert "roi dismissed" in log_text(win)
    finally:
        close_window(win)
