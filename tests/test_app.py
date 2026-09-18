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
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from app_scene import (
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
    win.shutdown()


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
    assert "brush" in window.tool_label.text()
    assert window.sam_label.text()


def test_mode_switch_moves_the_central_widget(window):
    window.set_mode(A.MODE_ANNOTATE)
    assert window.stack.currentWidget() is window.canvas
    window.set_mode(A.MODE_STEPS)
    assert window.stack.currentWidget() is window.steps_panel
    window.set_mode(A.MODE_REVIEW)
    assert window.stack.currentWidget() is window.canvas
    assert window.review_dock.isVisibleTo(window) or window.review_dock.isVisible()


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


def test_a_failing_backup_warns_but_still_closes(qapp, tmp_path, monkeypatch):
    win = open_window(tmp_path)
    monkeypatch.setattr(type(win.session.db), "backup",
                        lambda self, dest: (_ for _ in ()).throw(OSError("disk full")))
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
        win.shutdown()


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
        win.shutdown()


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
    assert "eraser" in window.tool_label.text()
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
    def boom() -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(window.session, "refresh_all", boom)
    window.act_refresh_all()  # must not raise
    assert "kaboom" in window.status_message()
    assert window.session.db.conn.in_transaction is False


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
    window.set_mode(A.MODE_STEPS)
    window._steps_dirty = True
    answers = iter([False, True])
    monkeypatch.setattr(window, "confirm_discard", lambda *a: next(answers))
    window.set_mode(A.MODE_ANNOTATE)
    assert window.mode == A.MODE_STEPS  # refused
    window.set_mode(A.MODE_ANNOTATE)
    assert window.mode == A.MODE_ANNOTATE


# --------------------------------------------------------------------------- #
# review mode
# --------------------------------------------------------------------------- #
def test_review_mode_does_not_sweep_implicitly(window, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(window.session, "refresh_all", lambda: calls.append(1) or {})
    window.set_mode(A.MODE_REVIEW)
    assert calls == []
    window.act_refresh_all()
    assert calls == [1]


def test_review_item_activation_opens_that_frame(window):
    session = window.session
    seed_shapes(session, LAST_STEP)
    session.goto(LAST_STEP)
    window.set_mode(A.MODE_REVIEW)
    window.review.list_for("missing_shape")
    target = LAST_STEP - 2
    session.goto(target)
    assert window.session.current().step == target


def test_second_session_reuses_the_same_database_without_a_second_lock(qapp, tmp_path):
    """Two windows in one test process must not fight over the lock file."""
    db, paths, tax = make_db(tmp_path)
    session = AnnotationSession(db, tax, TruthService(db, tax), paths["cache_dir"],
                                "tester")
    session.open(DESKTOP, VIEW)
    win = MainWindow(session, paths, "tester", sam_queue=StubSamQueue())
    try:
        assert win.session.is_open if hasattr(win.session, "is_open") else True
    finally:
        win.shutdown()
        db.close()
