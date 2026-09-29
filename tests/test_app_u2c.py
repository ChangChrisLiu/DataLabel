"""U2c: four things the controller found still confusing on real data.

1. The status line said "13 problem(s) — 见任务卡" over a task card that showed
   no problems at all (D13/scan, step 34 reached from 42).
2. The instance table's Vis column said ``out`` for a part nobody had drawn
   yet -- which reads as "somebody set it out of view".
3. The ROI bar said "请按 Shift+R 自己画" while the ROI editor was already open.
4. A view with one frame that has no image could never read ``n/n``.

Each has its tests here, against the real window on the synthetic D13 scene.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app_scene import (
    DESKTOP,
    LAST_STEP,
    VIEW,
    StubSamQueue,
    cell,
    chassis_instances,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
    write_paths_yaml,
)
from tda.core import masks
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.truth_inputs import annotatable_steps
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.panels.taskcard import instance_of

CHASSIS = "chassis"
#: At LAST_STEP - 1 nothing about the PSU changes, so no card row names it.
UNLISTED = "psu.01"


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, session=None, show=False, **kwargs) -> MainWindow:
    session = session if session is not None else make_session(tmp_path, **kwargs)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    if show:
        win.show()
        QApplication.processEvents()
    return win


@pytest.fixture
def window(qapp, tmp_path):
    win = open_window(tmp_path)
    yield win
    close_window(win)


def answer_roi(win: MainWindow) -> None:
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


def press(win: MainWindow, name: str) -> None:
    win.dispatch(A.action_named(name))
    QApplication.processEvents()


def open_rows(win: MainWindow) -> set[str]:
    return {str(r["instance"]) for r in win.task_card.rows() if not r["done"]}


def pane_codes(win: MainWindow) -> list[str]:
    return [r["code"] for r in win.task_card.problem_rows()]


# --------------------------------------------------------------------------- #
# item 1: the status line never points at something that is not on screen
# --------------------------------------------------------------------------- #
def test_arriving_where_every_problem_is_an_open_row_says_nothing(window):
    """The start frame's missing shapes *are* its rows: the card is the list."""
    answer_roi(window)
    session = window.session
    session.goto(LAST_STEP - 1, force=True)
    session.goto(LAST_STEP, force=True)
    QApplication.processEvents()

    problems = session.current_problems()
    assert problems, "the scene's start frame has missing shapes"
    assert {instance_of(p) for p in problems} <= open_rows(window)
    assert "problem(s)" not in window.status_message()
    assert not window.task_card.problems_visible()


def test_a_problem_no_row_mentions_is_in_the_pane_and_the_status_points_at_it(
        qapp, tmp_path):
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP, skip=(UNLISTED,))
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        session.goto(LAST_STEP, force=True)
        session.goto(LAST_STEP - 1, force=True)
        QApplication.processEvents()

        assert UNLISTED not in {str(r["instance"]) for r in win.task_card.rows()}
        assert f"missing_shape:{UNLISTED}" in session.current_problems()
        assert win.task_card.problems_visible()
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]
        assert "1 problem(s) — 见任务卡" in win.status_message()
        # nothing an open row already asks for is said twice
        assert not {instance_of(c) for c in pane_codes(win)} & open_rows(win)
    finally:
        close_window(win)


def test_the_frame_the_window_opens_on_already_has_its_pane(qapp, tmp_path):
    """The session announced it before the window existed (as ``main`` does)."""
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP, skip=(UNLISTED,))
    session.goto(LAST_STEP - 1, force=True)
    win = open_window(tmp_path, session=session)
    try:
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]
    finally:
        close_window(win)


def test_a_refused_space_still_lists_everything_rows_included(window):
    answer_roi(window)
    session = window.session
    session.goto(LAST_STEP, force=True)
    QApplication.processEvents()
    assert not window.task_card.problems_visible()

    assert window.act_confirm() is False

    rows = window.task_card.problem_rows()
    assert "cannot be verified" in rows[0]["text"]
    assert {r["code"] for r in rows[1:]} == set(session.current_problems())
    assert window.status_message().startswith(
        f"step {LAST_STEP} is not complete: {len(rows) - 1} problem(s)")


def test_a_failed_recheck_elsewhere_neither_wipes_the_pane_nor_points_at_it(
        qapp, tmp_path):
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP, skip=(UNLISTED,))
    session.goto(LAST_STEP - 1, force=True)
    win = open_window(tmp_path, session=session)
    try:
        win.report("")
        session.sigProblems.emit(["step 3: re-check failed: boom"])
        QApplication.processEvents()
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]
        assert "见任务卡" not in win.status_message()
    finally:
        close_window(win)
