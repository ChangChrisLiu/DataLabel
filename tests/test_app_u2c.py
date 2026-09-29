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
from tda.ui.app_roi import NO_CHASSIS_CLOSED, NO_CHASSIS_FOUND
from tda.ui.panels.instances import NOT_DRAWN
from tda.ui.panels.taskcard import instance_of
from tda.ui.panels.timeline import NO_IMAGE_TEXT, STEP_ROLE

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


# --------------------------------------------------------------------------- #
# item 2: a part nobody has drawn is "—", not "out"
# --------------------------------------------------------------------------- #
def vis_cell(win: MainWindow, key: str):
    table = win.instances.table()
    line = next(i for i in range(table.rowCount())
                if table.item(i, 1) is not None and table.item(i, 1).text() == key)
    return table.item(line, win.instances.COLUMNS.index("Vis"))


def row_of(win: MainWindow, key: str) -> dict:
    return next(r for r in win.instances.rows() if r["key"] == key)


#: What the cell shows for the labels a small drawn rectangle compiles to.
DRAWN_LABELS = {"visible": "vis", "visible_tiny": "v-tiny"}


def shows_its_label(win: MainWindow, key: str) -> bool:
    return vis_cell(win, key).text() == DRAWN_LABELS[row_of(win, key)["visibility"]]


def test_an_undrawn_part_shows_a_dash_and_says_why(window):
    answer_roi(window)
    window.session.goto(LAST_STEP, force=True)
    QApplication.processEvents()
    cell_ = vis_cell(window, CHASSIS)
    assert cell_.text() == NOT_DRAWN
    assert "还没画" in cell_.toolTip() and "not drawn yet" in cell_.toolTip()
    assert cell_.background().style() == Qt.BrushStyle.NoBrush   # not "set by hand"
    # display only: the value the compiler and the exports use is untouched
    assert row_of(window, CHASSIS)["visibility"] == "out_of_view"
    assert row_of(window, CHASSIS)["has_shape"] is False


def test_a_drawn_part_shows_its_label(window):
    answer_roi(window)
    session = window.session
    session.goto(LAST_STEP, force=True)
    session.begin_edit(CHASSIS)
    session.set_editing_mask(cell(9))
    session.commit_edit(api.SCOPE_KEYFRAME)
    session.clear_edit()
    window.instances.refresh()
    assert vis_cell(window, CHASSIS).text() != NOT_DRAWN
    assert shows_its_label(window, CHASSIS)


def test_a_label_set_by_hand_on_an_undrawn_part_keeps_its_marked_cell(window):
    answer_roi(window)
    window.session.goto(LAST_STEP, force=True)
    QApplication.processEvents()
    assert window.instances.select_instance(CHASSIS)
    press(window, "visibility_2")                          # 部分遮挡
    cell_ = vis_cell(window, CHASSIS)
    assert cell_.text() == "occ-p"
    assert "⚠" not in cell_.text()                         # nothing drawn to hide
    assert cell_.background().color().alpha() > 0, "the hand-set cell is not marked"
    assert "手动设的可见性" in cell_.toolTip()


def test_a_drawn_part_hidden_by_a_label_still_warns(window):
    answer_roi(window)
    session = window.session
    session.goto(LAST_STEP, force=True)
    session.begin_edit(CHASSIS)
    session.set_editing_mask(cell(9))
    session.commit_edit(api.SCOPE_KEYFRAME)
    session.clear_edit()
    window.instances.refresh()
    assert window.instances.select_instance(CHASSIS)
    press(window, "visibility_4")                          # 画面外
    assert vis_cell(window, CHASSIS).text() == "⚠ out"


def test_a_shape_drawn_for_this_frame_only_counts_as_drawn(window):
    """A frame override with pixels carries no keyframe id, and still is a shape."""
    answer_roi(window)
    session = window.session
    session.goto(LAST_STEP, force=True)
    session.begin_edit(CHASSIS)
    session.set_editing_mask(cell(9))
    session.commit_edit(api.SCOPE_FRAME_OVERRIDE)
    session.clear_edit()
    window.instances.refresh()

    key = session.current()
    assert window.db.frame_overrides(key)[CHASSIS].visible_rle is not None
    assert session.compiled().instances[CHASSIS].keyframe_id is None
    assert row_of(window, CHASSIS)["has_shape"] is True
    assert shows_its_label(window, CHASSIS)


# --------------------------------------------------------------------------- #
# item 3: an open ROI editor is never told to press Shift+R
# --------------------------------------------------------------------------- #
def roi_bar_text(win: MainWindow) -> str:
    return win.roi_bar.label.text() if win.roi_bar.isVisibleTo(win) else ""


def test_a_failed_proposal_says_what_to_do_in_the_open_editor(qapp, tmp_path,
                                                              monkeypatch):
    """View rs of D13: the detector found nothing and the editor was open."""
    from tda.ui import app_roi_worker

    monkeypatch.setattr(app_roi_worker, "suggest_roi_over",
                        lambda images, view: (0, 0, 64, 64))
    win = open_window(tmp_path, show=True)
    try:
        assert win.roi_editing, "the editor opens on a segment with no ROI"
        assert win.wait_for_roi_proposal() is True
        bar = roi_bar_text(win)
        assert bar.endswith(NO_CHASSIS_FOUND), bar
        assert "请按 Shift+R 自己画" not in bar
        assert NO_CHASSIS_FOUND in win.status_message()

        win.act_commit()                  # Enter on the untouched whole frame
        assert win.roi() is None, "a whole-frame ROI was stored"
        assert win.roi_editing is True
        assert NO_CHASSIS_FOUND in win.status_message()

        win.act_clear_edit()              # Esc: now the editor is closed
        assert win.roi_editing is False
        assert NO_CHASSIS_CLOSED in roi_bar_text(win)
        assert NO_CHASSIS_FOUND not in roi_bar_text(win)
        win.act_accept_roi_proposal()     # the button would store the whole frame
        assert NO_CHASSIS_CLOSED in win.status_message()
        assert win.roi() is None
    finally:
        close_window(win)


def test_a_too_small_rectangle_is_named_as_that_not_as_no_chassis(qapp, tmp_path):
    win = open_window(tmp_path, show=True)
    try:
        win.wait_for_roi_proposal()
        assert win.roi_editing
        win.on_roi_box((10.0, 10.0, 14.0, 14.0))
        bar = roi_bar_text(win)
        assert "太小" in bar and "没找到机箱" not in bar
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 4: a frame with no image is not work, so a finished view reads n/n
# --------------------------------------------------------------------------- #
FEW_STEPS = 5
MISSING_STEP = 3


def chooser_text(win: MainWindow) -> str:
    return win.desktop_combo.itemText(win.desktop_combo.currentIndex())


def seed_every_frame(session) -> None:
    """One rectangle per chassis instance any image frame needs, and the order."""
    wanted: list[str] = []
    for step in session.available_steps():
        for key in chassis_instances(session, step):
            if key not in wanted:
                wanted.append(key)
    for index, key in enumerate(wanted):
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=1,
            anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(cell(index)))],
        ))
    session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(k, "main") for k in wanted]))
    session.refresh_all()


def test_a_view_with_a_missing_frame_reaches_n_of_n(qapp, tmp_path, capsys):
    from tda.cli import main as cli_main

    session = make_session(tmp_path, last_step=FEW_STEPS, missing=(MISSING_STEP,))
    seed_every_frame(session)
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        image_steps = session.available_steps()
        assert MISSING_STEP not in image_steps and len(image_steps) == FEW_STEPS - 1
        assert image_steps == annotatable_steps(win.db, DESKTOP, VIEW, session.steps())
        n = len(image_steps)
        assert f"[0/{n}]" in chooser_text(win), chooser_text(win)

        session.goto(max(image_steps), force=True)
        for _ in image_steps:            # Space steps back, skipping the gap
            assert win.act_confirm() is True, win.task_card.problems()
        QApplication.processEvents()

        assert f"[{n}/{n}]" in chooser_text(win), chooser_text(win)
        # the timeline says why the gap is not "not done"
        timeline = win.timeline.list_widget()
        texts = {int(timeline.item(i).data(STEP_ROLE)): timeline.item(i).text()
                 for i in range(timeline.count())}
        assert NO_IMAGE_TEXT in texts[MISSING_STEP]
        assert all(NO_IMAGE_TEXT not in t for s, t in texts.items() if s != MISSING_STEP)

        # ... and so does the command line, from the same database
        capsys.readouterr()
        assert cli_main(["--paths", write_paths_yaml(tmp_path), "status",
                         "--desktop", str(DESKTOP)]) == 0
        line = next(ln for ln in capsys.readouterr().out.splitlines()
                    if ln.strip().startswith(VIEW))
        assert f"{n}/{n}" in line.replace(" ", ""), line
    finally:
        close_window(win)


def test_the_status_table_counts_what_the_navigation_walks(qapp, tmp_path):
    """The SQL counters and ``annotatable_steps`` must never disagree."""
    from tda.core.model import StepType

    session = make_session(tmp_path, last_step=FEW_STEPS, missing=(MISSING_STEP,))
    db = session.db
    steps = db.steps(DESKTOP)
    steps[0].step_type = StepType.IGNORE.value      # a skipped step has an image
    db.replace_steps(DESKTOP, steps, db.actions(DESKTOP))
    db.set_frame_flags(FrameKey(DESKTOP, 2, VIEW), review_status="verified")
    db.set_frame_flags(FrameKey(DESKTOP, MISSING_STEP, VIEW), review_status="verified")

    walked = annotatable_steps(db, DESKTOP, VIEW, [r["step"] for r in
                                                   db.frames_for(DESKTOP, VIEW)])
    assert walked == [2, 4, 5]
    assert db.count_per_view("work")[(DESKTOP, VIEW)] == len(walked)
    # a verified frame the view has no image for is not "done" either
    assert db.count_per_view("done")[(DESKTOP, VIEW)] == 1
    assert db.count_per_view("verified")[(DESKTOP, VIEW)] == 2   # stored as it was
    db.close()


def test_the_timeline_says_no_image_and_greys_the_row(qapp, tmp_path):
    session = make_session(tmp_path, last_step=FEW_STEPS, missing=(MISSING_STEP,))
    win = open_window(tmp_path, session=session)
    try:
        timeline = win.timeline.list_widget()
        item = next(timeline.item(i) for i in range(timeline.count())
                    if timeline.item(i).text().startswith(f"Step {MISSING_STEP} "))
        assert NO_IMAGE_TEXT in item.text()
        assert "不用做" in item.toolTip()
        assert item.foreground().color().getRgb()[:3] == (150, 150, 156)
        plain = next(timeline.item(i) for i in range(timeline.count())
                     if timeline.item(i).text() == "Step 2")
        assert plain.toolTip() == ""
    finally:
        close_window(win)
