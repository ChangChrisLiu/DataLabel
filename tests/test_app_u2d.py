"""U2d: the minors the U2c review left, each against the real window.

1. The ROI reminder bar (editor closed) said "no chassis" for any rectangle
   that would be refused -- a 20 px drag left behind by Esc included.
2. The open editor's bar said "Enter 保存；Esc 先跳过" twice and mentioned a
   Shift+R that only matters after Esc.
3. A good drag after a refused one left the refusal in the status line.
4. With every card row done the guide said "按 Space" while problems in the
   card's pane made Space refuse -- and a click on one of them did nothing.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import (
    LAST_STEP,
    StubSamQueue,
    cell,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.core import truth_verify
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_roi import (
    NO_CHASSIS_CLOSED,
    NO_CHASSIS_DRAG,
    NO_CHASSIS_FOUND,
    ROI_BAR_KEYS,
    ROI_BAR_KEYS_STORED,
    ROI_BOX_READY,
    ROI_TOO_SMALL,
    roi_min_side,
)
from tda.ui.panels.taskcard import INSTANCE_ROLE


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


def roi_bar_text(win: MainWindow) -> str:
    return win.roi_bar.label.text() if win.roi_bar.isVisibleTo(win) else ""


def answer_roi(win: MainWindow) -> None:
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


def pane_codes(win: MainWindow) -> list[str]:
    return [r["code"] for r in win.task_card.problem_rows()]


#: At LAST_STEP - 1 nothing about the PSU changes, so no card row names it;
#: the one row there is ram_latch.01's ✂.
UNLISTED = "psu.01"
SPLIT_ROW = "ram_latch.01"


def rows_done_but_one_unlisted(tmp_path: Path):
    """LAST_STEP - 1 with its only card row done and psu.01 still undrawn."""
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP - 1, skip=(UNLISTED,))
    session.goto(LAST_STEP, force=True)
    session.goto(LAST_STEP - 1, force=True)
    session.begin_edit(SPLIT_ROW)
    session.set_editing_mask(cell(40))
    session.commit_edit(api.SCOPE_SPLIT)
    session.clear_edit()
    assert [(r["instance"], r["done"]) for r in session.task_card()] == [(SPLIT_ROW, True)]
    assert session.current_problems() == [f"missing_shape:{UNLISTED}"]
    return session


def draw(session, instance: str, index: int, scope: str = api.SCOPE_KEYFRAME) -> None:
    session.begin_edit(instance)
    session.set_editing_mask(cell(index))
    session.commit_edit(scope)
    session.clear_edit()


# --------------------------------------------------------------------------- #
# item 1: the reminder bar names the refusal 确认建议框 would answer with
# --------------------------------------------------------------------------- #
def test_a_too_small_drag_left_behind_by_esc_is_called_too_small(qapp, tmp_path):
    """D13/rs: drag a 20 px box, Esc -- the bar used to say "no chassis"."""
    win = open_window(tmp_path, show=True)
    try:
        win.wait_for_roi_proposal()
        assert win.roi_editing
        win.on_roi_box((10.0, 10.0, 14.0, 14.0))
        win.act_clear_edit()                       # Esc: the editor closes
        assert win.roi_editing is False and win.roi_unanswered()

        too_small = ROI_TOO_SMALL.format(floor=roi_min_side(win.overlay.hw))
        bar = roi_bar_text(win)
        assert too_small in bar, bar
        assert NO_CHASSIS_CLOSED not in bar and "没找到机箱" not in bar
        # ... which is exactly what the button answers
        win.act_accept_roi_proposal()
        assert win.status_message() == too_small
        assert win.roi() is None
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 2: the open editor's bar says each thing once, the keys at the end
# --------------------------------------------------------------------------- #
def says_the_keys_once(bar: str, keys: str = ROI_BAR_KEYS) -> None:
    assert bar.endswith(keys), bar
    assert bar.count("Enter") == keys.count("Enter"), bar
    assert bar.count("Esc") == keys.count("Esc"), bar
    assert "Shift+R" not in bar, bar
    for how in ("拖边或角调整", "框里按住整体移动", "空白处拖动重画", "差异图"):
        assert bar.count(how) == 1, (how, bar)


def test_the_open_editor_bar_says_the_keys_once(qapp, tmp_path):
    win = open_window(tmp_path, show=True)
    try:
        win.wait_for_roi_proposal()
        assert win.roi_editing and win.roi() is None
        says_the_keys_once(roi_bar_text(win))

        win.on_roi_box((10.0, 10.0, 14.0, 14.0))          # too small: a reason
        bar = roi_bar_text(win)
        assert "太小" in bar
        says_the_keys_once(bar)
    finally:
        close_window(win)


def test_a_failed_proposal_is_one_reason_and_one_ending(qapp, tmp_path, monkeypatch):
    """The detector's whole frame: the reason once, no second Enter/Esc."""
    from tda.ui import app_roi_worker

    monkeypatch.setattr(app_roi_worker, "suggest_roi_over",
                        lambda images, view: (0, 0, 64, 64))
    win = open_window(tmp_path, show=True)
    try:
        assert win.wait_for_roi_proposal() is True
        bar = roi_bar_text(win)
        assert bar.count(NO_CHASSIS_DRAG) == 1
        says_the_keys_once(bar)
    finally:
        close_window(win)


def test_the_stored_editor_bar_says_esc_keeps_it(qapp, tmp_path):
    win = open_window(tmp_path, show=True)
    try:
        win.wait_for_roi_proposal()
        win.act_commit()                                  # store the proposal
        assert win.roi() is not None
        win.act_edit_roi()                                # Shift+R: open it again
        assert win.roi_editing
        bar = roi_bar_text(win)
        says_the_keys_once(bar, ROI_BAR_KEYS_STORED)
        assert "先跳过" not in bar

        win.on_roi_box((10.0, 10.0, 14.0, 14.0))
        bar = roi_bar_text(win)
        assert "太小" in bar
        says_the_keys_once(bar, ROI_BAR_KEYS_STORED)
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 3: a storable drag replaces the refusal the last one left
# --------------------------------------------------------------------------- #
def test_a_good_drag_after_refused_ones_says_it_is_ready(qapp, tmp_path, monkeypatch):
    from tda.ui import app_roi_worker

    monkeypatch.setattr(app_roi_worker, "suggest_roi_over",
                        lambda images, view: (0, 0, 64, 64))
    win = open_window(tmp_path, show=True)
    try:
        assert win.wait_for_roi_proposal() is True
        assert NO_CHASSIS_FOUND in win.status_message()

        win.on_roi_box((10.0, 10.0, 14.0, 14.0))
        too_small = ROI_TOO_SMALL.format(floor=roi_min_side(win.overlay.hw))
        assert win.status_message() == too_small

        win.on_roi_box((8.0, 8.0, 56.0, 56.0))
        assert win.status_message() == ROI_BOX_READY
        assert win.roi_refusal(win.roi_draft) == ""
        win.act_commit()                                  # ... and it is: Enter stores it
        assert win.roi() == (8, 8, 56, 56)
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 4: the guide never says "按 Space" while Space would be refused
# --------------------------------------------------------------------------- #
DONE_ROW = G.CardItem("chassis", api.KIND_ADD_SHAPE, True, "机箱 chassis")
ALL_DONE = G.GuideFacts(mode="annotate", step=13, neighbour=14, roi_stored=True,
                        items=(DONE_ROW,))


def test_the_plan_with_blockers_left_points_at_the_pane_not_at_space():
    plan = G.plan_for(replace(ALL_DONE, blockers=2))
    assert plan.phase == G.PHASE_BLOCKED
    assert "任务卡做完了，但下面还有 2 个问题挡住 Space：单击一条去处理" in plan.now
    assert "按 Space" not in plan.now
    assert plan.action == "", "no button is the next one"
    # the numbered list agrees: its last line is about the problems
    assert plan.steps[-1][0] == G.NOW and "2 个问题" in plan.steps[-1][1]
    assert all("Space 确认" not in text for _state, text in plan.steps)
    # ... and with none left it is Space again
    assert G.plan_for(ALL_DONE).phase == G.PHASE_CONFIRM
    assert G.plan_for(ALL_DONE).action == "confirm"


def test_open_rows_come_before_the_pane():
    todo = G.CardItem("cover.01", api.KIND_ADD_SHAPE, False, "导风罩 cover.01")
    plan = G.plan_for(replace(ALL_DONE, items=(DONE_ROW, todo), blockers=3))
    assert plan.phase == G.PHASE_PICK


def test_rows_done_and_a_blocker_left_the_window_does_not_say_space(qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]
        assert win.guide_facts().blockers == 1
        plan = win.guide_plan()
        assert plan.phase == G.PHASE_BLOCKED, plan
        assert "1 个问题挡住 Space" in plan.now
        assert win.guide.plan() == plan, "the panel shows what the window plans"
        assert win.act_confirm() is False              # ... and Space does refuse

        # a click on the problem goes to deal with it: here, drawing psu.01
        lw = win.task_card._problems_list
        missing = next(lw.item(i) for i in range(lw.count())
                       if lw.item(i).data(INSTANCE_ROLE) == UNLISTED)
        lw.itemClicked.emit(missing)
        QApplication.processEvents()
        assert session.editing_instance == UNLISTED
        win.act_clear_edit()

        draw(session, UNLISTED, 41)                     # drawn: nothing blocks now
        QApplication.processEvents()
        assert win.guide_facts().blockers == 0
        assert win.guide_plan().phase == G.PHASE_CONFIRM
        assert win.act_confirm() is True
    finally:
        close_window(win)


def test_the_guide_and_confirm_read_one_list_of_blocking_codes(qapp, tmp_path,
                                                            monkeypatch):
    """Take missing_shape off the list: the card, the guide and Space all follow."""
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        assert win.guide_plan().phase == G.PHASE_BLOCKED
        monkeypatch.setattr(truth_verify, "BLOCKING_PROBLEMS",
                            ("zorder_cycle:", "shape_size_mismatch:"))
        session.goto(LAST_STEP - 1, force=True)         # arrive again
        QApplication.processEvents()
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]   # still listed ...
        assert win.task_card.problem_count() == 0                 # ... as a note
        assert win.guide_plan().phase == G.PHASE_CONFIRM
        assert win.act_confirm() is True
    finally:
        close_window(win)


def test_a_click_on_a_problem_that_is_not_a_missing_shape_selects_its_part(
        qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        card = win.task_card
        card._show_problems([f"empty_visible:{SPLIT_ROW}"], "Problems")
        lw = card._problems_list
        lw.itemClicked.emit(lw.item(0))
        QApplication.processEvents()
        assert win.instances.selected_instance() == SPLIT_ROW
        assert session.editing_instance is None, "nothing to draw for this one"
        assert win.status_message() == lw.item(0).text()
    finally:
        close_window(win)
