"""U2d: the minors the U2c review left, each against the real window.

1. The ROI reminder bar (editor closed) said "no chassis" for any rectangle
   that would be refused -- a 20 px drag left behind by Esc included.
2. The open editor's bar said "Enter 保存；Esc 先跳过" twice and mentioned a
   Shift+R that only matters after Esc.
3. A good drag after a refused one left the refusal in the status line.
4. With every card row done the guide said "按 Space" while problems in the
   card's pane made Space refuse -- and a click on one of them did nothing.
5. The pane listed notes Space accepts (``empty_visible``...) among the
   problems that block it, and the status line said "见任务卡" for either.
6. A conflict verdict announced a frame change without the frame's problems,
   so the card's pane went empty.
7. Alt+Enter on a part with no keyframe here wrote an override the compiler
   still called ``missing_shape``: the row stayed open.
8. A step typed ``ignore`` that has an image read as a plain "Step N".
9. The window opened silent about its first frame's pane: the card emitted
   before anything listened.

Round 2:

R1. The card's header and its ✔ row still said "直接 Space" over a pane of
    problems that made Space refuse.
R2. A refused conflict verdict reached the status line as "conflict N
    refused: " -- the reason travelled only on ``sigProblems``.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
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
)
from tda.core import truth_refusals
from tda.ui import app_actions as A
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
from tda.ui.panels.taskcard import (
    ARRIVAL_TITLE,
    INSTANCE_ROLE,
    NOTES_HEADING,
    PROBLEM_SENTENCES,
    VIEW_ROLE,
    TaskCardPanel,
    explain_code,
)


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


def pane_item(win: MainWindow, instance: str):
    """The line of the card's problems pane about ``instance``."""
    lw = win.task_card._problems_list
    return next(lw.item(i) for i in range(lw.count())
                if lw.item(i).data(INSTANCE_ROLE) == instance)


def press(win: MainWindow, name: str) -> None:
    win.dispatch(A.action_named(name))
    QApplication.processEvents()


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
        win.task_card._problems_list.itemClicked.emit(pane_item(win, UNLISTED))
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
        # the list lives with the other refusals since U2e (truth_refusals)
        monkeypatch.setattr(truth_refusals, "BLOCKING_PROBLEMS",
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
        item = pane_item(win, SPLIT_ROW)
        card._problems_list.itemClicked.emit(item)
        QApplication.processEvents()
        assert win.instances.selected_instance() == SPLIT_ROW
        assert session.editing_instance is None, "nothing to draw for this one"
        assert win.status_message() == item.text()
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 5: blockers first, notes under their own heading, "见任务卡" for blockers
# --------------------------------------------------------------------------- #
def test_notes_follow_the_blockers_under_their_own_heading(qapp):
    panel = TaskCardPanel()
    shown: list[int] = []
    panel.sigProblemsShown.connect(shown.append)
    panel._show_arrival(["empty_visible:a.01", "missing_shape:psu.01",
                         "zorder_missing:b.01/main", "zorder_cycle:c.01,d.01",
                         "bench_missing:e.01", "pose_segment_ambiguous:f.01",
                         "shape_size_mismatch:g.01/main"])
    rows = panel.problem_rows()
    assert [r["code"] for r in rows] == [
        "missing_shape:psu.01", "zorder_cycle:c.01,d.01", "shape_size_mismatch:g.01/main",
        "empty_visible:a.01", "zorder_missing:b.01/main", "bench_missing:e.01",
        "pose_segment_ambiguous:f.01"]
    assert [r["note"] for r in rows] == [False] * 3 + [True] * 4
    lw = panel._problems_list
    lines = [lw.item(i).text() for i in range(lw.count())]
    assert lines.index(NOTES_HEADING) == 3, "the heading sits between the two"
    assert NOTES_HEADING.startswith("提示（不挡 Space）")
    assert lw.item(3).flags() == Qt.ItemFlag.NoItemFlags   # not a problem to pick
    assert NOTES_HEADING not in panel.problems()
    assert shown == [3] and panel.problem_count() == 3

    panel._show_arrival(["empty_visible:a.01"])            # notes alone
    assert panel.problems_visible() and panel.notes_heading_shown()
    assert shown[-1] == 0 and panel.problem_count() == 0

    panel._show_arrival(["missing_shape:psu.01"])          # blockers alone
    assert not panel.notes_heading_shown()


def test_every_sentence_says_what_to_do():
    wants = {
        "bench_missing:": "按 R", "shape_size_mismatch:": "重画",
        "zorder_cycle:": "Ctrl+Z", "zorder_missing:": "Ctrl+↑/↓",
        "empty_visible:": "选中它按 3（完全遮挡）", "pose_segment_ambiguous:": "Ctrl+Shift+B",
        "missing_shape:": "画出来",
        # U2e: the refusals that are not compiler problems
        "open_conflict:": "K 保留旧的 / N 采用新的", "frozen_disagreement:": "按 F5",
        "inputs_changed:": "再按一次 Space",
    }
    assert {prefix for prefix, _ in PROBLEM_SENTENCES} == set(wants)
    for prefix, fix in wants.items():
        assert fix in explain_code(prefix + ("7/x.01" if prefix == "open_conflict:"
                                             else "x.01")), prefix
    assert "完全挡住" in explain_code("empty_visible:x.01")


def test_notes_alone_neither_point_the_status_line_nor_block_space(qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    covered = [k for k in chassis_instances(session, LAST_STEP - 1) if k != UNLISTED][0]
    draw(session, UNLISTED, 0)            # on top of `covered`'s rectangle
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        session.goto(LAST_STEP, force=True)
        session.goto(LAST_STEP - 1, force=True)
        QApplication.processEvents()
        assert session.current_problems() == [f"empty_visible:{covered}"]
        rows = win.task_card.problem_rows()
        assert [(r["code"], r["note"]) for r in rows] == [(f"empty_visible:{covered}", True)]
        assert win.task_card.notes_heading_shown()
        assert win.task_card.problems_visible()
        assert "见任务卡" not in win.status_message()
        assert win.guide_plan().phase == G.PHASE_CONFIRM   # a note blocks nothing

        # the fix its sentence names: select it (a click on the line) and press 3
        win.task_card._problems_list.itemClicked.emit(pane_item(win, covered))
        assert win.instances.selected_instance() == covered
        press(win, "visibility_3")
        assert f"empty_visible:{covered}" not in session.current_problems()
        assert not win.task_card.problems_visible()
        assert win.act_confirm() is True
    finally:
        close_window(win)


def test_a_blocker_on_arrival_says_to_fix_it_first(qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        session.goto(LAST_STEP, force=True)
        session.goto(LAST_STEP - 1, force=True)
        QApplication.processEvents()
        assert win.task_card._problems_label.text() == ARRIVAL_TITLE
        assert "1 个问题要先处理 — 见任务卡" in win.status_message()
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 9: the frame the window opens on is an arrival like any other
# --------------------------------------------------------------------------- #
def test_the_window_opens_saying_what_blocks_its_first_frame(qapp, tmp_path):
    """The card filled its pane before the status line listened (as in ``main``)."""
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session)
    try:
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]
        assert "1 个问题要先处理 — 见任务卡" in win.status_message()
        assert win.guide_facts().blockers == 1          # the ROI question comes first
    finally:
        close_window(win)


def test_the_window_opens_quiet_over_notes_alone(qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    draw(session, UNLISTED, 0)            # covers the first part: a note, no blocker
    win = open_window(tmp_path, session=session)
    try:
        assert win.task_card.problems_visible() and win.task_card.problem_count() == 0
        assert "见任务卡" not in win.status_message()
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 6: a conflict verdict leaves the card's pane right
# --------------------------------------------------------------------------- #
def test_a_conflict_verdict_announces_the_frame_with_its_problems(qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    said: list[list[str]] = []
    session.sigProblems.connect(said.append)
    assert session.resolve_conflict(9999, api.RESOLVE_ACCEPT_NEW) == "refused"
    # the frame's own codes first, the verdict after them
    assert said[-2] == session.current_problems() == [f"missing_shape:{UNLISTED}"]
    assert said[-1][0].startswith("conflict 9999 not resolved")


def test_the_pane_survives_a_conflict_verdict(qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]
        win.resolve_conflict(9999, api.RESOLVE_ACCEPT_NEW)   # K / N in Review
        QApplication.processEvents()
        # the frame change emptied the pane and nothing filled it again
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]
        assert win.guide_facts().blockers == 1
        assert "conflict 9999 refused" in win.status_message()   # the verdict is said
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 7: Alt+Enter (只改这一帧) needs a shape under it
# --------------------------------------------------------------------------- #
def paint(win: MainWindow, dx: int = 8) -> None:
    """One brush stroke across the middle of the canvas (the window is shown)."""
    if win._tool_name in ("sam_point", "sam_box"):
        win.act_tool("brush")
    viewport = win.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(dx, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(dx, 0))
    QApplication.processEvents()


def test_the_session_refuses_a_frame_override_with_no_shape_under_it(qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    key = session.current()
    assert session.keyframe_applies(UNLISTED) is False
    session.begin_edit(UNLISTED)
    session.set_editing_mask(cell(41))
    with pytest.raises(api.SessionRefusal, match=re.escape(api.OVERRIDE_NEEDS_SHAPE)):
        session.commit_edit(api.SCOPE_FRAME_OVERRIDE)
    assert UNLISTED not in session.db.frame_overrides(key), "nothing was written"
    assert session.editing_instance == UNLISTED, "the layer is kept"

    session.commit_edit(api.SCOPE_KEYFRAME)          # Enter draws it ...
    session.clear_edit()
    assert session.keyframe_applies(UNLISTED) is True
    draw(session, UNLISTED, 42, api.SCOPE_FRAME_OVERRIDE)   # ... and now it is allowed
    assert session.db.frame_overrides(key)[UNLISTED].visible_rle is not None


def test_alt_enter_on_an_undrawn_part_is_greyed_and_refused_with_one_reason(
        qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session, show=True)
    try:
        answer_roi(win)
        win.on_request_edit(UNLISTED)
        paint(win)
        assert win.layer_facts()[2], "the layer holds uncommitted pixels"
        button = win.palette.button("commit_override")
        assert not button.isEnabled()
        assert button.reason() == api.OVERRIDE_NEEDS_SHAPE
        assert win.palette.button("commit_split").isEnabled()   # only Alt+Enter

        ops = len(win.db.ops(DESKTOP, VIEW, limit=10_000))
        win.report("")
        press(win, "commit_override")                    # the key: same guard
        assert api.OVERRIDE_NEEDS_SHAPE in win.status_message()
        assert len(win.db.ops(DESKTOP, VIEW, limit=10_000)) == ops, "the key acted"
        assert session.editing_instance == UNLISTED and win.layer_facts()[2]

        win.act_commit_override()                        # past the guard: the session
        assert api.OVERRIDE_NEEDS_SHAPE in win.status_message()
        assert UNLISTED not in win.db.frame_overrides(session.current())
    finally:
        close_window(win)


def test_alt_enter_is_live_on_a_part_that_has_a_shape_here(qapp, tmp_path):
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session, show=True)
    try:
        answer_roi(win)
        win.on_request_edit(SPLIT_ROW)                   # drawn on this frame
        paint(win)
        button = win.palette.button("commit_override")
        assert button.isEnabled(), button.reason()
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 8: a step typed ignore that has an image says 跳过
# --------------------------------------------------------------------------- #
FEW_STEPS = 5
IGNORED, IGNORED_AND_MISSING = 2, 4


def test_an_ignored_step_with_an_image_says_skip(qapp, tmp_path):
    """D66 oak1/oak2/rs step 13: navigation and counts skip it, the row said "Step 13"."""
    from tda.core.model import StepType
    from tda.core.truth_inputs import annotatable_steps
    from tda.ui.panels.timeline import NO_IMAGE_TEXT, SKIPPED_TEXT, SKIPPED_TIP, STEP_ROLE

    session = make_session(tmp_path, last_step=FEW_STEPS, missing=(IGNORED_AND_MISSING,))
    db = session.db
    steps = db.steps(DESKTOP)
    for rec in steps:
        if rec.step in (IGNORED, IGNORED_AND_MISSING):
            rec.step_type = StepType.IGNORE.value
    db.replace_steps(DESKTOP, steps, db.actions(DESKTOP))
    session.open(DESKTOP, VIEW)                      # the walk is read on open
    walked = annotatable_steps(db, DESKTOP, VIEW, session.steps())
    assert walked == session.available_steps() == [1, 3, 5]
    assert session.image_path(IGNORED) is not None, "it has an image"

    win = open_window(tmp_path, session=session)
    try:
        timeline = win.timeline.list_widget()
        items = {int(timeline.item(i).data(STEP_ROLE)): timeline.item(i)
                 for i in range(timeline.count())}
        skipped = items[IGNORED]
        assert skipped.text() == f"Step {IGNORED} · {SKIPPED_TEXT}"
        assert skipped.toolTip() == SKIPPED_TIP
        assert "标了 ignore" in SKIPPED_TIP and "不算进 [已确认/总数]" in SKIPPED_TIP
        assert skipped.foreground().color().getRgb()[:3] == (150, 150, 156)
        # no image wins: that row says why it has nothing to draw
        assert NO_IMAGE_TEXT in items[IGNORED_AND_MISSING].text()
        # every step the walk takes is a plain row
        for step in walked:
            assert items[step].text() == f"Step {step}" and items[step].toolTip() == ""
        # ... and it stays so when the statuses are re-read
        win.timeline.refresh_statuses()
        assert skipped.text() == f"Step {IGNORED} · {SKIPPED_TEXT}"
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# round 2, item 1: no surface says Space while Space would be refused
# --------------------------------------------------------------------------- #
#: Reached from 6, step 5 only fastens a screw: its card is one ≡ row, and
#: its header says "不用画，直接 Space".
STATE_ONLY_STEP = 5


def space_frame(tmp_path: Path, *, blocked: bool, confirm_row: bool = False):
    """Step 5 of the scene: every part drawn there, or only what step 14 has.

    ``confirm_row`` types the step ``dupli``, whose card is one ✔ row.
    """
    from tda.core.model import StepType

    session = make_session(tmp_path)
    if confirm_row:
        db = session.db
        steps = db.steps(DESKTOP)
        for rec in steps:
            if rec.step == STATE_ONLY_STEP:
                rec.step_type = StepType.DUPLI.value
        db.replace_steps(DESKTOP, steps, db.actions(DESKTOP))
        session.open(DESKTOP, VIEW)
    # step 14's parts only: what came back in between has no shape at step 5
    seed_shapes(session, LAST_STEP if blocked else STATE_ONLY_STEP)
    session.goto(STATE_ONLY_STEP + 1, force=True)
    session.goto(STATE_ONLY_STEP, force=True)
    return session


def surfaces(win: MainWindow) -> tuple[str, list[str], G.GuidePlan]:
    """The card's header, its row sentences, and the guide's plan."""
    answer_roi(win)
    QApplication.processEvents()
    views = [win.task_card.list_widget().item(i).data(VIEW_ROLE)
             for i in range(win.task_card.list_widget().count())]
    return win.task_card.header_text(), [v["sentence"] for v in views], win.guide_plan()


def says_space(text: str) -> bool:
    return "直接 Space" in text or "按 Space" in text


@pytest.mark.parametrize("confirm_row", [False, True], ids=["state_only", "confirm_row"])
def test_header_row_and_guide_agree_when_the_pane_blocks_space(qapp, tmp_path,
                                                                 confirm_row):
    session = space_frame(tmp_path, blocked=True, confirm_row=confirm_row)
    win = open_window(tmp_path, session=session)
    try:
        header, sentences, plan = surfaces(win)
        n = win.task_card.problem_count()
        assert n > 0 and n == win.guide_facts().blockers
        blocked = f"但下面还有 {n} 个问题挡住 Space（见下方）"
        assert header.endswith(blocked), header
        assert not says_space(header)
        if confirm_row:
            assert sentences == [f"这一帧不用画，{blocked}"]
        assert not any(says_space(s) for s in sentences), sentences
        assert plan.phase == G.PHASE_BLOCKED
        assert f"{n} 个问题挡住 Space" in plan.now and not says_space(plan.now)
        assert not any(says_space(text) for _state, text in plan.steps), plan.steps
        # ... and Space does refuse; afterwards all three still agree
        assert win.act_confirm() is False
        m = win.task_card.problem_count()
        assert win.task_card.header_text().endswith(f"但下面还有 {m} 个问题挡住 Space（见下方）")
        assert f"{m} 个问题挡住 Space" in win.guide_plan().now
    finally:
        close_window(win)


@pytest.mark.parametrize("confirm_row", [False, True], ids=["state_only", "confirm_row"])
def test_header_row_and_guide_say_space_when_nothing_blocks(qapp, tmp_path, confirm_row):
    session = space_frame(tmp_path, blocked=False, confirm_row=confirm_row)
    win = open_window(tmp_path, session=session)
    try:
        header, sentences, plan = surfaces(win)
        assert win.task_card.problem_count() == 0 == win.guide_facts().blockers
        assert header.endswith("直接 Space"), header
        if confirm_row:
            assert sentences == ["这一帧不用画，直接 Space"]
        assert plan.phase == G.PHASE_CONFIRM and "按 Space" in plan.now
        assert win.act_confirm() is True
    finally:
        close_window(win)


def test_the_card_resays_itself_when_the_pane_changes(qapp, tmp_path):
    """The header follows the pane, not only the frame: drawing the last
    blocker turns "挡住 Space" back into "直接 Space" without leaving the frame."""
    session = rows_done_but_one_unlisted(tmp_path)
    session.goto(LAST_STEP, force=True)
    draw(session, UNLISTED, 41)              # the start frame, every part drawn
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        card = win.task_card
        assert not card.problems_visible()
        assert card.header_text().endswith("直接 Space 确认"), card.header_text()
        # what the pane is handed decides, whatever put it there
        # (a wrong-size shape is a blocker the start card has no row for)
        card._show_problems(["shape_size_mismatch:chassis/main"], "Problems")
        assert card.header_text().endswith("但下面还有 1 个问题挡住 Space（见下方）")
        assert win.guide_plan().phase == G.PHASE_BLOCKED
        card._show_problems(["empty_visible:chassis"], "Problems")     # a note only
        assert card.header_text().endswith("直接 Space 确认")
        assert win.guide_plan().phase == G.PHASE_CONFIRM
    finally:
        close_window(win)


def test_the_pure_sentences_carry_the_count():
    from tda.ui.panels.taskcard import card_header, row_view

    assert card_header(5, 6, [api.KIND_STATE_ONLY], blockers=3).endswith(
        "不用画，但下面还有 3 个问题挡住 Space（见下方）")
    assert card_header(5, 6, [api.KIND_CONFIRM], blockers=1) == (
        "第 5 帧：这一帧不用画，但下面还有 1 个问题挡住 Space（见下方）")
    assert card_header(14, None, [api.KIND_ADD_SHAPE], first=1, last=14, done=[True],
                       blockers=2).endswith("都画好了，但下面还有 2 个问题挡住 Space（见下方）")
    assert card_header(5, 6, [api.KIND_CONFIRM]) == "第 5 帧：这一帧不用画，直接 Space"
    row = {"instance": "step 5", "kind": api.KIND_CONFIRM, "done": False}
    assert row_view(row)["sentence"] == "这一帧不用画，直接 Space"
    assert row_view(row, blockers=4)["sentence"] == (
        "这一帧不用画，但下面还有 4 个问题挡住 Space（见下方）")


# --------------------------------------------------------------------------- #
# round 2, item 2: a refused or superseded verdict says why
# --------------------------------------------------------------------------- #
def superseded_conflict(tmp_path: Path) -> tuple:
    """A real queued conflict whose inputs moved on before anybody settled it."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False          # the re-check is run here, in order
    seed_shapes(session, LAST_STEP - 1)
    session.goto(LAST_STEP, force=True)
    session.goto(LAST_STEP - 1, force=True)
    draw(session, SPLIT_ROW, 40, api.SCOPE_SPLIT)
    assert session.confirm_frame() is True                # step 13 verified
    session.goto(LAST_STEP - 1, force=True)
    draw(session, "chassis", 50)                          # disagrees with it
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    cid = session.queues()[api.QUEUE_CONFLICTS][0]["id"]
    draw(session, "chassis", 51)                          # ... and moves on again
    return session, cid


def test_the_verdict_carries_its_reason(qapp, tmp_path):
    session, cid = superseded_conflict(tmp_path)
    verdict = session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW)
    assert verdict == "superseded" and isinstance(verdict, api.Verdict)
    assert "stale" in verdict.reason
    again = session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW)
    assert again == "refused" and "already resolved" in again.reason
    unknown = session.resolve_conflict(9999, api.RESOLVE_ACCEPT_NEW)
    assert unknown == "refused" and unknown.reason


def test_a_refused_verdict_shows_its_reason_in_the_status_line(qapp, tmp_path):
    session, cid = superseded_conflict(tmp_path)
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        win.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW)          # N in Review
        said = win.status_message()
        assert said.startswith(f"conflict {cid}: superseded and re-queued"), said
        assert "stale" in said, "the reason reached the status line"

        win.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW)          # the same again
        said = win.status_message()
        assert said.startswith(f"conflict {cid} refused: "), said
        assert said != f"conflict {cid} refused: " and "already resolved" in said
        assert "见任务卡" not in said
    finally:
        close_window(win)


def test_a_failed_recheck_still_says_so_in_the_status_line(qapp, tmp_path):
    """The other non-code line that went out on sigProblems: it has its own
    signal (``sigSweepError``), and U2c took away the slot that buried it."""
    session = rows_done_but_one_unlisted(tmp_path)
    win = open_window(tmp_path, session=session)
    try:
        answer_roi(win)
        session._on_sweep_error(3, "the truth service fell over")
        QApplication.processEvents()
        said = win.status_message()
        assert "re-check of step 3 failed: the truth service fell over" in said, said
        assert "F5" in said and "见任务卡" not in said
        assert pane_codes(win) == [f"missing_shape:{UNLISTED}"]   # the pane is untouched
    finally:
        close_window(win)
