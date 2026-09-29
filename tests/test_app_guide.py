"""A window a first-time annotator understands (task U2b, user trial #2).

The trial ended with four complaints: nobody could tell what the task card's
other items were, a double-click "did nothing" (it had started an edit on an
empty layer and nothing on screen changed), the operation guide was a
document instead of something on the window, and there were no buttons for
the keys.  These tests pin the answers:

* the tool palette -- every button *is* its key: the label comes from
  ``ACTIONS``, the click runs the same slot, nothing takes the keyboard, and a
  button that cannot run says why;
* the brush size -- one number for the slider, the spin box, ``[``/``]``, the
  cursor ring and the badge;
* the "现在做什么 / What now" dock -- one phase per state;
* the task card -- a header per frame, rows in the annotator's words, a single
  click that starts the edit and marks it, a hover that outlines and changes
  nothing;
* the canvas banner -- visible feedback the moment an edit starts, which
  never takes a click.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
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
    cell,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.core import masks
from tda.core.model import ShapeKeyframe, ShapePart
from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_guide import BANNER_EMPTY, BANNER_LOADED, BANNER_PIXELS, HINT_DWELL_MS
from tda.ui.app_view import TOOL_LABELS
from tda.ui.panels.palette import (
    RADIUS_MAX,
    RADIUS_MIN,
    radius_to_slider,
    slider_to_radius,
)
from tda.ui.panels.taskcard import CHIP_EDITING, card_header, row_view

#: The buttons that are live in Annotate mode.
PALETTE = A.PALETTE_TOOLS + A.PALETTE_ACTIONS
#: Every button on the strip, Review's included (round 1, item 3).
EVERY_BUTTON = tuple(dict.fromkeys(PALETTE + A.PALETTE_REVIEW))


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, sam: bool = True, **kwargs) -> MainWindow:
    session = make_session(tmp_path, **kwargs)
    win = MainWindow(session, make_paths(tmp_path), "tester",
                     sam_queue=StubSamQueue() if sam else None)
    win.resize(1400, 900)
    return win


@pytest.fixture
def window(qapp, tmp_path):
    win = open_window(tmp_path)
    yield win
    close_window(win)


def answer_roi(win: MainWindow) -> None:
    """Save the proposed rectangle, as the trial annotator did.

    The proposal is measured on a worker, so it is waited for; if none came
    back the rectangle is skipped (``Esc``) instead.
    """
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


def first_row(win: MainWindow) -> tuple[int, str]:
    """``(row, instance)`` of the first drawable row on the card."""
    for index, row in enumerate(win.task_card.rows()):
        if row.get("kind") == api.KIND_ADD_SHAPE and not row.get("done"):
            return index, str(row["instance"])
    raise AssertionError("the card has nothing to draw")


def click_row(win: MainWindow, index: int, double: bool = False) -> None:
    """A real click on one card row, through the list's viewport."""
    lw = win.task_card.list_widget()
    lw.scrollToItem(lw.item(index))
    QApplication.processEvents()
    centre = lw.visualItemRect(lw.item(index)).center()
    QTest.mouseClick(lw.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre, 10)
    if double:
        QTest.mouseDClick(lw.viewport(), Qt.MouseButton.LeftButton,
                          Qt.KeyboardModifier.NoModifier, centre, 10)
    QApplication.processEvents()


def brush_stroke(win: MainWindow) -> None:
    """Pick the brush and drag across the canvas."""
    win.act_tool("brush")
    viewport = win.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(8, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(8, 0))
    QApplication.processEvents()


def phase(win: MainWindow) -> str:
    return win.guide.plan().phase


# --------------------------------------------------------------------------- #
# the palette is the keyboard, as buttons (ruling U2b-1)
# --------------------------------------------------------------------------- #
def test_every_palette_button_is_an_action_with_its_own_key(window):
    buttons = window.palette.buttons()
    assert set(buttons) == set(EVERY_BUTTON)
    for name in EVERY_BUTTON:
        action = A.action_named(name)
        button = buttons[name]
        assert action.short, f"{name} has no palette caption"
        assert (button.name, button.note) == A.short_parts(action)
        assert button.key == A.key_caption(action)
        assert button.key.replace("按住 ", "") in action.keys
        assert action.label_zh in button.toolTip()
        for key in action.keys:
            assert key in button.toolTip()


def test_the_palette_names_the_tools_the_way_the_badge_does(window):
    """The badge and the checked button are one state: one name for it."""
    for name in A.PALETTE_TOOLS:
        action = A.action_named(name)
        tool = action.args[0] if action.slot == "act_tool" else "roi"
        assert TOOL_LABELS[tool][0] == A.short_parts(action)[0]


def test_r_is_the_bench_box_and_shift_r_is_the_chassis_range(window):
    """The trial annotator pressed R expecting the ROI (ruling U2b-5)."""
    bench = window.palette.button("tool_bench_box")
    roi = window.palette.button("edit_roi")
    assert (bench.name, bench.key) == ("台面框", "R")
    assert "已拆下、放在台面上的零件" in bench.note
    assert (roi.name, roi.key) == ("机箱范围 ROI", "Shift+R")
    assert "台面框（已拆下、放在台面上的零件）" in A.cheat_sheet_html()
    assert "机箱范围 ROI" in A.cheat_sheet_html()


def test_nothing_on_the_palette_takes_the_keyboard(window):
    """Nothing but the brush-size number, which is typed into (round 1, item 5)."""
    widgets = [window.palette, window.palette.radius_slider]
    widgets += list(window.palette.buttons().values())
    for widget in widgets:
        assert widget.focusPolicy() == Qt.FocusPolicy.NoFocus, type(widget).__name__
    assert window.palette.radius_spin.focusPolicy() == Qt.FocusPolicy.ClickFocus
    for widget in (window.guide, window.guide.cheat_button,
                   window.task_card.list_widget()):
        assert widget.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_a_palette_click_leaves_the_keyboard_on_the_canvas(window):
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    window.canvas.setFocus()
    button = window.palette.button("tool_eraser")
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert window._tool_name == "eraser"
    assert QApplication.focusWidget() is window.canvas


@pytest.mark.parametrize("name", PALETTE)
def test_a_palette_click_runs_what_the_key_runs(window, name):
    """Same slot, same arguments -- the button cannot do what the key does not."""
    action = A.action_named(name)
    calls: list[tuple] = []
    setattr(window, action.slot, lambda *args: calls.append(args))
    button = window.palette.button(name)
    button.setEnabled(True)            # the gate is the slot's; here only the wiring
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    if action.hold:
        assert calls == [(True,), (False,)]
    else:
        assert calls == [tuple(action.args)]


def test_a_palette_click_is_refused_outside_the_actions_mode(window):
    """In Steps mode the palette hides; a stray call still keeps the modes."""
    calls: list[tuple] = []
    window.act_commit = lambda *a: calls.append(a)
    window.set_mode(A.MODE_STEPS)
    window.run_palette_action("commit")
    assert calls == []
    assert window.palette.isHidden()
    window.set_mode(A.MODE_ANNOTATE)
    assert not window.palette.isHidden()


def test_the_palette_goes_through_every_gate_its_key_does(window, monkeypatch):
    """A button is a key: the gates live in the slots, so they hold for clicks."""
    from tda.ui.app_adopt import ADOPT_FIRST
    from tda.ui.app_edit import BLOCK_HINT

    answer_roi(window)
    _row, instance = first_row(window)
    window.on_request_edit(instance)
    brush_stroke(window)
    step = window.session.current().step
    # the dirty-layer gate: confirming steps the frame back, so it is refused
    window.run_palette_action("confirm")
    assert window.session.current().step == step
    assert window.status_message() == BLOCK_HINT
    # the ghost owns every commit key but Enter
    monkeypatch.setattr(window, "showing_draft_ghost", lambda: True)
    committed: list = []
    monkeypatch.setattr(window.session, "commit_edit",
                        lambda *a, **k: committed.append(a) or {})
    window.run_palette_action("commit_split")
    assert committed == [] and window.status_message() == ADOPT_FIRST
    monkeypatch.undo()
    # Review is read-only: a tool button does nothing there
    window.act_clear_edit()
    window.set_mode(A.MODE_REVIEW)
    window.run_palette_action("tool_eraser")
    assert window._tool_name != "eraser"
    window.set_mode(A.MODE_ANNOTATE)


def test_the_roi_rectangle_owns_the_palettes_esc(window):
    """While the rectangle is up, 放弃编辑 answers the rectangle, not a layer."""
    assert window.roi_editing
    window.run_palette_action("clear_edit")
    assert not window.roi_editing
    assert window.roi_unanswered()


def test_a_palette_click_ends_a_stuck_flash_first(window):
    answer_roi(window)
    window.act_step(-1)
    window.act_flash_compare(True)
    if not window.is_flashing():
        pytest.skip("no neighbour image to flash")
    window.run_palette_action("tool_eraser")
    assert not window.is_flashing()
    assert window._tool_name == "eraser"


def test_the_armed_tool_is_the_checked_button(window):
    buttons = window.palette.buttons()
    assert buttons["edit_roi"].isChecked(), "the ROI rectangle is up on a fresh segment"
    answer_roi(window)
    assert buttons["tool_brush"].isChecked()
    window.act_tool("eraser")
    checked = [n for n, b in buttons.items() if b.isCheckable() and b.isChecked()]
    assert checked == ["tool_eraser"]
    assert "橡皮擦" in window.tool_label.text()


def test_a_refused_tool_click_does_not_leave_its_button_checked(qapp, tmp_path):
    win = open_window(tmp_path, sam=False)
    try:
        answer_roi(win)
        button = win.palette.button("tool_sam_point")
        button.setEnabled(True)
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        QApplication.processEvents()
        assert not button.isChecked()
        assert win.palette.button("tool_brush").isChecked()
    finally:
        close_window(win)


def test_sam_buttons_say_why_until_sam_is_ready(qapp, tmp_path):
    win = open_window(tmp_path, sam=False)
    try:
        answer_roi(win)
        for name in ("tool_sam_point", "tool_sam_box"):
            button = win.palette.button(name)
            assert not button.isEnabled()
            assert "SAM" in button.reason() and "现在不能用" in button.toolTip()
        win.set_sam_queue(StubSamQueue(), owns=False)
        assert win.palette.button("tool_sam_point").isEnabled()
        assert win.palette.button("tool_sam_point").reason() == ""
    finally:
        close_window(win)


def test_commit_is_enabled_only_when_there_is_something_to_commit(window):
    answer_roi(window)
    commit = window.palette.button("commit")
    assert not commit.isEnabled()
    assert "没有可提交的修改" in commit.reason()
    assert window.palette.button("confirm").isEnabled()
    _row, instance = first_row(window)
    window.on_request_edit(instance)
    brush_stroke(window)
    assert commit.isEnabled()
    assert window.palette.button("commit_override").isEnabled()
    assert window.palette.button("commit_split").isEnabled()
    confirm = window.palette.button("confirm")
    assert not confirm.isEnabled() and "Enter" in confirm.reason()
    assert window.palette.button("clear_edit").isEnabled()
    assert window.palette.button("undo").isEnabled()


def test_the_roi_rectangle_owns_the_commit_button(window):
    """Enter saves the rectangle while it is up; the button is Enter."""
    assert window.roi_editing
    assert window.palette.button("commit").isEnabled()
    assert window.palette.button("commit").suggested()
    assert not window.palette.button("commit_split").isEnabled()


def test_review_mode_keeps_only_what_works_there(window):
    """Round 1, item 3: the drawing tools and the brush size go, as in Steps;
    what stays is the difference map and the queue's own four keys."""
    answer_roi(window)
    window.set_mode(A.MODE_REVIEW)
    palette = window.palette
    assert not palette.isHidden()
    assert palette.tools_section_hidden()
    for name in A.PALETTE_TOOLS:
        assert palette.button(name).isHidden()
    assert palette.radius_slider.isHidden() and palette.radius_spin.isHidden()
    for name in ("commit", "commit_override", "commit_split", "confirm", "undo",
                 "redo", "flash_compare", "cycle_candidate", "clear_edit"):
        assert palette.button(name).isHidden(), name
    shown = [n for n, b in palette.buttons().items() if not b.isHidden()]
    assert sorted(shown) == sorted(A.PALETTE_REVIEW)
    assert palette.button("toggle_heat").isEnabled()          # D works in Review
    assert palette.button("review_accept").isEnabled()
    keep = palette.button("review_keep_old")
    assert not keep.isEnabled() and "冲突" in keep.reason()   # nothing selected
    window.set_mode(A.MODE_ANNOTATE)
    assert not palette.tools_section_hidden()
    assert all(palette.button(n).isHidden() for n in A.PALETTE_REVIEW
               if n not in A.PALETTE_ACTIONS)


@pytest.mark.parametrize("name,verdict", [("review_keep_old", api.RESOLVE_KEEP_OLD),
                                          ("review_accept_new", api.RESOLVE_ACCEPT_NEW)])
def test_the_palettes_k_and_n_resolve_the_selected_conflict(window, monkeypatch,
                                                              name, verdict):
    """Round 1b: the Review dock's own Keep old / Take new are gone; these are
    the one place, and they settle exactly the conflict the queue selected."""
    resolved: list[tuple] = []
    monkeypatch.setattr(window, "resolve_conflict",
                        lambda cid, resolution: resolved.append((cid, resolution)))
    monkeypatch.setattr(window.review, "selected_conflict", lambda: 7)
    window.set_mode(A.MODE_REVIEW)
    window.refresh_guidance()
    button = window.palette.button(name)
    assert button.isEnabled(), button.reason()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert resolved == [(7, verdict)]


@pytest.mark.parametrize("name", ["review_accept", "review_rework",
                                  "review_keep_old", "review_accept_new"])
def test_the_review_buttons_are_the_review_keys(window, name):
    action = A.action_named(name)
    calls: list[tuple] = []
    setattr(window, action.slot, lambda *args: calls.append(args))
    window.set_mode(A.MODE_REVIEW)
    button = window.palette.button(name)
    button.setEnabled(True)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert calls == [tuple(action.args)]
    assert (button.name, button.key) == (A.short_parts(action)[0], A.key_caption(action))


def test_the_difference_map_button_is_checked_with_the_map(window):
    answer_roi(window)
    button = window.palette.button("toggle_heat")
    window.act_toggle_heat()
    assert button.isChecked()
    window.act_toggle_heat()
    assert not button.isChecked()


def test_the_start_frame_has_nothing_to_flash_against(window):
    answer_roi(window)
    button = window.palette.button("flash_compare")
    assert not button.isEnabled() and "起点" in button.reason()
    window.act_step(-1)
    assert window.palette.button("flash_compare").isEnabled()


# --------------------------------------------------------------------------- #
# one brush size (ruling U2b-1)
# --------------------------------------------------------------------------- #
def test_the_slider_maps_both_ends_exactly_and_is_monotonic():
    assert slider_to_radius(0) == RADIUS_MIN
    assert slider_to_radius(100) == RADIUS_MAX
    values = [slider_to_radius(v) for v in range(101)]
    assert values == sorted(values)
    for radius in (1, 3, 8, 30, 120, 200):
        assert abs(slider_to_radius(radius_to_slider(radius)) - radius) <= max(2, radius // 10)


def test_the_slider_sets_the_brush_ring_and_badge(window):
    answer_roi(window)
    window.palette.radius_slider.setValue(radius_to_slider(30))
    radius = slider_to_radius(radius_to_slider(30))
    assert window.brush.radius == window.eraser.radius == window.occluder.radius == radius
    assert window.palette.radius_spin.value() == radius
    assert f"r={radius}" in window.tool_label.text()
    assert window.canvas.tool_cursor().radius == radius


def test_the_spin_box_sets_the_brush_too(window):
    answer_roi(window)
    window.palette.radius_spin.setValue(17)
    assert window.brush.radius == 17
    assert window.palette.radius_slider.value() == radius_to_slider(17)
    assert "r=17" in window.tool_label.text()


def test_the_bracket_keys_move_the_slider_and_the_spin_box(window):
    answer_roi(window)
    window.set_brush_radius(10)
    window.act_radius(+1)
    assert window.brush.radius == 11
    assert window.palette.radius_spin.value() == 11
    assert window.palette.radius_slider.value() == radius_to_slider(11)
    window.act_radius(-1)
    window.act_radius(-1)
    assert window.palette.radius_spin.value() == 9 == window.brush.radius


def test_the_brush_size_is_clamped_to_the_palettes_range(window):
    answer_roi(window)
    window.set_brush_radius(1)
    window.act_radius(-1)
    assert window.brush.radius == RADIUS_MIN
    window.set_brush_radius(10_000)
    assert window.brush.radius == RADIUS_MAX == window.palette.radius_spin.value()


def test_a_radius_set_on_the_tool_reaches_the_palette(window):
    """Anything that moves the brush -- a test, a future panel -- shows up."""
    answer_roi(window)
    window.brush.set_radius(42)
    window.update_status()
    assert window.palette.radius_spin.value() == 42


# --------------------------------------------------------------------------- #
# the guide, one phase per state (ruling U2b-2)
# --------------------------------------------------------------------------- #
def test_the_guide_walks_a_frame_from_roi_to_confirm(window, monkeypatch):
    assert phase(window) == G.PHASE_ROI
    assert "Enter" in window.guide.text()
    answer_roi(window)
    assert phase(window) == G.PHASE_PICK
    _row, instance = first_row(window)
    assert instance in window.guide.plan().now
    window.on_request_edit(instance)
    assert phase(window) == G.PHASE_DRAW_EMPTY
    brush_stroke(window)
    assert phase(window) == G.PHASE_DRAW_PIXELS
    monkeypatch.setattr(window.session, "suggest_scope",
                        lambda *a, **k: "zorder:above:other")
    window.act_commit()
    assert phase(window) == G.PHASE_SCOPE
    window.act_clear_edit()
    assert phase(window) == G.PHASE_PICK


def test_the_area_warning_is_its_own_phase(window, monkeypatch):
    from tda.ui import app_priors

    answer_roi(window)
    _row, instance = first_row(window)
    window.on_request_edit(instance)
    brush_stroke(window)
    monkeypatch.setattr(app_priors, "area_warning", lambda *a, **k: "面积异常")
    window.act_commit()
    assert window.warn_bar.isVisibleTo(window)
    assert phase(window) == G.PHASE_WARNING
    window.act_clear_edit()             # Esc answers the warning first
    assert phase(window) == G.PHASE_DRAW_PIXELS


def test_a_draft_ghost_is_its_own_phase(window, monkeypatch):
    answer_roi(window)
    _row, instance = first_row(window)
    window.on_request_edit(instance)
    monkeypatch.setattr(window, "showing_draft_ghost", lambda: True)
    window.refresh_guidance()
    assert phase(window) == G.PHASE_GHOST


def test_all_items_done_asks_for_space_and_a_confirmed_frame_says_so(window):
    answer_roi(window)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(LAST_STEP, force=True)
    assert phase(window) == G.PHASE_CONFIRM
    assert window.palette.button("confirm").suggested()
    assert window.act_confirm() is True
    window.timeline_goto(LAST_STEP)
    assert phase(window) == G.PHASE_CONFIRMED


def test_review_and_steps_modes_have_their_own_line(window):
    answer_roi(window)
    window.set_mode(A.MODE_REVIEW)
    assert phase(window) == G.PHASE_REVIEW
    window.set_mode(A.MODE_STEPS)
    assert phase(window) == G.PHASE_STEPS
    assert "Apply" in window.guide.text()


def test_a_frame_with_no_image_is_not_a_missing_drive(qapp, tmp_path):
    """A step this view never photographed: no image, and nothing to plug in.

    The drive-missing phase itself is tested with U2a's real resolver in
    tests/test_app_guide_r2.py (``test_the_raw_drive_sentence_says_f5_once``).
    """
    win = open_window(tmp_path, missing=(LAST_STEP,))
    try:
        win.session.goto(LAST_STEP)
        assert phase(win) == G.PHASE_NO_IMAGE
        assert win.session.images.why_unreadable(win.session.current()) is None
    finally:
        close_window(win)


def test_moving_the_mouse_costs_the_guide_nothing(window, monkeypatch):
    """The guide follows state changes; a mouse move over the canvas is none."""
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    _row, instance = first_row(window)
    window.on_request_edit(instance)
    calls: list[int] = []
    monkeypatch.setattr(window, "refresh_guidance", lambda: calls.append(1))
    monkeypatch.setattr(window, "_note_edit_facts", lambda *a, **k: calls.append(2))
    viewport = window.canvas.viewport()
    for dx in range(0, 60, 6):
        QTest.mouseMove(viewport, viewport.rect().center() + QPoint(dx, dx // 2))
    QApplication.processEvents()
    assert calls == []


def test_the_guide_repaints_only_when_the_plan_changes(window):
    answer_roi(window)
    plan = window.guide.plan()
    assert window.guide.show_plan(plan, window._cheat_for(window.mode)) is False


def test_the_cheat_sheet_lists_only_this_modes_keys(window):
    annotate = dict(A.mode_cheat_rows(A.MODE_ANNOTATE))
    review = dict(A.mode_cheat_rows(A.MODE_REVIEW))
    assert "B" in annotate and "K" not in annotate
    assert "K" in review and "B" not in review
    assert "1–7" in annotate
    window.guide.cheat_button.setChecked(True)
    assert not window.guide.cheat.isHidden()


# --------------------------------------------------------------------------- #
# a task card that explains itself (ruling U2b-3)
# --------------------------------------------------------------------------- #
def test_the_header_says_what_the_frame_is():
    assert card_header(42, None, last=42) == ("第 42 帧（起点，已经拆完的样子）："
                                     "把这一帧里还看得到的零件都画出来，每个都画完整形状"
                                     "（被别的零件挡住的部分也算它的，层级程序会处理）")
    assert card_header(40, 41) == ("第 40 帧：比第 41 帧多了下面这些零件"
                                   "（刚被拆掉的，要把它画回去）")


def test_the_card_header_follows_the_frame(window):
    assert "起点" in window.task_card.header_text()
    assert str(LAST_STEP) in window.task_card.header_text()
    answer_roi(window)
    window.act_step(-1)
    # Round 2 (I3): the header is built from the rows it sits over.
    kinds = [row["kind"] for row in window.task_card.rows()]
    steps = window.session.steps()
    header = window.task_card.header_text()
    assert f"第 {LAST_STEP - 1} 帧" in header
    assert header == card_header(LAST_STEP - 1, LAST_STEP, kinds,
                                 first=min(steps), last=max(steps))
    if api.KIND_ADD_SHAPE not in kinds:
        assert "多了" not in header
    assert "✚" in window.task_card.header.toolTip()


def test_a_row_names_the_part_the_log_and_what_to_do(window):
    texts = window.task_card.row_texts()
    rows = window.task_card.rows()
    cover = rows.index(next(r for r in rows if r["instance"] == "cover.01"))
    assert "导风罩 cover.01" in texts[cover]
    assert "日志：CPU Fan Cover" in texts[cover]
    assert "在这一帧画出它的完整形状" in texts[cover]
    assert "[待画]" in texts[cover]


def test_the_start_frame_says_what_a_complete_shape_is_once(window):
    """Round 1 item 6, round 1b: "完整形状" meant nothing to a first-time
    annotator -- said once, in the start frame's header, not on every row."""
    note = "（被别的零件挡住的部分也算它的，层级程序会处理）"
    assert window.task_card.header_text().count(note) == 1
    assert "完整形状" + note in window.task_card.header_text()
    assert not any(note in text for text in window.task_card.row_texts())
    answer_roi(window)
    window.act_step(-1)                   # not the start frame any more
    assert note not in window.task_card.header_text()
    assert not any(note in text for text in window.task_card.row_texts())
    view = row_view({"instance": "chassis", "kind": api.KIND_ADD_SHAPE, "done": False,
                     "cls": "chassis", "parent": "motherboard.01"}, start=True)
    # nothing "comes back in with" a parent on the start frame
    assert view["sentence"] == "在这一帧画出它的完整形状"


@pytest.mark.parametrize("kind,sentence", [
    (api.KIND_SPLIT_KEYFRAME, "Ctrl+K"),
    (api.KIND_STATE_ONLY, "不用画"),
    (api.KIND_ADD_BENCH_BOX, "台面框 R"),
    (api.KIND_CONFIRM, "Space"),
])
def test_each_kind_has_its_plain_sentence(kind, sentence):
    view = row_view({"instance": "cover.01", "kind": kind, "done": False,
                     "cls": "cover", "attrs": {"of": "cpu_cooler"},
                     "transition": ["open", "closed"]})
    assert sentence in view["sentence"]


def test_a_single_click_starts_the_edit_and_marks_the_row(window):
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    row, instance = first_row(window)
    click_row(window, row)
    assert window.session.editing_instance == instance
    assert CHIP_EDITING in window.task_card.row_texts()[row]
    assert window.task_card.editing_instance() == instance
    assert QApplication.focusWidget() is window.canvas


def test_a_double_click_still_works_and_asks_once(window, monkeypatch):
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    asked: list[str] = []
    original = window.on_request_edit
    monkeypatch.setattr(window, "on_request_edit",
                        lambda i, *a, **k: asked.append(i) or original(i, *a, **k))
    window.task_card.sigRequestEdit.disconnect()
    window.task_card.sigRequestEdit.connect(window.on_request_edit)
    row, instance = first_row(window)
    click_row(window, row, double=True)
    assert asked == [instance]
    assert window.session.editing_instance == instance


def test_the_task_card_has_no_action_buttons_of_its_own(window):
    """Round 1, item 2: the palette is the one place for actions."""
    from PySide6.QtWidgets import QPushButton

    card = window.task_card
    for name in ("commit_button", "override_button", "split_button", "confirm_button"):
        assert not hasattr(card, name), name
    assert not hasattr(card, "sigCommit") and not hasattr(card, "sigConfirm")
    assert card.findChildren(QPushButton) == []
    assert not hasattr(window, "on_panel_commit")


def test_a_row_that_is_not_work_explains_itself(window):
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(LAST_STEP, force=True)
    rows = window.task_card.rows()
    assert [r["kind"] for r in rows] == [api.KIND_CONFIRM]
    click_row(window, 0)
    assert window.session.editing_instance is None
    assert "Space" in window.status_message()


# --------------------------------------------------------------------------- #
# the hover outline: where it probably is, and nothing else
# --------------------------------------------------------------------------- #
def _dwell() -> None:
    deadline = time.monotonic() + 3 * HINT_DWELL_MS / 1000.0 + 0.2
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def _state(win: MainWindow) -> tuple:
    return (win._tool_name, win._prompt_box, win.session.editing_instance,
            win.task_card.list_widget().currentRow(), win.status_message(),
            int(win.overlay.editing.sum()), win.canvas._rubber_band,
            win.canvas.prompt_point(), win.session.current())


def test_hovering_a_row_outlines_the_shape_it_had_and_changes_nothing(window):
    answer_roi(window)
    instance = "cover.01"
    window.session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=instance, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=LAST_STEP - 2, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(cell(9)))],
    ))
    before = _state(window)
    window.task_card.sigHover.emit(instance)
    _dwell()
    hints = window.canvas.hint_boxes()
    assert hints, "nothing was outlined"
    box, label, _rgb = hints[0]
    assert box == (9.0, 9.0, 15.0, 15.0) and "的形状" in label
    assert _state(window) == before
    window.task_card.sigHover.emit("")
    QApplication.processEvents()
    assert window.canvas.hint_boxes() == []
    assert _state(window) == before


def test_sliding_past_rows_looks_nothing_up(window, monkeypatch):
    answer_roi(window)
    looked: list[str] = []
    monkeypatch.setattr(window, "card_hints", lambda i: looked.append(i) or [])
    for row in window.task_card.rows()[:5]:
        window.task_card.sigHover.emit(str(row["instance"]))
        QApplication.processEvents()
    window.task_card.sigHover.emit("")
    _dwell()
    assert looked == []


def test_the_difference_map_box_is_outlined_for_a_part_that_came_back(window):
    from tda.core.diffmap import DiffBlob

    answer_roi(window)
    window.act_step(-1)
    key = window.session.current()
    blob = DiffBlob(box=(3, 4, 20, 22), area=80, score=1.0)
    window.assist_result = {"key": key, "blobs": [blob], "unexplained": [blob],
                            "explained": []}
    window._prompt_box = None
    hints = window._diff_hint(key, {"kind": api.KIND_ADD_SHAPE})
    assert hints and hints[0][0] == (3.0, 4.0, 20.0, 22.0)
    assert window._diff_hint(key, {"kind": api.KIND_STATE_ONLY}) == []
    # the box SAM is actually armed with wins -- it may be a Shift+C alternate
    window._prompt_box = (5.0, 6.0, 10.0, 12.0)
    assert window._diff_hint(key, {"kind": api.KIND_ADD_SHAPE})[0][0] == (5.0, 6.0, 10.0, 12.0)


def test_outlines_of_one_place_are_one_outline_with_both_names():
    """The difference map's box and a draft on the same screw overlapped labels."""
    from tda.ui.app_guide import _merge_hints

    diff = ((898.0, 799.0, 922.0, 822.0), "差异图的提示框", (255, 150, 40))
    same = ((896.0, 800.0, 920.0, 821.0), "旧草稿", (96, 208, 255))
    again = ((897.0, 800.0, 921.0, 821.0), "旧草稿", (96, 208, 255))
    other = ((918.0, 354.0, 937.0, 372.0), "旧草稿", (96, 208, 255))
    merged = _merge_hints([diff, same, again, other])
    assert [label for _b, label, _c in merged] == ["差异图的提示框 + 旧草稿", "旧草稿"]
    assert merged[0][0] == diff[0]


# --------------------------------------------------------------------------- #
# the banner, and the tool a new shape starts with (ruling U2b-4)
# --------------------------------------------------------------------------- #
def test_an_empty_edit_shows_the_banner_and_arms_sam(window):
    answer_roi(window)
    assert window.canvas.banner_text() == ""
    window.on_request_edit("cover.01")
    banner = window.canvas.banner_text()
    assert banner == f"正在画：导风罩 cover.01（日志：CPU Fan Cover）— {BANNER_EMPTY}"
    assert window._tool_name == "sam_point"
    assert window.palette.button("tool_sam_point").isChecked()


def test_pixels_change_the_banner_to_enter_or_esc(window):
    answer_roi(window)
    window.on_request_edit("cover.01")
    brush_stroke(window)
    assert window.canvas.banner_text().endswith(BANNER_PIXELS)
    window.act_clear_edit()
    assert window.canvas.banner_text() == ""


def test_a_part_without_log_names_still_gets_a_banner(window):
    answer_roi(window)
    window.on_request_edit("chassis")
    assert window.canvas.banner_text() == f"正在画：机箱 chassis — {BANNER_EMPTY}"


def test_the_last_sam_tool_is_the_one_a_new_shape_starts_with(window):
    answer_roi(window)
    window.act_tool("sam_box")
    window.act_tool("brush")          # tidying up the last part...
    window.act_tool("sam_box")        # ...and the box is what they prompt with
    window.on_request_edit("cover.01")
    assert window._tool_name == "sam_box"
    window.act_clear_edit()
    window.act_tool("brush")
    window.on_request_edit("cover.02")
    # Round 2: "remember the last SAM tool, default S" -- a brush used for
    # tidying up in between does not make the next shape start with S.
    assert window._tool_name == "sam_box", "the last SAM tool was X"


def test_without_sam_the_tool_stays(qapp, tmp_path):
    win = open_window(tmp_path, sam=False)
    try:
        answer_roi(win)
        win.on_request_edit("cover.01")
        assert win._tool_name == "brush"
        assert "B 画笔" in win.canvas.banner_text()
    finally:
        close_window(win)


def test_a_shape_that_already_has_pixels_keeps_the_tool(window):
    answer_roi(window)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(LAST_STEP, force=True)
    window.act_tool("eraser")
    window.on_request_edit("cover.01")
    assert window._tool_name == "eraser"
    # Loaded and untouched: Enter would be refused, so the banner does not
    # offer it first (round 2, item 2).
    assert window.canvas.banner_text().endswith(BANNER_LOADED)


def test_a_stroke_that_adopts_the_card_item_keeps_the_brush(window):
    """The press is already on its way to the brush: no switch to SAM."""
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    window.act_tool("brush")
    viewport = window.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(6, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(6, 0))
    QApplication.processEvents()
    assert window._tool_name == "brush"
    assert window.session.editing_instance is not None
    assert window.overlay.editing.any()


def test_an_armed_bench_box_is_named_on_the_canvas_the_card_and_the_guide(window):
    answer_roi(window)
    window.begin_bench_box("cover.01")
    assert window.canvas.banner_text().startswith("正在框：导风罩 cover.01")
    assert phase(window) == G.PHASE_BENCH
    assert window.palette.button("tool_bench_box").isChecked()
    rows = window.task_card.rows()
    index = next(i for i, r in enumerate(rows) if r["instance"] == "cover.01")
    assert "✎ 正在框" in window.task_card.row_texts()[index]
    window.act_clear_edit()             # Esc disarms it
    assert window.canvas.banner_text() == ""
    assert "✎" not in window.task_card.row_texts()[index]


def test_the_banner_never_takes_a_click(window):
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    window.on_request_edit("cover.01")
    rect = window.canvas.banner_rect()
    assert rect is not None and rect.height() > 0
    # painted, not a widget: nothing on the canvas can catch the press
    assert window.canvas.viewport().childAt(rect.center()) is None
    presses: list[tuple] = []
    window.canvas.sigMousePress.connect(lambda x, y, ev: presses.append((x, y)))
    QTest.mousePress(window.canvas.viewport(), Qt.MouseButton.RightButton,
                     Qt.KeyboardModifier.NoModifier, rect.center())
    QTest.mouseRelease(window.canvas.viewport(), Qt.MouseButton.RightButton,
                       Qt.KeyboardModifier.NoModifier, rect.center())
    assert len(presses) == 1


def test_setting_the_same_banner_repaints_nothing(window, monkeypatch):
    answer_roi(window)
    window.canvas.set_banner("正在画：x")
    updates: list = []
    viewport = window.canvas.viewport()
    monkeypatch.setattr(viewport, "update", lambda *a: updates.append(a))
    window.canvas.set_banner("正在画：x")
    assert updates == []


# --------------------------------------------------------------------------- #
# round 1, item 4: hover drafts over the whole pose segment, at most five
# --------------------------------------------------------------------------- #
SCREW_LABEL = "CPU Cooling Fan Screw"        # -> screw, via configs/ls_label_map.yaml


def _screw_draft(win: MainWindow, ordinal: int, step: int, box) -> None:
    """One importer-shaped Label Studio draft of class ``screw``."""
    from tda.core.model import InstanceRec

    key = f"ls:{SCREW_LABEL}#{ordinal}"
    mask = np.zeros((64, 64), dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    win.db.upsert_instance(InstanceRec(key=key, desktop=DESKTOP, cls="screw",
                                       raw_names=[SCREW_LABEL]))
    win.db.add_keyframe(ShapeKeyframe(
        id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=0,
        anchor_step=step, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(mask))],
        amodal_complete=False, source="labelstudio", draft_id=7,
    ))


def _draft_boxes(hints) -> list:
    return [box for box, label, _rgb in hints if "旧草稿" in label]


def test_hover_drafts_come_from_the_whole_pose_segment_and_stop_at_five(window):
    answer_roi(window)
    # eight screws traced at step 3: eleven steps away from the start frame,
    # far outside Shift+A's +-2, inside the one pose segment (1..14)
    boxes = [(2 + 7 * i, 50, 5 + 7 * i, 53) for i in range(8)]
    for index, box in enumerate(boxes):
        _screw_draft(window, index + 1, 3, box)
    window._prompt_box = None
    window.assist_result = None
    shown = _draft_boxes(window.card_hints("screw.motherboard.01"))
    assert len(shown) == 5
    assert all(box in [tuple(float(v) for v in b) for b in boxes] for box in shown)


def test_the_five_hover_drafts_are_the_nearest_to_the_difference_box(window):
    from tda.core.diffmap import DiffBlob

    answer_roi(window)
    boxes = [(2 + 7 * i, 50, 5 + 7 * i, 53) for i in range(8)]
    for index, box in enumerate(boxes):
        _screw_draft(window, index + 1, 3, box)
    key = window.session.current()
    blob = DiffBlob(box=(52, 44, 58, 50), area=36, score=1.0)
    window.assist_result = {"key": key, "blobs": [blob], "unexplained": [blob],
                            "explained": []}
    window._prompt_box = None
    shown = _draft_boxes(window.card_hints("screw.motherboard.01"))
    nearest = sorted(boxes, key=lambda b: abs((b[0] + b[2]) / 2 - 55))[:5]
    assert sorted(shown) == sorted(tuple(float(v) for v in b) for b in nearest)


def test_shift_a_still_offers_only_drafts_within_two_steps(window):
    """The wider window is for the hover outline only; adoption is unchanged."""
    answer_roi(window)
    _screw_draft(window, 1, 3, (30, 30, 33, 33))
    window.on_request_edit("screw.motherboard.01")
    window.act_adopt_draft(at=(31.0, 31.0))
    assert not window.showing_draft_ghost()
    assert "没有可用的旧草稿" in window.status_message()


# --------------------------------------------------------------------------- #
# round 1, item 5: the brush size can be typed
# --------------------------------------------------------------------------- #
def _focus_spin(win: MainWindow):
    spin = win.palette.radius_spin
    QTest.mouseClick(spin, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, spin.rect().center())
    QApplication.processEvents()
    assert QApplication.focusWidget() is spin, QApplication.focusWidget()
    spin.selectAll()
    return spin


def test_typing_a_brush_size_applies_it_and_gives_the_keyboard_back(window):
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    window.act_tool("brush")
    spin = _focus_spin(window)
    QTest.keyClicks(spin, "24")
    assert window.brush.radius != 24, "applied before Enter: keyboard tracking is on"
    QTest.keyClick(spin, Qt.Key.Key_Return)
    QApplication.processEvents()
    assert window.brush.radius == window.eraser.radius == 24
    assert "r=24" in window.tool_label.text()
    assert window.canvas.tool_cursor().radius == 24
    assert QApplication.focusWidget() is window.canvas
    assert "r=24" in window.status_message()
    QTest.keyClick(window.canvas, Qt.Key.Key_E)
    assert window._tool_name == "eraser"
    QTest.keyClick(window.canvas, Qt.Key.Key_B)
    assert window._tool_name == "brush"


def test_escape_in_the_brush_size_puts_it_back(window):
    window.show()
    QApplication.processEvents()
    answer_roi(window)
    window.on_request_edit("cover.01")
    window.set_brush_radius(12)
    spin = _focus_spin(window)
    QTest.keyClicks(spin, "99")
    QTest.keyClick(spin, Qt.Key.Key_Escape)
    QApplication.processEvents()
    assert window.brush.radius == 12 and spin.value() == 12
    assert QApplication.focusWidget() is window.canvas
    # Esc was the number box's, not the window's: the edit is still open
    assert window.session.editing_instance == "cover.01"


def test_while_the_brush_size_has_the_keyboard_shortcuts_are_swallowed(window):
    from tda.ui.app_keys import KEY_SWALLOWED

    window.show()
    QApplication.processEvents()
    answer_roi(window)
    window.act_tool("brush")
    spin = _focus_spin(window)
    QTest.keyClick(spin, Qt.Key.Key_E)
    assert window._tool_name == "brush"
    assert window.status_message() == KEY_SWALLOWED


# --------------------------------------------------------------------------- #
# round 1, item 1: the raw drive's own sentence (task U2a)
# --------------------------------------------------------------------------- #
def test_the_raw_drive_reason_is_the_image_caches_own(qapp, tmp_path, monkeypatch):
    win = open_window(tmp_path, missing=(LAST_STEP,))
    try:
        win.session.goto(LAST_STEP)
        monkeypatch.setattr(win.session.images, "why_unreadable",
                            lambda key: "原始数据盘没有接上 — F:/scan/42.png", raising=False)
        win.refresh_guidance()
        assert phase(win) == G.PHASE_RAW_MISSING
        assert "原始数据盘没有接上 — F:/scan/42.png" in win.guide.text()
    finally:
        close_window(win)


def test_the_raw_drive_reason_falls_back_to_the_windows_resolver(qapp, tmp_path):
    from types import SimpleNamespace

    win = open_window(tmp_path, missing=(LAST_STEP,))
    try:
        win.session.goto(LAST_STEP)
        win.raw_root = SimpleNamespace(configured=True, connected=False,
                                       message="原始数据盘 F: 不在")
        win.refresh_guidance()
        assert phase(win) == G.PHASE_RAW_MISSING
        assert "原始数据盘 F: 不在" in win.guide.text()
        win.raw_root = SimpleNamespace(configured=True, connected=True, message="")
        win.refresh_guidance()
        assert phase(win) == G.PHASE_NO_IMAGE
    finally:
        close_window(win)
