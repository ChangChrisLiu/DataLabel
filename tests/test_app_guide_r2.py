"""U2b round 2: the instructions must be true, and nothing on screen may clip.

The review of U2b found the guide telling the annotator to do the wrong thing
in three places -- ``Enter`` on a ✂ row rewrites the other version of the
shape, ``Enter`` on an untouched loaded shape is refused, and a card header
that says "draw these back" over rows that only change state -- plus a guide
that clips on a 1280x720 screen and a minimap that turned left drags into pans.
Each has its test here, against the real window on the synthetic D13 scene.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
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
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart
from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.class_names import state_zh
from tda.ui.panels.taskcard import card_header, row_view

LATCH = "ram_latch.01"
SPLIT_STEP = LAST_STEP - 1          # open at 14, closed at 13: a ✂ row


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, size=(1400, 900), show=False, **kwargs) -> MainWindow:
    session = make_session(tmp_path, **kwargs)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(*size)
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


def brush_stroke(win: MainWindow) -> None:
    win.act_tool("brush")
    viewport = win.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(8, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(8, 0))
    QApplication.processEvents()


def card_row(win: MainWindow, instance: str) -> dict:
    return next(r for r in win.task_card.rows() if r["instance"] == instance)


def keyframe_rles(win: MainWindow, instance: str) -> dict:
    """``{anchor step: (rle of every part)}`` -- what "byte-identical" compares."""
    out = {}
    for kf in win.db.keyframes(DESKTOP, VIEW, instance):
        out[int(kf.anchor_step)] = tuple(repr(p.rle) for p in kf.parts)
    return out


def on_split_row(win: MainWindow) -> None:
    """Every part drawn at 14; stand on 13, where the latch closes."""
    answer_roi(win)
    seed_shapes(win.session, LAST_STEP)
    win.session.goto(SPLIT_STEP, force=True)
    QApplication.processEvents()
    assert card_row(win, LATCH)["kind"] == api.KIND_SPLIT_KEYFRAME


# --------------------------------------------------------------------------- #
# item 1: a ✂ row -- the guide says Ctrl+K, and the row is done only once split
# --------------------------------------------------------------------------- #
def test_a_split_row_is_not_done_because_the_old_shape_reaches_it(window):
    """The neighbour's shape covering this frame is exactly *not* a split."""
    on_split_row(window)
    assert card_row(window, LATCH)["done"] is False
    plan = window.guide.plan()
    assert plan.phase == G.PHASE_PICK, plan
    assert "Space" not in plan.now


def test_a_split_row_is_done_once_a_version_starts_here(qapp, tmp_path):
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP)
    session.goto(SPLIT_STEP)
    session.begin_edit(LATCH)
    session.set_editing_mask(cell(40))
    session.commit_edit(api.SCOPE_SPLIT)
    session.clear_edit()
    row = next(r for r in session.task_card() if r["instance"] == LATCH)
    assert row["done"] is True
    session.close()


def test_a_frame_override_here_also_ends_the_old_version(qapp, tmp_path):
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP)
    session.goto(SPLIT_STEP)
    session.begin_edit(LATCH)
    session.set_editing_mask(cell(41))
    session.commit_edit(api.SCOPE_FRAME_OVERRIDE)
    session.clear_edit()
    row = next(r for r in session.task_card() if r["instance"] == LATCH)
    assert row["done"] is True
    session.close()


def test_following_the_guide_on_a_split_row_starts_a_new_version(window):
    on_split_row(window)
    before = keyframe_rles(window, LATCH)
    assert list(before) == [LAST_STEP]
    window.task_card.sigRequestEdit.emit(LATCH)
    brush_stroke(window)
    plan = window.guide.plan()
    assert plan.phase == G.PHASE_DRAW_PIXELS
    assert plan.action == "commit_split"
    assert "Ctrl+K" in plan.now and "从这帧起新版本" in plan.now
    assert "Ctrl+K" in window.canvas.banner_text()
    assert window.palette.button("commit_split").suggested()
    assert not window.palette.button("commit").suggested()
    assert window.palette.button("commit").isEnabled(), "Enter stays available"

    window.run_palette_action(plan.action)      # do exactly what the guide says
    if window._pending_warning is not None:     # the synthetic latch is "too big"
        plan = window.guide.plan()
        assert plan.phase == G.PHASE_WARNING
        assert plan.action == "commit_split" and "Ctrl+K" in plan.now, plan
        window.run_palette_action(plan.action)  # ... and the guide still says Ctrl+K

    after = keyframe_rles(window, LATCH)
    assert after[LAST_STEP] == before[LAST_STEP], "the other version was rewritten"
    assert SPLIT_STEP in after, "no new version starts here"
    assert card_row(window, LATCH)["done"] is True


def test_on_a_split_row_the_step_line_names_ctrl_k_too():
    facts = G.GuideFacts(step=13, neighbour=14, roi_stored=True, editing="x",
                         editing_kind=api.KIND_SPLIT_KEYFRAME, layer_pixels=True,
                         layer_dirty=True,
                         items=(G.CardItem("x", api.KIND_SPLIT_KEYFRAME, False, "x"),))
    plan = G.plan_for(facts)
    line = [text for state, text in plan.steps if state == G.NOW][0]
    assert line.startswith("Ctrl+K")


# --------------------------------------------------------------------------- #
# item 2: a loaded, untouched shape -- Enter would be refused
# --------------------------------------------------------------------------- #
def test_a_loaded_untouched_shape_says_draw_or_esc(window):
    answer_roi(window)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(LAST_STEP, force=True)
    window.on_request_edit("cover.01")
    assert window.layer_facts()[1:] == (True, False)       # pixels, not dirty
    plan = window.guide.plan()
    assert plan.phase == G.PHASE_LOADED
    assert "这是已存的形状" in plan.now and "Esc" in plan.now
    assert plan.action == "clear_edit"
    assert window.palette.button("clear_edit").suggested()
    assert not window.palette.button("commit").isEnabled()
    assert "Esc" in window.canvas.banner_text()
    assert "Enter 提交" not in window.canvas.banner_text()


def test_editing_a_stored_roi_says_esc_keeps_it():
    """Item 6: on D13 the ROI is already stored; opening it to check it is not
    "a proposal you may ignore for now"."""
    stored = G.plan_for(G.GuideFacts(step=42, roi_stored=True, roi_editing=True))
    assert stored.phase == G.PHASE_ROI
    assert "已存的" in stored.now and "Esc" in stored.now and "暂时不想管" not in stored.now
    line = [text for state, text in stored.steps if state == G.NOW][0]
    assert "Esc 不改" in line and "跳过" not in line, line
    proposal = G.plan_for(G.GuideFacts(step=42, roi_stored=False, roi_editing=True))
    assert "暂时不想管" in proposal.now


def test_the_guide_title_on_a_forward_start_is_not_the_taken_apart_look():
    """Item 3, the guide's own title: step 1 as a start is the machine whole."""
    back = G.plan_for(G.GuideFacts(step=42, roi_stored=True))
    assert "已经拆完的样子" in back.title
    forward = G.plan_for(G.GuideFacts(step=1, roi_stored=True, forward_start=True))
    assert "已经拆完" not in forward.title and "还没开始拆" in forward.title


# --------------------------------------------------------------------------- #
# item 3: a header that agrees with its rows; Chinese state names
# --------------------------------------------------------------------------- #
def test_the_start_header_depends_on_the_direction_and_on_what_is_left():
    last = card_header(42, None, [api.KIND_ADD_SHAPE], first=1, last=42)
    assert "已经拆完的样子" in last and "都画出来" in last
    done = card_header(42, None, [api.KIND_CONFIRM], first=1, last=42)
    assert "都画出来" not in done and "Space" in done
    forward = card_header(1, None, [api.KIND_ADD_SHAPE], first=1, last=42)
    assert "已经拆完" not in forward and "还没开始拆" in forward


@pytest.mark.parametrize("kinds,wanted,unwanted", [
    ([api.KIND_ADD_SHAPE], "多了", None),
    ([api.KIND_STATE_ONLY, api.KIND_STATE_ONLY], "只有状态变化", "多了"),
    ([api.KIND_SPLIT_KEYFRAME], "形状从这一帧起变了", "多了"),
    ([api.KIND_CONFIRM], "这一帧不用画，直接 Space", "多了"),
    ([api.KIND_ADD_SHAPE, api.KIND_SPLIT_KEYFRAME, api.KIND_STATE_ONLY],
     "1 个要画回去", None),
])
def test_the_header_is_built_from_the_row_kinds(kinds, wanted, unwanted):
    text = card_header(40, 41, kinds, first=1, last=42)
    assert wanted in text, text
    if unwanted:
        assert unwanted not in text, text
    if len(set(kinds)) > 1:
        assert "1 个形状变了" in text and "1 个只是状态变了" in text


def test_the_window_header_on_a_split_frame(window):
    on_split_row(window)
    header = window.task_card.header_text()
    assert "形状从这一帧起变了" in header and "多了" not in header


def test_split_rows_show_the_states_in_chinese():
    assert state_zh("open") == "打开" and state_zh("closed") == "关上"
    assert state_zh("fastened") == "拧紧" and state_zh("unplugged") == "拔掉"
    assert state_zh("something_new") == "something_new"
    view = row_view({"instance": LATCH, "kind": api.KIND_SPLIT_KEYFRAME, "done": False,
                     "cls": "ram_latch", "transition": ["open", "closed"]})
    assert "（打开 → 关上）" in view["sentence"]
    assert "open" not in view["sentence"]


# --------------------------------------------------------------------------- #
# item 4: nothing clips at 1280x720; the wheel scrolls the palette
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [(1280, 720), (1600, 900)])
def test_every_guide_line_is_readable_or_reachable(qapp, tmp_path, size):
    win = open_window(tmp_path, size=size, show=True)
    try:
        answer_roi(win)
        win.on_request_edit("cover.01")    # the longest plan: five lines + now
        QApplication.processEvents()
        area = win.guide.scroll
        content = area.widget()
        for label in (win.guide.title, win.guide.now, win.guide.steps):
            if label.isHidden():
                continue
            wanted = label.heightForWidth(label.width())
            assert label.height() >= wanted - 1, (label.text()[:30], label.height(), wanted)
        bar = area.verticalScrollBar()
        assert content.height() <= area.viewport().height() + bar.maximum() + 1
        assert win.guide_dock.height() > 0
    finally:
        close_window(win)


def test_every_palette_button_is_reachable_at_1280x720(qapp, tmp_path):
    win = open_window(tmp_path, size=(1280, 720), show=True)
    try:
        answer_roi(win)
        scroll = win.palette.scroll
        bar = scroll.verticalScrollBar()
        inner = scroll.widget()
        for button in win.palette.buttons().values():
            if button.isHidden():
                continue
            assert button.height() >= button.sizeHint().height() - 1
            bottom = button.mapTo(inner, button.rect().bottomLeft()).y()
            assert bottom <= scroll.viewport().height() + bar.maximum() + 1
        assert win.canvas.minimap().isVisible(), "the minimap stays in view"
    finally:
        close_window(win)


def _wheel(widget, dy: int = -240) -> None:
    centre = QPointF(widget.rect().center())
    event = QWheelEvent(centre, QPointF(widget.mapToGlobal(centre.toPoint())), QPoint(0, 0),
                        QPoint(0, dy), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                        False)
    QApplication.sendEvent(widget, event)
    QApplication.processEvents()


def test_the_wheel_over_the_brush_size_scrolls_the_palette(qapp, tmp_path):
    win = open_window(tmp_path, size=(1280, 560), show=True)
    try:
        answer_roi(win)
        bar = win.palette.scroll.verticalScrollBar()
        assert bar.maximum() > 0, "the palette has to scroll for this to mean anything"
        for widget in (win.palette.radius_slider, win.palette.radius_spin):
            bar.setValue(0)
            radius = win.brush.radius
            _wheel(widget)
            assert bar.value() > 0, type(widget).__name__
            assert win.brush.radius == radius, "the wheel changed the brush instead"
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 5: the minimap lives on the palette; the canvas has nothing on top
# --------------------------------------------------------------------------- #
def test_the_minimap_is_not_on_the_canvas(qapp, tmp_path):
    win = open_window(tmp_path, size=(1280, 720), show=True)
    try:
        answer_roi(win)
        mini = win.canvas.minimap()
        assert not win.canvas.isAncestorOf(mini)
        assert win.palette.isAncestorOf(mini)
        assert mini.isVisible()
        viewport = win.canvas.viewport()
        for x, y in ((2, 2), (viewport.width() - 3, viewport.height() - 3),
                     (viewport.width() - 40, viewport.height() - 40)):
            point = viewport.mapTo(win.canvas, QPoint(x, y))
            assert win.canvas.childAt(point) in (None, viewport), (x, y)
        win.act_edit_roi()
        QApplication.processEvents()
        assert mini.isVisible(), "nothing to hide any more: it is not over the ROI"
    finally:
        close_window(win)


def test_a_click_on_the_minimap_recentres_the_canvas(qapp, tmp_path):
    win = open_window(tmp_path, size=(1280, 720), show=True)
    try:
        answer_roi(win)
        win.canvas.set_zoom(32.0)
        win.canvas.center_on((32.0, 32.0))
        QApplication.processEvents()
        before = win.canvas.viewport_image_rect()
        mini = win.canvas.minimap()
        QTest.mouseClick(mini, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(4, 4))
        QApplication.processEvents()
        assert win.canvas.viewport_image_rect() != before
    finally:
        close_window(win)


@pytest.mark.parametrize("tool", ["brush", "eraser", "occluder", "sam_point",
                                  "sam_box", "bench_box", "roi"])
def test_a_left_drag_from_the_old_minimap_corner_never_pans(qapp, tmp_path, tool):
    """Where the minimap used to sit, over the chassis on a zoomed frame, a left
    drag re-centred the view -- "dragging inside the box only moves the view"."""
    win = open_window(tmp_path, size=(1280, 720), show=True)
    try:
        answer_roi(win)
        win.canvas.set_zoom(8.0)
        win.canvas.center_on((32.0, 32.0))
        if tool == "roi":
            win.act_edit_roi()
        else:
            win._tool_name = tool
            win._attach_tool()
        QApplication.processEvents()
        bars = (win.canvas.horizontalScrollBar(), win.canvas.verticalScrollBar())
        before = [b.value() for b in bars]
        viewport = win.canvas.viewport()
        start = QPoint(viewport.width() - 30, viewport.height() - 30)
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, start)
        for i in range(1, 5):
            QTest.mouseMove(viewport, start - QPoint(20 * i, 15 * i))
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, start - QPoint(80, 60))
        QApplication.processEvents()
        assert [b.value() for b in bars] == before, f"{tool}: a left drag panned"
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 6: a new shape starts with the last SAM tool, S by default
# --------------------------------------------------------------------------- #
def test_a_new_shape_starts_with_the_last_sam_tool(window):
    answer_roi(window)
    window.on_request_edit("cover.01")
    assert window._tool_name == "sam_point"            # never used one: S
    window.act_clear_edit()
    window.act_tool("sam_box")
    window.act_tool("brush")                           # tidying up with the brush
    window.act_tool("eraser")
    window.on_request_edit("cover.02")
    assert window._tool_name == "sam_box", "the last SAM tool was X"


# --------------------------------------------------------------------------- #
# item 7: the minors
# --------------------------------------------------------------------------- #
def test_a_stale_swallowed_key_line_goes_when_the_canvas_has_the_keys_back(qapp, tmp_path):
    from tda.ui.app_keys import KEY_SWALLOWED

    win = open_window(tmp_path, show=True)
    try:
        answer_roi(win)
        spin = win.palette.radius_spin
        QTest.mouseClick(spin, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, spin.rect().center())
        QTest.keyClick(spin, Qt.Key.Key_E)
        assert win.status_message() == KEY_SWALLOWED
        win.canvas.setFocus(Qt.FocusReason.MouseFocusReason)
        QApplication.processEvents()
        assert win.status_message() != KEY_SWALLOWED
    finally:
        close_window(win)


def test_a_click_on_a_greyed_button_says_why_in_the_status_bar(qapp, tmp_path):
    win = open_window(tmp_path, show=True)
    try:
        answer_roi(win)
        commit = win.palette.button("commit")
        assert not commit.isEnabled()
        QTest.mouseClick(commit, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, commit.rect().center())
        QApplication.processEvents()
        assert commit.reason() in win.status_message()
        assert "提交" in win.status_message()
    finally:
        close_window(win)


def test_a_bench_box_end_row_is_not_work_and_starts_no_edit(window, monkeypatch):
    view = row_view({"instance": "psu.01", "kind": api.KIND_REMOVE_BENCH_BOX,
                     "done": False, "cls": "psu"})
    assert view["chip"] != "待画"
    rows = [{"instance": "psu.01", "kind": api.KIND_REMOVE_BENCH_BOX, "text": "ends",
             "done": False, "cls": "psu"}]
    monkeypatch.setattr(window.session, "task_card", lambda: [dict(r) for r in rows])
    window.task_card.refresh()
    asked: list[str] = []
    window.task_card.sigRequestEdit.connect(asked.append)
    item = window.task_card.list_widget().item(0)
    window.task_card._on_item_clicked(item)
    assert asked == []
    assert window.session.editing_instance is None


def test_the_scope_phase_speaks_chinese(window, monkeypatch):
    answer_roi(window)
    window.on_request_edit("cover.01")
    brush_stroke(window)
    monkeypatch.setattr(window.session, "suggest_scope",
                        lambda *a, **k: "zorder:above:chassis")
    window.act_commit()
    plan = window.guide.plan()
    assert plan.phase == G.PHASE_SCOPE
    assert "它在 机箱 chassis 上面（层级）" in plan.now
    assert "zorder:" not in plan.now


def test_the_raw_drive_sentence_says_f5_once(qapp, tmp_path, monkeypatch):
    """U2a's real accessor, with the drive really missing (its test scene)."""
    import shutil

    from test_rawroot_wiring import _raw_window

    win, g = _raw_window(tmp_path, monkeypatch, plugged=False)
    shutil.rmtree(g, ignore_errors=True)
    try:
        win.act_recheck_raw_data()
        win.refresh_guidance()
        plan = win.guide.plan()
        assert plan.phase == G.PHASE_RAW_MISSING, plan
        assert "原始数据盘没连上" in plan.now
        assert plan.now.count("F5") == 1, plan.now
    finally:
        close_window(win)


def test_the_hover_outline_is_looked_up_again_after_a_commit(window):
    answer_roi(window)
    looked: list[str] = []
    original = window._shape_hint
    window._shape_hint = lambda key, inst: looked.append(inst) or original(key, inst)
    window.card_hints("cover.01")
    window.card_hints("cover.01")
    assert looked == ["cover.01"], "cached within the frame"
    window.on_request_edit("cover.01")
    brush_stroke(window)
    window.act_commit()
    for _ in range(2):
        if window._pending_warning is not None or window._pending_scope is not None:
            window.act_commit()
    window.card_hints("cover.01")
    assert looked == ["cover.01", "cover.01"], "the committed shape was not looked up"
    window.act_undo()
    window.card_hints("cover.01")
    assert looked == ["cover.01"] * 3
