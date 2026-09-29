"""U2b round 3: what a row, a key and the first picture may claim.

The round-2 review found a ✂ row turned ✔ by a visibility label alone, a digit
key that hid the chassis without a word, a start card that hid what was drawn,
a first picture fitted to a canvas that had not been laid out yet, and keys
that did something different from the greyed button next to them.  Each has
its test here, against the real window on the synthetic D13 scene.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_roi import BENCH_NEEDS_ITEM
from tda.ui.class_names import visibility_zh
from tda.ui.panels.taskcard import card_header, row_view

LATCH = "ram_latch.01"
SPLIT_STEP = LAST_STEP - 1          # open at 14, closed at 13: a ✂ row
CHASSIS = "chassis"
STORED = (10, 10, 54, 54)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, size=(1400, 900), show=False, stored=None,
                **kwargs) -> MainWindow:
    session = make_session(tmp_path, **kwargs)
    if stored is not None:
        session.db.set_pose_segment_roi(DESKTOP, VIEW, 1, list(stored),
                                        annotator="tester", hw=(64, 64))
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


def card_row(win_or_session, instance: str) -> dict:
    rows = (win_or_session.task_card.rows() if isinstance(win_or_session, MainWindow)
            else win_or_session.task_card())
    return next(r for r in rows if r["instance"] == instance)


def press(win: MainWindow, name: str) -> None:
    """The key of action ``name``, through the window's own dispatch."""
    win.dispatch(A.action_named(name))
    QApplication.processEvents()


def draw_chassis_at_start(win: MainWindow) -> None:
    win.session.goto(LAST_STEP, force=True)
    win.session.begin_edit(CHASSIS)
    win.session.set_editing_mask(cell(9))
    win.session.commit_edit(api.SCOPE_KEYFRAME)
    win.session.clear_edit()
    win.task_card.refresh()
    win.instances.refresh()
    QApplication.processEvents()


# --------------------------------------------------------------------------- #
# item 1 (I-A): a visibility label alone never finishes a ✂ row ... mostly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("vis,done", [
    ("visible", False),
    ("occluded_partial", False),
    ("too_small", False),
    ("visible_tiny", False),
    ("motion_blur", False),
    ("out_of_view", True),        # nothing of it is here: nothing to draw
    ("occluded_full", True),
])
def test_a_visibility_label_finishes_a_split_row_only_when_nothing_shows(
        qapp, tmp_path, vis, done):
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP)
    session.goto(SPLIT_STEP)
    assert card_row(session, LATCH)["done"] is False
    session.set_visibility(LATCH, vis)
    assert card_row(session, LATCH)["done"] is done, vis
    session.close()


def test_a_drawn_frame_override_still_finishes_a_split_row(qapp, tmp_path):
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP)
    session.goto(SPLIT_STEP)
    session.set_visibility(LATCH, "occluded_partial")      # a label first ...
    session.begin_edit(LATCH)
    session.set_editing_mask(cell(41))
    session.commit_edit(api.SCOPE_FRAME_OVERRIDE)          # ... then a drawn shape
    session.clear_edit()
    assert card_row(session, LATCH)["done"] is True
    session.close()


def test_pressing_2_on_a_split_row_does_not_say_space(window):
    answer_roi(window)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(SPLIT_STEP, force=True)
    QApplication.processEvents()
    assert window.instances.select_instance(LATCH)
    press(window, "visibility_2")                        # 部分遮挡
    assert card_row(window, LATCH)["done"] is False
    plan = window.guide.plan()
    assert plan.phase != G.PHASE_CONFIRM and "Space" not in plan.now, plan


# --------------------------------------------------------------------------- #
# item 3: the start card lists drawn parts too, ✔ and last
# --------------------------------------------------------------------------- #
def test_the_start_card_keeps_a_drawn_part_as_a_done_row_at_the_end(window):
    answer_roi(window)
    draw_chassis_at_start(window)
    rows = window.task_card.rows()
    chassis = card_row(window, CHASSIS)
    assert chassis["done"] is True and chassis["kind"] == api.KIND_ADD_SHAPE
    open_rows = [r for r in rows if not r["done"]]
    assert open_rows, "the scene has more to draw at the start"
    assert rows.index(chassis) > max(rows.index(r) for r in open_rows)
    view = row_view(chassis, start=True)
    assert view["chip"] == "✔ 已完成"
    # the guide still names the first open part, not the drawn one
    plan = window.guide.plan()
    assert plan.phase == G.PHASE_PICK and "chassis" not in plan.now
    assert f"还剩 {len(open_rows)} 个" in plan.now


def test_the_start_header_counts_what_is_left_and_what_is_done():
    kinds = [api.KIND_ADD_SHAPE] * 3
    text = card_header(42, None, kinds, first=1, last=42, done=[False, False, True])
    assert "还剩 2 个" in text and "1 个画好了" in text
    done = card_header(42, None, kinds, first=1, last=42, done=[True] * 3)
    assert "都画好了" in done and "Space" in done and "都画出来" not in done
    assert "还剩" not in done


def test_a_start_frame_with_everything_drawn_lists_it_all_and_says_space(qapp, tmp_path):
    win = open_window(tmp_path)
    try:
        answer_roi(win)
        seed_shapes(win.session, LAST_STEP)
        win.session.goto(LAST_STEP, force=True)
        QApplication.processEvents()
        rows = win.task_card.rows()
        assert rows and all(r["done"] for r in rows)
        assert {r["kind"] for r in rows} == {api.KIND_ADD_SHAPE}
        assert "都画好了" in win.task_card.header_text()
        assert win.guide.plan().phase == G.PHASE_CONFIRM
    finally:
        close_window(win)


def test_a_click_on_a_done_start_row_opens_the_stored_shape(window):
    answer_roi(window)
    draw_chassis_at_start(window)
    lw = window.task_card.list_widget()
    index = next(i for i, r in enumerate(window.task_card.rows())
                 if r["instance"] == CHASSIS)
    window.task_card._on_item_clicked(lw.item(index))
    QApplication.processEvents()
    assert window.session.editing_instance == CHASSIS
    assert window.guide.plan().phase == G.PHASE_LOADED


# --------------------------------------------------------------------------- #
# item 4: the first picture is fitted once, after the window has its size
# --------------------------------------------------------------------------- #
def test_the_first_show_fits_the_stored_roi_and_later_resizes_do_not(qapp, tmp_path):
    win = open_window(tmp_path, size=(1300, 850), stored=STORED)
    try:
        win.show()
        QApplication.processEvents()
        QApplication.processEvents()
        zoom = win.canvas.zoom_factor()
        win.canvas.zoom_to(STORED)                   # what "F" would give now
        assert zoom == pytest.approx(win.canvas.zoom_factor(), rel=1e-6)
        win.canvas.set_zoom(zoom * 3.0)              # the annotator zooms in ...
        win.resize(1500, 950)                        # ... and resizes the window
        QApplication.processEvents()
        assert win.canvas.zoom_factor() == pytest.approx(zoom * 3.0, rel=1e-6)
    finally:
        close_window(win)


def test_the_first_show_fits_the_whole_frame_without_an_roi(qapp, tmp_path):
    win = open_window(tmp_path, size=(1300, 850))
    try:
        win.show()
        QApplication.processEvents()
        QApplication.processEvents()
        assert win.roi() is None        # a proposal may be up; nothing is stored
        zoom = win.canvas.zoom_factor()
        win.canvas.fit_image()
        assert zoom == pytest.approx(win.canvas.zoom_factor(), rel=1e-6)
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 5: a visibility change is never silent
# --------------------------------------------------------------------------- #
def test_a_digit_key_says_what_it_did_and_how_to_undo_it(window):
    answer_roi(window)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(LAST_STEP, force=True)
    QApplication.processEvents()
    assert window.instances.select_instance(CHASSIS)
    press(window, "visibility_4")                         # 画面外
    said = "机箱 chassis：这一帧的可见性 → 画面外（Ctrl+Z 撤销）"
    assert said in window.status_message()
    assert said in window.canvas.banner_text()
    window.expire_visibility_note()                       # the banner is short-lived
    assert said not in window.canvas.banner_text()


def test_a_digit_key_with_nothing_selected_says_so(window):
    answer_roi(window)
    window.instances.table().setCurrentCell(-1, -1)
    press(window, "visibility_4")
    assert "实例表" in window.status_message()


def test_an_overridden_visibility_is_marked_in_the_table_and_on_the_card(window):
    answer_roi(window)
    draw_chassis_at_start(window)
    assert window.instances.select_instance(CHASSIS)
    press(window, "visibility_4")
    row = next(r for r in window.instances.rows() if r["key"] == CHASSIS)
    assert row["vis_override"] is True and row["has_shape"] is True
    table = window.instances.table()
    line = next(i for i in range(table.rowCount())
                if table.item(i, 1) is not None and table.item(i, 1).text() == CHASSIS)
    vis = table.item(line, window.instances.COLUMNS.index("Vis"))
    assert "⚠" in vis.text()
    assert "已设为不可见" in vis.toolTip()
    assert vis.background().color().alpha() > 0, "the overridden cell is not marked"
    card = card_row(window, CHASSIS)
    assert card.get("invisible") == "out_of_view"
    assert "已设为不可见" in row_view(card, start=True)["sentence"]


def test_a_plain_label_is_marked_but_not_warned(window):
    answer_roi(window)
    draw_chassis_at_start(window)
    assert window.instances.select_instance(CHASSIS)
    press(window, "visibility_2")                          # 部分遮挡: still visible
    table = window.instances.table()
    line = next(i for i in range(table.rowCount())
                if table.item(i, 1) is not None and table.item(i, 1).text() == CHASSIS)
    vis = table.item(line, window.instances.COLUMNS.index("Vis"))
    assert "⚠" not in vis.text()
    assert vis.background().color().alpha() > 0
    assert "invisible" not in card_row(window, CHASSIS)


def test_ctrl_z_takes_a_visibility_back_exactly(window):
    answer_roi(window)
    draw_chassis_at_start(window)
    key = window.session.current()
    before_override = window.db.frame_overrides(key).get(CHASSIS)
    before_mask = window.session.compiled().instances[CHASSIS].visible.copy()
    assert window.instances.select_instance(CHASSIS)
    press(window, "visibility_4")
    assert window.db.frame_overrides(key)[CHASSIS].visibility == "out_of_view"
    press(window, "undo")
    assert window.db.frame_overrides(key).get(CHASSIS) == before_override is None
    after = window.session.compiled().instances[CHASSIS].visible
    assert after is not None and (after == before_mask).all()
    row = next(r for r in window.instances.rows() if r["key"] == CHASSIS)
    assert row["vis_override"] is False


def test_ctrl_z_keeps_a_drawn_frame_override_byte_exact(window):
    answer_roi(window)
    draw_chassis_at_start(window)
    session = window.session
    session.begin_edit(CHASSIS)
    session.set_editing_mask(cell(10))
    session.commit_edit(api.SCOPE_FRAME_OVERRIDE)
    session.clear_edit()
    key = session.current()
    before = window.db.frame_overrides(key)[CHASSIS]
    window.instances.refresh()
    assert window.instances.select_instance(CHASSIS)
    press(window, "visibility_3")
    press(window, "undo")
    after = window.db.frame_overrides(key)[CHASSIS]
    assert (after.visible_rle, after.visibility) == (before.visible_rle, before.visibility)


def test_the_visibility_names_in_chinese():
    assert visibility_zh("out_of_view") == "画面外"
    assert visibility_zh("occluded_full") == "完全遮挡"
    assert visibility_zh("occluded_partial") == "部分遮挡"
    assert visibility_zh("new_value") == "new_value"


# --------------------------------------------------------------------------- #
# item 6: minors
# --------------------------------------------------------------------------- #
def test_the_bench_box_button_is_greyed_with_the_r_sentence(window):
    answer_roi(window)
    window.instances.table().setCurrentCell(-1, -1)
    window.refresh_guidance()
    button = window.palette.button("tool_bench_box")
    assert not button.isEnabled()
    assert button.reason() == BENCH_NEEDS_ITEM


def test_a_loaded_split_row_suggests_no_button(window):
    answer_roi(window)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(SPLIT_STEP, force=True)
    QApplication.processEvents()
    window.task_card.sigRequestEdit.emit(LATCH)
    QApplication.processEvents()
    plan = window.guide.plan()
    assert plan.phase == G.PHASE_LOADED
    assert plan.action == "", plan
    assert not any(b.suggested() for b in window.palette.buttons().values())


def _no_image(win: MainWindow) -> None:
    answer_roi(win)                 # its worker's sentence must not land mid-test
    win.session.goto(LAST_STEP)
    QApplication.processEvents()
    win.refresh_guidance()
    assert win.guide.plan().phase == G.PHASE_NO_IMAGE


def _untouched(win: MainWindow) -> None:
    answer_roi(win)
    seed_shapes(win.session, LAST_STEP)
    win.session.goto(LAST_STEP, force=True)
    win.on_request_edit("cover.01")
    win.refresh_guidance()


@pytest.mark.parametrize("setup,missing,name", [
    (_no_image, (LAST_STEP,), "commit"),
    (_no_image, (LAST_STEP,), "commit_override"),
    (_no_image, (LAST_STEP,), "commit_split"),
    (_no_image, (LAST_STEP,), "confirm"),
    (_no_image, (LAST_STEP,), "tool_brush"),
    (_no_image, (LAST_STEP,), "toggle_heat"),
    (_untouched, (), "commit_override"),
    (_untouched, (), "commit_split"),
])
def test_a_key_refuses_with_the_greyed_buttons_reason(qapp, tmp_path, setup, missing, name):
    win = open_window(tmp_path, missing=missing)
    try:
        setup(win)
        button = win.palette.button(name)
        assert not button.isEnabled(), name
        reason = button.reason()
        assert reason
        ops_before = len(win.db.ops(DESKTOP, VIEW, limit=10_000))
        tool_before = win._tool_name
        win.report("")
        press(win, name)
        assert reason in win.status_message(), (name, win.status_message())
        assert win._tool_name == tool_before
        assert len(win.db.ops(DESKTOP, VIEW, limit=10_000)) == ops_before, "the key acted"
    finally:
        close_window(win)
