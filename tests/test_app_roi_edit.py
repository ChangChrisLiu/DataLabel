"""Editing a stored ROI, and what a left drag may never do (task U2a, trial #2).

"ROI 这个区域我没法更改，在框里移动只是修改视角" -- the annotator's second
trial. D13/scan already had a stored ROI, drawn only as a thin outline that no
gesture could reach; the way to edit it was ``Shift+R``, which nobody had told
them, and the ``R`` they pressed arms the *bench* box. A drag "inside the box"
then moved the view: on a frame zoomed to its ROI the minimap sat inside the
rectangle on screen, and a left drag on the minimap re-centres the canvas.
(U2b round 2 moved the minimap off the canvas, onto the tool palette.)

Every mouse gesture here is a real ``QTest`` event on the canvas viewport, so
what is tested is what a hand does, not what a slot does when called.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import (
    DESKTOP,
    LAST_STEP,
    SEGMENT_CUT,
    VIEW,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
)
from tda.ui import app_actions as A
from tda.ui.app import MainWindow

STORED = (10, 10, 54, 54)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, stored=None, **kwargs) -> MainWindow:
    session = make_session(tmp_path, **kwargs)
    if stored is not None:
        for seg, box in stored.items():
            session.db.set_pose_segment_roi(DESKTOP, VIEW, seg, list(box),
                                            annotator="tester", hw=(64, 64))
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    return win


def to_view(win: MainWindow, x: float, y: float) -> QPoint:
    """An image point, in canvas-viewport pixels."""
    return win.canvas.mapFromScene(QPointF(x, y))


def drag(win: MainWindow, start: QPoint, end: QPoint,
         button=Qt.MouseButton.LeftButton, steps: int = 4, widget=None) -> None:
    """A press-move-release on ``widget`` (the canvas viewport by default)."""
    target = widget if widget is not None else win.canvas.viewport()
    QTest.mousePress(target, button, Qt.KeyboardModifier.NoModifier, start)
    for i in range(1, steps + 1):
        QTest.mouseMove(target, start + (end - start) * i / steps)
    QTest.mouseRelease(target, button, Qt.KeyboardModifier.NoModifier, end)
    QApplication.processEvents()


def widget_at(win: MainWindow, point_in_canvas: QPoint):
    """What a hand pressing there would press: the topmost child of the canvas."""
    found = win.canvas.childAt(point_in_canvas)
    return found if found is not None else win.canvas.viewport()


def zoom_into(win: MainWindow, box) -> None:
    """Zoom so the rectangle covers the whole viewport, as on a real frame."""
    x0, y0, x1, y1 = box
    win.canvas.zoom_to(box)
    win.canvas.set_zoom(win.canvas.zoom_factor() * 2.0)
    win.canvas.center_on(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    QApplication.processEvents()


# --------------------------------------------------------------------------- #
# a stored ROI is editable directly
# --------------------------------------------------------------------------- #
def test_edit_roi_opens_the_stored_rectangle_with_its_handles(qapp, tmp_path):
    win = open_window(tmp_path, stored={1: STORED})
    try:
        assert win.roi_editing is False and win.roi() == STORED
        win.act_edit_roi()
        assert win.roi_editing is True
        assert win.roi_tool.rect == tuple(float(v) for v in STORED)
        assert win.active_tool is win.roi_tool
        assert win.canvas.roi_rect() == tuple(float(v) for v in STORED)
        assert set(win.canvas.roi_handle_points()) == {"nw", "n", "ne", "w", "e",
                                                       "sw", "s", "se"}
        assert "Esc" in win.roi_bar.label.text()
    finally:
        close_window(win)


def test_esc_keeps_the_stored_rectangle_exactly(qapp, tmp_path):
    win = open_window(tmp_path, stored={1: STORED})
    try:
        win.act_edit_roi()
        centre = to_view(win, 32, 32)
        drag(win, centre, centre + QPoint(20, 20))       # moved on screen ...
        assert tuple(win.roi_draft) != STORED
        win.act_clear_edit()                             # ... and Esc
        assert win.roi_editing is False
        assert win.roi() == STORED
        assert win.canvas.roi_rect() == tuple(float(v) for v in STORED)
    finally:
        close_window(win)


def test_a_left_drag_inside_moves_the_rectangle_and_never_the_view(qapp, tmp_path):
    win = open_window(tmp_path, stored={1: STORED})
    try:
        win.act_edit_roi()
        zoom_before = win.canvas.zoom_factor()
        view_before = win.canvas.viewport_image_rect()
        start = to_view(win, 30, 30)
        end = to_view(win, 34, 33)
        drag(win, start, end)
        moved = tuple(win.roi_draft)
        assert (moved[2] - moved[0], moved[3] - moved[1]) == (44, 44), "a move keeps the size"
        assert (moved[0], moved[1]) == (14, 13)
        assert win.canvas.viewport_image_rect() == view_before, "the view panned"
        assert win.canvas.zoom_factor() == zoom_before
        win.act_commit()                                  # Enter stores it
        assert win.roi_editing is False
        assert win.roi() == moved
    finally:
        close_window(win)


def spots(win: MainWindow, inset: int = 12) -> list[QPoint]:
    """Nine places across the viewport: its corners, edge middles and centre."""
    r = win.canvas.viewport().rect()
    xs = (r.left() + inset, r.center().x(), r.right() - inset)
    ys = (r.top() + inset, r.center().y(), r.bottom() - inset)
    return [QPoint(x, y) for y in ys for x in xs]


@pytest.mark.parametrize("tool", ["brush", "eraser", "occluder", "sam_point",
                                  "sam_box", "bench_box", "roi"])
def test_a_left_drag_anywhere_on_the_canvas_never_pans(qapp, tmp_path, tool):
    """U2b round 2: nothing floats over the canvas any more (the minimap that
    turned "a drag inside the ROI" into a pan now lives on the tool palette),
    so a left drag that starts *anywhere* on the picture -- the bottom-right
    corner it used to cover included -- is the armed tool's, never a pan."""
    win = open_window(tmp_path, stored={1: STORED})
    try:
        if tool == "roi":
            win.act_edit_roi()             # the bar appears and the canvas shrinks ...
        else:
            win._tool_name = tool
            win._attach_tool()
        QApplication.processEvents()
        zoom_into(win, STORED)             # ... so measure where things are now

        def origin():
            # The picture point under the viewport's top-left corner, and the
            # zoom: a pan moves the first. (Not the whole visible rectangle --
            # the ROI bar's sentence may wrap after a drag and shorten the
            # viewport, which is not a pan.)
            corner = win.canvas.mapToScene(QPoint(0, 0))
            return (round(corner.x(), 3), round(corner.y(), 3), win.canvas.zoom_factor())

        for spot in spots(win):
            where = win.canvas.viewport().mapTo(win.canvas, spot)
            assert widget_at(win, where) is win.canvas.viewport(), (
                f"{tool}: a widget sits over the canvas at {spot}")
            before = origin()
            drag(win, spot, spot - QPoint(30, 25))
            assert origin() == before, f"{tool}: a left drag from {spot} panned"
    finally:
        close_window(win)


@pytest.mark.parametrize("tool", ["brush", "eraser", "occluder", "sam_point",
                                  "sam_box", "bench_box", "roi"])
def test_a_left_drag_never_pans_whatever_the_tool(qapp, tmp_path, tool):
    win = open_window(tmp_path, stored={1: STORED})
    try:
        zoom_into(win, STORED)
        if tool == "roi":
            win.act_edit_roi()
        else:
            win._tool_name = tool          # armed as the tool, whatever act_tool allows
            win._attach_tool()
        QApplication.processEvents()
        before = win.canvas.viewport_image_rect()
        centre = win.canvas.viewport().rect().center()
        drag(win, centre - QPoint(60, 0), centre + QPoint(60, 40))
        assert win.canvas.viewport_image_rect() == before, f"{tool}: a left drag panned"
        # ... and the middle button is still the way to pan
        drag(win, centre, centre + QPoint(80, 60), button=Qt.MouseButton.MiddleButton)
        assert win.canvas.viewport_image_rect() != before
    finally:
        close_window(win)


def test_a_stroke_that_repaints_the_frame_leaves_the_view_exactly_where_it_was(
        qapp, tmp_path):
    """An occluder stroke is committed at once and the frame is drawn again;
    restoring the view through its centre crept by a pixel per stroke -- on
    the real D13 scan, (716, 483, 1296, 986) became (716, 483, 1295, 985)."""
    win = open_window(tmp_path, stored={1: STORED})
    try:
        win.canvas.set_zoom(13.37)
        win.canvas.center_on((29.3, 35.7))
        QApplication.processEvents()
        win._tool_name = "occluder"
        win._attach_tool()
        bars = (win.canvas.horizontalScrollBar(), win.canvas.verticalScrollBar())
        before = ([b.value() for b in bars], win.canvas.zoom_factor())
        centre = win.canvas.viewport().rect().center()
        for _ in range(3):
            drag(win, centre - QPoint(30, 0), centre + QPoint(30, 20))
        assert ([b.value() for b in bars], win.canvas.zoom_factor()) == before
    finally:
        close_window(win)


def test_the_status_bar_roi_label_opens_the_editor(qapp, tmp_path):
    win = open_window(tmp_path, stored={1: STORED})
    try:
        assert win.roi_label.text() == "ROI ✓"
        assert "Shift+R" in win.roi_label.toolTip()
        QTest.mouseClick(win.roi_label, Qt.MouseButton.LeftButton)
        QApplication.processEvents()
        assert win.roi_editing is True
        assert win.roi_tool.rect == tuple(float(v) for v in STORED)
    finally:
        close_window(win)


def test_edit_roi_from_review_mode_goes_to_annotate_first(qapp, tmp_path):
    win = open_window(tmp_path, stored={1: STORED})
    try:
        win.set_mode(A.MODE_REVIEW)
        win.act_edit_roi()
        assert win.mode == A.MODE_ANNOTATE
        assert win.roi_editing is True
        assert win.active_tool is win.roi_tool
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# bare R
# --------------------------------------------------------------------------- #
def test_bare_r_explains_itself_and_arms_nothing(qapp, tmp_path):
    win = open_window(tmp_path, stored={1: STORED})
    try:
        win.act_tool("brush")
        win.act_tool("bench_box")                        # R, nothing on the bench armed
        assert win.armed_tool_name() == "brush", "R armed a tool that eats drags"
        text = win.status_message()
        assert "R 是台面框" in text and "Shift+R" in text and "ROI" in text
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# leaving a segment is not an answer, and is not about the next segment
# --------------------------------------------------------------------------- #
def _key(win: MainWindow):
    return win.roi_key()


def test_leaving_for_a_stored_segment_dismisses_the_rectangle_of_the_old_one(
        qapp, tmp_path):
    """The trial's log: leaving D12 wrote "dismissed segment=(13, ...)"."""
    from tda.ui import app_support as S

    win = open_window(tmp_path, stored={1: STORED}, two_segments=True)
    try:
        assert win.session.current().step == LAST_STEP     # segment 2, no ROI
        assert win.roi_editing is True
        old = _key(win)
        win.timeline_goto(SEGMENT_CUT - 1)                  # segment 1, stored
        QApplication.processEvents()
        new = _key(win)
        assert new != old
        assert win.roi_editing is False
        assert old in win._roi_dismissed, "dismissed under the segment it belonged to"
        assert new not in win._roi_dismissed, "not under the segment arrived at"
        assert win.roi_unanswered() is False               # segment 1 has its ROI
        assert old in win._roi_pending, "leaving is not an answer"
        log = Path(S.log_path(win.paths)).read_text(encoding="utf-8")
        lines = [line for line in log.splitlines() if "roi rectangle dismissed" in line]
        assert lines and str(old) in lines[-1] and str(new) not in lines[-1]
        assert "unanswered=True" in lines[-1]

        win.timeline_goto(LAST_STEP)                        # and back
        QApplication.processEvents()
        assert win.roi_editing is False, "a dismissed rectangle does not pop up again"
        assert win.roi_unanswered() is True, "... but the question is still open"
    finally:
        close_window(win)


def test_leaving_for_an_unanswered_segment_asks_about_that_one(qapp, tmp_path):
    win = open_window(tmp_path, two_segments=True)
    try:
        win.wait_for_roi_proposal()
        old = _key(win)
        assert win.roi_editing is True
        win.timeline_goto(SEGMENT_CUT - 1)
        QApplication.processEvents()
        new = _key(win)
        assert old in win._roi_dismissed and old in win._roi_pending
        assert win.roi_editing is True, "segment 1 gets its own question"
        assert win._roi_wanted == new
        assert new not in win._roi_dismissed
    finally:
        close_window(win)
