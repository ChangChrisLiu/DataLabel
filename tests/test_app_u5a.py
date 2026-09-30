"""Task U5a: the ``P`` polygon and the ``Y`` circle -- a whole region in one gesture.

The annotator (2026-09-30): "After SAM segments the motherboard there are many
gaps in the middle; today I can only paint them one by one with the brush.
Could there be a tool where I click a few points and the region between them
is all selected, overlapping what is already labelled?"  And, for the screws,
a circle.

What is pinned here:

* the two rasterisers -- the polygon is :func:`tda.core.masks.polygons_to_mask`
  cropped to its own box, the disk is the pixels whose centres the preview
  circle holds;
* the polygon: vertices, ``Backspace``, closing by ``Enter`` / double-click /
  a click on the first vertex, fewer than three, the union with the layer,
  the erased set, one undo step, the ``Esc`` layering, a tool or frame switch
  dropping it unfilled, zoom and pan in the middle of it;
* the circle: drag, click, under a pixel, ``Esc`` mid-drag, zoom mid-drag;
* both refused and greyed exactly like the brush, named in the guide and the
  cheat sheet;
* and nothing else moved: a fill followed by an ``S`` click sends SAM the very
  request a brush dab followed by an ``S`` click does.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import (
    HW,
    LAST_STEP,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
)
from tda.core import masks as M
from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui.app import MainWindow
from tda.ui.app_edit import NO_INSTANCE_HINT, SHAPE_BUSY
from tda.ui.app_guide import BANNER_CIRCLE
from tda.ui.canvas.tools import CircleTool, PolygonTool, disk_patch, polygon_patch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, **kwargs) -> MainWindow:
    session = make_session(tmp_path, **kwargs)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    return win


def skip_roi(win: MainWindow) -> None:
    """Answer the ROI question with Esc: the rectangle owns the canvas until then."""
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()


@pytest.fixture
def window(qapp, tmp_path):
    win = open_window(tmp_path)
    skip_roi(win)
    yield win
    close_window(win)


def start_edit(win: MainWindow) -> str:
    """Begin an edit of the first part the card asks for; the layer is empty."""
    card = [row for row in win.session.task_card() if row.get("instance")]
    assert card, "the start frame should ask for shapes"
    instance = str(card[0]["instance"])
    win.on_request_edit(instance)
    assert win.session.editing_instance == instance
    return instance


def seed_block(win: MainWindow, box=(8, 8, 24, 24)) -> np.ndarray:
    """Put a solid block into the layer as one undoable change; returns the layer."""
    mask = np.zeros(win.overlay.hw, dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    win.set_editing_mask(mask, undoable=True)
    return win.overlay.editing.copy()


def click(win: MainWindow, x: float, y: float) -> None:
    """A left click at image ``(x, y)`` through the canvas signals (window first)."""
    win.canvas.sigMousePress.emit(float(x), float(y), None)
    win.canvas.sigMouseRelease.emit(float(x), float(y), None)
    QApplication.processEvents()


def key(win: MainWindow, spec: str) -> None:
    """A key press through the window's own key handling."""
    combination = QKeySequence.fromString(spec)[0]
    event = QKeyEvent(QKeyEvent.Type.KeyPress, int(combination.key()),
                      combination.keyboardModifiers())
    assert win.handle_key(event), spec
    QApplication.processEvents()


def at(win: MainWindow, x: float, y: float) -> QPoint:
    """Where image ``(x, y)`` is in the canvas viewport right now."""
    return win.canvas.mapFromScene(QPointF(float(x), float(y)))


def fill_of(points) -> np.ndarray:
    box, patch = polygon_patch(points, HW)
    out = np.zeros(HW, dtype=bool)
    out[box[1]:box[3], box[0]:box[2]] = patch
    return out


def disk_of(cx, cy, r) -> np.ndarray:
    box, patch = disk_patch(cx, cy, r, HW)
    out = np.zeros(HW, dtype=bool)
    out[box[1]:box[3], box[0]:box[2]] = patch
    return out


def ops(win: MainWindow) -> int:
    return len(win.session.undo_stack)


SQUARE = [(30.5, 30.5), (50.5, 30.5), (50.5, 50.5), (30.5, 50.5)]


# --------------------------------------------------------------------------- #
# the rasterisers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("points", [
    [(10.2, 10.7), (40.9, 12.5), (33.3, 44.4), (12.5, 30.5)],
    [(-9.0, 5.5), (30.5, -7.25), (70.0, 40.5), (20.5, 80.0)],      # over the edges
    [(2.5, 2.5), (3.5, 2.5), (2.5, 3.5)],                           # halves round to even
])
def test_the_polygon_is_polygons_to_mask_on_the_whole_frame_cropped(points):
    """Same pixels as ``masks.polygons_to_mask`` over the full frame, only cheaper."""
    grid = np.round(np.asarray(points, dtype=np.float64) - 0.5)
    whole = M.polygons_to_mask([grid.reshape(-1)], HW)
    box, patch = polygon_patch(points, HW)
    assert patch.shape == (box[3] - box[1], box[2] - box[0])
    assert np.array_equal(fill_of(points), whole)
    assert whole[box[1]:box[3], box[0]:box[2]].sum() == whole.sum(), "clipped a pixel"


def test_a_vertex_reaches_the_pixel_it_was_clicked_in():
    diamond = [(19.5, 20.5), (20.5, 19.5), (21.5, 20.5), (20.5, 21.5)]
    plus = fill_of(diamond)
    assert plus.sum() == 5 and plus[20, 20] and plus[20, 19] and plus[19, 20]


def test_the_disk_is_the_pixels_whose_centres_are_inside_it():
    disk = disk_of(20.5, 20.5, 1.0)
    assert sorted(zip(*np.nonzero(disk))) == [(19, 20), (20, 19), (20, 20), (20, 21),
                                              (21, 20)]
    big = disk_of(31.3, 29.8, 9.6)
    ys, xs = np.mgrid[0:HW[0], 0:HW[1]]
    inside = (xs + 0.5 - 31.3) ** 2 + (ys + 0.5 - 29.8) ** 2 <= 9.6 ** 2
    assert np.array_equal(big, inside)
    assert disk_patch(-50.0, -50.0, 3.0, HW) is None


# --------------------------------------------------------------------------- #
# arming P and Y
# --------------------------------------------------------------------------- #
def test_p_and_y_arm_their_tools_with_their_own_badge_cursor_and_button(window):
    start_edit(window)
    key(window, "P")
    assert window._tool_name == "polygon" and window.active_tool is window.polygon
    assert window.tool_label.text().startswith("工具：多边形 P")
    assert window.canvas.tool_cursor().kind == "polygon"
    assert window.palette.button("tool_polygon").isChecked()

    key(window, "Y")
    assert window._tool_name == "circle" and window.active_tool is window.circle
    radius = window.brush.radius
    assert window.tool_label.text().startswith(f"工具：圆形 Y r={radius}")
    cursor = window.canvas.tool_cursor()
    assert (cursor.kind, cursor.glyph, cursor.radius) == ("circle", "cross", radius)
    brush_ring = type(cursor)("circle", cursor.rgb, radius)
    assert cursor != brush_ring, "the circle tool must not look like the brush"
    assert window.palette.button("tool_circle").isChecked()
    assert not window.palette.button("tool_polygon").isChecked()


def test_the_new_keys_are_in_the_one_key_map_and_the_palette():
    polygon = A.action_named("tool_polygon")
    circle = A.action_named("tool_circle")
    assert (polygon.keys, polygon.short) == (("P",), "多边形")
    assert (circle.keys, circle.short) == (("Y",), "圆形")
    assert A.action_named("polygon_backspace").keys == ("Backspace",)
    assert {"tool_polygon", "tool_circle"} <= set(A.PALETTE_TOOLS)
    for spec in ("P", "Y"):       # C, O and R stay what they were
        others = [a.name for a in A.ACTIONS if spec in a.keys]
        assert len(others) == 1, others


# --------------------------------------------------------------------------- #
# the polygon
# --------------------------------------------------------------------------- #
def test_clicks_add_vertices_and_backspace_takes_them_back(window):
    start_edit(window)
    window.act_tool("polygon")
    for x, y in SQUARE[:3]:
        click(window, x, y)
    assert window.polygon.vertices == SQUARE[:3]
    shape = window.canvas.shape_preview()
    assert shape[0] == "polygon" and list(shape[1]) == SQUARE[:3]
    assert not window.overlay.editing.any() and ops(window) == 0, "nothing is filled yet"
    assert window.guide.plan().phase == G.PHASE_SHAPE
    assert "3 个点" in window.guide.plan().now
    assert "多边形 3 个点" in window.canvas.banner_text()

    key(window, "Backspace")
    assert window.polygon.vertices == SQUARE[:2]
    assert "还剩 2 个" in window.status_message()
    key(window, "Backspace")
    key(window, "Backspace")
    assert window.polygon.vertices == [] and window.canvas.shape_preview() is None
    key(window, "Backspace")                       # nothing left: said, not swallowed
    assert "没有正在画的多边形" in window.status_message()


def test_enter_fills_the_polygon_as_one_undo_step_over_what_is_there(window):
    start_edit(window)
    before = seed_block(window, (20, 20, 40, 40))  # overlaps the square
    count = ops(window)
    window.act_tool("polygon")
    for x, y in SQUARE:
        click(window, x, y)
    key(window, "Return")

    expected = before | fill_of(SQUARE)
    assert np.array_equal(window.overlay.editing, expected), "not the union"
    assert np.array_equal(window.session.editing_mask(), expected)
    assert (before & fill_of(SQUARE)).any(), "the test needs an overlap"
    assert ops(window) == count + 1
    assert window.session.undo_stack.ops[-1].kind == "edit_editing_mask"
    assert window.polygon.vertices == [] and window.canvas.shape_preview() is None
    assert window.has_uncommitted_edit()
    assert window._sidecar_pending is not None, "the crash sidecar was not queued"
    assert "已填进编辑层" in window.status_message()
    assert window.session.editing_instance, "Enter committed instead of filling"

    window.act_undo()
    QApplication.processEvents()
    assert np.array_equal(window.overlay.editing, before), "one undo, one polygon"


def test_a_click_on_the_first_vertex_closes_the_polygon(window):
    start_edit(window)
    window.act_tool("polygon")
    viewport = window.canvas.viewport()
    for x, y in SQUARE[:3]:
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, at(window, x, y))
    assert len(window.polygon.vertices) == 3
    near = at(window, *SQUARE[0]) + QPoint(3, -2)  # a few screen pixels off it
    window.polygon.on_move(*window.canvas.image_pos(QPointF(near)), None)
    assert window.canvas.shape_preview()[2] is True, "the first vertex is not lit"
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, near)
    QApplication.processEvents()
    assert window.polygon.vertices == []
    # The real clicks land within half a screen pixel of the pixel centres
    # asked for, which round to the same pixels.
    assert np.array_equal(window.overlay.editing, fill_of(SQUARE[:3]))
    assert ops(window) == 1


def test_near_the_first_vertex_a_click_is_a_vertex_until_there_are_three(window):
    """Round 2: the first vertex only closes a polygon that could close.

    At the fit zoom of a 12 MP frame its 8 screen px are ~44 image px, so a
    small part's second and third corners land "near the first vertex"; they
    were swallowed, and the "too few" line asked for clicks that would be
    swallowed again.  Zoomed out here so the whole part is within the 8 px.
    """
    start_edit(window)
    window.act_tool("polygon")
    window.canvas.set_zoom(0.5)                     # 8 screen px = 16 image px
    window.canvas.center_on((32.0, 32.0))
    part = [(30.5, 30.5), (36.5, 30.5), (33.5, 36.5)]
    for x, y in part[:2]:
        click(window, x, y)
        window.polygon.on_move(*part[0], None)
        assert window.canvas.shape_preview()[2] is False, "lit below three vertices"
    assert window.polygon.vertices == part[:2], "a corner near the first was swallowed"
    click(window, *part[2])
    assert window.polygon.vertices == part and ops(window) == 0
    window.polygon.on_move(*part[0], None)
    assert window.canvas.shape_preview()[2] is True
    click(window, part[0][0] + 2.0, part[0][1] + 1.0)   # now it closes
    assert window.polygon.vertices == [] and ops(window) == 1
    assert np.array_equal(window.overlay.editing, fill_of(part))


def test_a_double_click_closes_the_polygon(window):
    """Through the window, the way a real mouse arrives: Qt turns the second
    press of a double-click into ``MouseButtonDblClick`` for a widget, so the
    canvas sees press, release, double-click, release -- one vertex, then the
    close.  (``QTest.mouseDClick`` on the *widget* sends the double-click
    alone, which no real mouse does.)"""
    start_edit(window)
    window.act_tool("polygon")
    handle = window.windowHandle()
    viewport = window.canvas.viewport()

    def in_window(x, y):
        return viewport.mapTo(window, at(window, x, y))

    for x, y in SQUARE[:2]:
        QTest.mouseClick(handle, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, in_window(x, y))
    assert len(window.polygon.vertices) == 2
    QTest.mouseDClick(handle, Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier, in_window(*SQUARE[2]))
    QApplication.processEvents()
    assert window.polygon.vertices == []
    assert np.array_equal(window.overlay.editing, fill_of(SQUARE[:3]))
    assert ops(window) == 1


def test_fewer_than_three_vertices_fill_nothing_and_say_so(window):
    start_edit(window)
    window.act_tool("polygon")
    click(window, *SQUARE[0])
    click(window, *SQUARE[1])
    key(window, "Return")
    assert not window.overlay.editing.any() and ops(window) == 0
    said = window.status_message()
    assert "至少要 3 个点" in said and said.endswith("或 Esc 取消"), said
    assert window.polygon.vertices == SQUARE[:2], "the two clicks were thrown away"
    window.polygon.on_double_click(*SQUARE[1], None)
    assert not window.overlay.editing.any() and ops(window) == 0
    click(window, *SQUARE[2])                     # a third makes it one
    key(window, "Return")
    assert window.overlay.editing.any() and ops(window) == 1


def test_the_polygon_lifts_the_erased_pixels_it_covers(window):
    start_edit(window)
    seed_block(window, (30, 30, 52, 52))
    hole = np.zeros(HW, dtype=bool)
    hole[38:44, 38:44] = True
    window.set_editing_mask(window.overlay.editing & ~hole, undoable=True,
                            deliberate=True)       # what an eraser stroke leaves
    assert window.erased_mask() is not None and window.erased_mask().at(40, 40)

    window.act_tool("polygon")
    for x, y in [(36.5, 36.5), (46.5, 36.5), (46.5, 46.5), (36.5, 46.5)]:
        click(window, x, y)
    key(window, "Return")

    assert window.overlay.editing[40, 40], "the hole was not filled"
    left = window.erased_mask()
    assert left is None or not (left.full() & window.overlay.editing).any()


def test_esc_drops_the_polygon_and_only_the_polygon(window):
    instance = start_edit(window)
    layer = seed_block(window)
    window.act_tool("polygon")
    click(window, *SQUARE[0])
    click(window, *SQUARE[1])

    key(window, "Esc")
    assert window.polygon.vertices == [] and window.canvas.shape_preview() is None
    assert window.session.editing_instance == instance, "Esc discarded the edit"
    assert np.array_equal(window.overlay.editing, layer), "Esc touched the layer"
    assert "多边形已取消" in window.status_message()

    key(window, "Esc")                              # the next layer down
    assert window.session.editing_instance is None


def test_commit_keys_wait_for_a_half_drawn_polygon(window):
    start_edit(window)
    seed_block(window)
    window.act_tool("polygon")
    click(window, *SQUARE[0])
    count = ops(window)
    for name in ("commit_override", "commit_split", "confirm"):
        window.report("")
        window.dispatch(A.action_named(name))
        assert SHAPE_BUSY in window.status_message(), name
        assert window.session.editing_instance, name
    assert ops(window) == count and window.polygon.vertices == [SQUARE[0]]
    window.refresh_guidance()
    for name in ("commit_override", "commit_split", "confirm"):
        assert not window.palette.button(name).isEnabled(), name
    assert window.palette.button("commit").isEnabled(), "Enter closes it"


def test_switching_tool_drops_the_polygon_unfilled(window):
    start_edit(window)
    window.act_tool("polygon")
    for x, y in SQUARE[:3]:
        click(window, x, y)
    window.act_tool("brush")
    assert window.polygon.vertices == [] and window.canvas.shape_preview() is None
    assert not window.overlay.editing.any() and ops(window) == 0
    window.act_tool("polygon")
    assert window.polygon.vertices == [], "the polygon came back"


@pytest.mark.parametrize("mode", [A.MODE_REVIEW, A.MODE_STEPS])
def test_leaving_annotate_mode_drops_the_polygon_unfilled(window, mode):
    start_edit(window)
    window.act_tool("polygon")
    for x, y in SQUARE[:3]:
        click(window, x, y)
    window.set_mode(mode)
    assert window.mode == mode
    assert window.polygon.vertices == [] and window.canvas.shape_preview() is None
    window.set_mode(A.MODE_ANNOTATE)
    assert window.polygon.vertices == [] and not window.overlay.editing.any()


def test_a_frame_change_drops_the_polygon_unfilled(window):
    start_edit(window)
    window.act_tool("polygon")
    for x, y in SQUARE[:3]:
        click(window, x, y)
    step = window.session.current().step
    window.act_step(-1)
    QApplication.processEvents()
    assert window.session.current().step != step
    assert window.polygon.vertices == [] and window.canvas.shape_preview() is None
    assert not window.overlay.editing.any()


def test_an_undo_or_a_look_at_the_other_frame_keeps_the_polygon(window):
    start_edit(window)
    seed_block(window)
    window.act_tool("polygon")
    for x, y in SQUARE[:3]:
        click(window, x, y)
    window.act_undo()                               # the block goes, not the clicks
    QApplication.processEvents()
    assert window.polygon.vertices == SQUARE[:3]
    if window.guide_facts().neighbour is not None:
        window.act_flash_compare(True)
        QApplication.processEvents()
        window.act_flash_compare(False)
        QApplication.processEvents()
        assert window.polygon.vertices == SQUARE[:3], "Tab threw the polygon away"
    key(window, "Return")
    assert np.array_equal(window.overlay.editing, fill_of(SQUARE[:3]))


def test_another_part_drops_the_polygon(window):
    start_edit(window)
    window.act_tool("polygon")
    click(window, *SQUARE[0])
    others = [str(r["instance"]) for r in window.session.task_card()
              if r.get("instance") and r["instance"] != window.session.editing_instance]
    if not others:
        pytest.skip("the card has one part only")
    window.on_request_edit(others[0])
    assert window.session.editing_instance == others[0]
    assert window.polygon.vertices == []


def _wheel(win: MainWindow, notches: int) -> None:
    viewport = win.canvas.viewport()
    centre = QPointF(viewport.rect().center())
    event = QWheelEvent(centre, QPointF(viewport.mapToGlobal(centre.toPoint())),
                        QPoint(0, 0), QPoint(0, 120 * notches),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(viewport, event)
    QApplication.processEvents()


def test_zoom_and_pan_mid_polygon_keep_its_image_coordinates(window):
    start_edit(window)
    window.act_tool("polygon")
    viewport = window.canvas.viewport()
    window.canvas.set_zoom(6.0)
    for x, y in SQUARE[:2]:
        window.canvas.center_on((x, y))
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, at(window, x, y))
    placed = list(window.polygon.vertices)
    zoom = window.canvas.zoom_factor()
    for (px, py), (x, y) in zip(placed, SQUARE[:2]):
        assert abs(px - x) <= 1.0 / zoom and abs(py - y) <= 1.0 / zoom

    _wheel(window, 2)                               # the wheel still zooms
    assert window.canvas.zoom_factor() > zoom
    centre = viewport.rect().center()               # and the middle button pans
    view = window.canvas.viewport_image_rect()
    QTest.mousePress(viewport, Qt.MouseButton.MiddleButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(-40, -30))
    QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(-40, -30))
    QApplication.processEvents()
    assert window.canvas.viewport_image_rect() != view, "the pan did nothing"
    assert window.polygon.vertices == placed, "zoom or pan moved or added a vertex"
    assert list(window.canvas.shape_preview()[1]) == placed

    zoom = window.canvas.zoom_factor()
    for x, y in SQUARE[2:]:
        window.canvas.center_on((x, y))
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, at(window, x, y))
    assert len(window.polygon.vertices) == 4
    for (px, py), (x, y) in zip(window.polygon.vertices[2:], SQUARE[2:]):
        assert abs(px - x) <= 1.0 / zoom and abs(py - y) <= 1.0 / zoom
    key(window, "Return")
    assert np.array_equal(window.overlay.editing, fill_of(SQUARE))


# --------------------------------------------------------------------------- #
# the circle
# --------------------------------------------------------------------------- #
def drag(win: MainWindow, start, end, steps: int = 4) -> None:
    viewport = win.canvas.viewport()
    a, b = at(win, *start), at(win, *end)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, a)
    for i in range(1, steps + 1):
        QTest.mouseMove(viewport, a + (b - a) * i / steps)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, b)
    QApplication.processEvents()


def test_a_circle_drag_fills_the_disk_as_one_stroke_over_what_is_there(window):
    start_edit(window)
    before = seed_block(window, (8, 8, 30, 30))
    count = ops(window)
    window.act_tool("circle")
    viewport = window.canvas.viewport()
    centre = (24.5, 24.5)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, at(window, *centre))
    QTest.mouseMove(viewport, at(window, 32.5, 24.5))
    assert window.canvas.shape_preview()[0] == "circle"
    assert window.canvas.tool_cursor().kind == "cross", "a ring on the rim"
    assert "r=8" in window.tool_label.text(), window.tool_label.text()
    assert BANNER_CIRCLE in window.canvas.banner_text()
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, at(window, 32.5, 24.5))
    QApplication.processEvents()

    cx, cy = window.canvas.image_pos(QPointF(at(window, *centre)))
    rx, ry = window.canvas.image_pos(QPointF(at(window, 32.5, 24.5)))
    disk = disk_of(cx, cy, float(np.hypot(rx - cx, ry - cy)))
    assert np.array_equal(window.overlay.editing, before | disk)
    assert (before & disk).any() and (disk & ~before).any()
    assert ops(window) == count + 1
    assert window.canvas.shape_preview() is None
    assert window.canvas.tool_cursor().glyph == "cross", "the ring came back"
    assert "圆形 r=8 已填进编辑层" in window.status_message()
    window.act_undo()
    QApplication.processEvents()
    assert np.array_equal(window.overlay.editing, before)


def test_a_plain_click_fills_a_disk_of_the_brush_radius(window):
    start_edit(window)
    window.act_tool("circle")
    window.set_brush_radius(5)
    assert window.circle.click_radius == 5
    click(window, 30.5, 30.5)
    assert np.array_equal(window.overlay.editing, disk_of(30.5, 30.5, 5.0))
    assert ops(window) == 1


def test_a_release_under_a_pixel_is_a_click_of_the_brush_radius(window):
    """Round 2: click or drag is decided in image pixels, whatever the zoom --
    here 0.5 px is six screen pixels, and still a click."""
    start_edit(window)
    window.act_tool("circle")
    window.set_brush_radius(4)
    window.canvas.set_zoom(12.0)
    window.canvas.center_on((30, 30))
    window.circle.on_press(30.5, 30.5, None)
    window.circle.on_move(31.0, 30.5, None)
    shape = window.canvas.shape_preview()
    assert shape[2] == 4.0, "the preview must show the disk a release fills"
    assert "r=4" in window.tool_label.text(), window.tool_label.text()
    window.circle.on_release(31.0, 30.5, None)
    QApplication.processEvents()
    assert np.array_equal(window.overlay.editing, disk_of(30.5, 30.5, 4.0))
    assert ops(window) == 1


def test_at_a_quarter_zoom_a_short_drag_fills_the_radius_it_showed(window):
    """Round 2 (m1): at 25 % an 8 px drag is two screen pixels.  It was taken
    for a click and filled the brush radius, while the preview and the badge
    said r=8."""
    start_edit(window)
    window.act_tool("circle")
    window.set_brush_radius(3)
    window.canvas.set_zoom(0.25)
    window.canvas.center_on((32.0, 32.0))
    assert window.canvas.zoom_factor() == pytest.approx(0.25)
    window.circle.on_press(30.5, 30.5, None)
    window.circle.on_move(38.5, 30.5, None)
    assert window.canvas.shape_preview()[2] == pytest.approx(8.0)
    assert "r=8" in window.tool_label.text(), window.tool_label.text()
    window.circle.on_release(38.5, 30.5, None)
    QApplication.processEvents()
    assert np.array_equal(window.overlay.editing, disk_of(30.5, 30.5, 8.0))
    assert "圆形 r=8 已填进编辑层" in window.status_message()


def test_a_release_off_the_canvas_cancels_the_circle(window):
    """Round 2 (m2): a drag that ran off the canvas filled to the release
    point -- 1.1 Mpx on the scanner.  It is cancelled, and said."""
    start_edit(window)
    layer = seed_block(window)
    count = ops(window)
    window.act_tool("circle")
    window.canvas.set_zoom(12.0)
    window.canvas.center_on((30.0, 30.0))
    viewport = window.canvas.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, at(window, 30.5, 30.5))
    off = QPoint(viewport.width() + 40, viewport.height() // 2)
    QTest.mouseMove(viewport, off)
    assert window.circle.busy
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, off)
    QApplication.processEvents()
    assert not window.circle.busy and window.canvas.shape_preview() is None
    assert np.array_equal(window.overlay.editing, layer) and ops(window) == count
    assert "画布外" in window.status_message() and "已取消" in window.status_message()
    assert window.canvas.tool_cursor().glyph == "cross", "the ring came back"


def test_esc_mid_drag_drops_the_circle(window):
    start_edit(window)
    layer = seed_block(window)
    count = ops(window)
    window.act_tool("circle")
    viewport = window.canvas.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, at(window, 40.5, 40.5))
    QTest.mouseMove(viewport, at(window, 48.5, 40.5))
    key(window, "Return")                           # the release fills, not Enter
    assert window.circle.busy and ops(window) == count
    key(window, "Esc")
    assert not window.circle.busy and window.canvas.shape_preview() is None
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, at(window, 48.5, 40.5))
    QApplication.processEvents()
    assert np.array_equal(window.overlay.editing, layer) and ops(window) == count
    assert window.session.editing_instance, "Esc went past the circle"


def test_zoom_mid_drag_keeps_the_radius_in_image_pixels(window):
    start_edit(window)
    window.act_tool("circle")
    viewport = window.canvas.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, at(window, 32.5, 32.5))
    centre = window.circle.centre
    QTest.mouseMove(viewport, at(window, 38.5, 32.5))
    radius = window.circle.drag_radius
    _wheel(window, 2)
    assert window.circle.centre == centre and window.circle.drag_radius == radius
    rim = at(window, 38.5, 32.5)                    # the same image point, zoomed
    QTest.mouseMove(viewport, rim)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, rim)
    QApplication.processEvents()
    rx, ry = window.canvas.image_pos(QPointF(rim))
    expected = disk_of(centre[0], centre[1], float(np.hypot(rx - centre[0], ry - centre[1])))
    assert np.array_equal(window.overlay.editing, expected)
    assert abs(window.circle.last_radius - 6.0) < 0.5


# --------------------------------------------------------------------------- #
# the brush's rules: when it cannot run, and when nothing is being edited
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", ["annotate", "review", "no_image"])
def test_greyed_and_refused_exactly_like_the_brush(qapp, tmp_path, state):
    win = open_window(tmp_path, missing=(LAST_STEP,) if state == "no_image" else ())
    try:
        skip_roi(win)
        if state == "review":
            win.set_mode(A.MODE_REVIEW)
        if state == "no_image":
            win.session.goto(LAST_STEP)
            QApplication.processEvents()
        win.refresh_guidance()
        states = win.palette_states(win.guide_facts())
        for name in ("tool_polygon", "tool_circle"):
            assert states[name][0] == states["tool_brush"][0], (name, states[name])
            assert states[name][2] == states["tool_brush"][2], (name, states[name])
            assert win.key_refusal(name) == win.key_refusal("tool_brush")
            button, brush = win.palette.button(name), win.palette.button("tool_brush")
            assert button.isEnabled() == brush.isEnabled()
            assert button.reason() == brush.reason()
        if state == "no_image":
            tool = win._tool_name
            win.dispatch(A.action_named("tool_polygon"))
            assert win._tool_name == tool and "这一帧没有图像" in win.status_message()
    finally:
        close_window(win)


def test_with_nothing_to_edit_a_click_starts_no_polygon(window, monkeypatch):
    window.act_tool("polygon")
    monkeypatch.setattr(window, "_adoptable_instance", lambda: None)
    click(window, *SQUARE[0])
    assert window.polygon.vertices == []
    assert window.status_message() == NO_INSTANCE_HINT
    window.act_tool("circle")
    click(window, *SQUARE[0])
    assert not window.circle.busy and not window.overlay.editing.any()


def test_like_the_brush_the_first_click_adopts_the_cards_part(window):
    assert window.session.editing_instance is None
    window.act_tool("polygon")
    click(window, *SQUARE[0])
    assert window.session.editing_instance, "the card's open item was not adopted"
    assert window.polygon.vertices == [SQUARE[0]]


# --------------------------------------------------------------------------- #
# the guide, the cheat sheet, the documents
# --------------------------------------------------------------------------- #
def test_the_guide_and_the_cheat_sheet_name_p_and_y(window):
    assert "P 连点填一块" in G.DRAW and "圆形：拖一下画一颗螺丝" in G.DRAW
    start_edit(window)
    plan = window.guide.plan()
    assert any("P 连点填一块" in text for _state, text in plan.steps)
    html = A.cheat_sheet_html()
    for words in ("多边形", "圆形", "<b>P</b>", "<b>Y</b>", "<b>Backspace</b>"):
        assert words in html, words
    rows = dict(A.mode_cheat_rows(A.MODE_ANNOTATE))
    assert rows["P"].startswith("多边形") and rows["Y"].startswith("圆形")
    assert "Backspace" in rows


def test_the_documents_say_it():
    guide = (REPO / "docs" / "annotation_guide.md").read_text(encoding="utf-8")
    quick = (REPO / "docs" / "trial_quickstart.md").read_text(encoding="utf-8")
    flat = " ".join(guide.split())
    assert "`P`" in flat and "`Y`" in flat
    assert "`Shift+F`" in flat and "封闭" in flat
    assert "SAM 漏掉的大块用 P 点几下补上，封闭的小洞 Shift+F" in " ".join(quick.split())
    assert "img/u5a_polygon.jpg" in quick
    assert (REPO / "docs" / "img" / "u5a_polygon.jpg").is_file()


# --------------------------------------------------------------------------- #
# nothing else moved: the SAM request after a fill is the brush's
# --------------------------------------------------------------------------- #
def _diamond(px: int, py: int) -> list[tuple[float, float]]:
    """A polygon round pixel ``(px, py)`` whose fill is exactly an r=1 brush dab."""
    return [(px - 0.5, py + 0.5), (px + 0.5, py - 0.5),
            (px + 1.5, py + 0.5), (px + 0.5, py + 1.5)]


def _plus(px: int = 20, py: int = 20) -> np.ndarray:
    out = np.zeros(HW, dtype=bool)
    out[py, px - 1:px + 2] = True
    out[py - 1:py + 2, px] = True
    return out


def _s_click(win: MainWindow, at_xy=(30.0, 24.0)):
    win.act_tool("sam_point")
    before = len(win.sam_queue.requests)
    click(win, *at_xy)
    assert len(win.sam_queue.requests) == before + 1, "S sent nothing"
    return win.sam_queue.requests[-1]


def _same_request(a, b) -> None:
    assert a.points == b.points
    assert a.box == b.box
    assert a.multimask == b.multimask
    assert np.array_equal(a.image_crop, b.image_crop)
    assert (a.mask_input is None) == (b.mask_input is None)
    if a.mask_input is not None:
        assert np.array_equal(a.mask_input, b.mask_input)


def _edit_a_boxed_part(win: MainWindow) -> str:
    """Stand on a frame whose card part gets a difference-map box, and edit it."""
    from tda.ui import session_api as api

    for step in range(LAST_STEP - 1, 1, -1):
        win.session.goto(step)
        assert win.assist.wait(5.0)
        QApplication.processEvents()
        for row in win.session.task_card():
            if row.get("kind") != api.KIND_ADD_SHAPE or row.get("done"):
                continue
            win.on_request_edit(str(row["instance"]))
            if win._prompt_box is not None:
                return str(row["instance"])
            win.act_clear_edit()
    pytest.skip("no frame of the scene arms a prompt box")


def test_a_fill_then_s_sends_sam_what_a_brush_dab_then_s_does(window):
    """No ROI: the S click goes out point-only, after each of the three."""
    assert window.assist.wait(5.0)
    QApplication.processEvents()
    instance = start_edit(window)
    _fill_then_s(window, instance, (20, 20), (30.0, 24.0))


def test_with_a_box_armed_the_click_carries_it_after_any_of_them(qapp, tmp_path):
    """ROI stored: the difference map arms a box and the S click inside it
    carries it -- the same box, the same crop, the same prior mask after the
    brush, the polygon and the circle."""
    window = open_window(tmp_path)
    try:
        if window.roi_editing:
            window.wait_for_roi_proposal()
            window.act_commit()
        skip_roi(window)
        assert window.roi() is not None, "the scene's ROI was not stored"
        instance = _edit_a_boxed_part(window)
        x0, y0, x1, y1 = window._prompt_box
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        px = min(max(cx - 3, 1), HW[1] - 2)
        py = min(max(cy, 1), HW[0] - 2)
        request = _fill_then_s(window, instance, (px, py), (cx + 0.5, cy + 0.5))
        assert request.box is not None, "the click did not carry the armed box"
    finally:
        close_window(window)


def _fill_then_s(window: MainWindow, instance: str, pixel, click_xy):
    """Brush dab / polygon / circle click on ``pixel``, then S at ``click_xy``.

    Close enough that the diamond's corners (1.4 image px apart) are not within
    the first vertex's 8 screen px, and the dab and the S click are both inside
    the viewport crop SAM is sent.
    """
    px, py = pixel
    window.canvas.set_zoom(8.0)
    window.canvas.center_on(((px + click_xy[0]) / 2.0, (py + click_xy[1]) / 2.0))
    QApplication.processEvents()
    window.set_brush_radius(1)
    assert window.session.editing_instance == instance

    window.act_tool("brush")                        # the reference: one dab
    count = ops(window)
    click(window, float(px), float(py))
    assert np.array_equal(window.overlay.editing, _plus(px, py))
    assert ops(window) == count + 1
    by_brush = _s_click(window, click_xy)
    assert by_brush.mask_input is not None, "the dab should be what SAM refines"

    window.act_clear_edit()                         # Esc, and the same part again
    window.on_request_edit(instance)
    window.act_tool("polygon")
    for x, y in _diamond(px, py):
        click(window, x, y)
    count = ops(window)
    key(window, "Return")
    assert np.array_equal(window.overlay.editing, _plus(px, py))
    assert ops(window) == count + 1
    assert window.session.undo_stack.ops[-1].kind == "edit_editing_mask"
    _same_request(_s_click(window, click_xy), by_brush)

    window.act_clear_edit()
    window.on_request_edit(instance)
    window.act_tool("circle")                       # a click: the brush's radius
    count = ops(window)
    click(window, px + 0.5, py + 0.5)
    assert np.array_equal(window.overlay.editing, _plus(px, py))
    assert ops(window) == count + 1
    _same_request(_s_click(window, click_xy), by_brush)
    return by_brush


def test_the_fills_are_strokes_to_the_window(window):
    """Wired like the brush: the same slot, the same op, no second mechanism."""
    assert isinstance(window.polygon, PolygonTool)
    assert isinstance(window.circle, CircleTool)
    for tool in (window.polygon, window.circle):
        assert tool in window._all_tools()
        assert window._tool_for("polygon" if tool is window.polygon else "circle") is tool
