"""Offscreen tests for the canvas, label overlay, edit tools and undo stack.

Everything here runs on the ``offscreen`` Qt platform plugin (see
``conftest.py``); the env var is also set at import time because the
``QApplication`` is created by a session fixture that may be built before the
function-scoped autouse fixture runs.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from tda.core import masks as M
from tda.models.sam_service import SamRequest, SamResult
from tda.ui.canvas.overlay import (
    EDIT_RGB,
    OCCLUDER_RGB,
    PALETTE_64,
    LabelOverlay,
    palette_color,
)
from tda.ui.canvas.tools import (
    BrushTool,
    EraserTool,
    OccluderTool,
    SamBoxTool,
    SamPointTool,
    Tool,
)
from tda.ui.canvas.view import ImageCanvas
from tda.ui.commands import KINDS, Op, UndoStack, edit_editing_mask_op

MAIN_THREAD = threading.get_ident()


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def two_masks() -> tuple[dict[str, np.ndarray], list[str]]:
    a = np.zeros((40, 50), dtype=bool)
    a[5:15, 5:20] = True
    b = np.zeros((40, 50), dtype=bool)
    b[10:25, 15:35] = True  # overlaps a, painted on top
    return {"inst-a": a, "inst-b": b}, ["inst-a", "inst-b"]


def _rgb(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.arange(w, dtype=np.uint8)[None, :]
    img[..., 1] = np.arange(h, dtype=np.uint8)[:, None]
    img[..., 2] = 90
    return img


def _shown(canvas: ImageCanvas, w: int = 400, h: int = 400) -> ImageCanvas:
    canvas.resize(w, h)
    canvas.show()
    QApplication.processEvents()
    return canvas


def _press(x: float, y: float, button=Qt.MouseButton.LeftButton) -> QMouseEvent:
    pos = QPointF(x, y)
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        pos,
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )


def _argb(img, x: int, y: int) -> tuple[int, int, int, int]:
    """(a, r, g, b) of one pixel of a QImage."""
    px = img.pixel(x, y)
    return ((px >> 24) & 0xFF, (px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF)


def _pixels(img) -> np.ndarray:
    """The whole QImage as an HxW uint32 array (a copy)."""
    raw = np.frombuffer(bytes(img.constBits()), dtype=np.uint32)
    return raw.reshape(img.height(), img.width()).copy()


def _grow(rect, hw) -> tuple[int, int, int, int]:
    """The 1-px outline halo an outlined partial repaint must cover."""
    h, w = hw
    x0, y0, x1, y1 = rect
    return (max(0, x0 - 1), max(0, y0 - 1), min(w, x1 + 1), min(h, y1 + 1))


def _spin(predicate, timeout: float = 5.0) -> bool:
    """Run the Qt event loop until ``predicate()`` or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class StubQueue:
    """Stands in for ``SamQueue``: records requests, replies synchronously."""

    def __init__(self, maker=None) -> None:
        self.requests: list[SamRequest] = []
        self._maker = maker

    def submit(self, req: SamRequest, cb) -> None:
        self.requests.append(req)
        if self._maker is not None:
            cb(self._maker(req))

    @property
    def last(self) -> SamRequest:
        return self.requests[-1]


class ThreadedQueue(StubQueue):
    """Replies from a *worker* thread, like the real ``SamQueue`` does."""

    def __init__(self, maker) -> None:
        super().__init__(None)
        self._maker = maker
        self.threads: list[threading.Thread] = []

    def submit(self, req: SamRequest, cb) -> None:
        self.requests.append(req)
        maker = self._maker
        thread = threading.Thread(target=lambda: cb(maker(req)), daemon=True)
        self.threads.append(thread)
        thread.start()


def _blob_result(req: SamRequest) -> SamResult:
    h, w = req.image_crop.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4 : h // 2, w // 4 : w // 2] = True
    return SamResult(mask=mask, score=0.9, ms=1.0)


def _multi_result(req: SamRequest) -> SamResult:
    """Three nested candidates (part -> sub-assembly -> assembly), best first."""
    h, w = req.image_crop.shape[:2]
    candidates = []
    for divisor in (4, 3, 2):
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : h // 4 + h // divisor, w // 4 : w // 4 + w // divisor] = True
        candidates.append(mask)
    return SamResult(
        mask=candidates[0],
        score=0.9,
        ms=1.0,
        candidates=candidates,
        scores=[0.9, 0.8, 0.7],
    )


# ---------------------------------------------------------------------------
# LabelOverlay
# ---------------------------------------------------------------------------
def test_palette_has_64_distinct_colors():
    assert len(PALETTE_64) == 64
    assert len(set(PALETTE_64)) == 64
    assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in PALETTE_64)


def test_palette_color_is_stable_and_in_table():
    first = palette_color("D13/s010/psu#1")
    assert first == palette_color("D13/s010/psu#1")
    assert first in PALETTE_64
    assert palette_color("a") != palette_color("a-different-key")


def test_set_instances_builds_labelmap_and_palette(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)

    assert ov.labelmap.dtype == np.uint16
    assert ov.labelmap.shape == (40, 50)
    assert set(np.unique(ov.labelmap).tolist()) == {0, 1, 2}
    assert ov.labelmap[6, 6] == 1  # only in a
    assert ov.labelmap[20, 30] == 2  # only in b
    assert ov.labelmap[12, 17] == 2  # overlap: b is on top
    assert ov.id2key == {1: "inst-a", 2: "inst-b"}
    assert ov.palette[1] == palette_color("inst-a")
    assert ov.palette[2] == palette_color("inst-b")


def test_qimage_is_non_null_and_colors_match_palette(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)

    img = ov.qimage()
    assert not img.isNull()
    assert (img.width(), img.height()) == (50, 40)
    assert _argb(img, 0, 0) == (0, 0, 0, 0)  # background is transparent

    a_alpha, *a_rgb = _argb(img, 6, 6)
    assert tuple(a_rgb) == palette_color("inst-a")
    assert a_alpha == 110  # default fill alpha
    b_alpha, *b_rgb = _argb(img, 30, 20)
    assert tuple(b_rgb) == palette_color("inst-b")


def test_qimage_outline_is_opaque_and_toggleable(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)

    on = ov.qimage(outline=True)
    assert _argb(on, 5, 5)[0] == 255  # top-left corner of a: on the boundary
    assert _argb(on, 10, 10)[0] == 110  # interior of a

    off = ov.qimage(outline=False)
    assert _argb(off, 5, 5)[0] == 110


def test_qimage_alpha_argument_changes_fill(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    assert _argb(ov.qimage(alpha=200), 10, 10)[0] == 200


def test_paint_returns_dirty_rect_containing_the_stroke():
    ov = LabelOverlay((40, 50))
    ov.set_editing("inst-x", np.zeros((40, 50), dtype=bool))

    rect = ov.paint((25, 20), 5, True)
    assert rect == (20, 15, 31, 26)
    assert ov.editing[20, 25]
    assert not ov.editing[20, 31]  # just outside the disk
    ys, xs = np.nonzero(ov.editing)
    assert xs.min() >= rect[0] and xs.max() < rect[2]
    assert ys.min() >= rect[1] and ys.max() < rect[3]
    # a disk, not a square
    assert 0.6 < ov.editing.sum() / (11 * 11) < 0.9


def test_paint_clips_the_dirty_rect_to_the_image():
    ov = LabelOverlay((40, 50))
    rect = ov.paint((1, 1), 6, True)
    assert rect == (0, 0, 8, 8)
    assert ov.editing[0, 0]


def test_paint_with_add_false_erases():
    ov = LabelOverlay((40, 50))
    ov.paint((25, 20), 6, True)
    ov.paint((25, 20), 3, False)
    assert not ov.editing[20, 25]
    assert ov.editing[20, 31]


def test_paint_entirely_outside_the_image_returns_none():
    ov = LabelOverlay((40, 50))
    assert ov.paint((-40, -40), 3, True) is None
    assert ov.paint((200, 20), 3, True) is None
    assert not ov.editing.any()
    # nothing was marked dirty, so a later qimage() must not rebuild anything
    ov.qimage()
    ov.last_rebuild_rect = None
    assert ov.paint((-40, -40), 3, True) is None
    ov.qimage()
    assert ov.last_rebuild_rect is None


def test_paint_occluder_keeps_one_layer_per_type():
    ov = LabelOverlay((40, 50))
    assert ov.occluders == {}

    hand = ov.paint_occluder((10, 10), 3, True, "hand")
    cable = ov.paint_occluder((40, 30), 3, True, "cable")
    assert hand is not None and cable is not None
    assert set(ov.occluders) == {"hand", "cable"}
    assert ov.occluders["hand"][10, 10]
    assert not ov.occluders["hand"][30, 40], "a cable stroke leaked into hand"
    assert ov.occluders["cable"][30, 40]
    assert not ov.occluders["cable"][10, 10], "a hand stroke leaked into cable"
    assert not ov.editing.any()
    # every layer is rendered
    img = ov.qimage()
    assert tuple(_argb(img, 10, 10)[1:]) == OCCLUDER_RGB
    assert tuple(_argb(img, 40, 30)[1:]) == OCCLUDER_RGB


def test_paint_occluder_rejects_an_unknown_type():
    ov = LabelOverlay((40, 50))
    with pytest.raises(ValueError):
        ov.paint_occluder((10, 10), 3, True, "banana")


def test_set_occluder_replaces_one_type_only():
    ov = LabelOverlay((40, 50))
    ov.paint_occluder((10, 10), 3, True, "hand")
    mask = np.zeros((40, 50), dtype=bool)
    mask[30:35, 30:35] = True
    ov.set_occluder(mask, "cable")
    assert ov.occluders["hand"][10, 10]
    assert ov.occluders["cable"][32, 32]
    assert not ov.occluders["cable"][10, 10]


def test_editing_layer_draws_on_top_of_instances(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    edit = np.zeros((40, 50), dtype=bool)
    edit[8:12, 8:12] = True
    ov.set_editing("inst-c", edit)

    img = ov.qimage()
    assert tuple(_argb(img, 10, 10)[1:]) == EDIT_RGB
    assert tuple(_argb(img, 6, 6)[1:]) == palette_color("inst-a")


def test_qimage_rebuilds_only_the_dirty_rect(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    ov.qimage()
    assert ov.last_rebuild_rect == (0, 0, 50, 40)

    rect = ov.paint((40, 30), 3, True)
    img = ov.qimage(rect)
    # grown by 1 px so the outline of the newly exposed edge is correct
    assert ov.last_rebuild_rect == _grow(rect, (40, 50))
    assert tuple(_argb(img, 40, 30)[1:]) == EDIT_RGB
    # the untouched part of the buffer still carries the instance colours
    assert tuple(_argb(img, 6, 6)[1:]) == palette_color("inst-a")


def test_partial_repaint_matches_a_full_repaint_after_an_erase(two_masks):
    """An eraser must not leave the old outline ring outside the dirty rect."""
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    filled = np.zeros((40, 50), dtype=bool)
    filled[10:30, 10:40] = True
    ov.set_editing("inst-c", filled)
    ov.qimage()

    rect = ov.paint((25, 20), 5, False)  # erase a disc out of the middle
    partial = _pixels(ov.qimage(rect))

    ov.force_full_rebuild()
    full = _pixels(ov.qimage())
    assert int(np.count_nonzero(partial != full)) == 0


def test_partial_repaint_matches_a_full_repaint_after_a_brush(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    ov.qimage()

    rect = ov.paint((25, 20), 4, True)
    partial = _pixels(ov.qimage(rect))
    ov.force_full_rebuild()
    assert int(np.count_nonzero(partial != _pixels(ov.qimage()))) == 0


def test_visible_false_renders_transparent_and_restores(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    shown = _pixels(ov.qimage())
    assert shown.any()

    ov.visible = False
    hidden = _pixels(ov.qimage())
    assert not hidden.any(), "visible=False did not invalidate the cache"

    ov.visible = True
    assert int(np.count_nonzero(_pixels(ov.qimage()) != shown)) == 0


def test_qimage_never_returns_a_stale_buffer():
    ov = LabelOverlay((40, 50))
    ov.qimage()
    ov.paint((10, 10), 2, True)
    img = ov.qimage()  # no rect given, but a paint is pending
    assert tuple(_argb(img, 10, 10)[1:]) == EDIT_RGB


def test_set_editing_rejects_a_wrong_shape():
    ov = LabelOverlay((40, 50))
    with pytest.raises(ValueError):
        ov.set_editing("x", np.zeros((10, 10), dtype=bool))


def test_overlay_uses_no_python_pixel_loops():
    """A 1 MP overlay must render fast enough to be numpy-only."""
    ov = LabelOverlay((1000, 1000))
    m = np.zeros((1000, 1000), dtype=bool)
    m[100:900, 100:900] = True
    ov.set_instances({"k": m}, ["k"])
    t0 = time.perf_counter()
    ov.qimage()
    assert (time.perf_counter() - t0) < 0.5


# ---------------------------------------------------------------------------
# ImageCanvas
# ---------------------------------------------------------------------------
def test_set_image_sets_scene_rect_and_keeps_the_array(qapp):
    canvas = _shown(ImageCanvas())
    rgb = _rgb(300, 400)
    canvas.set_image(rgb)
    assert canvas.image_hw() == (300, 400)
    assert canvas.sceneRect().width() == 400
    assert canvas.sceneRect().height() == 300
    np.testing.assert_array_equal(canvas.image_rgb(), rgb)


def test_zoom_to_box_zooms_in_and_shows_the_box(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 400))
    canvas.zoom_to((100, 100, 200, 200))
    QApplication.processEvents()

    assert canvas.zoom_factor() > 1.0
    x0, y0, x1, y1 = canvas.viewport_image_rect()
    assert x0 <= 100 and y0 <= 100 and x1 >= 200 and y1 >= 200
    # 5% margin per side, square box in a square viewport
    assert (x1 - x0) < 150 and (y1 - y0) < 150


def test_viewport_image_rect_is_clipped_to_the_image(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(200, 300))
    canvas.set_zoom(0.2)  # whole image far smaller than the viewport
    QApplication.processEvents()
    assert canvas.viewport_image_rect() == (0, 0, 300, 200)


def test_wheel_zooms_about_the_cursor(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 400))
    canvas.set_zoom(2.0)
    QApplication.processEvents()

    pos = QPointF(120.0, 90.0)
    before = canvas.image_pos(pos)
    event = QWheelEvent(
        pos,
        canvas.viewport().mapToGlobal(pos),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(canvas.viewport(), event)
    QApplication.processEvents()

    assert canvas.zoom_factor() == pytest.approx(2.5, rel=1e-3)  # 2.0 * 1.25
    after = canvas.image_pos(pos)
    assert after[0] == pytest.approx(before[0], abs=1.0)
    assert after[1] == pytest.approx(before[1], abs=1.0)


def test_zoom_is_clamped(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(100, 100))
    canvas.set_zoom(10_000.0)
    assert canvas.zoom_factor() <= canvas.MAX_ZOOM
    canvas.set_zoom(1e-6)
    assert canvas.zoom_factor() >= canvas.MIN_ZOOM


def test_mouse_signals_carry_image_coordinates(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 400))
    canvas.set_zoom(1.0)
    canvas.center_on((200, 200))
    QApplication.processEvents()

    seen: list[tuple[float, float]] = []
    canvas.sigMousePress.connect(lambda x, y, ev: seen.append((x, y)))
    centre = QPointF(canvas.viewport().width() / 2, canvas.viewport().height() / 2)
    QApplication.sendEvent(canvas.viewport(), _press(centre.x(), centre.y()))

    assert len(seen) == 1
    assert seen[0][0] == pytest.approx(200.0, abs=1.5)
    assert seen[0][1] == pytest.approx(200.0, abs=1.5)


def test_middle_drag_pans_without_emitting_tool_signals(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(600, 600))
    canvas.set_zoom(4.0)
    QApplication.processEvents()

    presses: list[tuple[float, float]] = []
    canvas.sigMousePress.connect(lambda x, y, ev: presses.append((x, y)))
    before = canvas.viewport_image_rect()
    QApplication.sendEvent(
        canvas.viewport(), _press(200.0, 200.0, Qt.MouseButton.MiddleButton)
    )
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(240.0, 230.0),
        QPointF(240.0, 230.0),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas.viewport(), move)
    QApplication.processEvents()

    assert presses == []
    assert canvas.viewport_image_rect() != before


def test_canvas_renders_with_overlay_and_pixel_grid(qapp, two_masks):
    from PySide6.QtGui import QImage, QPainter

    masks, order = two_masks
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(40, 50))
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    canvas.set_overlay(ov)
    canvas.set_zoom(8.0)  # > 4 => pixel grid is drawn
    QApplication.processEvents()
    assert canvas.zoom_factor() > 4.0

    target = QImage(200, 200, QImage.Format.Format_ARGB32)
    target.fill(0)
    painter = QPainter(target)
    canvas.render(painter)
    painter.end()
    assert target.constBits() is not None


def test_refresh_with_a_dirty_rect_updates_only_that_region(qapp, two_masks):
    masks, order = two_masks
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(40, 50))
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    canvas.set_overlay(ov)
    canvas.refresh()

    rect = ov.paint((40, 30), 3, True)
    canvas.refresh(rect)
    assert ov.last_rebuild_rect == _grow(rect, (40, 50))
    # the item draws straight from the overlay buffer: no copy, full frame size
    assert canvas.overlay_image() is ov.qimage()
    assert canvas.overlay_image().size().toTuple() == (50, 40)


def test_refresh_ignores_an_empty_dirty_rect(qapp, two_masks):
    """A stroke entirely outside the image must not trigger a full repaint."""
    masks, order = two_masks
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(40, 50))
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    canvas.set_overlay(ov)
    canvas.refresh()

    tool = BrushTool(canvas, ov, radius=3)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)
    ov.last_rebuild_rect = None
    tool.on_press(-50.0, -50.0, None)
    tool.on_release(-60.0, -50.0, None)
    assert strokes == []
    assert ov.last_rebuild_rect is None
    assert not ov.editing.any()


def test_minimap_stays_in_the_corner_while_the_view_scrolls(qapp):
    canvas = _shown(ImageCanvas(), 400, 400)
    canvas.set_image(_rgb(900, 900))
    mini = canvas.minimap()
    corner = (
        canvas.viewport().width() - mini.width() - 8,
        canvas.viewport().height() - mini.height() - 8,
    )
    assert (mini.x(), mini.y()) == corner

    # zooming and panning scroll the scene; the minimap must not ride along
    canvas.zoom_to((300, 300, 600, 600))
    QApplication.processEvents()
    assert (mini.x(), mini.y()) == corner
    canvas.center_on((700.0, 700.0))
    QApplication.processEvents()
    assert (mini.x(), mini.y()) == corner


def test_minimap_tracks_the_viewport(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 500))
    canvas.zoom_to((100, 100, 200, 200))
    QApplication.processEvents()
    mini = canvas.minimap()
    assert mini.isVisibleTo(canvas)
    assert mini.image_hw == (400, 500)
    x0, y0, x1, y1 = mini.view_rect
    assert x1 > x0 and y1 > y0
    assert (x1 - x0) < 500  # not the whole image


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@pytest.fixture
def rig(qapp):
    canvas = _shown(ImageCanvas(), 300, 300)
    canvas.set_image(_rgb(60, 80))
    ov = LabelOverlay((60, 80))
    ov.set_editing("inst-x", np.zeros((60, 80), dtype=bool))
    canvas.set_overlay(ov)
    return canvas, ov


def test_tool_base_is_a_no_op(rig):
    canvas, ov = rig
    tool = Tool(canvas, ov)
    tool.on_press(1.0, 2.0, None)
    tool.on_move(1.0, 2.0, None)
    tool.on_release(1.0, 2.0, None)
    assert not ov.editing.any()


def test_tool_attach_connects_the_canvas_signals(rig):
    canvas, ov = rig
    tool = BrushTool(canvas, ov, radius=3)
    tool.attach()
    QApplication.sendEvent(canvas.viewport(), _press(150.0, 150.0))
    QApplication.processEvents()
    assert ov.editing.any()

    ov.editing[:] = False
    tool.detach()
    QApplication.sendEvent(canvas.viewport(), _press(150.0, 150.0))
    QApplication.processEvents()
    assert not ov.editing.any()


def test_brush_paints_a_continuous_stroke_and_emits_the_dirty_rect(rig):
    canvas, ov = rig
    tool = BrushTool(canvas, ov, radius=3)
    strokes: list[tuple[int, int, int, int]] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(10.0, 10.0, None)
    tool.on_move(40.0, 10.0, None)  # one big jump: must be interpolated
    tool.on_release(40.0, 10.0, None)

    assert ov.editing[10, 10] and ov.editing[10, 40]
    assert ov.editing[10, 25], "the stroke has a gap between samples"
    assert len(strokes) == 1
    x0, y0, x1, y1 = strokes[0]
    assert x0 <= 7 and x1 >= 44 and y0 <= 7 and y1 >= 14


def test_brush_snapshots_the_mask_before_the_stroke(rig):
    canvas, ov = rig
    tool = BrushTool(canvas, ov, radius=2)
    tool.on_press(20.0, 20.0, None)
    tool.on_release(20.0, 20.0, None)
    assert tool.stroke_before is not None
    assert not tool.stroke_before.any()
    assert ov.editing.any()


def test_brush_radius_is_adjustable(rig):
    canvas, ov = rig
    tool = BrushTool(canvas, ov, radius=2)
    tool.set_radius(9)
    assert tool.radius == 9
    rect = ov.paint((30, 30), tool.radius, True)
    assert rect == (21, 21, 40, 40)


def test_eraser_removes_pixels(rig):
    canvas, ov = rig
    ov.editing[:] = True
    tool = EraserTool(canvas, ov, radius=4)
    tool.on_press(30.0, 30.0, None)
    tool.on_release(30.0, 30.0, None)
    assert not ov.editing[30, 30]
    assert ov.editing[0, 0]


def test_occluder_tool_paints_the_layer_of_its_type(rig):
    canvas, ov = rig
    tool = OccluderTool(canvas, ov, radius=3)
    assert tool.occluder_type == "hand"
    tool.occluder_type = "tool"
    tool.on_press(25.0, 25.0, None)
    tool.on_release(25.0, 25.0, None)
    assert ov.occluders["tool"][25, 25]
    assert set(ov.occluders) == {"tool"}
    assert not ov.editing.any()


def test_occluder_tool_switching_type_does_not_merge_strokes(rig):
    canvas, ov = rig
    tool = OccluderTool(canvas, ov, radius=3)
    tool.occluder_type = "hand"
    tool.on_press(10.0, 10.0, None)
    tool.on_release(10.0, 10.0, None)
    tool.occluder_type = "cable"
    tool.on_press(40.0, 40.0, None)
    tool.on_release(40.0, 40.0, None)

    assert set(ov.occluders) == {"hand", "cable"}
    assert ov.occluders["hand"][10, 10] and not ov.occluders["hand"][40, 40]
    assert ov.occluders["cable"][40, 40] and not ov.occluders["cable"][10, 10]
    # the before-snapshot follows the type too, so undo stays per layer
    assert tool.stroke_before is not None and not tool.stroke_before.any()


def test_occluder_tool_rejects_an_unknown_type(rig):
    canvas, ov = rig
    with pytest.raises(ValueError):
        OccluderTool(canvas, ov, occluder_type="banana")


def test_sam_point_tool_submits_a_viewport_crop_with_crop_coords(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)  # whole 60x80 image visible
    QApplication.processEvents()
    queue = StubQueue()
    tool = SamPointTool(canvas, ov, queue)

    tool.on_press(20.0, 30.0, _press(0, 0, Qt.MouseButton.LeftButton))
    req = queue.last
    assert req.image_crop.shape == (60, 80, 3)
    assert req.image_crop.dtype == np.uint8
    assert req.points == [(20.0, 30.0, 1)]
    assert req.mask_input is None
    assert req.multimask is True  # a single point asks for candidates


def test_sam_point_tool_right_click_is_a_negative_point(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    queue = StubQueue()
    tool = SamPointTool(canvas, ov, queue)
    tool.on_press(20.0, 30.0, _press(0, 0, Qt.MouseButton.LeftButton))
    tool.on_press(40.0, 30.0, _press(0, 0, Qt.MouseButton.RightButton))
    assert queue.last.points == [(20.0, 30.0, 1), (40.0, 30.0, 0)]
    assert queue.last.multimask is False


def test_sam_point_tool_downscales_a_large_crop_and_scales_points(qapp):
    canvas = _shown(ImageCanvas(), 300, 300)
    canvas.set_image(_rgb(2000, 1500))
    ov = LabelOverlay((2000, 1500))
    canvas.set_overlay(ov)
    canvas.set_zoom(0.05)  # everything visible => crop is the whole image
    QApplication.processEvents()
    queue = StubQueue()
    tool = SamPointTool(canvas, ov, queue)

    tool.on_press(1000.0, 1000.0, None)
    req = queue.last
    h, w = req.image_crop.shape[:2]
    assert max(h, w) == 1024
    scale = 1024 / 2000
    px, py, label = req.points[0]
    assert px == pytest.approx(1000.0 * scale, abs=1.0)
    assert py == pytest.approx(1000.0 * scale, abs=1.0)
    assert label == 1


def test_sam_point_tool_refine_mode_passes_the_cropped_editing_mask(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    prior = np.zeros((60, 80), dtype=bool)
    prior[10:20, 10:20] = True
    ov.set_editing("inst-x", prior)
    queue = StubQueue()
    tool = SamPointTool(canvas, ov, queue, refine=True)

    tool.on_press(15.0, 15.0, None)
    req = queue.last
    assert req.mask_input is not None
    assert req.mask_input.shape == req.image_crop.shape[:2]
    assert req.mask_input[15, 15]
    assert not req.mask_input[50, 50]


def test_sam_point_tool_result_replaces_the_editing_layer(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    ov.editing[0:5, 0:5] = True
    queue = StubQueue(_blob_result)
    tool = SamPointTool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    # _blob_result fills rows h/4..h/2, cols w/4..w/2 of the 60x80 crop
    assert ov.editing[20, 25]
    assert not ov.editing[0, 0], "non-refine results replace the layer"


def test_sam_point_tool_refine_keeps_the_mask_outside_the_crop(qapp):
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(200, 200))
    ov = LabelOverlay((200, 200))
    prior = np.zeros((200, 200), dtype=bool)
    prior[180:200, 180:200] = True  # far from the zoomed viewport
    ov.set_editing("inst-x", prior)
    canvas.set_overlay(ov)
    canvas.zoom_to((0, 0, 60, 60))
    QApplication.processEvents()

    queue = StubQueue(_blob_result)
    tool = SamPointTool(canvas, ov, queue, refine=True)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)
    tool.on_press(20.0, 20.0, None)
    assert _spin(lambda: bool(strokes))
    assert ov.editing[190, 190], "refine must keep pixels outside the crop"


class ThreadRecordingSamTool(SamPointTool):
    """Records which thread actually touches the overlay."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.apply_threads: list[int] = []

    def _on_result(self, payload) -> None:
        self.apply_threads.append(threading.get_ident())
        super()._on_result(payload)


def test_sam_result_is_applied_on_the_gui_thread(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    queue = ThreadedQueue(_blob_result)
    tool = ThreadRecordingSamTool(canvas, ov, queue)

    tool.on_press(20.0, 30.0, None)
    for thread in queue.threads:
        thread.join(5.0)
        assert not thread.is_alive()
    # The worker callback has returned, yet nothing may have been applied yet:
    # anything else would mean the overlay was written from the worker thread.
    assert not ov.editing.any(), "the mask was applied off the GUI thread"
    assert tool.apply_threads == []

    assert _spin(lambda: bool(tool.apply_threads)), "no SAM result arrived"
    assert tool.apply_threads == [MAIN_THREAD]
    assert ov.editing[20, 25]


def test_sam_point_tool_clear_points(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    queue = StubQueue()
    tool = SamPointTool(canvas, ov, queue)
    tool.on_press(20.0, 30.0, None)
    tool.clear_points()
    assert tool.points == []
    tool.on_press(40.0, 40.0, None)
    assert queue.last.points == [(40.0, 40.0, 1)]


def test_sam_point_tool_cycles_through_the_candidates(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)  # the whole 60x80 image is the crop: mask == image coords
    QApplication.processEvents()
    queue = StubQueue(_multi_result)
    tool = SamPointTool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    assert tool.candidate_count == 3
    assert tool.candidate_index == 0
    # candidate 0: rows 15..30, cols 20..40
    assert ov.editing[20, 25] and not ov.editing[32, 25]

    assert tool.cycle_candidate() == 1
    assert tool.candidate_index == 1
    # candidate 1: rows 15..35, cols 20..46
    assert ov.editing[32, 25] and not ov.editing[40, 25]

    tool.cycle_candidate()
    assert tool.candidate_index == 2
    assert ov.editing[40, 25], "candidate 2 is the largest"

    tool.cycle_candidate()  # wraps around
    assert tool.candidate_index == 0
    assert not ov.editing[32, 25], "wrapping must restore the first candidate"

    tool.cycle_candidate(-1)  # and cycles backwards
    assert tool.candidate_index == 2
    assert ov.editing[40, 25]
    assert len(strokes) == 5, "every candidate swap is one undoable edit"


def test_sam_point_tool_resets_the_candidates_on_a_new_request(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    queue = StubQueue(_multi_result)
    tool = SamPointTool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    tool.cycle_candidate()
    assert tool.candidate_index == 1

    tool.clear_points()
    tool.on_press(30.0, 30.0, None)
    assert tool.candidate_index == 0, "a new request resets the index"
    assert tool.candidate_count == 0, "and forgets the stale candidates"
    assert _spin(lambda: len(strokes) > 2)
    assert tool.candidate_count == 3


def test_cycle_candidate_without_a_result_is_a_no_op(rig):
    canvas, ov = rig
    tool = SamPointTool(canvas, ov, StubQueue())
    assert tool.candidate_count == 0
    assert tool.cycle_candidate() == 0
    assert not ov.editing.any()


def test_cycle_candidate_does_nothing_when_sam_offered_only_one_mask(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    queue = StubQueue(_blob_result)  # a single-candidate result
    tool = SamPointTool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    assert tool.candidate_count == 1
    assert tool.cycle_candidate() == 0
    assert len(strokes) == 1, "a pointless swap must not create an undo step"


def test_sam_point_tool_sends_point_plus_box_when_a_prompt_box_is_set(rig):
    """The diff map (or the box tool) supplies the box; the click supplies the point."""
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    queue = StubQueue()
    tool = SamPointTool(canvas, ov, queue)

    tool.set_prompt_box((10.0, 12.0, 50.0, 52.0))
    tool.on_press(20.0, 30.0, None)
    req = queue.last
    assert req.points == [(20.0, 30.0, 1)]
    assert req.box == (10.0, 12.0, 50.0, 52.0)
    assert req.multimask is False, "point+box is unambiguous, so no candidates"

    tool.set_prompt_box(None)
    tool.clear_points()
    tool.on_press(20.0, 30.0, None)
    assert queue.last.box is None
    assert queue.last.multimask is True


def test_sam_point_tool_refine_does_not_ask_for_candidates(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    prior = np.zeros((60, 80), dtype=bool)
    prior[10:20, 10:20] = True
    ov.set_editing("inst-x", prior)
    queue = StubQueue()
    tool = SamPointTool(canvas, ov, queue, refine=True)

    tool.on_press(15.0, 15.0, None)
    assert queue.last.mask_input is not None
    assert queue.last.multimask is False


def test_sam_box_tool_submits_the_dragged_box(rig):
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    queue = StubQueue()
    tool = SamBoxTool(canvas, ov, queue)

    tool.on_press(40.0, 45.0, None)
    tool.on_move(10.0, 15.0, None)  # dragged backwards: must be normalised
    assert tool.box == (10.0, 15.0, 40.0, 45.0)
    tool.on_release(10.0, 15.0, None)

    req = queue.last
    assert req.box == (10.0, 15.0, 40.0, 45.0)
    assert req.points == []
    assert req.image_crop.shape == (60, 80, 3)


def test_sam_box_tool_ignores_a_degenerate_drag(rig):
    canvas, ov = rig
    queue = StubQueue()
    tool = SamBoxTool(canvas, ov, queue)
    tool.on_press(20.0, 20.0, None)
    tool.on_release(20.0, 20.0, None)
    assert queue.requests == []


def test_sam_tools_without_a_queue_do_not_crash(rig):
    canvas, ov = rig
    tool = SamPointTool(canvas, ov, None)
    tool.on_press(10.0, 10.0, None)  # no queue: collects the point, submits nothing
    assert tool.points == [(10.0, 10.0, 1)]


# ---------------------------------------------------------------------------
# UndoStack
# ---------------------------------------------------------------------------
def _mask(hw=(20, 20), box=(2, 2, 8, 8)) -> np.ndarray:
    m = np.zeros(hw, dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


class MaskHolder:
    """Minimal undo target: one editing mask per instance."""

    def __init__(self, hw=(20, 20)) -> None:
        self.hw = hw
        self.masks: dict[str, np.ndarray] = {}
        self.calls = 0

    def apply(self, payload: dict) -> None:
        self.calls += 1
        self.masks[payload["instance"]] = M.decode_rle(payload["rle_after"])


def test_op_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        Op("not-a-kind", {}, {})
    assert "edit_editing_mask" in KINDS


def test_edit_editing_mask_op_stores_rle_not_arrays():
    before, after = _mask(), _mask(box=(2, 2, 12, 12))
    op = edit_editing_mask_op("inst-a", before, after)
    assert op.kind == "edit_editing_mask"
    assert set(op.payload) == {"instance", "rle_before", "rle_after"}
    for value in op.payload.values():
        assert not isinstance(value, np.ndarray)
    assert isinstance(op.payload["rle_after"], dict)
    assert isinstance(op.payload["rle_after"]["counts"], str)
    # the inverse is the same op with the two masks swapped
    assert op.inverse["rle_after"] == op.payload["rle_before"]
    assert op.inverse["rle_before"] == op.payload["rle_after"]
    np.testing.assert_array_equal(M.decode_rle(op.payload["rle_after"]), after)


def test_push_applies_and_undo_redo_restore_masks_exactly():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)

    before, after = _mask(), _mask(box=(5, 5, 15, 15))
    stack.push(edit_editing_mask_op("inst-a", before, after))
    np.testing.assert_array_equal(holder.masks["inst-a"], after)

    op = stack.undo()
    assert op is not None and op.kind == "edit_editing_mask"
    np.testing.assert_array_equal(holder.masks["inst-a"], before)

    op = stack.redo()
    assert op is not None
    np.testing.assert_array_equal(holder.masks["inst-a"], after)


def test_push_without_apply_only_records():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)
    stack.push(edit_editing_mask_op("inst-a", _mask(), _mask(box=(1, 1, 3, 3))), apply=False)
    assert holder.calls == 0
    assert stack.can_undo
    stack.undo()
    assert holder.calls == 1


def test_undo_and_redo_are_empty_at_the_edges():
    stack = UndoStack()
    assert stack.undo() is None
    assert stack.redo() is None
    assert not stack.can_undo and not stack.can_redo


def test_a_new_push_clears_the_redo_branch():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)
    stack.push(edit_editing_mask_op("a", _mask(), _mask(box=(1, 1, 4, 4))))
    stack.undo()
    assert stack.can_redo
    stack.push(edit_editing_mask_op("a", _mask(), _mask(box=(6, 6, 9, 9))))
    assert not stack.can_redo
    assert stack.redo() is None


def test_stack_is_capped_at_200_entries():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)
    for i in range(250):
        stack.push(edit_editing_mask_op("a", _mask(), _mask(box=(1, 1, 2 + i % 10, 3))))
    assert stack.LIMIT == 200
    assert len(stack) == 200


def test_push_needs_a_registered_handler():
    stack = UndoStack()
    with pytest.raises(LookupError):
        stack.push(Op("set_zorder", {"a": 1}, {"a": 0}))


def test_other_kinds_round_trip_through_their_handlers():
    seen: list[tuple[str, dict]] = []
    stack = UndoStack()
    for kind in ("set_zorder", "set_pair_override", "set_frame_override",
                 "set_occluder", "commit_keyframe"):
        stack.register(
            kind,
            lambda p, k=kind: seen.append((f"do:{k}", p)),
            lambda p, k=kind: seen.append((f"undo:{k}", p)),
        )
        stack.push(Op(kind, {"v": 1}, {"v": 0}))
    assert [name for name, _ in seen] == [
        "do:set_zorder", "do:set_pair_override", "do:set_frame_override",
        "do:set_occluder", "do:commit_keyframe",
    ]
    stack.undo()
    assert seen[-1] == ("undo:commit_keyframe", {"v": 0})


def test_clear_drops_both_branches():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)
    stack.push(edit_editing_mask_op("a", _mask(), _mask(box=(1, 1, 4, 4))))
    stack.clear()
    assert not stack.can_undo and not stack.can_redo and len(stack) == 0
