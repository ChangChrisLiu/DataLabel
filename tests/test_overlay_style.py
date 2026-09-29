"""The canvas' outlines are readable at DPR 1.5 (task U2g).

"很多虚线的部分很不明显": the stored ROI was one device pixel of yellow at alpha
150 on an olive-yellow chassis, the hover hints were bare 2 px dashes with bare
labels.  What is pinned here:

* every outline is two-tone -- a dark under-stroke two logical pixels wider
  than a fully opaque bright stroke -- with the widths the ruling names;
* the widths are *logical*: a cosmetic pen's width is device pixels on this Qt,
  so the style multiplies by the ratio, measured here on a 1.5 image;
* no instance colour can be an overlay's colour;
* the labels are chips, drawn, kept inside the viewport, and the ROI's goes
  with the ``A`` toggle;
* the armed prompt box and a drag are two bands, not one slot.
"""
from __future__ import annotations

import colorsys
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from tda.ui.canvas import overlay_style as OS
from tda.ui.canvas.overlay import EDIT_RGB, GHOST_RGB, PALETTE_64
from tda.ui.canvas.view import ImageCanvas


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


STYLES = [OS.ROI_STORED, OS.ROI_EDITING, OS.PROMPT_BOX, OS.DRAG_BAND,
          OS.PROMPT_POINT, OS.hint_style(OS.DRAFT_RGB), OS.hint_style(OS.SHAPE_RGB),
          OS.hint_style(OS.PROMPT_RGB)]


def _canvas(w: int = 400, h: int = 300, image=None) -> ImageCanvas:
    canvas = ImageCanvas()
    canvas.resize(w, h)
    canvas.set_image(np.full((200, 200, 3), 128, dtype=np.uint8) if image is None else image)
    canvas.set_zoom(1.0)
    canvas.center_on((100, 100))
    canvas.show()
    QApplication.processEvents()
    return canvas


def _screen(canvas: ImageCanvas) -> np.ndarray:
    """The viewport as the annotator sees it, ``H x W x 3`` uint8 RGB."""
    QApplication.processEvents()
    image = canvas.viewport().grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    raw = np.frombuffer(bytes(image.constBits()), dtype=np.uint8)
    return raw.reshape(h, image.bytesPerLine())[:, :3 * w].reshape(h, w, 3).copy()


def _near(pixels: np.ndarray, rgb, tol: int = 40) -> np.ndarray:
    return np.all(np.abs(pixels.astype(int) - np.array(rgb)[None, :]) <= tol, axis=-1)


# --------------------------------------------------------------------------- #
# the style itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
@pytest.mark.parametrize("dpr", [1.0, 1.5])
def test_every_outline_is_a_dark_under_stroke_and_an_opaque_bright_one(qapp, style, dpr):
    under, over = style.under_pen(dpr), style.over_pen(dpr)
    assert under.isCosmetic() and over.isCosmetic()
    assert under.color().getRgb()[:3] == (0, 0, 0)
    assert under.color().alpha() >= 200
    assert over.color().alpha() == 255
    assert over.color().getRgb()[:3] == tuple(style.rgb)
    # logical widths, scaled to the device: under = over + 2
    assert over.widthF() == pytest.approx(style.width * dpr)
    assert under.widthF() == pytest.approx((style.width + 2.0) * dpr)
    assert under.style() == Qt.PenStyle.SolidLine
    if style.dashed:
        # A custom pattern whose dash and gap are DASH_PX / GAP_PX logical
        # pixels whatever the width -- not Qt's 4:2 widths, a fuzz at 1 px.
        on, off = over.dashPattern()
        assert on * over.widthF() == pytest.approx(OS.DASH_PX * dpr)
        assert off * over.widthF() == pytest.approx(OS.GAP_PX * dpr)
        assert over.capStyle() == Qt.PenCapStyle.FlatCap


def test_the_widths_are_the_ones_the_ruling_names():
    assert OS.ROI_STORED.width == 2.0
    assert OS.ROI_EDITING.width == 3.0
    assert OS.HINT_PX == 3.0 and OS.hint_style((1, 2, 3)).width == 3.0
    assert OS.DRAG_BAND.width == 2.0
    assert 2.0 <= OS.PROMPT_BOX.width <= 3.0
    assert OS.UNDER_EXTRA_PX == 2.0


def _runs(line: np.ndarray) -> list[int]:
    """Lengths of the True runs of a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.r_[0, line.astype(int), 0]))
    return [int(b - a) for a, b in zip(edges[::2], edges[1::2])]


def test_widths_are_logical_pixels_on_a_150_percent_device(qapp):
    """Measured, not assumed: a cosmetic width is device pixels on this Qt.

    Two logical pixels of bright over four of dark must be three over six on
    the annotator's screen -- a stroke that came out two device pixels wide
    would be the old "1 px" line all over again.
    """
    image = QImage(300, 300, QImage.Format.Format_RGB32)
    image.setDevicePixelRatio(1.5)
    image.fill(QColor(255, 255, 255))
    painter = QPainter(image)
    OS.draw_outline_rect(painter, QRectF(20, 20, 150, 150), OS.ROI_STORED, 1.5)
    painter.end()
    arr = np.frombuffer(bytes(image.constBits()), dtype=np.uint32).reshape(300, 300)
    r, g, b = (arr >> 16) & 0xFF, (arr >> 8) & 0xFF, arr & 0xFF
    # a column through a dash of the top edge (device rows 0..60)
    for column in range(40, 250):
        bright = (r[:60, column] > 200) & (g[:60, column] < 140) & (b[:60, column] > 180)
        if bright.any():
            break
    else:
        pytest.fail("no dash crosses the top edge")
    dark = (r[:60, column] < 60) & (g[:60, column] < 60) & (b[:60, column] < 60)
    assert _runs(bright) == [3], _runs(bright)          # 2 logical px x 1.5
    assert _runs(dark | bright) == [6], _runs(dark | bright)   # 4 x 1.5


# --------------------------------------------------------------------------- #
# colours
# --------------------------------------------------------------------------- #
def test_no_instance_colour_has_an_overlay_hue():
    for rgb in OS.RESERVED_RGBS:
        hue = OS.hue_degrees(rgb)
        nearest = min(OS.hue_distance(hue, OS.hue_degrees(p)) for p in PALETTE_64)
        # RGB rounding moves a hue by a fraction of a degree
        assert nearest >= OS.reserved_half_width(rgb) - 1.0, (rgb, nearest)


def test_only_the_roi_band_is_widened_to_25_degrees():
    """U2h: violet and pink 15-18 degrees from the ROI read as the ROI."""
    assert OS.reserved_half_width(OS.ROI_RGB) == OS.ROI_HUE_HALF_WIDTH == 25.0
    for rgb in (OS.PROMPT_RGB, OS.DRAFT_RGB, OS.SHAPE_RGB):
        assert OS.reserved_half_width(rgb) == OS.RESERVED_HUE_HALF_WIDTH == 15.0
    roi = OS.hue_degrees(OS.ROI_RGB)
    # D13's screw.motherboard.05 under U2g's palette, and the pink next to it
    for seen in ((217, 92, 242), (242, 29, 160)):
        assert OS.hue_distance(roi, OS.hue_degrees(seen)) < OS.ROI_HUE_HALF_WIDTH - 1.0
        assert seen not in PALETTE_64
    nearest = min(OS.hue_distance(roi, OS.hue_degrees(p)) for p in PALETTE_64)
    assert nearest >= OS.ROI_HUE_HALF_WIDTH - 1.0, nearest
    # the palette's wheel: 360 less four bands, the ROI's the wider
    from tda.ui.canvas.overlay import _free_arcs

    free = sum(hi - lo for lo, hi in _free_arcs())
    assert free == pytest.approx(360.0 - 2 * 25.0 - 3 * 2 * 15.0)


def test_the_drag_band_is_a_colour_the_palette_cannot_make():
    sat = colorsys.rgb_to_hsv(*(c / 255 for c in OS.DRAG_RGB))[1]
    assert sat < 0.1
    assert min(colorsys.rgb_to_hsv(*(c / 255 for c in p))[1] for p in PALETTE_64) >= 0.6


def test_the_palette_is_still_64_distinct_colours():
    assert len(set(PALETTE_64)) == 64


def test_every_overlay_kind_has_its_own_colour():
    rgbs = list(OS.RESERVED_RGBS)
    for i, a in enumerate(rgbs):
        for b in rgbs[i + 1:]:
            # the two bands do not overlap
            assert (OS.hue_distance(OS.hue_degrees(a), OS.hue_degrees(b))
                    >= OS.reserved_half_width(a) + OS.reserved_half_width(b))
    assert OS.DRAG_RGB not in OS.RESERVED_RGBS


def test_the_roi_is_not_yellow_any_more():
    """It sat on the olive-yellow chassis (186, 179, 22) in its own colour."""
    assert OS.hue_distance(OS.hue_degrees(OS.ROI_RGB), OS.hue_degrees(EDIT_RGB)) >= 90
    assert OS.hue_distance(OS.hue_degrees(OS.ROI_RGB),
                           OS.hue_degrees((186, 179, 22))) >= 90


def test_a_draft_is_one_colour_wherever_it_is_shown():
    assert GHOST_RGB == OS.DRAFT_RGB


# --------------------------------------------------------------------------- #
# on the canvas
# --------------------------------------------------------------------------- #
def test_a_stored_roi_is_drawn_two_tone_in_its_own_colour(qapp):
    canvas = _canvas()
    try:
        canvas.set_roi((40, 40, 160, 160), editing=False)
        screen = _screen(canvas)
        roi = _near(screen, OS.ROI_RGB)
        dark = _near(screen, (0, 0, 0), tol=40)
        assert roi.sum() > 100, "no ROI-coloured pixel on screen"
        # every bright pixel has dark directly beside it: the under-stroke
        ys, xs = np.nonzero(roi)
        y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
        around = dark[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        assert around.any(), "no dark under-stroke next to the ROI's stroke"
        canvas.roi_outline_visible = False
        canvas.viewport().update()
        assert not _near(_screen(canvas), OS.ROI_RGB).any(), "A did not hide it"
    finally:
        canvas.close()


def test_the_hints_are_two_tone_in_their_colours(qapp):
    canvas = _canvas()
    try:
        canvas.set_hint_boxes([((30, 30, 90, 90), "旧草稿 ls:X#1（第 3 帧）", OS.DRAFT_RGB),
                               ((110, 110, 170, 170), "第 3 帧画的形状", OS.SHAPE_RGB)])
        screen = _screen(canvas)
        assert _near(screen, OS.DRAFT_RGB, tol=20).sum() > 50
        assert _near(screen, OS.SHAPE_RGB, tol=20).sum() > 50
    finally:
        canvas.close()


def test_a_repaint_that_misses_the_stroke_band_strokes_nothing(qapp):
    """A brush dab inside the ROI must not pay for the ROI's dashed stroke."""
    from unittest import mock

    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(QColor(255, 255, 255))
    painter = QPainter(image)
    box = QRectF(20, 20, 150, 150)
    with mock.patch.object(QPainter, "drawRect", autospec=True) as draw:
        OS.draw_outline_rect(painter, box, OS.ROI_STORED, exposed=QRectF(80, 80, 10, 10))
        OS.draw_outline_rect(painter, box, OS.ROI_STORED, exposed=QRectF(190, 190, 5, 5))
        assert draw.call_count == 0, "stroked for a repaint that cannot show it"
        OS.draw_outline_rect(painter, box, OS.ROI_STORED, exposed=QRectF(15, 80, 10, 10))
        assert draw.call_count == 2, "a repaint across the edge must stroke it"
    painter.end()


def test_a_box_that_names_a_part_is_stroked_outside_it(qapp):
    """A screw's box is ~20 px: a stroke centred on its edge covered the screw."""
    canvas = _canvas()
    try:
        canvas.set_zoom(2.0)
        canvas.center_on((100, 100))
        QApplication.processEvents()
        canvas.set_rubber_band((90, 90, 110, 110), kind="prompt", label="")
        screen = _screen(canvas)
        corner = canvas.mapFromScene(90.0, 90.0)
        x, y = int(corner.x()), int(corner.y())
        mid = int(canvas.mapFromScene(100.0, 100.0).y())
        grey = screen[mid, x + 1:x + 6]           # just inside the left edge
        assert np.all(np.abs(grey.astype(int) - 128) <= 12), grey
        outer = OS.PROMPT_BOX.under_width
        stroke = screen[mid, x - int(outer):x]    # just outside it
        assert (_near(stroke, OS.PROMPT_RGB, tol=30) | _near(stroke, (0, 0, 0), 40)).all()
        # the ROI keeps its line on the rectangle (its handles sit on the corners)
        canvas.set_rubber_band(None, kind="prompt")
        canvas.set_roi((90, 90, 110, 110), editing=False)
        screen = _screen(canvas)
        edge = screen[mid, x - 1:x + 2]
        assert not np.all(np.abs(edge.astype(int) - 128) <= 12)
    finally:
        canvas.close()


def test_every_hint_and_the_roi_carry_a_chip_that_is_drawn(qapp):
    canvas = _canvas()
    try:
        canvas.set_roi((20, 20, 180, 180), editing=False)
        canvas.set_hint_boxes([((60, 60, 120, 120), "差异最大处（SAM 提示框）", OS.PROMPT_RGB),
                               ((130, 60, 170, 100), "第 7 帧画的形状", OS.SHAPE_RGB)])
        layout = canvas.chip_layout()
        texts = [text for text, _rect, _pix in layout]
        assert texts == [OS.ROI_CHIP, "差异最大处（SAM 提示框）", "第 7 帧画的形状"]
        view = QRectF(canvas.viewport().rect())
        rects = [rect for _t, rect, _p in layout]
        for rect in rects:
            assert view.contains(rect), rect
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                assert not a.intersects(b), "two chips on top of each other"
        # drawn: the chip's dark background is on screen where it was laid out
        screen = _screen(canvas)
        for rect in rects:
            inner = rect.adjusted(3, 3, -3, -3).toRect()
            patch = screen[inner.top():inner.bottom(), inner.left():inner.right()]
            assert _near(patch, OS.CHIP_BG.getRgb()[:3], tol=30).mean() > 0.3, rect
        # the chip names the box it sits on: at its top-left corner
        roi_rect = rects[0]
        corner = canvas.mapFromScene(20.0, 20.0)
        assert abs(roi_rect.left() - corner.x()) <= OS.ROI_STORED.under_width
        assert roi_rect.bottom() <= corner.y()
    finally:
        canvas.close()


def test_a_chip_stays_inside_the_viewport_when_its_corner_is_off_screen(qapp):
    canvas = _canvas(300, 200)
    try:
        canvas.set_zoom(4.0)
        canvas.center_on((150, 150))
        QApplication.processEvents()
        # the box's top-left corner (40, 40) is far outside the viewport
        canvas.set_hint_boxes([((40, 40, 170, 170), "旧草稿 ls:Screw#1（第 9 帧）",
                                OS.DRAFT_RGB)])
        corner = canvas.mapFromScene(40.0, 40.0)
        assert corner.x() < 0 and corner.y() < 0
        (_text, rect, _pix), = canvas.chip_layout()
        view = QRectF(canvas.viewport().rect())
        assert view.contains(rect)
        assert rect.left() <= OS.CHIP_MARGIN + 1 and rect.top() <= OS.CHIP_MARGIN + 4
        # a box nowhere on screen gets no chip at all
        canvas.set_hint_boxes([((0, 0, 10, 10), "第 1 帧画的形状", OS.SHAPE_RGB)])
        assert canvas.chip_layout() == []
    finally:
        canvas.close()


def test_a_chip_that_would_collide_moves_to_another_corner_of_its_own_box():
    """Five screws a few mm apart: each label has to stay next to its own screw."""
    view = QRectF(0, 0, 800, 600)
    first = OS.place_chip((200, 20), QRectF(300, 300, 20, 20), view)
    assert first == QRectF(300, 300 - OS.CHIP_GAP - 20, 200, 20)     # above-left
    second = OS.place_chip((200, 20), QRectF(200, 305, 20, 20), view, [first])
    assert not second.intersects(first)
    assert second == QRectF(200, 325 + OS.CHIP_GAP, 200, 20)         # under its own box
    third = OS.place_chip((200, 20), QRectF(250, 302, 20, 20), view, [first, second])
    assert not third.intersects(first) and not third.intersects(second)
    # a big box (the ROI under the banner) keeps its label at its top-left, inside
    banner = QRectF(0, 0, 800, 40)
    roi = OS.place_chip((120, 20), QRectF(50, 45, 600, 500), view, [banner])
    assert roi == QRectF(50, 45 + OS.CHIP_GAP, 120, 20)


def test_the_roi_chip_goes_with_the_outline_toggle_but_not_while_editing(qapp):
    canvas = _canvas()
    try:
        canvas.set_roi((20, 20, 180, 180), editing=False)
        assert [t for t, _r, _p in canvas.chip_layout()] == [OS.ROI_CHIP]
        canvas.roi_outline_visible = False
        assert canvas.chip_layout() == []
        # the rectangle being edited is a question, and is never hidden
        canvas.set_roi((20, 20, 180, 180), editing=True)
        assert [t for t, _r, _p in canvas.chip_layout()] == [OS.ROI_CHIP]
    finally:
        canvas.close()


# --------------------------------------------------------------------------- #
# the two bands (U2g addendum A)
# --------------------------------------------------------------------------- #
def test_a_drag_and_the_armed_prompt_box_are_two_bands(qapp):
    canvas = _canvas()
    try:
        canvas.set_rubber_band((50, 50, 70, 70), kind="prompt", label="SAM 提示框（程序猜的位置）")
        canvas.set_rubber_band((100, 100, 150, 150))           # a drag starts
        assert canvas.rubber_band_kind() == "drag"
        screen = _screen(canvas)
        assert _near(screen, OS.PROMPT_RGB, tol=20).sum() > 20, "the prompt box went"
        assert _near(screen, OS.DRAG_RGB, tol=10).sum() > 20, "no white drag band"
        canvas.set_rubber_band(None)                            # the drag ends
        assert canvas.prompt_band() == ((50.0, 50.0, 70.0, 70.0),
                                        "SAM 提示框（程序猜的位置）")
        assert canvas._rubber_band == (50.0, 50.0, 70.0, 70.0)
        texts = [t for t, _r, _p in canvas.chip_layout()]
        assert texts == ["SAM 提示框（程序猜的位置）"]
        canvas.set_rubber_band(None, kind="prompt")
        assert canvas.prompt_band() == (None, "")
    finally:
        canvas.close()


def test_the_prompt_chip_gives_way_to_a_hint_on_the_same_box(qapp):
    canvas = _canvas()
    try:
        canvas.set_rubber_band((50, 50, 70, 70), kind="prompt", label="SAM 提示框（程序猜的位置）")
        canvas.set_hint_boxes([((50, 50, 70, 70), "差异最大处（SAM 提示框）", OS.PROMPT_RGB)])
        assert [t for t, _r, _p in canvas.chip_layout()] == ["差异最大处（SAM 提示框）"]
    finally:
        canvas.close()


def test_the_prompt_box_is_not_drawn_while_the_roi_is_edited(qapp):
    canvas = _canvas()
    try:
        canvas.set_rubber_band((50, 50, 70, 70), kind="prompt", label="SAM 提示框（程序猜的位置）")
        canvas.set_roi((20, 20, 180, 180), editing=True)
        assert not _near(_screen(canvas), OS.PROMPT_RGB, tol=20).any()
        assert [t for t, _r, _p in canvas.chip_layout()] == [OS.ROI_CHIP]
        canvas.set_roi((20, 20, 180, 180), editing=False)       # Esc
        assert _near(_screen(canvas), OS.PROMPT_RGB, tol=20).any()
    finally:
        canvas.close()


def test_unknown_band_kinds_are_refused(qapp):
    canvas = ImageCanvas()
    with pytest.raises(ValueError):
        canvas.set_rubber_band((0, 0, 1, 1), kind="roi")


def test_a_pan_repaints_where_a_pinned_chip_was_and_is(qapp, monkeypatch):
    canvas = _canvas(300, 200)
    try:
        canvas.set_zoom(4.0)
        canvas.center_on((150, 150))
        canvas.set_hint_boxes([((40, 40, 170, 170), "第 2 帧画的形状", OS.SHAPE_RGB)])
        _screen(canvas)                                     # lays the chip out
        assert canvas._chip_rects
        updates: list = []
        viewport = canvas.viewport()
        real = viewport.update
        monkeypatch.setattr(viewport, "update", lambda *a: (updates.append(a), real(*a)))
        canvas.horizontalScrollBar().setValue(canvas.horizontalScrollBar().value() + 30)
        assert len(updates) >= 2, "the pinned chip would be smeared by the blit"
    finally:
        canvas.close()
