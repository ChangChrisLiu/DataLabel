"""Interactive image canvas: zoom, pan, overlay, pixel grid, minimap (spec 4.5).

Two ``QGraphicsPixmapItem``s share one scene: the frame image at the bottom and
the :class:`~tda.ui.canvas.overlay.LabelOverlay` on top.  The scene rect is the
image rect, so **scene coordinates are image coordinates** -- that is what lets
the tool signals report plain image pixels and the SAM crops be taken straight
from :meth:`ImageCanvas.viewport_image_rect`.

Interaction: wheel zooms about the cursor by a factor of 1.25 per notch,
middle-drag or ``Space``-drag pans, a pixel grid appears past 400% and a small
non-interactive minimap in the corner shows where the viewport sits.

Run ``python -m tda.ui.canvas.view --image <path>`` for a standalone smoke test.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from tda.ui.canvas.overlay import LabelOverlay

__all__ = ["ImageCanvas", "MiniMap", "OverlayItem"]

Rect = tuple[int, int, int, int]


class OverlayItem(QGraphicsItem):
    """Scene item drawing a ``QImage`` 1:1 in image coordinates.

    A ``QGraphicsPixmapItem`` would be the obvious choice, but ``setPixmap``
    invalidates the *whole* item, so every brush sample repaints the entire
    visible overlay: at 800% that measured ~22 ms per mouse move, against
    ~0.3 ms here.  This item instead draws straight from the overlay's ARGB
    buffer and is invalidated per dirty rect, so a stroke costs neither a
    pixmap upload nor a full-viewport blit.
    """

    def __init__(self) -> None:
        super().__init__()
        self._image: Optional[QImage] = None
        # Without this flag Qt reports the whole bounding rect as exposed.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    def set_image(self, image: Optional[QImage]) -> None:
        """Point the item at a new image (or clear it) and repaint fully."""
        self.prepareGeometryChange()
        self._image = image
        self.update()

    def image(self) -> Optional[QImage]:
        return self._image

    def boundingRect(self) -> QRectF:  # noqa: D102 - Qt override
        if self._image is None:
            return QRectF()
        return QRectF(0, 0, self._image.width(), self._image.height())

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: D102
        if self._image is None:
            return
        # Nearest neighbour: at high zoom the annotator must see real pixels.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        exposed = option.exposedRect.intersected(self.boundingRect())
        if exposed.isEmpty():
            return
        # Round outwards to whole image pixels: the painter is clipped to the
        # update region anyway, and whole pixels keep the blit seam-free.
        box = QRectF(
            float(np.floor(exposed.left())),
            float(np.floor(exposed.top())),
            float(np.ceil(exposed.width()) + 1),
            float(np.ceil(exposed.height()) + 1),
        ).intersected(self.boundingRect())
        painter.drawImage(box, self._image, box)


class MiniMap(QWidget):
    """Thumbnail of the whole frame with the current viewport outlined.

    A click (or a drag) centres the canvas on that point.  It used to be
    transparent to mouse events, which does not mean "the click is ignored": it
    means the press goes to the canvas *underneath*, at the minimap's own corner
    of the image, so aiming at the thumbnail painted a brush stroke in the far
    corner of the frame.
    """

    MAX_SIDE = 160

    #: Where on the image the annotator asked to look, in image pixels.
    sigCentreOn = Signal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.image_hw: tuple[int, int] = (0, 0)
        self.view_rect: Rect = (0, 0, 0, 0)
        self._thumb = QPixmap()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setVisible(False)

    # -- navigation ---------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: D102 - Qt override
        self._centre_on(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: D102 - Qt override
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._centre_on(event)

    def _centre_on(self, event) -> None:
        """Turn a position on the thumbnail into one on the image, and ask."""
        event.accept()          # swallowed either way: never a stroke underneath
        h, w = self.image_hw
        if w <= 0 or h <= 0 or self._thumb.isNull():
            return
        point = event.position() if hasattr(event, "position") else event.pos()
        x = float(point.x()) * w / max(1, self._thumb.width())
        y = float(point.y()) * h / max(1, self._thumb.height())
        self.sigCentreOn.emit(min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h)))

    def set_image(self, pixmap: QPixmap, hw: tuple[int, int]) -> None:
        self.image_hw = (int(hw[0]), int(hw[1]))
        self._thumb = pixmap.scaled(
            self.MAX_SIDE,
            self.MAX_SIDE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.resize(self._thumb.size())
        self.setVisible(not self._thumb.isNull())
        self.update()

    def set_view_rect(self, rect: Rect) -> None:
        self.view_rect = rect
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D102 - Qt override
        if self._thumb.isNull():
            return
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._thumb)
        h, w = self.image_hw
        if w > 0 and h > 0:
            sx = self._thumb.width() / w
            sy = self._thumb.height() / h
            x0, y0, x1, y1 = self.view_rect
            painter.setPen(QPen(QColor(255, 232, 64), 1))
            painter.drawRect(
                QRectF(x0 * sx, y0 * sy, max(1.0, (x1 - x0) * sx), max(1.0, (y1 - y0) * sy))
            )
        painter.setPen(QPen(QColor(0, 0, 0, 160), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        painter.end()


class ImageCanvas(QGraphicsView):
    """Zoomable image view with a label overlay and image-coordinate signals.

    The three mouse signals carry ``(x, y, event)`` with ``x``/``y`` in image
    pixels (floats, sub-pixel at high zoom).  The event is passed through for
    tools that need the button or modifiers; it is only valid *during* the
    emission, so handlers must not store it.  Pan gestures are swallowed and do
    not reach the tools.
    """

    sigMousePress = Signal(float, float, object)
    sigMouseMove = Signal(float, float, object)
    sigMouseRelease = Signal(float, float, object)
    #: The zoom factor changed, by whatever route -- the wheel, a fit, a
    #: programmatic set.  The status bar's percentage was rewritten only by the
    #: *actions* that zoom, so after a wheel notch it said the old number.
    sigZoomChanged = Signal(float)

    #: Zoom factor per wheel notch (spec 4.5).
    ZOOM_STEP = 1.25
    MIN_ZOOM = 0.02
    MAX_ZOOM = 64.0
    #: Pixel grid appears above this zoom (spec 4.5: "> 400%").
    GRID_ZOOM = 4.0
    #: Fraction of the box added on each side by :meth:`zoom_to`.
    FIT_MARGIN = 0.05

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Plain state first: the Qt setters below can already deliver a resize
        # event, and resizeEvent() reads these.
        self._rgb: Optional[np.ndarray] = None
        self._overlay: Optional[LabelOverlay] = None
        self.overlay_alpha = 110
        self.overlay_outline = True
        self._panning = False
        self._pan_origin = QPointF()
        self._space_down = False
        self._rubber_band: Optional[tuple[float, float, float, float]] = None
        self._last_zoom: Optional[float] = None
        # Parented to the view, not the viewport: QGraphicsView scrolls the
        # viewport's child widgets together with the scene, which would drag the
        # minimap off screen on the first pan.
        self._minimap = MiniMap(self)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = QGraphicsPixmapItem()
        # Nearest neighbour: at 800% the annotator must see real pixels.
        self._image_item.setTransformationMode(
            Qt.TransformationMode.FastTransformation
        )
        self._image_item.setZValue(0)
        self._overlay_item = OverlayItem()
        self._overlay_item.setZValue(1)
        self._scene.addItem(self._image_item)
        self._scene.addItem(self._overlay_item)

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Hidden scrollbars still carry the scroll range, which is what pan and
        # centerOn use; keeping them off makes the viewport size stable and the
        # fitInView arithmetic in zoom_to exact.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor(32, 32, 34))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._minimap.raise_()
        self._minimap.sigCentreOn.connect(lambda x, y: self.center_on((x, y)))
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(lambda _v: self._sync_minimap())

    # -- content ------------------------------------------------------------
    def set_image(self, rgb: np.ndarray) -> None:
        """Show an ``HxWx3`` uint8 RGB frame and fit it in the view."""
        arr = np.ascontiguousarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
            raise ValueError(
                f"expected an HxWx3 uint8 RGB image, got shape {arr.shape} "
                f"dtype {arr.dtype}"
            )
        self._rgb = arr
        h, w = arr.shape[:2]
        image = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)  # copies: the array may be replaced
        self._image_item.setPixmap(pixmap)
        self._overlay_item.set_image(None)
        self.setSceneRect(QRectF(0, 0, w, h))
        self._minimap.set_image(pixmap, (h, w))
        self._place_minimap()  # the thumbnail's size just changed
        self.fit_image()

    def image_rgb(self) -> Optional[np.ndarray]:
        """The frame as handed to :meth:`set_image` (source for SAM crops)."""
        return self._rgb

    def image_hw(self) -> tuple[int, int]:
        return (0, 0) if self._rgb is None else (self._rgb.shape[0], self._rgb.shape[1])

    def set_overlay(self, overlay: Optional[LabelOverlay]) -> None:
        """Attach a label overlay and draw it."""
        self._overlay = overlay
        self._overlay_item.set_image(None)
        self.refresh()

    def overlay(self) -> Optional[LabelOverlay]:
        return self._overlay

    def overlay_image(self) -> Optional[QImage]:
        """The image the overlay item is drawing (shares the overlay buffer)."""
        return self._overlay_item.image()

    def minimap(self) -> MiniMap:
        return self._minimap

    def refresh(self, rect: Optional[Rect] = None) -> None:
        """Recomposite the overlay, invalidating only what actually changed.

        ``rect`` is a hint; the overlay may rebuild a slightly larger region
        (the outline halo) and reports it as ``last_rebuild_rect``, which is
        what gets invalidated.  ``None`` there means nothing was stale, so no
        repaint is scheduled at all.
        """
        if self._overlay is None:
            self._overlay_item.set_image(None)
            return
        image = self._overlay.qimage(
            rect, alpha=self.overlay_alpha, outline=self.overlay_outline
        )
        if self._overlay_item.image() is not image:
            # First draw, or the overlay rebuilt its buffer: repaint it all.
            self._overlay_item.set_image(image)
            return
        rebuilt = self._overlay.last_rebuild_rect
        if rebuilt is None:
            return
        x0, y0, x1, y1 = (int(v) for v in rebuilt)
        self._overlay_item.update(
            QRectF(x0, y0, max(0, x1 - x0), max(0, y1 - y0))
        )

    def set_rubber_band(
        self, box: Optional[tuple[float, float, float, float]]
    ) -> None:
        """Show (or clear) a dashed box, e.g. while dragging a SAM box prompt."""
        self._rubber_band = box
        self.viewport().update()

    # -- zoom / pan ---------------------------------------------------------
    def zoom_factor(self) -> float:
        return float(self.transform().m11())

    def set_zoom(self, factor: float) -> None:
        """Set an absolute zoom factor, keeping the viewport centre."""
        target = self._clamp_zoom(factor)
        centre = self.mapToScene(self.viewport().rect().center())
        self.setTransform(QTransform.fromScale(target, target))
        self.centerOn(centre)
        self._sync_minimap()

    def zoom_at(self, pos: QPointF, factor: float) -> None:
        """Multiply the zoom by ``factor``, keeping the image point under ``pos``."""
        current = self.zoom_factor()
        target = self._clamp_zoom(current * factor)
        step = target / current if current else 1.0
        if abs(step - 1.0) < 1e-9:
            return
        before = self.image_pos(pos)
        self.scale(step, step)
        after = self.image_pos(pos)
        centre = self.mapToScene(self.viewport().rect().center())
        self.centerOn(
            centre.x() + (before[0] - after[0]), centre.y() + (before[1] - after[1])
        )
        self._sync_minimap()

    def fit_image(self) -> None:
        """Fit the whole frame in the view."""
        h, w = self.image_hw()
        if w and h:
            self.zoom_to((0, 0, w, h))

    def zoom_to(self, box: Rect) -> None:
        """Fit ``box`` = ``(x0, y0, x1, y1)`` (image coords) plus a 5% margin."""
        x0, y0, x1, y1 = (float(v) for v in box)
        w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        mx, my = w * self.FIT_MARGIN, h * self.FIT_MARGIN
        self.fitInView(
            QRectF(x0 - mx, y0 - my, w + 2 * mx, h + 2 * my),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        zoom = self.zoom_factor()
        clamped = self._clamp_zoom(zoom)
        if abs(clamped - zoom) > 1e-9:
            self.set_zoom(clamped)
        self._sync_minimap()

    def center_on(self, xy: tuple[float, float]) -> None:
        """Centre the view on an image point."""
        self.centerOn(float(xy[0]), float(xy[1]))
        self._sync_minimap()

    def _clamp_zoom(self, factor: float) -> float:
        return float(min(max(float(factor), self.MIN_ZOOM), self.MAX_ZOOM))

    # -- coordinates --------------------------------------------------------
    def image_pos(self, pos: QPointF) -> tuple[float, float]:
        """Viewport position -> image coordinates (floats, sub-pixel)."""
        inverted, ok = self.viewportTransform().inverted()
        point = inverted.map(QPointF(pos)) if ok else QPointF(pos)
        return (float(point.x()), float(point.y()))

    def viewport_image_rect(self) -> Rect:
        """Visible image region ``(x0, y0, x1, y1)``, clipped to the image.

        This is the crop window handed to SAM (spec 4.6: viewport crop at native
        resolution), so it is rounded outwards to whole pixels.
        """
        h, w = self.image_hw()
        if not w or not h:
            return (0, 0, 0, 0)
        bounds = self.mapToScene(self.viewport().rect()).boundingRect()
        x0 = min(max(int(np.floor(bounds.left())), 0), w)
        y0 = min(max(int(np.floor(bounds.top())), 0), h)
        x1 = min(max(int(np.ceil(bounds.right())), 0), w)
        y1 = min(max(int(np.ceil(bounds.bottom())), 0), h)
        return (x0, y0, max(x0, x1), max(y0, y1))

    def _sync_minimap(self) -> None:
        self._minimap.set_view_rect(self.viewport_image_rect())
        self._announce_zoom()

    def _announce_zoom(self) -> None:
        """Tell whoever is showing the percentage, once per real change.

        Every path that zooms already syncs the minimap, which makes this the
        one place that sees all of them -- the wheel included, which is the one
        the status bar used to miss.
        """
        zoom = self.zoom_factor()
        if self._last_zoom is None or abs(zoom - self._last_zoom) > 1e-9:
            self._last_zoom = zoom
            self.sigZoomChanged.emit(zoom)

    def _place_minimap(self, margin: int = 8) -> None:
        self._minimap.move(
            max(0, self.viewport().width() - self._minimap.width() - margin),
            max(0, self.viewport().height() - self._minimap.height() - margin),
        )

    # -- Qt events ----------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().resizeEvent(event)
        self._place_minimap()
        self._sync_minimap()

    def wheelEvent(self, event) -> None:  # noqa: D102 - Qt override
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        self.zoom_at(event.position(), self.ZOOM_STEP ** (delta / 120.0))
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: D102 - Qt override
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: D102 - Qt override
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: D102
        pan = event.button() == Qt.MouseButton.MiddleButton or (
            self._space_down and event.button() == Qt.MouseButton.LeftButton
        )
        if pan:
            self._panning = True
            self._pan_origin = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        x, y = self.image_pos(event.position())
        self.sigMousePress.emit(x, y, event)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: D102
        if self._panning:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
            hbar.setValue(hbar.value() - int(round(delta.x())))
            vbar.setValue(vbar.value() - int(round(delta.y())))
            event.accept()
            return
        x, y = self.image_pos(event.position())
        self.sigMouseMove.emit(x, y, event)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: D102
        if self._panning:
            self._panning = False
            if self._space_down:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.viewport().unsetCursor()
            self._sync_minimap()
            event.accept()
            return
        x, y = self.image_pos(event.position())
        self.sigMouseRelease.emit(x, y, event)
        event.accept()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: D102
        super().drawForeground(painter, rect)
        if self._rubber_band is not None:
            x0, y0, x1, y1 = self._rubber_band
            pen = QPen(QColor(255, 232, 64), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        if self.zoom_factor() <= self.GRID_ZOOM:
            return
        h, w = self.image_hw()
        if not w or not h:
            return
        left = int(max(0, np.floor(rect.left())))
        right = int(min(w, np.ceil(rect.right())))
        top = int(max(0, np.floor(rect.top())))
        bottom = int(min(h, np.ceil(rect.bottom())))
        if (right - left) + (bottom - top) > 8000:  # sanity guard
            return
        pen = QPen(QColor(255, 255, 255, 40))
        pen.setCosmetic(True)
        painter.setPen(pen)
        lines = [QLineF(x, top, x, bottom) for x in range(left, right + 1)]
        lines += [QLineF(left, y, right, y) for y in range(top, bottom + 1)]
        painter.drawLines(lines)


def _main(argv: Optional[list[str]] = None) -> int:
    """Manual smoke check: ``python -m tda.ui.canvas.view --image <path>``."""
    import argparse
    import sys

    import cv2
    from PySide6.QtWidgets import QApplication

    from tda.ui.canvas.tools import BrushTool

    parser = argparse.ArgumentParser(description="TDA canvas smoke test")
    parser.add_argument("--image", required=True, help="path to an RGB image")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    canvas = ImageCanvas()
    canvas.set_image(rgb)
    overlay = LabelOverlay((h, w))
    dummy = np.zeros((h, w), dtype=bool)
    dummy[h // 4 : h // 2, w // 4 : w // 2] = True
    overlay.set_instances({"dummy#1": dummy}, ["dummy#1"])
    overlay.set_editing("dummy#2", np.zeros((h, w), dtype=bool))
    canvas.set_overlay(overlay)
    brush = BrushTool(canvas, overlay, radius=6)
    brush.attach()
    canvas.setWindowTitle(f"TDA canvas - {args.image} ({w}x{h}) - drag to paint")
    canvas.resize(1100, 850)
    canvas.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(_main())
