"""Interactive image canvas: zoom, pan, overlay, pixel grid, minimap (spec 4.5).

Two ``QGraphicsPixmapItem``s share one scene: the frame image at the bottom and
the :class:`~tda.ui.canvas.overlay.LabelOverlay` on top.  The scene rect is the
image rect, so **scene coordinates are image coordinates** -- that is what lets
the tool signals report plain image pixels and the SAM crops be taken straight
from :meth:`ImageCanvas.viewport_image_rect`.

Interaction: wheel zooms about the cursor by a factor of 1.25 per notch,
middle-drag pans, a pixel grid appears past 400% and a small
non-interactive minimap in the corner shows where the viewport sits.

Run ``python -m tda.ui.canvas.view --image <path>`` for a standalone smoke test.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
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

__all__ = ["CURSOR_MAX_PX", "ImageCanvas", "MiniMap", "OverlayItem", "ToolCursor",
           "circle_cursor"]

Rect = tuple[int, int, int, int]

#: Largest ring a circle cursor is drawn at, in screen pixels.  Above this the
#: platform's cursor would be scaled down (Windows) or refused, so the cursor
#: falls back to a crosshair and the status bar's ``r<n>`` is what says how big
#: the brush is.  A radius that big is a fill, not a stroke.
CURSOR_MAX_PX = 128
#: Below this the ring is smaller than the hole in the middle of it and the
#: annotator cannot aim; a crosshair is more honest.
CURSOR_MIN_PX = 7


class ToolCursor:
    """What cursor a tool wants: ``(kind, rgb, radius)`` with the drawing rules.

    ``kind`` is one of :data:`KINDS`; ``radius`` is in **image** pixels for
    ``circle`` (the canvas turns it into screen pixels with its own zoom, so the
    ring is the size of the stroke the annotator is about to make) and ignored
    otherwise.  ``dashed`` is how the eraser is told apart from the brush at a
    glance, without relying on colour alone.
    """

    KINDS = ("circle", "cross", "arrow", "forbidden")

    def __init__(self, kind: str, rgb: tuple[int, int, int] = (255, 232, 64),
                 radius: int = 0, dashed: bool = False) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown cursor kind {kind!r}; expected {self.KINDS}")
        self.kind = kind
        self.rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self.radius = max(0, int(radius))
        self.dashed = bool(dashed)

    def key(self, zoom: float, dpr: float) -> tuple:
        """Cache key: two specs with this key produce the same cursor."""
        return (self.kind, self.rgb, self.radius, self.dashed,
                round(float(zoom), 4), round(float(dpr), 3))

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, ToolCursor) and (
            (self.kind, self.rgb, self.radius, self.dashed)
            == (other.kind, other.rgb, other.radius, other.dashed)
        )

    def __repr__(self) -> str:  # noqa: D105 - for a failing assert
        return (f"ToolCursor({self.kind!r}, rgb={self.rgb}, radius={self.radius}, "
                f"dashed={self.dashed})")


def circle_cursor(diameter: int, rgb: tuple[int, int, int], dashed: bool,
                  dpr: float = 1.0) -> QCursor:
    """A ring ``diameter`` screen pixels across, with a dark halo and a centre dot.

    The halo matters: the chassis is dark metal and the scan bed is white paper,
    and a one-colour ring disappears into one of them.  The dot is the pixel the
    stamp is centred on, which a ring alone does not say at low zoom.
    """
    size = max(CURSOR_MIN_PX, int(diameter)) + 6
    dpr = max(1.0, float(dpr))
    pixmap = QPixmap(int(round(size * dpr)), int(round(size * dpr)))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    box = QRectF(3.0, 3.0, float(size - 6), float(size - 6))
    halo = QPen(QColor(0, 0, 0, 170), 3.0)
    painter.setPen(halo)
    painter.drawEllipse(box)
    pen = QPen(QColor(*rgb), 1.6)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.drawEllipse(box)
    centre = size / 2.0
    painter.setPen(QPen(QColor(0, 0, 0, 200), 3.0))
    painter.drawPoint(QPointF(centre, centre))
    painter.setPen(QPen(QColor(*rgb), 1.0))
    painter.drawPoint(QPointF(centre, centre))
    painter.end()
    return QCursor(pixmap, size // 2, size // 2)


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
        self._ensure = None
        # Without this flag Qt reports the whole bounding rect as exposed.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    def set_image(self, image: Optional[QImage]) -> None:
        """Point the item at a new image (or clear it) and repaint fully."""
        self.prepareGeometryChange()
        self._image = image
        self.update()

    def set_ensure(self, ensure) -> None:
        """Install the callback that composites a region before it is drawn.

        The overlay only composites what somebody is about to look at, so the
        buffer this item draws from is not promised to be right everywhere --
        it is promised to be right where it has been asked for.  This item is
        the last thing that knows, exactly, which pixels are about to reach
        the screen, so it is the one that asks.
        """
        self._ensure = ensure

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
        if self._ensure is not None:
            self._ensure((
                int(np.floor(box.left())), int(np.floor(box.top())),
                int(np.ceil(box.right())), int(np.ceil(box.bottom())),
            ))
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
    #: Fraction of the viewport composited beyond it, so that a pan does not
    #: ask the overlay for a new sliver on every mouse move.
    OVERLAY_MARGIN = 0.25

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
        self._rubber_band: Optional[tuple[float, float, float, float]] = None
        #: The chassis-range rectangle, and whether it is being edited.  Drawn
        #: in :meth:`drawForeground` like the rubber band rather than into the
        #: overlay image: it changes on every mouse move of a drag and the
        #: overlay is a 12 MP composite (B7's budgets).
        self._roi: Optional[tuple[float, float, float, float]] = None
        self._roi_editing = False
        #: Whether a *stored* rectangle's thin outline is drawn; the overlays
        #: key ``A`` owns it, like every other layer.
        self.roi_outline_visible = True
        #: A pixel inside the armed box prompt -- "click here" (``Shift+C``).
        #: Painted in :meth:`drawForeground` next to the rubber band rather
        #: than added to the scene, so there is nothing under the cursor that
        #: could swallow a press or shift the coordinate a tool is handed.
        self._prompt_point: Optional[tuple[int, int]] = None
        self._last_zoom: Optional[float] = None
        #: What the armed tool looks like under the mouse; ``None`` means the
        #: platform's own arrow.  The one thing on screen that says *what a
        #: press will do*, which is why it is the canvas' business and not the
        #: window's: the ring has to follow the zoom, and only the canvas knows
        #: the zoom has changed (U1 report 1, ruling R1).
        self._tool_cursor: Optional[ToolCursor] = None
        self._cursor_cache: dict[tuple, QCursor] = {}
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
        self._overlay_item.set_ensure(self._ensure_overlay)
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
        what gets invalidated.  ``None`` there means nothing was stale *inside
        the part being composited*, so no repaint is scheduled at all.

        What is composited is the viewport plus :data:`OVERLAY_MARGIN`, not the
        frame: an OAK frame is 12 MP and a 950x770 viewport at 59 % zoom holds
        a sixth of it, so the other five sixths would be 90 ms of every
        gesture spent on pixels nobody can see.  They are not skipped, only
        deferred -- :class:`OverlayItem` composites whatever it is about to
        draw, so panning to them pays for them then, and in a strip.
        """
        if self._overlay is None:
            self._overlay_item.set_image(None)
            return
        image = self._overlay.qimage(
            rect, alpha=self.overlay_alpha, outline=self.overlay_outline,
            clip=self.overlay_clip(),
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

    def overlay_clip(self) -> Rect:
        """The region of the frame worth compositing now: the viewport + margin.

        The margin is there so that a slow drag does not ask for a new sliver
        on every mouse move; it is a fraction of the viewport, so it is small
        when zoomed in (where a sliver is cheap) and large when zoomed out
        (where the whole frame is on screen anyway).
        """
        x0, y0, x1, y1 = self.viewport_image_rect()
        mx = int(round((x1 - x0) * self.OVERLAY_MARGIN))
        my = int(round((y1 - y0) * self.OVERLAY_MARGIN))
        return (x0 - mx, y0 - my, x1 + mx, y1 + my)

    def _ensure_overlay(self, box: Rect) -> None:
        """Composite ``box`` before the item draws it (see :meth:`refresh`)."""
        if self._overlay is not None:
            self._overlay.qimage(
                alpha=self.overlay_alpha, outline=self.overlay_outline, clip=box
            )

    def set_rubber_band(
        self, box: Optional[tuple[float, float, float, float]]
    ) -> None:
        """Show (or clear) a dashed box, e.g. while dragging a SAM box prompt."""
        self._rubber_band = box
        self.viewport().update()

    # -- the chassis range --------------------------------------------------
    #: Half-width, in screen pixels, of a resize handle's square.
    HANDLE_PX = 5
    #: The eight handles, as ``(x fraction, y fraction)`` of the rectangle.
    HANDLES: tuple[tuple[str, float, float], ...] = (
        ("nw", 0.0, 0.0), ("n", 0.5, 0.0), ("ne", 1.0, 0.0),
        ("w", 0.0, 0.5), ("e", 1.0, 0.5),
        ("sw", 0.0, 1.0), ("s", 0.5, 1.0), ("se", 1.0, 1.0),
    )
    #: How dark the frame outside the rectangle goes while it is being edited.
    DIM_ALPHA = 90

    def set_roi(self, box: Optional[tuple], editing: bool = False) -> None:
        """Show the chassis-range rectangle (``None`` clears it).

        ``editing`` draws the thick dashed outline with its eight handles and
        dims everything outside it; otherwise a stored rectangle gets a thin,
        subtle outline that says where the difference map and the SAM prompt
        boxes are being computed without competing with the masks.
        """
        value = None if box is None else tuple(float(v) for v in box)
        if value == self._roi and bool(editing) == self._roi_editing:
            return
        self._roi, self._roi_editing = value, bool(editing)
        self.viewport().update()

    def roi_rect(self) -> Optional[tuple]:
        """The rectangle being drawn, or ``None``."""
        return self._roi

    def roi_handle_points(self) -> dict[str, tuple[float, float]]:
        """Where the eight handles sit, in image coordinates."""
        if self._roi is None:
            return {}
        x0, y0, x1, y1 = self._roi
        return {name: (x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)
                for name, fx, fy in self.HANDLES}

    def _draw_roi(self, painter: QPainter) -> None:
        """The rectangle, its handles and the dimmed surround (ruling U-ROI-2)."""
        if self._roi is None:
            return
        if not self._roi_editing and not self.roi_outline_visible:
            return
        x0, y0, x1, y1 = self._roi
        box = QRectF(x0, y0, x1 - x0, y1 - y0)
        if not self._roi_editing:
            pen = QPen(QColor(255, 232, 64, 150), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)
            return
        # Everything outside the rectangle goes dark, so "inside" is a thing
        # you can see rather than a thing you have to trace with your eye.
        h, w = self.image_hw()
        if w and h:
            outside = QPainterPath()
            outside.addRect(QRectF(0, 0, float(w), float(h)))
            inside = QPainterPath()
            inside.addRect(box)
            painter.fillPath(outside.subtracted(inside),
                             QColor(0, 0, 0, self.DIM_ALPHA))
        # Two strokes: a dark one under a bright dashed one, so the outline is
        # visible both on the dark chassis and on the white scan bed.
        under = QPen(QColor(0, 0, 0, 220), 5)
        under.setCosmetic(True)
        painter.setPen(under)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box)
        over = QPen(QColor(255, 232, 64), 3, Qt.PenStyle.DashLine)
        over.setCosmetic(True)
        painter.setPen(over)
        painter.drawRect(box)
        # Handles, sized in screen pixels so they stay grabbable at any zoom.
        half = self.HANDLE_PX / max(self.zoom_factor(), 1e-6)
        edge = QPen(QColor(0, 0, 0, 230), 1)
        edge.setCosmetic(True)
        painter.setPen(edge)
        painter.setBrush(QColor(255, 255, 255))
        for _name, (hx, hy) in self.roi_handle_points().items():
            painter.drawRect(QRectF(hx - half, hy - half, 2 * half, 2 * half))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def set_prompt_point(self, point: Optional[tuple]) -> None:
        """Mark one pixel inside the armed box prompt, or clear it with ``None``.

        The distance-transform point of a ``Shift+C`` alternate: the box tells
        SAM where to look and this tells the annotator where their own click
        will be unambiguous.  Setting the value it already holds repaints
        nothing, which is what keeps the frame-change path free of a repaint it
        never used to make.
        """
        value = (None if point is None
                 else (int(round(float(point[0]))), int(round(float(point[1])))))
        if value == self._prompt_point:
            return
        self._prompt_point = value
        self.viewport().update()

    def prompt_point(self) -> Optional[tuple[int, int]]:
        """The marked pixel, or ``None``."""
        return self._prompt_point

    # -- the armed tool, under the mouse ------------------------------------
    def set_tool_cursor(self, spec: Optional[ToolCursor]) -> None:
        """Say what the armed tool is; ``None`` restores the arrow.

        The annotator's first trial ended with four box prompts where they
        believed they were brushing.  Nothing on screen said which tool was
        armed: the status bar read ``sam_box`` in 9 pt English in the corner and
        the cursor was the same arrow for every tool.
        """
        self._tool_cursor = spec
        self._apply_tool_cursor()

    def tool_cursor(self) -> Optional[ToolCursor]:
        """The spec the canvas was last given (for the window and the tests)."""
        return self._tool_cursor

    def cursor_diameter(self) -> int:
        """On-screen diameter of the ring, ``0`` when no circle is shown.

        A brush of radius ``r`` image pixels stamps ``2r + 1`` of them, so the
        ring is that many screen pixels across at the current zoom -- which is
        what "the brush's true on-screen radius" means and what makes it follow
        both ``[``/``]`` and the wheel.
        """
        spec = self._tool_cursor
        if spec is None or spec.kind != "circle":
            return 0
        return int(round((2 * spec.radius + 1) * self.zoom_factor()))

    def _apply_tool_cursor(self) -> None:
        """Put the spec on the viewport, building (and caching) the pixmap."""
        spec = self._tool_cursor
        if spec is None:
            self.viewport().unsetCursor()
            return
        if spec.kind == "circle":
            diameter = self.cursor_diameter()
            if not CURSOR_MIN_PX <= diameter <= CURSOR_MAX_PX:
                # Too small to aim with, or bigger than the platform will draw.
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
                return
            dpr = float(self.devicePixelRatioF() or 1.0)
            key = spec.key(self.zoom_factor(), dpr)
            cursor = self._cursor_cache.get(key)
            if cursor is None:
                if len(self._cursor_cache) > 64:
                    self._cursor_cache.clear()   # a long wheel spin, not a leak
                cursor = circle_cursor(diameter, spec.rgb, spec.dashed, dpr)
                self._cursor_cache[key] = cursor
            self.viewport().setCursor(cursor)
            return
        self.viewport().setCursor({
            "cross": Qt.CursorShape.CrossCursor,
            "arrow": Qt.CursorShape.ArrowCursor,
            "forbidden": Qt.CursorShape.ForbiddenCursor,
        }[spec.kind])

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
            # The brush ring is the size of the stroke, so it follows the wheel
            # as well as ``[``/``]``; this is the one place every zoom passes.
            self._apply_tool_cursor()
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

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: D102
        # Middle-drag only.  Space-drag used to be the other way to pan and
        # was unreachable: Space is "confirm the frame" in the window, which
        # swallows it long before the canvas sees it.
        if event.button() == Qt.MouseButton.MiddleButton:
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
            # Not ``unsetCursor``: that would drop the armed tool's ring and
            # leave an arrow over the canvas until the next tool switch.
            self._apply_tool_cursor()
            self._sync_minimap()
            event.accept()
            return
        x, y = self.image_pos(event.position())
        self.sigMouseRelease.emit(x, y, event)
        event.accept()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: D102
        super().drawForeground(painter, rect)
        self._draw_roi(painter)
        if self._rubber_band is not None:
            x0, y0, x1, y1 = self._rubber_band
            pen = QPen(QColor(255, 232, 64), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        if self._prompt_point is not None:
            # A cross with a hole in the middle, sized in *screen* pixels, so
            # the marked pixel itself stays visible at 800 % and the mark stays
            # findable at 20 %.
            cx, cy = self._prompt_point[0] + 0.5, self._prompt_point[1] + 0.5
            arm = 9.0 / max(self.zoom_factor(), 1e-6)
            gap = arm / 3.0
            pen = QPen(QColor(255, 232, 64), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLines([
                QLineF(cx - arm, cy, cx - gap, cy),
                QLineF(cx + gap, cy, cx + arm, cy),
                QLineF(cx, cy - arm, cx, cy - gap),
                QLineF(cx, cy + gap, cx, cy + arm),
            ])
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
