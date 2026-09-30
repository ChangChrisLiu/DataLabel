"""Canvas edit tools: the pixel tools (brush, eraser, occluder brush), spec 4.6.

A tool is a stateless-ish adapter between the canvas mouse signals and the
overlay's edit layers.  All coordinates arriving at ``on_press``/``on_move``/
``on_release`` are **image** coordinates (floats) produced by
:class:`~tda.ui.canvas.view.ImageCanvas`; nothing here knows about the viewport
transform.

The pixel tools never trigger inference (spec 4.3) and never touch the committed
label map -- they write to the *editing* layer only, so an undoable op is always
a single mask replacement.

The two **filled-shape** tools (task U5a) are pixel tools too: ``P``
(:class:`PolygonTool`) fills the region between the vertices the annotator
clicks, ``Y`` (:class:`CircleTool`) the disk they drag out.  Each fill is added
to the editing layer the way one brush stroke is -- ``stroke_before`` plus
``sigStroke`` -- so the window cannot tell them apart from a stroke, and does
not have to.

The asynchronous SAM prompt tools live in :mod:`tda.ui.canvas.sam_tools`, which
imports :class:`Tool` from here.  They stay importable from this module (see
:func:`__getattr__`) so that ``from tda.ui.canvas.tools import SamPointTool``
keeps working; the indirection is lazy because a plain import in this direction
would close an import cycle.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, Signal

from tda.core import masks as _masks
from tda.ui.canvas.overlay import OCCLUDER_TYPES, LabelOverlay

__all__ = [
    "OCCLUDER_TYPES",
    "MAX_SAM_SIDE",
    "Tool",
    "PaintTool",
    "BrushTool",
    "EraserTool",
    "OccluderTool",
    "ShapeFillTool",
    "PolygonTool",
    "CircleTool",
    "disk_patch",
    "polygon_patch",
    "SamResultBridge",
    "SamToolBase",
    "SamPointTool",
    "SamBoxTool",
    "viewport_crop",
]

Rect = tuple[int, int, int, int]
Point = tuple[float, float, int]
Box = tuple[float, float, float, float]

#: Names re-exported from :mod:`tda.ui.canvas.sam_tools` for backward
#: compatibility; resolved on first access by :func:`__getattr__`.
_SAM_NAMES = frozenset(
    {
        "MAX_SAM_SIDE",
        "SamResultBridge",
        "SamToolBase",
        "SamPointTool",
        "SamBoxTool",
        "viewport_crop",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve the SAM tool names against :mod:`tda.ui.canvas.sam_tools` (PEP 562).

    Importing ``sam_tools`` at module level would be circular -- it needs
    :class:`Tool` from here -- so the lookup happens on first access instead.
    """
    if name in _SAM_NAMES:
        from tda.ui.canvas import sam_tools

        return getattr(sam_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _union(a: Optional[Rect], b: Optional[Rect]) -> Optional[Rect]:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------
class Tool(QObject):
    """Base tool: receives image-space mouse callbacks, does nothing.

    ``sigStroke`` carries the dirty rect ``(x0, y0, x1, y1)`` of a finished
    edit, which the session layer turns into an undoable op.
    """

    sigStroke = Signal(object)
    #: Something went wrong inside the tool; the payload is for the status bar.
    #: The three mouse slots are Qt slots, so an exception raised in one of them
    #: goes straight into the event loop: ``queue.submit`` raising left the SAM
    #: label saying "ready" while every click repeated the failure in silence.
    sigError = Signal(str)

    def __init__(
        self,
        canvas: Any = None,
        overlay: Optional[LabelOverlay] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.canvas = canvas
        self.overlay = overlay
        self._attached = False

    def attach(self) -> None:
        """Connect to the canvas mouse signals (idempotent)."""
        if self.canvas is None or self._attached:
            return
        self.canvas.sigMousePress.connect(self._press)
        self.canvas.sigMouseMove.connect(self._move)
        self.canvas.sigMouseRelease.connect(self._release)
        self._attached = True

    def detach(self) -> None:
        """Disconnect from the canvas mouse signals (idempotent)."""
        if self.canvas is None or not self._attached:
            return
        self.canvas.sigMousePress.disconnect(self._press)
        self.canvas.sigMouseMove.disconnect(self._move)
        self.canvas.sigMouseRelease.disconnect(self._release)
        self._attached = False

    # -- the guarded slots --------------------------------------------------
    def _press(self, x: float, y: float, ev: Any) -> None:
        self._guarded(self.on_press, x, y, ev)

    def _move(self, x: float, y: float, ev: Any) -> None:
        self._guarded(self.on_move, x, y, ev)

    def _release(self, x: float, y: float, ev: Any) -> None:
        self._guarded(self.on_release, x, y, ev)

    def _guarded(self, handler, x: float, y: float, ev: Any) -> None:
        """Run one mouse handler; a failure becomes a status line, not a crash."""
        try:
            handler(x, y, ev)
        except Exception as exc:  # noqa: BLE001 - this is a Qt slot
            self.sigError.emit(f"{type(exc).__name__}: {exc}")

    def on_press(self, x: float, y: float, ev: Any) -> None:
        """Mouse down at image coords ``(x, y)``."""

    def on_move(self, x: float, y: float, ev: Any) -> None:
        """Mouse moved to image coords ``(x, y)``."""

    def on_release(self, x: float, y: float, ev: Any) -> None:
        """Mouse up at image coords ``(x, y)``."""


# ---------------------------------------------------------------------------
# pixel tools
# ---------------------------------------------------------------------------
class PaintTool(Tool):
    """Circular stamp along the cursor path, into one boolean overlay layer.

    Consecutive samples are joined by interpolated stamps: at 800% a fast drag
    delivers mouse moves several image pixels apart, and without the
    interpolation the stroke would come out as a dotted line.

    :attr:`stroke_before` holds the layer as it was when the stroke started, so
    the session can build an ``edit_editing_mask`` op on ``sigStroke`` without
    the tool depending on the undo stack.
    """

    #: True adds pixels, False removes them.
    ADD = True

    def __init__(
        self,
        canvas: Any = None,
        overlay: Optional[LabelOverlay] = None,
        radius: int = 8,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(canvas, overlay, parent)
        self.radius = max(0, int(radius))
        self.stroke_before: Optional[np.ndarray] = None
        self._active = False
        self._dirty: Optional[Rect] = None
        self._last: Optional[tuple[float, float]] = None

    def set_radius(self, radius: int) -> None:
        """Set the brush radius in image pixels (``[`` / ``]`` in the UI)."""
        self.radius = max(0, int(radius))

    # -- layer binding (overridden by OccluderTool) -------------------------
    def _target(self) -> np.ndarray:
        """The layer this tool writes, for the before-stroke snapshot."""
        assert self.overlay is not None
        return self.overlay.editing

    def _paint(self, x: float, y: float) -> Optional[Rect]:
        """Stamp one disc; ``None`` when it falls outside the image."""
        assert self.overlay is not None
        return self.overlay.paint((x, y), self.radius, self.ADD)

    # -- events -------------------------------------------------------------
    @staticmethod
    def _is_right(ev: Any) -> bool:
        """Was this the right button? (Qt6 events answer, stubs may not.)"""
        from PySide6.QtCore import Qt as _Qt

        button = getattr(ev, "button", None)
        try:
            return button is not None and button() == _Qt.MouseButton.RightButton
        except TypeError:  # pragma: no cover - a stub without a callable button
            return False

    def on_press(self, x: float, y: float, ev: Any) -> None:
        if self.overlay is None:
            return
        if self._is_right(ev):
            # The right button belongs to the SAM tools (a negative point) and
            # to the context menu; painting with it was never asked for and a
            # right-click meant as "undo that point" would add pixels instead.
            return
        self.stroke_before = self._target().copy()
        self._active = True
        self._dirty = None
        self._last = None
        self._stamp(x, y)

    def on_move(self, x: float, y: float, ev: Any) -> None:
        if self._active:
            self._stamp(x, y)

    def on_release(self, x: float, y: float, ev: Any) -> None:
        if not self._active:
            return
        self._stamp(x, y)
        self._active = False
        self._last = None
        dirty, self._dirty = self._dirty, None
        if dirty is not None:
            self.sigStroke.emit(dirty)

    # -- painting -----------------------------------------------------------
    def _stamp(self, x: float, y: float) -> None:
        if self.overlay is None:
            return
        segment: Optional[Rect] = None
        for px, py in self._path(x, y):
            segment = _union(segment, self._paint(px, py))
        self._last = (float(x), float(y))
        self._dirty = _union(self._dirty, segment)
        # A stroke entirely outside the image dirties nothing, and asking the
        # canvas to refresh an empty rect would cost a full-item repaint.
        if segment is not None and self.canvas is not None:
            self.canvas.refresh(segment)

    def _path(self, x: float, y: float) -> list[tuple[float, float]]:
        """Stamp centres from the previous sample up to ``(x, y)``."""
        if self._last is None:
            return [(float(x), float(y))]
        lx, ly = self._last
        dx, dy = float(x) - lx, float(y) - ly
        step = max(1.0, self.radius / 2.0)
        steps = min(4096, int(math.hypot(dx, dy) / step) + 1)
        return [(lx + dx * i / steps, ly + dy * i / steps) for i in range(1, steps + 1)]


class BrushTool(PaintTool):
    """Adds pixels to the instance being edited."""

    ADD = True


class EraserTool(PaintTool):
    """Removes pixels from the instance being edited."""

    ADD = False


class OccluderTool(PaintTool):
    """Paints the frame occluder layer of its current type (spec 4.6, ``O``).

    Switching :attr:`occluder_type` switches the layer written, so strokes of
    two types never merge -- each ``occluder_type`` is stored and subtracted
    separately by the compiler.
    """

    ADD = True

    def __init__(
        self,
        canvas: Any = None,
        overlay: Optional[LabelOverlay] = None,
        radius: int = 8,
        occluder_type: str = "hand",
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(canvas, overlay, radius, parent)
        self.occluder_type = occluder_type

    @property
    def occluder_type(self) -> str:
        return self._occluder_type

    @occluder_type.setter
    def occluder_type(self, value: str) -> None:
        if value not in OCCLUDER_TYPES:
            raise ValueError(
                f"unknown occluder_type {value!r}; expected one of {OCCLUDER_TYPES}"
            )
        self._occluder_type = value

    def _target(self) -> np.ndarray:
        assert self.overlay is not None
        return self.overlay.occluder_layer(self._occluder_type)

    def _paint(self, x: float, y: float) -> Optional[Rect]:
        assert self.overlay is not None
        return self.overlay.paint_occluder(
            (x, y), self.radius, self.ADD, self._occluder_type
        )


# ---------------------------------------------------------------------------
# filled shapes (task U5a)
# ---------------------------------------------------------------------------
def polygon_patch(points: Sequence[tuple[float, float]],
                  hw: tuple[int, int]) -> Optional[tuple[Rect, np.ndarray]]:
    """The polygon through ``points`` filled, as ``(box, patch)``; ``None`` off the image.

    ``points`` are the canvas' image coordinates, in which pixel ``(x, y)``
    covers ``[x, x+1) x [y, y+1)``; :func:`tda.core.masks.polygons_to_mask`
    puts vertex ``(i, j)`` on the *centre* of pixel ``(i, j)``, so they are
    moved half a pixel onto that grid first -- the pixel a vertex was clicked
    in is the pixel the fill reaches, as the preview shows it.

    Rasterised by ``polygons_to_mask`` -- the one polygon fill the package
    has -- inside the polygon's own bounding box rather than over the whole
    frame: at 12 MP a frame-sized canvas is 12 MB allocated and converted for
    a shape that may be forty pixels across.  The vertices are rounded here,
    exactly as ``polygons_to_mask`` rounds them, before they are moved into
    the box, so the patch is the full-frame fill cropped (numpy rounds halves
    to even, and moving a half by an odd offset would change which way it
    went).
    """
    pts = np.round(np.asarray(points, dtype=np.float64).reshape(-1, 2) - 0.5)
    if len(pts) < 3:
        return None
    h, w = int(hw[0]), int(hw[1])
    x0 = max(0, int(pts[:, 0].min()))
    y0 = max(0, int(pts[:, 1].min()))
    x1 = min(w, int(pts[:, 0].max()) + 1)
    y1 = min(h, int(pts[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    local = (pts - np.array([x0, y0], dtype=np.float64)).reshape(-1)
    patch = _masks.polygons_to_mask([local], (y1 - y0, x1 - x0))
    return (x0, y0, x1, y1), patch


def disk_patch(cx: float, cy: float, radius: float,
               hw: tuple[int, int]) -> Optional[tuple[Rect, np.ndarray]]:
    """Every pixel whose centre lies within ``radius`` of ``(cx, cy)``, as ``(box, patch)``.

    Pixel ``(x, y)`` covers ``[x, x+1) x [y, y+1)`` of the image coordinates the
    canvas reports, so its centre is ``(x + 0.5, y + 0.5)``: the disk is the
    one the preview circle is drawn as, not half a pixel off it.  ``None`` when
    it misses the image.
    """
    h, w = int(hw[0]), int(hw[1])
    r = max(0.0, float(radius))
    x0 = max(0, int(math.floor(cx - r - 0.5)))
    y0 = max(0, int(math.floor(cy - r - 0.5)))
    x1 = min(w, int(math.ceil(cx + r + 0.5)))
    y1 = min(h, int(math.ceil(cy + r + 0.5)))
    if x1 <= x0 or y1 <= y0:
        return None
    dx = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5 - float(cx)
    dy = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5 - float(cy)
    return (x0, y0, x1, y1), (dx * dx + dy * dy) <= r * r


def _left_button(ev: Any) -> bool:
    """A left-button event -- or one that cannot say (a test's ``None``, a stub)."""
    from PySide6.QtCore import Qt as _Qt

    button = getattr(ev, "button", None)
    if button is None:
        return True
    try:
        return button() == _Qt.MouseButton.LeftButton
    except TypeError:  # pragma: no cover - a stub without a callable button
        return True


class ShapeFillTool(Tool):
    """A shape outlined on the canvas and filled into the editing layer (task U5a).

    The fill is **added** to the layer the way a brush stroke adds its disks:
    :attr:`stroke_before` holds the layer as it was and :attr:`sigStroke`
    carries the dirty rect, so the window makes one ``edit_editing_mask`` op of
    it -- one undo step, the crash sidecar queued, the erased set lifted where
    the fill covers it (ruling E1: a pixel back in the layer is not erased).
    It never removes a pixel, so what is already labelled stays labelled and
    the overlap is simply part of the union.

    :attr:`sigShape` says what happened to the shape in progress -- ``"start"``,
    ``"vertex"``, ``"remove"``, ``"radius"``, ``"cancel"``, ``"fill"``, and the
    three that fill nothing: ``"too_few"`` (a polygon of under three vertices),
    ``"outside"`` (a circle released off the viewport) and ``"empty"`` (a
    shape that misses the image) -- for the
    status line, the badge and the guide.  :attr:`accepts` is asked on every
    press; the window answers ``False`` when it has turned the press down (no
    part is being edited), so no shape starts that could never be filled.
    """

    sigShape = Signal(str)
    #: The same flag the brush family carries: this tool adds pixels.
    ADD = True

    def __init__(
        self,
        canvas: Any = None,
        overlay: Optional[LabelOverlay] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(canvas, overlay, parent)
        self.stroke_before: Optional[np.ndarray] = None
        self.accepts: Optional[Callable[[], bool]] = None

    @property
    def busy(self) -> bool:
        """Is a shape in progress (a vertex placed, a drag under way)?"""
        return False

    def cancel(self) -> bool:
        """Drop the shape in progress, filling nothing; ``False`` when there was none."""
        return False

    def _accepted(self) -> bool:
        return self.accepts is None or bool(self.accepts())

    def _zoom(self) -> float:
        """Screen pixels per image pixel, 1.0 for a canvas that cannot say."""
        zoom = getattr(self.canvas, "zoom_factor", None)
        try:
            value = float(zoom()) if callable(zoom) else 1.0
        except Exception:  # noqa: BLE001 - a stub canvas
            value = 1.0
        return value if value > 0.0 else 1.0

    def _preview(self, shape: Any) -> None:
        """Hand the canvas what to draw over the frame (``None`` takes it away)."""
        setter = getattr(self.canvas, "set_shape_preview", None)
        if setter is not None:
            setter(shape)

    def _fill(self, box: Rect, patch: np.ndarray) -> Optional[Rect]:
        """Add ``patch`` at ``box`` to the editing layer: one stroke's worth."""
        if self.overlay is None:
            return None
        before = self.overlay.editing.copy()
        rect = self.overlay.add_editing(patch, box)
        if rect is None:
            return None
        self.stroke_before = before
        if self.canvas is not None:
            self.canvas.refresh(rect)
        self.sigStroke.emit(rect)
        return rect


class PolygonTool(ShapeFillTool):
    """``P``: click the corners of a region, close it, and it is filled (task U5a).

    Asked for after SAM segmented the motherboard with gaps all over it: the
    brush filled them one dab at a time.  A left click adds a vertex; ``Enter``
    (the window's), a double-click, or -- once there are :attr:`MIN_VERTICES`
    -- a click within :attr:`CLOSE_PX` screen pixels of the first vertex closes
    the polygon and fills its interior into the editing layer.  Fewer than
    :attr:`MIN_VERTICES` vertices fill nothing.
    :meth:`remove_last` is ``Backspace`` and :meth:`cancel` is ``Esc``.

    The vertices are **image** coordinates, so zooming and panning in the
    middle of a polygon keep it where it was drawn.  :meth:`detach` does not
    drop it -- holding ``Tab`` to compare detaches every tool for a moment --
    so the window cancels it itself when the tool, the frame, or the part
    being edited really changes.
    """

    MIN_VERTICES = 3
    #: How close to the first vertex, in screen pixels, a click closes on it.
    CLOSE_PX = 8.0

    def __init__(
        self,
        canvas: Any = None,
        overlay: Optional[LabelOverlay] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(canvas, overlay, parent)
        self.vertices: list[tuple[float, float]] = []
        #: The pointer is close enough to the first vertex for a click to close.
        self.hot = False
        self._double_attached = False

    @property
    def busy(self) -> bool:
        return bool(self.vertices)

    # -- the double-click, which the base tools never listen to -------------
    def attach(self) -> None:
        super().attach()
        signal = getattr(self.canvas, "sigMouseDoubleClick", None)
        if self._attached and signal is not None and not self._double_attached:
            signal.connect(self._double)
            self._double_attached = True

    def detach(self) -> None:
        super().detach()
        if self._double_attached:
            self.canvas.sigMouseDoubleClick.disconnect(self._double)
            self._double_attached = False

    def _double(self, x: float, y: float, ev: Any) -> None:
        self._guarded(self.on_double_click, x, y, ev)

    # -- events -------------------------------------------------------------
    def on_press(self, x: float, y: float, ev: Any) -> None:
        if self.overlay is None or not _left_button(ev) or not self._accepted():
            return
        # Only a polygon that could close closes on its first vertex -- the
        # same rule that lights it (``on_move``).  Below three vertices a click
        # there is a vertex: at the fit zoom of a 12 MP frame the first
        # vertex's 8 screen px are ~44 image px, the whole of a small part.
        if len(self.vertices) >= self.MIN_VERTICES and self._near_first(x, y):
            self.close()
            return
        self.vertices.append((float(x), float(y)))
        self.hot = False
        self._show()
        self.sigShape.emit("start" if len(self.vertices) == 1 else "vertex")

    def on_move(self, x: float, y: float, ev: Any) -> None:
        if not self.vertices:
            return
        # Only a change of "would a click here close it?" repaints: moving the
        # pointer about costs nothing otherwise.
        hot = len(self.vertices) >= self.MIN_VERTICES and self._near_first(x, y)
        if hot != self.hot:
            self.hot = hot
            self._show()

    def on_double_click(self, x: float, y: float, ev: Any) -> None:
        """Close the polygon.  The double-click's first press already added its vertex."""
        if self.vertices and _left_button(ev):
            self.close()

    # -- what the window asks -----------------------------------------------
    def close(self) -> str:
        """Fill the polygon: ``"fill"``, ``"empty"``, ``"too_few"``, or ``""`` for none.

        ``"too_few"`` keeps the vertices: the annotator may still add one.
        """
        if not self.vertices:
            return ""
        if len(self.vertices) < self.MIN_VERTICES:
            self.sigShape.emit("too_few")
            return "too_few"
        points, self.vertices = list(self.vertices), []
        self.hot = False
        self._show()
        found = None if self.overlay is None else polygon_patch(points, self.overlay.hw)
        rect = None if found is None else self._fill(*found)
        outcome = "fill" if rect is not None else "empty"
        self.sigShape.emit(outcome)
        return outcome

    def remove_last(self) -> bool:
        """``Backspace``: take the last vertex back; ``False`` when there was none."""
        if not self.vertices:
            return False
        self.vertices.pop()
        self.hot = False
        self._show()
        self.sigShape.emit("remove")
        return True

    def cancel(self) -> bool:
        if not self.vertices:
            return False
        self.vertices = []
        self.hot = False
        self._show()
        self.sigShape.emit("cancel")
        return True

    # -- helpers ------------------------------------------------------------
    def _near_first(self, x: float, y: float) -> bool:
        fx, fy = self.vertices[0]
        return math.hypot(float(x) - fx, float(y) - fy) * self._zoom() <= self.CLOSE_PX

    def _show(self) -> None:
        self._preview(("polygon", tuple(self.vertices), bool(self.hot))
                      if self.vertices else None)


class CircleTool(ShapeFillTool):
    """``Y``: press at the centre, drag to the rim, release -- the disk is filled (U5a).

    Asked for next to the polygon, for the screws.  **What a release fills is
    what the preview and the badge showed**, decided in image pixels -- so the
    wheel may zoom in the middle of a drag, and the zoom never changes the
    answer: the drag's radius when it is at least :attr:`MIN_RADIUS`, and
    below that the brush radius (:attr:`click_radius`, the one ``[``/``]`` and
    the slider set) -- a plain click fills a disk of the brush size, so a screw
    can be one click at a matching brush.  (Deciding "click" by screen pixels
    instead turned a 2-px drag at a 25 % zoom -- an 8 px circle on the preview
    -- into a brush-sized disk.)  A release outside the viewport cancels the
    circle: a drag that ran off the canvas is not a disk that size (on the
    scanner it was 1.1 Mpx).  ``Esc`` (the window's :meth:`cancel`) drops a
    drag in progress.
    """

    #: Drags shorter than this, in image pixels, are a click: the brush radius.
    MIN_RADIUS = 1.0

    def __init__(
        self,
        canvas: Any = None,
        overlay: Optional[LabelOverlay] = None,
        radius: int = 8,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(canvas, overlay, parent)
        self.click_radius = max(0, int(radius))
        self.centre: Optional[tuple[float, float]] = None
        self.drag_radius = 0.0
        #: The radius of the last disk filled, image pixels (for the status line).
        self.last_radius = 0.0

    @property
    def busy(self) -> bool:
        return self.centre is not None

    @property
    def radius(self) -> int:
        """What the status badge's ``r=`` shows: the disk a release would fill
        while a drag is under way, the click's otherwise (image pixels, like the
        brush's)."""
        if self.centre is not None:
            return int(round(self.fill_radius()))
        return self.click_radius

    def fill_radius(self, drag: Optional[float] = None) -> float:
        """The radius a release at drag radius ``drag`` fills (default: the current one)."""
        radius = self.drag_radius if drag is None else float(drag)
        return radius if radius >= self.MIN_RADIUS else float(self.click_radius)

    def set_radius(self, radius: int) -> None:
        """The brush radius, which a plain click fills with (``[`` / ``]``)."""
        self.click_radius = max(0, int(radius))

    def detach(self) -> None:
        # A drag is one press and one release; a tool that stops listening in
        # between would never see the release, so the drag ends here.
        super().detach()
        self.cancel()

    # -- events -------------------------------------------------------------
    def on_press(self, x: float, y: float, ev: Any) -> None:
        if self.overlay is None or not _left_button(ev) or not self._accepted():
            return
        self.centre = (float(x), float(y))
        self.drag_radius = 0.0
        self._show()
        self.sigShape.emit("start")

    def on_move(self, x: float, y: float, ev: Any) -> None:
        if self.centre is None:
            return
        radius = self._radius_to(x, y)
        if radius == self.drag_radius:
            return
        self.drag_radius = radius
        self._show()
        self.sigShape.emit("radius")

    def on_release(self, x: float, y: float, ev: Any) -> None:
        if self.centre is None or not _left_button(ev):
            return
        cx, cy = self.centre
        radius = self.fill_radius(self._radius_to(x, y))
        self.centre = None
        self.drag_radius = 0.0
        self._preview(None)
        if not self._in_view(x, y):
            self.sigShape.emit("outside")
            return
        if radius < self.MIN_RADIUS:           # a brush radius of 0
            self.sigShape.emit("empty")
            return
        self.last_radius = radius
        found = None if self.overlay is None else disk_patch(cx, cy, radius, self.overlay.hw)
        rect = None if found is None else self._fill(*found)
        self.sigShape.emit("fill" if rect is not None else "empty")

    def cancel(self) -> bool:
        if self.centre is None:
            return False
        self.centre = None
        self.drag_radius = 0.0
        self._preview(None)
        self.sigShape.emit("cancel")
        return True

    # -- helpers ------------------------------------------------------------
    def _radius_to(self, x: float, y: float) -> float:
        cx, cy = self.centre  # type: ignore[misc]
        return math.hypot(float(x) - cx, float(y) - cy)

    def _in_view(self, x: float, y: float) -> bool:
        """Is image point ``(x, y)`` on the canvas' viewport (``True`` for a stub)?"""
        shows = getattr(self.canvas, "shows_image_point", None)
        return True if shows is None else bool(shows(x, y))

    def _show(self) -> None:
        if self.centre is None:
            self._preview(None)
        else:
            # The disk a release here would fill -- the brush's while the drag
            # is under a pixel -- so the preview never shows another circle.
            self._preview(("circle", self.centre, self.fill_radius()))


