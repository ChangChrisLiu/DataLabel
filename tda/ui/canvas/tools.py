"""Canvas edit tools: brush, eraser, occluder brush and SAM prompts (spec 4.6).

A tool is a stateless-ish adapter between the canvas mouse signals and the
overlay's edit layers.  All coordinates arriving at ``on_press``/``on_move``/
``on_release`` are **image** coordinates (floats) produced by
:class:`~tda.ui.canvas.view.ImageCanvas`; nothing here knows about the viewport
transform.

The pixel tools never trigger inference (spec 4.3), and the SAM tools never
touch the committed label map -- both write to the *editing* layer only, so an
undoable op is always a single mask replacement.

Threading: :class:`~tda.models.sam_service.SamQueue` invokes its callback on the
worker thread.  :class:`SamResultBridge` is the only thing that touches it
there; it re-emits the payload through a queued signal so the mask is applied
on the GUI thread.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal

from tda.models.sam_service import SamRequest, SamResult
from tda.ui.canvas.overlay import OCCLUDER_TYPES, LabelOverlay

__all__ = [
    "OCCLUDER_TYPES",
    "MAX_SAM_SIDE",
    "Tool",
    "PaintTool",
    "BrushTool",
    "EraserTool",
    "OccluderTool",
    "SamResultBridge",
    "SamToolBase",
    "SamPointTool",
    "SamBoxTool",
    "viewport_crop",
]

Rect = tuple[int, int, int, int]
Point = tuple[float, float, int]
Box = tuple[float, float, float, float]

#: SAM 2.1 resizes its input to 1024 anyway, so a longer crop wastes work
#: and costs boundary precision on the way back (spec 4.6).
MAX_SAM_SIDE = 1024


def _union(a: Optional[Rect], b: Optional[Rect]) -> Optional[Rect]:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _norm_box(a: tuple[float, float], b: tuple[float, float]) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------
class Tool(QObject):
    """Base tool: receives image-space mouse callbacks, does nothing.

    ``sigStroke`` carries the dirty rect ``(x0, y0, x1, y1)`` of a finished
    edit, which the session layer turns into an undoable op.
    """

    sigStroke = Signal(object)

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
        self.canvas.sigMousePress.connect(self.on_press)
        self.canvas.sigMouseMove.connect(self.on_move)
        self.canvas.sigMouseRelease.connect(self.on_release)
        self._attached = True

    def detach(self) -> None:
        """Disconnect from the canvas mouse signals (idempotent)."""
        if self.canvas is None or not self._attached:
            return
        self.canvas.sigMousePress.disconnect(self.on_press)
        self.canvas.sigMouseMove.disconnect(self.on_move)
        self.canvas.sigMouseRelease.disconnect(self.on_release)
        self._attached = False

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
    def on_press(self, x: float, y: float, ev: Any) -> None:
        if self.overlay is None:
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
# SAM tools
# ---------------------------------------------------------------------------
def viewport_crop(
    canvas: Any, max_side: int = MAX_SAM_SIDE
) -> Optional[tuple[np.ndarray, Rect, float]]:
    """``(crop, rect, scale)`` for the visible image region, or ``None``.

    ``rect`` is the crop window in image coordinates and ``scale`` the factor
    applied to fit ``max_side`` (1.0 when the viewport is already small enough,
    which is the normal case once the annotator has zoomed in).

    A viewport larger than ``max_side`` is **downscaled rather than tiled**
    (spec 4.6 mentions tiling; deferred to P2).  SAM 2 resizes whatever it gets
    to 1024x1024 internally, so tiling would buy detail only where the
    annotator is already expected to zoom in, and there the crop is native
    resolution.  The one visible consequence: ``SamService`` measures its local
    refinement radius (:data:`~tda.models.sam_service.REFINE_RADIUS_PX`, 48 px)
    in *crop* pixels, so the region a refinement click can change spans
    ``REFINE_RADIUS_PX / scale`` **image** pixels -- a zoomed-out view refines
    coarsely.  Zoom in for a tight correction.
    """
    rgb = canvas.image_rgb()
    if rgb is None:
        return None
    rect = canvas.viewport_image_rect()
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return None
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        crop = cv2.resize(
            crop,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(crop), rect, scale


class SamResultBridge(QObject):
    """Moves a SAM result from the worker thread onto the GUI thread.

    ``SamQueue`` calls its callback on its own thread; touching the overlay or
    the scene from there would be a data race.  :meth:`deliver` is the callback
    and does nothing but emit -- the connection is queued, so the slot runs in
    the thread that owns this object (the GUI thread).
    """

    sigResult = Signal(object)

    def deliver(self, payload: object) -> None:
        """Callback for ``SamQueue.submit`` -- runs on the worker thread."""
        self.sigResult.emit(payload)


class SamToolBase(Tool):
    """Shared plumbing for the SAM prompt tools.

    Attributes:
        queue: a ``SamQueue`` (or any object with ``submit(req, cb)``); ``None``
            disables submission so the UI still works without a checkpoint.
        refine: when True the current editing mask is sent as ``mask_input`` and
            only the region near the new points changes (spec 4.6).
        instance: instance key the result belongs to; ``None`` keeps whatever
            the overlay is already editing.
    """

    def __init__(
        self,
        canvas: Any = None,
        overlay: Optional[LabelOverlay] = None,
        queue: Any = None,
        refine: bool = False,
        instance: Optional[str] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(canvas, overlay, parent)
        self.queue = queue
        self.refine = bool(refine)
        self.instance = instance
        self.last_result: Optional[SamResult] = None
        self._bridge = SamResultBridge(self)
        self._bridge.sigResult.connect(
            self._on_result, Qt.ConnectionType.QueuedConnection
        )

    # -- submission ---------------------------------------------------------
    def _submit(self, points: Sequence[Point], box: Optional[Box] = None) -> None:
        if self.queue is None or self.canvas is None or self.overlay is None:
            return
        prepared = viewport_crop(self.canvas)
        if prepared is None:
            return
        crop, rect, scale = prepared
        x0, y0, x1, y1 = rect

        crop_points: list[Point] = [
            ((px - x0) * scale, (py - y0) * scale, int(label))
            for px, py, label in points
            if x0 <= px < x1 and y0 <= py < y1
        ]
        crop_box: Optional[Box] = None
        if box is not None:
            bx0 = (min(max(box[0], x0), x1) - x0) * scale
            by0 = (min(max(box[1], y0), y1) - y0) * scale
            bx1 = (min(max(box[2], x0), x1) - x0) * scale
            by1 = (min(max(box[3], y0), y1) - y0) * scale
            crop_box = (bx0, by0, bx1, by1)
        if not crop_points and crop_box is None:
            return

        req = SamRequest(
            image_crop=crop,
            points=crop_points,
            box=crop_box,
            mask_input=self._mask_input(rect, crop.shape[:2]),
            # One point is ambiguous (part vs. whole assembly), so let SAM
            # propose three candidates and keep its best; with more prompts the
            # user has already disambiguated.
            multimask=len(crop_points) == 1 and crop_box is None,
        )
        bridge, refine = self._bridge, self.refine
        self.queue.submit(req, lambda res: bridge.deliver((res, rect, refine)))

    def _mask_input(
        self, rect: Rect, crop_hw: tuple[int, int]
    ) -> Optional[np.ndarray]:
        """The editing mask cropped to ``rect``, or ``None`` outside refine mode."""
        if not self.refine or self.overlay is None:
            return None
        x0, y0, x1, y1 = rect
        prior = self.overlay.editing[y0:y1, x0:x1]
        if not prior.any():
            return None
        if prior.shape != tuple(crop_hw):
            prior = cv2.resize(
                prior.astype(np.uint8),
                (crop_hw[1], crop_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return np.ascontiguousarray(prior, dtype=bool)

    # -- result -------------------------------------------------------------
    def _on_result(self, payload: object) -> None:
        """Apply a SAM mask to the editing layer (GUI thread)."""
        if self.overlay is None:
            return
        result, rect, refine = payload  # type: ignore[misc]
        self.last_result = result
        x0, y0, x1, y1 = rect
        mask = np.asarray(result.mask).astype(bool)
        if mask.shape != (y1 - y0, x1 - x0):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (x1 - x0, y1 - y0),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        # Outside the crop the prediction says nothing: in refine mode the prior
        # mask survives there, otherwise the layer is replaced outright.
        full = (
            self.overlay.editing.copy()
            if refine
            else np.zeros(self.overlay.hw, dtype=bool)
        )
        full[y0:y1, x0:x1] = mask
        instance = self.instance or self.overlay.editing_instance or "editing"
        self.overlay.set_editing(instance, full)
        if self.canvas is not None:
            self.canvas.refresh(rect)
        self.sigStroke.emit(rect)


class SamPointTool(SamToolBase):
    """Point prompts: left click = positive, right click = negative.

    Points accumulate so every click refines the same proposal; the session
    calls :meth:`clear_points` when the target instance changes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.points: list[Point] = []

    def on_press(self, x: float, y: float, ev: Any) -> None:
        label = 1
        button = getattr(ev, "button", None)
        if button is not None and button() == Qt.MouseButton.RightButton:
            label = 0
        self.points.append((float(x), float(y), label))
        self._submit(self.points)

    def clear_points(self) -> None:
        """Forget the collected prompts (e.g. after accepting the mask)."""
        self.points = []


class SamBoxTool(SamToolBase):
    """Box prompt dragged over the part; shows a rubber band while dragging."""

    #: Drags smaller than this (image px on either side) are treated as clicks.
    MIN_BOX = 2.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.box: Optional[Box] = None
        self._start: Optional[tuple[float, float]] = None
        self._dragging = False

    def on_press(self, x: float, y: float, ev: Any) -> None:
        self._start = (float(x), float(y))
        self._dragging = True
        self.box = None

    def on_move(self, x: float, y: float, ev: Any) -> None:
        if not self._dragging or self._start is None:
            return
        self.box = _norm_box(self._start, (float(x), float(y)))
        if self.canvas is not None:
            self.canvas.set_rubber_band(self.box)

    def on_release(self, x: float, y: float, ev: Any) -> None:
        if not self._dragging or self._start is None:
            return
        self._dragging = False
        box = _norm_box(self._start, (float(x), float(y)))
        self._start = None
        if self.canvas is not None:
            self.canvas.set_rubber_band(None)
        if box[2] - box[0] < self.MIN_BOX or box[3] - box[1] < self.MIN_BOX:
            self.box = None
            return
        self.box = box
        self._submit([], box=box)
