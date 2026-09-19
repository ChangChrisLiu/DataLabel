"""Canvas edit tools: the pixel tools (brush, eraser, occluder brush), spec 4.6.

A tool is a stateless-ish adapter between the canvas mouse signals and the
overlay's edit layers.  All coordinates arriving at ``on_press``/``on_move``/
``on_release`` are **image** coordinates (floats) produced by
:class:`~tda.ui.canvas.view.ImageCanvas`; nothing here knows about the viewport
transform.

The pixel tools never trigger inference (spec 4.3) and never touch the committed
label map -- they write to the *editing* layer only, so an undoable op is always
a single mask replacement.

The asynchronous SAM prompt tools live in :mod:`tda.ui.canvas.sam_tools`, which
imports :class:`Tool` from here.  They stay importable from this module (see
:func:`__getattr__`) so that ``from tda.ui.canvas.tools import SamPointTool``
keeps working; the indirection is lazy because a plain import in this direction
would close an import cycle.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

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


