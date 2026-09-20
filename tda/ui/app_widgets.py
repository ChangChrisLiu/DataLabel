"""The two small widgets the editing half draws with: a box drag and a bar.

Neither belongs to the editing *logic* -- a rectangle dragged on the canvas is
the ROI or a bench box depending on who armed the tool, and the bar is a
one-line non-modal strip used both for a scope suggestion and for a recovered
edit.  They live here so :mod:`tda.ui.app_edit` can be about edits.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from tda.ui.canvas.tools import Tool

__all__ = ["MIN_BOX_PX", "Bar", "BoxDragTool", "RoiBoxTool"]

#: A drag shorter than this on either side is a click, not a box.
MIN_BOX_PX = 2.0


class BoxDragTool(Tool):
    """Drag a rectangle on the canvas; used for the ROI and for bench boxes.

    It writes nothing: the box is reported and whoever armed the tool decides
    what it means -- the pose segment's ROI, or the staging-area box of a part
    that is now on the bench (spec 4.2 S4).
    """

    sigBox = Signal(object)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.box: Optional[tuple[float, float, float, float]] = None
        self._start: Optional[tuple[float, float]] = None

    def on_press(self, x: float, y: float, ev: Any) -> None:
        self._start = (float(x), float(y))

    def on_move(self, x: float, y: float, ev: Any) -> None:
        if self._start is None:
            return
        self.box = self._norm(x, y)
        if self.canvas is not None:
            self.canvas.set_rubber_band(self.box)

    def on_release(self, x: float, y: float, ev: Any) -> None:
        if self._start is None:
            return
        box = self._norm(x, y)
        self._start = None
        if box[2] - box[0] < MIN_BOX_PX or box[3] - box[1] < MIN_BOX_PX:
            return
        self.box = box
        self.sigBox.emit(box)

    def _norm(self, x: float, y: float) -> tuple[float, float, float, float]:
        sx, sy = self._start or (x, y)
        return (min(sx, x), min(sy, y), max(sx, x), max(sy, y))


class RoiBoxTool(BoxDragTool):
    """The chassis rectangle: drag a handle to resize, inside to move, else redraw.

    The first annotator to use the tool did not understand that the rectangle
    was waiting for an answer, or that it could be adjusted at all (task U1,
    report 2).  A rectangle you can only replace by drawing a new one from
    scratch does not *look* adjustable, so this adds the two gestures every
    other cropping tool has -- eight handles and a grab-inside-to-move -- and
    a cursor that changes over them, which is what says they are there.

    Every gesture reports through :attr:`sigBox` exactly like
    :class:`BoxDragTool`, so what the window does with the rectangle is
    unchanged; :attr:`sigPreview` is the same box *during* the drag, for the
    canvas to draw.
    """

    #: How close to a handle (screen px) still counts as grabbing it.
    GRAB_PX = 9
    #: Smallest rectangle a resize may leave, in image pixels.
    MIN_SIDE = 4.0
    _CURSORS = {
        "nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
        "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
        "n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
        "w": Qt.CursorShape.SizeHorCursor, "e": Qt.CursorShape.SizeHorCursor,
        "inside": Qt.CursorShape.SizeAllCursor,
    }

    #: The rectangle as it stands mid-drag; the window stores it on release.
    sigPreview = Signal(object)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: The rectangle being edited, in image coordinates.
        self.rect: Optional[tuple[float, float, float, float]] = None
        self._mode: Optional[str] = None
        self._anchor: Optional[tuple[float, float]] = None
        self._before: Optional[tuple[float, float, float, float]] = None

    # -- state --------------------------------------------------------------
    def set_rect(self, box: Optional[tuple]) -> None:
        """Tell the tool which rectangle is on screen (``None``: none yet)."""
        self.rect = None if box is None else tuple(float(v) for v in box)

    def detach(self) -> None:  # noqa: D102 - a half-finished drag is not a box
        super().detach()
        self._mode = self._anchor = self._before = None

    def _tolerance(self) -> float:
        zoom = 1.0 if self.canvas is None else max(self.canvas.zoom_factor(), 1e-6)
        return self.GRAB_PX / zoom

    def hit(self, x: float, y: float) -> Optional[str]:
        """Which part of the rectangle is under ``(x, y)``: a handle, or inside.

        The grab radius is capped at a third of each side and the **nearest**
        handle wins, not the first in the list. On a rectangle narrower than
        the tolerance every point used to be within reach of ``nw``, so a
        rectangle too small to resize was also one you could not move (round 2,
        M3) -- and moving is exactly what you want when it is that small.
        """
        if self.rect is None:
            return None
        x0, y0, x1, y1 = self.rect
        near = self._tolerance()
        near_x = min(near, max(1e-6, (x1 - x0) / 3.0))
        near_y = min(near, max(1e-6, (y1 - y0) / 3.0))
        best, best_d = None, None
        for name, fx, fy in getattr(type(self.canvas), "HANDLES", ()):
            hx, hy = x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy
            if abs(x - hx) <= near_x and abs(y - hy) <= near_y:
                distance = (x - hx) ** 2 + (y - hy) ** 2
                if best_d is None or distance < best_d:
                    best, best_d = name, distance
        if best is not None:
            return best
        if x0 <= x <= x1 and y0 <= y <= y1:
            return "inside"
        return None

    def cursor_for(self, x: float, y: float):
        """The cursor the rectangle wants at ``(x, y)``; a crosshair off it."""
        return self._CURSORS.get(self.hit(x, y) or "", Qt.CursorShape.CrossCursor)

    # -- gestures -----------------------------------------------------------
    def on_press(self, x: float, y: float, ev: Any) -> None:  # noqa: D102
        where = self.hit(x, y)
        self._before = self.rect
        self._anchor = (float(x), float(y))
        if where is None:
            self._mode = "draw"
            super().on_press(x, y, ev)
            return
        self._mode = where
        self._start = None      # not a fresh drag: BoxDragTool must stay out

    def on_move(self, x: float, y: float, ev: Any) -> None:  # noqa: D102
        if self._mode is None:
            # Hovering: the cursor is the only thing that says the handles exist.
            if self.canvas is not None:
                self.canvas.viewport().setCursor(self.cursor_for(x, y))
            return
        if self._mode == "draw":
            super().on_move(x, y, ev)
            self.rect = self.box
            self.sigPreview.emit(self.box)
            return
        moved = self._moved(x, y, self._mode)
        if moved is not None:
            self.rect = moved
            self.sigPreview.emit(moved)

    def on_release(self, x: float, y: float, ev: Any) -> None:  # noqa: D102
        mode, self._mode = self._mode, None
        if mode is None:
            return
        if mode == "draw":
            before, self._before = self._before, None
            box = self._norm(x, y) if self._start is not None else None
            super().on_release(x, y, ev)
            if box is None or (box[2] - box[0] < MIN_BOX_PX
                               or box[3] - box[1] < MIN_BOX_PX):
                # A click, or a drag too small to be a rectangle: ``sigBox``
                # was not emitted, so the window still holds the old draft and
                # the canvas has to be put back to it rather than left showing
                # the sliver the preview drew on the way.
                self.rect = before
                self.sigPreview.emit(before)
                return
            self.rect = self.box
            return
        moved = self._moved(x, y, mode)
        self._anchor = self._before = None
        if moved is None:
            return
        self.rect = self.box = moved
        self.sigBox.emit(moved)

    def _moved(self, x: float, y: float, mode: Optional[str]):
        """The rectangle after dragging ``mode`` (a handle, or inside) to ``(x, y)``.

        ``mode`` is passed rather than read off the tool: ``on_release`` has to
        clear it before it can emit, and reading it back there produced a
        release that moved nothing at all.
        """
        if self._before is None or self._anchor is None or mode is None:
            return None
        x0, y0, x1, y1 = self._before
        dx, dy = float(x) - self._anchor[0], float(y) - self._anchor[1]
        if mode == "inside":
            return self._clamped((x0 + dx, y0 + dy, x1 + dx, y1 + dy), mode)
        if "w" in mode:
            x0 = min(x0 + dx, x1 - self.MIN_SIDE)
        if "e" in mode:
            x1 = max(x1 + dx, x0 + self.MIN_SIDE)
        if "n" in mode:
            y0 = min(y0 + dy, y1 - self.MIN_SIDE)
        if "s" in mode:
            y1 = max(y1 + dy, y0 + self.MIN_SIDE)
        return self._clamped((x0, y0, x1, y1), mode)

    def _clamped(self, box, mode: Optional[str]):
        """Keep the rectangle inside the frame, moving it rather than shrinking."""
        if self.canvas is None:
            return box
        h, w = self.canvas.image_hw()
        if not w or not h:
            return box
        x0, y0, x1, y1 = box
        if mode == "inside":
            # A move keeps its size: pushing it off an edge slides it back.
            width, height = x1 - x0, y1 - y0
            x0 = min(max(0.0, x0), max(0.0, float(w) - width))
            y0 = min(max(0.0, y0), max(0.0, float(h) - height))
            return (x0, y0, min(float(w), x0 + width), min(float(h), y0 + height))
        return (max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1))


class Bar(QWidget):
    """A one-line non-modal bar under the canvas (scope suggestion, restore offer)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(self.label, 1)
        self.buttons = layout
        self.hide()

    def add_button(self, text: str, slot) -> QPushButton:
        """A button whose ``clicked(bool)`` never reaches a no-argument slot."""
        button = QPushButton(text, self)
        button.clicked.connect(lambda _checked=False: slot())
        self.buttons.addWidget(button)
        return button

    def show_text(self, text: str) -> None:
        self.label.setText(text)
        self.show()


