"""The two small widgets the editing half draws with: a box drag and a bar.

Neither belongs to the editing *logic* -- a rectangle dragged on the canvas is
the ROI or a bench box depending on who armed the tool, and the bar is a
one-line non-modal strip used both for a scope suggestion and for a recovered
edit.  They live here so :mod:`tda.ui.app_edit` can be about edits.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from tda.ui.canvas.tools import Tool

__all__ = ["MIN_BOX_PX", "Bar", "BoxDragTool"]

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


