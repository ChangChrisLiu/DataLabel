"""The "现在做什么 / What now" dock: the operation guide, on the main window.

The trial annotator asked for "the operation guide's hints on the main window
so I can follow along" (task U2b).  This panel shows a
:class:`tda.ui.guide.GuidePlan`: one highlighted line naming the exact next
key or button, the frame's five steps with the current one marked, and -- one
click away -- the keys that work in the current mode, generated from the same
table the keyboard uses.

It never takes the keyboard (every widget is ``NoFocus`` and no text is
selectable), and it repaints only when the plan it is handed differs from the
one on screen: the window may hand it the same plan on every status update
and that costs one tuple comparison.
"""
from __future__ import annotations

import html
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from tda.ui import guide as G

__all__ = ["WhatNowPanel"]

_MARKERS = ("①", "②", "③", "④", "⑤", "⑥", "⑦")


def _steps_html(steps) -> str:
    rows = []
    for index, (state, text) in enumerate(steps):
        marker = _MARKERS[index] if index < len(_MARKERS) else f"{index + 1}."
        body = html.escape(text)
        if state == G.NOW:
            rows.append(f"<tr><td valign='top' style='color:#1e5bb0'><b>▶ {marker}</b></td>"
                        f"<td style='background:#fff3b8'><b>{body}</b></td></tr>")
        elif state == G.DONE:
            rows.append(f"<tr><td valign='top' style='color:#3a8a4a'>✓ {marker}</td>"
                        f"<td style='color:#8a8a90'>{body}</td></tr>")
        elif state == G.WARN:
            rows.append(f"<tr><td valign='top' style='color:#c06000'>⚠ {marker}</td>"
                        f"<td style='color:#a05000'>{body}</td></tr>")
        else:
            rows.append(f"<tr><td valign='top' style='color:#6a6a70'>&nbsp;&nbsp;{marker}</td>"
                        f"<td style='color:#404046'>{body}</td></tr>")
    return "<table cellspacing='2' cellpadding='1'>" + "".join(rows) + "</table>"


class WhatNowPanel(QWidget):
    """Title, the "现在：…" line, the numbered steps and a folded cheat sheet."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._plan: Optional[G.GuidePlan] = None
        self._cheat = ""
        self.title = self._label()
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.now = self._label()
        self.now.setStyleSheet("QLabel { background: #fff3b8; border-left: 4px solid "
                               "#e0a800; padding: 4px 6px; font-weight: bold; }")
        self.steps = self._label()
        self.steps.setTextFormat(Qt.TextFormat.RichText)
        self.cheat_button = QToolButton()
        self.cheat_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cheat_button.setCheckable(True)
        self.cheat_button.setAutoRaise(True)
        self.cheat_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.cheat_button.toggled.connect(self._fold)
        self.cheat = self._label()
        self.cheat.setTextFormat(Qt.TextFormat.RichText)
        self.cheat.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        for widget in (self.title, self.now, self.steps, self.cheat_button, self.cheat):
            layout.addWidget(widget)
        layout.addStretch(1)
        self._fold(False)

    @staticmethod
    def _label() -> QLabel:
        label = QLabel("")
        label.setWordWrap(True)
        label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        # A long sentence wraps; it never asks the dock to be wider.
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        label.setMinimumWidth(1)
        return label

    def _fold(self, open_: bool) -> None:
        self.cheat.setVisible(bool(open_))
        self.cheat_button.setText(("▾ " if open_ else "▸ ") + "快捷键（这个模式能用的）")

    def show_plan(self, plan: G.GuidePlan, cheat_html: str = "") -> bool:
        """Put ``plan`` on screen; ``False`` (and no repaint) when it already is."""
        if plan == self._plan and cheat_html == self._cheat:
            return False
        if plan != self._plan:
            self._plan = plan
            self.title.setText(plan.title)
            self.now.setText(plan.now)
            self.steps.setText(_steps_html(plan.steps) if plan.steps else "")
            self.steps.setVisible(bool(plan.steps))
        if cheat_html != self._cheat:
            self._cheat = cheat_html
            self.cheat.setText(cheat_html)
        return True

    def plan(self) -> Optional[G.GuidePlan]:
        """The plan on screen."""
        return self._plan

    def text(self) -> str:
        """Everything the panel says, as plain text (for the tests)."""
        if self._plan is None:
            return ""
        lines = [self._plan.title, self._plan.now]
        lines += [text for _state, text in self._plan.steps]
        return "\n".join(lines)
