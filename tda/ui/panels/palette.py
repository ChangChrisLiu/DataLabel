"""The tool palette: every tool and every commit key as a button (task U2b).

The second trial's annotator asked for "shortcuts or buttons I can simply
click to choose a function".  Everything here already had a key; this is the
same set of keys as buttons, docked left of the canvas where the eye already
is.

**One key map.**  A button is built from an :class:`~tda.ui.app_actions.Action`
by name: its caption is the action's ``short``, its key is read off the
action's own ``keys`` and its tooltip is ``label_zh`` + ``label``.  Clicking it
emits :attr:`ToolPalette.sigAction` with that name, and the window runs the
very slot the key runs -- so a button cannot do something its key does not,
and its label cannot drift from the keyboard.

**Nothing here takes the keyboard** but the brush-size number.  Every button
and the slider are ``NoFocus``: the first trial ended with a combo box holding
the focus and every shortcut silently switched off, and a palette that stole
the focus on each click would be that bug with seventeen doors.  The number is
typed into (round 1): it takes the focus on a click, shortcuts stand aside the
way they do for every text field, and ``Enter`` / ``Esc`` give it back.

**The window decides.**  The palette never knows whether an action can run;
:meth:`ToolPalette.apply_states` is told which buttons are enabled, which one
is checked (the armed tool -- the same state the status badge reads) and why a
disabled one is disabled, and says it in the tooltip.

Why vertical and on the left: the scanner frames are square and the canvas is
16:9, so at 1920x1080 and 2560x1440 the frame is limited by the canvas'
*height* and the canvas has spare width on both sides.  A strip down the left
costs the frame nothing there (and about 5 % on a 4:3 OAK frame); a toolbar
across the top would shrink every frame by its own height.
"""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tda.ui import app_actions as A
from tda.ui.canvas.overlay import EDIT_RGB as _EDIT_RGB
from tda.ui.canvas.overlay import OCCLUDER_RGB as _OCCLUDER_RGB

__all__ = ["PALETTE_WIDTH", "RADIUS_MAX", "RADIUS_MIN", "PaletteButton",
           "ToolPalette", "radius_to_slider", "slider_to_radius", "tool_icon"]

#: The strip's width in logical pixels (see the module docstring for why a
#: strip at all).  Measured against ``tests/test_app.py``'s canvas share.
PALETTE_WIDTH = 124
#: The brush, the eraser and the occluder share one radius, in image pixels.
RADIUS_MIN = 1
RADIUS_MAX = 200
#: The slider runs 0..SLIDER_STEPS and maps to a radius *quadratically*: on a
#: strip a hundred pixels wide a linear 1..200 put every useful brush (3-30 px)
#: into the first fifteen pixels of travel.
SLIDER_STEPS = 100


def slider_to_radius(value: int) -> int:
    """Slider position -> radius; quadratic, exact at both ends."""
    frac = min(max(int(value), 0), SLIDER_STEPS) / SLIDER_STEPS
    return int(round(RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * frac * frac))


def radius_to_slider(radius: int) -> int:
    """Radius -> the slider position that shows it best."""
    r = min(max(int(radius), RADIUS_MIN), RADIUS_MAX)
    return int(round(SLIDER_STEPS * math.sqrt((r - RADIUS_MIN) / (RADIUS_MAX - RADIUS_MIN))))


# --------------------------------------------------------------------------- #
# icons: what the tool looks like under the mouse, in small
# --------------------------------------------------------------------------- #
ICON_PX = 18


def tool_icon(name: str, dpr: float = 1.0) -> Optional[QPixmap]:
    """A small drawn icon for a tool button, or ``None`` for an action.

    Drawn, not loaded: no files to ship, and the brush and eraser icons are the
    canvas cursors in miniature -- a yellow ring and a dashed white one -- so
    the button and the pointer look like the same thing.
    """
    painters = {
        "tool_brush": _icon_brush, "tool_eraser": _icon_eraser,
        "tool_sam_point": _icon_point, "tool_sam_box": _icon_box,
        "tool_occluder": _icon_occluder, "tool_bench_box": _icon_bench,
        "edit_roi": _icon_roi,
    }
    draw = painters.get(name)
    if draw is None:
        return None
    dpr = max(1.0, float(dpr))
    pixmap = QPixmap(int(round(ICON_PX * dpr)), int(round(ICON_PX * dpr)))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    draw(painter, float(ICON_PX))
    painter.end()
    return pixmap


def _ring(p: QPainter, s: float, rgb, dashed: bool, fill: bool = False) -> None:
    box = QRectF(2.5, 2.5, s - 5, s - 5)
    p.setPen(QPen(QColor(0, 0, 0, 170), 3.0))
    p.drawEllipse(box)
    pen = QPen(QColor(*rgb), 1.8)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    if fill:
        p.setBrush(QColor(rgb[0], rgb[1], rgb[2], 150))
    p.drawEllipse(box)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _icon_brush(p: QPainter, s: float) -> None:
    _ring(p, s, _EDIT_RGB, dashed=False, fill=True)


def _icon_eraser(p: QPainter, s: float) -> None:
    _ring(p, s, (245, 245, 245), dashed=True)


def _icon_occluder(p: QPainter, s: float) -> None:
    _ring(p, s, _OCCLUDER_RGB, dashed=False, fill=True)


def _icon_point(p: QPainter, s: float) -> None:
    c = s / 2.0
    p.setPen(QPen(QColor(40, 170, 90), 1.6))
    p.drawEllipse(QRectF(c - 6, c - 6, 12, 12))
    p.setBrush(QColor(40, 170, 90))
    p.drawEllipse(QRectF(c - 2.5, c - 2.5, 5, 5))
    p.setBrush(Qt.BrushStyle.NoBrush)


def _icon_box(p: QPainter, s: float) -> None:
    p.setPen(QPen(QColor(40, 170, 90), 1.6, Qt.PenStyle.DashLine))
    p.drawRect(QRectF(2.5, 3.5, s - 5, s - 7))
    p.setBrush(QColor(40, 170, 90))
    p.drawEllipse(QRectF(s - 6, s - 6, 4, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)


def _icon_bench(p: QPainter, s: float) -> None:
    p.setPen(QPen(QColor(230, 130, 30), 1.8))
    p.drawRect(QRectF(3.5, 3.5, s - 7, s - 9))
    p.setPen(QPen(QColor(120, 120, 120), 2.0))
    p.drawLine(1, int(s - 2), int(s - 1), int(s - 2))


def _icon_roi(p: QPainter, s: float) -> None:
    p.setPen(QPen(QColor(0, 0, 0, 200), 3))
    p.drawRect(QRectF(3, 3, s - 6, s - 6))
    p.setPen(QPen(QColor(*_EDIT_RGB), 1.6, Qt.PenStyle.DashLine))
    p.drawRect(QRectF(3, 3, s - 6, s - 6))
    p.setPen(QPen(QColor(0, 0, 0), 0.8))
    p.setBrush(QColor(255, 255, 255))
    for x, y in ((3, 3), (s - 3, 3), (3, s - 3), (s - 3, s - 3)):
        p.drawRect(QRectF(x - 1.8, y - 1.8, 3.6, 3.6))
    p.setBrush(Qt.BrushStyle.NoBrush)


# --------------------------------------------------------------------------- #
# one button
# --------------------------------------------------------------------------- #
class PaletteButton(QAbstractButton):
    """Icon, name, key in a keycap, and an optional note -- painted, not styled.

    Painted by hand so that the three things a first-time user reads -- what
    it is, which key does the same, and which one is armed -- have one fixed
    place each, and the armed tool can wear exactly the status badge's colours.
    Checkable buttons do **not** toggle themselves on a click
    (:meth:`nextCheckState`): whether a tool is armed is the window's answer,
    and a refused click must not leave a button looking armed.
    """

    PAD = 5
    GAP = 4

    #: A press on the button while it is greyed out: say why (round 2).
    sigRefusedClick = Signal()

    def event(self, event) -> bool:  # noqa: D102 - Qt override
        # A disabled widget still receives its mouse events here (Qt only
        # ignores them afterwards); a greyed button answers a click with its
        # reason instead of with nothing.
        if not self.isEnabled() and event.type() == QEvent.Type.MouseButtonRelease:
            self.sigRefusedClick.emit()
            return True
        return super().event(event)

    def __init__(self, action: A.Action, checkable: bool = False,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.action = action
        self.name, self.note = A.short_parts(action)
        self.key = A.key_caption(action)
        self._icon = tool_icon(action.name, self.devicePixelRatioF() or 1.0)
        self._suggested = False
        self._reason = ""
        self.setCheckable(checkable)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setText(f"{self.name} {self.key}")
        self._base_tip = (f"{action.label_zh}（{' / '.join(action.keys)}）"
                          f"\n{action.label}")
        self.setToolTip(self._base_tip)

    # -- state the window sets ----------------------------------------------
    def nextCheckState(self) -> None:  # noqa: D102 - the window decides
        pass

    def set_reason(self, reason: str) -> None:
        """Why the button is disabled; shown in the tooltip."""
        reason = str(reason or "")
        if reason != self._reason:
            self._reason = reason
            self.setToolTip(self._base_tip + (f"\n现在不能用：{reason}" if reason else ""))

    def reason(self) -> str:
        return self._reason

    def set_suggested(self, on: bool) -> None:
        """Wear the "this is the next thing to press" border."""
        if bool(on) != self._suggested:
            self._suggested = bool(on)
            self.update()

    def suggested(self) -> bool:
        return self._suggested

    # -- geometry -----------------------------------------------------------
    def _fonts(self) -> tuple[QFont, QFont]:
        name = QFont(self.font())
        name.setBold(True)
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, self.font().pointSizeF() - 1.0))
        return name, small

    def _text_left(self) -> int:
        return self.PAD + ((ICON_PX + self.GAP) if self._icon is not None else 0)

    def _layout(self, width: int) -> dict:
        name_font, small_font = self._fonts()
        nm, sm = QFontMetrics(name_font), QFontMetrics(small_font)
        left = self._text_left()
        room = max(20, width - left - self.PAD)
        key_w = sm.horizontalAdvance(self.key) + 8
        key_h = sm.height() + 2
        name_rect = nm.boundingRect(QRect(0, 0, room, 1000),
                                    int(Qt.TextFlag.TextWordWrap), self.name)
        one_line = name_rect.height() <= nm.height() + 1 and \
            nm.horizontalAdvance(self.name) + 6 + key_w <= room
        height = self.PAD + max(name_rect.height(), key_h if one_line else 0)
        key_top = self.PAD if one_line else height + 2
        if not one_line:
            height = key_top + key_h
        note_rect = QRect()
        if self.note:
            note_x = left + (0 if one_line else key_w + 4)
            note_room = max(20, width - note_x - self.PAD)
            note_rect = sm.boundingRect(QRect(0, 0, note_room, 1000),
                                        int(Qt.TextFlag.TextWordWrap), self.note)
            note_top = height + 2 if one_line else key_top
            note_rect = QRect(note_x, note_top, note_room, note_rect.height())
            height = max(height, note_rect.bottom() + 1)
        height += self.PAD
        return {"left": left, "room": room, "one_line": one_line,
                "name_h": name_rect.height(), "key_w": key_w, "key_h": key_h,
                "key_top": key_top, "note": note_rect, "height": max(height, ICON_PX + 2 * self.PAD)}

    def sizeHint(self) -> QSize:  # noqa: D102
        width = PALETTE_WIDTH - 12
        return QSize(width, self._layout(width)["height"])

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        return self.sizeHint()

    def hasHeightForWidth(self) -> bool:  # noqa: D102
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: D102
        return self._layout(width)["height"]

    # -- paint --------------------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: D102
        lay = self._layout(self.width())
        name_font, small_font = self._fonts()
        enabled = self.isEnabled()
        checked = self.isCheckable() and self.isChecked()
        hover = enabled and self.underMouse()
        if checked:
            bg, fg, sub = QColor(47, 50, 56), QColor(255, 232, 64), QColor(210, 210, 200)
        elif self.isDown():
            bg, fg, sub = QColor(207, 216, 230), QColor(20, 20, 24), QColor(80, 80, 90)
        elif hover:
            bg, fg, sub = QColor(226, 234, 246), QColor(20, 20, 24), QColor(80, 80, 90)
        else:
            bg, fg, sub = QColor(246, 247, 249), QColor(20, 20, 24), QColor(96, 96, 104)
        if not enabled:
            fg, sub = QColor(160, 160, 166), QColor(176, 176, 182)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        border = QColor(255, 232, 64) if checked else QColor(201, 204, 210)
        width = 1.0
        if self._suggested and enabled:
            border, width = QColor(42, 130, 218), 2.5
        painter.setPen(QPen(border, width))
        painter.setBrush(bg)
        painter.drawRoundedRect(QRectF(1, 1, self.width() - 2, self.height() - 2), 5, 5)
        if self._icon is not None:
            painter.setOpacity(1.0 if enabled else 0.35)
            painter.drawPixmap(self.PAD, self.PAD + max(0, (lay["name_h"] - ICON_PX) // 2),
                               self._icon)
            painter.setOpacity(1.0)
        left, room = lay["left"], lay["room"]
        painter.setFont(name_font)
        painter.setPen(fg)
        painter.drawText(QRect(left, self.PAD, room, lay["name_h"]),
                         int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft), self.name)
        # the key, in a keycap: right of the name when it fits, under it if not
        painter.setFont(small_font)
        key_x = (left + room - lay["key_w"]) if lay["one_line"] else left
        cap = QRectF(key_x, lay["key_top"], lay["key_w"], lay["key_h"])
        painter.setPen(QPen(sub, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(cap.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)
        painter.drawText(cap, int(Qt.AlignmentFlag.AlignCenter), self.key)
        if self.note:
            painter.setPen(sub)
            painter.drawText(lay["note"], int(Qt.TextFlag.TextWordWrap
                                              | Qt.AlignmentFlag.AlignLeft), self.note)
        painter.end()


# --------------------------------------------------------------------------- #
# the strip
# --------------------------------------------------------------------------- #
class ToolPalette(QWidget):
    """Tools, the brush size, the commit/confirm/undo keys -- and the minimap.

    Per mode (:meth:`set_mode`): Annotate shows everything but the Review
    queue's keys; Review hides the tools and the brush size -- its canvas is
    read-only -- and keeps the difference map and the queue's four keys; Steps
    hides the whole strip (the window does that).

    The buttons scroll (:attr:`scroll`) on a short screen; the minimap sits
    below them, outside the scroll area, so it is always in view (U2b round 2:
    it used to float over the canvas, where a left drag that started on it
    moved the view instead of doing what the armed tool does).
    """

    #: ``(action name, pressed)``: a click (``pressed`` is ``True``), or the
    #: press and release of a held one (``对比上一帧``: ``Tab``).
    sigAction = Signal(str, bool)
    #: The annotator moved the slider or the spin box to this radius.
    sigRadius = Signal(int)
    #: Typing in the brush-size box ended (``Enter`` or ``Esc``): the window
    #: takes the keyboard back to the canvas.  The payload is the radius shown.
    sigRadiusTyped = Signal(int)
    #: A greyed button was clicked: ``(action name, why it cannot run)``.
    sigRefused = Signal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("tool_palette")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedWidth(PALETTE_WIDTH)
        self.scroll = QScrollArea(self)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self._minimap_host = QVBoxLayout()
        self._minimap_host.setContentsMargins(4, 2, 4, 4)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.scroll, 1)
        outer.addLayout(self._minimap_host)
        inner = QWidget()
        inner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        self._buttons: dict[str, PaletteButton] = {}
        self._mode = ""
        #: The drawing half of the strip, hidden in Review (round 1, item 3).
        self._tool_widgets: list[QWidget] = []

        header = self._header("工具")
        layout.addWidget(header)
        self._tool_widgets.append(header)
        for name in A.PALETTE_TOOLS:
            button = self._add(name, checkable=True)
            layout.addWidget(button)
            self._tool_widgets.append(button)

        self.radius_label = self._header("笔刷大小 r")
        self.radius_label.setToolTip("画笔、橡皮擦、遮挡共用一个大小，单位是图像像素；"
                                     "拖滑块、点数字框直接输入（Enter 确定，Esc 取消），"
                                     "或键盘 [ 变小、] 变大（可长按）")
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(0, SLIDER_STEPS)
        self.radius_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.radius_slider.setToolTip(self.radius_label.toolTip())
        # The one field on the strip that takes the keyboard, and only on a
        # click: while it has it, the window's shortcuts stand aside the way
        # they do for every text field (``blocks_shortcuts``), and Enter or Esc
        # hand it back to the canvas.  No keyboard tracking: a radius of 2 on
        # the way to 24 is not something anybody asked for.
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(RADIUS_MIN, RADIUS_MAX)
        self.radius_spin.setSuffix(" px")
        self.radius_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.radius_spin.setKeyboardTracking(False)
        self.radius_spin.setToolTip(self.radius_label.toolTip())
        self.radius_spin.setAccelerated(True)
        self.radius_spin.installEventFilter(self)
        # The wheel over either of them scrolls the strip unless the widget has
        # the keyboard (round 2): on a 1280x720 screen the strip scrolls, and
        # a wheel meant for it must not quietly resize the brush.
        self.radius_slider.installEventFilter(self)
        self.radius_spin.editingFinished.connect(self._on_spin_finished)
        layout.addSpacing(4)
        for widget in (self.radius_label, self.radius_slider, self.radius_spin):
            layout.addWidget(widget)
            self._tool_widgets.append(widget)
        self.radius_slider.valueChanged.connect(self._on_slider)
        self.radius_spin.valueChanged.connect(self._on_spin)

        layout.addSpacing(4)
        layout.addWidget(self._header("操作"))
        # Review's own keys first: in Review they are the whole strip, with
        # 差异图 after them; in Annotate they are hidden.
        review_first = tuple(n for n in A.PALETTE_REVIEW if n not in A.PALETTE_ACTIONS)
        for name in dict.fromkeys(review_first + A.PALETTE_ACTIONS + A.PALETTE_REVIEW):
            layout.addWidget(self._add(name, checkable=(name == "toggle_heat")))
        layout.addStretch(1)
        self.scroll.setWidget(inner)
        self.set_radius(8)
        self.set_mode(A.MODE_ANNOTATE)

    def set_minimap(self, minimap: QWidget) -> None:
        """Put the minimap at the foot of the strip, below the scrolling part.

        Taking it into this layout reparents it: the canvas has no widget over
        its picture any more, so a left drag on the canvas is always the armed
        tool's (U2b round 2).
        """
        label = self._header("小地图")
        label.setToolTip(minimap.toolTip())
        self._minimap_host.addWidget(label)
        if hasattr(minimap, "set_max_side"):
            minimap.set_max_side(PALETTE_WIDTH - 12)
        self._minimap_host.addWidget(minimap, 0, Qt.AlignmentFlag.AlignHCenter)

    @staticmethod
    def _header(text: str) -> QLabel:
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet("QLabel { color: #50545c; padding: 2px 2px 0 2px; }")
        return label

    def _add(self, name: str, checkable: bool) -> PaletteButton:
        action = A.action_named(name)
        button = PaletteButton(action, checkable=checkable)
        button.sigRefusedClick.connect(lambda n=name, b=button: self.sigRefused.emit(n, b.reason()))
        if action.hold:
            button.pressed.connect(lambda n=name: self.sigAction.emit(n, True))
            button.released.connect(lambda n=name: self.sigAction.emit(n, False))
        else:
            button.clicked.connect(lambda _c=False, n=name: self.sigAction.emit(n, True))
        self._buttons[name] = button
        return button

    # -- what the window reads and sets ---------------------------------------
    def button(self, name: str) -> PaletteButton:
        """The button of action ``name``."""
        return self._buttons[name]

    def buttons(self) -> dict[str, PaletteButton]:
        """Every button by action name (a copy)."""
        return dict(self._buttons)

    def apply_states(self, states: dict, suggested: str = "") -> None:
        """``{name: (enabled, checked, reason)}``; buttons not named are left alone."""
        for name, (enabled, checked, reason) in states.items():
            button = self._buttons.get(name)
            if button is None:
                continue
            if button.isEnabled() != bool(enabled):
                button.setEnabled(bool(enabled))
            if button.isCheckable() and button.isChecked() != bool(checked):
                button.setChecked(bool(checked))
            button.set_reason("" if enabled else reason)
        for name, button in self._buttons.items():
            button.set_suggested(name == suggested)

    def set_radius(self, radius: int) -> None:
        """Show ``radius`` on the slider and the spin box without re-announcing it."""
        r = min(max(int(radius), RADIUS_MIN), RADIUS_MAX)
        for widget, value in ((self.radius_spin, r),
                              (self.radius_slider, radius_to_slider(r))):
            if widget.value() != value:
                blocked = widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(blocked)

    def radius(self) -> int:
        return int(self.radius_spin.value())

    # -- per mode -------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Show what works in ``mode``: a button is shown when its action is live.

        The tools and the brush size are Annotate's alone; a button is shown
        exactly in the modes its ``ACTIONS`` row lists, so 差异图 stays in
        Review and the queue's 接受 / 返工 / K / N appear only there.
        """
        if mode == self._mode:
            return
        self._mode = mode
        drawing = mode == A.MODE_ANNOTATE
        for widget in self._tool_widgets:
            widget.setVisible(drawing)
        for name, button in self._buttons.items():
            if name in A.PALETTE_TOOLS:
                continue
            button.setVisible(mode in button.action.modes)

    def mode(self) -> str:
        return self._mode

    def tools_section_hidden(self) -> bool:
        """Are the drawing tools and the brush size put away (Review)?"""
        return all(widget.isHidden() for widget in self._tool_widgets)

    # -- typing a radius ------------------------------------------------------
    def eventFilter(self, obj, event) -> bool:  # noqa: D102 - the spin box's Esc
        if (event.type() == QEvent.Type.Wheel
                and obj in (self.radius_slider, self.radius_spin) and not obj.hasFocus()):
            # Not focused: the wheel belongs to the strip, as in any list of
            # controls (the standard "ignore the wheel when not focused" rule).
            QApplication.sendEvent(self.scroll.verticalScrollBar(), event)
            return True
        if (obj is self.radius_spin and event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Escape):
            # Put back what is in force, typed digits and all, and hand the
            # keyboard back: Esc here answers the box, never the edit behind it.
            shown = int(self.radius_spin.value())
            blocked = self.radius_spin.blockSignals(True)
            self.radius_spin.setValue(shown)
            self.radius_spin.blockSignals(blocked)
            self.sigRadiusTyped.emit(shown)
            return True
        return super().eventFilter(obj, event)

    def _on_spin_finished(self) -> None:
        """``Enter`` (or the focus leaving): the value is in; give the keys back.

        Only while the box still has the keyboard -- that is ``Enter``.  When
        the focus has already gone somewhere the annotator clicked, it stays
        there.
        """
        if self.radius_spin.hasFocus():
            self.sigRadiusTyped.emit(int(self.radius_spin.value()))

    def _on_slider(self, value: int) -> None:
        radius = slider_to_radius(value)
        blocked = self.radius_spin.blockSignals(True)
        self.radius_spin.setValue(radius)
        self.radius_spin.blockSignals(blocked)
        self.sigRadius.emit(radius)

    def _on_spin(self, value: int) -> None:
        blocked = self.radius_slider.blockSignals(True)
        self.radius_slider.setValue(radius_to_slider(value))
        self.radius_slider.blockSignals(blocked)
        self.sigRadius.emit(int(value))
