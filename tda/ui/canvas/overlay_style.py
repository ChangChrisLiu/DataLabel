"""How every outline drawn over the frame looks, in one place (task U2g).

The annotator's screen runs at a device pixel ratio of 1.5, and at that ratio
the canvas' dashed outlines were close to invisible ("很多虚线的部分很不明显"):
the stored ROI was one *device* pixel of yellow at alpha 150, on a chassis the
palette happened to paint olive-yellow, and the hover hints were bare 2 px
dashes with bare-text labels.  Two things made it worse than it looked in the
code, and both are fixed here rather than at each call site:

* **Cosmetic widths are device pixels.**  Measured on this Qt (6.11) at DPR
  1.5: a cosmetic pen of width 2 draws two device pixels, not three.  Every
  width below is in *logical* pixels and :meth:`OutlineStyle.over_pen` /
  :meth:`OutlineStyle.under_pen` multiply by the painter's ratio -- so an
  outline is as wide on the 150 % screen as it is at 100 %, and as wide at the
  29 % an OAK frame opens at as at 800 % (a cosmetic pen ignores the zoom).
* **One colour could vanish into another.**  Each overlay kind has its own
  colour and :func:`tda.ui.canvas.overlay._build_palette` leaves a band of
  :data:`RESERVED_HUE_HALF_WIDTH` degrees around each of them empty, so no
  instance mask can ever share an overlay's hue; the drag band is white, which
  the palette's saturation floor (0.62) cannot produce either.

The look itself is the one the ROI editor already had (ruling U-ROI-2): a
dark, solid under-stroke two logical pixels wider than a fully opaque bright
over-stroke, so the outline reads on the dark chassis *and* on the white scan
bed.  Dashes are a fixed length in logical pixels (:data:`DASH_PX` on,
:data:`GAP_PX` off, flat caps) whatever the width: Qt's own ``DashLine`` is
4:2 pen widths, which at one device pixel is a grey fuzz.

Labels are **chips** -- text on a dark rounded background with a border in the
outline's colour -- placed at the outlined box's top-left corner and kept
inside the viewport (:func:`place_chip`).
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Optional, Sequence

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap

__all__ = [
    "CHIP_BG", "CHIP_GAP", "CHIP_MARGIN", "CHIP_MAX_W", "CURSOR_RING_PX", "DASH_PX",
    "DRAFT_RGB", "DRAG_BAND", "DRAG_RGB", "GAP_PX", "HANDLE_EDGE", "HINT_PX",
    "OutlineStyle", "PROMPT_BOX", "PROMPT_POINT", "PROMPT_RGB", "RESERVED_HUE_HALF_WIDTH",
    "RESERVED_RGBS", "ROI_CHIP", "ROI_EDITING", "ROI_RGB", "ROI_STORED", "SHAPE_RGB",
    "UNDER_ALPHA", "UNDER_EXTRA_PX", "chip_pixmap", "draw_outline_rect",
    "hint_style", "hue_degrees", "hue_distance", "place_chip", "reserved_hue_bands",
]

RGB = tuple[int, int, int]

# --------------------------------------------------------------------------- #
# the colours: one per kind, none of them an instance colour
# --------------------------------------------------------------------------- #
#: The chassis range (ROI), stored and being edited alike -- it is one
#: rectangle.  Magenta: nothing in the rig is magenta (the scan bed is white,
#: the tape square yellow/orange, the chassis dark metal or painted by the
#: palette, boards green), it is none of the other overlays' colours, and it
#: is not the minimap's blue viewport frame, which the second trial took for
#: the ROI.  It used to be the editing layer's yellow, and on D13 the chassis
#: instance is palette olive-yellow (186, 179, 22): the stored outline sat on
#: exactly its own colour.
ROI_RGB: RGB = (255, 72, 240)
#: The difference map's box -- the SAM prompt box it arms, its "click here"
#: cross, and the hover hint that points at it.  One box, one colour.
PROMPT_RGB: RGB = (255, 150, 40)
#: Old Label Studio drafts: the hover hint *and* the ``Shift+A`` ghost
#: (:data:`tda.ui.canvas.overlay.GHOST_RGB` is this colour), so "a draft" is
#: one colour wherever it is shown.
DRAFT_RGB: RGB = (96, 208, 255)
#: The part's own shape on another keyframe of the segment (hover hint).
SHAPE_RGB: RGB = (80, 220, 120)
#: A box being dragged right now (SAM box ``X``, bench box ``R``).  White:
#: achromatic, so no palette colour -- saturation 0.62 or more -- comes near it.
DRAG_RGB: RGB = (255, 255, 255)

#: The hued overlay colours the instance palette keeps clear of.
RESERVED_RGBS: tuple[RGB, ...] = (ROI_RGB, PROMPT_RGB, DRAFT_RGB, SHAPE_RGB)
#: Half the width of the hue band left empty around each of them, degrees.
#: 15 keeps the nearest instance colour a clearly different hue (orange vs
#: amber, green vs lime) while leaving two thirds of the wheel to the palette.
RESERVED_HUE_HALF_WIDTH = 15.0

# --------------------------------------------------------------------------- #
# the strokes
# --------------------------------------------------------------------------- #
#: The dark under-stroke: black, this opaque, this much wider than the bright
#: stroke on top of it (one logical pixel of dark on each side).
UNDER_ALPHA = 215
UNDER_EXTRA_PX = 2.0
#: Dash and gap, logical pixels, whatever the stroke's width.
DASH_PX = 10.0
GAP_PX = 6.0


@dataclass(frozen=True)
class OutlineStyle:
    """One overlay kind's outline: colour, width (logical px), dashed or solid."""

    name: str
    rgb: RGB
    width: float
    dashed: bool = True

    @property
    def under_width(self) -> float:
        """The dark stroke's width, logical pixels."""
        return self.width + UNDER_EXTRA_PX

    def over_pen(self, dpr: float = 1.0) -> QPen:
        """The bright stroke: fully opaque, cosmetic, ``width`` logical px."""
        scale = max(1.0, float(dpr))
        pen = QPen(QColor(*self.rgb, 255), self.width * scale)
        pen.setCosmetic(True)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        if self.dashed:
            # In units of the pen's width, which is itself scaled: DASH_PX and
            # GAP_PX come out in logical pixels at any ratio.
            pen.setDashPattern([DASH_PX / self.width, GAP_PX / self.width])
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        return pen

    def under_pen(self, dpr: float = 1.0) -> QPen:
        """The dark stroke under it: solid, so the gaps read dark too."""
        scale = max(1.0, float(dpr))
        pen = QPen(QColor(0, 0, 0, UNDER_ALPHA), self.under_width * scale)
        pen.setCosmetic(True)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        return pen


#: A stored ROI: quieter than when it is being edited, still two tones.
ROI_STORED = OutlineStyle("roi", ROI_RGB, 2.0)
#: The ROI being edited (the proposal, ``Shift+R``): the editor's old 3 over 5.
ROI_EDITING = OutlineStyle("roi_editing", ROI_RGB, 3.0)
#: Hover hints (task card): the widest, they are only up while pointed at.
HINT_PX = 3.0
#: The SAM prompt box the difference map arms.
PROMPT_BOX = OutlineStyle("prompt", PROMPT_RGB, 2.5)
#: The prompt box's "click here" cross (``Shift+C``): solid, not dashed.
PROMPT_POINT = OutlineStyle("prompt_point", PROMPT_RGB, 2.0, dashed=False)
#: A drag in progress (the rubber band).
DRAG_BAND = OutlineStyle("drag", DRAG_RGB, 2.0)
#: The ROI handles' edge, solid black: logical, like everything else.
HANDLE_EDGE = OutlineStyle("handle_edge", (0, 0, 0), 1.0, dashed=False)
#: The brush/eraser ring (drawn into a cursor pixmap, logical px there too):
#: bright ring over a dark halo of ring + :data:`UNDER_EXTRA_PX`.
CURSOR_RING_PX = 1.8


def hint_style(rgb: Sequence[int]) -> OutlineStyle:
    """The outline of one hover hint in its source's colour."""
    return OutlineStyle("hint", (int(rgb[0]), int(rgb[1]), int(rgb[2])), HINT_PX)


def draw_outline_rect(painter: QPainter, rect: QRectF, style: OutlineStyle,
                      dpr: float = 1.0) -> None:
    """Stroke ``rect`` (in the painter's coordinates) in ``style``: dark, then bright."""
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(style.under_pen(dpr))
    painter.drawRect(rect)
    painter.setPen(style.over_pen(dpr))
    painter.drawRect(rect)


# --------------------------------------------------------------------------- #
# hues: what the palette has to stay away from
# --------------------------------------------------------------------------- #
def hue_degrees(rgb: Sequence[int]) -> float:
    """Hue of an RGB colour, degrees in [0, 360)."""
    h, _s, _v = colorsys.rgb_to_hsv(*(float(c) / 255.0 for c in rgb[:3]))
    return (h * 360.0) % 360.0


def hue_distance(a: float, b: float) -> float:
    """Shortest distance between two hues, degrees."""
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def reserved_hue_bands() -> list[tuple[float, float]]:
    """The hue bands no instance colour may fall in, as ``(lo, hi)`` degrees.

    Each band is :data:`RESERVED_HUE_HALF_WIDTH` either side of an overlay
    hue; ``lo`` may be negative or ``hi`` above 360 for a band that wraps.
    """
    return [(hue_degrees(rgb) - RESERVED_HUE_HALF_WIDTH,
             hue_degrees(rgb) + RESERVED_HUE_HALF_WIDTH) for rgb in RESERVED_RGBS]


# --------------------------------------------------------------------------- #
# chips: a label on a dark rounded background
# --------------------------------------------------------------------------- #
CHIP_BG = QColor(18, 20, 24, 235)
CHIP_PAD = (7.0, 3.0)
CHIP_RADIUS = 5.0
#: Between a chip and the outline it names, and between two chips.
CHIP_GAP = 3.0
#: How close to the viewport's edge a chip may sit.
CHIP_MARGIN = 4.0
#: Longer labels are elided (a merged "a + b + c" hint, a long draft key).
CHIP_MAX_W = 420.0
#: The ROI's own chip.
ROI_CHIP = "机箱范围 ROI"


def chip_font(base: QFont) -> QFont:
    """The chip's font: the widget's, bold, one point larger."""
    font = QFont(base)
    font.setBold(True)
    font.setPointSizeF(max(9.0, font.pointSizeF() + 1.0))
    return font


def chip_pixmap(text: str, rgb: Sequence[int], base_font: QFont, dpr: float) -> QPixmap:
    """``text`` in the overlay's colour on a dark rounded chip, at ``dpr``."""
    font = chip_font(base_font)
    metrics = QFontMetrics(font)
    padx, pady = CHIP_PAD
    shown = metrics.elidedText(str(text), Qt.TextElideMode.ElideRight,
                               int(CHIP_MAX_W - 2 * padx))
    w = float(metrics.horizontalAdvance(shown)) + 2 * padx
    h = float(metrics.height()) + 2 * pady
    scale = max(1.0, float(dpr))
    pixmap = QPixmap(int(round(w * scale)), int(round(h * scale)))
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    colour = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    painter.setPen(QPen(colour, 1.5))
    painter.setBrush(CHIP_BG)
    painter.drawRoundedRect(QRectF(0.75, 0.75, w - 1.5, h - 1.5), CHIP_RADIUS, CHIP_RADIUS)
    painter.setFont(font)
    painter.setPen(colour)
    painter.drawText(QRectF(padx, pady, w - 2 * padx, h - 2 * pady),
                     int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), shown)
    painter.end()
    return pixmap


def place_chip(size: tuple[float, float], box: QRectF, viewport: QRectF,
               taken: Sequence[QRectF] = (), gap: float = CHIP_GAP) -> Optional[QRectF]:
    """Where a ``size`` chip naming ``box`` goes, all in viewport coordinates.

    At the box's top-left corner, sitting just above its top edge (so it
    covers the outline's corner, not the part), or just inside the box when
    there is no room above.  The corner used is the box's *visible* top-left,
    and the chip is kept :data:`CHIP_MARGIN` inside the viewport, so a box
    whose corner is off-screen still has its label on screen.  A chip that
    would cover one already ``taken`` moves right along the same line, then
    down a line.  ``None`` when the box is not on screen at all -- a label
    for an outline nobody can see is a label pointing at nothing.
    """
    w, h = float(size[0]), float(size[1])
    box = QRectF(box.x(), box.y(), max(1.0, box.width()), max(1.0, box.height()))
    if not box.intersects(viewport):
        return None
    left = max(box.left(), viewport.left())
    top = max(box.top(), viewport.top())
    lo_x = viewport.left() + CHIP_MARGIN
    hi_x = viewport.right() - CHIP_MARGIN - w
    lo_y = viewport.top() + CHIP_MARGIN
    hi_y = viewport.bottom() - CHIP_MARGIN - h
    y = top - gap - h
    if box.top() < viewport.top() or y < lo_y:
        y = top + gap                      # no room above: just inside the box
    x = min(max(left, lo_x), max(lo_x, hi_x))
    y = min(max(y, lo_y), max(lo_y, hi_y))
    rect = QRectF(x, y, w, h)
    for _ in range(24):
        hit = next((r for r in taken if r.adjusted(-1, -1, 1, 1).intersects(rect)), None)
        if hit is None:
            break
        nx = hit.right() + CHIP_GAP
        if nx <= hi_x:
            rect.moveLeft(nx)
        else:
            rect.moveLeft(min(max(left, lo_x), max(lo_x, hi_x)))
            rect.moveTop(min(hit.bottom() + CHIP_GAP, max(lo_y, hi_y)))
    return rect
