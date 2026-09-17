"""Label overlay: uint16 label map + edit layers rendered to an ARGB QImage.

Rendering model (spec 10.2): one ``uint16`` label map holds the *committed*
visible masks of a frame, the instance being edited lives in its own boolean
array on top, and frame-level occluders in a third.  Colour comes from a fixed
64-entry palette indexed by a CRC of the instance key, so an instance keeps its
colour across frames, sessions and machines without any stored state.

Compositing is a palette lookup over the label map plus boolean-mask writes --
pure numpy, no per-pixel Python.  The ARGB buffer is allocated once and kept on
the object; :meth:`LabelOverlay.qimage` returns a ``QImage`` that *shares* that
buffer, which is what makes dirty-rect repaints cheap: a brush stroke rebuilds
only the disk it touched instead of the whole frame.
"""
from __future__ import annotations

import colorsys
import zlib
from typing import Literal, Optional

import numpy as np
from PySide6.QtGui import QImage

from tda.core import masks as _masks

__all__ = [
    "PALETTE_64",
    "EDIT_RGB",
    "OCCLUDER_RGB",
    "LabelOverlay",
    "palette_color",
]

Rect = tuple[int, int, int, int]
RGB = tuple[int, int, int]
Layer = Literal["editing", "occluder"]


def _build_palette() -> tuple[RGB, ...]:
    """64 distinct, evenly separated colours.

    Hues advance by the golden ratio so that neighbouring indices -- and any
    two instances of one frame -- land far apart on the wheel; saturation and
    value alternate on a 2x2 pattern to keep the colours apart when printed or
    seen by a colour-deficient eye.
    """
    colors: list[RGB] = []
    for i in range(64):
        hue = (i * 0.6180339887498949) % 1.0
        sat = 0.62 + 0.26 * ((i >> 1) & 1)
        val = 0.95 - 0.22 * (i & 1)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((round(r * 255), round(g * 255), round(b * 255)))
    return tuple(colors)


#: Fixed instance palette; see :func:`palette_color`.
PALETTE_64: tuple[RGB, ...] = _build_palette()

#: Highlight colour of the instance currently being edited (spec 4.3): one
#: fixed colour, never a palette entry, so "what am I painting" is unambiguous.
EDIT_RGB: RGB = (255, 232, 64)
#: Frame-level occluder layer colour (spec 3.1 ``OccluderMask``).
OCCLUDER_RGB: RGB = (255, 72, 72)

_ALL = "all"  # sentinel: the whole buffer is stale


def palette_color(key: str) -> RGB:
    """Stable colour for an instance key.

    ``crc32`` is used rather than :func:`hash` because Python's string hash is
    randomised per process -- an instance would change colour every launch.
    """
    return PALETTE_64[zlib.crc32(key.encode("utf-8")) % len(PALETTE_64)]


def _union(a: Optional[Rect], b: Optional[Rect]) -> Optional[Rect]:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _argb(rgb: RGB, alpha: int) -> np.uint32:
    """Pack ``(r, g, b)`` + alpha into one ``Format_ARGB32`` word."""
    r, g, b = rgb
    return np.uint32(((int(alpha) & 0xFF) << 24) | (r << 16) | (g << 8) | b)


class LabelOverlay:
    """Composited label/edit layers of one frame, renderable as a ``QImage``.

    Attributes:
        labelmap: ``uint16`` ``HxW``; 0 is background, ids index :attr:`palette`.
        palette: label id -> RGB, derived from the instance keys.
        id2key: label id -> instance key (inverse of the painting order).
        editing: boolean ``HxW`` layer of the instance under the cursor.
        occluder: boolean ``HxW`` frame occluder layer.
        visible: when False :meth:`qimage` renders fully transparent, which is
            how the "hide masks" shortcut works without touching any data.
    """

    def __init__(self, hw: tuple[int, int]) -> None:
        h, w = int(hw[0]), int(hw[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"overlay size must be positive, got {(h, w)!r}")
        self.hw: tuple[int, int] = (h, w)
        self.labelmap = np.zeros((h, w), dtype=np.uint16)
        self.palette: dict[int, RGB] = {}
        self.id2key: dict[int, str] = {}
        self.editing_instance: Optional[str] = None
        self.editing = np.zeros((h, w), dtype=bool)
        self.occluder = np.zeros((h, w), dtype=bool)
        self.visible = True

        # ARGB32 words; the QImage below shares this memory.
        self._buffer = np.zeros((h, w), dtype=np.uint32)
        self._image: Optional[QImage] = None
        self._style: Optional[tuple[int, bool]] = None  # (alpha, outline)
        self._luts: Optional[tuple[int, np.ndarray, np.ndarray]] = None
        self._dirty: object = _ALL
        #: Region rebuilt by the last :meth:`qimage` call (for tests/profiling).
        self.last_rebuild_rect: Optional[Rect] = None

    # -- content ------------------------------------------------------------
    def set_instances(self, masks: dict[str, np.ndarray], order: list[str]) -> None:
        """Rebuild the label map by painting ``masks`` bottom-to-top in ``order``."""
        self.labelmap, self.id2key = _masks.labelmap_from_masks(masks, order, self.hw)
        self.palette = {lab: palette_color(key) for lab, key in self.id2key.items()}
        self._luts = None
        self._dirty = _ALL

    def set_editing(self, instance: str, mask: np.ndarray) -> None:
        """Replace the editing layer (a separate bool layer drawn on top)."""
        self.editing = self._coerce(mask, "editing mask")
        self.editing_instance = instance
        self._dirty = _ALL

    def clear_editing(self) -> None:
        """Drop the editing layer without touching the committed label map."""
        self.editing = np.zeros(self.hw, dtype=bool)
        self.editing_instance = None
        self._dirty = _ALL

    def set_occluder(self, mask: np.ndarray) -> None:
        """Replace the frame occluder layer."""
        self.occluder = self._coerce(mask, "occluder mask")
        self._dirty = _ALL

    def _coerce(self, mask: np.ndarray, what: str) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.shape != self.hw:
            raise ValueError(
                f"{what} has shape {arr.shape!r}, expected {self.hw!r}"
            )
        return np.array(arr, dtype=bool, copy=True)

    def layer(self, name: Layer) -> np.ndarray:
        """The boolean layer ``name`` ("editing" or "occluder")."""
        if name == "editing":
            return self.editing
        if name == "occluder":
            return self.occluder
        raise ValueError(f"unknown layer {name!r}")

    # -- painting -----------------------------------------------------------
    def paint(
        self,
        xy: tuple[int, int],
        radius: int,
        add: bool,
        layer: Layer = "editing",
    ) -> Rect:
        """Stamp a filled circle into a boolean layer; return the dirty rect.

        ``radius`` is measured in image pixels, so the stamp spans
        ``2 * radius + 1`` pixels.  The returned rect is clipped to the image
        and may be empty when the stamp falls entirely outside it.
        """
        target = self.layer(layer)
        h, w = self.hw
        cx, cy = int(round(xy[0])), int(round(xy[1]))
        r = max(0, int(radius))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            cx, cy = min(max(cx, 0), w), min(max(cy, 0), h)
            return (cx, cy, cx, cy)

        dy = np.arange(y0, y1, dtype=np.int32)[:, None] - cy
        dx = np.arange(x0, x1, dtype=np.int32)[None, :] - cx
        disk = (dx * dx + dy * dy) <= r * r
        window = target[y0:y1, x0:x1]
        if add:
            window |= disk
        else:
            window &= ~disk

        rect = (x0, y0, x1, y1)
        self._dirty = _ALL if self._dirty is _ALL else _union(self._dirty, rect)
        return rect

    # -- rendering ----------------------------------------------------------
    def qimage(
        self, rect: Optional[Rect] = None, alpha: int = 110, outline: bool = True
    ) -> QImage:
        """Full-size ``Format_ARGB32`` image of label map + edit layers.

        Only the stale region is recomputed: ``rect`` (when given) unioned with
        whatever :meth:`paint` has marked dirty since the last call.  Changing
        ``alpha`` or ``outline`` forces a full rebuild, since they apply
        everywhere.  The returned image always covers the whole frame and
        shares this object's buffer, so it must not outlive the overlay.
        """
        style = (int(alpha), bool(outline))
        h, w = self.hw
        full: Rect = (0, 0, w, h)

        if self._image is None or style != self._style:
            self._image = QImage(
                self._buffer.data, w, h, QImage.Format.Format_ARGB32
            )
            self._style = style
            target: Optional[Rect] = full
        elif self._dirty is _ALL:
            target = full
        else:
            target = _union(rect, self._dirty)  # type: ignore[arg-type]

        if target is not None:
            self._render(self._clip(target), style[0], style[1])
        self._dirty = None
        return self._image

    def _clip(self, rect: Rect) -> Rect:
        h, w = self.hw
        x0, y0, x1, y1 = rect
        return (
            min(max(int(x0), 0), w),
            min(max(int(y0), 0), h),
            min(max(int(x1), 0), w),
            min(max(int(y1), 0), h),
        )

    def _render(self, rect: Rect, alpha: int, outline: bool) -> None:
        self.last_rebuild_rect = rect
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            return
        if not self.visible:
            self._buffer[y0:y1, x0:x1] = 0
            return

        fill_lut, line_lut = self._label_luts(alpha)
        labels = self.labelmap[y0:y1, x0:x1]
        out = fill_lut[labels]
        if outline:
            edges = self._edges(self.labelmap, rect) & (labels != 0)
            np.copyto(out, line_lut[labels], where=edges)

        for layer, rgb in ((self.editing, EDIT_RGB), (self.occluder, OCCLUDER_RGB)):
            window = layer[y0:y1, x0:x1]
            if not window.any():
                continue
            out[window] = _argb(rgb, alpha)
            if outline:
                out[self._edges(layer, rect) & window] = _argb(rgb, 255)

        self._buffer[y0:y1, x0:x1] = out

    def _label_luts(self, alpha: int) -> tuple[np.ndarray, np.ndarray]:
        """``(fill, outline)`` ARGB lookup tables indexed by label id.

        Cached: a dirty-rect repaint must not pay for rebuilding the tables,
        and they only change when the palette or the alpha does.
        """
        if self._luts is not None and self._luts[0] == alpha:
            return self._luts[1], self._luts[2]
        size = max(self.palette, default=0) + 1
        fill = np.zeros(size, dtype=np.uint32)
        line = np.zeros(size, dtype=np.uint32)
        for label, rgb in self.palette.items():
            fill[label] = _argb(rgb, alpha)
            line[label] = _argb(rgb, 255)
        self._luts = (alpha, fill, line)
        return fill, line

    def _edges(self, layer: np.ndarray, rect: Rect) -> np.ndarray:
        """Pixels of ``rect`` whose 4-neighbourhood holds a different value.

        The slice is grown by one pixel before the comparison so that an
        outline computed for a dirty rect matches the one a full rebuild would
        draw; outside the image the value counts as background, which makes an
        instance touching the frame border outlined there too.
        """
        h, w = self.hw
        x0, y0, x1, y1 = rect
        ex0, ey0 = max(0, x0 - 1), max(0, y0 - 1)
        ex1, ey1 = min(w, x1 + 1), min(h, y1 + 1)
        grown = layer[ey0:ey1, ex0:ex1]
        padded = np.zeros(
            (grown.shape[0] + 2, grown.shape[1] + 2), dtype=grown.dtype
        )
        padded[1:-1, 1:-1] = grown
        centre = padded[1:-1, 1:-1]
        edges = (
            (centre != padded[:-2, 1:-1])
            | (centre != padded[2:, 1:-1])
            | (centre != padded[1:-1, :-2])
            | (centre != padded[1:-1, 2:])
        )
        iy, ix = y0 - ey0, x0 - ex0
        return edges[iy : iy + (y1 - y0), ix : ix + (x1 - x0)]
