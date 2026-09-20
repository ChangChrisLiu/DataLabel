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
from typing import Optional

import numpy as np
from PySide6.QtGui import QImage

from tda.core import masks as _masks

__all__ = [
    "PALETTE_64",
    "EDIT_RGB",
    "GHOST_RGB",
    "OCCLUDER_RGB",
    "OCCLUDER_TYPES",
    "LabelOverlay",
    "palette_color",
]

Rect = tuple[int, int, int, int]
RGB = tuple[int, int, int]

#: ``OccluderMask.occluder_type`` domain (spec 3.1).  Each type is a separate
#: layer, because the compiler stores (and subtracts) one mask per type.
OCCLUDER_TYPES: tuple[str, ...] = ("hand", "arm", "body", "tool", "cable", "other")


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
#: Colour of a proposal nobody has accepted yet -- today the Label Studio draft
#: on offer (``Shift+A``).  Neither a palette entry nor :data:`EDIT_RGB`: "this
#: is not yours until you press Enter" has to be visible at a glance.
GHOST_RGB: RGB = (96, 208, 255)

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
        ghost: boolean ``HxW`` preview layer -- a proposal being *shown*, never
            an edit; :attr:`has_ghost` says whether one is up.
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
        self.ghost = np.zeros((h, w), dtype=bool)
        #: Whether :attr:`ghost` is being shown.  A flag rather than
        #: ``ghost.any()``: clearing it runs on every commit, undo and frame
        #: change, and scanning 12 MP to find out there was nothing to clear
        #: would be a repaint nobody asked for.
        self.has_ghost = False
        self._ghost_rect: Optional[Rect] = None
        self.occluders: dict[str, np.ndarray] = {}
        self._visible = True

        # ARGB32 words; the QImage below shares this memory.
        self._buffer = np.zeros((h, w), dtype=np.uint32)
        self._image: Optional[QImage] = None
        self._style: Optional[tuple[int, bool]] = None  # (alpha, outline)
        self._luts: Optional[tuple[int, np.ndarray, np.ndarray]] = None
        self._dirty: object = _ALL
        #: Region rebuilt by the last :meth:`qimage` call (for tests/profiling).
        self.last_rebuild_rect: Optional[Rect] = None

    # -- content ------------------------------------------------------------
    @property
    def visible(self) -> bool:
        """Whether :meth:`qimage` draws anything at all.

        A property rather than a plain attribute: toggling it has to invalidate
        the cached buffer, or ``qimage()`` would hand back the last render.
        """
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        value = bool(value)
        if value != self._visible:
            self._visible = value
            self._dirty = _ALL

    def force_full_rebuild(self) -> None:
        """Mark the whole buffer stale (for a palette/alpha change or a test)."""
        self._dirty = _ALL

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

    def set_ghost(self, mask: np.ndarray, rect: Optional[Rect] = None) -> None:
        """Show a proposal over the frame without making it an edit.

        The ghost is drawn *under* the editing layer, so a proposal can never
        cover the pixels the annotator has already painted, and it takes part
        in nothing else: no label id, no undo entry, no commit.  Whoever put it
        up (:mod:`tda.ui.app_adopt`) is the only one who can turn it into pixels.

        ``rect`` is the region the proposal covers -- a caller that already
        knows it (from the draft's stored bounding box) hands it over and only
        that region is re-composited, together with whatever the previous ghost
        covered.  Without one the whole buffer is marked stale, which on a
        12 MP frame is a 123 ms repaint per keypress.
        """
        stale = _union(self._ghost_rect, rect) if rect is not None else None
        self.ghost = self._coerce(mask, "ghost mask")
        self.has_ghost = True
        self._ghost_rect = None if rect is None else tuple(int(v) for v in rect)
        self._mark(stale)

    def clear_ghost(self) -> None:
        """Take the proposal off the screen; a no-op when none is showing."""
        if not self.has_ghost:
            return
        stale, self._ghost_rect = self._ghost_rect, None
        self.ghost = np.zeros(self.hw, dtype=bool)
        self.has_ghost = False
        self._mark(stale)

    @property
    def ghost_rect(self) -> Optional[Rect]:
        """The region the ghost on screen covers, when its owner said so."""
        return self._ghost_rect

    def _mark(self, rect: Optional[Rect]) -> None:
        """Mark ``rect`` stale, or the whole buffer when the caller cannot say."""
        if rect is None:
            self._dirty = _ALL
            return
        self._dirty = _ALL if self._dirty is _ALL else _union(self._dirty, rect)

    def occluder_layer(self, occluder_type: str) -> np.ndarray:
        """The occluder layer of ``occluder_type``, created empty if needed."""
        if occluder_type not in OCCLUDER_TYPES:
            raise ValueError(
                f"unknown occluder_type {occluder_type!r}; expected one of "
                f"{OCCLUDER_TYPES}"
            )
        layer = self.occluders.get(occluder_type)
        if layer is None:
            layer = np.zeros(self.hw, dtype=bool)
            self.occluders[occluder_type] = layer
        return layer

    def set_occluder(self, mask: np.ndarray, occluder_type: str) -> None:
        """Replace the occluder layer of one type; others are untouched."""
        if occluder_type not in OCCLUDER_TYPES:
            raise ValueError(f"unknown occluder_type {occluder_type!r}")
        self.occluders[occluder_type] = self._coerce(mask, "occluder mask")
        self._dirty = _ALL

    def _coerce(self, mask: np.ndarray, what: str) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.shape != self.hw:
            raise ValueError(
                f"{what} has shape {arr.shape!r}, expected {self.hw!r}"
            )
        return np.array(arr, dtype=bool, copy=True)

    # -- painting -----------------------------------------------------------
    def paint(
        self, xy: tuple[int, int], radius: int, add: bool
    ) -> Optional[Rect]:
        """Stamp a filled circle into the editing layer; return the dirty rect.

        ``radius`` is measured in image pixels, so the stamp spans
        ``2 * radius + 1`` pixels.  The rect is clipped to the image; a stamp
        that falls entirely outside it changes nothing and returns ``None``.
        """
        return self._stamp(self.editing, xy, radius, add)

    def paint_occluder(
        self,
        xy: tuple[int, int],
        radius: int,
        add: bool,
        occluder_type: str = "hand",
    ) -> Optional[Rect]:
        """Stamp into the occluder layer of ``occluder_type`` (spec 4.6, ``O``).

        One layer per type: the compiler subtracts each type separately and the
        database keys ``occluder_mask`` on it, so strokes of two types must
        never land in the same array.
        """
        return self._stamp(
            self.occluder_layer(occluder_type), xy, radius, add
        )

    def _stamp(
        self, target: np.ndarray, xy: tuple[int, int], radius: int, add: bool
    ) -> Optional[Rect]:
        h, w = self.hw
        cx, cy = int(round(xy[0])), int(round(xy[1]))
        r = max(0, int(radius))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None

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
        whatever :meth:`paint` has marked dirty since the last call, and then
        grown by one pixel when ``outline`` is on -- editing a pixel changes
        whether its *neighbours* sit on a boundary, so a repaint limited to the
        edited rect would leave the old outline standing just outside it.
        :attr:`last_rebuild_rect` reports the region actually rendered (``None``
        when nothing was stale), which is what a caller should invalidate.

        Changing ``alpha`` or ``outline`` forces a full rebuild, since they
        apply everywhere.  The returned image always covers the whole frame and
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

        self.last_rebuild_rect = None
        if target is not None:
            target = self._clip(target)
            if outline:
                target = self._halo(target)
            self._render(target, style[0], style[1])
        self._dirty = None
        return self._image

    def _halo(self, rect: Rect) -> Rect:
        """Grow ``rect`` by one pixel, clipped to the image."""
        h, w = self.hw
        x0, y0, x1, y1 = rect
        return (max(0, x0 - 1), max(0, y0 - 1), min(w, x1 + 1), min(h, y1 + 1))

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
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            return
        self.last_rebuild_rect = rect
        if not self._visible:
            self._buffer[y0:y1, x0:x1] = 0
            return

        fill_lut, line_lut = self._label_luts(alpha)
        labels = self.labelmap[y0:y1, x0:x1]
        out = fill_lut[labels]
        if outline:
            edges = self._edges(self.labelmap, rect) & (labels != 0)
            np.copyto(out, line_lut[labels], where=edges)

        # A proposal goes under the editing layer: it is an offer, and the
        # annotator's own pixels outrank it.  Left out entirely while none is
        # up, so the ordinary repaint pays nothing for it.
        layers = [(self.ghost, GHOST_RGB)] if self.has_ghost else []
        layers += [(self.editing, EDIT_RGB)]
        # Occluders sit on top of the instance being edited: they mark what the
        # annotator cannot see, so they must not be hidden by it.
        layers += [
            (self.occluders[key], OCCLUDER_RGB) for key in sorted(self.occluders)
        ]
        for layer, rgb in layers:
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
