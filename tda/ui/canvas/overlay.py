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
import weakref
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

#: How many separate stale regions are tracked before they are collapsed into
#: the one box that covers them all.  Collapsing only ever composites pixels
#: that were already correct, so it costs time and never an answer; the cap
#: stops a long pan from turning the stale list into a thousand slivers.
MAX_DIRTY_RECTS = 12


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


def _intersect(a: Rect, b: Rect) -> Optional[Rect]:
    """The region both rects cover, or ``None`` when they do not meet."""
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return None if box[2] <= box[0] or box[3] <= box[1] else box


def _subtract(a: Rect, b: Rect) -> list[Rect]:
    """``a`` minus ``b`` as up to four rects (the ring left around ``b``)."""
    keep = _intersect(a, b)
    if keep is None:
        return [a]
    ax0, ay0, ax1, ay1 = a
    kx0, ky0, kx1, ky1 = keep
    out: list[Rect] = []
    if ay0 < ky0:
        out.append((ax0, ay0, ax1, ky0))
    if ky1 < ay1:
        out.append((ax0, ky1, ax1, ay1))
    if ax0 < kx0:
        out.append((ax0, ky0, kx0, ky1))
    if kx1 < ax1:
        out.append((kx1, ky0, ax1, ky1))
    return out


def _bounds(rects: list[Rect]) -> Optional[Rect]:
    """The box covering every rect in the list (``None`` for an empty list)."""
    found: Optional[Rect] = None
    for rect in rects:
        found = _union(found, rect)
    return found


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
        #: A box the editing layer is empty outside, or ``None`` when it is
        #: empty altogether.  Kept as a *superset*: a stroke grows it and an
        #: erase leaves it alone, which is what makes "repaint where the layer
        #: was and where it is now" a cheap question rather than a 12 MP scan.
        self._editing_rect: Optional[Rect] = None
        self.ghost = np.zeros((h, w), dtype=bool)
        #: Whether :attr:`ghost` is being shown.  A flag rather than
        #: ``ghost.any()``: clearing it runs on every commit, undo and frame
        #: change, and scanning 12 MP to find out there was nothing to clear
        #: would be a repaint nobody asked for.
        self.has_ghost = False
        self._ghost_rect: Optional[Rect] = None
        self.occluders: dict[str, np.ndarray] = {}
        #: Per type, a box that layer is empty outside (see ``_editing_rect``).
        self._occluder_rects: dict[str, Optional[Rect]] = {}
        self._visible = True

        # ARGB32 words; the QImage below shares this memory.
        self._buffer = np.zeros((h, w), dtype=np.uint32)
        self._image: Optional[QImage] = None
        self._style: Optional[tuple[int, bool]] = None  # (alpha, outline)
        self._luts: Optional[tuple[int, np.ndarray, np.ndarray]] = None
        #: The regions of :attr:`_buffer` that do **not** hold what the layers
        #: say they should.  A list rather than one box because the canvas
        #: composites what it is about to draw and leaves the rest for when it
        #: is drawn: subtracting the viewport from "the whole frame" is a ring,
        #: and a ring is not a rectangle.
        self._dirty: list[Rect] = [(0, 0, w, h)]
        #: What the last :meth:`set_instances` painted: the keys in paint order
        #: and a weak reference to each mask.  Weak, so that the identity check
        #: cannot be fooled by a freed array's address being handed to the next
        #: one -- a dead reference simply reads as "this layer changed".
        self._layers: list[tuple[str, "weakref.ReferenceType"]] = []
        self._layer_windows: dict[str, Optional[Rect]] = {}
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
            self._mark_all()

    def force_full_rebuild(self) -> None:
        """Mark the whole buffer stale (for a palette/alpha change or a test).

        It also forgets what the layers were, so the next
        :meth:`set_instances` cannot decide it has nothing to do: this is the
        escape hatch for a caller that knows something changed behind the
        overlay's back, and it has to mean it.
        """
        self._layers = []
        self._mark_all()

    def set_instances(self, masks: dict[str, np.ndarray], order: list[str],
                      windows: Optional[dict[str, Optional[Rect]]] = None) -> None:
        """Rebuild the label map by painting ``masks`` bottom-to-top in ``order``.

        ``windows`` is the caller saying where each instance's pixels are --
        :attr:`CompiledInstance.window <tda.core.compiler.CompiledInstance>`,
        which the compiler has already measured.  It saves measuring the boxes
        again (140 ms per 12 MP frame change with forty instances) *and* it is
        what makes a change localisable: a layer that is not in the new set,
        or whose mask is a different array, dirties its own window and nobody
        else's.

        Two shortcuts, both resting on the same promise -- that a mask handed
        over here is not written into afterwards, which is true of everything
        the compiler produces:

        * the same keys in the same order with the same arrays means the
          picture cannot have changed, so nothing is repainted at all.  A
          commit repaints the layers twice (once in ``_commit``, once when the
          session re-announces the frame) and the second pass is free;
        * anything else repaints only the windows of the layers that moved --
          unless the *stacking* of the layers they share the canvas with
          changed, which is not a local question, or a window is missing,
          which is not a promise.
        """
        painted = [key for key in order if key in masks]
        fresh = [(key, weakref.ref(masks[key])) for key in painted]
        boxes = {key: (None if windows is None else windows.get(key))
                 for key in painted}
        if self._layers and self._same_layers(fresh):
            return
        stale = self._layer_dirty(fresh, boxes)
        self.labelmap, self.id2key = _masks.labelmap_from_masks(
            masks, order, self.hw, windows=boxes, out=self.labelmap
        )
        self.palette = {lab: palette_color(key) for lab, key in self.id2key.items()}
        self._luts = None
        self._layers = fresh
        self._layer_windows = boxes
        if stale is None:
            self._mark_all()
        else:
            for rect in stale:
                self._mark(rect)

    def _same_layers(self, fresh: list) -> bool:
        """Is this the very set of arrays, in the very order, already painted?"""
        if len(fresh) != len(self._layers):
            return False
        for (key, ref), (was_key, was_ref) in zip(fresh, self._layers):
            if key != was_key:
                return False
            mask = ref()
            if mask is None or was_ref() is not mask:
                return False
        return True

    def _layer_dirty(self, fresh: list,
                     boxes: dict[str, Optional[Rect]]) -> Optional[list[Rect]]:
        """The regions this layer set changes, or ``None`` for "all of it".

        ``None`` is returned whenever the change cannot be localised: nothing
        was painted before, a window is missing, or the relative order of the
        layers that survive has moved -- which instance wins a pixel is a
        question about the whole stack, not about one box.
        """
        if not self._layers:
            return None
        was = {key: ref for key, ref in self._layers}
        now = {key: ref for key, ref in fresh}
        kept_now = [key for key, _ref in fresh if key in was]
        kept_was = [key for key, _ref in self._layers if key in now]
        if kept_now != kept_was:
            return None
        rects: list[Rect] = []
        for key in set(was) | set(now):
            here, there = was.get(key), now.get(key)
            if here is not None and there is not None:
                mask = there()
                if mask is not None and here() is mask:
                    continue
            if here is not None:
                if self._layer_windows.get(key) is None:
                    return None
                rects.append(self._layer_windows[key])  # type: ignore[arg-type]
            if there is not None:
                if boxes.get(key) is None:
                    return None
                rects.append(boxes[key])  # type: ignore[arg-type]
        return rects

    def set_editing(self, instance: str, mask: np.ndarray,
                    rect: Optional[Rect] = None) -> None:
        """Replace the editing layer (a separate bool layer drawn on top).

        ``rect`` is a caller that already knows where the new layer's pixels
        are saying so; without one the box is measured, which is two ``any``
        reductions against a whole-frame composite.  What is repainted is that
        box together with whatever the previous layer covered -- a SAM result,
        a restored sidecar and an undo all land here, and at 12 MP marking the
        whole buffer stale for a mask the size of a fan cost 95 ms a time.
        """
        arr = self._checked(mask, "editing mask")
        np.copyto(self.editing, arr)
        was, self._editing_rect = self._editing_rect, (
            _masks.bbox(arr) if rect is None else self._clip(rect)
        )
        self.editing_instance = instance
        self._mark_maybe(_union(was, self._editing_rect))

    def clear_editing(self) -> None:
        """Drop the editing layer without touching the committed label map."""
        was, self._editing_rect = self._editing_rect, None
        if was is not None:
            x0, y0, x1, y1 = was
            self.editing[y0:y1, x0:x1] = False
        self.editing_instance = None
        self._mark_maybe(was)

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
        # Where the *previous* proposal was matters as much as where this one
        # goes: one that was put up without a rect can be anywhere, so knowing
        # this one's box says nothing about taking that one off the screen.
        unknown = rect is None or (self.has_ghost and self._ghost_rect is None)
        stale = None if unknown else _union(self._ghost_rect, rect)
        self.ghost = self._coerce(mask, "ghost mask")
        self.has_ghost = True
        self._ghost_rect = None if rect is None else self._clip(rect)
        self._mark_unknown(stale)

    def clear_ghost(self) -> None:
        """Take the proposal off the screen; a no-op when none is showing."""
        if not self.has_ghost:
            return
        stale, self._ghost_rect = self._ghost_rect, None
        self.ghost = np.zeros(self.hw, dtype=bool)
        self.has_ghost = False
        self._mark_unknown(stale)

    @property
    def ghost_rect(self) -> Optional[Rect]:
        """The region the ghost on screen covers, when its owner said so."""
        return self._ghost_rect

    # -- what the buffer no longer holds ------------------------------------
    def _mark(self, rect: Rect) -> None:
        """Mark a region of the buffer stale, plus the one-pixel outline halo.

        The halo is added **here**, where a change is recorded, rather than
        where one is repaired: changing a pixel changes whether its
        *neighbours* sit on a boundary, and a region that has already been
        composited needs no ring around it.  Growing at repair time instead
        re-composited a one-pixel seam every time a clipped repaint left a
        remainder behind -- on every pan, for ever.
        """
        box = self._clip(self._halo(rect))
        if box is None:
            return
        self._dirty.append(box)
        if len(self._dirty) > MAX_DIRTY_RECTS:
            bounds = _bounds(self._dirty)
            self._dirty = [] if bounds is None else [bounds]

    def _mark_all(self) -> None:
        """Mark the whole buffer stale."""
        h, w = self.hw
        self._dirty = [(0, 0, w, h)]

    def _mark_maybe(self, rect: Optional[Rect]) -> None:
        """Mark ``rect``; ``None`` here means *nothing changed*."""
        if rect is not None:
            self._mark(rect)

    def _mark_unknown(self, rect: Optional[Rect]) -> None:
        """Mark ``rect``; ``None`` here means *the caller could not say where*."""
        if rect is None:
            self._mark_all()
        else:
            self._mark(rect)

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

    def set_occluder(self, mask: np.ndarray, occluder_type: str,
                     rect: Optional[Rect] = None) -> None:
        """Replace the occluder layer of one type; others are untouched.

        Like :meth:`set_editing`, only the box the layer used to cover and the
        one it covers now are repainted; ``rect`` is a caller that knows the
        latter saying so, and without one it is measured.
        """
        if occluder_type not in OCCLUDER_TYPES:
            raise ValueError(f"unknown occluder_type {occluder_type!r}")
        arr = self._coerce(mask, "occluder mask")
        self.occluders[occluder_type] = arr
        was = self._occluder_rects.get(occluder_type)
        self._occluder_rects[occluder_type] = (
            _masks.bbox(arr) if rect is None else self._clip(rect)
        )
        self._mark_maybe(_union(was, self._occluder_rects[occluder_type]))

    def _coerce(self, mask: np.ndarray, what: str) -> np.ndarray:
        return np.array(self._checked(mask, what), dtype=bool, copy=True)

    def _checked(self, mask: np.ndarray, what: str) -> np.ndarray:
        """``mask`` as a bool array of this overlay's shape (no copy made)."""
        arr = np.asarray(mask)
        if arr.shape != self.hw:
            raise ValueError(
                f"{what} has shape {arr.shape!r}, expected {self.hw!r}"
            )
        return arr if arr.dtype == bool else arr.astype(bool)

    # -- painting -----------------------------------------------------------
    def paint(
        self, xy: tuple[int, int], radius: int, add: bool
    ) -> Optional[Rect]:
        """Stamp a filled circle into the editing layer; return the dirty rect.

        ``radius`` is measured in image pixels, so the stamp spans
        ``2 * radius + 1`` pixels.  The rect is clipped to the image; a stamp
        that falls entirely outside it changes nothing and returns ``None``.
        """
        rect = self._stamp(self.editing, xy, radius, add)
        if rect is not None and add:
            self._editing_rect = _union(self._editing_rect, rect)
        return rect

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
        rect = self._stamp(self.occluder_layer(occluder_type), xy, radius, add)
        if rect is not None and add:
            self._occluder_rects[occluder_type] = _union(
                self._occluder_rects.get(occluder_type), rect
            )
        return rect

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
        self._mark(rect)
        return rect

    # -- rendering ----------------------------------------------------------
    def qimage(
        self, rect: Optional[Rect] = None, alpha: int = 110, outline: bool = True,
        clip: Optional[Rect] = None,
    ) -> QImage:
        """Full-size ``Format_ARGB32`` image of label map + edit layers.

        Only the stale regions are recomputed: ``rect`` (when given) added to
        whatever the layers and :meth:`paint` have marked stale since the last
        call, and then grown by one pixel when ``outline`` is on -- editing a
        pixel changes whether its *neighbours* sit on a boundary, so a repaint
        limited to the edited rect would leave the old outline standing just
        outside it.  :attr:`last_rebuild_rect` reports the box the rebuilt
        regions fit in (``None`` when nothing was stale), which is what a
        caller should invalidate.

        ``clip`` is the caller saying which part of the frame it is about to
        draw: only the stale regions that meet it are composited and the rest
        stays stale, to be paid for when it is looked at.  That is what keeps a
        12 MP frame affordable -- at 59 % zoom the viewport holds a sixth of
        it, and compositing the other five sixths is 90 ms per gesture for
        pixels nobody can see.  Whatever is composited is composited exactly as
        a whole-frame pass would have: a region is rendered from the label map
        and the layers alone, never from its neighbours.

        Changing ``alpha`` or ``outline`` makes the whole buffer stale, since
        they apply everywhere.  The returned image always covers the whole
        frame and shares this object's buffer, so it must not outlive the
        overlay.
        """
        style = (int(alpha), bool(outline))
        h, w = self.hw

        if self._image is None or style != self._style:
            self._image = QImage(
                self._buffer.data, w, h, QImage.Format.Format_ARGB32
            )
            self._style = style
            self._mark_all()
        elif rect is not None:
            self._mark(rect)

        limit: Optional[Rect] = (0, 0, w, h) if clip is None else self._clip(clip)
        stale, rendered = self._dirty, []
        self._dirty = []
        for region in stale:
            part = None if limit is None else _intersect(region, limit)
            if part is None:
                self._dirty.append(region)
                continue
            rendered.append(part)
            self._dirty.extend(_subtract(region, limit))
        # One composite over the box the stale parts fit in, rather than one
        # per region: a repaint pays a fixed price in edge padding and slicing,
        # and the box can never be larger than the clip the caller named.  With
        # no clip this is exactly the single union rect the overlay always
        # rebuilt.
        target = _bounds(rendered)
        if target is not None:
            self._render(target, style[0], style[1])
        if len(self._dirty) > MAX_DIRTY_RECTS:
            bounds = _bounds(self._dirty)
            self._dirty = [] if bounds is None else [bounds]
        self.last_rebuild_rect = target
        return self._image

    def _halo(self, rect: Rect) -> Rect:
        """Grow ``rect`` by one pixel, clipped to the image."""
        h, w = self.hw
        x0, y0, x1, y1 = rect
        return (max(0, x0 - 1), max(0, y0 - 1), min(w, x1 + 1), min(h, y1 + 1))

    def _clip(self, rect: Rect) -> Optional[Rect]:
        """``rect`` inside the image, or ``None`` when none of it is."""
        h, w = self.hw
        x0, y0, x1, y1 = (int(v) for v in rect)
        box = (min(max(x0, 0), w), min(max(y0, 0), h),
               min(max(x1, 0), w), min(max(y1, 0), h))
        return None if box[2] <= box[0] or box[3] <= box[1] else box

    def _render(self, rect: Rect, alpha: int, outline: bool) -> None:
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            return
        if not self._visible:
            self._buffer[y0:y1, x0:x1] = 0
            return

        fill_lut, line_lut = self._label_luts(alpha)
        labels = self.labelmap[y0:y1, x0:x1]
        # Straight into the shared buffer: building the answer in a fresh array
        # and copying it over was a second 48 MB pass per 12 MP composite.
        out = self._buffer[y0:y1, x0:x1]
        out[...] = fill_lut[labels]
        if outline:
            edges = self._edges(self.labelmap, rect)
            edges &= labels != 0
            # Only the boundary pixels are looked up, rather than the whole
            # rect being re-coloured and then masked back.
            out[edges] = line_lut[labels[edges]]

        # A proposal goes under the editing layer: it is an offer, and the
        # annotator's own pixels outrank it.  Left out entirely while none is
        # up, so the ordinary repaint pays nothing for it.
        layers = [(self.ghost, GHOST_RGB, self._ghost_rect)] if self.has_ghost else []
        layers += [(self.editing, EDIT_RGB, self._editing_rect)]
        # Occluders sit on top of the instance being edited: they mark what the
        # annotator cannot see, so they must not be hidden by it.
        layers += [
            (self.occluders[key], OCCLUDER_RGB, self._occluder_rects.get(key))
            for key in sorted(self.occluders)
        ]
        for layer, rgb, where in layers:
            # ``where`` is a box the layer is empty outside; when it misses
            # this rect there is nothing here and the ``any`` -- a full pass
            # over the rect, three times per composite -- is not made.
            if where is not None and _intersect(where, rect) is None:
                continue
            window = layer[y0:y1, x0:x1]
            if not window.any():
                continue
            out[window] = _argb(rgb, alpha)
            if outline:
                out[self._edges(layer, rect) & window] = _argb(rgb, 255)

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
        # One scratch array instead of seven: at 12 MP the four comparisons and
        # the three unions used to allocate (and free) 84 MB per composite.
        edges = np.not_equal(centre, padded[:-2, 1:-1])
        scratch = np.empty_like(edges)
        for other in (padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
            np.not_equal(centre, other, out=scratch)
            edges |= scratch
        iy, ix = y0 - ey0, x0 - ex0
        return edges[iy : iy + (y1 - y0), ix : ix + (x1 - x0)]
