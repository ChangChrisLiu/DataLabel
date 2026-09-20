"""Mask toolbox for the Teardown Annotator (pure numpy/cv2, no Qt).

Conventions used across ``tda``:

* A *mask* is a ``np.ndarray`` of dtype ``bool`` with shape ``(H, W)``.
* A *COCO RLE* is ``{"size": [h, w], "counts": str}`` -- ``counts`` is an
  ASCII ``str`` (not ``bytes``) so that it survives a JSON/SQLite round trip.
* A *box* is ``(x0, y0, x1, y1)`` with ``x1``/``y1`` exclusive, like slicing.
* A *polygon* is a flat list ``[x0, y0, x1, y1, ...]`` of floats (COCO style).
* ``Similarity(scale, theta, tx, ty)`` means ``x' = s * R(theta) * x + t``
  about the image origin, with ``theta`` in **radians**, y pointing down.

This module is consumed by the layer compiler, the truth table and the
canvas overlay.  Everything here is deterministic -- the same arguments always
give the same answer -- but three things are not side-effect free:

* :func:`paste` writes into its destination in place;
* :func:`decode_rle_shared` keeps a bounded, content-keyed memo of decoded
  shapes (:data:`DECODE_CACHE_BYTES`, emptied by :func:`clear_decode_cache`);
* :func:`encode_rle` with a ``window`` builds the run lengths in a scratch
  canvas this thread keeps, and puts it back to empty afterwards.

None of the three can change an answer; they only decide how much work it
takes to arrive at it.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from tda.core.model import Similarity

__all__ = [
    "CHECK_ENCODE_WINDOW",
    "DECODE_CACHE_BYTES",
    "clear_decode_cache",
    "decode_cache_stats",
    "encode_rle",
    "encode_rle_boxed",
    "decode_rle",
    "decode_rle_shared",
    "rle_area",
    "rle_bbox_xywh",
    "rle_counts",
    "bbox",
    "min_side",
    "rle_min_side",
    "area",
    "fill_holes",
    "remove_small_components",
    "tolerant_sym_diff",
    "is_conflict",
    "mask_to_polygons",
    "polygons_to_mask",
    "warp_mask",
    "crop",
    "paste",
    "labelmap_from_masks",
]

Box = tuple[int, int, int, int]
HW = tuple[int, int]


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _as_bool(mask: np.ndarray) -> np.ndarray:
    """Coerce any 2-D array-like to a boolean mask (no copy when possible)."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {arr.shape!r}")
    return arr if arr.dtype == bool else arr.astype(bool)


def _as_u8(mask: np.ndarray) -> np.ndarray:
    """Contiguous uint8 view of a mask, 0/1 valued -- what cv2 wants."""
    return np.ascontiguousarray(_as_bool(mask).astype(np.uint8))


def _square_kernel(size: int) -> np.ndarray:
    return np.ones((size, size), np.uint8)


def _boundary(mask_u8: np.ndarray) -> np.ndarray:
    """Inner boundary: the pixels of ``mask`` that touch the background.

    Computed as ``mask - erode(mask, 3x3)``, so it is one pixel thick and is
    always a subset of the mask itself.
    """
    eroded = cv2.erode(mask_u8, _square_kernel(3), borderValue=0)
    return cv2.subtract(mask_u8, eroded)


# ---------------------------------------------------------------------------
# RLE
# ---------------------------------------------------------------------------


#: One reusable Fortran-ordered scratch canvas per thread, for the windowed
#: encode below. Per thread because the truth sweeper encodes on its own one.
_scratch = threading.local()


#: Check every ``encode_rle(mask, window)`` call's promise that the mask really
#: is empty outside its window.
#:
#: A window that does not contain the mask silently truncates stored geometry,
#: which is the one way this optimisation could lose an annotation, and the
#: check is four ``any`` passes over the bands outside the window -- cheap
#: enough for a test run, not free enough for a 12 MP commit. So it is **on in
#: the test suite** (``tests/conftest.py`` sets it, which makes every call site
#: in the codebase guarded on every run) and off in the annotator, and
#: ``TDA_CHECK_ENCODE_WINDOW=1`` turns it on anywhere else.
CHECK_ENCODE_WINDOW = os.environ.get("TDA_CHECK_ENCODE_WINDOW", "") not in ("", "0")


def _refuse_pixels_outside(arr: np.ndarray, window: Box) -> None:
    """Raise when ``arr`` has a set pixel outside ``window`` (:data:`CHECK_ENCODE_WINDOW`)."""
    x0, y0, x1, y1 = window
    height, width = arr.shape
    x0, y0 = max(0, min(x0, width)), max(0, min(y0, height))
    x1, y1 = max(x0, min(x1, width)), max(y0, min(y1, height))
    for band in (arr[:y0, :], arr[y1:, :], arr[y0:y1, :x0], arr[y0:y1, x1:]):
        if band.size and band.any():
            raise ValueError(
                f"encode_rle was given the window {tuple(window)!r}, but the "
                f"mask has pixels outside it: the run lengths would be missing "
                f"them. Only a caller that produced the mask may pass a window."
            )


def _encode_buffer(hw: HW) -> np.ndarray:
    buffer = getattr(_scratch, "buffer", None)
    if buffer is None or buffer.shape != tuple(hw):
        buffer = np.zeros((int(hw[0]), int(hw[1])), dtype=np.uint8, order="F")
        _scratch.buffer = buffer
    return buffer


def encode_rle(mask: np.ndarray, window: Optional[Box] = None) -> dict:
    """Encode a boolean mask as a COCO RLE dict with ``str`` counts.

    Two copies used to happen on the way in and both are gone; the bytes that
    come out are the same ones, which
    :mod:`tests.test_compiler_golden` pins.

    * ``.view(np.uint8)`` instead of ``.astype(np.uint8)``. A NumPy ``bool_``
      **is** one byte holding 0 or 1, so reinterpreting it costs nothing, while
      the cast copied the whole canvas -- 3 ms per instance at 12 MP. It is a
      reinterpretation, so it relies on the bytes really being 0/1; every mask
      in ``tda`` is produced either by a NumPy operation or by
      :func:`decode_rle`, and both guarantee that.
    * the caller is expected to hand over a **Fortran-ordered** mask.
      ``coco_mask.encode`` reads column-major, so a C-ordered array has to be
      transposed into a new buffer first -- 39 ms per instance at 12 MP, which
      was the single most expensive thing a commit did. :func:`decode_rle`
      returns Fortran order and :func:`tda.core.compiler.compile_frame` builds
      its canvases in it, so the copy below is skipped on everything the truth
      table stores. A C-ordered mask still works; it just pays for the
      transpose.

    ``window`` is the caller **promising** that ``mask`` is empty outside
    ``(x0, y0, x1, y1)`` -- which the compiler knows, because a shape is only
    ever painted inside its own bounding box. The run lengths are then built in
    a scratch canvas this thread keeps, so the encode reads memory that is
    already resident instead of faulting in a freshly allocated 12 MB one:
    5.6 ms per instance becomes 2.7 ms, and at forty instances that is a tenth
    of the commit budget. The answer is the same either way -- but a window that
    does not contain the mask silently truncates it, so only a caller that
    produced the mask may pass one.
    """
    arr = _as_bool(mask)
    if window is None or arr.size == 0:
        rle = coco_mask.encode(np.asfortranarray(arr.view(np.uint8)))
    else:
        if CHECK_ENCODE_WINDOW:
            _refuse_pixels_outside(arr, window)
        x0, y0, x1, y1 = (int(v) for v in window)
        buffer = _encode_buffer(arr.shape)
        try:
            if x1 > x0 and y1 > y0:
                buffer[y0:y1, x0:x1] = arr[y0:y1, x0:x1].view(np.uint8)
            rle = coco_mask.encode(buffer)
        finally:
            # the canvas is reused, so it goes back to empty whatever happened:
            # a half-written buffer would poison every later encode
            if x1 > x0 and y1 > y0:
                buffer[y0:y1, x0:x1] = 0
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].decode("ascii"),
    }


def encode_rle_boxed(mask: np.ndarray) -> dict:
    """:func:`encode_rle` with the window measured off the mask itself.

    For a caller holding a full-canvas mask that does not know where its pixels
    are -- the editing layer, the crash sidecar, an undo record, an occluder, a
    frame override. Measuring the box is two ``any`` reductions (0.5 ms at
    12 MP) and it saves transposing and scanning the whole canvas (40 ms), so
    it pays for itself by eighty to one on an OAK frame and costs nothing
    measurable on a scanner one. The bytes out are :func:`encode_rle`'s.
    """
    arr = _as_bool(mask)
    return encode_rle(arr, bbox(arr))


def rle_counts(rle: Optional[dict]) -> Optional[str]:
    """The ``counts`` of an RLE as a ``str``, whatever it was handed as.

    Three places identify an RLE by its run lengths rather than by its pixels --
    the compiler's ``input_hash``, the truth table's conflict deduplication and
    the input digest -- and all three have to agree on what "the same mask"
    means. :func:`encode_rle` produces ``str`` counts, but pycocotools produces
    ``bytes`` and a caller may hand one straight on: the digest that compared
    them without normalising simply never matched its own stored value again, so
    that frame was recompiled for ever. ``None`` means "no RLE / no counts".
    """
    if not rle:
        return None
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        return counts.decode("ascii")
    return None if counts is None else str(counts)


def _coco_rle(rle: dict) -> dict:
    """Our RLE dict in the exact shape pycocotools wants (``bytes`` counts)."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def rle_area(rle: dict) -> int:
    """Pixels covered, straight off the run lengths.

    The encoding already carries the answer, so a caller that only needs the
    area -- an export writing one number per annotation -- has no reason to
    build the ``H x W`` array first.
    """
    return int(coco_mask.area(_coco_rle(rle)))


def rle_bbox_xywh(rle: dict) -> list[float]:
    """COCO ``[x, y, w, h]`` straight off the run lengths (``[0, 0, 0, 0]`` if empty)."""
    return [float(v) for v in coco_mask.toBbox(_coco_rle(rle))]


def decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict (``str`` or ``bytes`` counts) to a bool mask.

    The result is **Fortran-ordered**, because that is what pycocotools writes
    and what :func:`encode_rle` wants back; nothing re-orders it on the way.

    ``.view(bool)`` rather than ``.astype(bool)``: pycocotools fills the buffer
    with 0 and 1 only, a ``bool_`` is one byte, and the cast was a second pass
    over the whole canvas -- 5 ms per instance at 12 MP.
    """
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    h, w = int(rle["size"][0]), int(rle["size"][1])
    decoded = coco_mask.decode({"size": [h, w], "counts": counts})
    return decoded.view(bool)


# ---------------------------------------------------------------------------
# the decode memo
# ---------------------------------------------------------------------------
#: How many bytes of decoded shapes :func:`decode_rle_shared` keeps.
#:
#: What is kept is each shape **cropped to its own bounding box**, which is what
#: makes the memo affordable: a part of an OAK frame decodes to a 12 MB canvas
#: whatever its size, and forty of those would be 500 MB, while the same forty
#: cropped are a few tens of MB. The working set is exactly one frame's shapes
#: -- the compiler decodes the same forty keyframes every time the annotator
#: commits, confirms or walks to the next frame, and at 12 MP that decode was
#: 0.4 s of every one of those gestures.
DECODE_CACHE_BYTES = 128 * 1024 * 1024

_decoded: "OrderedDict[tuple, tuple[np.ndarray, Box]]" = OrderedDict()
_decoded_bytes = 0
_decode_lock = threading.Lock()


def decode_rle_shared(rle: dict) -> tuple[np.ndarray, Box]:
    """``(mask inside its own bounding box, that box)`` -- shared, **read-only**.

    The same pixels :func:`decode_rle` produces, with the empty margin left off:
    ``decode_rle(rle)[y0:y1, x0:x1]`` for the returned ``(x0, y0, x1, y1)``, and
    that box is read off the run lengths (:func:`rle_bbox_xywh`) rather than
    measured, so nothing scans the canvas to find it. An RLE with no pixels set
    gives an empty array and ``(0, 0, 0, 0)``.

    Memoised, and it cannot go stale: the key *is* the content (the ``counts``
    string and the size), so a re-traced shape is a different key rather than a
    stale entry. The array is shared and therefore marked non-writeable, which
    turns "somebody painted into a decoded mask" from a corruption of every
    later answer into a ``ValueError`` on the line that did it; a caller that
    needs to write takes its own copy.

    Two threads asking for the same shape at once both decode it and one of the
    two entries is dropped, which costs a decode and is otherwise harmless --
    worth not holding the lock across the decode, since that is the expensive
    part.
    """
    global _decoded_bytes

    size = rle.get("size")
    key = (rle_counts(rle), None if size is None else (int(size[0]), int(size[1])))
    with _decode_lock:
        hit = _decoded.get(key)
        if hit is not None:
            _decoded.move_to_end(key)
            return hit

    x, y, width, height = rle_bbox_xywh(rle)
    box: Box = ((int(x), int(y), int(x + width), int(y + height))
                if width > 0 and height > 0 else (0, 0, 0, 0))
    full = decode_rle(rle)
    mask = np.asfortranarray(full[box[1]:box[3], box[0]:box[2]])
    mask.flags.writeable = False
    found = (mask, box)
    with _decode_lock:
        if key not in _decoded:
            _decoded[key] = found
            _decoded_bytes += mask.nbytes
        while len(_decoded) > 1 and _decoded_bytes > DECODE_CACHE_BYTES:
            _key, (dropped, _box) = _decoded.popitem(last=False)
            _decoded_bytes -= dropped.nbytes
    return found


def decode_cache_stats() -> dict:
    """``{"entries", "bytes"}`` of the memo (tests and memory reporting)."""
    with _decode_lock:
        return {"entries": len(_decoded), "bytes": _decoded_bytes}


def clear_decode_cache() -> None:
    """Forget every memoised shape. Only ever costs time, never an answer."""
    global _decoded_bytes

    with _decode_lock:
        _decoded.clear()
        _decoded_bytes = 0


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


def bbox(mask: np.ndarray) -> Optional[Box]:
    """Tight bounding box ``(x0, y0, x1, y1)`` (exclusive), ``None`` if empty."""
    m = _as_bool(mask)
    rows = np.flatnonzero(m.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(m.any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def min_side(mask: np.ndarray) -> int:
    """Shorter side of the bounding box in pixels (0 for an empty mask)."""
    box = bbox(mask)
    if box is None:
        return 0
    x0, y0, x1, y1 = box
    return int(min(x1 - x0, y1 - y0))


def rle_min_side(rle: Optional[dict]) -> int:
    """:func:`min_side` straight off the run lengths, without decoding.

    The same number as ``min_side(decode_rle(rle))`` -- pycocotools measures the
    tight box the same way -- for a caller that has a stored RLE and wants one
    integer out of it, not an ``H x W`` array.
    """
    if not rle:
        return 0
    _x, _y, width, height = rle_bbox_xywh(rle)
    return int(min(width, height))


def area(mask: np.ndarray) -> int:
    """Number of set pixels."""
    return int(np.count_nonzero(_as_bool(mask)))


# ---------------------------------------------------------------------------
# morphology / cleanup
# ---------------------------------------------------------------------------


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that are fully enclosed by the mask.

    Background connected (4-connectivity) to the image border is preserved,
    so a notch cut in from the outside stays open.
    """
    m = _as_u8(mask)
    h, w = m.shape
    # Pad by one so a mask touching the border still has an outside to flood.
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = m
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 1)
    holes = padded[1:-1, 1:-1] == 0
    return _as_bool(mask) | holes


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop 8-connected components smaller than ``min_px`` pixels."""
    if min_px <= 1:
        return _as_bool(mask).copy()
    m = _as_u8(mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n_labels, dtype=bool)
    for label in range(1, n_labels):  # 0 is the background
        keep[label] = stats[label, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


# ---------------------------------------------------------------------------
# tolerant comparison
# ---------------------------------------------------------------------------


def tolerant_sym_diff(a: np.ndarray, b: np.ndarray, tol_px: int = 2) -> int:
    """Symmetric difference of ``a`` and ``b``, ignoring a band around ``a``.

    ``a`` is the **reference** mask (the frozen truth): the band is the
    dilation of ``boundary(a)`` alone by a square kernel of size
    ``2 * tol_px + 1``, and every XOR pixel inside it is treated as an
    irrelevant re-trace of the same silhouette.  A pure translation therefore
    registers as soon as it exceeds ``tol_px`` pixels.

    The function is deliberately **not symmetric** in its arguments -- swapping
    them measures the difference against the other silhouette's tolerance.

    Returns the number of XOR pixels left outside the band.
    """
    am, bm = _as_u8(a), _as_u8(b)
    if am.shape != bm.shape:
        raise ValueError(f"shape mismatch: {am.shape!r} vs {bm.shape!r}")
    xor = cv2.bitwise_xor(am, bm)
    if not xor.any():
        return 0
    tol = max(0, int(tol_px))
    band = cv2.dilate(_boundary(am), _square_kernel(2 * tol + 1), borderValue=0)
    return int(np.count_nonzero(xor.astype(bool) & ~band.astype(bool)))


def is_conflict(
    old: np.ndarray,
    new: np.ndarray,
    area_frac: float = 0.02,
    min_px: int = 20,
    tol_px: int = 2,
) -> bool:
    """True when ``new`` differs from ``old`` beyond re-tracing tolerance.

    ``old`` is passed as the reference mask of :func:`tolerant_sym_diff`, so
    the tolerance band is measured around the frozen truth.  The change must
    exceed ``max(area_frac * area(old), min_px)`` pixels of tolerant
    symmetric difference; ``min_px`` is the floor that keeps tiny
    instances from tripping the check on a couple of pixels.
    """
    threshold = max(area_frac * area(old), float(min_px))
    return tolerant_sym_diff(old, new, tol_px=tol_px) > threshold


# ---------------------------------------------------------------------------
# polygons
# ---------------------------------------------------------------------------


def mask_to_polygons(mask: np.ndarray, tol: float = 1.0) -> list[list[float]]:
    """Outer contours of a mask as COCO-style flat polygons.

    Each component yields one polygon simplified with ``cv2.approxPolyDP``
    (epsilon ``tol``).  Degenerate contours (< 3 points) are dropped.

    Only external contours are returned, so **holes are not representable**:
    a polygon round trip fills any enclosed hole and is therefore lossy --
    use :func:`encode_rle` / :func:`decode_rle` when fidelity matters.
    """
    m = _as_u8(mask)
    if not m.any():
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, float(tol), True)
        if len(approx) < 3:
            continue
        polys.append([float(v) for v in approx.reshape(-1)])
    return polys


def polygons_to_mask(polys: Iterable[Sequence[float]], hw: HW) -> np.ndarray:
    """Rasterise flat polygons into a bool mask; polygons are unioned."""
    h, w = int(hw[0]), int(hw[1])
    canvas = np.zeros((h, w), np.uint8)
    for poly in polys:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas.astype(bool)


# ---------------------------------------------------------------------------
# geometric transforms
# ---------------------------------------------------------------------------


def warp_mask(mask: np.ndarray, sim: Similarity, hw_out: HW) -> np.ndarray:
    """Apply ``x' = s * R(theta) * x + t`` with nearest-neighbour sampling."""
    h_out, w_out = int(hw_out[0]), int(hw_out[1])
    cos_t = float(sim.scale) * float(np.cos(sim.theta))
    sin_t = float(sim.scale) * float(np.sin(sim.theta))
    matrix = np.array(
        [[cos_t, -sin_t, float(sim.tx)], [sin_t, cos_t, float(sim.ty)]],
        dtype=np.float64,
    )
    warped = cv2.warpAffine(
        _as_u8(mask),
        matrix,
        (w_out, h_out),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(bool)


def crop(mask: np.ndarray, box: Box) -> np.ndarray:
    """Cut out ``box`` = ``(x0, y0, x1, y1)``; parts outside the mask are 0.

    The result always has shape ``(y1 - y0, x1 - x0)``, so the box may hang
    over the image edge.
    """
    m = _as_bool(mask)
    x0, y0, x1, y1 = (int(v) for v in box)
    out = np.zeros((max(0, y1 - y0), max(0, x1 - x0)), dtype=bool)
    h, w = m.shape
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = m[sy0:sy1, sx0:sx1]
    return out


def paste(dst: np.ndarray, src: np.ndarray, xy: tuple[int, int]) -> None:
    """Write ``src`` into ``dst`` with its top-left corner at ``xy``.

    The destination region is *overwritten* (so ``crop``/``paste`` round-trip)
    and anything falling outside ``dst`` is clipped away.  ``dst`` is modified
    in place.

    This is a replace, **not** a union: callers that want to composite layers
    must OR explicitly (``dst[y:y+h, x:x+w] |= src``).
    """
    s = _as_bool(src)
    x0, y0 = int(xy[0]), int(xy[1])
    h, w = dst.shape
    sh, sw = s.shape
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x0 + sw, w), min(y0 + sh, h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    dst[dy0:dy1, dx0:dx1] = s[dy0 - y0 : dy1 - y0, dx0 - x0 : dx1 - x0]


# ---------------------------------------------------------------------------
# label maps
# ---------------------------------------------------------------------------


def labelmap_from_masks(
    masks: dict[str, np.ndarray], order: list[str], hw: HW
) -> tuple[np.ndarray, dict[int, str]]:
    """Paint ``masks`` bottom-to-top into a uint16 label map.

    Instances are painted in ``order`` and numbered 1..N in the same order, so
    a later (higher) instance both overwrites the ones below it and carries a
    larger id.  0 is the background.  Keys of ``order`` without a mask, and
    masks without a place in ``order``, are ignored.

    Each instance is painted **inside its own bounding box**, which is measured
    here rather than taken on trust.  A boolean-mask assignment walks every
    element of the canvas it is given, so at 12 MP a Fortran-ordered instance
    mask -- which is what the compiler now produces, because that is what the
    run lengths are stored in -- cost 24 ms, and forty of them a second of the
    GUI thread on every frame change.  Measuring the box costs 0.5 ms and the
    assignment inside it 0.07 ms.  An empty mask is skipped, but it still takes
    its id: the ids are the paint order and a caller reads ``id2key`` by them.
    """
    h, w = int(hw[0]), int(hw[1])
    painted = [key for key in order if key in masks]
    if len(painted) > np.iinfo(np.uint16).max:
        raise ValueError(
            f"{len(painted)} instances exceed the uint16 label map capacity "
            f"({np.iinfo(np.uint16).max})"
        )
    labelmap = np.zeros((h, w), dtype=np.uint16)
    id2key: dict[int, str] = {}
    for idx, key in enumerate(painted, start=1):
        m = _as_bool(masks[key])
        if m.shape != (h, w):
            raise ValueError(
                f"mask {key!r} has shape {m.shape!r}, expected {(h, w)!r}"
            )
        id2key[idx] = key
        window = bbox(m)
        if window is None:
            continue
        x0, y0, x1, y1 = window
        labelmap[y0:y1, x0:x1][m[y0:y1, x0:x1]] = idx
    return labelmap, id2key
