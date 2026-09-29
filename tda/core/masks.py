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
from pycocotools import _mask as _low_level
from pycocotools import mask as coco_mask

from tda.core.model import Similarity

__all__ = [
    "BoxedMask",
    "CHECK_ENCODE_WINDOW",
    "DECODE_CACHE_BYTES",
    "RunLengths",
    "clear_decode_cache",
    "decode_cache_stats",
    "encode_rle",
    "encode_rle_boxed",
    "encode_rle_patch",
    "encode_rle_windowed",
    "decode_rle",
    "decode_rle_shared",
    "rle_area",
    "rle_bbox",
    "rle_bbox_xywh",
    "rle_contains",
    "rle_counts",
    "rle_iou",
    "rle_overlap",
    "bbox",
    "bbox_in",
    "min_side",
    "rle_min_side",
    "area",
    "fill_holes",
    "remove_small_components",
    "tolerant_sym_diff",
    "tolerance_reach",
    "conflict_pixels",
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


def _refuse_pixels_outside(arr: np.ndarray, window: Box, who: str = "encode_rle",
                           loss: str = "the run lengths would be missing them") -> None:
    """Raise when ``arr`` has a set pixel outside ``window`` (:data:`CHECK_ENCODE_WINDOW`).

    ``who`` and ``loss`` name the caller trusting the window and what a wrong
    one would cost it; the encoders keep the words they always had.
    """
    x0, y0, x1, y1 = (int(v) for v in window)
    height, width = arr.shape
    x0, y0 = max(0, min(x0, width)), max(0, min(y0, height))
    x1, y1 = max(x0, min(x1, width)), max(y0, min(y1, height))
    for band in (arr[:y0, :], arr[y1:, :], arr[y0:y1, :x0], arr[y0:y1, x1:]):
        if band.size and band.any():
            raise ValueError(
                f"{who} was given the window {tuple(window)!r}, but the "
                f"mask has pixels outside it: {loss}. Only a caller that "
                f"produced the mask may pass a window."
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


def encode_rle_patch(patch: np.ndarray, box: Box, hw: HW) -> dict:
    """:func:`encode_rle` for a caller holding only the box and its contents.

    The same scratch canvas and the same bytes out, without ever materialising
    the full-frame array: an eraser stroke's protected set is a few hundred
    pixels across on a 12 MP frame, and building 12 MB to encode it twice per
    stroke was 6 ms of every stroke (round 3).
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    buffer = _encode_buffer(hw)
    try:
        if x1 > x0 and y1 > y0:
            buffer[y0:y1, x0:x1] = _as_bool(patch).view(np.uint8)
        rle = coco_mask.encode(buffer)
    finally:
        if x1 > x0 and y1 > y0:
            buffer[y0:y1, x0:x1] = 0
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].decode("ascii"),
    }


def encode_rle_windowed(mask: np.ndarray, window: Optional[Box]) -> dict:
    """:func:`encode_rle`'s bytes for ``(mask, window)``, from the window alone.

    ``encode_rle`` hands pycocotools the whole canvas, so even a windowed
    encode walks all 12 MP of an OAK frame: 2.4 ms whatever the part's size.
    This finds the run lengths inside the window -- where a column of the
    window changes value, as positions in the column-major order the counts
    are written in -- and gives them to pycocotools' own string writer
    (``frUncompressedRLE``, the ``rleToString`` that ``encode`` ends in), so
    the cost is the window's and the bytes are the same ones.
    ``tests/test_disagreement_window.py`` pins that byte for byte.

    ``window`` is the same promise as :func:`encode_rle`'s: the mask is empty
    outside it (checked under :data:`CHECK_ENCODE_WINDOW`). ``None`` measures
    the box instead (:func:`bbox`). Task U2f: what the frozen-row comparison
    encodes a compiled instance with before it decides anything else.
    """
    arr = _as_bool(mask)
    height, width = arr.shape
    total = height * width
    if window is None:
        window = bbox(arr) or (0, 0, 0, 0)
    elif CHECK_ENCODE_WINDOW:
        _refuse_pixels_outside(arr, window, "encode_rle_windowed")
    if total == 0:
        return encode_rle(arr)
    clipped = _clip_box(window, (height, width))
    positions = np.empty(0, dtype=np.int64)
    if clipped is not None:
        x0, y0, x1, y1 = clipped
        span = y1 - y0
        # one row per column of the window, padded with the background above
        # and below it: a change of value is a run boundary
        padded = np.zeros((x1 - x0, span + 2), dtype=np.uint8)
        padded[:, 1:-1] = arr.T[x0:x1, y0:y1].view(np.uint8)
        changes = np.flatnonzero(padded[:, 1:] != padded[:, :-1])
        column, row = np.divmod(changes, span + 1)
        positions = (column + x0) * height + (row + y0)
        if y0 == 0 and y1 == height and positions.size > 1:
            # a run that goes on from the bottom of one column into the top of
            # the next ended and began at the same position: it is one run
            seam = np.flatnonzero(positions[1:] == positions[:-1])
            if seam.size:
                positions = np.delete(positions, np.concatenate((seam, seam + 1)))
        if positions.size and positions[-1] == total:
            positions = positions[:-1]  # the canvas ends there, not a run
    counts = np.diff(np.concatenate(([0], positions, [total]))).astype(np.uint32)
    rle = _low_level.frUncompressedRLE(
        [{"counts": counts, "size": [height, width]}], height, width)[0]
    return {"size": [int(height), int(width)], "counts": rle["counts"].decode("ascii")}


class BoxedMask:
    """A mostly-empty full-frame boolean kept as its bounding box and patch.

    What the annotator has erased during one edit is a handful of blobs on a
    12 MP canvas, and every full-frame operation on it -- the copy, the two
    RLE encodes per stroke, the ``& ~layer`` that keeps it honest -- was paid
    at 12 MP whatever its size. Keeping the box means the work is the size of
    what is in it (round 3).

    Instances are treated as **immutable**: every operation returns a new one,
    so a value that has been handed to the undo stack or the sidecar cannot
    change underneath them. ``None`` is the empty set: this class never
    represents one, so "is there anything protected?" is ``is not None`` and
    costs nothing.
    """

    __slots__ = ("hw", "box", "patch")

    def __init__(self, hw: HW, box: Box, patch: np.ndarray) -> None:
        self.hw = (int(hw[0]), int(hw[1]))
        self.box = tuple(int(v) for v in box)
        self.patch = patch

    # -- construction -------------------------------------------------------
    @classmethod
    def of(cls, mask, hw: Optional[HW] = None) -> Optional["BoxedMask"]:
        """Box a full-frame mask; ``None`` when it holds nothing."""
        if mask is None:
            return None
        if isinstance(mask, BoxedMask):
            return mask
        arr = _as_bool(mask)
        box = bbox(arr)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        return cls(hw or arr.shape, box,
                   np.array(arr[y0:y1, x0:x1], dtype=bool, copy=True))

    @classmethod
    def of_patch(cls, hw: HW, box: Box, patch: np.ndarray) -> Optional["BoxedMask"]:
        """Re-tighten ``patch`` (which sits at ``box``); ``None`` when empty."""
        inner = bbox(patch)
        if inner is None:
            return None
        ix0, iy0, ix1, iy1 = inner
        x0, y0 = int(box[0]) + ix0, int(box[1]) + iy0
        return cls(hw, (x0, y0, int(box[0]) + ix1, int(box[1]) + iy1),
                   np.array(patch[iy0:iy1, ix0:ix1], dtype=bool, copy=True))

    @classmethod
    def from_rle(cls, rle: Optional[dict]) -> Optional["BoxedMask"]:
        """Read either shape: :meth:`rle`'s, or a plain full-frame COCO RLE.

        The second is what round 2 wrote, and a crash sidecar from then must
        still restore -- it is the annotator's unfinished work.
        """
        if not rle:
            return None
        if "box" in rle and "rle" in rle:
            patch = decode_rle(rle["rle"])
            return cls.of_patch(tuple(rle["hw"]), tuple(rle["box"]), patch)
        return cls.of(decode_rle(rle))

    # -- reading ------------------------------------------------------------
    def full(self) -> np.ndarray:
        """The whole frame, for a caller that really needs the canvas."""
        out = np.zeros(self.hw, dtype=bool)
        x0, y0, x1, y1 = self.box
        out[y0:y1, x0:x1] = self.patch
        return out

    def crop(self, rect: Box) -> np.ndarray:
        """The window ``rect`` of the full mask, as a bool array of its size."""
        x0, y0, x1, y1 = (int(v) for v in rect)
        out = np.zeros((max(0, y1 - y0), max(0, x1 - x0)), dtype=bool)
        bx0, by0, bx1, by1 = self.box
        ox0, oy0 = max(x0, bx0), max(y0, by0)
        ox1, oy1 = min(x1, bx1), min(y1, by1)
        if ox1 > ox0 and oy1 > oy0:
            out[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = self.patch[
                oy0 - by0:oy1 - by0, ox0 - bx0:ox1 - bx0]
        return out

    def at(self, x: int, y: int) -> bool:
        """Is the pixel ``(x, y)`` in the set?"""
        x0, y0, x1, y1 = self.box
        if not (x0 <= x < x1 and y0 <= y < y1):
            return False
        return bool(self.patch[int(y) - y0, int(x) - x0])

    def count(self) -> int:
        return int(np.count_nonzero(self.patch))

    def copy(self) -> "BoxedMask":
        return BoxedMask(self.hw, self.box, self.patch.copy())

    def rle(self) -> dict:
        """The box, the frame it sits in, and the **patch's** run lengths.

        Not the frame's: ``coco_mask.encode`` walks whatever canvas it is
        given, so encoding a 500x300 erasure as a 12 MP mask was 2.5 ms, twice
        per stroke, for a few hundred bytes of output.  This is a private
        format -- the undo payload and the crash sidecar -- never stored
        geometry, so it is free to say where the box is instead (round 3).
        """
        return {"box": list(self.box), "hw": [self.hw[0], self.hw[1]],
                "rle": encode_rle(self.patch)}

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, BoxedMask):
            return NotImplemented
        return (self.hw == other.hw and self.box == other.box
                and np.array_equal(self.patch, other.patch))

    def __repr__(self) -> str:  # noqa: D105 - for a failing assert
        return f"BoxedMask(hw={self.hw}, box={self.box}, {self.count()} px)"


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


def rle_bbox(rle: Optional[dict]) -> Optional[Box]:
    """:func:`bbox` straight off the run lengths (``None`` for an empty mask).

    The same ``(x0, y0, x1, y1)`` convention as :func:`bbox`, so a caller can
    order a few hundred stored masks by where they are without decoding one.
    """
    if not rle:
        return None
    x, y, width, height = rle_bbox_xywh(rle)
    if width <= 0 or height <= 0:
        return None
    return (int(x), int(y), int(x + width), int(y + height))


def _point_rle(x: int, y: int, h: int, w: int) -> dict:
    """A one-pixel RLE at ``(x, y)``, built from run lengths, not from an array.

    ``counts`` is column-major (Fortran order), so the pixel's index is
    ``x * h + y`` and the mask is three runs. Encoding a real ``H x W`` array
    to ask about one pixel would allocate 12 MB on an OAK frame.
    """
    index = int(x) * int(h) + int(y)
    counts = [index, 1, int(h) * int(w) - index - 1]
    return _low_level.frUncompressedRLE(
        [{"counts": counts, "size": [int(h), int(w)]}], int(h), int(w)
    )[0]


def rle_contains(rle: Optional[dict], x: int, y: int) -> bool:
    """Is image pixel ``(x, y)`` set in this RLE?  No decoding.

    Measured at 0.06 ms on a 4032x3040 mask against 13 ms (and 12 MB) for a
    full decode, which is what makes "which of these forty drafts is the
    annotator pointing at?" a question worth asking on every ``Shift+A``.
    """
    if not rle:
        return False
    h, w = int(rle["size"][0]), int(rle["size"][1])
    if not (0 <= int(x) < w and 0 <= int(y) < h):
        return False
    merged = coco_mask.merge([_coco_rle(rle), _point_rle(int(x), int(y), h, w)],
                             intersect=1)
    return bool(int(coco_mask.area(merged)))


def rle_overlap(a: Optional[dict], b: Optional[dict]) -> int:
    """Pixels two RLEs share, off the run lengths; ``0`` when either is missing.

    Masks of two different sizes belong to two different frames and share
    nothing, which is an answer rather than an error (see :func:`rle_iou`).
    """
    if not a or not b:
        return 0
    if [int(v) for v in a["size"]] != [int(v) for v in b["size"]]:
        return 0
    return int(coco_mask.area(coco_mask.merge([_coco_rle(a), _coco_rle(b)],
                                              intersect=1)))


def rle_iou(a: Optional[dict], b: Optional[dict]) -> float:
    """Intersection over union of two RLEs, straight off the run lengths.

    Neither mask is decoded, which is the point: ranking a frame's draft
    polygons against what the annotator is drawing means comparing a handful of
    stored RLEs, and at 4032x3040 each decode is a 12 MB array nobody would
    keep. Masks of different sizes are different frames, so their overlap is
    ``0.0`` rather than an error -- the caller that cares about the mismatch
    (:func:`tda.core.ls_adopt.drafts_for`) refuses them by size first.
    """
    if not a or not b:
        return 0.0
    if [int(v) for v in a["size"]] != [int(v) for v in b["size"]]:
        return 0.0
    out = np.asarray(coco_mask.iou([_coco_rle(a)], [_coco_rle(b)], [0]), dtype=float)
    return float(out.reshape(-1)[0]) if out.size else 0.0


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
# run lengths without the canvas (task U2f)
# ---------------------------------------------------------------------------
#: The longest value :func:`_read_counts` reads, in characters. Six characters
#: carry thirty bits, a canvas of half a billion pixels; a longer value is left
#: to pycocotools.
_MAX_VALUE_CHARS = 6


def _read_counts(rle: dict, total: int) -> Optional[np.ndarray]:
    """The run lengths a COCO ``counts`` string spells, or ``None``.

    pycocotools' ``rleFrString`` (``maskApi.c``) done with numpy, so that the
    lengths can be read without decoding the canvas they describe. Each value
    is written five bits per character, least significant first, as
    ``chr(48 + bits)`` with ``0x20`` set on every character but the last and
    ``0x10`` of the last one the sign; from the fourth value on (index 3) each
    is stored as the difference to the value two before it.

    ``None`` whenever this reader could not promise pycocotools' answer: counts
    that are not a string, a character outside the 48..111 alphabet, a value
    cut off or longer than :data:`_MAX_VALUE_CHARS`, a negative length, or
    lengths that do not add up to the canvas. The caller then decodes with
    pycocotools after all (:class:`RunLengths`), so an odd RLE costs time and
    never a different answer.
    """
    counts = rle.get("counts")
    if isinstance(counts, str):
        if not counts.isascii():
            return None
        raw = counts.encode("ascii")
    elif isinstance(counts, (bytes, bytearray)):
        raw = bytes(counts)
    else:
        return None
    if not raw:
        return None
    chars = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) - 48
    if chars.min() < 0 or chars.max() > 63:
        return None
    last = (chars & 0x20) == 0          # the last character of its value
    if not last[-1]:
        return None
    ends = np.flatnonzero(last)
    starts = np.concatenate(([0], ends[:-1] + 1))
    length = ends - starts + 1
    if int(length.max()) > _MAX_VALUE_CHARS:
        return None
    place = np.arange(chars.size) - np.repeat(starts, length)
    values = np.add.reduceat((chars & 0x1F) << (5 * place), starts)
    negative = (chars[ends] & 0x10) != 0
    values[negative] -= np.left_shift(1, 5 * length[negative])
    # index 0, 1 and 2 are written as they are; from 3 on, as the step from
    # the value two places back -- so odd and even places are running sums
    values[1::2] = np.cumsum(values[1::2])
    values[2::2] = np.cumsum(values[2::2])
    if values.min() < 0 or int(values.sum()) != int(total):
        return None
    return values


class RunLengths:
    """One COCO RLE, answered off its run lengths instead of its canvas.

    :func:`decode_rle` builds the whole frame -- 12 MB and 8 ms on an OAK frame
    -- to say anything about a part that may be twenty pixels across. This
    reads the counts string once (:func:`_read_counts`) and answers the three
    questions a comparison asks without that canvas: how many pixels
    (:meth:`area`), where (:meth:`bbox`), and which ones inside a window
    (:meth:`window`). Each is the answer ``decode_rle(rle)`` would give:

    * the counts alternate background and foreground runs in column-major
      order, starting with background, so the foreground is exactly the
      odd-numbered runs ``[start, end)`` of that order;
    * pixel ``(x, y)`` is position ``x * h + y``.

    An RLE the reader cannot read exactly is decoded by pycocotools instead,
    and every answer is then taken off that canvas -- which is what the caller
    did before, so an odd RLE costs time and never a different answer.
    ``hw`` is the canvas either way. Task U2f.
    """

    __slots__ = ("hw", "_starts", "_ends", "_full")

    def __init__(self, rle: dict) -> None:
        height, width = int(rle["size"][0]), int(rle["size"][1])
        self.hw = (height, width)
        self._full: Optional[np.ndarray] = None
        counts = _read_counts(rle, height * width) if height * width > 0 else None
        if counts is None:
            self._full = decode_rle(rle)
            self.hw = (int(self._full.shape[0]), int(self._full.shape[1]))
            self._starts = self._ends = np.empty(0, dtype=np.int64)
            return
        ends = np.cumsum(counts)
        starts = ends - counts
        # the foreground runs, without the empty ones a hand-made RLE may carry
        keep = counts[1::2] > 0
        self._starts, self._ends = starts[1::2][keep], ends[1::2][keep]

    @property
    def decoded(self) -> bool:
        """Did this RLE need pycocotools after all? (Tests.)"""
        return self._full is not None

    def area(self) -> int:
        """``area(decode_rle(rle))``: the foreground runs' lengths added up."""
        if self._full is not None:
            return area(self._full)
        return int((self._ends - self._starts).sum())

    def bbox(self) -> Optional[Box]:
        """``bbox(decode_rle(rle))``: columns from the runs, rows too unless one wraps.

        A run that goes on past the bottom of a column covers that column's
        last row and the next column's first, so any such run makes the box
        the canvas' full height; otherwise every run lies in one column and
        the rows are its own.
        """
        if self._full is not None:
            return bbox(self._full)
        if not self._starts.size:
            return None
        height = self.hw[0]
        first = self._starts // height
        last = (self._ends - 1) // height
        x0, x1 = int(first.min()), int(last.max()) + 1
        if bool((last > first).any()):
            return (x0, 0, x1, height)
        y0 = int((self._starts - first * height).min())
        y1 = int((self._ends - 1 - last * height).max()) + 1
        return (x0, y0, x1, y1)

    def window(self, box: Box) -> np.ndarray:
        """``decode_rle(rle)[y0:y1, x0:x1]`` for ``box`` clipped to the canvas.

        **Fortran-ordered**, like :func:`decode_rle`, and built at the window's
        size: every foreground run is cut into the columns it crosses, each
        piece is cut to the window's rows, and the pieces are painted as
        ``+1`` at their top and ``-1`` below their bottom and summed down each
        column. The runs never overlap, so the sum is 0 or 1 everywhere.
        """
        height, width = self.hw
        x0, y0, x1, y1 = (int(v) for v in box)
        x0, y0 = max(0, min(x0, width)), max(0, min(y0, height))
        x1, y1 = max(x0, min(x1, width)), max(y0, min(y1, height))
        if self._full is not None:
            return self._full[y0:y1, x0:x1]
        across, down = x1 - x0, y1 - y0
        paint = np.zeros((across, down + 1), dtype=np.int8)
        if across and down and self._starts.size:
            starts = np.maximum(self._starts, x0 * height)
            ends = np.minimum(self._ends, x1 * height)
            inside = ends > starts
            starts, ends = starts[inside], ends[inside]
            first = starts // height
            last = (ends - 1) // height
            columns = last - first + 1
            run = np.repeat(np.arange(starts.size), columns)
            column = first[run] + (np.arange(run.size)
                                   - np.repeat(np.cumsum(columns) - columns, columns))
            top = np.where(column == first[run], starts[run] - column * height, 0)
            bottom = np.where(column == last[run], ends[run] - column * height, height)
            top = np.clip(top, y0, y1) - y0
            bottom = np.clip(bottom, y0, y1) - y0
            kept = bottom > top
            row = (column[kept] - x0) * (down + 1)
            flat = paint.reshape(-1)
            flat[row + top[kept]] = 1
            flat[row + bottom[kept]] -= 1
            np.cumsum(paint, axis=1, dtype=np.int8, out=paint)
        return paint[:, :down].view(bool).T


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


def bbox_in(mask: np.ndarray, window: Optional[Box]) -> Optional[Box]:
    """:func:`bbox` of a mask known to be empty outside ``window``.

    Measured inside the window and put back in frame coordinates, so the cost
    is the window's rather than the canvas'. ``window`` is the promise
    :attr:`tda.core.compiler.CompiledInstance.window` makes, checked under
    :data:`CHECK_ENCODE_WINDOW` like an encode's; ``None`` measures the whole
    mask. Task U2f.
    """
    arr = _as_bool(mask)
    if window is None:
        return bbox(arr)
    if CHECK_ENCODE_WINDOW:
        _refuse_pixels_outside(arr, window, "bbox_in",
                               "its bounding box would be measured short")
    clipped = _clip_box(window, arr.shape)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    found = bbox(arr[y0:y1, x0:x1])
    if found is None:
        return None
    return (found[0] + x0, found[1] + y0, found[2] + x0, found[3] + y0)


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

    **Nothing is copied on the way in** (task U2f). Every mask the truth table
    compares is column-major -- :func:`decode_rle` and the compiler both make
    Fortran order, and a window of one is column-major too -- and cv2 reads
    rows, so the old ``np.ascontiguousarray(mask.astype(np.uint8))`` made two
    12 MB copies per side, 84 of them per 12 MP refresh: 2.4 s of the 3.1 s
    it spent comparing. Two column-major masks are compared **transposed**
    instead, which is free, and gives the same number: the kernels are
    squares centred on their pixel and the border is 0 on every side, so
    eroding or dilating the transpose is transposing the erosion or the
    dilation, and XOR and a count do not care which way the grid is read.
    ``bool`` is read as ``uint8`` in place (:func:`_u8_pair`), and the count
    is cv2's saturating subtraction -- ``xor - band`` is 1 exactly where the
    XOR is set and the band is not -- instead of two casts and a mask.
    """
    am, bm = _u8_pair(a, b)
    xor = cv2.bitwise_xor(am, bm)
    if not xor.any():
        return 0
    tol = max(0, int(tol_px))
    band = cv2.dilate(_boundary(am), _square_kernel(2 * tol + 1), borderValue=0)
    return int(cv2.countNonZero(cv2.subtract(xor, band)))


def _u8_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two masks as 0/1 ``uint8`` arrays cv2 can read where they are.

    Same shape or ``ValueError`` (the message :func:`tolerant_sym_diff` always
    raised). A ``bool`` is one byte holding 0 or 1 (see :func:`encode_rle`), so
    it is viewed as ``uint8`` rather than cast; when both masks are laid out a
    column at a time, both are transposed -- a view -- so that cv2 reads them
    along their rows without making a C-ordered copy. Whether to transpose
    only decides what it costs: :func:`tolerant_sym_diff` gives the same
    number either way. Not :func:`_as_u8`, whose contiguous copy its other
    callers are written against.
    """
    am, bm = _as_bool(a), _as_bool(b)
    if am.shape != bm.shape:
        raise ValueError(f"shape mismatch: {am.shape!r} vs {bm.shape!r}")
    if am.strides[0] < am.strides[1] and bm.strides[0] < bm.strides[1]:
        am, bm = am.T, bm.T
    return am.view(np.uint8), bm.view(np.uint8)


def tolerance_reach(tol_px: int = 2) -> int:
    """How far from a pixel :func:`tolerant_sym_diff` looks to decide it.

    The band is ``boundary(a)`` dilated by ``tol_px``, and the boundary of a
    pixel depends on its 3x3 neighbourhood: ``tol_px + 1``. A window around
    both masks grown by this much sees every pixel the full-canvas answer
    depends on (task U2f; the proof is in the U2f report).
    """
    return max(0, int(tol_px)) + 1


def conflict_pixels(
    old: np.ndarray,
    new: np.ndarray,
    area_frac: float = 0.02,
    min_px: int = 20,
    tol_px: int = 2,
) -> Optional[int]:
    """:func:`tolerant_sym_diff` when :func:`is_conflict` holds, else ``None``.

    The frozen-row comparison needs both the verdict and the pixel count, and
    asked for them as ``tolerant_sym_diff(old, new) if is_conflict(old, new)``
    -- the whole comparison twice for every disagreement (task U2f).
    """
    threshold = max(area_frac * area(old), float(min_px))
    diff = tolerant_sym_diff(old, new, tol_px=tol_px)
    return diff if diff > threshold else None


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
    return conflict_pixels(old, new, area_frac, min_px, tol_px) is not None


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
    masks: dict[str, np.ndarray],
    order: list[str],
    hw: HW,
    windows: Optional[dict[str, Optional[Box]]] = None,
    out: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Paint ``masks`` bottom-to-top into a uint16 label map.

    Instances are painted in ``order`` and numbered 1..N in the same order, so
    a later (higher) instance both overwrites the ones below it and carries a
    larger id.  0 is the background.  Keys of ``order`` without a mask, and
    masks without a place in ``order``, are ignored.

    Each instance is painted **inside a window it is known to be empty
    outside**.  A boolean-mask assignment walks every element of the canvas it
    is given, so at 12 MP a Fortran-ordered instance mask -- which is what the
    compiler now produces, because that is what the run lengths are stored in
    -- cost 24 ms, and forty of them a second of the GUI thread on every frame
    change.  Where the window comes from is the caller's business:

    * ``windows[key]`` when given -- :attr:`CompiledInstance.window
      <tda.core.compiler.CompiledInstance.window>` is exactly this promise and
      the compiler has already paid for it.  It may be *looser* than the tight
      box; that only paints over more background, never a different map.  A key
      whose entry is ``None`` (or absent) is measured instead.
    * otherwise :func:`bbox`, which is 0.5 ms per instance at 12 MP against
      0.07 ms for the assignment inside it -- forty of those measurements were
      140 ms of every 12 MP frame change, which is why the caller is now asked.

    An empty mask is skipped, but it still takes its id: the ids are the paint
    order and a caller reads ``id2key`` by them.

    ``out`` is a ``(H, W)`` uint16 canvas to paint into.  It is zeroed first,
    so the result is the same array a fresh allocation would have produced; it
    exists so that a caller repainting one frame after another does not ask the
    allocator for a new 24 MB canvas each time.
    """
    h, w = int(hw[0]), int(hw[1])
    painted = [key for key in order if key in masks]
    if len(painted) > np.iinfo(np.uint16).max:
        raise ValueError(
            f"{len(painted)} instances exceed the uint16 label map capacity "
            f"({np.iinfo(np.uint16).max})"
        )
    if out is None:
        labelmap = np.zeros((h, w), dtype=np.uint16)
    else:
        if out.shape != (h, w) or out.dtype != np.uint16:
            raise ValueError(
                f"label map canvas is {out.shape!r} {out.dtype}, expected "
                f"{(h, w)!r} uint16"
            )
        labelmap = out
        labelmap[...] = 0
    id2key: dict[int, str] = {}
    for idx, key in enumerate(painted, start=1):
        m = _as_bool(masks[key])
        if m.shape != (h, w):
            raise ValueError(
                f"mask {key!r} has shape {m.shape!r}, expected {(h, w)!r}"
            )
        id2key[idx] = key
        window = (windows or {}).get(key)
        window = bbox(m) if window is None else _clip_box(window, (h, w))
        if window is None:
            continue
        x0, y0, x1, y1 = window
        labelmap[y0:y1, x0:x1][m[y0:y1, x0:x1]] = idx
    return labelmap, id2key


def _clip_box(box: Box, hw: HW) -> Optional[Box]:
    """``box`` clipped to the canvas, or ``None`` when nothing is left of it."""
    h, w = int(hw[0]), int(hw[1])
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)
