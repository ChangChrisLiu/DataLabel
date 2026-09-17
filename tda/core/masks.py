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
canvas overlay; everything here is deterministic and side-effect free except
:func:`paste`, which writes into its destination in place.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from tda.core.model import Similarity

__all__ = [
    "encode_rle",
    "decode_rle",
    "bbox",
    "min_side",
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


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a boolean mask as a COCO RLE dict with ``str`` counts."""
    m = np.asfortranarray(_as_bool(mask).astype(np.uint8))
    rle = coco_mask.encode(m)
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].decode("ascii"),
    }


def decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict (``str`` or ``bytes`` counts) to a bool mask."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    h, w = int(rle["size"][0]), int(rle["size"][1])
    decoded = coco_mask.decode({"size": [h, w], "counts": counts})
    return decoded.astype(bool)


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
        labelmap[m] = idx
        id2key[idx] = key
    return labelmap, id2key
