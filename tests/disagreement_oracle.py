"""The frozen-row comparison as main had it before task U2f -- the reference.

Not a test module. :func:`disagreement` here is ``tda.core.truth_conflicts.
disagreement`` as it stood at ``df8333d``, with everything it reached copied
beside it line for line -- the full-canvas decode, the contiguous ``uint8``
copies, the tolerant symmetric difference and the box rule -- so that nothing
the production module changes can move this one with it. Docstrings and
annotations are left out and ``masks.x`` calls the copy of ``x`` here; with
that, every function's syntax tree is ``df8333d``'s.

It decides which conflicts are queued, so the fast one must return exactly
what this returns for every input (U2f ruling 1):
``tests/test_disagreement_window.py`` holds them side by side.  It is slow on
purpose: 12 MB per side per call at 12 MP.  Do not "optimise" it.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from pycocotools import mask as coco_mask

GEOM_BOX = "box"
BOX_TOL_PX = 2


# -- tda/core/masks.py @ df8333d -------------------------------------------- #
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
    eroded = cv2.erode(mask_u8, _square_kernel(3), borderValue=0)
    return cv2.subtract(mask_u8, eroded)


def decode_rle(rle: dict) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    h, w = int(rle["size"][0]), int(rle["size"][1])
    decoded = coco_mask.decode({"size": [h, w], "counts": counts})
    return decoded.view(bool)


def bbox(mask: np.ndarray):
    m = _as_bool(mask)
    rows = np.flatnonzero(m.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(m.any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(_as_bool(mask)))


def tolerant_sym_diff(a: np.ndarray, b: np.ndarray, tol_px: int = 2) -> int:
    am, bm = _as_u8(a), _as_u8(b)
    if am.shape != bm.shape:
        raise ValueError(f"shape mismatch: {am.shape!r} vs {bm.shape!r}")
    xor = cv2.bitwise_xor(am, bm)
    if not xor.any():
        return 0
    tol = max(0, int(tol_px))
    band = cv2.dilate(_boundary(am), _square_kernel(2 * tol + 1), borderValue=0)
    return int(np.count_nonzero(xor.astype(bool) & ~band.astype(bool)))


def is_conflict(old, new, area_frac: float = 0.02, min_px: int = 20,
                tol_px: int = 2) -> bool:
    threshold = max(area_frac * area(old), float(min_px))
    return tolerant_sym_diff(old, new, tol_px=tol_px) > threshold


# -- tda/core/truth_conflicts.py @ df8333d ---------------------------------- #
def is_box(compiled_inst) -> bool:
    return (
        compiled_inst.visible is None
        and compiled_inst.amodal is None
        and compiled_inst.box is not None
    )


def rounded_box(box):
    if box is None:
        return None
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def box_sym_diff(a, b) -> int:
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return int(area_a + area_b - 2 * overlap)


def _stored_box(row: dict):
    if row.get("geom_type") == GEOM_BOX:
        return rounded_box(row.get("box"))
    rle = row.get("visible_rle")
    return None if rle is None else bbox(decode_rle(rle))


def _compiled_box(compiled_inst):
    if compiled_inst.visible is not None:
        return bbox(compiled_inst.visible)
    return rounded_box(compiled_inst.box)


def _box_disagreement(old_box, new_box) -> Optional[int]:
    if old_box is None and new_box is None:
        return None
    if old_box is None or new_box is None:
        present = new_box if old_box is None else old_box
        return int((present[2] - present[0]) * (present[3] - present[1]))
    if max(abs(a - b) for a, b in zip(old_box, new_box)) <= BOX_TOL_PX:
        return None
    return box_sym_diff(old_box, new_box)


def disagreement(row: dict, compiled_inst) -> Optional[int]:
    """``truth_conflicts.disagreement`` at ``df8333d``, line for line."""
    if row.get("geom_type") == GEOM_BOX or is_box(compiled_inst):
        return _box_disagreement(_stored_box(row), _compiled_box(compiled_inst))
    stored_rle = row.get("visible_rle")
    old = None if stored_rle is None else decode_rle(stored_rle)
    new = compiled_inst.visible
    if old is None and new is None:
        return None
    if old is None:
        return area(new)
    if new is None:
        return area(old)
    if old.shape != new.shape:
        return max(area(old), area(new))
    return tolerant_sym_diff(old, new) if is_conflict(old, new) else None
