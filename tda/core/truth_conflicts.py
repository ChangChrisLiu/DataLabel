"""Truth-row geometry and the frozen-vs-recompiled comparison (spec 3.4).

Split out of :mod:`tda.core.truth` so the service keeps to the workflow -- what
to write, when to freeze, when to demote -- while everything about *shapes*
lives here:

* :func:`row_values` -- one compiled instance as the columns of a truth row.
  A mask instance stores its visible RLE, a bench part tracked by a rectangle
  stores a ``box`` instead, and ``geom_type`` says which of the two is the
  truth.
* :func:`disagreement` -- does a re-compilation differ from a frozen row
  enough to be a conflict, and by how many pixels? Masks go through
  :func:`tda.core.masks.is_conflict` (tolerant symmetric difference, spec 3.4);
  boxes conflict when any corner moves more than :data:`BOX_TOL_PX`.
* :func:`geom_payload` / :func:`payload_geometry` -- the conflict queue has one
  JSON column per side, so a side is either a COCO RLE (``{"size", "counts"}``)
  or a box (``{"box": [x0, y0, x1, y1]}``). Everything that reads or writes a
  conflict goes through this pair instead of guessing.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

from tda.core import masks
from tda.core.compiler import CompiledInstance

__all__ = [
    "BOX_TOL_PX",
    "GEOM_BOX",
    "GEOM_MASK",
    "RowValues",
    "disagreement",
    "geom_payload",
    "payload_geometry",
    "row_payload",
    "row_values",
]

GEOM_MASK = "mask"
GEOM_BOX = "box"

#: How far a box corner may move before a frozen box row is in conflict.
BOX_TOL_PX = 2


class RowValues(NamedTuple):
    """The column values of one ``compiled_mask`` row."""

    geom_type: str
    visible_rle: Optional[dict]
    box: Optional[list]
    occlusion_ratio: float
    visibility: str
    placement: str


# --------------------------------------------------------------------------- #
# geometry of a row
# --------------------------------------------------------------------------- #
def is_box(compiled_inst: CompiledInstance) -> bool:
    """Box-only geometry: a box, no visible mask and no amodal shape behind it."""
    return (
        compiled_inst.visible is None
        and compiled_inst.amodal is None
        and compiled_inst.box is not None
    )


def row_values(compiled_inst: CompiledInstance) -> RowValues:
    """One compiled instance as the columns of its truth row."""
    if is_box(compiled_inst):
        geom_type: str = GEOM_BOX
        visible_rle: Optional[dict] = None
        box: Optional[list] = [float(v) for v in compiled_inst.box]
    else:
        geom_type = GEOM_MASK
        visible_rle = (
            None if compiled_inst.visible is None else masks.encode_rle(compiled_inst.visible)
        )
        box = None
    return RowValues(
        geom_type=geom_type,
        visible_rle=visible_rle,
        box=box,
        occlusion_ratio=float(compiled_inst.occlusion_ratio),
        visibility=compiled_inst.visibility,
        placement=compiled_inst.placement,
    )


# --------------------------------------------------------------------------- #
# conflict payloads
# --------------------------------------------------------------------------- #
def geom_payload(visible_rle: Optional[dict], box) -> Optional[dict]:
    """One side of a conflict: the RLE, or the box wrapped as ``{"box": [...]}``."""
    if visible_rle is not None:
        return visible_rle
    if box is not None:
        return {"box": [float(v) for v in box]}
    return None


def row_payload(row: dict) -> Optional[dict]:
    """The same, for a stored truth row."""
    return geom_payload(row.get("visible_rle"), row.get("box"))


def payload_geometry(payload: Optional[dict]) -> tuple[str, Optional[dict], Optional[list]]:
    """``(geom_type, visible_rle, box)`` of a conflict side; empty -> a mask row."""
    if not payload:
        return (GEOM_MASK, None, None)
    if "box" in payload:
        return (GEOM_BOX, None, [float(v) for v in payload["box"]])
    return (GEOM_MASK, payload, None)


def payload_row(payload: Optional[dict]) -> dict:
    """One conflict side shaped like a stored row, so :func:`disagreement` reads it.

    That is how a queued value is compared against a fresh compilation with
    exactly the rule that queued it in the first place.
    """
    geom_type, visible_rle, box = payload_geometry(payload)
    return {"geom_type": geom_type, "visible_rle": visible_rle, "box": box}


# --------------------------------------------------------------------------- #
# boxes
# --------------------------------------------------------------------------- #
def rounded_box(box) -> Optional[tuple[int, int, int, int]]:
    """``box`` on whole pixels, or ``None`` when it has no area left."""
    if box is None:
        return None
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def box_sym_diff(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Pixels covered by exactly one of two axis-aligned boxes."""
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return int(area_a + area_b - 2 * overlap)


def _stored_box(row: dict) -> Optional[tuple[int, int, int, int]]:
    """The box of a stored row: its own, or the bounding box of its mask."""
    if row.get("geom_type") == GEOM_BOX:
        return rounded_box(row.get("box"))
    rle = row.get("visible_rle")
    return None if rle is None else masks.bbox(masks.decode_rle(rle))


def _compiled_box(compiled_inst: CompiledInstance) -> Optional[tuple[int, int, int, int]]:
    """The same for a fresh compilation."""
    if compiled_inst.visible is not None:
        return masks.bbox(compiled_inst.visible)
    return rounded_box(compiled_inst.box)


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #
def disagreement(row: dict, compiled_inst: CompiledInstance) -> Optional[int]:
    """Differing pixels between a frozen row and a re-compilation, or ``None``.

    ``None`` means "close enough to be the same annotation": within the
    re-tracing tolerance of :func:`tda.core.masks.is_conflict` for masks, or
    within :data:`BOX_TOL_PX` on every corner for boxes. Geometry appearing or
    disappearing always counts, and so does a change of canvas size, which no
    comparison could survive.

    Boxes decide whenever *either* side is a box row -- a part that moved from
    the chassis to the bench between two compilations is compared on its
    rectangle, which is the only thing the two sides still have in common.
    """
    if row.get("geom_type") == GEOM_BOX or is_box(compiled_inst):
        return _box_disagreement(_stored_box(row), _compiled_box(compiled_inst))
    stored_rle = row.get("visible_rle")
    old = None if stored_rle is None else masks.decode_rle(stored_rle)
    new = compiled_inst.visible
    if old is None and new is None:
        return None
    if old is None:
        return masks.area(new)
    if new is None:
        return masks.area(old)
    if old.shape != new.shape:
        return max(masks.area(old), masks.area(new))
    return masks.tolerant_sym_diff(old, new) if masks.is_conflict(old, new) else None


def _box_disagreement(old_box, new_box) -> Optional[int]:
    """Has a corner moved more than the tolerance? Then by how many pixels."""
    if old_box is None and new_box is None:
        return None
    if old_box is None or new_box is None:
        present = new_box if old_box is None else old_box
        return int((present[2] - present[0]) * (present[3] - present[1]))
    if max(abs(a - b) for a, b in zip(old_box, new_box)) <= BOX_TOL_PX:
        return None
    return box_sym_diff(old_box, new_box)
