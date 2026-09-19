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
from tda.core.compiler_visibility import visibility_for
from tda.core.model import FrameOverride, Visibility

__all__ = [
    "BOX_TOL_PX",
    "GEOM_BOX",
    "GEOM_MASK",
    "LABELS_KEY",
    "LABEL_FIELDS",
    "STATE_OWNED_FIELDS",
    "RowValues",
    "disagreement",
    "geom_payload",
    "label_changes",
    "label_text",
    "labels_still_hold",
    "payload_geometry",
    "payload_labels",
    "row_payload",
    "row_values",
    "row_visibility",
    "was_forced",
]

GEOM_MASK = "mask"
GEOM_BOX = "box"

#: How far a box corner may move before a frozen box row is in conflict.
BOX_TOL_PX = 2

#: The compiled fields a truth row carries besides its pixels. A frozen row
#: whose label moved is in disagreement as surely as one whose outline moved:
#: ``visibility`` is written per annotation into every COCO export and is the
#: ground truth of a VLM question, and a row that says ``visible`` where the
#: annotator pressed 2 for ``occluded_partial`` is simply wrong.
#:
#: ``visibility`` counts **only where a frame override states it** (see
#: :func:`label_changes`); ``placement`` always does, because it is the state
#: machine's answer rather than the pixels'.
#:
#: Two fields are deliberately not in here:
#:
#: ``state``
#:     the truth row has no such column. The state machine's decision reaches
#:     the row as ``placement``, and the state itself is read from the event log
#:     when an export needs it, so it cannot go stale against a frozen row.
#: ``occlusion_ratio``
#:     it is a quotient of two pixel counts and nothing else, so the pixels'
#:     own tolerance (:func:`tda.core.masks.is_conflict`) is the authority on
#:     whether it moved. It also has no override to pin it with, so ``keep_old``
#:     could not settle a conflict about it: the disagreement would come
#:     straight back.
LABEL_FIELDS = ("visibility", "placement")

#: Fields an override may not pin: the step table and the event log decide where
#: a part is, and "keep the old value" would mean contradicting them for ever.
STATE_OWNED_FIELDS = frozenset({"placement"})

#: Where a conflict side carries its label changes, next to its geometry.
LABELS_KEY = "labels"


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
def geom_payload(visible_rle: Optional[dict], box,
                 labels: Optional[list[dict]] = None) -> Optional[dict]:
    """One side of a conflict: the RLE, or the box wrapped as ``{"box": [...]}``.

    ``labels`` is :func:`label_changes`' answer, carried under
    :data:`LABELS_KEY` on the same object -- the conflict table has one JSON
    column per side and no room for a third, and the two belong together
    anyway: they are both "what this side says about the instance".
    """
    if visible_rle is not None:
        payload: Optional[dict] = dict(visible_rle)
    elif box is not None:
        payload = {"box": [float(v) for v in box]}
    else:
        payload = None
    if labels:
        payload = dict(payload or {})
        payload[LABELS_KEY] = [dict(change) for change in labels]
    return payload


def row_payload(row: dict) -> Optional[dict]:
    """The same, for a stored truth row."""
    return geom_payload(row.get("visible_rle"), row.get("box"))


def payload_labels(payload: Optional[dict]) -> list[dict]:
    """The label changes a conflict side carries, ``[]`` when it carries none."""
    if not payload:
        return []
    return list(payload.get(LABELS_KEY) or [])


def payload_geometry(payload: Optional[dict]) -> tuple[str, Optional[dict], Optional[list]]:
    """``(geom_type, visible_rle, box)`` of a conflict side; empty -> a mask row.

    The label changes travelling on the same object are not geometry and are
    stripped here, so everything that decodes a conflict side sees the RLE it
    expects.
    """
    if not payload:
        return (GEOM_MASK, None, None)
    if "box" in payload:
        return (GEOM_BOX, None, [float(v) for v in payload["box"]])
    if "counts" not in payload:
        return (GEOM_MASK, None, None)  # labels only: this side has no pixels
    if LABELS_KEY not in payload:
        return (GEOM_MASK, payload, None)
    return (GEOM_MASK, {k: v for k, v in payload.items() if k != LABELS_KEY}, None)


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


def row_visibility(row: dict) -> str:
    """What a stored row's **own geometry** says its visibility is.

    The compiler's ladder (:func:`tda.core.compiler_visibility.visibility_for`)
    read off the row rather than off an array, so it costs one bbox measurement
    on the run lengths and no decode. The two branches that do not go through
    the ladder are reproduced from the same place they come from (spec 3.3
    step 7): a bench box carries no occlusion and is plainly ``visible``, and a
    row with no geometry at all is ``out_of_view``.

    Its use is :func:`was_forced`: a row whose stored label is *not* this is one
    a human typed.
    """
    if row.get("geom_type") == GEOM_BOX:
        return Visibility.VISIBLE.value
    rle = row.get("visible_rle")
    if not rle:
        return Visibility.OUT_OF_VIEW.value
    return visibility_for(masks.rle_min_side(rle),
                          float(row.get("occlusion_ratio") or 0.0))


def was_forced(row: dict) -> bool:
    """Did a human put this row's ``visibility`` there? (spec 4.3, the 1-7 keys)

    A frozen row carries no flag saying so, and it needs none: a label the
    compiler derived is a function of the row's own geometry, so a label that
    is *not* that function's answer can only have come from a
    :class:`~tda.core.model.FrameOverride`. That is what makes "the annotator
    pressed Ctrl+Z on the override" a disagreement the frozen row has to be
    told about -- otherwise it keeps a label neither a human nor its own pixels
    stand behind, and the export writes it beside a segmentation that
    contradicts it.
    """
    return (row.get("visibility") or None) != row_visibility(row)


def label_changes(row: dict, compiled_inst: CompiledInstance,
                  override: Optional[FrameOverride] = None) -> list[dict]:
    """The non-geometric fields of a frozen row a re-compilation disagrees with.

    ``[{"field", "old", "new"}, ...]`` in :data:`LABEL_FIELDS` order, empty when
    the labels still hold. This is the other half of :func:`disagreement`: the
    pixels can be identical while the row says something different about the
    instance, and that is exactly what a confirmed frame was silently throwing
    away -- the refresh saw "no disagreement", left the frozen row saying
    ``visible`` and stamped the digest, so nothing ever looked at the frame
    again. ``visibility`` is written into every COCO annotation and is the
    ground truth of a VLM question.

    ``visibility`` is compared when **either side of it was a human's**: the
    frame carries an override stating it now (``override``), or the frozen row
    itself was forced (:func:`was_forced`, which is how a *removed* override is
    caught -- Ctrl+Z on the 1-7 keys over a frame somebody had confirmed).
    Where neither is, the label is a pure function of the pixels (the occlusion
    ratio and the size of the visible box, spec 3.3 step 7), so two outlines
    :func:`disagreement` calls the same annotation cannot honestly disagree
    about it -- a one-pixel re-trace that happens to cross the 6 px
    ``too_small`` threshold is noise, not a decision.

    ``placement`` needs no such gate: it comes from the state machine.

    A stored ``NULL`` is "not recorded" rather than a value to disagree with.
    """
    stated = override is not None and bool(override.visibility)
    out: list[dict] = []
    for field in LABEL_FIELDS:
        if field == "visibility" and not (stated or was_forced(row)):
            continue
        old = row.get(field)
        new = getattr(compiled_inst, field)
        if (old or None) != (new or None):
            out.append({"field": field, "old": old, "new": new})
    return out


def labels_still_hold(labels: list[dict], compiled_inst: CompiledInstance) -> bool:
    """Does a queued conflict's ``new`` side still describe this compilation?

    What :meth:`~tda.core.truth_resolve.ResolveMixin.resolve_conflict` asks
    before accepting one: the labels may have moved again since it was queued,
    and confirming a value nobody has seen is the thing ``accept_new`` exists
    not to do.
    """
    for change in labels:
        field = change.get("field")
        if field not in LABEL_FIELDS:
            continue
        if (getattr(compiled_inst, field, None) or None) != (change.get("new") or None):
            return False
    return True


def label_text(labels: list[dict]) -> str:
    """``"visibility: visible -> occluded_partial"`` for a review-queue line."""
    return ", ".join(
        f"{c.get('field')}: {c.get('old')} → {c.get('new')}" for c in labels
    )


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
