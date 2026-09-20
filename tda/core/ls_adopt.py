"""Which old Label Studio polygons could be the shape being drawn (spec 3.2).

:mod:`tda.core.ls_import` wrote 11,191 draft keyframes on provisional
``ls:<Label>#<n>`` keys -- the team's own annotations of fourteen desktops,
traced before this tool existed. They are visible-only polygons, often traced
at a neighbouring step, and a draft that is 80 % right is faster to fix than to
redraw. This module is the read side of that: given the class the annotator is
drawing, it offers the drafts that could be it.

**The ordinal is not an identity.** ``ls:RAM Module#2`` means "the second
``RAM Module`` polygon from the left on that frame", so it says nothing about
which ``ram_module.02`` it is; two frames of one desktop number their drafts
independently. Nothing here ever maps an ordinal onto an instance. What is
offered is every draft of the right **class** within a few steps, ranked by
geometry -- how much it overlaps the mask the caller is already looking at (the
editing layer, or the difference map's armed proposal) and then how far its
step is from this one. The annotator picks.

**A draft is reference material.** It is never modified, never deleted, never
compiled and never exported (:func:`tda.core.model.is_provisional`), and it is
never *resized*: a polygon traced on a 1280x800 upload is a different crop of
the scene, so a draft whose RLE does not match this frame's size is skipped and
counted rather than stretched to fit. The masks handed out are fresh arrays, so
a caller may paint on one without touching the database.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from tda.core.db import Db
from tda.core.ls_export import LS_LABEL_MAP_PATH, load_label_map
from tda.core.masks import decode_rle, encode_rle, rle_iou
from tda.core.model import LS_PREFIX, ShapeKeyframe, is_provisional
from tda.core.taxonomy import Taxonomy

__all__ = ["MAIN", "MAX_CANDIDATES", "NEAR_STEPS", "DraftCandidate",
           "draft_label", "drafts_for"]

log = logging.getLogger(__name__)

#: The part name the importer gives every draft (it traced one polygon).
MAIN = "main"
#: How far from the open step a draft may have been traced and still be offered.
NEAR_STEPS = 2
#: How many candidates are handed back at most. A frame can carry twenty screws
#: of one class; decoding twenty 12 MP masks to offer them would cost 240 MB and
#: nobody walks past the first few anyway.
MAX_CANDIDATES = 8


@dataclass(frozen=True)
class DraftCandidate:
    """One old polygon that could be the shape being drawn.

    Attributes:
        key: the provisional instance key, ``ls:<Label>#<n>``.
        step: the step the draft was traced on (not necessarily the open one).
        mask: bool, full-frame, in **this view's** coordinates; the caller's own.
        label: the Label Studio label the key carries.
        iou_with_editing: overlap with the reference mask the caller passed,
            ``0.0`` when it passed none (or an empty one).
    """

    key: str
    step: int
    mask: np.ndarray
    label: str
    iou_with_editing: float


def draft_label(key: str) -> Optional[str]:
    """The Label Studio label inside a draft key, or ``None`` for a real one.

    >>> draft_label("ls:Motherboard Screw#12")
    'Motherboard Screw'
    """
    if not is_provisional(key):
        return None
    return str(key)[len(LS_PREFIX):].rsplit("#", 1)[0]


def _label_classes(tax: Taxonomy, path: str | Path) -> dict[str, str]:
    """``{label: taxonomy class}`` from the label map, hints and drops left out."""
    out: dict[str, str] = {}
    for label, entry in load_label_map(path, tax).items():
        if entry is not None and entry.cls:
            out[str(label)] = str(entry.cls)
    return out


def _main_rle(kf: ShapeKeyframe) -> Optional[dict]:
    """The draft's single polygon, or ``None`` when the row carries no mask."""
    for part in kf.parts or ():
        if part.name == MAIN and part.rle:
            return part.rle
    return None


def _reference_rle(editing: Optional[np.ndarray], hw: tuple[int, int]):
    """The caller's reference mask as an RLE, or ``None`` when there is none.

    Encoded once here rather than decoded per candidate: the ranking then costs
    one ``pycocotools`` call per draft and no ``H x W`` array at all.
    """
    if editing is None:
        return None
    arr = np.asarray(editing, dtype=bool)
    if arr.shape != (int(hw[0]), int(hw[1])) or not arr.any():
        # A layer that has not been touched yet says nothing about which draft
        # this is; ranking by an all-zero mask would only shuffle ties.
        return None
    return encode_rle(arr)


def drafts_for(db: Db, tax: Taxonomy, desktop: int, view: str, step: int, cls: str, *,
               hw: tuple[int, int], near_steps: int = NEAR_STEPS,
               editing: Optional[np.ndarray] = None,
               label_map_path: str | Path = LS_LABEL_MAP_PATH,
               max_candidates: int = MAX_CANDIDATES,
               stats: Optional[dict] = None) -> list[DraftCandidate]:
    """Draft polygons of this view that could be ``cls`` at ``step``.

    Only this desktop and this view, only keys the importer wrote
    (``ls:*``), only drafts whose label maps onto ``cls`` through
    ``configs/ls_label_map.yaml`` -- falling back to the class the importer
    stored on the provisional instance when the map no longer knows the label --
    and only those traced within ``near_steps`` of ``step``.

    Ranking: when ``editing`` holds pixels (the editing layer, or the
    difference map's armed proposal) the candidate that overlaps it most comes
    first and the step distance breaks ties, because that is the only signal
    that tells two screws of one frame apart. With no reference mask there is
    no geometry to go on and the nearest step wins, then the key, so the order
    is deterministic.

    ``stats`` is filled in when a dict is passed: ``considered``, ``wrong_size``
    (skipped, never resized), ``empty`` and ``offered``. Nothing is written to
    the database and every mask handed back belongs to the caller.
    """
    counts = {"considered": 0, "wrong_size": 0, "empty": 0, "offered": 0}
    if stats is not None:
        stats.clear()
        stats.update(counts)
    want, size = str(cls), [int(hw[0]), int(hw[1])]
    lo, hi = int(step) - int(near_steps), int(step) + int(near_steps)
    classes = _label_classes(tax, label_map_path)
    stored: Optional[dict] = None
    reference = _reference_rle(editing, hw)

    ranked: list[tuple[tuple, ShapeKeyframe, dict, float]] = []
    for kf in db.keyframes(int(desktop), str(view)):
        if not is_provisional(kf.instance) or not lo <= int(kf.anchor_step) <= hi:
            continue
        label = draft_label(kf.instance) or ""
        found = classes.get(label)
        if found is None:
            if stored is None:
                stored = db.instances(int(desktop))
            record = stored.get(kf.instance)
            found = None if record is None else str(record.cls)
        if found != want:
            continue
        counts["considered"] += 1
        rle = _main_rle(kf)
        if rle is None:
            counts["empty"] += 1
            continue
        if [int(v) for v in rle["size"]] != size:
            # Traced on another upload: a different crop of the scene, so its
            # percentages do not transfer. Resizing would invent geometry.
            counts["wrong_size"] += 1
            continue
        iou = rle_iou(rle, reference) if reference is not None else 0.0
        order = ((-iou,) if reference is not None else ()) + (
            abs(int(kf.anchor_step) - int(step)), str(kf.instance))
        ranked.append((order, kf, rle, iou))

    ranked.sort(key=lambda item: item[0])
    out = [
        DraftCandidate(key=str(kf.instance), step=int(kf.anchor_step),
                       mask=decode_rle(rle), label=draft_label(kf.instance) or "",
                       iou_with_editing=float(iou))
        for _order, kf, rle, iou in ranked[:max(0, int(max_candidates))]
    ]
    counts["offered"] = len(out)
    if stats is not None:
        stats.update(counts)
    log.info("drafts D%s/%s step %s cls=%s: %d offered, %d considered, "
             "%d wrong size, %d empty", desktop, view, step, want, counts["offered"],
             counts["considered"], counts["wrong_size"], counts["empty"])
    return out
