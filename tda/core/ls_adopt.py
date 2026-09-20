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
independently. Nothing here ever maps an ordinal onto an instance, and nothing
here ever *ranks* by one either: on the classes that carry most of the drafts
-- connectors, screws, RAM, latches -- a frame holds fifteen or thirty of them
and an alphabetical order over meaningless ordinals means the right one is
somewhere in the middle. So the caller passes **where the annotator is
pointing** and that decides: the draft under the cursor first, then the ones
around it, then the neighbouring steps. Every draft of the class stays in the
list; none of them is cut off.

Labels are noisy, so after the drafts of the wanted class come the drafts of
*any* class whose mask contains the cursor -- what the annotator is pointing at
may have been labelled something else five months ago. Each candidate carries
its own label and class, so the status bar can say what it was called.

**A draft is reference material.** It is never modified, never deleted, never
compiled and never exported (:func:`tda.core.model.is_provisional`), and it is
never *resized*: a polygon traced on a 1280x800 upload is a different crop of
the scene, so a draft whose RLE does not match this frame's size is skipped and
counted rather than stretched to fit. A draft anchored in another **pose
segment** is another pose of the machine and is not offered either (the drafts
were all imported into segment 0, so the step range is what tells them apart).

**Nothing is decoded to rank it.** Ordering runs on the stored run lengths --
:func:`tda.core.masks.rle_bbox` for the distance, :func:`~tda.core.masks.rle_contains`
for the "is the cursor inside it" test -- and only the candidate actually being
shown is turned into an ``H x W`` array, by asking for :attr:`DraftCandidate.mask`.
At 4032x3040 that is the difference between 0.06 ms and 12 MB per draft.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from tda.core.db import Db
from tda.core.ls_export import LS_LABEL_MAP_PATH, load_label_map
from tda.core.masks import decode_rle, encode_rle, rle_bbox, rle_contains, rle_iou
from tda.core.model import LS_PREFIX, FrameKey, ShapeKeyframe, is_provisional
from tda.core.taxonomy import Taxonomy

__all__ = ["MAIN", "NEAR_STEPS", "DraftCandidate", "draft_label", "draft_rle",
           "drafts_for"]

log = logging.getLogger(__name__)

#: The part name the importer gives every draft (it traced one polygon).
MAIN = "main"
#: How far from the open step a draft may have been traced and still be offered.
NEAR_STEPS = 2


@dataclass
class DraftCandidate:
    """One old polygon that could be the shape being drawn.

    Attributes:
        key: the provisional instance key, ``ls:<Label>#<n>``.
        step: the step the draft was traced on (not necessarily the open one).
        label: the Label Studio label the key carries.
        cls: the taxonomy class that label maps onto.
        rle: the stored mask, untouched; :attr:`mask` decodes it on demand.
        keyframe_id: ``shape_keyframe.id`` of the draft, so a commit that
            adopted it can point back at the exact row (spec 3.1 初稿引用).
        same_class: whether this is a draft of the class that was asked for, or
            one of another class that happens to lie under the cursor.
        distance: from the cursor to the draft -- ``0.0`` when the cursor is
            inside the mask, otherwise the distance to its bounding box;
            ``inf`` when the caller gave no cursor.
        iou_with_editing: overlap with the reference mask the caller passed,
            ``0.0`` when it passed none (or an empty one).
    """

    key: str
    step: int
    label: str
    cls: str
    rle: dict
    keyframe_id: Optional[int] = None
    same_class: bool = True
    distance: float = math.inf
    iou_with_editing: float = 0.0
    _mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def mask(self) -> np.ndarray:
        """The draft as a full-frame bool array; decoded once, then cached.

        The array belongs to the caller: it is decoded from the stored RLE, so
        painting on it cannot reach the database.
        """
        if self._mask is None:
            self._mask = decode_rle(self.rle)
        return self._mask

    @property
    def decoded(self) -> bool:
        """Is the ``H x W`` array being held right now?"""
        return self._mask is not None

    def release(self) -> None:
        """Drop the decoded array; :attr:`mask` would decode it again."""
        self._mask = None

    @property
    def box(self):
        """``(x0, y0, x1, y1)`` of the draft, off the run lengths."""
        return rle_bbox(self.rle)

    @property
    def area(self) -> int:
        """Pixels the draft covers, off the run lengths."""
        from tda.core.masks import rle_area

        return rle_area(self.rle)


def draft_label(key: str) -> Optional[str]:
    """The Label Studio label inside a draft key, or ``None`` for a real one.

    >>> draft_label("ls:Motherboard Screw#12")
    'Motherboard Screw'
    """
    if not is_provisional(key):
        return None
    return str(key)[len(LS_PREFIX):].rsplit("#", 1)[0]


def draft_rle(db: Db, desktop: int, view: str, key: str,
              keyframe_id: Optional[int]) -> Optional[dict]:
    """The stored mask of one adopted draft, by its keyframe id.

    What a commit needs to measure how much of the mask it is about to write
    really came from the draft it claims (``overlap_px``), long after the
    candidate list is gone.
    """
    for kf in db.keyframes(int(desktop), str(view), str(key)):
        if keyframe_id is None or kf.id == int(keyframe_id):
            return _main_rle(kf)
    return None


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


def _box_distance(box, xy: tuple[float, float]) -> float:
    """Distance from a point to a box; ``0.0`` inside it."""
    if box is None:
        return math.inf
    x0, y0, x1, y1 = box
    dx = max(x0 - xy[0], 0.0, xy[0] - (x1 - 1))
    dy = max(y0 - xy[1], 0.0, xy[1] - (y1 - 1))
    return math.hypot(dx, dy)


def _segment_window(db: Db, desktop: int, view: str, step: int,
                    lo: int, hi: int) -> tuple[int, int]:
    """Clamp the step window to the pose segment the open frame belongs to.

    Shapes do not carry across a pose break (spec 2.5), and the drafts were all
    imported with ``pose_segment = 0`` and are never re-keyed by a re-cut, so
    the segment's **step range** is what tells a draft of this pose from a
    draft of the one next door.  No segment row: no clamping, which is the
    state of every desktop that has not been through ``split_pose_segments``.
    """
    row = db.pose_segment_for(FrameKey(int(desktop), int(step), str(view))) or {}
    start, end = row.get("start_step"), row.get("end_step")
    if start is not None:
        lo = max(lo, int(start))
    if end is not None:
        hi = min(hi, int(end))
    return lo, hi


def drafts_for(db: Db, tax: Taxonomy, desktop: int, view: str, step: int, cls: str, *,
               hw: tuple[int, int], near_steps: int = NEAR_STEPS,
               editing: Optional[np.ndarray] = None,
               cursor: Optional[tuple[float, float]] = None,
               label_map_path: str | Path = LS_LABEL_MAP_PATH,
               max_candidates: Optional[int] = None,
               stats: Optional[dict] = None) -> list[DraftCandidate]:
    """Draft polygons of this view that could be ``cls`` at ``step``.

    Only this desktop and this view, only keys the importer wrote (``ls:*``),
    only drafts whose label maps onto ``cls`` through
    ``configs/ls_label_map.yaml`` -- falling back to the class the importer
    stored on the provisional instance when the map no longer knows the label
    -- and only those traced within ``near_steps`` of ``step`` **and inside the
    open frame's pose segment**.

    Ranking:

    * with a ``cursor`` (image coordinates), by where the annotator is
      pointing: inside the mask first, then by distance to the bounding box,
      then ``|Δstep|``, then the key. Drafts of *other* classes whose mask
      contains the cursor follow the whole same-class list.
    * with no cursor but a non-empty ``editing`` reference (the editing layer,
      or the difference map's armed proposal), by overlap with it, then step
      distance.
    * with neither, by step distance then key, which is deterministic.

    ``max_candidates`` is ``None`` by default: every draft of the class stays
    reachable, because the masks are not decoded until one is shown. ``stats``
    is filled in when a dict is passed: ``considered``, ``wrong_size``
    (skipped, never resized), ``empty`` and ``offered``. Nothing is written.
    """
    counts = {"considered": 0, "wrong_size": 0, "empty": 0, "offered": 0}
    if stats is not None:
        stats.clear()
        stats.update(counts)
    want, size = str(cls), [int(hw[0]), int(hw[1])]
    lo, hi = _segment_window(db, desktop, view, step,
                             int(step) - int(near_steps), int(step) + int(near_steps))
    classes = _label_classes(tax, label_map_path)
    stored: Optional[dict] = None
    reference = _reference_rle(editing, hw)

    mine: list[tuple[tuple, DraftCandidate]] = []
    others: list[tuple[tuple, DraftCandidate]] = []
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
        same = found == want
        if not same and cursor is None:
            continue                     # nothing to point at: class only
        rle = _main_rle(kf)
        if rle is None:
            if same:
                counts["considered"] += 1
                counts["empty"] += 1
            continue
        if [int(v) for v in rle["size"]] != size:
            # Traced on another upload: a different crop of the scene, so its
            # percentages do not transfer. Resizing would invent geometry.
            if same:
                counts["considered"] += 1
                counts["wrong_size"] += 1
            continue
        inside = cursor is not None and rle_contains(rle, int(cursor[0]), int(cursor[1]))
        if not same and not inside:
            continue                     # another class is offered only under the cursor
        counts["considered"] += 1
        distance = (math.inf if cursor is None
                    else (0.0 if inside else _box_distance(rle_bbox(rle), cursor)))
        candidate = DraftCandidate(
            key=str(kf.instance), step=int(kf.anchor_step), label=label,
            cls=str(found or ""), rle=rle, keyframe_id=kf.id, same_class=same,
            distance=distance,
            iou_with_editing=(rle_iou(rle, reference) if reference is not None else 0.0),
        )
        step_distance = abs(int(kf.anchor_step) - int(step))
        if cursor is not None:
            order = (0 if inside else 1, distance, step_distance, candidate.key)
        elif reference is not None:
            order = (-candidate.iou_with_editing, step_distance, candidate.key)
        else:
            order = (step_distance, candidate.key)
        (mine if same else others).append((order, candidate))

    mine.sort(key=lambda item: item[0])
    others.sort(key=lambda item: item[0])
    out = [candidate for _order, candidate in mine + others]
    if max_candidates is not None:
        out = out[:max(0, int(max_candidates))]
    counts["offered"] = len(out)
    if stats is not None:
        stats.update(counts)
    log.info("drafts D%s/%s step %s cls=%s cursor=%s: %d offered (%d other class), "
             "%d considered, %d wrong size, %d empty", desktop, view, step, want,
             cursor, counts["offered"], sum(1 for c in out if not c.same_class),
             counts["considered"], counts["wrong_size"], counts["empty"])
    return out
