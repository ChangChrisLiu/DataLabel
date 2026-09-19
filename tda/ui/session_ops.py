"""Undoable operations over the annotation inputs, and the reads they share.

:mod:`tda.ui.session_edit` decides *what* an edit means; this module is the
layer underneath it: the vocabulary both it and :mod:`tda.ui.session_tasks`
read the database with (which steps can be compiled, where an instance is, what
its keyframe chain looks like), plus the five handlers that write one
:class:`~tda.ui.commands.Op` back.

Op payloads are *states*, not deltas
------------------------------------
Each handler is registered as both the ``do`` and the ``undo`` direction of its
op kind: an op's ``payload`` and its ``inverse`` are two states of the same
rows, so redoing and undoing are the same call with a different argument.  A
keyframe that did not exist yet is written as ``exists: False``, which the
handler turns into a delete.  Both directions share one mutable ``ref`` dict
holding the row id, and a redo re-inserts under that same id: several ops may
name one keyframe, so an id that changed on a redo would strand all the others.

Nothing here is Qt-aware, and nothing here decides policy.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from tda.core import masks
from tda.core.db import Db
from tda.core.model import (
    FrameKey,
    FrameOverride,
    OccluderMask,
    PairOverride,
    Placement,
    ShapeKeyframe,
    ShapePart,
    ZOrderRec,
)
from tda.core.compiler import select_keyframe
from tda.core.states import needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth import VERIFIED, TruthService
from tda.ui.session_api import SessionRefusal
from tda.core.truth_inputs import (
    InputCache,
    annotatable_steps,
    instances_of,
    pose_segment_of,
    state_of,
)

__all__ = [
    "BENCH_KINDS",
    "DIRECTIONS",
    "FORWARD",
    "GEOM_BOX",
    "GEOM_MASK",
    "IN_CHASSIS",
    "MAIN",
    "ON_BENCH",
    "REVERSE",
    "annotatable_steps",
    "apply_frame_override",
    "apply_keyframes",
    "apply_occluder",
    "apply_pair_override",
    "apply_zorder",
    "as_mask",
    "chain_for",
    "chain_steps",
    "default_anchor",
    "keyframe_state",
    "new_keyframe",
    "occluder_union",
    "placement_of",
    "refresh_steps",
    "settle",
    "mask_steps",
    "segment_steps",
    "write_zorder",
]

IN_CHASSIS = Placement.IN_CHASSIS.value
ON_BENCH = Placement.ON_BENCH.value
GEOM_MASK = "mask"
GEOM_BOX = "box"
MAIN = "main"

#: Browsing directions a split keyframe can be anchored in (spec 3.3).
REVERSE = "reverse"
FORWARD = "forward"
DIRECTIONS = (REVERSE, FORWARD)

#: Geometry kinds :func:`tda.core.states.needs_geom` reports for a bench part.
BENCH_KINDS = ("box",)


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #
def refresh_steps(db: Db, truth: TruthService, desktop: int, view: str,
                  steps: Iterable[int]) -> dict:
    """Recompile these steps, keeping each step's problems under its own key.

    A thin alias for :meth:`~tda.core.truth.TruthService.refresh_range` with
    ``per_step``: the session needs to know *where* a missing shape is in order
    to fill the queue of spec 4.4, and there is one implementation of the sweep.
    """
    return truth.refresh_range(desktop, view, steps, per_step=True)


def is_verified(db: Db, desktop: int, view: str, step: int) -> bool:
    """Has a human frozen this frame? (spec 3.4)"""
    frame = db.get_frame(FrameKey(desktop, int(step), view)) or {}
    return frame.get("review_status") == VERIFIED


def settle(db: Db, truth: TruthService, desktop: int, view: str, steps: Iterable[int],
           current: Optional[int] = None) -> dict:
    """Bring the frame in hand up to date; defer the rest of the interval.

    The rule every write goes through (spec 3.4 read as "an unverified frame's
    compiled rows are a cache of a pure function"): compile the frame the
    annotator is looking at, leave the other unverified frames stale for
    whoever visits them, and hand the **frozen** ones to the persisted re-check
    queue, because a frozen row is the one thing the compiler may not overwrite
    and a disagreement nobody looks for is a conflict that never gets raised.

    Returns what :func:`refresh_steps` does plus ``rechecks`` (the frames
    queued), ``compiled`` (the ones actually done now) and ``frame`` -- the
    :class:`~tda.core.compiler.CompiledFrame` of the current step, handed back
    so the session installs it instead of compiling the same frame again.
    """
    wanted = annotatable_steps(db, desktop, view, steps)
    now = [int(current)] if current is not None and int(current) in wanted else wanted[:1]
    stats: dict = {"updated": 0, "conflicts": 0, "standing": 0, "skipped": 0,
                   "problems": {}, "frame": None}
    for step in now:
        one = truth.refresh(FrameKey(desktop, int(step), view), want_compiled=True)
        for counter in ("updated", "conflicts", "standing", "skipped"):
            stats[counter] += one[counter]
        stats["problems"][int(step)] = list(one["problems"])
        stats["frame"] = one["compiled"]  # the caller installs it: no second compile
    deferred = [s for s in wanted if s not in now and is_verified(db, desktop, view, s)]
    stats["rechecks"] = truth.queue_rechecks(desktop, view, deferred)
    stats["compiled"] = now
    return stats


def as_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Validate an edited mask against the frame canvas and make it boolean."""
    arr = np.asarray(mask)
    if arr.shape != tuple(hw):
        raise SessionRefusal(f"mask has shape {arr.shape!r}, expected {tuple(hw)!r}")
    return arr.astype(bool, copy=False)


def placement_of(db: Db, tax: Taxonomy, key: FrameKey, instance: str,
                 cache: Optional[InputCache] = None) -> str:
    """Where the instance is at this step; the chassis chain when it is unknown."""
    inst = state_of(db, tax, key.desktop, key.step, cache).get(instance)
    return IN_CHASSIS if inst is None else inst.placement


def occluder_union(db: Db, key: FrameKey, hw: tuple[int, int]) -> np.ndarray:
    """Every occluder layer of one frame in a single mask."""
    out = np.zeros(hw, dtype=bool)
    for occ in db.occluders(key):
        if occ.rle:
            out |= masks.decode_rle(occ.rle)
    return out


def chain_for(db: Db, key: FrameKey, instance: str, seg: int,
              placement: str) -> list[ShapeKeyframe]:
    """The instance's keyframes of one pose segment and one placement chain.

    The ``in_chassis`` and ``on_bench`` chains of an instance are independent
    (spec 3.3), which is what lets one part carry a mask before its removal step
    and a staging-area box after it.
    """
    return [
        kf
        for kf in db.keyframes(key.desktop, key.view, instance)
        if kf.pose_segment == seg and kf.placement == placement
    ]


def default_anchor(db: Db, tax: Taxonomy, key: FrameKey, instance: str, seg: int,
                   placement: str, cache: Optional[InputCache] = None) -> int:
    """Anchor for a shape drawn for the first time (spec 3.3, 关键帧与标注方向).

    An instance's lifetime is fixed by the step table before anyone draws
    anything, so the default anchor is the **last logical step of this pose
    segment where the instance still needs geometry in this placement** -- which
    makes the one shape cover the instance's whole life in that chain.  Falls
    back to the current step when the state machine says the instance needs
    geometry nowhere (a shape drawn against the rules is still kept).
    """
    instances = instances_of(db, key.desktop, cache)
    bench_roi = db.bench_roi(key.desktop, key.view, seg)
    best: Optional[int] = None
    for rec in db.steps(key.desktop):
        step = rec.step
        if pose_segment_of(db, FrameKey(key.desktop, step, key.view), cache) != seg:
            continue
        state = state_of(db, tax, key.desktop, step, cache)
        held = state.get(instance)
        if held is None or held.placement != placement:
            continue
        if instance in needs_geom(instances, state, tax, bench_roi=bench_roi):
            best = step
    return key.step if best is None else best


def chain_steps(db: Db, tax: Taxonomy, key: FrameKey, instance: str, seg: int,
                placement: str, chain: list[ShapeKeyframe], target: ShapeKeyframe,
                cache: Optional[InputCache] = None) -> list[int]:
    """Steps whose compilation would select ``target`` out of ``chain``.

    The same rule as :meth:`tda.core.truth.TruthService.affected_steps`, but
    over a chain the caller may have *simulated*: that is what lets the session
    answer "影响 N 帧" (spec 4.3) before anything is written.
    """
    instances = instances_of(db, key.desktop, cache)
    bench_roi = db.bench_roi(key.desktop, key.view, seg)
    out: list[int] = []
    for rec in db.steps(key.desktop):
        step = rec.step
        if pose_segment_of(db, FrameKey(key.desktop, step, key.view), cache) != seg:
            continue
        state = state_of(db, tax, key.desktop, step, cache)
        held = state.get(instance)
        if held is None or held.placement != placement:
            continue
        if instance not in needs_geom(instances, state, tax, bench_roi=bench_roi):
            continue
        if select_keyframe(chain, step) is target:
            out.append(step)
    return annotatable_steps(db, key.desktop, key.view, out)


def segment_steps(db: Db, tax: Taxonomy, key: FrameKey, instance: str, seg: int,
                  cache: Optional[InputCache] = None,
                  geom: Optional[str] = None) -> list[int]:
    """Annotatable steps of this pose segment where ``instance`` carries geometry.

    ``geom`` narrows it to one kind: ``"mask"`` for the steps where the instance
    is a *layer*, which is the only place the layering can be argued about.
    """
    instances = instances_of(db, key.desktop, cache)
    bench_roi = db.bench_roi(key.desktop, key.view, seg)
    out = []
    for rec in db.steps(key.desktop):
        step = rec.step
        if pose_segment_of(db, FrameKey(key.desktop, step, key.view), cache) != seg:
            continue
        needs = needs_geom(instances, state_of(db, tax, key.desktop, step, cache), tax,
                           bench_roi=bench_roi)
        if instance in needs and (geom is None or needs[instance] == geom):
            out.append(step)
    return annotatable_steps(db, key.desktop, key.view, out)


def mask_steps(db: Db, tax: Taxonomy, key: FrameKey, instance: str, seg: int,
               cache: Optional[InputCache] = None) -> list[int]:
    """Steps of this pose segment where ``instance`` is a mask layer."""
    return segment_steps(db, tax, key, instance, seg, cache, geom=GEOM_MASK)


# --------------------------------------------------------------------------- #
# keyframe states
# --------------------------------------------------------------------------- #
def new_keyframe(key: FrameKey, instance: str, seg: int, anchor: int, placement: str,
                 parts: list[ShapePart], geom_type: str) -> ShapeKeyframe:
    """An unsaved keyframe for one instance of one chain."""
    return ShapeKeyframe(
        id=None, instance=instance, desktop=key.desktop, view=key.view,
        pose_segment=seg, anchor_step=anchor, placement=placement,
        geom_type=geom_type, parts=parts,
    )


def _part_state(part: ShapePart) -> dict:
    return {"name": part.name, "rle": part.rle,
            "box": None if part.box is None else [float(v) for v in part.box]}


def _parts_of(state: dict) -> list[ShapePart]:
    return [
        ShapePart(p["name"], p.get("rle"),
                  None if p.get("box") is None else tuple(float(v) for v in p["box"]))
        for p in state["parts"]
    ]


def keyframe_state(kf: Optional[ShapeKeyframe], ref: dict,
                   template: Optional[ShapeKeyframe] = None) -> dict:
    """One keyframe as an undo-friendly dict; ``None`` becomes "it did not exist".

    ``ref`` is the mutable holder of the row id that both directions of the op
    share, so an undo that deletes the row and a redo that re-creates it stay in
    step with each other.
    """
    if kf is None:
        base = template
        return {"ref": ref, "exists": False,
                "instance": None if base is None else base.instance,
                "desktop": None if base is None else base.desktop,
                "view": None if base is None else base.view,
                "pose_segment": None if base is None else base.pose_segment,
                "anchor_step": None, "placement": None, "geom_type": None, "parts": []}
    return {
        "ref": ref, "exists": True, "instance": kf.instance, "desktop": kf.desktop,
        "view": kf.view, "pose_segment": kf.pose_segment, "anchor_step": kf.anchor_step,
        "placement": kf.placement, "geom_type": kf.geom_type, "version": kf.version,
        "parts": [_part_state(p) for p in kf.parts],
    }


# --------------------------------------------------------------------------- #
# the five handlers -- one per op kind of tda.ui.commands.KINDS
# --------------------------------------------------------------------------- #
def apply_keyframes(db: Db, truth: TruthService, payload: dict,
                    current: Optional[int] = None) -> dict:
    """Write back a list of keyframe states (and a layer order), then settle."""
    with db.transaction():
        for state in payload.get("keyframes", ()):
            _apply_keyframe(db, state)
        zorder = payload.get("zorder")
        if zorder is not None:
            write_zorder(db, payload["desktop"], payload["view"], zorder)
    return settle(db, truth, payload["desktop"], payload["view"], payload["steps"], current)


def _apply_keyframe(db: Db, state: dict) -> None:
    """Make one keyframe row look like ``state``; ``exists: False`` removes it.

    The row id in ``ref`` survives a delete: several ops can name the same
    keyframe (a create and every re-trace after it), and each holds its own
    ``ref``, so a redo that re-inserted under a *fresh* id would leave the other
    ops pointing at a row that no longer exists -- they would each insert a
    keyframe of their own and the instance would end up with duplicates that
    nothing can undo. Re-inserting with ``keep_id`` keeps every op in agreement.
    """
    ref = state["ref"]
    kid = ref.get("keyframe_id")
    if not state["exists"]:
        if kid is not None:
            db.delete_keyframe(kid)
        return  # ... but the id is kept, so a redo can restore that very row
    existing = None
    if kid is not None:
        existing = next(
            (kf for kf in db.keyframes(state["desktop"], state["view"], state["instance"])
             if kf.id == kid),
            None,
        )
    parts = _parts_of(state)
    if existing is not None:
        existing.anchor_step = state["anchor_step"]
        existing.placement = state["placement"]
        existing.geom_type = state["geom_type"]
        existing.parts = parts
        db.update_keyframe(existing)
        return
    kf = new_keyframe(
        FrameKey(state["desktop"], state["anchor_step"], state["view"]),
        state["instance"], state["pose_segment"], state["anchor_step"],
        state["placement"], parts, state["geom_type"],
    )
    kf.version = int(state.get("version") or 1)
    ref["keyframe_id"] = db.add_keyframe(kf, keep_id=kid)


def write_zorder(db: Db, desktop: int, view: str, state: dict) -> None:
    """Store one ``(view, pose segment)`` layer order from an op payload."""
    db.set_zorder(
        ZOrderRec(desktop, view, state["pose_segment"],
                  [tuple(entry) for entry in state["order"]], version=state["version"])
    )


def apply_zorder(db: Db, truth: TruthService, payload: dict,
                 current: Optional[int] = None) -> dict:
    """Write one stored layer order back and settle the frames it reaches."""
    with db.transaction():
        write_zorder(db, payload["desktop"], payload["view"], payload)
    return settle(db, truth, payload["desktop"], payload["view"], payload["steps"], current)


def apply_pair_override(db: Db, truth: TruthService, payload: dict,
                        current: Optional[int] = None) -> dict:
    """Add or remove one "above beats below" exception, then settle."""
    po = PairOverride(payload["desktop"], payload["view"], payload["pose_segment"],
                      payload["above"], payload["below"])
    with db.transaction():
        if payload["exists"]:
            db.set_pair_override(po)
        else:
            db.delete_pair_override(po)
    return settle(db, truth, payload["desktop"], payload["view"], payload["steps"], current)


def apply_frame_override(db: Db, truth: TruthService, payload: dict,
                         current: Optional[int] = None) -> dict:
    """Write, or drop, one instance's single-frame override, then settle."""
    key = FrameKey(payload["desktop"], payload["step"], payload["view"])
    with db.transaction():
        if payload["exists"]:
            db.set_frame_override(
                FrameOverride(key, payload["instance"], payload.get("visible_rle"),
                              payload.get("visibility"))
            )
        else:
            db.delete_frame_override(key, payload["instance"])
    return settle(db, truth, key.desktop, key.view, payload["steps"], current)


def apply_occluder(db: Db, truth: TruthService, payload: dict,
                   current: Optional[int] = None) -> dict:
    """Write, or drop, one occluder layer of one frame, then settle."""
    key = FrameKey(payload["desktop"], payload["step"], payload["view"])
    with db.transaction():
        if payload["exists"]:
            db.set_occluder(OccluderMask(key, payload["occluder_type"], payload["rle"]))
        else:
            db.delete_occluder(key, payload["occluder_type"])
    return settle(db, truth, key.desktop, key.view, payload["steps"], current)
