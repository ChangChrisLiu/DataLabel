"""What one annotation gesture writes, without Qt (spec 4.3 编辑语义).

:class:`tda.ui.session.AnnotationSession` owns "which frame is open" and the
signals the panels listen to; this module is everything it *writes*, as plain
functions over a :class:`~tda.core.db.Db` and a
:class:`~tda.core.truth.TruthService`.  That split keeps the session thin and
makes the rules below testable without a ``QApplication``.

The four scopes of spec 4.3 -- re-trace the keyframe, split it, override this
one frame, change the layering -- are :func:`commit_edit` and
:func:`commit_pair_override`; :func:`suggest_scope` is which of them a given
edit defaults to.  The operations they log, and the reads they share with
:mod:`tda.ui.session_tasks`, live in :mod:`tda.ui.session_ops`.

What every mutating function does
---------------------------------
1. Write the annotation inputs inside one
   :meth:`~tda.core.db.Db.transaction`, together with the ``op_log`` entry that
   records the change and its inverse (spec 4.6).
2. Work out which logical steps the change reaches: for a shape that is
   :meth:`~tda.core.truth.TruthService.affected_steps`, for a single-frame
   override it is this frame alone.
3. Recompile exactly those frames and report how many frozen rows disagreed.

They all return ``{"affected", "conflicts", "problems", "op"}``: the steps
actually recompiled, the number of queued conflicts, the compiler's problems per
step, and the :class:`~tda.ui.commands.Op` the session pushes onto its undo
stack.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from tda.core import masks
from tda.core.compiler import CompiledFrame, select_keyframe
from tda.core.db import Db
from tda.core.model import (
    FrameKey,
    FrameOverride,
    OccluderMask,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    ZOrderRec,
)
from tda.core.truth import VERIFIED, TruthService
from tda.core.truth_inputs import InputCache, frame_hw, pose_segment_of
from tda.ui import session_api as api
from tda.ui.commands import Op
from tda.ui.session_ops import (
    DIRECTIONS,
    FORWARD,
    GEOM_BOX,
    GEOM_MASK,
    MAIN,
    REVERSE,
    annotatable_steps,
    apply_frame_override,
    apply_keyframes,
    apply_occluder,
    apply_pair_override,
    apply_zorder,
    as_mask,
    chain_for,
    chain_steps,
    default_anchor,
    keyframe_state,
    new_keyframe,
    occluder_union,
    placement_of,
    refresh_steps,
    segment_steps,
    write_zorder,
)
from tda.ui.session_tasks import task_card_for

__all__ = [
    "DIRECTIONS",
    "FORWARD",
    "REVERSE",
    "ZORDER_HINT_FRAC",
    "annotatable_steps",
    "apply_frame_override",
    "apply_keyframes",
    "apply_occluder",
    "apply_pair_override",
    "apply_zorder",
    "commit_box",
    "commit_edit",
    "commit_occluder",
    "commit_pair_override",
    "default_anchor",
    "preview",
    "refresh_steps",
    "require_instance",
    "split_zorder_scope",
    "set_visibility",
    "set_zorder_move",
    "suggest_scope",
    "task_card_for",
]

#: How much of the *changed* pixels has to land inside another instance before
#: the default scope becomes "change the layering" (spec 4.3).
ZORDER_HINT_FRAC = 0.6

#: ``suggest_scope`` answers, and what :meth:`AnnotationSession.commit_edit`
#: accepts besides the three :data:`~tda.ui.session_api.COMMIT_SCOPES`.
SCOPE_ZORDER_ABOVE = "zorder:above:"
SCOPE_ZORDER_BELOW = "zorder:below:"


def split_zorder_scope(scope: str) -> Optional[tuple[str, bool]]:
    """``(other instance, this one goes above)`` for a layering scope, else ``None``."""
    for prefix, above in ((SCOPE_ZORDER_ABOVE, True), (SCOPE_ZORDER_BELOW, False)):
        if scope.startswith(prefix):
            return scope[len(prefix):], above
    return None


def _result(db: Db, truth: TruthService, key: FrameKey, steps: Sequence[int], op: Op,
            extra: Optional[dict] = None) -> dict:
    """Recompile ``steps`` and package what the session reports to the panels."""
    affected = annotatable_steps(db, key.desktop, key.view, steps)
    stats = refresh_steps(db, truth, key.desktop, key.view, affected)
    out = {"affected": affected, "conflicts": stats["conflicts"],
           "problems": stats["problems"], "op": op}
    out.update(extra or {})
    return out


# --------------------------------------------------------------------------- #
# pixels
# --------------------------------------------------------------------------- #
def commit_edit(db: Db, truth: TruthService, key: FrameKey, instance: str,
                mask: np.ndarray, scope: str, direction: str = REVERSE,
                annotator: str = "system") -> dict:
    """Write one pixel edit back with the scope the annotator chose (spec 4.3).

    ``scope`` is one of :data:`tda.ui.session_api.COMMIT_SCOPES`:

    ``keyframe``
        Re-trace the keyframe in force at this step, bumping its version, so the
        change reaches every frame of its interval.  When no keyframe applies
        yet the shape is created with the default anchor of
        :func:`~tda.ui.session_ops.default_anchor` and the instance is appended
        to the layer order, i.e. on top of everything (spec 4.2).
    ``split``
        A new version of the shape from this step on, in the browsing direction
        (spec 3.3): annotating in reverse the new keyframe is anchored at the
        current step, so it covers this frame and every earlier one while the
        old keyframe keeps the frames after it.  Going forward the old keyframe
        is pulled back to ``k-1`` and the new one inherits its anchor.
    ``frame_override``
        Only this frame: the edited pixels, minus the frame's occluders, become
        the instance's visible mask here and nothing else changes.

    Raises ``ValueError`` for an unknown scope or direction, or for a mask that
    is not in the frame's coordinates.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    cache = InputCache()
    hw = frame_hw(db, key)
    edited = as_mask(mask, hw)
    if scope == api.SCOPE_FRAME_OVERRIDE:
        return _commit_frame_override(db, truth, key, instance, edited, hw, annotator)
    if scope not in (api.SCOPE_KEYFRAME, api.SCOPE_SPLIT):
        raise ValueError(f"unknown commit scope {scope!r}")
    parts = [ShapePart(MAIN, masks.encode_rle(edited))]
    return _commit_shape(db, truth, key, instance, parts, GEOM_MASK, scope, direction,
                         cache, annotator)


def commit_box(db: Db, truth: TruthService, key: FrameKey, instance: str,
               box: Sequence[float], scope: str = api.SCOPE_KEYFRAME,
               direction: str = REVERSE, annotator: str = "system") -> dict:
    """Draw the staging-area rectangle of a part on the bench (spec 4.2 S4).

    Bench geometry is a box rather than a mask: the part is tracked only well
    enough to say that it left the chassis and where it went, so the keyframe
    carries one ``box`` part and takes no place in the layer order -- a bench box
    neither occludes anything nor is occluded (spec 3.3 step 5).
    """
    parts = [ShapePart(MAIN, None, tuple(float(v) for v in box))]
    return _commit_shape(db, truth, key, instance, parts, GEOM_BOX, scope, direction,
                         InputCache(), annotator)


def _commit_shape(db: Db, truth: TruthService, key: FrameKey, instance: str,
                  parts: list[ShapePart], geom_type: str, scope: str, direction: str,
                  cache: InputCache, annotator: str) -> dict:
    """The shared body of :func:`commit_edit` and :func:`commit_box`."""
    seg = pose_segment_of(db, key, cache)
    placement = placement_of(db, truth.tax, key, instance, cache)
    chosen = select_keyframe(chain_for(db, key, instance, seg, placement), key.step)
    ref: dict = {"keyframe_id": None}
    old_ref: dict = {"keyframe_id": None if chosen is None else chosen.id}
    before: list[dict] = []
    after: list[dict] = []

    if scope == api.SCOPE_SPLIT and _cannot_split(chosen, key.step, direction):
        # there is nothing left to cut off: the keyframe in force already ends
        # exactly here, so "split" means "re-trace this one" (see _cannot_split)
        scope = api.SCOPE_KEYFRAME

    with db.transaction():
        if scope == api.SCOPE_SPLIT:
            anchor = key.step
            if direction == FORWARD and chosen is not None:
                before.append(keyframe_state(chosen, old_ref))
                anchor = chosen.anchor_step
                chosen.anchor_step = key.step - 1
                db.update_keyframe(chosen)
                after.append(keyframe_state(chosen, old_ref))
            kf = new_keyframe(key, instance, seg, anchor, placement, parts, geom_type)
            # a new version of the shape has to outrank the one it was cut from:
            # select_keyframe breaks an anchor tie on the higher version
            kf.version = 1 if chosen is None else int(chosen.version) + 1
            before.append(keyframe_state(None, ref, template=kf))
            ref["keyframe_id"] = db.add_keyframe(kf)
            after.append(keyframe_state(kf, ref))
        elif chosen is not None and chosen.geom_type == geom_type:
            ref["keyframe_id"] = chosen.id
            before.append(keyframe_state(chosen, ref))
            chosen.parts = parts
            db.update_keyframe(chosen)
            after.append(keyframe_state(chosen, ref))
            kf = chosen
        else:
            anchor = default_anchor(db, truth.tax, key, instance, seg, placement, cache)
            kf = new_keyframe(key, instance, seg, anchor, placement, parts, geom_type)
            before.append(keyframe_state(None, ref, template=kf))
            ref["keyframe_id"] = db.add_keyframe(kf)
            after.append(keyframe_state(kf, ref))

        zorder = None if geom_type == GEOM_BOX else _append_to_zorder(db, key, seg, instance)
        steps = annotatable_steps(
            db, key.desktop, key.view,
            set(truth.affected_steps(key.desktop, key.view, instance, kf)) | {key.step},
        )
        common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                  "instance": instance, "scope": scope, "direction": direction,
                  "steps": steps}
        payload = common | {"keyframes": after,
                            "zorder": None if zorder is None else zorder["after"]}
        inverse = common | {"keyframes": list(reversed(before)),
                            "zorder": None if zorder is None else zorder["before"]}
        db.log_op(key.desktop, key.view, "commit_keyframe",
                  _loggable(payload), _loggable(inverse), annotator)

    op = Op(kind="commit_keyframe", payload=payload, inverse=inverse)
    return _result(db, truth, key, steps, op,
                   {"keyframe_id": kf.id, "anchor_step": kf.anchor_step, "scope": scope,
                    "changed": True})


def preview(db: Db, truth: TruthService, key: FrameKey, instance: str, scope: str,
            direction: str = REVERSE, geom_type: str = GEOM_MASK) -> dict:
    """Which frames an edit *would* reach, without writing or compiling anything.

    This is the "影响 N 帧 / 将产生 N 个冲突" strip of spec 4.3: the annotator has
    to see how far a change carries -- and how many frozen frames it will
    disturb -- **before** deciding on the scope.  The chain is simulated rather
    than written, and no mask is touched, so the answer costs a few state
    lookups.

    Returns ``{"steps", "verified_steps"}``; the second is the subset already
    confirmed by a human, i.e. the frames that would go to the conflict queue.
    """
    cache = InputCache()
    seg = pose_segment_of(db, key, cache)
    placement = placement_of(db, truth.tax, key, instance, cache)
    chain = chain_for(db, key, instance, seg, placement)
    chosen = select_keyframe(chain, key.step)

    if scope == api.SCOPE_FRAME_OVERRIDE:
        steps = annotatable_steps(db, key.desktop, key.view, [key.step])
    else:
        if scope == api.SCOPE_SPLIT and not _cannot_split(chosen, key.step, direction):
            target = new_keyframe(key, instance, seg, key.step, placement, [], geom_type)
            target.version = 1 if chosen is None else int(chosen.version) + 1
            simulated = chain + [target]
        elif chosen is not None and chosen.geom_type == geom_type:
            target, simulated = chosen, chain
        else:
            anchor = default_anchor(db, truth.tax, key, instance, seg, placement, cache)
            target = new_keyframe(key, instance, seg, anchor, placement, [], geom_type)
            simulated = chain + [target]
        steps = chain_steps(db, truth.tax, key, instance, seg, placement, simulated,
                            target, cache)
    verified = [
        step for step in steps
        if (db.get_frame(FrameKey(key.desktop, step, key.view)) or {}).get("review_status")
        == VERIFIED
    ]
    return {"steps": steps, "verified_steps": verified}


def _cannot_split(chosen: Optional[ShapeKeyframe], step: int, direction: str) -> bool:
    """Would a split produce a keyframe covering no frame the old one kept?

    Annotating in reverse a split anchors the new shape at the current step, so
    it covers ``(previous anchor, step]`` and the old one keeps everything after
    ``step``.  When the old keyframe's own anchor *is* ``step`` there is nothing
    after it to keep: the two would share an anchor and the newer one would win
    everywhere, which is a re-trace wearing a second row.  Going forward the same
    thing happens at the other end -- the old keyframe would be pulled back to
    ``step - 1`` and inherit nothing.
    """
    if chosen is None:
        return False
    return direction == REVERSE and int(chosen.anchor_step) == int(step)


def _loggable(payload: dict) -> dict:
    """The op-log copy of a payload: the shared ``ref`` holders flattened away."""
    return payload | {
        "keyframes": [
            {k: v for k, v in state.items() if k != "ref"}
            | {"keyframe_id": state["ref"].get("keyframe_id")}
            for state in payload.get("keyframes", ())
        ]
    }


def _append_to_zorder(db: Db, key: FrameKey, seg: int, instance: str,
                      part: str = MAIN) -> Optional[dict]:
    """Put a newly drawn instance on top of the layer order (spec 4.2).

    Returns the before/after states for the undo payload, or ``None`` when the
    instance was in the order already and nothing had to change.
    """
    rec = db.zorder(key.desktop, key.view, seg)
    order = [tuple(entry) for entry in rec.order]
    if any(entry[0] == instance for entry in order):
        return None
    after = order + [(instance, part)]
    db.set_zorder(ZOrderRec(key.desktop, key.view, seg, after, version=rec.version + 1))
    return {
        "before": {"pose_segment": seg, "order": [list(e) for e in order],
                   "version": rec.version},
        "after": {"pose_segment": seg, "order": [list(e) for e in after],
                  "version": rec.version + 1},
    }


def _commit_frame_override(db: Db, truth: TruthService, key: FrameKey, instance: str,
                           edited: np.ndarray, hw: tuple[int, int], annotator: str) -> dict:
    """Spec 4.3 单帧覆盖: the edit applies to this frame and to no other.

    The stored mask is the *visible* one, so the frame's occluders come off it
    here -- the compiler subtracts them before an override is applied, never
    after (spec 3.3 steps 6-7), and a pinned mask has to mean the same thing as
    the value it replaces.
    """
    visible = edited & ~occluder_union(db, key, hw)
    previous = db.frame_overrides(key).get(instance)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": [key.step]}
    payload = common | {"exists": True, "visible_rle": masks.encode_rle(visible),
                        "visibility": None if previous is None else previous.visibility}
    inverse = common | {"exists": previous is not None,
                        "visible_rle": None if previous is None else previous.visible_rle,
                        "visibility": None if previous is None else previous.visibility}
    with db.transaction():
        db.set_frame_override(
            FrameOverride(key, instance, payload["visible_rle"], payload["visibility"])
        )
        db.log_op(key.desktop, key.view, "set_frame_override", payload, inverse, annotator)
    return _result(db, truth, key, [key.step], Op("set_frame_override", payload, inverse))


# --------------------------------------------------------------------------- #
# layering, visibility, occluders
# --------------------------------------------------------------------------- #
def suggest_scope(compiled: CompiledFrame, instance: str, before: np.ndarray,
                  edited: np.ndarray) -> str:
    """The scope an edit defaults to (spec 4.3, 默认触发).

    The question is about the pixels that actually *changed*, never about the
    whole shape -- loading an instance into the editing layer and touching
    nothing is not a statement about anything:

    * pixels **added** where another instance ``B`` currently paints *over* this
      one say "I want to see this one there instead" -- i.e. put it above ``B``;
    * pixels **erased** exactly where ``B``'s shape lies under this one say the
      opposite: ``B`` should have been on top all along;
    * anything else is a change to the silhouette, so it edits the keyframe.

    Both answers name the pair and the direction, ``zorder:above:<B>`` and
    ``zorder:below:<B>``, because "change the layering" alone does not say which
    way round.  A hint is only given when at least
    :data:`ZORDER_HINT_FRAC` of the changed pixels fall inside ``B``.
    """
    before = np.asarray(before, dtype=bool)
    edited = np.asarray(edited, dtype=bool)
    added, erased = edited & ~before, before & ~edited

    covering = _partner(compiled, instance, added, above=True)
    if covering is not None:
        return f"zorder:above:{covering}"
    covered = _partner(compiled, instance, erased, above=False)
    if covered is not None:
        return f"zorder:below:{covered}"
    return api.SCOPE_KEYFRAME


def _partner(compiled: CompiledFrame, instance: str, changed: np.ndarray,
             above: bool) -> Optional[str]:
    """The instance the changed pixels are a layering statement about, if any.

    ``above=True`` looks for an instance painting *over* ``instance`` whose
    visible pixels the edit reached into; ``above=False`` for one painting
    *under* it, compared on the amodal shapes, since what lies under is by
    definition not visible.
    """
    total = int(changed.sum())
    if total == 0:
        return None
    best, best_overlap = None, 0
    for other, inst in compiled.instances.items():
        if other == instance or _paints_above(compiled, other, instance) is not above:
            continue
        region = inst.visible if above else inst.amodal
        if region is None:
            continue
        overlap = int(np.count_nonzero(changed & region))
        if overlap > best_overlap:
            best, best_overlap = other, overlap
    return best if best_overlap >= ZORDER_HINT_FRAC * total else None


def _paints_above(compiled: CompiledFrame, other: str, instance: str) -> Optional[bool]:
    """Is ``other`` painted over ``instance``? ``None`` when they never meet."""
    for order in compiled.painted.values():
        if other in order and instance in order:
            return order.index(other) > order.index(instance)
    return None


def require_instance(known: Optional[set[str]], instance: str) -> None:
    """Refuse a layering gesture that names something this frame does not have.

    Silently treating an unknown neighbour as "the top of the stack" is how a
    typo, or a stale panel row, turns into a z-order nobody asked for.
    """
    if known is not None and instance not in known:
        raise ValueError(f"no such instance in this frame: {instance!r}")


def set_zorder_move(db: Db, truth: TruthService, key: FrameKey, instance: str,
                    above_of: str, annotator: str = "system",
                    known: Optional[set[str]] = None) -> dict:
    """Move ``instance`` directly above ``above_of`` in the stored layer order.

    The order is a total order over ``(instance, part)`` pairs, bottom first, so
    a multi-part instance moves as one block and keeps its internal order.  A
    neighbour the order does not mention yet is first pinned at the top and the
    instance placed above it, which fixes their relative order and leaves every
    other pair where it was; a neighbour that is not in ``known`` -- the
    instances this frame actually has -- is refused.
    """
    require_instance(known, above_of)
    cache = InputCache()
    seg = pose_segment_of(db, key, cache)
    rec = db.zorder(key.desktop, key.view, seg)
    order = [tuple(entry) for entry in rec.order]
    moved = [entry for entry in order if entry[0] == instance] or [(instance, MAIN)]
    rest = [entry for entry in order if entry[0] != instance]
    if not any(entry[0] == above_of for entry in rest):
        rest = rest + [(above_of, MAIN)]  # unordered so far: pin it, then go above it
    cut = max(i for i, entry in enumerate(rest) if entry[0] == above_of)
    after = rest[: cut + 1] + moved + rest[cut + 1:]
    steps = segment_steps(db, truth.tax, key, instance, seg, cache)

    common = {"desktop": key.desktop, "view": key.view, "pose_segment": seg, "steps": steps}
    payload = common | {"order": [list(e) for e in after], "version": rec.version + 1}
    inverse = common | {"order": [list(e) for e in order], "version": rec.version}
    with db.transaction():
        write_zorder(db, key.desktop, key.view, payload)
        db.log_op(key.desktop, key.view, "set_zorder", payload, inverse, annotator)
    return _result(db, truth, key, steps, Op("set_zorder", payload, inverse))


def commit_pair_override(db: Db, truth: TruthService, key: FrameKey, above: str,
                         below: str, annotator: str = "system",
                         known: Optional[set[str]] = None) -> dict:
    """Record "``above`` beats ``below``" for this pose segment (spec 4.3 改层级).

    A pairwise exception rather than a new global order: it says one thing about
    one pair and leaves every other relation where the annotator put it.
    """
    require_instance(known, above)
    require_instance(known, below)
    cache = InputCache()
    seg = pose_segment_of(db, key, cache)
    po = PairOverride(key.desktop, key.view, seg, above, below)
    existed = any(p == po for p in db.pair_overrides(key.desktop, key.view, seg))
    steps = segment_steps(db, truth.tax, key, above, seg, cache)

    common = {"desktop": key.desktop, "view": key.view, "pose_segment": seg,
              "above": above, "below": below, "steps": steps}
    payload = common | {"exists": True}
    inverse = common | {"exists": existed}
    with db.transaction():
        db.set_pair_override(po)
        db.log_op(key.desktop, key.view, "set_pair_override", payload, inverse, annotator)
    return _result(db, truth, key, steps, Op("set_pair_override", payload, inverse))


def set_visibility(db: Db, truth: TruthService, key: FrameKey, instance: str, vis: str,
                   annotator: str = "system") -> dict:
    """Override one instance's ``visibility`` label on this frame (spec 6.2).

    A label is a frame-level statement, so it is stored as a
    :class:`~tda.core.model.FrameOverride` -- merged into the one already there,
    which keeps a hand-drawn single-frame mask intact.
    """
    if vis not in api.VISIBILITY_VALUES:
        raise ValueError(f"visibility must be one of {api.VISIBILITY_VALUES}, got {vis!r}")
    previous = db.frame_overrides(key).get(instance)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": [key.step]}
    payload = common | {"exists": True, "visibility": vis,
                        "visible_rle": None if previous is None else previous.visible_rle}
    inverse = common | {"exists": previous is not None,
                        "visibility": None if previous is None else previous.visibility,
                        "visible_rle": None if previous is None else previous.visible_rle}
    with db.transaction():
        db.set_frame_override(FrameOverride(key, instance, payload["visible_rle"], vis))
        db.log_op(key.desktop, key.view, "set_frame_override", payload, inverse, annotator)
    return _result(db, truth, key, [key.step], Op("set_frame_override", payload, inverse))


def commit_occluder(db: Db, truth: TruthService, key: FrameKey, mask: np.ndarray,
                    occluder_type: str = "hand", annotator: str = "system") -> dict:
    """Store one occluder layer of this frame (spec 3.1, 4.2 step 4).

    One layer per type, because the compiler subtracts each type separately and
    the database keys ``occluder_mask`` on it.
    """
    layer = as_mask(mask, frame_hw(db, key))
    previous = next((o for o in db.occluders(key) if o.occluder_type == occluder_type), None)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "occluder_type": occluder_type, "steps": [key.step]}
    payload = common | {"exists": True, "rle": masks.encode_rle(layer)}
    inverse = common | {"exists": previous is not None,
                        "rle": None if previous is None else previous.rle}
    with db.transaction():
        db.set_occluder(OccluderMask(key, occluder_type, payload["rle"]))
        db.log_op(key.desktop, key.view, "set_occluder", payload, inverse, annotator)
    return _result(db, truth, key, [key.step], Op("set_occluder", payload, inverse))
