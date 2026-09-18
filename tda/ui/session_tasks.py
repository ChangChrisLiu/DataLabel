"""The task card: what reverse-order annotation asks for at one step (spec 4.2).

Annotating backwards means reading the step table backwards: the change an
action *caused* at step ``k`` is the thing that has to be *undone* on the way to
``k - 1``.  :func:`tda.core.states.diff_states` between the two snapshots
therefore yields the instructions almost directly, and this module is the
mapping from a state change to the gesture it implies -- draw the part back in,
split its keyframe, change only the label, retire its staging-area box, or just
confirm the frame.

Nothing here writes: the card is derived from the step table and from whichever
shapes happen to exist, which is also what makes an item ``done``.
"""
from __future__ import annotations

from typing import Optional

from tda.core.db import Db
from tda.core.compiler import select_keyframe
from tda.core.model import FrameKey, InstanceRec
from tda.core.states import diff_states, needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import InputCache, instances_of, pose_segment_of, state_of
from tda.ui import session_api as api
from tda.ui.session_ops import (
    BENCH_KINDS,
    GEOM_BOX,
    GEOM_MASK,
    IN_CHASSIS,
    ON_BENCH,
    chain_for,
    placement_of,
)

__all__ = ["SPLIT_TRANSITIONS", "STATE_ONLY_TRANSITIONS", "task_card_for"]

REMOVED = "removed"
_VERIFIED = "verified"

#: State transitions that keep the shape but need a new version of it at k-1.
SPLIT_TRANSITIONS = frozenset(
    {("open", "closed"), ("unplugged", "plugged"), ("displaced", "installed")}
)
#: ``loosened -> fastened``: the shape carries over untouched.
STATE_ONLY_TRANSITIONS = frozenset({("loosened", "fastened")})

#: Order the kinds appear in for one instance: draw first, confirm last.
_KIND_RANK = {
    api.KIND_ADD_SHAPE: 0,
    api.KIND_SPLIT_KEYFRAME: 1,
    api.KIND_STATE_ONLY: 2,
    api.KIND_REMOVE_BENCH_BOX: 3,
    api.KIND_CONFIRM: 4,
}


# --------------------------------------------------------------------------- #
# what already exists
# --------------------------------------------------------------------------- #
def _has_shape(db: Db, tax: Taxonomy, desktop: int, view: str, instance: str, step: int,
               cache: Optional[InputCache] = None,
               geom_type: Optional[str] = None) -> bool:
    """Does a keyframe of the right chain apply to ``instance`` at ``step``?"""
    key = FrameKey(desktop, step, view)
    seg = pose_segment_of(db, key, cache)
    placement = placement_of(db, tax, key, instance, cache)
    chosen = select_keyframe(chain_for(db, key, instance, seg, placement), step)
    if chosen is None:
        return False
    return geom_type is None or chosen.geom_type == geom_type


def _has_bench_chain(db: Db, desktop: int, view: str, instance: str) -> bool:
    """Has anybody drawn a staging-area box for this instance in this view?"""
    return any(kf.placement == ON_BENCH for kf in db.keyframes(desktop, view, instance))


# --------------------------------------------------------------------------- #
# one change -> one instruction
# --------------------------------------------------------------------------- #
def _kind_for(changes: dict, needed_now: bool, needed_before: bool) -> str:
    """Map one instance's ``k -> k-1`` change onto a kind (spec 4.2 step 2).

    The three named transition sets come straight from the spec.  Anything else
    falls back on where geometry is required: a shape that exists at ``k-1`` but
    not at ``k`` has to be drawn, one that exists on both sides needs a second
    version, and one that exists on neither is a label change.
    """
    transition = changes.get("state")
    if transition is not None:
        old, new = transition
        if old == REMOVED and new != REMOVED:
            return api.KIND_ADD_SHAPE
        if transition in SPLIT_TRANSITIONS:
            return api.KIND_SPLIT_KEYFRAME
        if transition in STATE_ONLY_TRANSITIONS:
            return api.KIND_STATE_ONLY
    if needed_before and not needed_now:
        return api.KIND_ADD_SHAPE
    if needed_before and needed_now:
        return api.KIND_SPLIT_KEYFRAME
    return api.KIND_STATE_ONLY


def _text_for(kind: str, instance: str, changes: dict, rec: Optional[InstanceRec]) -> str:
    """The one-line instruction the task card shows."""
    cls = "" if rec is None else f" ({rec.cls})"
    transition = changes.get("state")
    arrow = "" if transition is None else f": {transition[0]} -> {transition[1]}"
    if kind == api.KIND_ADD_SHAPE:
        with_parent = (f", back in with {rec.parent}"
                       if rec is not None and rec.attached and rec.parent else "")
        return f"Draw {instance}{cls} back inside the chassis{with_parent}"
    if kind == api.KIND_SPLIT_KEYFRAME:
        return f"Split the keyframe of {instance}{cls}{arrow}"
    if kind == api.KIND_REMOVE_BENCH_BOX:
        return f"The staging-area box of {instance}{cls} ends here"
    if kind == api.KIND_CONFIRM:
        return f"Nothing to draw here{arrow} - confirm the frame"
    return f"State only for {instance}{cls}{arrow}"


def _item(instance: str, kind: str, changes: dict, done: bool,
          rec: Optional[InstanceRec]) -> dict:
    return {"instance": instance, "kind": kind,
            "text": _text_for(kind, instance, changes, rec), "done": bool(done)}


def _ordered(items: list[dict], instances: dict[str, InstanceRec]) -> list[dict]:
    """Parent before the children that ride out of the chassis with it."""
    def sort_key(item: dict) -> tuple:
        rec = instances.get(item["instance"])
        parent = rec.parent if rec is not None and rec.attached and rec.parent else None
        return (parent or item["instance"], 0 if parent is None else 1,
                item["instance"], _KIND_RANK.get(item["kind"], 9))

    return sorted(items, key=sort_key)


# --------------------------------------------------------------------------- #
# the card
# --------------------------------------------------------------------------- #
def task_card_for(db: Db, tax: Taxonomy, desktop: int, view: str, step: int,
                  start_step: Optional[int] = None) -> list[dict]:
    """What has to be annotated to go from step ``k`` back to ``k-1`` (spec 4.2).

    A part that is ``removed`` at ``k`` and installed at ``k-1`` has to be drawn
    back into the chassis -- together with the attached children that come back
    in with it, which is why the card groups a parent with them.  A latch that
    is ``open`` at ``k`` needs a second version of its shape; a screw that is
    merely ``loosened`` needs no new pixels at all.

    Two frames are special.  The **start frame** has nothing annotated yet, so
    its card lists every instance that needs geometry there (spec 4.2 step 1).
    A ``dupli`` or ``failed`` step -- and the first step of the teardown, which
    has no predecessor -- carries a single ``confirm`` item.

    ``done`` says whether the geometry the item asks for is already in the
    database, so the panel can grey an item out without the annotator having to
    remember what they drew.
    """
    cache = InputCache()
    records = {rec.step: rec for rec in db.steps(desktop)}
    if step not in records:
        return []
    rec = records[step]
    instances = instances_of(db, desktop, cache)
    start = max(records) if start_step is None else start_step
    frame = db.get_frame(FrameKey(desktop, step, view)) or {}
    confirmed = frame.get("review_status") == _VERIFIED

    if rec.dupli or rec.step_type in ("dupli", "failed"):
        return [_item(rec.raw_name or f"step {step}", api.KIND_CONFIRM, {}, confirmed, None)]
    if step >= start:
        return _start_card(db, tax, desktop, view, step, instances, cache)
    if step - 1 not in records:
        return [_item(rec.raw_name or f"step {step}", api.KIND_CONFIRM, {}, confirmed, None)]

    state_now = state_of(db, tax, desktop, step, cache)
    state_before = state_of(db, tax, desktop, step - 1, cache)
    needs_now = needs_geom(instances, state_now, tax)
    needs_before = needs_geom(instances, state_before, tax)

    changed: dict[str, dict] = {}
    for instance, attr, old, new in diff_states(state_now, state_before):
        changed.setdefault(instance, {})[attr] = (old, new)

    items: list[dict] = []
    for instance, changes in changed.items():
        if instance not in instances:
            continue  # a virtual cable node: it never carries geometry
        rec_i = instances.get(instance)
        kind = _kind_for(changes, instance in needs_now, instance in needs_before)
        done = (True if kind == api.KIND_STATE_ONLY
                else _has_shape(db, tax, desktop, view, instance, step - 1, cache))
        items.append(_item(instance, kind, changes, done, rec_i))
        if (changes.get("placement") == (ON_BENCH, IN_CHASSIS)
                and _has_bench_chain(db, desktop, view, instance)):
            # spec 4.2: the part is back in the chassis, so its on_bench chain
            # ends at k -- done exactly when no bench box reaches k-1 any more
            retired = not _has_shape(db, tax, desktop, view, instance, step - 1, cache,
                                     GEOM_BOX)
            items.append(_item(instance, api.KIND_REMOVE_BENCH_BOX, changes, retired, rec_i))
    return _ordered(items, instances)


def _start_card(db: Db, tax: Taxonomy, desktop: int, view: str, step: int,
                instances: dict[str, InstanceRec], cache: InputCache) -> list[dict]:
    """Spec 4.2 step 1: on the start frame everything present has to be drawn."""
    state = state_of(db, tax, desktop, step, cache)
    items = []
    for instance, kind in sorted(needs_geom(instances, state, tax).items()):
        geom = GEOM_BOX if kind in BENCH_KINDS else GEOM_MASK
        item = _item(instance, api.KIND_ADD_SHAPE, {},
                     _has_shape(db, tax, desktop, view, instance, step, cache, geom),
                     instances.get(instance))
        item["text"] = (f"Draw the staging-area box of {instance}" if geom == GEOM_BOX
                        else f"Draw {instance} in the chassis")
        items.append(item)
    return _ordered(items, instances)
