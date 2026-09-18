"""The task card: what has to be annotated on the frame in front of the annotator.

Reverse-order annotation (spec 4.2) arrives at frame ``j`` from ``j + 1``, which
is already done.  The work that belongs to ``j`` is the difference between the
two, read on the image the annotator is looking at: a part that is ``removed``
at ``j + 1`` and installed at ``j`` is a part they can *see* there and have to
draw, a latch that is ``open`` at ``j + 1`` and shut at ``j`` needs a second
version of its shape, and a screw that was merely loosened needs no new pixels
at all.

Everything is therefore expressed in two frames: **``j``, the frame on screen**,
and **``neighbour``, the frame it is diffed against** -- ``j + 1`` browsing
backwards, ``j - 1`` browsing forwards for a repair, and the next one that
actually has an image when the one next door is missing.  ``done`` is always
evaluated at ``j``: the question is whether the geometry the item asks for
exists *here*.

The **start frame** has no neighbour.  Nothing has been annotated yet, so its
card is one item per instance that needs geometry there and has none, ordered
so the chassis is drawn before the things that sit on top of it, and a single
``confirm`` once they are all drawn.

Bench work is asked for only where there is a bench to see.  Spec 4.2 says
若该视角有堆放区 ROI -- *if this view has a staging-area ROI* -- and the scanner
looks straight down at the board, so on it the parts that have been taken out
are simply not in the picture.  ``add_bench_box`` and ``remove_bench_box`` are
therefore emitted only when the frame's pose segment has one recorded, and they
carry their own kind so the window arms the box tool rather than the brush.
"""
from __future__ import annotations

from typing import Optional

from tda.core.compiler import select_keyframe
from tda.core.db import Db
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

__all__ = ["LAYER_RANK", "SPLIT_TRANSITIONS", "STATE_ONLY_TRANSITIONS", "has_bench_roi",
           "item_text", "task_card_for"]

REMOVED = "removed"
_VERIFIED = "verified"

#: State transitions that keep the shape but need a new version of it here.
SPLIT_TRANSITIONS = frozenset(
    {("open", "closed"), ("unplugged", "plugged"), ("displaced", "installed")}
)
#: ``loosened -> fastened``: the shape carries over untouched.
STATE_ONLY_TRANSITIONS = frozenset({("loosened", "fastened")})

#: Taxonomy group -> how low it sits in a frame, so the start frame's card can
#: be drawn from the bottom up: the chassis is behind everything, the parts
#: bolted into it come next, and the small things that fasten or connect them
#: are on top. Only the order matters, not the numbers.
LAYER_RANK: dict[str, int] = {
    "structure": 0,
    "part": 1,
    "interface": 2,
    "latch": 3,
    "fastener": 4,
}
_CHASSIS_CLASS = "chassis"

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


def has_bench_roi(db: Db, key: FrameKey, cache: Optional[InputCache] = None) -> bool:
    """Can this view see a staging area at this frame? (spec 4.2 item 1)

    Without one there is nothing to box: a part that has been taken out is not
    in the picture at all, and asking for thirty rectangles nobody can draw is
    how a task card stops being read.
    """
    seg = pose_segment_of(db, key, cache)
    return db.bench_roi(key.desktop, key.view, seg) is not None


# --------------------------------------------------------------------------- #
# one change -> one instruction
# --------------------------------------------------------------------------- #
def _kind_for(changes: dict, needs_neighbour: bool, needs_here: bool) -> str:
    """Map one instance's difference from the neighbour onto a kind (spec 4.2).

    ``changes`` reads ``neighbour value -> value on this frame``.  The three
    named transition sets come straight from the spec.  Anything else falls back
    on where geometry is required: a shape needed *here* but not next door has
    to be drawn, one needed on both sides needs a second version, and one needed
    on neither is a label change.
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
    if changes.get("placement") is not None and needs_here:
        # the chassis and bench chains of an instance are independent (spec 3.3),
        # so a part that changed side carries no shape on this one yet, whichever
        # way the annotator is walking
        return api.KIND_ADD_SHAPE
    if needs_here and not needs_neighbour:
        return api.KIND_ADD_SHAPE
    if needs_here and needs_neighbour:
        return api.KIND_SPLIT_KEYFRAME
    return api.KIND_STATE_ONLY


def item_text(kind: str, instance: str, changes: dict, rec: Optional[InstanceRec],
              span: Optional[list[int]] = None) -> str:
    """The one-line instruction the task card shows.

    The one formatter: :meth:`AnnotationSession.confirm_frame` says a missing
    shape in exactly these words too, so a panel can match a problem to the item
    it belongs to instead of guessing.
    """
    cls = "" if rec is None else f" ({rec.cls})"
    transition = changes.get("state")
    arrow = "" if transition is None else f": {transition[0]} -> {transition[1]}"
    note = _span_note(span)
    if kind == api.KIND_ADD_SHAPE:
        with_parent = (f", back in with {rec.parent}"
                       if rec is not None and rec.attached and rec.parent else "")
        return f"Draw {instance}{cls} on this frame{with_parent}{note}"
    if kind == api.KIND_ADD_BENCH_BOX:
        return f"Draw the staging-area box of {instance}{cls}{note}"
    if kind == api.KIND_SPLIT_KEYFRAME:
        return f"Split the keyframe of {instance}{cls}{arrow}{note}"
    if kind == api.KIND_REMOVE_BENCH_BOX:
        return f"The staging-area box of {instance}{cls} ends here{note}"
    if kind == api.KIND_CONFIRM:
        nothing = changes.get("nothing")
        if nothing is not None:
            return (f"Nothing to draw here: no change against step {nothing[0]}"
                    f"{note} - confirm the frame")
        return f"Nothing to draw here{arrow}{note} - confirm the frame"
    return f"State only for {instance}{cls}{arrow}{note}"


def _span_note(span: Optional[list[int]]) -> str:
    """Say so when the card covers more than one action step (a missing frame)."""
    if not span or len(span) < 2:
        return ""
    skipped = ", ".join(f"step {s} has no image" for s in span[:-1])
    return f" (covers steps {span[0]}-{span[-1]}; {skipped})"


def _item(instance: str, kind: str, changes: dict, done: bool,
          rec: Optional[InstanceRec], span: Optional[list[int]] = None) -> dict:
    return {"instance": instance, "kind": kind,
            "text": item_text(kind, instance, changes, rec, span), "done": bool(done)}


def _ordered(items: list[dict], instances: dict[str, InstanceRec]) -> list[dict]:
    """Parent before the children that ride out of the chassis with it."""
    def sort_key(item: dict) -> tuple:
        rec = instances.get(item["instance"])
        parent = rec.parent if rec is not None and rec.attached and rec.parent else None
        return (parent or item["instance"], 0 if parent is None else 1,
                item["instance"], _KIND_RANK.get(item["kind"], 9))

    return sorted(items, key=sort_key)


def _bottom_up(items: list[dict], instances: dict[str, InstanceRec],
               tax: Taxonomy) -> list[dict]:
    """Order a start-frame card so the thing everything sits on is drawn first."""
    def sort_key(item: dict) -> tuple:
        rec = instances.get(item["instance"])
        cls = "" if rec is None else rec.cls
        return (0 if cls == _CHASSIS_CLASS else 1,
                LAYER_RANK.get(tax.group_of(cls), len(LAYER_RANK)),
                item["instance"])

    return sorted(items, key=sort_key)


# --------------------------------------------------------------------------- #
# the card
# --------------------------------------------------------------------------- #
def task_card_for(db: Db, tax: Taxonomy, desktop: int, view: str, step: int,
                  neighbour: Optional[int] = None,
                  span: Optional[list[int]] = None) -> list[dict]:
    """What has to be annotated on frame ``step`` (spec 4.2).

    ``neighbour`` is the already-annotated frame the card is diffed against --
    ``step + 1`` in the reverse order annotation runs in, ``step - 1`` when the
    annotator is browsing forwards to repair something, and ``None`` on the
    start frame, which has nobody after it.

    A part that is ``removed`` next door and installed here has to be drawn back
    into the chassis, together with the attached children that came back in with
    it, which is why the card groups a parent with them. ``done`` says whether
    the geometry the item asks for already exists **on this frame**, so the
    panel can grey an item out without the annotator having to remember.
    """
    cache = InputCache()
    records = {rec.step: rec for rec in db.steps(desktop)}
    if step not in records:
        return []
    rec = records[step]
    instances = instances_of(db, desktop, cache)
    frame = db.get_frame(FrameKey(desktop, step, view)) or {}
    confirmed = frame.get("review_status") == _VERIFIED

    if rec.dupli or rec.step_type in ("dupli", "failed"):
        return [_item(rec.raw_name or f"step {step}", api.KIND_CONFIRM, {}, confirmed,
                      None, None)]
    if neighbour is None or neighbour not in records:
        return _start_card(db, tax, desktop, view, step, instances, cache, confirmed)

    span = span or [neighbour]
    bench = has_bench_roi(db, FrameKey(desktop, step, view), cache)
    state_here = state_of(db, tax, desktop, step, cache)
    state_there = state_of(db, tax, desktop, neighbour, cache)
    needs_here = needs_geom(instances, state_here, tax)
    needs_there = needs_geom(instances, state_there, tax)

    changed: dict[str, dict] = {}
    for instance, attr, there, here in diff_states(state_there, state_here):
        changed.setdefault(instance, {})[attr] = (there, here)

    items: list[dict] = []
    for instance, changes in changed.items():
        if instance not in instances:
            continue  # a virtual cable node: it never carries geometry
        rec_i = instances.get(instance)
        kind = _kind_for(changes, instance in needs_there, instance in needs_here)
        wants_box = needs_here.get(instance) in BENCH_KINDS
        if wants_box:
            if not bench:
                continue  # this view cannot see the staging area: not its work
            if kind == api.KIND_ADD_SHAPE:
                kind = api.KIND_ADD_BENCH_BOX
        done = (True if kind == api.KIND_STATE_ONLY
                else _has_shape(db, tax, desktop, view, instance, step, cache,
                                GEOM_BOX if wants_box else None))
        items.append(_item(instance, kind, changes, done, rec_i, span))
        if (bench and changes.get("placement") == (ON_BENCH, IN_CHASSIS)
                and _has_bench_chain(db, desktop, view, instance)):
            # the part is back in the chassis here, so its on_bench chain ended
            # next door -- done exactly when no bench box reaches this frame
            retired = not _has_shape(db, tax, desktop, view, instance, step, cache,
                                     GEOM_BOX)
            items.append(_item(instance, api.KIND_REMOVE_BENCH_BOX, changes, retired,
                               rec_i, span))
    if not items:
        return [_item(f"step {step}", api.KIND_CONFIRM,
                      {"nothing": (str(neighbour), str(step))}, confirmed, None, span)]
    return _ordered(items, instances)


def _start_card(db: Db, tax: Taxonomy, desktop: int, view: str, step: int,
                instances: dict[str, InstanceRec], cache: InputCache,
                confirmed: bool) -> list[dict]:
    """Spec 4.2 step 1: on the start frame everything present has to be drawn.

    Only what is still missing is listed -- an instance already drawn is not
    work -- so the card empties as the annotator goes and ends as a single
    confirmation.
    """
    state = state_of(db, tax, desktop, step, cache)
    bench = has_bench_roi(db, FrameKey(desktop, step, view), cache)
    items = []
    for instance, geom_kind in sorted(needs_geom(instances, state, tax).items()):
        on_bench = geom_kind in BENCH_KINDS
        if on_bench and not bench:
            continue  # this view cannot see the staging area
        geom = GEOM_BOX if on_bench else GEOM_MASK
        if _has_shape(db, tax, desktop, view, instance, step, cache, geom):
            continue
        kind = api.KIND_ADD_BENCH_BOX if on_bench else api.KIND_ADD_SHAPE
        items.append(_item(instance, kind, {}, False, instances.get(instance), None))
    if not items:
        return [_item(f"step {step}", api.KIND_CONFIRM, {}, confirmed, None, None)]
    return _bottom_up(items, instances, tax)
