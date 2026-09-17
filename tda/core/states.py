"""State machine and geometry policy -- steps 1-2 of the compiler (spec 3.3).

The whole module is pure: it turns the recorded action list into an explicit
event log, folds that log into a per-step snapshot, and answers which instances
need geometry at a step.

* :func:`initial_state`      -- every instance at its class default, in the chassis.
* :func:`events_from_actions`-- actions (spec 6.3) -> :class:`~tda.core.model.StateEvent` log.
* :func:`state_at`           -- initial state + every event with ``step <= k``.
* :func:`needs_geom`         -- which instances need a mask or a bench box at a step.
* :func:`diff_states`        -- what changed between two snapshots.
* :func:`validate_events`    -- the legal-transition checks of spec 6.2.

Conventions
-----------
Every instance starts ``in_chassis`` (the chassis itself included). An action
whose ``result`` is not ``"success"`` produces no events at all (spec 6.3), and
an action whose verb does not apply to the target class -- or whose effect is
already the current value -- produces no event either.

Whenever an instance reaches ``removed``, it also leaves the chassis
(``placement: in_chassis -> on_bench``) and drags its attached children with it.
A ``connector`` is the one exception: removing it means the cable was taken out
as a whole, which is a state change only, because an unplugged/removed
connector carries no geometry of its own (spec 6.2). That cascade is enforced
twice over: :func:`events_from_actions` materialises it as explicit events, and
:func:`state_at` closes over it again, so a hand-entered (``auto=False``) parent
removal never leaves an attached child behind asking for a mask.

Virtual nodes are targets starting with ``cable:`` (or, for actions, anything
absent from ``instances``). They have their own small state machine and are
carried in a :data:`FrameState` once an event mentions them, but they never
carry geometry, because :func:`needs_geom` only answers for real instances.
"""
from __future__ import annotations

from dataclasses import dataclass

from tda.core.model import ActionRec, InstanceRec, Placement, StateEvent
from tda.core.taxonomy import Taxonomy

__all__ = [
    "InstState",
    "FrameState",
    "diff_states",
    "events_from_actions",
    "initial_state",
    "is_virtual_target",
    "needs_geom",
    "state_at",
    "validate_events",
]

ATTR_STATE = "state"
ATTR_PLACEMENT = "placement"
REMOVED = "removed"
CABLE_PREFIX = "cable:"
CABLE_CLASS = "cable"
CONNECTOR_CLASS = "connector"
IN_CHASSIS = Placement.IN_CHASSIS.value
ON_BENCH = Placement.ON_BENCH.value
_PLACEMENTS = frozenset(p.value for p in Placement)
_SUCCESS = "success"


@dataclass
class InstState:
    """One instance's state and placement at one logical step."""

    state: str
    placement: str


#: instance key -> its state at one logical step
FrameState = dict[str, InstState]


def is_virtual_target(target: str, instances: dict[str, InstanceRec]) -> bool:
    """Is this action target a virtual node rather than a real instance?"""
    return target.startswith(CABLE_PREFIX) or target not in instances


def _virtual_class(target: str) -> str | None:
    """The virtual-node class of ``target``, or ``None`` when it has none."""
    return CABLE_CLASS if target.startswith(CABLE_PREFIX) else None


# --------------------------------------------------------------------------- #
# 1. state
# --------------------------------------------------------------------------- #
def initial_state(instances: dict[str, InstanceRec], tax: Taxonomy) -> FrameState:
    """The state before step 1: class defaults, everything still in the chassis."""
    return {
        key: InstState(state=tax.default_state(rec.cls), placement=IN_CHASSIS)
        for key, rec in instances.items()
    }


def _attached_children(instances: dict[str, InstanceRec]) -> dict[str, list[str]]:
    """parent key -> sorted keys of the children that leave the chassis with it."""
    children: dict[str, list[str]] = {}
    for key, rec in instances.items():
        if rec.attached and rec.parent:
            children.setdefault(rec.parent, []).append(key)
    for keys in children.values():
        keys.sort()
    return children


def _cascades_on_removal(rec: InstanceRec) -> bool:
    """Does removing this instance take its attached children out with it?

    Shared by :func:`events_from_actions` and :func:`state_at` so the event log
    and the closure over it can never disagree. A ``connector`` does not
    cascade: removing one means the cable went out as a whole (spec 6.3).
    """
    return rec.cls != CONNECTOR_CLASS


def events_from_actions(
    instances: dict[str, InstanceRec],
    actions: list[ActionRec],
    tax: Taxonomy,
) -> list[StateEvent]:
    """Compile the recorded actions into a chronological state-event log.

    Actions are processed in ``(step, idx)`` order, so ``old`` on every event is
    the value that actually held when the action was performed. All events are
    ``auto=True``: they are derived, not typed in by an annotator.
    """
    frame = initial_state(instances, tax)
    cable_states: dict[str, str] = {}
    children = _attached_children(instances)
    events: list[StateEvent] = []

    def emit(action: ActionRec, target: str, attr: str, new: str) -> None:
        inst = frame[target]
        old = getattr(inst, attr)
        if old == new:
            return
        events.append(
            StateEvent(
                desktop=action.desktop,
                step=action.step,
                target=target,
                attr=attr,
                old=old,
                new=new,
                auto=True,
            )
        )
        setattr(inst, attr, new)

    def detach(action: ActionRec, target: str, seen: set[str]) -> None:
        """Move ``target`` to the bench, then its attached children with it.

        ``seen`` keeps a malformed parent cycle from recursing forever.
        """
        if frame[target].placement == IN_CHASSIS:
            emit(action, target, ATTR_PLACEMENT, ON_BENCH)
        for child in children.get(target, ()):
            if child in seen:
                continue
            seen.add(child)
            emit(action, child, ATTR_STATE, REMOVED)
            detach(action, child, seen)

    for action in sorted(actions, key=lambda a: (a.step, a.idx)):
        if action.result != _SUCCESS:
            continue
        if is_virtual_target(action.target, instances):
            cls = _virtual_class(action.target)
            if cls is None:
                continue
            effect = tax.apply_verb(cls, {}, action.verb)
            if effect is None or effect[0] != ATTR_STATE:
                continue
            old = cable_states.get(action.target, tax.default_state(cls))
            if old == effect[1]:
                continue
            events.append(
                StateEvent(
                    desktop=action.desktop,
                    step=action.step,
                    target=action.target,
                    attr=ATTR_STATE,
                    old=old,
                    new=effect[1],
                    auto=True,
                )
            )
            cable_states[action.target] = effect[1]
            continue

        rec = instances[action.target]
        effect = tax.apply_verb(rec.cls, rec.attrs, action.verb)
        if effect is None:
            continue
        attr, new = effect
        emit(action, action.target, attr, new)
        if attr == ATTR_STATE and new == REMOVED and _cascades_on_removal(rec):
            detach(action, action.target, {action.target})
    return events


def _close_attached_cascade(
    instances: dict[str, InstanceRec],
    frame: FrameState,
) -> None:
    """Force the spec-3.3 cascade on ``frame`` in place, idempotently.

    Every ``attached`` child of an instance already in state ``removed`` is
    removed as well, transitively; a child still ``in_chassis`` moves to the
    bench (one already ``elsewhere`` stays there -- it is out of every view).
    Running this on a frame that already satisfies the cascade -- anything folded
    from an :func:`events_from_actions` log -- changes nothing.
    """
    children = _attached_children(instances)
    if not children:
        return

    def walk(parent: str, seen: set[str]) -> None:
        for child in children.get(parent, ()):
            if child in seen:
                continue
            seen.add(child)
            inst = frame.get(child)
            if inst is None:
                continue
            inst.state = REMOVED
            if inst.placement == IN_CHASSIS:
                inst.placement = ON_BENCH
            walk(child, seen)

    for key in sorted(children):  # only instances that actually have children
        rec = instances.get(key)
        inst = frame.get(key)
        if rec is None or inst is None or inst.state != REMOVED:
            continue
        if _cascades_on_removal(rec):
            walk(key, {key})


def state_at(
    instances: dict[str, InstanceRec],
    events: list[StateEvent],
    step: int,
    tax: Taxonomy,
) -> FrameState:
    """The state of every instance at logical step ``step``.

    Initial state, plus every event with ``event.step <= step``, plus the
    attached-child closure of spec 3.3 step 1. :func:`events_from_actions`
    already emits that cascade as explicit events, so the closure is a no-op on
    a compiled log; it is what makes a hand-entered (``auto=False``) parent
    removal drag its attached children along too.

    A virtual ``cable:*`` node enters the snapshot as soon as an event mentions
    it (placement ``in_chassis``, it is nowhere else); it carries no geometry.
    Events on any other unknown key are skipped.
    """
    frame = initial_state(instances, tax)
    for event in sorted(events, key=lambda e: e.step):  # stable: keeps intra-step order
        if event.step > step:
            break
        inst = frame.get(event.target)
        if inst is None:
            cls = _virtual_class(event.target)
            if cls is None:
                continue
            inst = InstState(state=tax.default_state(cls), placement=IN_CHASSIS)
            frame[event.target] = inst
        if event.attr == ATTR_STATE:
            inst.state = event.new
        elif event.attr == ATTR_PLACEMENT:
            inst.placement = event.new
    _close_attached_cascade(instances, frame)
    return frame


# --------------------------------------------------------------------------- #
# 2. geometry policy
# --------------------------------------------------------------------------- #
def needs_geom(
    instances: dict[str, InstanceRec],
    fs: FrameState,
    tax: Taxonomy,
) -> dict[str, str]:
    """Which instances need their own geometry in this frame state.

    ``"mask"`` for an instance still in the chassis whose state is tracked by
    the per-class table of spec 6.2, ``"box"`` for one lying in the bench area
    (the chassis excepted -- it never goes on the bench). Everything else is
    omitted: an unplugged connector, anything ``elsewhere`` or in some other
    placement, and any key that is not a real instance (a ``cable:*`` node).
    """
    geom: dict[str, str] = {}
    for key, inst in fs.items():
        rec = instances.get(key)
        if rec is None:
            continue
        if inst.placement == IN_CHASSIS:
            kind = "mask"
        elif inst.placement == ON_BENCH:
            kind = "box"
        else:
            continue
        if tax.needs_mask(rec.cls, inst.state, inst.placement):
            geom[key] = kind
    return geom


def diff_states(a: FrameState, b: FrameState) -> list[tuple[str, str, str, str]]:
    """``(key, attr, old, new)`` for every difference, sorted by key.

    Keys missing from either side are ignored: the caller compares two
    snapshots of the same instance table.
    """
    rows: list[tuple[str, str, str, str]] = []
    for key in sorted(a.keys() & b.keys()):
        before, after = a[key], b[key]
        if before.state != after.state:
            rows.append((key, ATTR_STATE, before.state, after.state))
        if before.placement != after.placement:
            rows.append((key, ATTR_PLACEMENT, before.placement, after.placement))
    return rows


# --------------------------------------------------------------------------- #
# validation (spec 6.2 legal transitions)
# --------------------------------------------------------------------------- #
def validate_events(
    instances: dict[str, InstanceRec],
    events: list[StateEvent],
    tax: Taxonomy,
) -> list[str]:
    """Report every illegal event as ``"<key>: <old> → <new> at step <k>: <reason>"``.

    Three things are flagged, in that order per event:

    1. leaving ``removed`` -- a removed part cannot come back (manual events may
       otherwise reverse a transition, e.g. a latch ``open -> closed``);
    2. an ``old`` value that does not match the value actually held at that step;
    3. a ``new`` value that is not a state of the target's class (or not a known
       placement).

    Events are replayed in ``(step, list)`` order and applied even when flagged,
    so later events are checked against the recorded trajectory. Targets that
    are neither instances nor virtual nodes are ignored, as are attributes other
    than ``state`` and ``placement``.
    """
    problems: list[str] = []
    current: dict[str, InstState] = initial_state(instances, tax)

    for event in sorted(events, key=lambda e: e.step):  # stable: keeps intra-step order
        rec = instances.get(event.target)
        if rec is not None:
            cls = rec.cls
        else:
            virtual = _virtual_class(event.target)
            if virtual is None:
                continue
            cls = virtual
        inst = current.get(event.target)
        if inst is None:
            inst = InstState(state=tax.default_state(cls), placement=IN_CHASSIS)
            current[event.target] = inst
        if event.attr not in (ATTR_STATE, ATTR_PLACEMENT):
            continue

        held = getattr(inst, event.attr)
        reasons: list[str] = []
        if event.attr == ATTR_STATE and held == REMOVED and event.new != REMOVED:
            reasons.append(f"{REMOVED!r} is final, an instance cannot leave it")
        if event.old != held:
            reasons.append(f"the current {event.attr} at that step is {held!r}")
        if event.attr == ATTR_STATE:
            if event.new not in tax.states_of(cls):
                reasons.append(f"{event.new!r} is not a state of class {cls!r}")
        elif event.new not in _PLACEMENTS:
            reasons.append(f"{event.new!r} is not a known placement")

        problems.extend(
            f"{event.target}: {event.old} → {event.new} at step {event.step}: {reason}"
            for reason in reasons
        )
        setattr(inst, event.attr, event.new)
    return problems
