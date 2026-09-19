"""Removal planning over the constraint graph -- the "what next" answer.

Split out of :mod:`tda.core.graph` to keep both modules readable; the reasoning
layer re-exports :func:`remaining_plan`, so callers still import it from
``tda.core.graph``.
"""
from __future__ import annotations

from typing import Optional

from tda.core.graph_rules import (
    CABLE_PREFIX,
    REMOVED,
    REQUIRED_STATES,
    Edge,
    VerbTarget,
    active_edges,
    cable_nodes,
    gated_verbs,
    verb_applies,
    verb_effect,
)
from tda.core.model import InstanceRec
from tda.core.states import FrameState
from tda.core.taxonomy import Taxonomy

__all__ = ["remaining_plan"]

#: The intermediate verbs :func:`remaining_plan` tries, in order, to bring a
#: goal into a state ``remove`` can start from. ``disconnect`` is here rather
#: than left to ``remove`` alone so a planned cable always comes out unplugged.
REMOVAL_PREFIX_VERBS = ("unscrew", "disconnect")

#: Which verb to reach for when a blocker has to change state, least
#: destructive first.
VERB_PREFERENCE = ("open", "release", "unscrew", "disconnect", "displace", "remove")


def _choose_verb(
    tax: Taxonomy,
    rec: InstanceRec,
    current: str,
    wanted: frozenset[str],
) -> Optional[str]:
    """The least destructive verb that puts ``rec`` into one of ``wanted``."""
    for verb in VERB_PREFERENCE:
        new = verb_effect(tax, rec.cls, rec.attrs, verb)
        if new in wanted and verb_applies(tax, rec.cls, rec.attrs, verb, current):
            return verb
    return None


def _removal_chain(tax: Taxonomy, rec: InstanceRec, current: str) -> Optional[list[str]]:
    """The verbs that take ``rec`` from ``current`` to ``removed``, in order.

    ``None`` when the class cannot be removed at all -- a latch, a lever or the
    chassis is not in ``remove.applies_to`` (spec 6.3), and asking to plan one
    out of the machine is a question with no answer, not an empty plan.

    Otherwise the chain is the intermediate verbs of
    :data:`REMOVAL_PREFIX_VERBS` that apply, then ``remove`` -- so a screw is
    unscrewed and a plug pulled first. It stops early when an intermediate verb
    already reaches ``removed``, which is what happens to a non-captive screw.
    """
    if current == REMOVED:
        return []
    if verb_effect(tax, rec.cls, rec.attrs, "remove") is None:
        return None
    chain: list[str] = []
    state = current
    for verb in REMOVAL_PREFIX_VERBS:
        if not verb_applies(tax, rec.cls, rec.attrs, verb, state):
            continue
        chain.append(verb)
        state = verb_effect(tax, rec.cls, rec.attrs, verb) or state
        if state == REMOVED:
            return chain
    if not verb_applies(tax, rec.cls, rec.attrs, "remove", state):
        return None
    chain.append("remove")
    return chain


def _removal_goal(instances: dict[str, InstanceRec], goal: str) -> str:
    """Which instance actually has to leave for ``goal`` to be gone.

    A captive screw stays in its bracket for good (spec 7.1), so ``removed`` is
    reachable for it only through the attached cascade when its parent comes
    out: planning one is really planning its parent.
    """
    rec = instances.get(goal)
    if rec is None or rec.cls != "screw" or not rec.attrs.get("captive"):
        return goal
    if rec.parent and rec.attached and rec.parent in instances:
        return rec.parent
    return goal


def remaining_plan(
    instances: dict[str, InstanceRec],
    edges: list[Edge],
    state: FrameState,
    goal: str,
    tax: Taxonomy,
) -> Optional[list[VerbTarget]]:
    """A shortest legal sequence from ``state`` that ends with ``goal`` removed.

    A breadth-first search over abstract states is hopeless here (a 40-instance
    desktop has far too many), but the constraint graph already *is* the search:
    to remove a node you must first put each of its unmet blockers into an
    accepted state, which is itself an action with its own blockers. So this
    expands the graph depth-first and emits the actions in topological order --
    every blocker before the thing it blocks -- which is the shortest sequence
    that touches nothing irrelevant.

    The goal itself is reached through :func:`_removal_chain`, so a screw is
    unscrewed, a plug is pulled before its cable is taken out, and a captive
    screw is planned as the removal of the parent it can never leave without.

    Recommended edges are honoured as well as required ones. Returns ``[]`` when
    the goal is already removed, and ``None`` when there is no answer: the goal
    is not an instance, its class cannot be removed at all (a latch, a lever,
    the chassis -- spec 6.3), or the constraints are cyclic (spec 7.4 forbids
    that, and :func:`find_cycles` names the offenders).
    """
    active = active_edges(edges)
    sim: dict[str, str] = {key: inst.state for key, inst in state.items()}
    for key, current in cable_nodes(active, state).items():
        sim.setdefault(key, current)
    for key in instances:
        sim.setdefault(key, tax.default_state(instances[key].cls))

    by_target: dict[str, list[Edge]] = {}
    for edge in active:
        by_target.setdefault(edge.target, []).append(edge)

    children = {
        key: sorted(k for k, r in instances.items() if r.attached and r.parent == key)
        for key in instances
    }
    plan: list[VerbTarget] = []
    visiting: set[str] = set()

    def apply(key: str, verb: str) -> None:
        rec = instances.get(key)
        cls = rec.cls if rec is not None else "cable"
        attrs = rec.attrs if rec is not None else {}
        new = verb_effect(tax, cls, attrs, verb)
        if new is None:
            return
        sim[key] = new
        if new == REMOVED and cls != "connector":  # spec 3.3 attached cascade
            for child in children.get(key, ()):
                sim[child] = REMOVED

    def perform(key: str, verb: str) -> bool:
        """Emit ``verb`` on ``key`` after clearing everything that blocks it."""
        if key in visiting:
            return False  # a cycle: the graph is not a partial order
        visiting.add(key)
        try:
            for edge in by_target.get(key, ()):
                # the same table the checker reads: a `connected_to` edge is in
                # the way of taking the part out, not of swinging it aside, so a
                # `displace` step does not schedule its disconnects
                if verb not in gated_verbs(edge.type):
                    continue
                wanted = REQUIRED_STATES.get(edge.type, frozenset())
                if not ensure(edge.blocker, wanted):
                    return False
        finally:
            visiting.discard(key)
        plan.append((verb, key))
        apply(key, verb)
        return True

    def ensure(key: str, wanted: frozenset[str]) -> bool:
        """Get ``key`` into one of ``wanted``, doing whatever that takes."""
        current = sim.get(key)
        if current is None or current == REMOVED or current in wanted:
            return True  # unknown or already good enough
        rec = instances.get(key)
        if rec is None:
            if not key.startswith(CABLE_PREFIX):
                return True
            rec = InstanceRec(key=key, desktop=0, cls="cable")
        verb = _choose_verb(tax, rec, current, wanted)
        if verb is None:
            return False
        return perform(key, verb)

    if sim.get(goal) == REMOVED:
        return []
    if goal not in instances:
        return None

    target = _removal_goal(instances, goal)
    chain = _removal_chain(tax, instances[target], sim.get(target, ""))
    if chain is None:
        return None
    for verb in chain:
        if not perform(target, verb):
            return None
    if sim.get(goal) != REMOVED:  # the plan did not actually achieve the goal
        return None
    return plan
