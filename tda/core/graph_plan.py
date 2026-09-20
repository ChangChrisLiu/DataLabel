"""Removal planning over the constraint graph -- the "what next" answer.

Split out of :mod:`tda.core.graph` to keep both modules readable; the reasoning
layer re-exports :func:`remaining_plan`, so callers still import it from
``tda.core.graph``.

It also owns the spec 7.4 acyclicity check, :func:`find_deadlocks`, because
the planner is where "the graph is usable" is actually decided. The check used
to be :func:`find_cycles`: a loop in the ``target -> blocker`` graph of
*instances*, whatever each edge gated. Since an edge only gates some verbs
(:data:`~tda.core.graph_rules.GATES`, and ``blocked_by`` by its mode) that was
both too strict and too vague -- a bracket that sits on a screw head is
``fastened_by(bracket, screw)`` plus ``blocked_by(screw, bracket,
physical_path)``, which no more deadlocks than a knot ties itself: nothing
gates ``unscrew``, so the screw comes out and the bracket follows. The panel
refused that edge and the command line would have exited 1 on a graph the
planner answers for.

So a cycle is a **deadlock of actions**, not of instances: the nodes are
``(verb, instance)`` pairs, an action depends on the action that clears each
edge gating it, and a deadlock is a loop among those. One definition, used by
the editor, the panel, the derivation and the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from tda.core.graph_rules import (
    CABLE_PREFIX,
    REMOVED,
    REQUIRED_STATES,
    Edge,
    VerbTarget,
    active_edges,
    cable_nodes,
    gated_verbs,
    is_provisional,
    verb_applies,
    verb_effect,
)
from tda.core.model import InstanceRec
from tda.core.states import FrameState, initial_state
from tda.core.taxonomy import Taxonomy

__all__ = ["Deadlock", "clearing_action", "find_deadlocks", "remaining_plan"]

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
    #: The actions being expanded right now. Keyed by ``(verb, key)`` and not by
    #: the instance: "to take A out, first swing B aside; to swing B aside,
    #: first swing A aside" is a plan (swing A, swing B, take A out), while an
    #: instance-level guard called it a cycle and answered "no plan" for a graph
    #: with no deadlock in it -- which is the definition
    #: :func:`find_deadlocks` now uses, and the two must agree.
    visiting: set[VerbTarget] = set()

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
        if (verb, key) in visiting:
            return False  # a deadlock: this action is waiting for itself
        visiting.add((verb, key))
        try:
            for edge in by_target.get(key, ()):
                # the same table the checker reads: a `connected_to` edge (and a
                # `cable_tension` block) is in the way of taking the part out,
                # not of swinging it aside, so a `displace` step does not
                # schedule its disconnects
                if verb not in gated_verbs(edge.type, edge.mode):
                    continue
                wanted = REQUIRED_STATES.get(edge.type, frozenset())
                if not ensure(edge.blocker, wanted):
                    return False
        finally:
            visiting.discard((verb, key))
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


# --------------------------------------------------------------------------- #
# spec 7.4: no deadlock (the acyclicity check)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Deadlock:
    """A loop of actions, each waiting for the next: nothing in it can be done.

    ``actions`` is the loop in order, ``edges`` the constraint edges that
    created each step of it -- which is what a human has to change to get out.
    """

    actions: tuple[VerbTarget, ...]
    edges: tuple[Edge, ...]

    def chain(self) -> str:
        """``unscrew screw.01 -> remove bracket.01 -> unscrew screw.01``."""
        loop = [*self.actions, self.actions[0]]
        return " -> ".join(f"{verb} {node}" for verb, node in loop)

    def label(self) -> str:
        """The chain plus the edges that hold it together."""
        return f"{self.chain()} [{', '.join(e.label() for e in self.edges)}]"


def _binding(edges: Iterable[Edge]) -> list[Edge]:
    """The edges that actually gate something: active, required, hard, settled."""
    return [e for e in active_edges(edges)
            if e.type in REQUIRED_STATES
            and e.necessity != "recommended"
            and not is_provisional(e.target) and not is_provisional(e.blocker)]


def _simulated(edges: list[Edge], instances: dict[str, InstanceRec], tax: Taxonomy,
               state: Optional[FrameState]) -> dict[str, str]:
    """Every node's state to reason from: the frame state, then class defaults."""
    state = state if state is not None else initial_state(instances, tax)
    sim = {key: inst.state for key, inst in state.items()}
    for key, current in cable_nodes(edges, state).items():
        sim.setdefault(key, current)
    for key, rec in instances.items():
        sim.setdefault(key, tax.default_state(rec.cls))
    return sim


def clearing_action(
    edge: Edge, instances: dict[str, InstanceRec], sim: dict[str, str], tax: Taxonomy
) -> Optional[VerbTarget]:
    """The action that satisfies ``edge``, or ``None`` when none is needed.

    The planner's own answer (:func:`_choose_verb`): the least destructive verb
    that puts the blocker into a state :data:`REQUIRED_STATES` accepts. ``None``
    when the blocker already satisfies the edge, is not a node of this desktop,
    or cannot be moved at all -- the last is a dead end rather than a loop, and
    :func:`remaining_plan` reports it as "no plan" without any cycle.
    """
    wanted = REQUIRED_STATES.get(edge.type, frozenset())
    current = sim.get(edge.blocker)
    if current is None or current == REMOVED or current in wanted:
        return None
    rec = instances.get(edge.blocker)
    if rec is None:
        if not edge.blocker.startswith(CABLE_PREFIX):
            return None
        rec = InstanceRec(key=edge.blocker, desktop=0, cls="cable")
    verb = _choose_verb(tax, rec, current, wanted)
    return None if verb is None else (verb, edge.blocker)


def _goal_actions(instances: dict[str, InstanceRec], sim: dict[str, str],
                  tax: Taxonomy) -> list[VerbTarget]:
    """Every action a caller could ask for: taking each part out of the machine."""
    roots: list[VerbTarget] = []
    for key in sorted(instances):
        if is_provisional(key):
            continue
        target = _removal_goal(instances, key)
        chain = _removal_chain(tax, instances[target], sim.get(target, ""))
        roots.extend((verb, target) for verb in (chain or ()))
    return roots


def find_deadlocks(
    edges: list[Edge],
    instances: dict[str, InstanceRec],
    tax: Taxonomy,
    state: Optional[FrameState] = None,
) -> list[Deadlock]:
    """Every deadlock of actions the graph holds (spec 7.4), in a stable order.

    Nodes are ``(verb, instance)`` actions reachable from some part's removal;
    ``(v, T)`` depends on :func:`clearing_action` of every active required edge
    ``(T, B)`` whose type and mode gate ``v``. A strongly connected component of
    that graph (or a self-loop) is a deadlock: each action in it waits for the
    next, so none of them can ever be done, and :func:`remaining_plan` answers
    ``None`` for every node in the loop.

    ``state`` defaults to the initial state, which is the state the graph is
    written about.
    """
    binding = _binding(edges)
    sim = _simulated(binding, instances, tax, state)
    by_target: dict[str, list[Edge]] = {}
    for edge in binding:
        by_target.setdefault(edge.target, []).append(edge)

    def dependencies(action: VerbTarget) -> list[tuple[VerbTarget, Edge]]:
        verb, node = action
        out: list[tuple[VerbTarget, Edge]] = []
        for edge in by_target.get(node, ()):
            if verb not in gated_verbs(edge.type, edge.mode):
                continue
            clear = clearing_action(edge, instances, sim, tax)
            if clear is not None:
                out.append((clear, edge))
        return sorted(out, key=lambda pair: pair[0])

    # expand from the goal actions only: an action nobody could ever ask for is
    # not a deadlock, it is unreachable
    graph: dict[VerbTarget, list[VerbTarget]] = {}
    why: dict[tuple[VerbTarget, VerbTarget], Edge] = {}
    queue = sorted(set(_goal_actions(instances, sim, tax)))
    while queue:
        action = queue.pop()
        if action in graph:
            continue
        deps = dependencies(action)
        graph[action] = [nxt for nxt, _edge in deps]
        for nxt, edge in deps:
            why.setdefault((action, nxt), edge)
            if nxt not in graph:
                queue.append(nxt)

    found: list[Deadlock] = []
    for component in _components(graph):
        loop = _one_cycle(graph, component)
        if loop is None:
            continue
        pairs = [(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]
        found.append(Deadlock(
            actions=tuple(loop),
            edges=tuple(dict.fromkeys(why[pair] for pair in pairs if pair in why)),
        ))
    return sorted(found, key=lambda d: d.actions)


def _components(graph: dict[VerbTarget, list[VerbTarget]]) -> list[list[VerbTarget]]:
    """Tarjan's strongly connected components, iterative and stable.

    Iterative because a pathological chain must not blow the recursion limit;
    sorted because two runs of the same graph have to report the same thing.
    """
    index: dict[VerbTarget, int] = {}
    low: dict[VerbTarget, int] = {}
    on_stack: set[VerbTarget] = set()
    stack: list[VerbTarget] = []
    counter = 0
    out: list[list[VerbTarget]] = []

    for root in sorted(graph):
        if root in index:
            continue
        work: list[tuple[VerbTarget, int]] = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, at = work[-1]
            succ = graph.get(node, ())
            if at < len(succ):
                work[-1] = (node, at + 1)
                nxt = succ[at]
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component: list[VerbTarget] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                out.append(sorted(component))
    return out


def _one_cycle(graph: dict[VerbTarget, list[VerbTarget]],
               component: list[VerbTarget]) -> Optional[list[VerbTarget]]:
    """One concrete loop inside a component, in order, or ``None`` for no loop.

    A single node is a deadlock only if it waits for itself; anything larger is
    walked once, taking the first successor inside the component each time,
    which ends in the loop the reader can follow.
    """
    members = set(component)
    if len(component) == 1:
        node = component[0]
        return [node] if node in graph.get(node, ()) else None
    start = component[0]
    path: list[VerbTarget] = [start]
    seen = {start: 0}
    node = start
    while True:
        nxt = next((n for n in graph.get(node, ()) if n in members), None)
        if nxt is None:  # cannot happen inside a component; never loop for ever
            return None
        if nxt in seen:
            return path[seen[nxt]:]
        seen[nxt] = len(path)
        path.append(nxt)
        node = nxt
