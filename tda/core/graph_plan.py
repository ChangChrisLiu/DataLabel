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

__all__ = ["Deadlock", "Plan", "clearing_actions", "find_deadlocks", "plan_removal",
           "remaining_plan"]

#: Spec 7.1's weaker necessity: a preference about the order, never a law. The
#: checker, the legal-action set and the deadlock check all bind ``required``
#: only, and now so does the planner when the preference leaves it no plan.
RECOMMENDED = "recommended"

#: The intermediate verbs :func:`remaining_plan` tries, in order, to bring a
#: goal into a state ``remove`` can start from. ``disconnect`` is here rather
#: than left to ``remove`` alone so a planned cable always comes out unplugged.
REMOVAL_PREFIX_VERBS = ("unscrew", "disconnect")

#: Which verb to reach for when a blocker has to change state, least
#: destructive first.
VERB_PREFERENCE = ("open", "release", "unscrew", "disconnect", "displace", "remove")


def _choose_verbs(
    tax: Taxonomy,
    rec: InstanceRec,
    current: str,
    wanted: frozenset[str],
) -> list[str]:
    """Every verb that puts ``rec`` into one of ``wanted``, least destructive first.

    There is usually more than one way to get a blocker out of the way, and they
    are not gated alike: a ``cable_clip`` is satisfied by ``open`` *or*
    ``release`` -- both leave it ``open`` -- but ``open`` waits on the clip's own
    screws and covers while ``release`` waits only on what covers it. Taking the
    first answer and stopping called a plannable machine deadlocked (the panel
    then refused a legitimate edge), so the caller gets the whole set and tries
    them in order.
    """
    out: list[str] = []
    for verb in VERB_PREFERENCE:
        new = verb_effect(tax, rec.cls, rec.attrs, verb)
        if new in wanted and verb_applies(tax, rec.cls, rec.attrs, verb, current):
            out.append(verb)
    return out


def _choose_verb(
    tax: Taxonomy,
    rec: InstanceRec,
    current: str,
    wanted: frozenset[str],
) -> Optional[str]:
    """The least destructive verb that puts ``rec`` into one of ``wanted``."""
    verbs = _choose_verbs(tax, rec, current, wanted)
    return verbs[0] if verbs else None


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


@dataclass(frozen=True)
class Plan:
    """A removal plan, and whether a preference had to be set aside for it."""

    actions: tuple[VerbTarget, ...]
    #: ``True`` when the plan ignores the recommended edges because honouring
    #: them left no plan at all (spec 7.1: ``recommended`` is a preference).
    relaxed: bool = False
    #: The recommended edges that were dropped to get it.
    dropped: tuple[Edge, ...] = ()

    def __iter__(self):
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)


def plan_removal(
    instances: dict[str, InstanceRec],
    edges: list[Edge],
    state: FrameState,
    goal: str,
    tax: Taxonomy,
) -> Optional[Plan]:
    """:func:`remaining_plan`, plus whether a preference was set aside.

    Two passes, because spec 7.1's ``recommended`` is a preference and not a
    necessity: first with every active edge, so a plan that can honour the
    preferred order does; and if that leaves no plan, again with the
    ``required`` edges alone, which is what the checker, the legal-action set
    and the deadlock check all bind to. The second answer is marked
    :attr:`Plan.relaxed` and carries the edges it ignored, so a caller can say
    so instead of pretending the preference was met.

    ``None`` only when there is no plan under the required edges either.
    """
    strict = _plan_once(instances, active_edges(edges), state, goal, tax)
    if strict is not None:
        return Plan(actions=tuple(strict))
    required = [e for e in active_edges(edges) if e.necessity != RECOMMENDED]
    dropped = tuple(e for e in active_edges(edges) if e.necessity == RECOMMENDED)
    if not dropped:
        return None
    relaxed = _plan_once(instances, required, state, goal, tax)
    if relaxed is None:
        return None
    return Plan(actions=tuple(relaxed), relaxed=True, dropped=dropped)


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

    Recommended edges are honoured when they can be; when they cannot, they are
    dropped rather than allowed to fail the plan (:func:`plan_removal` says
    which, and is the call to make when that matters). Returns ``[]`` when the
    goal is already removed, and ``None`` when there is no answer under the
    required edges: the goal is not an instance, its class cannot be removed at
    all (a latch, a lever, the chassis -- spec 6.3), a blocker can never reach a
    state the edge accepts (a dead end), or the required constraints deadlock
    (spec 7.4 forbids that, and :func:`find_deadlocks` names the offenders).
    """
    plan = plan_removal(instances, edges, state, goal, tax)
    return None if plan is None else list(plan.actions)


def _plan_once(
    instances: dict[str, InstanceRec],
    active: list[Edge],
    state: FrameState,
    goal: str,
    tax: Taxonomy,
) -> Optional[list[VerbTarget]]:
    """One pass of the planner over exactly the edges it is given."""
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

    def applies(key: str, verb: str) -> bool:
        rec = instances.get(key)
        cls = rec.cls if rec is not None else "cable"
        attrs = rec.attrs if rec is not None else {}
        return verb_applies(tax, cls, attrs, verb, sim.get(key, ""))

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
        if not applies(key, verb):
            # clearing the way changed this node too -- a clip that had to be
            # released so the part could be swung aside is open already, and
            # emitting "open the clip" now would be an illegal step. The caller
            # rolls back and tries its next alternative, which finds the world
            # as it was.
            return False
        plan.append((verb, key))
        apply(key, verb)
        return True

    def ensure(key: str, wanted: frozenset[str]) -> bool:
        """Get ``key`` into one of ``wanted``, doing whatever that takes.

        Every verb that would satisfy the edge is tried, least destructive
        first, and the first one that can actually be planned wins: a cable clip
        that cannot be ``open``ed (its own screws are in the way) may still be
        ``release``d. A failed attempt is rolled back -- the actions it emitted
        and the states it simulated -- so the next alternative starts from where
        this one did.
        """
        current = sim.get(key)
        if current is None or current == REMOVED or current in wanted:
            return True  # unknown or already good enough
        rec = instances.get(key)
        if rec is None:
            if not key.startswith(CABLE_PREFIX):
                return True
            rec = InstanceRec(key=key, desktop=0, cls="cable")
        for verb in _choose_verbs(tax, rec, current, wanted):
            mark, snapshot = len(plan), dict(sim)
            if perform(key, verb):
                return True
            del plan[mark:]
            sim.clear()
            sim.update(snapshot)
        return False

    if sim.get(goal) == REMOVED:
        return []
    if goal not in instances:
        return None

    target = _removal_goal(instances, goal)
    chain = _removal_chain(tax, instances[target], sim.get(target, ""))
    if chain is None:
        return None
    for verb in chain:
        if sim.get(goal) == REMOVED:
            break            # an attached cascade took the goal out on the way
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


def _binding(edges: Iterable[Edge], necessity: str = "required") -> list[Edge]:
    """The edges that actually gate something: active, hard, settled.

    ``necessity`` is the weakest level to weigh, like
    :func:`~tda.core.graph.unmet`: the default ``required`` is what a deadlock
    is judged on, because a ``recommended`` edge is a preference the planner
    drops rather than fails on. ``recommended`` weighs both, which is how the
    panel finds preferences that contradict each other and says so without
    refusing anything.
    """
    return [e for e in active_edges(edges)
            if e.type in REQUIRED_STATES
            and (necessity == RECOMMENDED or e.necessity != RECOMMENDED)
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


#: What :func:`clearing_actions` says when the edge needs nothing done at all
#: (the blocker already satisfies it, or is not a node of this desktop).
SATISFIED: list[VerbTarget] = []


def clearing_actions(
    edge: Edge, instances: dict[str, InstanceRec], sim: dict[str, str], tax: Taxonomy
) -> Optional[list[VerbTarget]]:
    """Every action that would satisfy ``edge``, least destructive first.

    ``None`` when nothing needs doing -- the blocker already satisfies the edge,
    or is not a node of this desktop. An **empty list** is the opposite and the
    important case: something has to change and no verb can change it, a *dead
    end* rather than a loop (:func:`remaining_plan` answers "no plan" for it and
    :func:`find_deadlocks` reports nothing, because there is no loop to name).

    The planner's own alternatives (:func:`_choose_verbs`), so the two agree on
    what "clearing a blocker" can mean.
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
    return [(verb, edge.blocker)
            for verb in _choose_verbs(tax, rec, current, wanted)]


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
    necessity: str = "required",
) -> list[Deadlock]:
    """Every deadlock of actions the graph holds (spec 7.4), in a stable order.

    Nodes are ``(verb, instance)`` actions reachable from some part's removal;
    ``(v, T)`` waits on :func:`clearing_actions` of every binding edge ``(T, B)``
    whose type and mode gate ``v``. The dependency is **AND over the edges, OR
    over each edge's alternatives**: a part waits for *all* of its edges, but
    each edge is satisfied by *any* verb that would put its blocker into an
    accepted state -- a cable clip can be opened or released, and the two are
    not gated alike.

    So the answer is computed in three steps:

    1. expand the reachable action graph from the goal actions;
    2. a least fixpoint over it: an action is **doable** when every edge gating
       it either needs nothing or has at least one doable alternative. Iterating
       from the actions that wait on nothing settles it;
    3. what is left is stuck, for one of two reasons. An edge with **no**
       alternative at all is a *dead end* -- nothing can ever clear it, which is
       a modelling gap, not a loop, and is not reported here (the planner
       answers "no plan" for it). An edge all of whose alternatives are stuck is
       a *loop*: those are the arrows this looks for a cycle in, and each cycle
       found is one :class:`Deadlock`.

    ``necessity`` is the weakest level that binds (see :func:`_binding`):
    ``required`` by default, because a recommended edge is a preference the
    planner drops. ``state`` defaults to the initial state, which is the state
    the graph is written about.
    """
    binding = _binding(edges, necessity)
    sim = _simulated(binding, instances, tax, state)
    by_target: dict[str, list[Edge]] = {}
    for edge in binding:
        by_target.setdefault(edge.target, []).append(edge)

    def gating(action: VerbTarget) -> list[tuple[Edge, list[VerbTarget]]]:
        """The edges in this action's way, each with its alternatives."""
        verb, node = action
        out: list[tuple[Edge, list[VerbTarget]]] = []
        for edge in by_target.get(node, ()):
            if verb not in gated_verbs(edge.type, edge.mode):
                continue
            alternatives = clearing_actions(edge, instances, sim, tax)
            if alternatives is not None:      # None: nothing to do for this edge
                out.append((edge, sorted(alternatives)))
        return out

    # 1. the reachable action graph (an action nobody could ask for is not a
    #    deadlock, it is unreachable)
    waits: dict[VerbTarget, list[tuple[Edge, list[VerbTarget]]]] = {}
    queue = sorted(set(_goal_actions(instances, sim, tax)))
    while queue:
        action = queue.pop()
        if action in waits:
            continue
        waits[action] = gating(action)
        for _edge, alternatives in waits[action]:
            queue.extend(a for a in alternatives if a not in waits)

    # 2. least fixpoint: who can be done at all
    doable: set[VerbTarget] = set()
    changed = True
    while changed:
        changed = False
        for action, gates in waits.items():
            if action in doable:
                continue
            if all(any(a in doable for a in alternatives) for _edge, alternatives in gates):
                doable.add(action)
                changed = True

    # 3. among the stuck ones, the arrows that are loops rather than dead ends
    graph: dict[VerbTarget, list[VerbTarget]] = {}
    why: dict[tuple[VerbTarget, VerbTarget], Edge] = {}
    for action, gates in waits.items():
        if action in doable:
            continue
        successors: list[VerbTarget] = []
        for edge, alternatives in gates:
            if not alternatives or any(a in doable for a in alternatives):
                continue                      # satisfied, or a dead end: no arrow
            for nxt in alternatives:
                successors.append(nxt)
                why.setdefault((action, nxt), edge)
        graph[action] = sorted(set(successors))

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
