"""The constraint graph: legal actions, sequence validation, planning (spec 7).

Each desktop carries a set of hard-constraint edges (:class:`Edge`, spec 7.2)
saying which node must change state before another may be acted on. From that
graph plus a :data:`~tda.core.states.FrameState` this module derives everything
the dataset needs and the spec deliberately does *not* store (spec 7.1):

* :func:`applicable_preconditions` -- which edges gate one ``(verb, target)``;
* :func:`unmet`                    -- which of them are not satisfied yet;
* :func:`legal_actions`            -- "what can be removed now" (V-task truth);
* :func:`validate_sequence`        -- the spec 7.4 replay check over a real log;
* :func:`find_cycles`              -- the spec 7.4 acyclicity check;
* :func:`remaining_plan`           -- "what next", as a shortest legal sequence.

The rules that *derive* the edges live in :mod:`tda.core.graph_rules` and the
family templates in :mod:`tda.core.graph_templates`; both are re-exported here,
so ``from tda.core.graph import ...`` is the only import a caller needs.

Everything is pure except :func:`edges_to_db` / :func:`edges_from_db`.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

from tda.core.graph_rules import (
    BLOCKED_MODES,
    CABLE_PREFIX,
    Edge,
    HARD_TYPES,
    REQUIRED_STATES,
    cable_owner,
    infer_relational_fields,
    propose_edges,
)
from tda.core.graph_templates import apply_template, save_template
from tda.core.model import ActionRec, InstanceRec
from tda.core.states import FrameState, events_from_actions, state_at
from tda.core.taxonomy import Taxonomy

__all__ = [
    "BLOCKED_MODES",
    "Edge",
    "HARD_TYPES",
    "REQUIRED_STATES",
    "apply_template",
    "applicable_preconditions",
    "cable_owner",
    "edges_from_db",
    "edges_to_db",
    "find_cycles",
    "infer_relational_fields",
    "legal_actions",
    "propose_edges",
    "remaining_plan",
    "save_template",
    "unmet",
    "validate_sequence",
]

REMOVED = "removed"
REJECTED = "rejected"

#: The default state of a ``cable:*`` node, mirroring ``taxonomy.yaml``. A
#: cable enters the frame state only once an event names it (see
#: :func:`tda.core.states.state_at`), so an edge may well point at one that is
#: not in the snapshot yet; it is still routed until something releases it.
CABLE_DEFAULT_STATE = "routed"

#: ``necessity`` from strongest to weakest. ``unmet(..., necessity=n)`` looks at
#: every edge at least as strong as ``n``.
NECESSITY_ORDER = ("required", "recommended")

#: Verbs that a hard constraint can block (spec 7.2). ``reorient`` is a capture
#: action on the chassis, not a disassembly step, so nothing gates it.
GATED_VERBS = frozenset({"remove", "displace", "open", "unscrew", "disconnect", "release"})

#: Which states a verb may be applied *from*, where the taxonomy's effect table
#: alone is too permissive. ``remove`` is handled by :data:`REMOVE_FROM_STATES`.
VERB_FROM_STATES: dict[str, frozenset[str]] = {
    "unscrew": frozenset({"fastened"}),
    "disconnect": frozenset({"plugged"}),
    "open": frozenset({"closed"}),
    "release": frozenset({"closed", "routed"}),
    "displace": frozenset({"installed"}),
}

#: Classes that need an intermediate step before ``remove``: a screw has to come
#: loose first, a plug has to come out first. Every other class (a part, a
#: cover, a cage) can be taken straight out of whatever state it is in.
REMOVE_FROM_STATES: dict[str, frozenset[str]] = {
    "screw": frozenset({"loosened"}),
    "connector": frozenset({"unplugged"}),
}

#: Which verb to reach for when a blocker has to change state, least
#: destructive first. Used by :func:`remaining_plan`.
VERB_PREFERENCE = ("open", "release", "unscrew", "disconnect", "displace", "remove")

VerbTarget = tuple[str, str]
ActionLike = Union[ActionRec, VerbTarget]


# --------------------------------------------------------------------------- #
# 1. preconditions
# --------------------------------------------------------------------------- #
def _verb_target(action: ActionLike) -> VerbTarget:
    if isinstance(action, ActionRec):
        return action.verb, action.target
    verb, target = action
    return verb, target


def _active(edges: Iterable[Edge]) -> list[Edge]:
    """Edges a human has not rejected (spec 7.3: proposed / accepted / rejected)."""
    return [e for e in edges if e.status != REJECTED]


def applicable_preconditions(edges: list[Edge], action: ActionLike) -> list[Edge]:
    """Which edges gate this ``(verb, target)``.

    Spec 7.2 gates ``remove`` / ``displace`` / ``open`` on a part, cover or
    latch, and ``unscrew`` / ``disconnect`` on a fastener or plug, on *every*
    hard edge whose ``target`` is that node -- the edge type says what has to
    give way, not which verb it applies to. A verb nothing can block (only
    ``reorient`` today) has no preconditions at all.

    ``action`` is an :class:`~tda.core.model.ActionRec` or a plain
    ``(verb, target)`` pair. Rejected edges are dropped.
    """
    verb, target = _verb_target(action)
    if verb not in GATED_VERBS:
        return []
    return [e for e in _active(edges) if e.target == target]


def _necessity_rank(necessity: str) -> int:
    """How binding this necessity is; 0 is the strongest.

    Spec 7.1 allows only ``required`` and ``recommended``. Anything else is a
    data error, and is ranked as ``required`` on purpose: a typo then makes the
    edge over-binding and noisy rather than silently dropping a constraint out
    of the ground truth.
    """
    try:
        return NECESSITY_ORDER.index(necessity)
    except ValueError:
        return 0


def _blocker_state(state: FrameState, blocker: str) -> Optional[str]:
    """The blocker's state, or ``None`` when it is not a node of this desktop."""
    inst = state.get(blocker)
    if inst is not None:
        return inst.state
    if blocker.startswith(CABLE_PREFIX):
        return CABLE_DEFAULT_STATE
    return None


def unmet(
    edges: list[Edge],
    state: FrameState,
    necessity: str = "required",
) -> list[Edge]:
    """Which of ``edges`` are not satisfied in ``state`` (spec 7.2).

    An edge is satisfied when its blocker's state is one of
    :data:`~tda.core.graph_rules.REQUIRED_STATES` for the edge's type; a blocker
    that is ``removed`` satisfies any edge, and one that is not a node of this
    desktop at all is treated as satisfied rather than invented as a violation.
    A ``cable:*`` blocker missing from the snapshot counts as ``routed``, its
    taxonomy default.

    ``necessity`` is the weakest level to take into account: the default
    ``"required"`` ignores recommended edges, ``"recommended"`` weighs both.
    """
    limit = _necessity_rank(necessity)
    out: list[Edge] = []
    for edge in _active(edges):
        if _necessity_rank(edge.necessity) > limit:
            continue
        current = _blocker_state(state, edge.blocker)
        if current is None or current == REMOVED:
            continue
        if current not in REQUIRED_STATES.get(edge.type, frozenset()):
            out.append(edge)
    return out


# --------------------------------------------------------------------------- #
# 2. legal actions
# --------------------------------------------------------------------------- #
def _verb_effect(tax: Taxonomy, rec_cls: str, attrs: dict, verb: str) -> Optional[str]:
    """The state this verb would put the class in, or ``None`` for no effect."""
    effect = tax.apply_verb(rec_cls, attrs, verb)
    if effect is None or effect[0] != "state":
        return None
    return effect[1]


def _verb_applies(tax: Taxonomy, cls: str, attrs: dict, verb: str, current: str) -> bool:
    """Can this verb be performed on an instance of ``cls`` in state ``current``?

    Independent of the constraint graph: it only asks whether the verb belongs
    to the class, does something, and starts from a sane state.
    """
    new = _verb_effect(tax, cls, attrs, verb)
    if new is None or current == REMOVED or new == current:
        return False
    if verb == "remove":
        allowed = REMOVE_FROM_STATES.get(cls)
    else:
        allowed = VERB_FROM_STATES.get(verb)
    return allowed is None or current in allowed


def _cable_nodes(edges: list[Edge], state: FrameState) -> dict[str, str]:
    """Every virtual cable node this graph mentions, with its current state."""
    keys = {e.blocker for e in edges if e.blocker.startswith(CABLE_PREFIX)}
    keys |= {k for k in state if k.startswith(CABLE_PREFIX)}
    return {k: _blocker_state(state, k) or CABLE_DEFAULT_STATE for k in keys}


def legal_actions(
    instances: dict[str, InstanceRec],
    edges: list[Edge],
    state: FrameState,
    tax: Taxonomy,
    strict: bool = True,
) -> list[VerbTarget]:
    """Every ``(verb, target)`` that is physically possible in ``state``.

    A pair is legal when the verb applies to the target's class (spec 6.3), the
    target is in a state the verb can start from -- a screw is only unscrewed
    while ``fastened``, a latch only opened while ``closed``, a plug only pulled
    while ``plugged``, a part only removed while it is still there -- and no
    precondition of spec 7.2 is unmet. Instances already ``removed`` are skipped.

    ``strict`` (the default) requires recommended edges to be satisfied too;
    ``strict=False`` weighs only the required ones, which is the wider set used
    when checking a real log that took a shortcut.

    Virtual ``cable:*`` nodes named by the graph are included, so a cable that
    has to be released before a part comes free shows up as a legal action too.
    The result is sorted and free of duplicates.
    """
    necessity = "recommended" if strict else "required"
    active = _active(edges)
    out: list[VerbTarget] = []

    def allowed(verb: str, target: str) -> bool:
        return not unmet(applicable_preconditions(active, (verb, target)), state, necessity)

    for key, rec in instances.items():
        inst = state.get(key)
        if inst is None or inst.state == REMOVED:
            continue
        for verb in tax.verbs:
            if not _verb_applies(tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            if allowed(verb, key):
                out.append((verb, key))

    for key, current in _cable_nodes(active, state).items():
        if _verb_applies(tax, "cable", {}, "release", current) and allowed("release", key):
            out.append(("release", key))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# 3. sequence validation (spec 7.4)
# --------------------------------------------------------------------------- #
#: Larger than any logical step, so ``state_at`` folds a whole prefix.
_ALL_STEPS = 10**9


def validate_sequence(
    instances: dict[str, InstanceRec],
    edges: list[Edge],
    actions: list[ActionRec],
    tax: Taxonomy,
) -> list[str]:
    """Replay the observed actions and report every spec 7.4 breach.

    Two checks, per action, against the state the moment before it happened:

    * a **successful** action must have no unmet *required* precondition, else
      ``"step k: <verb> <target> violates <edge>"`` -- one line per unmet edge,
      naming the blocker's actual state;
    * a **failed** attempt must have at least one unmet precondition, else
      ``"step k: failed <verb> <target> has no unmet constraint — missing edge?"``
      which is the cue to add the missing ``blocked_by`` edge by hand.

    Actions are replayed in ``(step, idx)`` order through
    :func:`tda.core.states.events_from_actions`, so two actions in one compound
    step see each other's effect. Recommended edges are ignored: they are a
    preference, not a physical law, and a log that skips one is not wrong.
    """
    active = _active(edges)
    ordered = sorted(actions, key=lambda a: (a.step, a.idx))
    problems: list[str] = []

    for i, action in enumerate(ordered):
        events = events_from_actions(instances, ordered[:i], tax)
        state = state_at(instances, events, _ALL_STEPS, tax)
        gating = applicable_preconditions(active, action)
        bad = unmet(gating, state, "required")
        head = f"step {action.step}: {action.verb} {action.target}"
        if action.result == "success":
            problems.extend(
                f"{head} violates {edge.label()} "
                f"({edge.blocker} is {_blocker_state(state, edge.blocker)!r})"
                for edge in bad
            )
        elif not bad:
            problems.append(
                f"step {action.step}: failed {action.verb} {action.target} "
                "has no unmet constraint — missing edge?"
            )
    return problems


# --------------------------------------------------------------------------- #
# 4. cycles (spec 7.4: the graph must be acyclic)
# --------------------------------------------------------------------------- #
def find_cycles(edges: list[Edge]) -> list[list[str]]:
    """Every cycle of the ``target -> blocker`` graph, as sorted node lists.

    Iterative Tarjan, so a pathological chain cannot blow the recursion limit.
    Returns each strongly connected component of more than one node, plus every
    self-loop; an acyclic graph gives ``[]``. Components and their nodes come
    back sorted, so the result is stable across runs.
    """
    graph: dict[str, list[str]] = {}
    selfish: set[str] = set()
    for edge in _active(edges):
        graph.setdefault(edge.target, []).append(edge.blocker)
        graph.setdefault(edge.blocker, [])
        if edge.target == edge.blocker:
            selfish.add(edge.target)
    for succ in graph.values():
        succ.sort()

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    found: list[list[str]] = []

    for root in sorted(graph):
        if root in index:
            continue
        # (node, iterator position) frames, expanded by hand instead of recursing
        work: list[tuple[str, int]] = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, at = work[-1]
            succ = graph[node]
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
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1 or component[0] in selfish:
                    found.append(sorted(component))
    return sorted(found)


# --------------------------------------------------------------------------- #
# 5. planning
# --------------------------------------------------------------------------- #
def _choose_verb(
    tax: Taxonomy,
    rec: InstanceRec,
    current: str,
    wanted: frozenset[str],
) -> Optional[str]:
    """The least destructive verb that puts ``rec`` into one of ``wanted``."""
    for verb in VERB_PREFERENCE:
        new = _verb_effect(tax, rec.cls, rec.attrs, verb)
        if new in wanted and _verb_applies(tax, rec.cls, rec.attrs, verb, current):
            return verb
    return None


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

    Recommended edges are honoured as well as required ones. Returns ``[]`` when
    the goal is already removed, and ``None`` when the constraints are cyclic
    (spec 7.4 forbids that, :func:`find_cycles` names the offenders).
    """
    active = _active(edges)
    sim: dict[str, str] = {key: inst.state for key, inst in state.items()}
    for key, current in _cable_nodes(active, state).items():
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
        new = _verb_effect(tax, cls, attrs, verb)
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
    if not perform(goal, "remove"):
        return None
    return plan


# --------------------------------------------------------------------------- #
# 6. persistence
# --------------------------------------------------------------------------- #
def edges_to_db(db, desktop: int, edges: list[Edge]) -> list[int]:
    """Upsert every edge into the ``relation`` table; returns their row ids.

    ``(desktop, type, target, blocker)`` is unique, so re-running this after a
    re-import updates the existing rows instead of doubling them.
    """
    return [
        db.add_relation(
            desktop,
            edge.type,
            edge.target,
            edge.blocker,
            necessity=edge.necessity,
            mode=edge.mode,
            reason=edge.reason or None,
            source=edge.source,
            evidence_step=edge.evidence_step,
            status=edge.status,
        )
        for edge in edges
    ]


def edges_from_db(db, desktop: int) -> list[Edge]:
    """Read one desktop's constraint edges back, in insertion order."""
    return [
        Edge(
            type=row["type"],
            target=row["target"],
            blocker=row["blocker"],
            necessity=row["necessity"],
            mode=row["mode"],
            reason=row["reason"] or "",
            source=row["source"],
            evidence_step=row["evidence_step"],
            status=row["status"],
        )
        for row in db.relations(desktop)
    ]
