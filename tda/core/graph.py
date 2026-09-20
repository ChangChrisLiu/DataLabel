"""The constraint graph: legal actions, sequence validation, cycles (spec 7).

Each desktop carries a set of hard-constraint edges (:class:`Edge`, spec 7.2)
saying which node must change state before another may be acted on. From that
graph plus a :data:`~tda.core.states.FrameState` this module derives everything
the dataset needs and the spec deliberately does *not* store (spec 7.1):

* :func:`applicable_preconditions` -- which edges gate one ``(verb, target)``;
* :func:`unmet`                    -- which of them are not satisfied yet;
* :func:`legal_actions`            -- "what can be removed now" (V-task truth);
* :func:`validate_sequence`        -- the spec 7.4 replay check over a real log;
* :func:`find_deadlocks`           -- the spec 7.4 acyclicity check, as a loop
  of *actions* (it lives with the planner, which decides the same question);
* :func:`remaining_plan`           -- "what next", as a shortest legal sequence;
* :func:`graph_version`            -- which graph an export shipped.

The graph stack is four modules, bottom up: :mod:`tda.core.graph_rules` (the
vocabulary, the spec 7.2 semantics and the spec 7.3 derivation rules),
:mod:`tda.core.graph_plan` (the planner), :mod:`tda.core.graph_templates` (the
family templates) and this one. Everything a caller needs is re-exported here,
so ``from tda.core.graph import ...`` is the only import needed.

Everything is pure except :func:`edges_to_db` / :func:`edges_from_db`.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Union

from tda.core.graph_plan import (
    DeadEnd,
    Deadlock,
    find_dead_ends,
    find_deadlocks,
    remaining_plan,
)
from tda.core.graph_rules import (
    BLOCKED_MODES,
    Edge,
    GATED_VERBS,
    GATES,
    HARD_TYPES,
    REMOVED,
    REQUIRED_STATES,
    VerbTarget,
    active_edges,
    blocker_state,
    cable_nodes,
    cable_owner,
    connector_owner,
    gated_verbs,
    infer_relational_fields,
    is_provisional,
    propose_edges,
    unresolved_fan_owners,
    unresolved_kind,
    unresolved_relations,
    verb_applies,
)
from tda.core.graph_templates import apply_template, save_template
from tda.core.model import ActionRec, InstanceRec
from tda.core.states import FrameState, events_from_actions, state_at
from tda.core.taxonomy import Taxonomy

__all__ = [
    "BLOCKED_MODES",
    "DeadEnd",
    "Deadlock",
    "Edge",
    "GATED_VERBS",
    "GATES",
    "HARD_TYPES",
    "REQUIRED_STATES",
    "apply_template",
    "applicable_preconditions",
    "cable_owner",
    "connector_owner",
    "constraint_edges",
    "edge_digest",
    "edges_from_db",
    "edges_to_db",
    "find_dead_ends",
    "find_deadlocks",
    "gated_verbs",
    "graph_version",
    "infer_relational_fields",
    "is_provisional",
    "legal_actions",
    "propose_edges",
    "remaining_plan",
    "save_template",
    "unmet",
    "unresolved_fan_owners",
    "unresolved_kind",
    "unresolved_relations",
    "validate_sequence",
]

#: ``necessity`` from strongest to weakest. ``unmet(..., necessity=n)`` looks at
#: every edge at least as strong as ``n``.
NECESSITY_ORDER = ("required", "recommended")

ActionLike = Union[ActionRec, VerbTarget]


# --------------------------------------------------------------------------- #
# 1. preconditions
# --------------------------------------------------------------------------- #
def _verb_target(action: ActionLike) -> VerbTarget:
    if isinstance(action, ActionRec):
        return action.verb, action.target
    verb, target = action
    return verb, target


def applicable_preconditions(edges: list[Edge], action: ActionLike) -> list[Edge]:
    """Which edges gate this ``(verb, target)``.

    Every hard edge whose ``target`` is that node **and whose type gates this
    verb** -- see :data:`~tda.core.graph_rules.GATES`. The type says what has to
    give way *and* what it holds up: a screw stops the part moving but not the
    plug in its socket being pulled, and a plugged cable stops the part being
    taken away but not swung aside. A verb nothing can block (only ``reorient``
    today) has no preconditions at all.

    A ``blocked_by`` edge gates by its ``mode`` (spec 7.1): a cable under
    tension stops the part leaving, a blocked path stops it moving as well, and
    only ``tool_access`` stops you reaching its own screws and plugs.

    ``action`` is an :class:`~tda.core.model.ActionRec` or a plain
    ``(verb, target)`` pair. Rejected and orphaned edges are dropped.
    """
    verb, target = _verb_target(action)
    if verb not in GATED_VERBS:
        return []
    return [e for e in active_edges(edges)
            if e.target == target and verb in gated_verbs(e.type, e.mode)]


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
    for edge in active_edges(edges):
        if _necessity_rank(edge.necessity) > limit:
            continue
        current = blocker_state(state, edge.blocker)
        if current is None or current == REMOVED:
            continue
        if current not in REQUIRED_STATES.get(edge.type, frozenset()):
            out.append(edge)
    return out


# --------------------------------------------------------------------------- #
# 2. legal actions
# --------------------------------------------------------------------------- #
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
    active = active_edges(edges)
    out: list[VerbTarget] = []

    def allowed(verb: str, target: str) -> bool:
        return not unmet(applicable_preconditions(active, (verb, target)), state, necessity)

    for key, rec in instances.items():
        inst = state.get(key)
        if inst is None or inst.state == REMOVED:
            continue
        for verb in tax.verbs:
            if not verb_applies(tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            if allowed(verb, key):
                out.append((verb, key))

    for key, current in cable_nodes(active, state).items():
        if verb_applies(tax, "cable", {}, "release", current) and allowed("release", key):
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
      ``"step k: failed <verb> <target> has no unmet constraint - missing edge?"``
      which is the cue to add the missing ``blocked_by`` edge by hand.

    Actions are replayed in ``(step, idx)`` order through
    :func:`tda.core.states.events_from_actions`, so two actions in one compound
    step see each other's effect. Recommended edges are ignored: they are a
    preference, not a physical law, and a log that skips one is not wrong.
    """
    active = active_edges(edges)
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
                f"({edge.blocker} is {blocker_state(state, edge.blocker)!r})"
                for edge in bad
            )
        elif not bad:
            problems.append(
                f"step {action.step}: failed {action.verb} {action.target} "
                "has no unmet constraint - missing edge?"
            )
    return problems


# --------------------------------------------------------------------------- #
# 6. persistence
# --------------------------------------------------------------------------- #
def edges_to_db(db, desktop: int, edges: list[Edge]) -> list[int]:
    """Upsert every edge into the ``relation`` table; returns their row ids.

    ``(desktop, type, target, blocker)`` is unique, so re-running this after a
    re-import updates the existing rows instead of doubling them.

    One transaction for the whole list, not one per edge: a desktop's graph is a
    set, and a run interrupted after two thousand of its two and a half thousand
    edges would leave a graph that is neither the old one nor the new one. The
    block is re-entrant, so a caller that is already inside a transaction of its
    own still commits once, with everything else it did.
    """
    with db.transaction():
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


# --------------------------------------------------------------------------- #
# 7. graph version
# --------------------------------------------------------------------------- #
def constraint_edges(edges: list[Edge]) -> list[Edge]:
    """The subset of an edge list that is actually the constraint graph.

    The five hard types of spec 7.2, on settled instances. Two things share the
    ``relation`` table without being constraints and must not reach a digest, a
    cycle check or a replay:

    * the ``partner_of`` / ``is_pre-request_of`` / ``related_to`` rows the Label
      Studio import files there -- annotations of another kind, which gate
      nothing and have no :data:`REQUIRED_STATES` entry;
    * any edge naming a provisional ``ls:*`` key, which is a shape somebody drew
      rather than a part anybody has decided exists (spec 3.2).
    """
    return [
        e for e in edges
        if e.type in HARD_TYPES
        and not is_provisional(e.target) and not is_provisional(e.blocker)
    ]


def edge_digest(edges: list[Edge]) -> Optional[str]:
    """A 16-hex-character content hash of an edge set, or ``None`` when it is empty.

    Only the constraint edges count (:func:`constraint_edges`), so the caller
    cannot change the answer by handing in more or less of the table than the
    next caller did -- which is exactly how the stamped version and the accessor
    came to disagree on the two desktops that carry Label Studio rows.

    What goes in is what changes the *meaning* of the graph -- type, target,
    blocker, necessity, mode and status -- sorted, so two runs that derive the
    same constraints agree whatever order they found them in. ``reason`` and
    ``evidence_step`` are prose and provenance: re-wording a reason must not
    look like a different graph. ``source`` is out for the same reason, so a
    human accepting a rule edge by hand does not invalidate every export that
    quoted the version.
    """
    wanted = constraint_edges(edges)
    if not wanted:
        return None
    body = "\n".join(sorted(
        f"{e.type}|{e.target}|{e.blocker}|{e.necessity}|{e.mode or ''}|{e.status}"
        for e in wanted
    ))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def graph_version(db, desktop: int) -> Optional[str]:
    """The content hash of one desktop's stored constraint graph.

    The one definition: ``constraints`` stamps exactly this, read back after its
    write, and the exports quote exactly this. Computed from the ``relation``
    rows rather than read back from the meta stamp, so it cannot go stale -- a
    hand-added edge changes the answer immediately, and an export that quotes it
    is quoting what it shipped. ``None`` for a desktop with no constraint edges
    at all, which is what ``python -m tda.cli constraints`` is for.
    """
    return edge_digest(edges_from_db(db, desktop))
