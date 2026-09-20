"""Editing a desktop's constraint graph by hand (spec 7.3 item 3, stage S6).

``python -m tda.cli constraints`` derives the spec 7.3 edges from the instance
attributes and re-derives them on every run. That covers about 90 % of the
graph; the rest is a human's job, and so is disagreeing with a rule. This
module is the pure layer under the Relations tab -- a list of
:class:`~tda.core.graph_rules.Edge` in, a new list out, or
:class:`GraphEditError` with a reason a panel can show as it is.

Two provenances, two behaviours:

* a **manual** edge (:data:`MANUAL`) is the annotator's own assertion. The
  rules never derive it and never touch it: ``constraints`` keeps every row
  whose ``source`` is not ``rule`` exactly as it is, so a manual edge survives
  every re-run byte for byte.
* a **rule** edge (:data:`RULE`) is derived, so editing one by hand would be
  undone silently by the next run. The human's way to disagree is the spec 7.3
  decision -- ``accepted`` / ``rejected`` -- recorded by
  :func:`set_rule_decision` as an :data:`OVERRIDE` row carrying the same
  ``(type, target, blocker)`` and the rule's own reason. The re-run keeps that
  row (it is not a ``rule`` row any more) and skips re-deriving the triple, so
  the decision sticks; :func:`~tda.core.graph_rules.active_edges` drops a
  rejected edge from every replay, plan and legal-action set. Clearing a
  decision writes the rule's edge back, so nothing is lost either way.

Everything here is pure. The staging, the transaction and the ``graph_version``
re-stamp live in :mod:`tda.ui.steps_relations`, which is what the S1 ``Apply``
calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from tda.core.graph import (
    HARD_TYPES,
    NECESSITY_ORDER,
    Edge,
    active_edges,
    constraint_edges,
    find_cycles,
    is_provisional,
    validate_sequence,
)
from tda.core.graph_derive import MANUAL, OVERRIDE, RULE
from tda.core.graph_rules import BLOCKED_MODES, CABLE_PREFIX, is_orphan
from tda.core.model import ActionRec, InstanceRec

__all__ = [
    "DECISIONS",
    "MANUAL",
    "OVERRIDE",
    "RULE",
    "GraphEditError",
    "Violation",
    "add_manual_edge",
    "adopt_orphan",
    "drop_orphan",
    "editable",
    "is_orphan",
    "remove_manual_edge",
    "set_rule_decision",
    "violations",
    "violations_of",
]

#: What :func:`set_rule_decision` accepts. ``proposed`` clears the decision and
#: gives the rules their edge back.
DECISIONS = ("proposed", "accepted", "rejected")

#: The status a human-written edge starts in: they did not propose it, they
#: asserted it (spec 7.3).
MANUAL_STATUS = "accepted"

#: ``step 12: ...`` -- how :func:`~tda.core.graph.validate_sequence` starts
#: every line. Parsed rather than re-derived so the panel's list and
#: ``constraints --validate`` cannot drift apart: there is one replay.
_STEP = re.compile(r"^step (\d+):")

#: The tail :func:`~tda.core.graph.validate_sequence` puts on a failed attempt
#: nothing in the graph explains.
MISSING_EDGE = "missing edge?"


class GraphEditError(ValueError):
    """An edit the graph refuses. The message is bilingual and UI-ready."""


@dataclass(frozen=True)
class Violation:
    """One line of the spec 7.4 replay, with the step it belongs to.

    ``kind`` is ``"breach"`` (a successful action that broke a hard constraint
    -- most likely a gap in the log, S1 work) or ``"hint"`` (a failed attempt
    with nothing in its way -- the cue for a manual ``blocked_by``, S6 work).
    """

    step: int
    kind: str
    text: str


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _triple(edge: Edge) -> tuple[str, str, str]:
    return (edge.type, edge.target, edge.blocker)


def _find(edges: Iterable[Edge], target: str, kind: str, blocker: str) -> Optional[Edge]:
    wanted = (kind, target, blocker)
    for edge in edges:
        if _triple(edge) == wanted:
            return edge
    return None


def editable(edge: Edge) -> bool:
    """May this edge be removed by hand? Only a manual one may."""
    return edge.source == MANUAL


def _check_endpoint(
    instances: dict[str, InstanceRec], key: str, role: str, *, cable_ok: bool
) -> None:
    """Refuse an endpoint that is not a part of this desktop (spec 3.2)."""
    if key.startswith(CABLE_PREFIX):
        if not cable_ok:
            raise GraphEditError(
                f"线缆节点只能作为阻挡方 / a {CABLE_PREFIX}* node can only be the "
                f"blocker, not the {role} ({key})"
            )
        if not key[len(CABLE_PREFIX):].strip():
            raise GraphEditError(f"线缆节点没有名字 / the cable node {key!r} has no owner")
        return
    if is_provisional(key):
        raise GraphEditError(
            f"{key} 还只是 Label Studio 草稿 / {key} is a Label Studio draft, not a "
            f"part anybody has decided exists; settle it in S1 first"
        )
    if key not in instances:
        raise GraphEditError(
            f"本机没有实例 {key} / D has no instance {key!r} (deleted or never existed)"
        )


def _check_kind(kind: str) -> None:
    if kind not in HARD_TYPES:
        raise GraphEditError(
            f"{kind!r} 不是硬约束类型 / {kind!r} is not one of the spec 7.2 hard "
            f"constraint types ({', '.join(HARD_TYPES)})"
        )


def _check_mode(kind: str, mode: Optional[str]) -> None:
    if kind == "blocked_by":
        if not mode:
            raise GraphEditError(
                "blocked_by 必须写清阻挡方式 / a blocked_by edge needs a mode "
                f"({', '.join(BLOCKED_MODES)}) -- spec 7.1"
            )
        if mode not in BLOCKED_MODES:
            raise GraphEditError(
                f"{mode!r} 不是阻挡方式 / {mode!r} is not a blocked_by mode "
                f"({', '.join(BLOCKED_MODES)})"
            )
        return
    if mode:
        raise GraphEditError(
            f"只有 blocked_by 带 mode / only a blocked_by edge carries a mode, "
            f"{kind!r} does not"
        )


def _check_necessity(necessity: str) -> None:
    if necessity not in NECESSITY_ORDER:
        raise GraphEditError(
            f"{necessity!r} 不是约束强度 / {necessity!r} is not a necessity "
            f"({', '.join(NECESSITY_ORDER)}) -- spec 7.1"
        )


def _check_duplicate(edges: Iterable[Edge], target: str, kind: str, blocker: str) -> None:
    found = _find(edges, target, kind, blocker)
    if found is None:
        return
    raise GraphEditError(
        f"这条边已经存在（来源 {found.source}，状态 {found.status}） / "
        f"{found.label()} already exists as a {found.source} edge "
        f"({found.status}); one row per (type, target, blocker)"
    )


def _check_no_new_cycle(before: list[Edge], after: list[Edge]) -> None:
    """Spec 7.4: the graph is acyclic. Name the loop the edge would close."""
    was = {tuple(c) for c in find_cycles(constraint_edges(active_edges(before)))}
    now = [c for c in find_cycles(constraint_edges(active_edges(after)))
           if tuple(c) not in was]
    if now:
        named = "; ".join(" -> ".join(cycle) for cycle in now)
        raise GraphEditError(
            f"这条边会形成环 / that edge closes a cycle, which spec 7.4 forbids: {named}"
        )


# --------------------------------------------------------------------------- #
# the three edits
# --------------------------------------------------------------------------- #
def add_manual_edge(
    edges: list[Edge],
    src: str,
    kind: str,
    dst: str,
    *,
    necessity: str = "required",
    mode: Optional[str] = None,
    note: str = "",
    instances: dict[str, InstanceRec],
) -> list[Edge]:
    """``edges`` plus one hand-written edge ``kind(src, dst)``, or a refusal.

    ``src`` is the node being acted on and ``dst`` the one that must change
    state first, so the arguments read like :meth:`Edge.label`:
    ``blocked_by(optical_drive.01, psu.01)`` is "the PSU is in the drive's way".

    Refused, with a reason, when an endpoint is unknown, deleted or still a
    Label Studio draft (spec 3.2), when the two are the same node, when any
    edge with that triple already exists whatever its source or status, when
    ``kind`` is not one of the five hard types of spec 7.2, when a
    ``blocked_by`` carries no ``mode`` (or another type carries one), when
    ``necessity`` is not of spec 7.1, or when the edge would close a cycle
    (spec 7.4) -- the reason then names the cycle.

    The input list is never mutated. A ``cable:*`` node may be the blocker: a
    clipped cable is a spec 7.3 ``blocked_by`` reason and has no instance row.
    """
    _check_kind(kind)
    _check_mode(kind, mode)
    _check_necessity(necessity)
    if src == dst:
        raise GraphEditError(f"自环 / {src} cannot block itself")
    _check_endpoint(instances, src, "target", cable_ok=False)
    _check_endpoint(instances, dst, "blocker", cable_ok=True)
    _check_duplicate(edges, src, kind, dst)

    edge = Edge(type=kind, target=src, blocker=dst, necessity=necessity, mode=mode,
                reason=note or "", source=MANUAL, status=MANUAL_STATUS)
    out = [*edges, edge]
    _check_no_new_cycle(edges, out)
    return out


def remove_manual_edge(edges: list[Edge], src: str, kind: str, dst: str) -> list[Edge]:
    """``edges`` without the manual edge ``kind(src, dst)``, or a refusal.

    Only a manual edge can be removed. A rule edge is re-derived on the next
    ``constraints`` run, so deleting one would come back silently: the way to
    disagree with it is :func:`set_rule_decision`.
    """
    found = _find(edges, src, kind, dst)
    if found is None:
        raise GraphEditError(
            f"没有这条边 / D has no edge {kind}({src}, {dst})"
        )
    if not editable(found):
        raise GraphEditError(
            f"{found.source} 边不能手工删除 / a {found.source} edge cannot be deleted "
            f"by hand -- it is derived again on every constraints run; reject it "
            f"instead (spec 7.3)"
        )
    return [e for e in edges if e is not found]


def set_rule_decision(
    edges: list[Edge], src: str, kind: str, dst: str, decision: str
) -> list[Edge]:
    """Record the spec 7.3 decision about a rule edge: accept, reject, or clear.

    ``accepted`` and ``rejected`` replace the derived row with an
    :data:`OVERRIDE` row -- the same triple, the same reason, the human's
    status -- which ``constraints`` keeps instead of re-deriving, so the
    decision survives every re-run. ``proposed`` clears the decision and hands
    the triple back to the rules.

    Either of the two that make the edge **active again** is checked for
    acyclicity, because rejecting one edge and writing its reverse by hand is a
    legal pair of edits whose *undo* would close a loop -- and a stored cycle is
    a graph the planner cannot answer for and the next ``constraints`` run
    refuses. Rejecting is never refused: it can only take an edge out.

    A manual edge carries no decision (remove it instead), and neither does an
    orphaned one -- the rule edge it was about is not derived any more, so
    :func:`adopt_orphan` and :func:`drop_orphan` are what is left.
    """
    if decision not in DECISIONS:
        raise GraphEditError(
            f"{decision!r} 不是一个决定 / {decision!r} is not a decision "
            f"({', '.join(DECISIONS)}) -- spec 7.3"
        )
    found = _find(edges, src, kind, dst)
    if found is None:
        raise GraphEditError(f"没有这条边 / D has no edge {kind}({src}, {dst})")
    if is_orphan(found):
        raise GraphEditError(
            f"规则已经不再推导这条边 / the rules no longer derive {found.label()}, so "
            f"there is nothing left to accept or reject: keep it as a manual edge "
            f"or clear it"
        )
    if found.source not in (RULE, OVERRIDE):
        raise GraphEditError(
            f"{found.source} 边没有接受/拒绝 / a {found.source} edge is not accepted or "
            f"rejected, it is simply removed (spec 7.3)"
        )
    if decision == "proposed":
        new = replace(found, source=RULE, status="proposed")
    else:
        new = replace(found, source=OVERRIDE, status=decision)
    out = [new if e is found else e for e in edges]
    if decision != "rejected":
        _check_no_new_cycle(edges, out)
    return out


def adopt_orphan(
    edges: list[Edge],
    src: str,
    kind: str,
    dst: str,
    *,
    note: str = "",
    instances: dict[str, InstanceRec],
) -> list[Edge]:
    """Keep an orphaned decision as a manual edge of the annotator's own.

    The rules stopped deriving the edge this decision was about, so the decision
    has nothing left to be about (:mod:`tda.core.graph_derive`). If the
    annotator still believes the constraint, it becomes theirs: same triple,
    same mode and necessity, ``source='manual'``, and their own reason. It goes
    through every check a new edge goes through -- endpoints, cycles, the lot --
    because that is exactly what it now is.
    """
    found = _find(edges, src, kind, dst)
    if found is None or not is_orphan(found):
        raise GraphEditError(
            f"这不是一条已失去规则的决定 / {kind}({src}, {dst}) is not an orphaned "
            f"decision"
        )
    without = [e for e in edges if e is not found]
    return add_manual_edge(without, src, kind, dst, necessity=found.necessity,
                           mode=found.mode, note=note or found.reason,
                           instances=instances)


def drop_orphan(edges: list[Edge], src: str, kind: str, dst: str) -> list[Edge]:
    """Clear an orphaned decision: the edge is gone, so is the decision."""
    found = _find(edges, src, kind, dst)
    if found is None or not is_orphan(found):
        raise GraphEditError(
            f"这不是一条已失去规则的决定 / {kind}({src}, {dst}) is not an orphaned "
            f"decision"
        )
    return [e for e in edges if e is not found]


# --------------------------------------------------------------------------- #
# the violations list (spec 7.4)
# --------------------------------------------------------------------------- #
def violations_of(
    instances: dict[str, InstanceRec],
    edges: list[Edge],
    actions: list[ActionRec],
    tax,
) -> list[Violation]:
    """The spec 7.4 replay of ``actions`` against ``edges``, one record per line.

    Exactly what ``python -m tda.cli constraints --validate`` prints: the same
    :func:`~tda.core.graph.validate_sequence` over the same
    :func:`~tda.core.graph.constraint_edges` subset, so a violation the panel
    shows as gone is gone from the report too. The step is read back off the
    line rather than re-derived, because there is only one replay and only one
    wording.
    """
    out: list[Violation] = []
    for text in validate_sequence(instances, constraint_edges(edges), list(actions), tax):
        match = _STEP.match(text)
        out.append(Violation(
            step=int(match.group(1)) if match else 0,
            kind="hint" if text.endswith(MISSING_EDGE) else "breach",
            text=text,
        ))
    return out


def violations(db, desktop: int, tax=None) -> list[Violation]:
    """:func:`violations_of` for one stored desktop.

    The settled instances only (a ``ls:*`` draft carries no constraints), the
    desktop's edges and its actions -- the same three inputs
    :mod:`tda.cli_graph` replays.
    """
    from tda.core.graph import edges_from_db

    if tax is None:
        from tda.core.taxonomy import load_taxonomy

        tax = load_taxonomy()
    instances = {k: rec for k, rec in db.instances(desktop).items()
                 if not is_provisional(k)}
    return violations_of(instances, edges_from_db(db, desktop), db.actions(desktop), tax)
