"""The constraint graph as an S1 edit session (stage S6, spec 7.3 item 3).

:class:`RelationsData` is to the ``relation`` table what
:class:`~tda.ui.steps_model.StepTableData` is to the step and instance tables:
the annotator's edits are **staged**, nothing reaches the database until
``Apply``, and ``Apply`` writes them inside the step table's one transaction --
so the steps, the actions, the instances, the pose re-cut and the edges either
all land or none of them do.

Four things this layer adds to the pure :mod:`tda.core.graph_edit`:

* the endpoints are checked against the **staged** instance table, so an edge
  may name an instance S1 has only just created (it is written first, in the
  same transaction);
* the rule edges are re-derived from the staged instances by
  :func:`~tda.core.graph_derive.derive_edges` -- the same function the
  ``constraints`` command runs. Without it the tab described the instance table
  the annotator had just corrected: clearing a ``screw.fastens`` left its
  ``fastened_by`` edge in the graph, and in the spec 7.4 replay, until somebody
  remembered the command line. The staged view previews the derivation and the
  ``Apply`` writes it, so what the annotator checked is what is stored;
* ``graph_version``, ``graph_edges`` and ``graph_cycles`` are re-stamped when
  the edge set really changed (:func:`~tda.core.graph.edge_digest` ignores
  prose, so re-wording a reason is not a new graph), the version read back
  through :func:`~tda.core.graph.graph_version` exactly as ``constraints`` does,
  so the meta stamp and the accessor are the same answer by construction;
* the write is recorded in the op log with the inverse patch.

The violations list is the spec 7.4 replay of the *staged* session -- derived
edges, staged edges, staged actions -- so adding the right ``blocked_by`` for a
failed attempt makes the line disappear before ``Apply``, not after it. The
cycle list is :func:`~tda.core.graph.find_cycles` over the same staged graph:
the panel makes a cycle impossible to create, but a database written elsewhere
can hold one, and a graph with a loop is one the planner cannot answer for.

No Qt here: :mod:`tda.ui.panels.relations` is the widget layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

from tda.core.db import Db
from tda.core.graph import (
    HARD_TYPES,
    Edge,
    edge_digest,
    edges_from_db,
    find_dead_ends,
    find_deadlocks,
    graph_version,
    is_provisional,
)
from tda.core.graph_derive import Derivation, derive_edges, settled_instances, write_derivation
from tda.core.graph_edit import (
    GraphEditError,
    Violation,
    add_manual_edge,
    adopt_orphan,
    drop_orphan,
    remove_manual_edge,
    set_rule_decision,
    violations_of,
)
from tda.core.graph_rules import CABLE_PREFIX
from tda.ui.steps_values import EditError

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, fine for typing
    from tda.ui.steps_model import StepTableData

__all__ = ["ANNOTATOR", "OP_KIND", "OP_VIEW", "RelationsData", "edge_sort_key"]

#: ``op_log.kind`` of the row an ``Apply`` with staged edges writes.
OP_KIND = "relations"
#: ``op_log`` is scoped per ``(desktop, view)``; an edge belongs to no view.
OP_VIEW = "-"
#: ``op_log.annotator`` when the window did not say who is editing.
ANNOTATOR = "ui:relations"


def edge_sort_key(edge: Edge) -> tuple:
    """Group the table by type (spec 7.2 order), then by target and blocker."""
    order = HARD_TYPES.index(edge.type) if edge.type in HARD_TYPES else len(HARD_TYPES)
    return (order, edge.target, edge.blocker)


@dataclass
class RelationsData:
    """One desktop's constraint edges, staged for the step table's ``Apply``."""

    data: "StepTableData"
    stored: list[Edge] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    #: Who the op log says did this; the window sets it from its annotator.
    annotator: str = ANNOTATOR
    #: The last previewed derivation, dropped whenever anything could change it.
    _view: Optional[Derivation] = None
    #: What the last :meth:`write` put in the table, for :meth:`committed`.
    _written: list[Edge] = field(default_factory=list)

    # -- loading ----------------------------------------------------------- #
    def reload(self, db: Db) -> None:
        """Throw away every staged edit and read the stored edges back."""
        self.stored = edges_from_db(db, self.data.desktop)
        self.edges = list(self.stored)
        self.invalidate()

    def invalidate(self) -> None:
        """Forget the previewed derivation (the instance table may have moved)."""
        self._view = None

    def view(self) -> Derivation:
        """What the rules make of the staged session right now.

        The same :func:`~tda.core.graph_derive.derive_edges` the ``constraints``
        command runs, over the **staged** instance table: the tab therefore
        shows the graph this ``Apply`` will write, not the one the last CLI run
        left behind.
        """
        if self._view is None:
            self._view = derive_edges(self.data.instances, self.edges, self.data.tax)
        return self._view

    # -- what the panel shows ---------------------------------------------- #
    @property
    def rows(self) -> list[Edge]:
        """Every edge of this desktop, re-derived and grouped for display."""
        view = self.view()
        return sorted([*view.edges, *view.other], key=edge_sort_key)

    @property
    def dirty(self) -> bool:
        """Is an edit of *this tab* staged? (An S1 edit is dirty on its own.)"""
        return self._by_triple(self.edges) != self._by_triple(self.stored)

    def instance_keys(self) -> list[str]:
        """The settled instances an edge may name (spec 3.2: no ``ls:*`` draft)."""
        return [k for k in self.data.instance_keys() if not is_provisional(k)]

    def blocker_keys(self) -> list[str]:
        """What may block: every settled instance, plus the cable nodes in use.

        A clipped cable is a spec 7.3 ``blocked_by`` reason and has no instance
        row of its own, so the virtual nodes the graph already mentions are
        offered next to the real parts rather than typed in from memory.
        """
        cables = {e.blocker for e in self.edges if e.blocker.startswith(CABLE_PREFIX)}
        cables |= {rec.cable for rec in self.data.instances.values() if rec.cable}
        return [*self.instance_keys(), *sorted(cables)]

    def violations(self) -> list[Violation]:
        """The spec 7.4 replay of the staged session (same as ``--validate``)."""
        return violations_of(settled_instances(self.data.instances), self.view().edges,
                             self.data.actions, self.data.tax)

    def cycles(self) -> list:
        """The spec 7.4 deadlock check over the staged **active** graph."""
        return find_deadlocks(self.view().edges, self.data.instances, self.data.tax)

    def dead_ends(self) -> list:
        """Instances that cannot be planned out, and the edge nothing can clear.

        Not a refusal: an edge no action satisfies is a modelling gap, and the
        way out may be to create the missing part rather than to change the
        edge. It must not be *silent* either -- one ``Add edge`` click can write
        one (round 4, I-1).
        """
        return find_dead_ends(self.view().edges, self.data.instances, self.data.tax)

    def soft_conflicts(self) -> list:
        """Recommended edges that contradict each other -- a note, not a refusal.

        Spec 7.1 calls ``recommended`` a preference: it cannot make a graph
        impossible, and the planner drops it rather than fail. Two preferences
        that point at each other are still worth saying out loud, because the
        order they ask for cannot be honoured.
        """
        strict = {d.actions for d in self.cycles()}
        return [d for d in find_deadlocks(self.view().edges, self.data.instances,
                                          self.data.tax, necessity="recommended")
                if d.actions not in strict]

    def names(self, key: str) -> list[Edge]:
        """The **staged** edges naming ``key``, which a delete has to answer for."""
        staged = {self._triple(e) for e in self.edges} - {self._triple(e) for e in self.stored}
        return [e for e in self.edges
                if self._triple(e) in staged and key in (e.target, e.blocker)]

    # -- edits -------------------------------------------------------------- #
    def add(self, target: str, kind: str, blocker: str, *, necessity: str = "required",
            mode: Optional[str] = None, note: str = "") -> None:
        """Stage one hand-written edge; see :func:`~tda.core.graph_edit.add_manual_edge`."""
        self._stage(lambda edges: add_manual_edge(
            edges, target, kind, blocker, necessity=necessity, mode=mode,
            note=note, instances=self.data.instances, tax=self.data.tax,
        ))

    def remove(self, target: str, kind: str, blocker: str) -> None:
        """Stage the removal of a manual edge (a rule edge is refused)."""
        self._stage(lambda edges: remove_manual_edge(edges, target, kind, blocker))

    def decide(self, target: str, kind: str, blocker: str, decision: str) -> None:
        """Stage the spec 7.3 decision about a rule edge: accept / reject / clear."""
        self._stage(lambda edges: set_rule_decision(
            edges, target, kind, blocker, decision,
            instances=self.data.instances, tax=self.data.tax))

    def adopt(self, target: str, kind: str, blocker: str, note: str = "") -> None:
        """Keep an orphaned decision as a manual edge of the annotator's own."""
        self._stage(lambda edges: adopt_orphan(
            edges, target, kind, blocker, note=note,
            instances=self.data.instances, tax=self.data.tax))

    def drop(self, target: str, kind: str, blocker: str) -> None:
        """Clear an orphaned decision: its rule edge is not derived any more."""
        self._stage(lambda edges: drop_orphan(edges, target, kind, blocker))

    def _stage(self, run) -> None:
        """Run one edit over the **previewed** graph and record its difference.

        The edit sees the derivation, not the raw stored rows: an orphan status,
        and a rule edge an S1 correction has just created, are part of what the
        annotator is looking at and of what the checks (duplicate, cycle) have
        to weigh. Only the difference is kept, so :attr:`dirty` still means
        "this tab has an unsaved edit" rather than "the rules moved".
        """
        before = self.view().edges
        try:
            after = run(before)
        except GraphEditError as error:  # the panel knows this one
            raise EditError(str(error)) from error
        was = self._by_triple(before)
        now = self._by_triple(after)
        staged = self._by_triple(self.edges)
        for triple in [t for t in staged if t in was and t not in now]:
            staged.pop(triple)
        for triple, edge in now.items():
            if was.get(triple) != edge:
                staged[triple] = edge
        self.edges = list(staged.values())
        self.invalidate()

    # -- writing ------------------------------------------------------------ #
    def write(self, db: Db) -> dict:
        """Derive and write this desktop's graph. **Inside the caller's transaction.**

        The rule edges are re-derived here, with the same function the
        ``constraints`` command uses, so an S1 correction of ``fastens`` /
        ``socket_host`` / ``of`` reaches the graph on the ``Apply`` that made
        it. Returns the op-log payload, or ``{}`` when nothing changed at all --
        an ``Apply`` that moved no edge must not re-stamp ``graph_version`` or
        leave a row in the op log.
        """
        derivation = derive_edges(self.data.instances, self.edges, self.data.tax)
        self._refuse_deadlock(derivation)
        rows = [*derivation.edges, *derivation.other]
        before = self._by_triple(self.stored)
        after = self._by_triple(rows)
        self._written = rows
        dropped = [t for t in before if t not in after]
        changed = [t for t, e in after.items() if t in before and before[t] != e]
        added = [t for t in after if t not in before]
        if not dropped and not changed and not added:
            return {}

        write_derivation(db, self.data.desktop, derivation, self.stored)
        payload = {
            "added": len(added),
            "changed": len(changed),
            "removed": len(dropped),
            "orphaned": len(derivation.orphaned),
            "readopted": len(derivation.readopted),
            "version": None,
        }
        if edge_digest(list(after.values())) != edge_digest(list(before.values())):
            from tda.pipeline import merge_desktop_meta   # late: tda.pipeline is heavy

            payload["version"] = graph_version(db, self.data.desktop)
            merge_desktop_meta(db, self.data.desktop, {
                "graph_version": payload["version"],
                "graph_edges": len(derivation.active),
                "graph_cycles": 0,   # refused above if there were any
            })
        db.log_op(
            self.data.desktop, OP_VIEW, OP_KIND, payload,
            {"restore": [asdict(before[t]) for t in dropped],
             "drop": [list(t) for t in added]},
            self.annotator,
        )
        return payload

    def _refuse_deadlock(self, derivation: Derivation) -> None:
        """An ``Apply`` may never store a deadlock (spec 7.4).

        Each edit is checked on its own, which is not enough: an S1 correction
        can re-activate a rule edge that deadlocks with a manual edge written
        while that rule edge was gone, and neither edit was wrong when it was
        made. The derivation is where the two meet, so this is where it is
        caught -- the whole ``Apply`` is refused, the transaction rolls back and
        the staged edits stay staged, because the annotator is one small change
        away from a graph that works.
        """
        found = find_deadlocks(derivation.edges, self.data.instances, self.data.tax)
        if not found:
            return
        named = "; ".join(d.label() for d in found)
        raise EditError(
            f"这次 Apply 会存下动作死锁，已全部回滚 / this Apply would store a deadlock "
            f"and was rolled back whole: {named}. 改掉环里的手动边，或拒绝环里的规则边，"
            f"再 Apply / remove or change the manual edge in the loop, or reject the "
            f"rule edge in it, then Apply again"
        )

    def committed(self) -> None:
        """The transaction went through: what was written is now what is stored."""
        self.stored = list(self._written)
        self.edges = list(self._written)
        self.invalidate()

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _triple(edge: Edge) -> tuple[str, str, str]:
        return (edge.type, edge.target, edge.blocker)

    @classmethod
    def _by_triple(cls, edges: list[Edge]) -> dict[tuple[str, str, str], Edge]:
        return {cls._triple(e): e for e in edges}
