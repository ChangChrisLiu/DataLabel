"""The constraint graph as an S1 edit session (stage S6, spec 7.3 item 3).

:class:`RelationsData` is to the ``relation`` table what
:class:`~tda.ui.steps_model.StepTableData` is to the step and instance tables:
the annotator's edits are **staged**, nothing reaches the database until
``Apply``, and ``Apply`` writes them inside the step table's one transaction --
so the steps, the actions, the instances, the pose re-cut and the edges either
all land or none of them do.

Three things this layer adds to the pure :mod:`tda.core.graph_edit`:

* the endpoints are checked against the **staged** instance table, so an edge
  may name an instance S1 has only just created (it is written first, in the
  same transaction);
* ``graph_version`` is re-stamped only when the edge set really changed
  (:func:`~tda.core.graph.edge_digest` ignores prose, so re-wording a reason is
  not a new graph), and it is read back through
  :func:`~tda.core.graph.graph_version` exactly as ``constraints`` does, so the
  meta stamp and the accessor are the same answer by construction;
* the write is recorded in the op log with the inverse patch.

The violations list is the spec 7.4 replay of the *staged* session -- staged
edges, staged actions -- so adding the right ``blocked_by`` for a failed
attempt makes the line disappear before ``Apply``, not after it.

No Qt here: :mod:`tda.ui.panels.relations` is the widget layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

from tda.core.db import Db
from tda.core.graph import (
    HARD_TYPES,
    Edge,
    constraint_edges,
    edge_digest,
    edges_from_db,
    edges_to_db,
    graph_version,
    is_provisional,
)
from tda.core.graph_edit import (
    GraphEditError,
    Violation,
    add_manual_edge,
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

    # -- loading ----------------------------------------------------------- #
    @classmethod
    def load(cls, db: Db, data: "StepTableData") -> "RelationsData":
        """Read the desktop's stored edges into a fresh session."""
        out = cls(data=data)
        out.reload(db)
        return out

    def reload(self, db: Db) -> None:
        """Throw away every staged edit and read the stored edges back."""
        self.stored = edges_from_db(db, self.data.desktop)
        self.edges = list(self.stored)

    # -- what the panel shows ---------------------------------------------- #
    @property
    def rows(self) -> list[Edge]:
        """Every edge of this desktop, grouped for display."""
        return sorted(self.edges, key=edge_sort_key)

    @property
    def dirty(self) -> bool:
        """Is anything staged that ``Apply`` would write?"""
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
        instances = {k: rec for k, rec in self.data.instances.items()
                     if not is_provisional(k)}
        return violations_of(instances, self.edges, self.data.actions, self.data.tax)

    def names(self, key: str) -> list[Edge]:
        """The **staged** edges naming ``key``, which a delete has to answer for."""
        staged = {self._triple(e) for e in self.edges} - {self._triple(e) for e in self.stored}
        return [e for e in self.edges
                if self._triple(e) in staged and key in (e.target, e.blocker)]

    # -- edits -------------------------------------------------------------- #
    def add(self, target: str, kind: str, blocker: str, *, necessity: str = "required",
            mode: Optional[str] = None, note: str = "") -> None:
        """Stage one hand-written edge; see :func:`~tda.core.graph_edit.add_manual_edge`."""
        self._stage(lambda: add_manual_edge(
            self.edges, target, kind, blocker, necessity=necessity, mode=mode,
            note=note, instances=self.data.instances,
        ))

    def remove(self, target: str, kind: str, blocker: str) -> None:
        """Stage the removal of a manual edge (a rule edge is refused)."""
        self._stage(lambda: remove_manual_edge(self.edges, target, kind, blocker))

    def decide(self, target: str, kind: str, blocker: str, decision: str) -> None:
        """Stage the spec 7.3 decision about a rule edge: accept / reject / clear."""
        self._stage(lambda: set_rule_decision(self.edges, target, kind, blocker, decision))

    def _stage(self, run) -> None:
        try:
            self.edges = run()
        except GraphEditError as error:  # the panel knows this one
            raise EditError(str(error)) from error

    # -- writing ------------------------------------------------------------ #
    def write(self, db: Db) -> dict:
        """Write the staged edits. **Call inside the caller's transaction.**

        Returns the op-log payload, or ``{}`` when nothing was staged -- an
        ``Apply`` that touched no edge must not re-stamp ``graph_version`` or
        leave a row in the op log.
        """
        before = self._by_triple(self.stored)
        after = self._by_triple(self.edges)
        dropped = [t for t in before if t not in after]
        written = [e for t, e in after.items() if before.get(t) != e]
        if not dropped and not written:
            return {}

        for triple in dropped:
            db.delete_relation(self.data.desktop, *triple)
        edges_to_db(db, self.data.desktop, written)

        payload = {
            "added": sum(1 for t, e in after.items() if t not in before),
            "changed": sum(1 for t, e in after.items()
                           if t in before and before[t] != e),
            "removed": len(dropped),
            "version": None,
        }
        if edge_digest(constraint_edges(list(after.values()))) != \
                edge_digest(constraint_edges(list(before.values()))):
            from tda.pipeline import merge_desktop_meta   # late: tda.pipeline is heavy

            payload["version"] = graph_version(db, self.data.desktop)
            merge_desktop_meta(db, self.data.desktop, {
                "graph_version": payload["version"],
                "graph_edges": len(constraint_edges(list(after.values()))),
            })
        db.log_op(
            self.data.desktop, OP_VIEW, OP_KIND, payload,
            {"restore": [asdict(before[t]) for t in dropped],
             "drop": [list(t) for t, e in after.items() if t not in before]},
            self.annotator,
        )
        return payload

    def committed(self) -> None:
        """The transaction went through: what was staged is now what is stored."""
        self.stored = list(self.edges)

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _triple(edge: Edge) -> tuple[str, str, str]:
        return (edge.type, edge.target, edge.blocker)

    @classmethod
    def _by_triple(cls, edges: list[Edge]) -> dict[tuple[str, str, str], Edge]:
        return {cls._triple(e): e for e in edges}
