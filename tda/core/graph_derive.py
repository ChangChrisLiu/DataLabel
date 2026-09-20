"""Deriving one desktop's constraint graph -- the one derivation (spec 7.3).

``python -m tda.cli constraints`` used to own this logic, which meant the S1
``Apply`` could change ``screw.fastens`` and leave the stored graph, the
Relations tab and its spec 7.4 replay describing the *old* instance table until
somebody remembered to run the command. Both callers now come here:

* :func:`derive_edges` is pure -- instances and the desktop's current rows in,
  the rows it should hold afterwards out;
* :func:`write_derivation` brings the table to that state inside the caller's
  transaction.

What it does, per desktop:

1. the spec 7.3 rules propose their edges from the **settled** instances (a
   ``ls:*`` draft carries no relations);
2. rows the rules do not own -- ``manual``, ``override``, the Label Studio
   imports -- are kept exactly as they are, and a proposed edge whose triple one
   of them already holds is not written a second time (one row per triple);
3. a rule row the rules no longer propose is **stale** and goes;
4. an ``override`` row -- a human's spec 7.3 decision *about* a rule edge -- is
   only meaningful while that rule edge is still derived. When it is not, the
   row is kept but **orphaned** (``accepted_orphan`` / ``rejected_orphan``):
   inactive everywhere (:func:`~tda.core.graph_rules.active_edges`), counted and
   listed separately, and offered to the annotator as "keep as a manual edge" or
   "clear". If a later derivation proposes the triple again the decision is
   adopted back, unchanged -- accepting an edge does not freeze it against every
   later correction of the instance table, and rejecting one is not forgotten.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from tda.core.graph import Edge, constraint_edges, edges_to_db, propose_edges
from tda.core.graph_rules import (
    HARD_TYPES,
    ORPHAN_OF,
    SETTLED_OF,
    active_edges,
    is_orphan,
    is_provisional,
)
from tda.core.model import InstanceRec
from tda.core.taxonomy import Taxonomy

__all__ = [
    "MANUAL",
    "OVERRIDE",
    "RULE",
    "Derivation",
    "derive_edges",
    "settled_instances",
    "write_derivation",
]

#: ``relation.source`` values this module distinguishes. Everything else (the
#: Label Studio import) is "imported": kept, counted, never touched.
RULE = "rule"
OVERRIDE = "override"
MANUAL = "manual"

Triple = tuple[str, str, str]


def _triple(edge: Edge) -> Triple:
    return (edge.type, edge.target, edge.blocker)


def settled_instances(instances: dict[str, InstanceRec]) -> dict[str, InstanceRec]:
    """The desktop's instances without the Label Studio drafts (spec 3.2)."""
    return {k: rec for k, rec in instances.items() if not is_provisional(k)}


@dataclass(frozen=True)
class Derivation:
    """What one desktop's ``relation`` rows should be, and what changed."""

    #: Every hard-constraint row of the desktop afterwards (rule + kept).
    edges: list[Edge] = field(default_factory=list)
    #: Rows of a non-constraint type, or naming a draft: untouched, not counted
    #: as constraints (the Label Studio ``partner_of`` rows and their kin).
    other: list[Edge] = field(default_factory=list)
    #: Rule rows the rules no longer propose; :func:`write_derivation` drops them.
    stale: list[Triple] = field(default_factory=list)
    #: Decisions this derivation orphaned, and decisions it adopted back.
    orphaned: list[Edge] = field(default_factory=list)
    readopted: list[Edge] = field(default_factory=list)

    @property
    def active(self) -> list[Edge]:
        """The edges that actually gate something (no rejected, no orphans)."""
        return active_edges(self.edges)

    @property
    def by_type(self) -> dict[str, int]:
        """Active edges per spec 7.2 type -- what "the graph has N edges" means."""
        counts = {t: 0 for t in HARD_TYPES}
        for edge in self.active:
            counts[edge.type] = counts.get(edge.type, 0) + 1
        return {t: n for t, n in counts.items() if t in HARD_TYPES}

    @property
    def orphans(self) -> list[Edge]:
        """Every orphaned decision the desktop holds, not only the new ones."""
        return [e for e in self.edges if is_orphan(e)]

    @property
    def rejected(self) -> list[Edge]:
        """Rule edges a human rejected (and that are still derived)."""
        return [e for e in self.edges
                if e.status == "rejected" and e.source == OVERRIDE]

    def of_source(self, source: str) -> list[Edge]:
        return [e for e in self.edges if e.source == source]

    @property
    def imported(self) -> list[Edge]:
        """Hard edges from somewhere else entirely (the Label Studio import)."""
        return [e for e in self.edges if e.source not in (RULE, OVERRIDE, MANUAL)]

    def counts(self) -> dict[str, int]:
        """The numbers the ``constraints`` report and the op log quote."""
        return {
            "active": len(self.active),
            "rule": len(self.of_source(RULE)),
            "decided": len(self.of_source(OVERRIDE)),
            "manual": len(self.of_source(MANUAL)),
            "imported": len(self.imported),
            "rejected": len(self.rejected),
            "orphaned": len(self.orphans),
            "other": len(self.other),
            "stale": len(self.stale),
        }


def _decided_status(edge: Edge, derived: bool) -> str:
    """The status this decision should carry, given whether its rule still exists."""
    if derived:
        return SETTLED_OF.get(edge.status, edge.status)
    return ORPHAN_OF.get(edge.status, edge.status)


def derive_edges(
    instances: dict[str, InstanceRec],
    existing: Iterable[Edge],
    tax: Taxonomy,
) -> Derivation:
    """The rows one desktop should hold, from its instances and its current rows.

    Pure. ``instances`` may contain drafts (they are dropped here, once);
    ``existing`` is everything :func:`~tda.core.graph.edges_from_db` returned.
    """
    settled = settled_instances(instances)
    proposed = [replace(e, source=RULE) for e in propose_edges(settled, tax)]
    proposed_triples = {_triple(p) for p in proposed}

    existing = list(existing)
    keep = [e for e in existing if e.source != RULE]
    hard = constraint_edges(keep)
    hard_ids = {id(e) for e in hard}
    other = [e for e in keep if id(e) not in hard_ids]

    protected: dict[Triple, Edge] = {}
    orphaned: list[Edge] = []
    readopted: list[Edge] = []
    for edge in hard:
        if edge.source == OVERRIDE:
            wanted = _decided_status(edge, _triple(edge) in proposed_triples)
            if wanted != edge.status:
                edge = replace(edge, status=wanted)
                (orphaned if is_orphan(edge) else readopted).append(edge)
        protected[_triple(edge)] = edge

    fresh = [p for p in proposed if _triple(p) not in protected]
    stale = [_triple(e) for e in existing
             if e.source == RULE and _triple(e) not in proposed_triples]
    return Derivation(
        edges=fresh + list(protected.values()),
        other=other,
        stale=stale,
        orphaned=orphaned,
        readopted=readopted,
    )


def write_derivation(
    db, desktop: int, derivation: Derivation, stored: Optional[Iterable[Edge]] = None
) -> dict:
    """Bring the desktop's rows to ``derivation``. **Inside a transaction.**

    ``stored`` is what the table holds now (default: read it back). Everything
    is a diff against it, so a row whose content did not change is not rewritten
    and keeps its id -- which is what "a manual edge survives a re-run byte for
    byte" means -- and a row that vanished from the derivation (a stale rule
    edge, an edge the annotator removed) is deleted.
    """
    from tda.core.graph import edges_from_db

    before = {_triple(e): e for e in (
        list(stored) if stored is not None else edges_from_db(db, desktop))}
    after = {_triple(e): e for e in [*derivation.edges, *derivation.other]}
    dropped = [t for t in before if t not in after]
    written = [e for t, e in after.items() if before.get(t) != e]
    for triple in dropped:
        db.delete_relation(desktop, *triple)
    edges_to_db(db, desktop, written)
    return {"written": len(written), "dropped": len(dropped)}
