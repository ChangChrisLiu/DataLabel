"""One derivation for the CLI and for the S1 Apply (:mod:`tda.core.graph_derive`).

The part that needed a ruling: a human's spec 7.3 decision is *about* a rule
edge, so it only means anything while the rules still derive that edge. When a
later S1 correction stops deriving it the decision is kept but **orphaned** --
inactive, counted apart, offered to the annotator -- and adopted back unchanged
if the triple comes round again. Accepting an edge must not freeze it against
every later correction of the instance table.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from graph_scenes import DESKTOP, bench_instances, good_sequence

from tda.core.db import Db
from tda.core.graph import Edge, edges_from_db, graph_version
from tda.core.graph_derive import (
    MANUAL,
    OVERRIDE,
    RULE,
    derive_edges,
    write_derivation,
)
from tda.core.graph_rules import active_edges, is_orphan
from tda.core.taxonomy import load_taxonomy

SCREW = "screw.motherboard.01"
FASTENED = ("fastened_by", "motherboard.01", SCREW)


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def instances():
    return bench_instances()


def triples(edges):
    return {(e.type, e.target, e.blocker) for e in edges}


def one(edges, triple) -> Edge:
    return next(e for e in edges if (e.type, e.target, e.blocker) == triple)


def decided(triple, status: str, reason: str = "the rule's own reason") -> Edge:
    kind, target, blocker = triple
    return Edge(type=kind, target=target, blocker=blocker, reason=reason,
                source=OVERRIDE, status=status)


# --------------------------------------------------------------------------- #
# the rules
# --------------------------------------------------------------------------- #
def test_an_empty_desktop_gets_every_proposed_edge(instances, tax):
    out = derive_edges(instances, [], tax)
    assert FASTENED in triples(out.edges)
    assert {e.source for e in out.edges} == {RULE}
    assert out.stale == [] and out.orphaned == []


def test_a_rule_edge_the_instances_no_longer_imply_is_stale(instances, tax):
    stored = derive_edges(instances, [], tax).edges
    instances[SCREW].fastens = None

    out = derive_edges(instances, stored, tax)

    assert FASTENED in out.stale
    assert FASTENED not in triples(out.edges)


def test_a_draft_instance_derives_nothing(instances, tax):
    from tda.core.model import InstanceRec

    instances["ls:Screw#1"] = InstanceRec(key="ls:Screw#1", desktop=DESKTOP, cls="screw",
                                          fastens="motherboard.01")
    assert not [e for e in derive_edges(instances, [], tax).edges
                if "ls:" in e.blocker or "ls:" in e.target]


def test_a_manual_edge_is_kept_untouched(instances, tax):
    manual = Edge(type="blocked_by", target="psu.01", blocker="motherboard.01",
                  mode="cable_tension", reason="mine", source=MANUAL, status="accepted")
    out = derive_edges(instances, [manual], tax)
    assert one(out.edges, ("blocked_by", "psu.01", "motherboard.01")) == manual
    assert out.stale == []


def test_a_non_constraint_row_is_left_where_it_is(instances, tax):
    partner = Edge(type="partner_of", target="psu.01", blocker="motherboard.01",
                   source="labelstudio")
    out = derive_edges(instances, [partner], tax)
    assert out.other == [partner]
    assert partner not in out.edges
    assert out.counts()["other"] == 1


def test_a_kept_edge_stops_the_rule_edge_being_written_twice(instances, tax):
    manual = Edge(**{**vars(one(derive_edges(instances, [], tax).edges, FASTENED)),
                     "source": MANUAL, "status": "accepted"})
    out = derive_edges(instances, [manual], tax)
    assert [e for e in out.edges if (e.type, e.target, e.blocker) == FASTENED] == [manual]


# --------------------------------------------------------------------------- #
# decisions and orphans (I2)
# --------------------------------------------------------------------------- #
def test_a_decision_about_a_derived_edge_keeps_its_status(instances, tax):
    out = derive_edges(instances, [decided(FASTENED, "rejected")], tax)
    edge = one(out.edges, FASTENED)
    assert (edge.source, edge.status) == (OVERRIDE, "rejected")
    assert out.orphaned == [] and out.readopted == []
    assert edge not in out.active


def test_a_decision_whose_rule_edge_is_gone_is_orphaned(instances, tax):
    instances[SCREW].fastens = None

    out = derive_edges(instances, [decided(FASTENED, "accepted")], tax)

    edge = one(out.edges, FASTENED)
    assert edge.status == "accepted_orphan"
    assert is_orphan(edge)
    assert out.orphaned == [edge] and out.orphans == [edge]
    assert edge not in out.active, "an orphaned decision gates nothing"
    assert out.counts()["orphaned"] == 1
    assert edge.reason == "the rule's own reason", "the decision itself is kept"


def test_an_orphaned_rejection_is_orphaned_too(instances, tax):
    instances[SCREW].fastens = None
    out = derive_edges(instances, [decided(FASTENED, "rejected")], tax)
    assert one(out.edges, FASTENED).status == "rejected_orphan"


def test_an_orphan_is_adopted_back_when_the_rule_returns(instances, tax):
    """The human's decision is preserved across a correction and its undo."""
    out = derive_edges(instances, [decided(FASTENED, "accepted_orphan")], tax)
    edge = one(out.edges, FASTENED)
    assert edge.status == "accepted"
    assert out.readopted == [edge] and out.orphans == []
    assert edge in out.active


def test_a_rejected_orphan_that_is_derived_again_stays_rejected(instances, tax):
    out = derive_edges(instances, [decided(FASTENED, "rejected_orphan")], tax)
    edge = one(out.edges, FASTENED)
    assert edge.status == "rejected"
    assert edge not in out.active


def test_an_orphaned_decision_does_not_hold_back_the_rule_edge(instances, tax):
    """Its triple is no longer proposed, so there is nothing to hold back."""
    instances[SCREW].fastens = "cpu_cooler.fan.01"
    out = derive_edges(instances, [decided(FASTENED, "accepted")], tax)
    assert ("fastened_by", "cpu_cooler.fan.01", SCREW) in triples(out.active)
    assert one(out.edges, FASTENED).status == "accepted_orphan"


def test_a_manual_edge_is_never_orphaned(instances, tax):
    """It is nobody's decision about a rule: it stands on its own."""
    manual = Edge(type="fastened_by", target="cpu.01", blocker=SCREW, source=MANUAL,
                  status="accepted")
    instances[SCREW].fastens = None
    out = derive_edges(instances, [manual], tax)
    assert one(out.edges, ("fastened_by", "cpu.01", SCREW)).status == "accepted"


# --------------------------------------------------------------------------- #
# counts (I3)
# --------------------------------------------------------------------------- #
def test_the_counts_are_over_the_active_edges(instances, tax):
    stored = derive_edges(instances, [], tax).edges
    rejected = [decided(FASTENED, "rejected")]
    out = derive_edges(instances, rejected, tax)
    plain = derive_edges(instances, [], tax)
    assert out.counts()["active"] == plain.counts()["active"] - 1
    assert out.by_type["fastened_by"] == plain.by_type["fastened_by"] - 1
    assert out.counts()["rejected"] == 1
    assert out.counts()["decided"] == 1
    assert len(stored) == len(out.edges)


# --------------------------------------------------------------------------- #
# writing it (the diff, so ids survive)
# --------------------------------------------------------------------------- #
@pytest.fixture
def db(tmp_path: Path, instances) -> Db:
    conn = Db(str(tmp_path / "tda.sqlite"))
    conn.upsert_desktop(DESKTOP, {"brand": "Bench"})
    for rec in instances.values():
        conn.upsert_instance(rec)
    conn.replace_steps(DESKTOP, [], good_sequence())
    yield conn
    conn.close()


def write(db, instances, tax, extra=()):
    stored = edges_from_db(db, DESKTOP)
    out = derive_edges(instances, [*stored, *extra], tax)
    with db.transaction():
        write_derivation(db, DESKTOP, out, stored)
    return out


def test_writing_twice_changes_no_row(db, instances, tax):
    write(db, instances, tax)
    first = db.relations(DESKTOP)
    write(db, instances, tax)
    assert db.relations(DESKTOP) == first


def test_a_stale_rule_row_is_deleted(db, instances, tax):
    write(db, instances, tax)
    instances[SCREW].fastens = None
    write(db, instances, tax)
    assert FASTENED not in triples(edges_from_db(db, DESKTOP))


def test_a_manual_row_keeps_its_id_across_a_derivation(db, instances, tax):
    manual = Edge(type="blocked_by", target="psu.01", blocker="motherboard.01",
                  mode="cable_tension", reason="mine", source=MANUAL, status="accepted")
    write(db, instances, tax, extra=[manual])
    before = [r for r in db.relations(DESKTOP) if r["source"] == MANUAL]
    write(db, instances, tax)
    assert [r for r in db.relations(DESKTOP) if r["source"] == MANUAL] == before


def test_orphaning_a_decision_changes_the_graph_version(db, instances, tax):
    write(db, instances, tax, extra=[decided(FASTENED, "accepted")])
    before = graph_version(db, DESKTOP)
    instances[SCREW].fastens = None

    write(db, instances, tax)

    row = [r for r in db.relations(DESKTOP)
           if (r["type"], r["target"], r["blocker"]) == FASTENED]
    assert [r["status"] for r in row] == ["accepted_orphan"]
    assert graph_version(db, DESKTOP) != before
