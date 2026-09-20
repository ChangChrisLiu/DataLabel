"""A cycle is a deadlock of *actions*, not of instances (round 2, I-1).

``find_cycles`` asked whether the ``target -> blocker`` graph of instances had
a loop, whatever each edge gated. Since an edge gates only some verbs -- and a
``blocked_by`` only the verbs of its mode -- that both refused legitimate edges
and reported loops the planner walks straight through. The bracket that sits on
a screw head is the case this tab exists for:

* ``fastened_by(bracket, screw)`` gates remove / displace / open on the bracket;
* ``blocked_by(screw, bracket, physical_path)`` gates remove / displace on the
  screw;
* ``unscrew`` is gated by neither, so the screw comes out and the bracket
  follows. No deadlock.

:func:`tda.core.graph_plan.find_deadlocks` is the one definition, and this file
pins it against the planner itself: empty deadlocks if and only if every part
can still be planned out.
"""
from __future__ import annotations

import random

import pytest
from graph_scenes import DESKTOP, bench_instances

from tda.core.graph import Edge, propose_edges, remaining_plan
from tda.core.graph_plan import find_deadlocks
from tda.core.model import InstanceRec
from tda.core.states import initial_state
from tda.core.taxonomy import load_taxonomy

BRACKET, SCREW = "cooler_bracket.01", "screw.cooler_bracket.01"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def bench():
    return bench_instances()


def block(target: str, blocker: str, mode: str) -> Edge:
    return Edge(type="blocked_by", target=target, blocker=blocker, mode=mode,
                source="manual", status="accepted")


def deadlocks(edges, instances, tax):
    return find_deadlocks(list(edges), instances, tax)


# --------------------------------------------------------------------------- #
# the D36 pair
# --------------------------------------------------------------------------- #
@pytest.fixture
def bracket(bench) -> dict:
    """A bracket held by one screw, in the bench desktop."""
    bench[BRACKET] = InstanceRec(key=BRACKET, desktop=DESKTOP, cls="cooler_bracket",
                                 mounted_on="motherboard.01")
    bench[SCREW] = InstanceRec(key=SCREW, desktop=DESKTOP, cls="screw",
                               attrs={"role": "cpu_cooler", "captive": False},
                               fastens=BRACKET)
    return bench


def test_the_bracket_on_the_screw_head_is_not_a_deadlock(bracket, tax):
    """`fastened_by` does not gate `unscrew`, so the pair unwinds itself."""
    edges = [*propose_edges(bracket, tax), block(SCREW, BRACKET, "physical_path")]
    assert deadlocks(edges, bracket, tax) == []
    for key in (BRACKET, SCREW):
        assert remaining_plan(bracket, edges, initial_state(bracket, tax), key, tax) \
            is not None


def test_the_same_pair_with_tool_access_is_a_deadlock(bracket, tax):
    """`tool_access` gates every verb, so the screw cannot be reached at all."""
    edges = [*propose_edges(bracket, tax), block(SCREW, BRACKET, "tool_access")]
    found = deadlocks(edges, bracket, tax)
    assert len(found) == 1
    chain = found[0].chain()
    assert f"unscrew {SCREW}" in chain and BRACKET in chain
    assert chain.count("->") >= 2
    assert {e.type for e in found[0].edges} == {"fastened_by", "blocked_by"}
    for key in (BRACKET, SCREW):
        assert remaining_plan(bracket, edges, initial_state(bracket, tax), key, tax) \
            is None


def test_a_cable_tension_pair_is_not_a_deadlock(bench, tax):
    """Each side may still be swung aside, so both can be planned out."""
    edges = [*propose_edges(bench, tax),
             block("psu.01", "storage_drive.hdd.01", "cable_tension"),
             block("storage_drive.hdd.01", "psu.01", "cable_tension")]
    assert deadlocks(edges, bench, tax) == []


def test_two_physical_path_edges_that_really_do_block_each_other(bench, tax):
    """Neither part can move at all until the other has: that is a deadlock."""
    edges = [*propose_edges(bench, tax),
             block("psu.01", "storage_drive.hdd.01", "physical_path"),
             block("storage_drive.hdd.01", "psu.01", "physical_path")]
    found = deadlocks(edges, bench, tax)
    assert len(found) == 1
    assert "psu.01" in found[0].chain() and "storage_drive.hdd.01" in found[0].chain()


def test_a_three_action_loop_through_a_rule_edge_and_a_decision(bracket, tax):
    """A rule edge, an accepted override and a staged manual edge in one loop."""
    rules = propose_edges(bracket, tax)
    accepted = [Edge(**{**vars(e), "source": "override", "status": "accepted"})
                if (e.type, e.target) == ("covered_by", "cpu.01") else e
                for e in rules]
    edges = [*accepted,
             block("cpu_cooler.fan.01", BRACKET, "tool_access"),
             block(BRACKET, "cpu.01", "tool_access")]
    found = deadlocks(edges, bracket, tax)
    assert found, "cpu -> cooler -> bracket -> cpu is a loop of actions"
    sources = {e.source for e in found[0].edges}
    assert {"manual", "override"} <= sources or {"manual", "rule"} <= sources


def test_a_rejected_edge_cannot_deadlock(bracket, tax):
    edges = [*propose_edges(bracket, tax),
             Edge(type="blocked_by", target=SCREW, blocker=BRACKET, mode="tool_access",
                  source="manual", status="rejected")]
    assert deadlocks(edges, bracket, tax) == []


def test_an_orphaned_decision_cannot_deadlock(bracket, tax):
    edges = [*propose_edges(bracket, tax),
             Edge(type="blocked_by", target=SCREW, blocker=BRACKET, mode="tool_access",
                  source="override", status="accepted_orphan")]
    assert deadlocks(edges, bracket, tax) == []


def test_a_recommended_edge_is_not_binding(bracket, tax):
    edges = [*propose_edges(bracket, tax),
             Edge(type="blocked_by", target=SCREW, blocker=BRACKET, mode="tool_access",
                  necessity="recommended", source="manual", status="accepted")]
    assert deadlocks(edges, bracket, tax) == []


def test_the_bench_desktop_holds_no_deadlock(bench, tax):
    assert deadlocks(propose_edges(bench, tax), bench, tax) == []


def test_a_self_edge_would_be_a_deadlock_of_one_action(bench, tax):
    """The editor refuses self edges; a database could still hold one."""
    edges = [block("psu.01", "psu.01", "tool_access")]
    found = deadlocks(edges, bench, tax)
    assert len(found) == 1 and len(found[0].actions) == 1


def test_the_label_names_the_edges_that_hold_the_loop(bracket, tax):
    edges = [*propose_edges(bracket, tax), block(SCREW, BRACKET, "tool_access")]
    label = deadlocks(edges, bracket, tax)[0].label()
    assert "fastened_by(" in label and "blocked_by(" in label


# --------------------------------------------------------------------------- #
# the property: no deadlock <=> the planner can empty the machine
# --------------------------------------------------------------------------- #
#: Random graphs over this many parts; every part is removable and every
#: blocker clearable, so the only way to fail is a loop.
PARTS = 5
GRAPHS = 200
KINDS = (("fastened_by", None), ("covered_by", None), ("connected_to", None),
         ("blocked_by", "physical_path"), ("blocked_by", "tool_access"),
         ("blocked_by", "cable_tension"))


def random_scene(rng: random.Random) -> tuple[dict, list[Edge]]:
    """``PARTS`` expansion cards and a handful of random edges between them."""
    keys = [f"expansion_card.{i:02d}" for i in range(1, PARTS + 1)]
    instances = {k: InstanceRec(key=k, desktop=DESKTOP, cls="expansion_card",
                                mounted_on="motherboard.01") for k in keys}
    edges: list[Edge] = []
    seen = set()
    for _ in range(rng.randint(1, PARTS * 2)):
        target, blocker = rng.sample(keys, 2)
        kind, mode = rng.choice(KINDS)
        if (kind, target, blocker) in seen:
            continue
        seen.add((kind, target, blocker))
        edges.append(Edge(type=kind, target=target, blocker=blocker, mode=mode,
                          source="manual", status="accepted"))
    return instances, edges


def test_no_deadlock_if_and_only_if_every_part_can_be_planned_out(tax):
    rng = random.Random(20260920)
    both = {"deadlock+plan": 0, "deadlock+noplan": 0, "clean+plan": 0, "clean+noplan": 0}
    for _ in range(GRAPHS):
        instances, edges = random_scene(rng)
        state = initial_state(instances, tax)
        found = find_deadlocks(edges, instances, tax)
        planned = all(remaining_plan(instances, edges, state, key, tax) is not None
                      for key in instances)
        both[("deadlock" if found else "clean") + ("+plan" if planned else "+noplan")] += 1
    assert both["deadlock+plan"] == 0, "a deadlock the planner walked through"
    assert both["clean+noplan"] == 0, "a plan that failed with no deadlock to blame"
    assert both["deadlock+noplan"] > 0 and both["clean+plan"] > 0, both
