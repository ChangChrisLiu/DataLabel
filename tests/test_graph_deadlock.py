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

from tda.core.graph import Edge, legal_actions, propose_edges, remaining_plan
from tda.core.graph_plan import find_deadlocks, plan_removal
from tda.core.graph_rules import REQUIRED_STATES, active_edges, verb_effect
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
    """A preference cannot deadlock anything -- and must not stop the planner.

    Round 3, C-1: the check ignored recommended edges (spec 7.1 calls them a
    preference) while the planner honoured them and failed on them, so a pair
    of contradictory preferences stored clean and left two nodes unplannable.
    """
    edges = [*propose_edges(bracket, tax),
             Edge(type="blocked_by", target=SCREW, blocker=BRACKET, mode="tool_access",
                  necessity="recommended", source="manual", status="accepted")]
    assert deadlocks(edges, bracket, tax) == []
    state = initial_state(bracket, tax)
    for key in (BRACKET, SCREW):
        assert remaining_plan(bracket, edges, state, key, tax) is not None


def test_two_contradictory_preferences_leave_a_relaxed_plan(bracket, tax):
    edges = [*propose_edges(bracket, tax),
             Edge(type="blocked_by", target=SCREW, blocker=BRACKET, mode="tool_access",
                  necessity="recommended", source="manual", status="accepted"),
             Edge(type="blocked_by", target=BRACKET, blocker=SCREW, mode="tool_access",
                  necessity="recommended", source="manual", status="accepted")]
    state = initial_state(bracket, tax)

    assert deadlocks(edges, bracket, tax) == [], "a preference is not a law"
    for key in (BRACKET, SCREW):
        plan = plan_removal(bracket, edges, state, key, tax)
        assert plan is not None and plan.relaxed
        assert {e.necessity for e in plan.dropped} == {"recommended"}


def test_a_preference_that_can_be_honoured_is_not_relaxed(bench, tax):
    edges = [*propose_edges(bench, tax),
             Edge(type="blocked_by", target="psu.01", blocker="storage_drive.hdd.01",
                  mode="physical_path", necessity="recommended", source="manual",
                  status="accepted")]
    plan = plan_removal(bench, edges, initial_state(bench, tax), "psu.01", tax)
    assert plan is not None and not plan.relaxed
    assert ("displace", "storage_drive.hdd.01") in plan.actions


def test_the_contradiction_can_still_be_found_when_asked_for(bracket, tax):
    """The panel says so as a note; nothing is refused (spec 7.1)."""
    edges = [*propose_edges(bracket, tax),
             Edge(type="blocked_by", target=SCREW, blocker=BRACKET, mode="tool_access",
                  necessity="recommended", source="manual", status="accepted"),
             Edge(type="blocked_by", target=BRACKET, blocker=SCREW, mode="tool_access",
                  necessity="recommended", source="manual", status="accepted")]
    soft = find_deadlocks(edges, bracket, tax, necessity="recommended")
    assert soft and any(e.necessity == "recommended" for e in soft[0].edges)


# --------------------------------------------------------------------------- #
# more than one way to clear a blocker (round 3, minor A)
# --------------------------------------------------------------------------- #
CLIP, CLIP_SCREW, PART = "cable_clip.01", "screw.cable_clip.01", "expansion_card.01"


@pytest.fixture
def clipped(bench) -> dict:
    """A part held by a clip, the clip held by a screw, the screw behind the part."""
    bench[CLIP] = InstanceRec(key=CLIP, desktop=DESKTOP, cls="cable_clip")
    bench[CLIP_SCREW] = InstanceRec(key=CLIP_SCREW, desktop=DESKTOP, cls="screw",
                                    attrs={"role": "other", "captive": False})
    bench[PART] = InstanceRec(key=PART, desktop=DESKTOP, cls="expansion_card",
                              mounted_on="motherboard.01")
    return bench


def clip_scene() -> list[Edge]:
    """The reviewer's scene: `open` the clip deadlocks, `release` does not."""
    return [
        Edge(type="locked_by", target=PART, blocker=CLIP, source="manual",
             status="accepted"),
        Edge(type="fastened_by", target=CLIP, blocker=CLIP_SCREW, source="manual",
             status="accepted"),
        Edge(type="blocked_by", target=CLIP_SCREW, blocker=PART, mode="tool_access",
             source="manual", status="accepted"),
    ]


def test_a_second_way_to_clear_the_blocker_is_not_a_deadlock(clipped, tax):
    """`open` waits on the clip's screw, `release` does not: the scene unwinds."""
    edges = clip_scene()
    assert deadlocks(edges, clipped, tax) == []
    plan = remaining_plan(clipped, edges, initial_state(clipped, tax), PART, tax)
    assert plan is not None
    assert ("release", CLIP) in plan, plan
    assert ("open", CLIP) not in plan


def test_it_is_a_deadlock_once_every_alternative_is_gated_too(clipped, tax):
    """Cover the clip as well and `release` waits on the part like `open` does."""
    edges = [*clip_scene(),
             Edge(type="covered_by", target=CLIP, blocker=PART, source="manual",
                  status="accepted")]
    found = deadlocks(edges, clipped, tax)
    assert found, "now nothing clears the clip"
    assert remaining_plan(clipped, edges, initial_state(clipped, tax), PART, tax) is None


def test_a_dead_end_is_not_reported_as_a_deadlock(bench, tax):
    """Nothing can move the chassis, so the edge is a modelling gap, not a loop."""
    edges = [Edge(type="blocked_by", target="psu.01", blocker="chassis",
                  mode="physical_path", source="manual", status="accepted")]
    assert deadlocks(edges, bench, tax) == []
    assert remaining_plan(bench, edges, initial_state(bench, tax), "psu.01", tax) is None


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
# the property: a deadlock is exactly what stops the planner
# --------------------------------------------------------------------------- #
#: Random graphs over this many parts, plus the chassis -- which nothing can
#: move, so an edge pointing at it is a **dead end**: the planner has no answer
#: and there is no loop to name. That is a legitimate "clean, no plan" and the
#: bucket below asserts it happens, so that nobody later "fixes" a dead end
#: into a deadlock and makes the panel refuse a modelling gap.
PARTS = 5
GRAPHS = 200
KINDS = (("fastened_by", None), ("covered_by", None), ("connected_to", None),
         ("blocked_by", "physical_path"), ("blocked_by", "tool_access"),
         ("blocked_by", "cable_tension"))
#: Drawn per edge, so the invariants are asserted over preferences and over
#: rows a human has taken out of the graph as well.
NECESSITIES = ("required", "required", "required", "recommended")
STATUSES = ("accepted", "accepted", "accepted", "proposed", "rejected",
            "accepted_orphan")
DEAD_END = "chassis"


def random_scene(rng: random.Random) -> tuple[dict, list[Edge]]:
    """``PARTS`` expansion cards, the chassis, and random edges between them."""
    keys = [f"expansion_card.{i:02d}" for i in range(1, PARTS + 1)]
    instances = {k: InstanceRec(key=k, desktop=DESKTOP, cls="expansion_card",
                                mounted_on="motherboard.01") for k in keys}
    instances[DEAD_END] = InstanceRec(key=DEAD_END, desktop=DESKTOP, cls="chassis")
    edges: list[Edge] = []
    seen = set()
    for _ in range(rng.randint(1, PARTS * 2)):
        target = rng.choice(keys)
        blocker = rng.choice([*keys, DEAD_END])   # the chassis blocks, never blocked
        kind, mode = rng.choice(KINDS)
        if target == blocker or (kind, target, blocker) in seen:
            continue
        seen.add((kind, target, blocker))
        edges.append(Edge(type=kind, target=target, blocker=blocker, mode=mode,
                          necessity=rng.choice(NECESSITIES),
                          source="manual", status=rng.choice(STATUSES)))
    return instances, edges


def has_dead_end(instances, edges, tax) -> bool:
    """Is some binding edge's blocker unable to reach a state it accepts?

    Computed here, independently of the module under test, so "clean but no
    plan" has to be explained by the scene rather than by the code.
    """
    state = initial_state(instances, tax)
    for edge in active_edges(edges):
        if edge.necessity == "recommended":
            continue
        wanted = REQUIRED_STATES.get(edge.type, frozenset())
        rec = instances.get(edge.blocker)
        current = state[edge.blocker].state if rec is not None else None
        if rec is None or current in wanted or current == "removed":
            continue
        if not any(verb_effect(tax, rec.cls, rec.attrs, verb) in wanted
                   for verb in tax.verbs):
            return True
    return False


def replays_legally(instances, edges, plan, tax) -> bool:
    """Every action of the plan is legal under the **required** edges when it runs."""
    from dataclasses import replace

    required = [e for e in edges if e.necessity != "recommended"]
    state = dict(initial_state(instances, tax))
    for verb, target in plan:
        if (verb, target) not in legal_actions(instances, required, state, tax,
                                               strict=False):
            return False
        rec = instances[target]
        new = verb_effect(tax, rec.cls, rec.attrs, verb)
        state[target] = replace(state[target], state=new)
    return True


def test_no_deadlock_if_and_only_if_every_part_can_be_planned_out(tax):
    rng = random.Random(20260920)
    both = {"deadlock+plan": 0, "deadlock+noplan": 0, "clean+plan": 0, "clean+noplan": 0}
    relaxed = 0
    for _ in range(GRAPHS):
        instances, edges = random_scene(rng)
        state = initial_state(instances, tax)
        required = [e for e in edges if e.necessity != "recommended"]
        found = find_deadlocks(edges, instances, tax)
        plans = {key: plan_removal(instances, edges, state, key, tax)
                 for key in instances if key != DEAD_END}
        planned = all(plan is not None for plan in plans.values())
        for key, plan in plans.items():
            # a preference can change the order of a plan, never its existence
            assert (plan is not None) == (
                remaining_plan(instances, required, state, key, tax) is not None), key
            if plan is None:
                continue
            relaxed += bool(plan.relaxed)
            assert replays_legally(instances, edges, plan.actions, tax), (key, plan)
        both[("deadlock" if found else "clean") + ("+plan" if planned else "+noplan")] += 1
        if not found and not planned:
            assert has_dead_end(instances, edges, tax), "clean, no plan, and no dead end"
    assert both["deadlock+plan"] == 0, "a deadlock the planner walked through"
    assert both["deadlock+noplan"] > 0 and both["clean+plan"] > 0, both
    assert both["clean+noplan"] > 0, "the dead-end bucket must not be empty"
    assert relaxed > 0, "no scene needed a preference dropped"
