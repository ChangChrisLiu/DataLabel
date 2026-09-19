"""Each hard-edge type gates its own verbs (spec 7.2, refined).

Every edge used to gate every gateable verb, which said a slim-case PSU could
not be *swung out of the way* until its whole harness was unplugged. That is
backwards: swinging it out is how you reach the plugs. Across the 66 sheets it
produced 34 "displace psu.01 violates connected_to(...)" lines, all of them
describing correct work.

One table in :mod:`tda.core.graph_rules` now says which verbs each type gates,
and the three readers -- :func:`applicable_preconditions` (hence
:func:`legal_actions` and :func:`validate_sequence`) and the planner -- all
consult it, so they cannot drift apart.
"""
from __future__ import annotations

import pytest
from graph_scenes import DESKTOP, act, bench_instances, state_after
from graph_scenes import triples as _triples

from tda.core.graph import (
    GATED_VERBS,
    applicable_preconditions,
    legal_actions,
    propose_edges,
    remaining_plan,
    validate_sequence,
)
from tda.core.graph_rules import GATES, gated_verbs
from tda.core.states import initial_state
from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def bench():
    return bench_instances()


@pytest.fixture
def edges(bench, tax):
    return propose_edges(bench, tax)


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def test_the_table_covers_every_hard_type():
    from tda.core.graph import HARD_TYPES

    assert set(GATES) == set(HARD_TYPES)
    for verbs in GATES.values():
        assert verbs <= GATED_VERBS


def test_connected_to_gates_only_remove():
    assert GATES["connected_to"] == frozenset({"remove"})


def test_a_structural_edge_gates_the_three_moving_verbs():
    moving = frozenset({"remove", "displace", "open"})
    assert GATES["fastened_by"] == moving
    assert GATES["locked_by"] == moving


def test_no_access_means_every_verb():
    assert GATES["covered_by"] == GATED_VERBS
    assert GATES["blocked_by"] == GATED_VERBS


def test_an_unknown_type_falls_back_to_gating_everything():
    assert gated_verbs("something_new") == GATED_VERBS


# --------------------------------------------------------------------------- #
# what that changes for one action
# --------------------------------------------------------------------------- #
def test_a_plugged_cable_does_not_stop_the_psu_being_swung_aside(edges):
    gating = _triples(applicable_preconditions(edges, ("displace", "psu.01")))
    assert ("connected_to", "psu.01", "connector.atx_24pin.01") not in gating
    assert ("locked_by", "psu.01", "psu_latch.01") in gating


def test_a_plugged_cable_still_stops_the_psu_being_taken_away(edges):
    gating = _triples(applicable_preconditions(edges, ("remove", "psu.01")))
    assert ("connected_to", "psu.01", "connector.atx_24pin.01") in gating
    assert ("locked_by", "psu.01", "psu_latch.01") in gating


def test_a_screw_does_not_have_to_come_out_before_a_plug_is_pulled(edges):
    """``fastened_by`` is about moving the part, not about reaching its socket."""
    gating = applicable_preconditions(edges, ("disconnect", "connector.atx_24pin.01"))
    assert all(e.type != "fastened_by" for e in gating)


def test_a_cover_still_gates_everything_underneath_it(edges):
    gating = _triples(applicable_preconditions(edges, ("unscrew", "screw.motherboard.01")))
    assert ("covered_by", "screw.motherboard.01", "cover.01") in gating


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #
def test_swinging_a_plugged_psu_out_is_no_longer_a_violation(bench, edges, tax):
    actions = [act(1, "psu_latch.01", "open"), act(2, "psu.01", "displace")]
    assert validate_sequence(bench, edges, actions, tax) == []


def test_taking_a_plugged_psu_away_still_is(bench, edges, tax):
    actions = [
        act(1, "psu_latch.01", "open"),
        act(2, "psu.01", "displace"),
        act(3, "psu.01", "remove"),
    ]
    problems = validate_sequence(bench, edges, actions, tax)
    assert len(problems) == 1
    assert "step 3" in problems[0] and "connected_to" in problems[0]


# --------------------------------------------------------------------------- #
# what is legal now
# --------------------------------------------------------------------------- #
def test_a_plugged_psu_is_displaceable_once_its_latch_is_open(bench, edges, tax):
    state = state_after(bench, [act(1, "psu_latch.01", "open")], tax)
    legal = legal_actions(bench, edges, state, tax)
    assert ("displace", "psu.01") in legal
    assert ("remove", "psu.01") not in legal


# --------------------------------------------------------------------------- #
# the planner reads the same table
# --------------------------------------------------------------------------- #
def test_a_removal_plan_still_pulls_the_plugs_first(bench, edges, tax):
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "psu.01", tax)
    assert plan is not None
    assert ("disconnect", "connector.atx_24pin.01") in plan
    assert plan.index(("disconnect", "connector.atx_24pin.01")) < \
        plan.index(("remove", "psu.01"))


def test_a_displace_inside_a_plan_does_not_drag_the_cables_along(bench, edges, tax):
    """The planner must honour the table too, not its own copy of the rule."""
    from tda.core.graph_plan import remaining_plan as planner

    assert planner is remaining_plan
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "motherboard.01", tax)
    assert plan is not None
    # the board's own plug must still be pulled before the board comes out
    assert ("disconnect", "connector.atx_24pin.01") in plan
