"""Constraint-graph rules: edges, preconditions and legal actions (spec 7.1-7.3).

The reasoning half -- sequence validation, cycles, planning, persistence and
family templates -- is in ``test_graph_plan.py``. Both share ``graph_scenes.py``.
"""
from __future__ import annotations

import pytest
from graph_scenes import EXPECTED
from graph_scenes import act as _act
from graph_scenes import bench_instances
from graph_scenes import inst as _inst
from graph_scenes import state_after as _state
from graph_scenes import triples as _triples

from tda.core import graph
from tda.core.graph import (
    Edge,
    applicable_preconditions,
    cable_owner,
    legal_actions,
    propose_edges,
    unmet,
)
from tda.core.model import InstanceRec
from tda.core.states import initial_state
from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def bench() -> dict[str, InstanceRec]:
    return bench_instances()


def test_propose_edges_exact_set(bench, tax):
    edges = propose_edges(bench, tax)
    assert _triples(edges) == EXPECTED
    assert len(edges) == len(EXPECTED)  # no duplicates


def test_proposed_edges_carry_provenance(bench, tax):
    for edge in propose_edges(bench, tax):
        assert edge.necessity == "required"
        assert edge.source == "rule"
        assert edge.status == "proposed"
        assert edge.type in graph.HARD_TYPES
        assert edge.reason  # every rule edge explains itself
        assert edge.mode is None  # mode is for blocked_by, which no rule derives


def test_sata_drive_needs_only_its_own_end(bench, tax):
    """Spec 7.2: the two ends of a SATA data cable are independent."""
    edges = propose_edges(bench, tax)
    drive = applicable_preconditions(edges, ("remove", "storage_drive.hdd.01"))
    assert _triples(drive) == {
        ("connected_to", "storage_drive.hdd.01", "connector.sata_data.01")
    }


def test_psu_harness_blocks_the_psu(bench, tax):
    """Spec 7.2: every plug of the PSU harness gates the PSU itself."""
    edges = propose_edges(bench, tax)
    psu = _triples(applicable_preconditions(edges, ("remove", "psu.01")))
    assert ("connected_to", "psu.01", "connector.atx_24pin.01") in psu
    assert ("locked_by", "psu.01", "psu_latch.01") in psu


def test_drive_cables_have_no_owner(bench, tax):
    """Spec 7.2 again, this time the way ``logs.py`` actually writes it.

    The importer tags both ends of a drive's SATA cables ``cable:storage_drive``
    as a grouping key, which is not the spec's "the cable is captive to this
    part". Taking that literally would make the motherboard-end plug gate the
    drive, and every real log would look like a violation.
    """
    for key in ("connector.sata_data.01", "connector.sata_data.02"):
        bench[key].cable = "cable:storage_drive"
    edges = propose_edges(bench, tax)
    drive = applicable_preconditions(edges, ("remove", "storage_drive.hdd.01"))
    assert _triples(drive) == {
        ("connected_to", "storage_drive.hdd.01", "connector.sata_data.01")
    }


def test_a_captive_cable_still_gates_its_owner(bench, tax):
    """A fan lead is moulded into the fan, so it gates the fan as well."""
    bench["connector.fan.01"] = _inst(
        "connector.fan.01", "connector", attrs={"kind": "fan"},
        socket_host="motherboard.01", cable="cable:cpu_fan",
    )
    edges = propose_edges(bench, tax)
    cooler = _triples(applicable_preconditions(edges, ("remove", "cpu_cooler.fan.01")))
    assert ("connected_to", "cpu_cooler.fan.01", "connector.fan.01") in cooler


def test_power_leads_belong_to_the_psu_harness(bench, tax):
    """A drive's *power* lead is part of the PSU harness, whatever it is tagged.

    ``taxonomy_map`` groups a drive's SATA power lead under the drive, but the
    far end of it is moulded into the PSU. Spec 7.2 therefore wants both: the
    plug gates the drive it sits on *and* the PSU it comes from.
    """
    bench["connector.sata_power.01"] = _inst(
        "connector.sata_power.01", "connector", attrs={"kind": "sata_power"},
        socket_host="storage_drive.hdd.01", cable="cable:storage_drive",
    )
    edges = propose_edges(bench, tax)
    assert _triples(applicable_preconditions(edges, ("remove", "storage_drive.hdd.01"))) == {
        ("connected_to", "storage_drive.hdd.01", "connector.sata_data.01"),
        ("connected_to", "storage_drive.hdd.01", "connector.sata_power.01"),
    }
    assert ("connected_to", "psu.01", "connector.sata_power.01") in _triples(
        applicable_preconditions(edges, ("remove", "psu.01"))
    )


@pytest.mark.parametrize("kind", ["atx_24pin", "cpu_power", "sata_power", "molex"])
def test_every_power_lead_kind_gates_the_psu(bench, tax, kind):
    bench["connector.x.01"] = _inst(
        "connector.x.01", "connector", attrs={"kind": kind},
        socket_host="motherboard.01", cable="cable:optical_drive",
    )
    edges = propose_edges(bench, tax)
    assert ("connected_to", "psu.01", "connector.x.01") in _triples(edges)


def test_sata_data_stays_ownerless_whatever_it_is_tagged(bench, tax):
    bench["connector.sata_data.02"].cable = "cable:psu"  # a mis-tagged data lead
    edges = propose_edges(bench, tax)
    psu = _triples(applicable_preconditions(edges, ("remove", "psu.01")))
    assert ("connected_to", "psu.01", "connector.sata_data.02") not in psu


def test_cable_owner():
    assert cable_owner("cable:psu") == "psu"
    assert cable_owner("cable:cpu_fan") == "cpu_cooler"  # alias onto a real class
    assert cable_owner("motherboard.01") is None
    assert cable_owner("cable:") is None
    assert cable_owner("cable:storage_drive") is None  # detachable at both ends
    assert cable_owner("cable:ssd") is None


def test_cable_owner_resolves_to_a_unique_instance(bench):
    assert cable_owner("cable:psu", bench) == "psu.01"
    assert cable_owner("cable:front_panel", bench) is None  # no such instance here


def test_unknown_or_missing_fields_derive_no_edge(tax):
    lone = {"screw.other.01": _inst("screw.other.01", "screw", attrs={"role": "other"})}
    assert propose_edges(lone, tax) == []


# --------------------------------------------------------------------------- #
# 2. preconditions and unmet
# --------------------------------------------------------------------------- #
def test_applicable_preconditions_by_verb(bench, tax):
    edges = propose_edges(bench, tax)
    mb = _triples(applicable_preconditions(edges, ("remove", "motherboard.01")))
    assert mb == {
        ("fastened_by", "motherboard.01", "screw.motherboard.01"),
        ("fastened_by", "motherboard.01", "screw.motherboard.02"),
        ("fastened_by", "motherboard.01", "screw.motherboard.03"),
        ("connected_to", "motherboard.01", "connector.atx_24pin.01"),
        ("connected_to", "motherboard.01", "connector.sata_data.02"),
    }
    # displace is gated by the structural edges only: a board whose cables are
    # still plugged can be lifted clear of its standoffs, just not taken away
    assert _triples(applicable_preconditions(edges, ("displace", "motherboard.01"))) == {
        t for t in mb if t[0] != "connected_to"
    }
    # a screw hidden under a cover cannot be unscrewed
    assert _triples(applicable_preconditions(edges, ("unscrew", "screw.motherboard.01"))) == {
        ("covered_by", "screw.motherboard.01", "cover.01")
    }
    # reorient is a capture action, not a disassembly one
    assert applicable_preconditions(edges, ("reorient", "chassis")) == []


def test_applicable_preconditions_accepts_an_action_record(bench, tax):
    edges = propose_edges(bench, tax)
    action = _act(1, "motherboard.01", "remove")
    assert applicable_preconditions(edges, action) == applicable_preconditions(
        edges, ("remove", "motherboard.01")
    )


def test_unmet_in_the_initial_state(bench, tax):
    edges = propose_edges(bench, tax)
    state = initial_state(bench, tax)
    gating = applicable_preconditions(edges, ("remove", "motherboard.01"))
    assert len(unmet(gating, state)) == 5


def test_unmet_shrinks_as_blockers_change_state(bench, tax):
    edges = propose_edges(bench, tax)
    gating = applicable_preconditions(edges, ("remove", "motherboard.01"))
    actions = [
        _act(1, "cover.01", "open"),
        _act(2, "connector.atx_24pin.01", "disconnect"),
        _act(3, "connector.sata_data.02", "disconnect"),
        _act(4, "screw.motherboard.01", "unscrew"),
        _act(5, "screw.motherboard.02", "unscrew"),
    ]
    state = _state(bench, actions, tax)
    assert _triples(unmet(gating, state)) == {
        ("fastened_by", "motherboard.01", "screw.motherboard.03"),
    }
    actions.append(_act(6, "screw.motherboard.03", "unscrew"))
    assert unmet(gating, _state(bench, actions, tax)) == []


def test_unmet_respects_necessity(bench, tax):
    state = initial_state(bench, tax)
    soft = Edge("blocked_by", "motherboard.01", "psu.01", necessity="recommended",
                mode="physical_path")
    assert unmet([soft], state, "required") == []
    assert unmet([soft], state, "recommended") == [soft]


def test_unmet_never_drops_an_edge_with_a_bad_necessity(bench, tax):
    """A typo must make an edge over-binding, not make it disappear."""
    state = initial_state(bench, tax)
    typo = Edge("locked_by", "motherboard.01", "psu_latch.01", necessity="requred")
    assert unmet([typo], state, "required") == [typo]


def test_unmet_ignores_rejected_edges(bench, tax):
    state = initial_state(bench, tax)
    rejected = Edge("locked_by", "motherboard.01", "psu_latch.01", status="rejected")
    assert unmet([rejected], state) == []
    assert applicable_preconditions([rejected], ("remove", "motherboard.01")) == []


def test_unmet_treats_removed_and_absent_blockers_as_satisfied(bench, tax):
    state = initial_state(bench, tax)
    ghost = Edge("blocked_by", "motherboard.01", "no_such_instance")
    assert unmet([ghost], state) == []
    state["psu.01"].state = "removed"
    assert unmet([Edge("locked_by", "motherboard.01", "psu.01")], state) == []


def test_unmet_uses_the_cable_default_for_an_unseen_cable_node(bench, tax):
    """A ``cable:*`` node enters the frame state only once an event names it."""
    state = initial_state(bench, tax)
    tension = Edge("blocked_by", "motherboard.01", "cable:psu", mode="cable_tension")
    assert unmet([tension], state) == [tension]  # still 'routed'
    released = _state(bench, [_act(1, "cable:psu", "release")], tax)
    assert unmet([tension], released) == []


# --------------------------------------------------------------------------- #
# 3. legal_actions
# --------------------------------------------------------------------------- #
def test_legal_actions_in_the_initial_state(bench, tax):
    edges = propose_edges(bench, tax)
    got = set(legal_actions(bench, edges, initial_state(bench, tax), tax))
    assert got == {
        ("disconnect", "connector.atx_24pin.01"),
        ("disconnect", "connector.sata_data.01"),
        ("disconnect", "connector.sata_data.02"),
        ("open", "cover.01"),
        ("open", "cpu_socket_lever.01"),
        ("open", "psu_latch.01"),
        ("open", "ram_latch.01"),
        ("open", "ram_latch.02"),
        # spec 6.3: a connector may be removed with its cable, plugged or not
        ("remove", "connector.atx_24pin.01"),
        ("remove", "connector.sata_data.01"),
        ("remove", "connector.sata_data.02"),
        ("remove", "cover.01"),
        # its two SATA plugs gate taking the drive away, not sliding it out of
        # the cage: `connected_to` gates `remove` only
        ("displace", "storage_drive.hdd.01"),
        ("unscrew", "screw.cpu_cooler.01"),
        ("unscrew", "screw.cpu_cooler.02"),
    }
    # the motherboard is screwed down, plugged in, and its screws are covered
    assert ("remove", "motherboard.01") not in got
    assert ("unscrew", "screw.motherboard.01") not in got
    # ... and still screwed down, so it cannot be moved at all
    assert ("displace", "motherboard.01") not in got
    # the drive may slide out of its cage, but not leave with its cables on
    assert ("remove", "storage_drive.hdd.01") not in got


def test_a_plugged_connector_can_be_removed_with_its_cable(bench, tax):
    """Spec 6.3: ``remove`` applies to a connector taken out with its cable."""
    edges = propose_edges(bench, tax)
    got = legal_actions(bench, edges, initial_state(bench, tax), tax)
    assert ("remove", "connector.atx_24pin.01") in got
    assert ("disconnect", "connector.atx_24pin.01") in got
    # and still so once it has been unplugged
    state = _state(bench, [_act(1, "connector.atx_24pin.01", "disconnect")], tax)
    after = legal_actions(bench, edges, state, tax)
    assert ("remove", "connector.atx_24pin.01") in after
    assert ("disconnect", "connector.atx_24pin.01") not in after


def test_legal_actions_is_sorted_and_stable(bench, tax):
    edges = propose_edges(bench, tax)
    got = legal_actions(bench, edges, initial_state(bench, tax), tax)
    assert got == sorted(got)
    assert len(got) == len(set(got))


def test_opening_the_cover_unlocks_the_motherboard_screws(bench, tax):
    edges = propose_edges(bench, tax)
    state = _state(bench, [_act(1, "cover.01", "open")], tax)
    got = set(legal_actions(bench, edges, state, tax))
    assert ("unscrew", "screw.motherboard.01") in got
    assert ("open", "cover.01") not in got  # already open


def test_motherboard_becomes_removable_only_when_everything_is_clear(bench, tax):
    edges = propose_edges(bench, tax)
    actions = [
        _act(1, "cover.01", "open"),
        _act(2, "connector.atx_24pin.01", "disconnect"),
        _act(3, "screw.motherboard.01", "unscrew"),
        _act(4, "screw.motherboard.02", "unscrew"),
        _act(5, "screw.motherboard.03", "unscrew"),
    ]
    for k in range(len(actions) + 1):
        state = _state(bench, actions[:k], tax)
        got = legal_actions(bench, edges, state, tax)
        # the motherboard-side SATA plug is still in, so it is never removable
        assert ("remove", "motherboard.01") not in got
    actions.append(_act(6, "connector.sata_data.02", "disconnect"))
    got = legal_actions(bench, edges, _state(bench, actions, tax), tax)
    assert ("remove", "motherboard.01") in got
    assert ("displace", "motherboard.01") in got


def test_legal_actions_skips_removed_instances(bench, tax):
    edges = propose_edges(bench, tax)
    actions = [_act(1, "cover.01", "open"), _act(2, "cover.01", "remove")]
    got = legal_actions(bench, edges, _state(bench, actions, tax), tax)
    assert not [pair for pair in got if pair[1] == "cover.01"]


def test_captive_screws_stay_loosened_and_then_removable(bench, tax):
    edges = propose_edges(bench, tax)
    state = _state(bench, [_act(1, "screw.cpu_cooler.01", "unscrew")], tax)
    assert state["screw.cpu_cooler.01"].state == "loosened"
    got = legal_actions(bench, edges, state, tax)
    assert ("unscrew", "screw.cpu_cooler.01") not in got
    assert ("remove", "screw.cpu_cooler.01") in got


def test_strict_mode_also_honours_recommended_edges(bench, tax):
    edges = propose_edges(bench, tax) + [
        Edge("blocked_by", "cover.01", "psu.01", necessity="recommended", mode="tool_access")
    ]
    state = initial_state(bench, tax)
    assert ("open", "cover.01") not in legal_actions(bench, edges, state, tax, strict=True)
    assert ("open", "cover.01") in legal_actions(bench, edges, state, tax, strict=False)

