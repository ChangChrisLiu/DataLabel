"""Tests for the constraint graph (spec 7) -- tda.core.graph / graph_rules / graph_templates."""
from __future__ import annotations

from pathlib import Path

import pytest

from tda.core import graph, logs
from tda.core.db import Db
from tda.core.graph import (
    Edge,
    applicable_preconditions,
    cable_owner,
    edges_from_db,
    edges_to_db,
    find_cycles,
    legal_actions,
    propose_edges,
    remaining_plan,
    unmet,
    validate_sequence,
)
from tda.core.graph_rules import infer_relational_fields
from tda.core.graph_templates import apply_template, save_template
from tda.core.model import ActionRec, InstanceRec
from tda.core.states import events_from_actions, initial_state, state_at
from tda.core.taxonomy import load_taxonomy

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
DESKTOP = 900


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


# --------------------------------------------------------------------------- #
# A small hand-made desktop
# --------------------------------------------------------------------------- #
def _inst(key: str, cls: str, **kw) -> InstanceRec:
    return InstanceRec(key=key, desktop=DESKTOP, cls=cls, **kw)


@pytest.fixture
def bench() -> dict[str, InstanceRec]:
    """One motherboard, one cooler, RAM, a PSU, a SATA drive and a screw cover.

    Slot ids are the family-template slots; every relational field the rules
    read (``fastens``, ``socket_host``, ``cable``, ``of``) is filled in, so
    :func:`propose_edges` can run without the heuristics.
    """
    recs = [
        _inst("chassis", "chassis", slot_id="chassis"),
        _inst("motherboard.01", "motherboard", mounted_on="chassis", slot_id="mb"),
        _inst("cpu.01", "cpu", mounted_on="motherboard.01", slot_id="cpu"),
        _inst("cpu_cooler.fan.01", "cpu_cooler", attrs={"kind": "fan"},
              mounted_on="motherboard.01", slot_id="cooler"),
        _inst("cpu_socket_lever.01", "cpu_socket_lever", slot_id="lever"),
        _inst("ram_module.01", "ram_module", mounted_on="motherboard.01", slot_id="ram1"),
        _inst("psu.01", "psu", mounted_on="chassis", slot_id="psu"),
        _inst("psu_latch.01", "psu_latch", slot_id="psu_latch"),
        _inst("storage_drive.hdd.01", "storage_drive", attrs={"kind": "hdd"}, slot_id="hdd"),
        _inst("cover.01", "cover", attrs={"of": "motherboard_screws"}, slot_id="mb_cover"),
    ]
    for i in (1, 2, 3):
        recs.append(_inst(f"screw.motherboard.{i:02d}", "screw",
                          attrs={"role": "motherboard", "captive": False, "head": "PH2"},
                          fastens="motherboard.01", slot_id=f"mb_screw{i}"))
    for i in (1, 2):
        recs.append(_inst(f"screw.cpu_cooler.{i:02d}", "screw",
                          attrs={"role": "cpu_cooler", "captive": True, "head": "PH2"},
                          fastens="cpu_cooler.fan.01", parent="cpu_cooler.fan.01",
                          attached=True, slot_id=f"cooler_screw{i}"))
        recs.append(_inst(f"ram_latch.{i:02d}", "ram_latch", attrs={"of": "ram_module.01"},
                          slot_id=f"ram1_latch{i}"))
    # PSU 24-pin: socket on the motherboard, cable owned by the PSU.
    recs.append(_inst("connector.atx_24pin.01", "connector", attrs={"kind": "atx_24pin"},
                      socket_host="motherboard.01", cable="cable:psu", slot_id="atx"))
    # SATA data: two independent ends, no cable owner (spec 7.2).
    recs.append(_inst("connector.sata_data.01", "connector", attrs={"kind": "sata_data"},
                      socket_host="storage_drive.hdd.01", slot_id="sata_drive_end"))
    recs.append(_inst("connector.sata_data.02", "connector", attrs={"kind": "sata_data"},
                      socket_host="motherboard.01", slot_id="sata_mb_end"))
    return {r.key: r for r in recs}


#: Every edge :func:`propose_edges` must derive from ``bench``.
EXPECTED = {
    ("fastened_by", "motherboard.01", "screw.motherboard.01"),
    ("fastened_by", "motherboard.01", "screw.motherboard.02"),
    ("fastened_by", "motherboard.01", "screw.motherboard.03"),
    ("fastened_by", "cpu_cooler.fan.01", "screw.cpu_cooler.01"),
    ("fastened_by", "cpu_cooler.fan.01", "screw.cpu_cooler.02"),
    ("connected_to", "motherboard.01", "connector.atx_24pin.01"),
    ("connected_to", "psu.01", "connector.atx_24pin.01"),
    ("connected_to", "storage_drive.hdd.01", "connector.sata_data.01"),
    ("connected_to", "motherboard.01", "connector.sata_data.02"),
    ("locked_by", "ram_module.01", "ram_latch.01"),
    ("locked_by", "ram_module.01", "ram_latch.02"),
    ("locked_by", "cpu.01", "cpu_socket_lever.01"),
    ("locked_by", "psu.01", "psu_latch.01"),
    ("covered_by", "screw.motherboard.01", "cover.01"),
    ("covered_by", "screw.motherboard.02", "cover.01"),
    ("covered_by", "screw.motherboard.03", "cover.01"),
    ("covered_by", "cpu.01", "cpu_cooler.fan.01"),
}


def _triples(edges) -> set[tuple[str, str, str]]:
    return {(e.type, e.target, e.blocker) for e in edges}


def _act(step: int, target: str, verb: str, result: str = "success", idx: int = 0) -> ActionRec:
    return ActionRec(desktop=DESKTOP, step=step, idx=idx, target=target, verb=verb, result=result)


def _state(bench, actions, tax, step: int = 10**6):
    events = events_from_actions(bench, actions, tax)
    return state_at(bench, events, step, tax)


# --------------------------------------------------------------------------- #
# 1. propose_edges (spec 7.3)
# --------------------------------------------------------------------------- #
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
    # displace is gated exactly like remove
    assert _triples(applicable_preconditions(edges, ("displace", "motherboard.01"))) == mb
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
        ("remove", "cover.01"),
        ("unscrew", "screw.cpu_cooler.01"),
        ("unscrew", "screw.cpu_cooler.02"),
    }
    # the motherboard is screwed down, plugged in, and its screws are covered
    assert ("remove", "motherboard.01") not in got
    assert ("unscrew", "screw.motherboard.01") not in got


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


# --------------------------------------------------------------------------- #
# 4. validate_sequence (spec 7.4)
# --------------------------------------------------------------------------- #
def _good_sequence() -> list[ActionRec]:
    return [
        _act(1, "cover.01", "open"),
        _act(2, "connector.atx_24pin.01", "disconnect"),
        _act(3, "connector.sata_data.02", "disconnect"),
        _act(4, "screw.motherboard.01", "unscrew"),
        _act(5, "screw.motherboard.02", "unscrew"),
        _act(6, "screw.motherboard.03", "unscrew"),
        _act(7, "motherboard.01", "remove"),
    ]


def test_validate_sequence_accepts_a_correct_teardown(bench, tax):
    edges = propose_edges(bench, tax)
    assert validate_sequence(bench, edges, _good_sequence(), tax) == []


def test_validate_sequence_flags_a_violation(bench, tax):
    edges = propose_edges(bench, tax)
    actions = _good_sequence()
    actions.remove(_act(6, "screw.motherboard.03", "unscrew"))
    problems = validate_sequence(bench, edges, actions, tax)
    assert len(problems) == 1
    assert problems[0].startswith("step 7: remove motherboard.01 violates ")
    assert "fastened_by(motherboard.01, screw.motherboard.03)" in problems[0]
    assert "fastened" in problems[0]


def test_validate_sequence_reports_every_unmet_edge_of_one_action(bench, tax):
    edges = propose_edges(bench, tax)
    problems = validate_sequence(bench, edges, [_act(1, "motherboard.01", "remove")], tax)
    assert len(problems) == 5
    assert all(p.startswith("step 1: remove motherboard.01 violates ") for p in problems)


def test_validate_sequence_flags_a_failure_without_a_constraint(bench, tax):
    edges = propose_edges(bench, tax)
    actions = [_act(1, "screw.cpu_cooler.01", "unscrew", result="failed")]
    problems = validate_sequence(bench, edges, actions, tax)
    assert problems == [
        "step 1: failed unscrew screw.cpu_cooler.01 has no unmet constraint — missing edge?"
    ]


def test_validate_sequence_accepts_a_failure_with_a_constraint(bench, tax):
    edges = propose_edges(bench, tax)
    actions = [_act(1, "motherboard.01", "remove", result="failed")]
    assert validate_sequence(bench, edges, actions, tax) == []


def test_validate_sequence_orders_actions_within_a_step(bench, tax):
    """Two actions in one compound step: the later one sees the earlier one's effect."""
    edges = propose_edges(bench, tax)
    actions = [
        _act(1, "connector.sata_data.01", "disconnect", idx=0),
        _act(1, "storage_drive.hdd.01", "remove", idx=1),
    ]
    assert validate_sequence(bench, edges, actions, tax) == []
    swapped = [
        _act(1, "storage_drive.hdd.01", "remove", idx=0),
        _act(1, "connector.sata_data.01", "disconnect", idx=1),
    ]
    assert len(validate_sequence(bench, edges, swapped, tax)) == 1


# --------------------------------------------------------------------------- #
# 5. cycles
# --------------------------------------------------------------------------- #
def test_find_cycles_on_an_acyclic_graph(bench, tax):
    assert find_cycles(propose_edges(bench, tax)) == []


def test_find_cycles_finds_a_two_node_loop(bench, tax):
    edges = propose_edges(bench, tax) + [
        Edge("blocked_by", "psu.01", "motherboard.01", mode="physical_path"),
        Edge("blocked_by", "motherboard.01", "psu.01", mode="physical_path"),
    ]
    assert find_cycles(edges) == [["motherboard.01", "psu.01"]]


def test_find_cycles_finds_a_self_loop_and_a_longer_loop():
    edges = [
        Edge("blocked_by", "a", "a"),
        Edge("blocked_by", "b", "c"),
        Edge("blocked_by", "c", "d"),
        Edge("blocked_by", "d", "b"),
    ]
    assert find_cycles(edges) == [["a"], ["b", "c", "d"]]


def test_find_cycles_survives_a_long_chain():
    """Tarjan is iterative, so a chain far deeper than the recursion limit is fine."""
    edges = [Edge("blocked_by", f"n{i}", f"n{i + 1}") for i in range(5000)]
    edges.append(Edge("blocked_by", "n5000", "n0"))
    assert len(find_cycles(edges)) == 1
    assert len(find_cycles(edges)[0]) == 5001


# --------------------------------------------------------------------------- #
# 6. remaining_plan
# --------------------------------------------------------------------------- #
def test_remaining_plan_from_the_initial_state(bench, tax):
    edges = propose_edges(bench, tax)
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "motherboard.01", tax)
    assert plan is not None
    assert plan[-1] == ("remove", "motherboard.01")
    # the cover has to come off before its screws, and every mainboard plug first
    assert plan.index(("open", "cover.01")) < plan.index(("unscrew", "screw.motherboard.01"))
    for pair in (
        ("disconnect", "connector.atx_24pin.01"),
        ("disconnect", "connector.sata_data.02"),
        ("unscrew", "screw.motherboard.03"),
    ):
        assert pair in plan
    # nothing irrelevant: the RAM, the CPU and the drive stay where they are
    assert not [p for p in plan if p[1].startswith(("ram_", "cpu", "storage_"))]


def test_remaining_plan_replays_cleanly(bench, tax):
    edges = propose_edges(bench, tax)
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "motherboard.01", tax)
    actions = [_act(k, target, verb) for k, (verb, target) in enumerate(plan, start=1)]
    assert validate_sequence(bench, edges, actions, tax) == []


def test_remaining_plan_continues_from_a_partial_teardown(bench, tax):
    edges = propose_edges(bench, tax)
    done = [_act(1, "cover.01", "open"), _act(2, "connector.atx_24pin.01", "disconnect")]
    plan = remaining_plan(bench, edges, _state(bench, done, tax), "motherboard.01", tax)
    assert ("open", "cover.01") not in plan
    assert ("disconnect", "connector.atx_24pin.01") not in plan
    assert plan[-1] == ("remove", "motherboard.01")


def test_remaining_plan_is_already_done(bench, tax):
    edges = propose_edges(bench, tax)
    state = _state(bench, _good_sequence(), tax)
    assert remaining_plan(bench, edges, state, "motherboard.01", tax) == []


def test_remaining_plan_returns_none_on_a_cycle(bench, tax):
    edges = propose_edges(bench, tax) + [
        Edge("blocked_by", "motherboard.01", "psu.01"),
        Edge("blocked_by", "psu.01", "motherboard.01"),
    ]
    assert remaining_plan(bench, edges, initial_state(bench, tax), "motherboard.01", tax) is None


def test_remaining_plan_releases_a_clipped_cable(bench, tax):
    edges = propose_edges(bench, tax) + [
        Edge("blocked_by", "motherboard.01", "cable:psu", mode="cable_tension")
    ]
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "motherboard.01", tax)
    assert ("release", "cable:psu") in plan
    assert plan.index(("release", "cable:psu")) < plan.index(("remove", "motherboard.01"))


def test_remaining_plan_opens_a_latch_before_pulling_the_ram(bench, tax):
    edges = propose_edges(bench, tax)
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "ram_module.01", tax)
    assert plan == [
        ("open", "ram_latch.01"),
        ("open", "ram_latch.02"),
        ("remove", "ram_module.01"),
    ]


# --------------------------------------------------------------------------- #
# 7. database round trip
# --------------------------------------------------------------------------- #
def test_edges_db_roundtrip(bench, tax, tmp_db_path):
    db = Db(tmp_db_path)
    try:
        db.upsert_desktop(DESKTOP, {"brand": "bench"})
        edges = propose_edges(bench, tax)
        edges.append(Edge("blocked_by", "psu.01", "cable:psu", mode="cable_tension",
                          reason="手够不到", source="manual", evidence_step=7, status="accepted"))
        ids = edges_to_db(db, DESKTOP, edges)
        assert len(ids) == len(edges)
        back = edges_from_db(db, DESKTOP)
        assert sorted(back, key=lambda e: (e.type, e.target, e.blocker)) == sorted(
            edges, key=lambda e: (e.type, e.target, e.blocker)
        )
    finally:
        db.close()


def test_edges_to_db_is_idempotent(bench, tax, tmp_db_path):
    db = Db(tmp_db_path)
    try:
        db.upsert_desktop(DESKTOP, {})
        edges = propose_edges(bench, tax)
        edges_to_db(db, DESKTOP, edges)
        edges_to_db(db, DESKTOP, edges)
        assert len(edges_from_db(db, DESKTOP)) == len(edges)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 8. family templates (spec 7.3 item 2)
# --------------------------------------------------------------------------- #
def _sibling(bench: dict[str, InstanceRec]) -> dict[str, InstanceRec]:
    """The same family, different instance keys, one slot missing."""
    out: dict[str, InstanceRec] = {}
    for rec in bench.values():
        if rec.slot_id == "mb_screw3":  # this machine has two mainboard screws
            continue
        key = rec.key.replace(".", "_") + ".x"
        out[key] = InstanceRec(key=key, desktop=901, cls=rec.cls, attrs=dict(rec.attrs),
                               slot_id=rec.slot_id)
    return out


def test_template_roundtrip(bench, tax, tmp_path):
    edges = propose_edges(bench, tax)
    path = tmp_path / "family.yaml"
    save_template(path, bench, edges)
    assert path.exists()

    report: list[str] = []
    applied = apply_template(path, _sibling(bench), report)
    slot_of = {r.key: r.slot_id for r in _sibling(bench).values()}
    key_of = {slot: key for key, slot in slot_of.items()}
    # every edge whose two endpoints exist on the sibling is instantiated
    expected = {
        (t, key_of[bench[a].slot_id], key_of[bench[b].slot_id])
        for t, a, b in EXPECTED
        if bench[a].slot_id in key_of and bench[b].slot_id in key_of
    }
    assert _triples(applied) == expected
    assert all(e.source == "template" and e.status == "proposed" for e in applied)
    # the missing screw slot is skipped and reported
    assert len(report) == 1
    assert "mb_screw3" in report[0]


def test_template_falls_back_to_the_instance_key(tax, tmp_path):
    """An instance with no slot id is keyed by its own instance key."""
    recs = {
        "psu.01": _inst("psu.01", "psu"),
        "psu_latch.01": _inst("psu_latch.01", "psu_latch"),
    }
    edges = propose_edges(recs, tax)
    path = tmp_path / "noslots.yaml"
    save_template(path, recs, edges)
    assert _triples(apply_template(path, recs)) == {("locked_by", "psu.01", "psu_latch.01")}


def test_template_keeps_edge_attributes(bench, tmp_path):
    edges = [Edge("blocked_by", "motherboard.01", "psu.01", necessity="recommended",
                  mode="physical_path", reason="cable tension")]
    path = tmp_path / "attrs.yaml"
    save_template(path, bench, edges)
    applied = apply_template(path, bench)
    assert len(applied) == 1
    assert applied[0].necessity == "recommended"
    assert applied[0].mode == "physical_path"
    assert applied[0].reason == "cable tension"


# --------------------------------------------------------------------------- #
# 9. infer_relational_fields (the importer leaves these empty)
# --------------------------------------------------------------------------- #
def _d13(tax):
    rows, meta = logs.read_desktop_csv(FIXTURES / "desktop_13.csv")
    return logs.import_log(13, rows, meta, tax)


def test_infer_fills_screw_fastens_and_captive_parents(tax):
    imported = _d13(tax)
    instances = imported.instances
    assert instances["screw.motherboard.01"].fastens is None  # the importer leaves it empty
    filled = infer_relational_fields(instances, tax)
    assert instances["screw.motherboard.01"].fastens == "motherboard.01"
    assert instances["screw.cpu_cooler.01"].fastens == "cpu_cooler.fan.01"
    # captive cooler screws travel with the cooler
    assert instances["screw.cpu_cooler.01"].parent == "cpu_cooler.fan.01"
    assert instances["screw.cpu_cooler.01"].attached is True
    assert any("screw.motherboard.01.fastens" in line for line in filled)


def test_infer_pairs_ram_latches_with_the_nearest_module(tax):
    instances = _d13(tax).instances
    infer_relational_fields(instances, tax)
    assert instances["ram_latch.01"].attrs["of"] == "ram_module.01"
    assert instances["ram_latch.02"].attrs["of"] == "ram_module.01"
    assert instances["ram_latch.03"].attrs["of"] == "ram_module.02"
    assert instances["ram_latch.04"].attrs["of"] == "ram_module.02"


def test_infer_resolves_a_class_level_socket_host(tax):
    instances = _d13(tax).instances
    assert instances["connector.03"].socket_host == "motherboard"  # a class name, not a key
    infer_relational_fields(instances, tax)
    assert instances["connector.03"].socket_host == "motherboard.01"


def test_infer_is_idempotent_and_never_overwrites(tax, bench):
    assert infer_relational_fields(bench, tax) == []  # everything is already filled
    instances = _d13(tax).instances
    first = infer_relational_fields(instances, tax)
    assert first
    assert infer_relational_fields(instances, tax) == []


# --------------------------------------------------------------------------- #
# 10. real-data smoke test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("desktop", [1, 13, 63])
def test_real_log_smoke(desktop, tax):
    path = FIXTURES / f"desktop_{desktop:02d}.csv"
    if not path.exists():
        pytest.skip(f"{path} is missing")
    rows, meta = logs.read_desktop_csv(path)
    imported = logs.import_log(desktop, rows, meta, tax)
    infer_relational_fields(imported.instances, tax)
    edges = propose_edges(imported.instances, tax)
    assert edges  # the attribute rules find something on every real machine
    assert find_cycles(edges) == []
    problems = validate_sequence(imported.instances, edges, imported.actions, tax)
    assert isinstance(problems, list)
    assert all(isinstance(p, str) for p in problems)


#: Classes a teardown actually takes out of the chassis.
_PART_CLASSES = frozenset({
    "motherboard", "cpu", "cpu_cooler", "ram_module", "psu", "storage_drive",
    "optical_drive", "expansion_card", "case_fan", "misc_part", "drive_cage", "cover",
})


@pytest.mark.parametrize("desktop", [1, 13, 63])
def test_every_real_part_has_a_plan_that_replays_cleanly(desktop, tax):
    """End to end: plan each part out of a real machine, then validate the plan."""
    path = FIXTURES / f"desktop_{desktop:02d}.csv"
    if not path.exists():
        pytest.skip(f"{path} is missing")
    rows, meta = logs.read_desktop_csv(path)
    imported = logs.import_log(desktop, rows, meta, tax)
    infer_relational_fields(imported.instances, tax)
    edges = propose_edges(imported.instances, tax)
    state = initial_state(imported.instances, tax)
    goals = [k for k, r in sorted(imported.instances.items()) if r.cls in _PART_CLASSES]
    assert goals
    for goal in goals:
        plan = remaining_plan(imported.instances, edges, state, goal, tax)
        assert plan is not None, goal
        assert plan[-1] == ("remove", goal)
        replay = [_act(k, t, v) for k, (v, t) in enumerate(plan, start=1)]
        assert validate_sequence(imported.instances, edges, replay, tax) == [], goal


def test_real_log_legal_actions_are_never_empty_at_the_start(tax):
    path = FIXTURES / "desktop_13.csv"
    if not path.exists():
        pytest.skip(f"{path} is missing")
    rows, meta = logs.read_desktop_csv(path)
    imported = logs.import_log(13, rows, meta, tax)
    infer_relational_fields(imported.instances, tax)
    edges = propose_edges(imported.instances, tax)
    got = legal_actions(imported.instances, edges, initial_state(imported.instances, tax), tax)
    assert got
    assert ("remove", "motherboard.01") not in got
