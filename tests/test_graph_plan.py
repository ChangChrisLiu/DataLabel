"""Constraint-graph reasoning: validation, cycles, plans, storage, templates.

The rule half -- how the edges are derived in the first place -- is in
``test_graph.py``. Both share ``graph_scenes.py``.
"""
from __future__ import annotations

import pytest
from graph_scenes import BENCH_CLASSES, DESKTOP, EXPECTED, FIXTURES, NOT_REMOVABLE
from graph_scenes import act as _act
from graph_scenes import bench_instances
from graph_scenes import good_sequence as _good_sequence
from graph_scenes import inst as _inst
from graph_scenes import state_after as _state
from graph_scenes import triples as _triples

from tda.core import logs
from tda.core.db import Db
from tda.core.graph import (
    Edge,
    edges_from_db,
    edges_to_db,
    find_deadlocks,
    legal_actions,
    propose_edges,
    remaining_plan,
    validate_sequence,
)
from tda.core.graph_rules import infer_relational_fields
from tda.core.graph_templates import apply_template, save_template
from tda.core.model import InstanceRec
from tda.core.states import initial_state
from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def bench() -> dict[str, InstanceRec]:
    return bench_instances()


# --------------------------------------------------------------------------- #
# 4. validate_sequence (spec 7.4)
# --------------------------------------------------------------------------- #
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
        "step 1: failed unscrew screw.cpu_cooler.01 has no unmet constraint - missing edge?"
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
# 5. deadlocks (the spec 7.4 check; see tests/test_graph_deadlock.py for the
#    definition itself -- these two pin the planner and the check together)
# --------------------------------------------------------------------------- #
def test_the_bench_graph_deadlocks_nothing(bench, tax):
    assert find_deadlocks(propose_edges(bench, tax), bench, tax) == []


def test_a_two_node_block_is_a_deadlock_and_the_planner_agrees(bench, tax):
    edges = propose_edges(bench, tax) + [
        Edge("blocked_by", "psu.01", "motherboard.01", mode="physical_path"),
        Edge("blocked_by", "motherboard.01", "psu.01", mode="physical_path"),
    ]
    found = find_deadlocks(edges, bench, tax)
    assert len(found) == 1
    assert remaining_plan(bench, edges, initial_state(bench, tax), "psu.01", tax) is None


def test_a_long_chain_does_not_blow_the_recursion_limit(tax):
    """Tarjan is iterative, so a chain far deeper than the limit is fine."""
    keys = [f"expansion_card.{i:04d}" for i in range(2000)]
    instances = {k: InstanceRec(key=k, desktop=900, cls="expansion_card") for k in keys}
    edges = [Edge("blocked_by", keys[i], keys[i + 1], mode="physical_path")
             for i in range(len(keys) - 1)]
    assert find_deadlocks(edges, instances, tax) == []
    edges.append(Edge("blocked_by", keys[-1], keys[0], mode="physical_path"))
    found = find_deadlocks(edges, instances, tax)
    assert len(found) == 1
    assert len(found[0].actions) == len(keys)


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


def test_remaining_plan_over_every_class(bench, tax):
    """A plan must exist exactly when the goal can actually be removed.

    For every instance of the bench desktop: an unremovable class gives
    ``None``, and any other goal gives a plan that both replays cleanly *and*
    genuinely leaves the goal ``removed`` -- which is what catches a plan that
    merely emits ``("remove", goal)`` for something ``remove`` cannot touch.
    """
    edges = propose_edges(bench, tax)
    start = initial_state(bench, tax)
    seen = set()
    for key, rec in sorted(bench.items()):
        seen.add(rec.cls)
        plan = remaining_plan(bench, edges, start, key, tax)
        if rec.cls in NOT_REMOVABLE:
            assert plan is None, f"{key} ({rec.cls}) cannot be removed"
            continue
        assert plan is not None, key
        replay = [_act(k, t, v) for k, (v, t) in enumerate(plan, start=1)]
        assert validate_sequence(bench, edges, replay, tax) == [], key
        assert _state(bench, replay, tax)[key].state == "removed", key
    assert seen == BENCH_CLASSES  # the sweep really did cover every class


def test_remaining_plan_for_a_loose_screw_stops_at_unscrew(bench, tax):
    """``unscrew`` already removes a non-captive screw; no ``remove`` follows."""
    edges = propose_edges(bench, tax)
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "screw.motherboard.01", tax)
    assert plan == [("open", "cover.01"), ("unscrew", "screw.motherboard.01")]


def test_remaining_plan_for_a_captive_screw_removes_its_parent(bench, tax):
    """A captive screw stays in the bracket: it leaves only with the cooler."""
    edges = propose_edges(bench, tax)
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "screw.cpu_cooler.01", tax)
    assert plan[-1] == ("remove", "cpu_cooler.fan.01")
    assert ("remove", "screw.cpu_cooler.01") not in plan
    assert ("unscrew", "screw.cpu_cooler.01") in plan


def test_remaining_plan_disconnects_a_connector_before_removing_it(bench, tax):
    edges = propose_edges(bench, tax)
    plan = remaining_plan(bench, edges, initial_state(bench, tax), "connector.atx_24pin.01", tax)
    assert plan == [
        ("disconnect", "connector.atx_24pin.01"),
        ("remove", "connector.atx_24pin.01"),
    ]


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


def test_template_roundtrips_a_virtual_cable_endpoint(bench, tmp_path):
    """The manual ``cable_tension`` edges are the priciest human input of all.

    A ``cable:*`` node has no ``slot_id`` because it is not an instance; its own
    id is stable across a family, so it *is* its slot.
    """
    edge = Edge("blocked_by", "motherboard.01", "cable:psu", mode="cable_tension",
                reason="the harness is pulled taut", source="manual")
    path = tmp_path / "cables.yaml"
    skipped: list[str] = []
    assert save_template(path, bench, [edge], skipped) == 1
    assert skipped == []

    report: list[str] = []
    applied = apply_template(path, bench, report)
    assert report == []
    assert len(applied) == 1
    assert (applied[0].target, applied[0].blocker) == ("motherboard.01", "cable:psu")
    assert applied[0].mode == "cable_tension"
    assert applied[0].source == "template"


def test_template_skips_a_cable_the_sibling_does_not_have(bench, tmp_path):
    edge = Edge("blocked_by", "motherboard.01", "cable:psu", mode="cable_tension")
    path = tmp_path / "cables.yaml"
    save_template(path, bench, [edge])
    cableless = {k: v for k, v in bench.items() if v.cls != "connector"}
    report: list[str] = []
    assert apply_template(path, cableless, report) == []
    assert len(report) == 1
    assert "cable:psu" in report[0]


def test_save_template_reports_what_it_drops(bench, tmp_path):
    edges = [
        Edge("blocked_by", "motherboard.01", "ghost.01"),
        Edge("locked_by", "psu.01", "psu_latch.01"),
    ]
    path = tmp_path / "partial.yaml"
    skipped: list[str] = []
    assert save_template(path, bench, edges, skipped) == 1
    assert len(skipped) == 1
    assert "ghost.01" in skipped[0]


def test_apply_template_reports_an_unknown_edge_type(bench, tmp_path):
    path = tmp_path / "bogus.yaml"
    save_template(path, bench, [Edge("supports", "motherboard.01", "chassis")])
    report: list[str] = []
    assert apply_template(path, bench, report) == []
    assert len(report) == 1
    assert "supports" in report[0]


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
    assert find_deadlocks(edges, imported.instances, tax) == []
    problems = validate_sequence(imported.instances, edges, imported.actions, tax)
    assert isinstance(problems, list)
    assert all(isinstance(p, str) for p in problems)


#: Latch-like classes and the chassis: ``remove`` does not apply (spec 6.3).
_UNREMOVABLE_CLASSES = frozenset({
    "chassis", "cpu_socket_lever", "psu_latch", "ram_latch", "drive_latch",
    "card_latch", "cooler_latch", "cable_clip",
})


@pytest.mark.parametrize("desktop", [1, 13, 63])
def test_every_real_instance_plans_or_honestly_refuses(desktop, tax):
    """End to end over a real machine, for *every* instance, not just the parts.

    Each goal either has no answer (its class cannot be removed) or gets a plan
    that replays clean through ``validate_sequence`` and genuinely ends with the
    goal ``removed`` -- the check that a plan was really achieved rather than
    merely emitted.
    """
    path = FIXTURES / f"desktop_{desktop:02d}.csv"
    if not path.exists():
        pytest.skip(f"{path} is missing")
    rows, meta = logs.read_desktop_csv(path)
    imported = logs.import_log(desktop, rows, meta, tax)
    instances = imported.instances
    infer_relational_fields(instances, tax)
    edges = propose_edges(instances, tax)
    state = initial_state(instances, tax)
    planned = 0
    for goal, rec in sorted(instances.items()):
        plan = remaining_plan(instances, edges, state, goal, tax)
        if rec.cls in _UNREMOVABLE_CLASSES:
            assert plan is None, f"{goal} ({rec.cls})"
            continue
        assert plan is not None, goal
        replay = [_act(k, t, v) for k, (v, t) in enumerate(plan, start=1)]
        assert validate_sequence(instances, edges, replay, tax) == [], goal
        assert _state(instances, replay, tax)[goal].state == "removed", goal
        planned += 1
    assert planned  # the machine has something to take apart


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
