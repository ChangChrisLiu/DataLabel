"""The constraint editor's pure layer (:mod:`tda.core.graph_edit`), task B5 step 1.

Two provenances live in the same table and behave differently on purpose:

* a **rule** edge is derived from the instance attributes on every
  ``constraints`` run, so editing one by hand would be undone silently. The
  human's way to disagree is the spec 7.3 decision -- ``accepted`` /
  ``rejected`` -- recorded as an ``override`` row the re-run keeps;
* a **manual** edge is somebody's own assertion. The rules never derive it,
  never touch it, and it survives every re-run byte for byte.

Everything here is pure: a list of edges in, a new list out, or
:class:`GraphEditError` with a reason the panel can show.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from graph_scenes import DESKTOP, bench_instances, good_sequence, triples

from tda.core.db import Db
from tda.core.graph import (
    Edge,
    active_edges,
    edge_digest,
    edges_from_db,
    find_cycles,
    graph_version,
    legal_actions,
    propose_edges,
)
from tda.core.graph_edit import (
    MANUAL,
    OVERRIDE,
    RULE,
    GraphEditError,
    add_manual_edge,
    remove_manual_edge,
    set_rule_decision,
    violations,
    violations_of,
)
from tda.core.model import ActionRec
from tda.core.states import state_at, events_from_actions
from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def instances():
    return bench_instances()


@pytest.fixture
def rules(instances, tax) -> list[Edge]:
    """The desktop's rule edges, exactly as ``constraints`` would store them."""
    return [Edge(**{**vars(e), "source": RULE}) for e in propose_edges(instances, tax)]


def add(edges, instances, target, kind, blocker, **kw):
    return add_manual_edge(edges, target, kind, blocker, instances=instances, **kw)


BLOCK = {"mode": "physical_path"}


# --------------------------------------------------------------------------- #
# adding a manual edge
# --------------------------------------------------------------------------- #
def test_a_manual_edge_is_appended_with_its_own_provenance(rules, instances):
    out = add(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01",
              note="the PSU is in the way", **BLOCK)
    assert len(out) == len(rules) + 1
    edge = out[-1]
    assert (edge.type, edge.target, edge.blocker) == (
        "blocked_by", "storage_drive.hdd.01", "psu.01")
    assert edge.source == MANUAL
    assert edge.status == "accepted"      # a human wrote it: it is not a proposal
    assert edge.mode == "physical_path"
    assert edge.reason == "the PSU is in the way"
    assert edge.necessity == "required"


def test_adding_does_not_mutate_the_list_it_was_given(rules, instances):
    before = list(rules)
    add(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01", **BLOCK)
    assert rules == before


def test_a_recommended_manual_edge_keeps_its_necessity(rules, instances):
    out = add(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01",
              necessity="recommended", **BLOCK)
    assert out[-1].necessity == "recommended"


# --------------------------------------------------------------------------- #
# every refusal (review focus 5)
# --------------------------------------------------------------------------- #
def refusal(edges, instances, *args, **kw) -> str:
    with pytest.raises(GraphEditError) as excinfo:
        add(edges, instances, *args, **kw)
    text = str(excinfo.value)
    assert "/" in text, "every refusal is bilingual: 中文 / English"
    return text


def test_unknown_endpoint_is_refused_by_name(rules, instances):
    text = refusal(rules, instances, "psu.99", "blocked_by", "psu.01", **BLOCK)
    assert "psu.99" in text


def test_a_deleted_endpoint_is_refused(rules, instances):
    """S1 deleted the drive; an edge that still names it can never be written."""
    del instances["storage_drive.hdd.01"]
    text = refusal(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01",
                   **BLOCK)
    assert "storage_drive.hdd.01" in text


def test_a_provisional_endpoint_is_refused(rules, instances, tax):
    from tda.core.model import InstanceRec

    instances["ls:PSU#1"] = InstanceRec(key="ls:PSU#1", desktop=DESKTOP, cls="psu")
    text = refusal(rules, instances, "storage_drive.hdd.01", "blocked_by", "ls:PSU#1",
                   **BLOCK)
    assert "ls:PSU#1" in text


def test_a_self_edge_is_refused(rules, instances):
    refusal(rules, instances, "psu.01", "blocked_by", "psu.01", **BLOCK)


def test_a_duplicate_of_a_rule_edge_is_refused(rules, instances):
    text = refusal(rules, instances, "motherboard.01", "fastened_by",
                   "screw.motherboard.01")
    assert "rule" in text


def test_a_duplicate_of_a_manual_edge_is_refused(rules, instances):
    once = add(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01", **BLOCK)
    text = refusal(once, instances, "storage_drive.hdd.01", "blocked_by", "psu.01",
                   **BLOCK)
    assert "manual" in text


def test_a_duplicate_of_a_rejected_edge_is_still_a_duplicate(rules, instances):
    """The row exists; a second one cannot, the unique index says so."""
    rejected = set_rule_decision(rules, "motherboard.01", "fastened_by",
                                 "screw.motherboard.01", "rejected")
    refusal(rejected, instances, "motherboard.01", "fastened_by", "screw.motherboard.01")


def test_a_kind_outside_the_spec_7_set_is_refused(rules, instances):
    text = refusal(rules, instances, "psu.01", "partner_of", "motherboard.01")
    assert "fastened_by" in text  # the reason lists what is allowed


def test_blocked_by_without_a_mode_is_refused(rules, instances):
    text = refusal(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01")
    assert "cable_tension" in text


def test_a_mode_outside_the_spec_7_set_is_refused(rules, instances):
    refusal(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01",
            mode="in_the_way")


def test_a_mode_on_a_type_that_carries_none_is_refused(rules, instances):
    text = refusal(rules, instances, "storage_drive.hdd.01", "covered_by", "psu.01",
                   mode="physical_path")
    assert "blocked_by" in text


def test_an_unknown_necessity_is_refused(rules, instances):
    refusal(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01",
            necessity="nice_to_have", **BLOCK)


def test_a_cycle_among_hard_constraints_is_refused_and_named(rules, instances):
    """``cpu covered_by cooler`` is a rule edge; the reverse closes a loop."""
    text = refusal(rules, instances, "cpu_cooler.fan.01", "blocked_by", "cpu.01",
                   **BLOCK)
    assert "cpu.01" in text and "cpu_cooler.fan.01" in text
    assert "->" in text


def test_a_longer_cycle_is_refused_too(rules, instances):
    one = add(rules, instances, "psu.01", "blocked_by", "storage_drive.hdd.01", **BLOCK)
    two = add(one, instances, "storage_drive.hdd.01", "blocked_by", "motherboard.01",
              **BLOCK)
    text = refusal(two, instances, "motherboard.01", "blocked_by", "psu.01", **BLOCK)
    for key in ("psu.01", "storage_drive.hdd.01", "motherboard.01"):
        assert key in text


def test_a_cycle_through_a_rejected_edge_is_not_a_cycle(rules, instances):
    """A rejected edge gates nothing, so it cannot close a loop either."""
    without = set_rule_decision(rules, "cpu.01", "covered_by", "cpu_cooler.fan.01",
                                "rejected")
    out = add(without, instances, "cpu_cooler.fan.01", "blocked_by", "cpu.01", **BLOCK)
    assert find_cycles(active_edges(out)) == []


def test_a_cable_node_may_block(rules, instances):
    """Spec 7.3: a clipped cable blocks a target; the node is virtual, not an instance."""
    out = add(rules, instances, "motherboard.01", "blocked_by", "cable:psu",
              mode="cable_tension")
    assert out[-1].blocker == "cable:psu"


def test_a_cable_node_cannot_be_the_target(rules, instances):
    refusal(rules, instances, "cable:psu", "blocked_by", "motherboard.01", **BLOCK)


# --------------------------------------------------------------------------- #
# removing
# --------------------------------------------------------------------------- #
def test_removing_a_manual_edge_drops_exactly_it(rules, instances):
    out = add(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01", **BLOCK)
    back = remove_manual_edge(out, "storage_drive.hdd.01", "blocked_by", "psu.01")
    assert back == rules


def test_removing_a_rule_edge_is_refused_and_points_at_the_decision(rules):
    with pytest.raises(GraphEditError) as excinfo:
        remove_manual_edge(rules, "motherboard.01", "fastened_by", "screw.motherboard.01")
    text = str(excinfo.value)
    assert "/" in text
    assert "reject" in text


def test_removing_an_edge_that_is_not_there_is_refused(rules):
    with pytest.raises(GraphEditError):
        remove_manual_edge(rules, "psu.01", "blocked_by", "motherboard.01")


# --------------------------------------------------------------------------- #
# deciding about a rule edge (spec 7.3 status, spec 4.1 S6)
# --------------------------------------------------------------------------- #
def one(edges, kind, target, blocker) -> Edge:
    return next(e for e in edges
                if (e.type, e.target, e.blocker) == (kind, target, blocker))


def test_rejecting_a_rule_edge_records_an_override(rules):
    out = set_rule_decision(rules, "cpu.01", "covered_by", "cpu_cooler.fan.01", "rejected")
    edge = one(out, "covered_by", "cpu.01", "cpu_cooler.fan.01")
    assert edge.source == OVERRIDE
    assert edge.status == "rejected"
    assert edge.reason == one(rules, "covered_by", "cpu.01", "cpu_cooler.fan.01").reason
    assert edge not in active_edges(out)
    assert len(out) == len(rules)  # a decision, not a deletion


def test_accepting_a_rule_edge_records_an_override_too(rules):
    out = set_rule_decision(rules, "cpu.01", "covered_by", "cpu_cooler.fan.01", "accepted")
    edge = one(out, "covered_by", "cpu.01", "cpu_cooler.fan.01")
    assert (edge.source, edge.status) == (OVERRIDE, "accepted")
    assert edge in active_edges(out)


def test_a_decision_can_be_taken_back(rules):
    out = set_rule_decision(rules, "cpu.01", "covered_by", "cpu_cooler.fan.01", "rejected")
    back = set_rule_decision(out, "cpu.01", "covered_by", "cpu_cooler.fan.01", "proposed")
    assert back == rules, "clearing a decision leaves the rules' own edge"


def test_deciding_about_a_manual_edge_is_refused(rules, instances):
    out = add(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01", **BLOCK)
    with pytest.raises(GraphEditError) as excinfo:
        set_rule_decision(out, "storage_drive.hdd.01", "blocked_by", "psu.01", "rejected")
    assert "/" in str(excinfo.value)


def test_an_unknown_decision_is_refused(rules):
    with pytest.raises(GraphEditError):
        set_rule_decision(rules, "cpu.01", "covered_by", "cpu_cooler.fan.01", "maybe")


def test_deciding_about_an_edge_that_is_not_there_is_refused(rules):
    with pytest.raises(GraphEditError):
        set_rule_decision(rules, "psu.01", "blocked_by", "motherboard.01", "rejected")


# --------------------------------------------------------------------------- #
# what the graph then says (the exports read exactly this)
# --------------------------------------------------------------------------- #
def test_a_manual_edge_removes_an_action_from_the_legal_set(rules, instances, tax):
    state = state_at(instances, [], 1, tax)
    assert ("disconnect", "connector.sata_data.01") in legal_actions(
        instances, active_edges(rules), state, tax, strict=False)
    out = add(rules, instances, "connector.sata_data.01", "blocked_by", "psu.01", **BLOCK)
    assert ("disconnect", "connector.sata_data.01") not in legal_actions(
        instances, active_edges(out), state, tax, strict=False)


def test_rejecting_a_rule_edge_puts_an_action_back_into_the_legal_set(instances, tax,
                                                                     rules):
    state = state_at(instances, [], 1, tax)
    assert ("remove", "cpu.01") not in legal_actions(
        instances, active_edges(rules), state, tax, strict=False)
    out = set_rule_decision(rules, "cpu.01", "covered_by", "cpu_cooler.fan.01", "rejected")
    out = set_rule_decision(out, "cpu.01", "locked_by", "cpu_socket_lever.01", "rejected")
    assert ("remove", "cpu.01") in legal_actions(
        instances, active_edges(out), state, tax, strict=False)


# --------------------------------------------------------------------------- #
# the violations list -- the same replay `constraints --validate` prints
# --------------------------------------------------------------------------- #
def failed_attempt() -> list[ActionRec]:
    """A teardown in which taking the drive out is tried and does not work.

    Its SATA lead is pulled first, so by step 9 the graph knows of nothing in
    the drive's way -- which is exactly the "missing edge?" case of spec 7.4.
    """
    return [*good_sequence(),
            ActionRec(desktop=DESKTOP, step=8, idx=0, target="connector.sata_data.01",
                      verb="disconnect", result="success"),
            ActionRec(desktop=DESKTOP, step=9, idx=0, target="storage_drive.hdd.01",
                      verb="remove", result="failed", failure_reason="blocked")]


def test_a_failed_attempt_nothing_explains_is_a_hint(instances, rules, tax):
    found = violations_of(instances, rules, failed_attempt(), tax)
    assert [(v.step, v.kind) for v in found] == [(9, "hint")]
    assert "missing edge?" in found[0].text


def test_the_right_blocked_by_makes_the_hint_disappear(instances, rules, tax):
    out = add(rules, instances, "storage_drive.hdd.01", "blocked_by", "psu.01", **BLOCK)
    assert violations_of(instances, out, failed_attempt(), tax) == []


def test_a_blocker_that_is_already_gone_does_not_explain_the_attempt(instances, rules,
                                                                    tax):
    """The edge has to name something still in the way at that step."""
    out = add(rules, instances, "storage_drive.hdd.01", "blocked_by", "motherboard.01",
              **BLOCK)
    assert [v.kind for v in violations_of(instances, out, failed_attempt(), tax)] == \
        ["hint"]


def test_a_breach_carries_its_step_and_text(instances, rules, tax):
    actions = [ActionRec(desktop=DESKTOP, step=3, idx=0, target="motherboard.01",
                         verb="remove", result="success")]
    found = violations_of(instances, rules, actions, tax)
    assert found and all(v.kind == "breach" for v in found)
    assert all(v.step == 3 for v in found)
    assert any("fastened_by" in v.text for v in found)


def test_violations_ignores_the_label_studio_rows(instances, rules, tax):
    """`partner_of` is an annotation of another kind and gates nothing."""
    noise = [*rules, Edge(type="partner_of", target="psu.01", blocker="motherboard.01",
                          source="labelstudio")]
    assert violations_of(instances, noise, failed_attempt(), tax) == \
        violations_of(instances, rules, failed_attempt(), tax)


# --------------------------------------------------------------------------- #
# on a database: the `constraints` re-run (review focus 5)
# --------------------------------------------------------------------------- #
@pytest.fixture
def db(tmp_path: Path) -> Db:
    conn = Db(str(tmp_path / "tda.sqlite"))
    conn.upsert_desktop(DESKTOP, {"brand": "Bench"})
    for rec in bench_instances().values():
        conn.upsert_instance(rec)
    conn.replace_steps(DESKTOP, [], failed_attempt())
    yield conn
    conn.close()


def rerun(db, tax) -> None:
    from tda.cli_graph import constraints_into_db

    constraints_into_db(db, tax, {DESKTOP})


def rows(db) -> list[dict]:
    return [r for r in db.relations(DESKTOP)]


def test_a_manual_edge_survives_a_constraints_rerun_byte_for_byte(db, tax, instances):
    rerun(db, tax)
    from tda.core.graph import edges_to_db

    edges_to_db(db, DESKTOP, [add(edges_from_db(db, DESKTOP), instances,
                                  "storage_drive.hdd.01", "blocked_by", "psu.01",
                                  note="the PSU is in the way", **BLOCK)[-1]])
    before = [r for r in rows(db) if r["source"] == MANUAL]
    assert len(before) == 1
    rerun(db, tax)
    after = [r for r in rows(db) if r["source"] == MANUAL]
    assert after == before


def test_the_rerun_keeps_a_rejected_rule_edge_rejected(db, tax):
    rerun(db, tax)
    stored = edges_from_db(db, DESKTOP)
    decided = set_rule_decision(stored, "cpu.01", "covered_by", "cpu_cooler.fan.01",
                                "rejected")
    from tda.core.graph import edges_to_db

    edges_to_db(db, DESKTOP, decided)
    rerun(db, tax)
    edge = one(edges_from_db(db, DESKTOP), "covered_by", "cpu.01", "cpu_cooler.fan.01")
    assert (edge.source, edge.status) == (OVERRIDE, "rejected")


def test_the_rerun_rederives_a_rule_edge_the_instances_still_imply(db, tax):
    rerun(db, tax)
    db.delete_relation(DESKTOP, "fastened_by", "motherboard.01", "screw.motherboard.01")
    rerun(db, tax)
    assert ("fastened_by", "motherboard.01", "screw.motherboard.01") in triples(
        edges_from_db(db, DESKTOP))


def test_graph_version_changes_only_when_the_edge_set_does(db, tax, instances):
    rerun(db, tax)
    first = graph_version(db, DESKTOP)
    rerun(db, tax)
    assert graph_version(db, DESKTOP) == first
    from tda.core.graph import edges_to_db

    edges_to_db(db, DESKTOP, [add(edges_from_db(db, DESKTOP), instances,
                                  "storage_drive.hdd.01", "blocked_by", "psu.01",
                                  **BLOCK)[-1]])
    second = graph_version(db, DESKTOP)
    assert second != first
    rerun(db, tax)
    assert graph_version(db, DESKTOP) == second


def test_a_reason_is_prose_and_does_not_change_the_version(db, tax, instances):
    rerun(db, tax)
    before = graph_version(db, DESKTOP)
    from tda.core.graph import edges_to_db

    edge = one(edges_from_db(db, DESKTOP), "locked_by", "psu.01", "psu_latch.01")
    edges_to_db(db, DESKTOP, [Edge(**{**vars(edge), "reason": "re-worded"})])
    assert graph_version(db, DESKTOP) == before


def test_violations_reads_the_database_the_way_the_cli_does(db, tax):
    rerun(db, tax)
    found = violations(db, DESKTOP, tax)
    assert [(v.step, v.kind) for v in found] == [(9, "hint")]


def test_the_export_quotes_a_graph_that_holds_the_manual_edge(db, tax, instances):
    from tda.core.export.vlm import graph_version_of

    rerun(db, tax)
    before = graph_version_of(db, DESKTOP)
    from tda.core.graph import edges_to_db

    edges_to_db(db, DESKTOP, [add(edges_from_db(db, DESKTOP), instances,
                                  "storage_drive.hdd.01", "blocked_by", "psu.01",
                                  **BLOCK)[-1]])
    assert graph_version_of(db, DESKTOP) not in (None, before)
    assert edge_digest(edges_from_db(db, DESKTOP)) == graph_version_of(db, DESKTOP)
