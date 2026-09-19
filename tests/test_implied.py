"""Implied instances: the part every frame shows but no step ever touches.

Four of the 66 desktops (D49, D62, D63, D64) stop before the motherboard is
lifted out, so the log importer never names one -- and the 33 connectors and
screws that point at ``motherboard`` stay unresolved while the board itself,
plainly visible in every single frame, would go unlabelled.
:func:`tda.core.implied.implied_instances` creates exactly one such instance,
and only for the classes ``configs/taxonomy.yaml`` allows it for.

The synthetic fixture here mirrors D64 (motherboard screws undone, connectors
unplugged, "change a direction", end of sheet); the real ``desktop_63.csv``
fixture is one of the four and ``desktop_13.csv`` is the control -- its sheet
removes the board, so nothing may be implied for it.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from test_cli import d13_steps, env, open_db, run  # noqa: F401  (re-used fixtures)

from tda.cli import EXIT_OK
from tda.core.graph_infer import NO_CANDIDATE, infer_relational_fields, unresolved_relations
from tda.core.implied import (
    IMPLIED_ATTR,
    OP_KIND,
    implied_instances,
    is_implied,
    referencing_instances,
)
from tda.core.model import ActionRec, InstanceRec
from tda.core.taxonomy import load_taxonomy
from tda.ui.steps_issues import orphan_issues, unresolved_issues


@pytest.fixture
def tax():
    return load_taxonomy()


# --------------------------------------------------------------------------- #
# a synthetic D64: the board is unscrewed and unplugged, then never lifted
# --------------------------------------------------------------------------- #
def _d64_like(desktop: int = 64) -> dict[str, InstanceRec]:
    """Two motherboard screws and two motherboard-side connectors, no board."""
    out = {
        "chassis.01": InstanceRec("chassis.01", desktop, "chassis"),
        "screw.motherboard.01": InstanceRec(
            "screw.motherboard.01", desktop, "screw", attrs={"role": "motherboard"}),
        "screw.motherboard.02": InstanceRec(
            "screw.motherboard.02", desktop, "screw", attrs={"role": "motherboard"}),
        "connector.atx_24pin.01": InstanceRec(
            "connector.atx_24pin.01", desktop, "connector",
            attrs={"kind": "atx_24pin"}, socket_host="motherboard"),
        "connector.cpu_power.01": InstanceRec(
            "connector.cpu_power.01", desktop, "connector",
            attrs={"kind": "cpu_power"}, socket_host="motherboard"),
    }
    return out


def _d64_actions(desktop: int = 64) -> list[ActionRec]:
    return [
        ActionRec(desktop, 2, 0, "connector.atx_24pin.01", "disconnect"),
        ActionRec(desktop, 3, 0, "connector.cpu_power.01", "disconnect"),
        ActionRec(desktop, 4, 0, "screw.motherboard.01", "unscrew"),
        ActionRec(desktop, 5, 0, "screw.motherboard.02", "unscrew"),
    ]


def test_a_never_lifted_board_gets_one_implied_instance(tax):
    instances = _d64_like()
    made = implied_instances(instances, _d64_actions(), tax)
    assert [rec.key for rec in made] == ["motherboard.01"]
    rec = made[0]
    assert rec.cls == "motherboard"
    assert rec.desktop == 64
    assert rec.attrs[IMPLIED_ATTR] is True
    assert is_implied(rec)
    # four references: two screws by role, two connectors by socket_host
    assert "referenced by 4 instances" in rec.attrs["note"]
    assert "never operated in the log" in rec.attrs["note"]


def test_the_implied_board_is_installed_in_the_chassis_for_the_whole_run(tax):
    """No action targets it, so the state machine leaves it at its defaults."""
    instances = _d64_like()
    actions = _d64_actions()
    made = implied_instances(instances, actions, tax)
    assert not any(a.target == made[0].key for a in actions)
    assert tax.default_state("motherboard") == "installed"


def test_only_a_class_the_taxonomy_allows_may_ever_be_implied(tax):
    """A referenced class that is not listed stays unresolved, on purpose."""
    assert tax.implied_when_referenced == ["motherboard"]
    instances = _d64_like()
    instances["screw.psu.01"] = InstanceRec(
        "screw.psu.01", 64, "screw", attrs={"role": "psu"})
    made = implied_instances(instances, _d64_actions(), tax)
    assert [rec.cls for rec in made] == ["motherboard"]


def test_a_real_instance_blocks_the_implied_one(tax):
    instances = _d64_like()
    instances["motherboard.01"] = InstanceRec("motherboard.01", 64, "motherboard")
    assert implied_instances(instances, _d64_actions(), tax) == []


def test_a_label_studio_draft_does_not_count_as_the_real_instance(tax):
    """``ls:*`` rows are drafts: they never stand in for a settled instance."""
    instances = _d64_like()
    instances["ls:Motherboard#1"] = InstanceRec("ls:Motherboard#1", 64, "motherboard")
    assert [rec.key for rec in implied_instances(instances, _d64_actions(), tax)] \
        == ["motherboard.01"]


def test_a_draft_is_not_a_reference_either(tax):
    """D66's case: eight drawn screws, no imported sheet - imply nothing."""
    instances = {
        "ls:Motherboard Screw#1": InstanceRec(
            "ls:Motherboard Screw#1", 66, "screw", attrs={"role": "motherboard"}),
        "ls:PSU to Motherboard Connector#1": InstanceRec(
            "ls:PSU to Motherboard Connector#1", 66, "connector",
            socket_host="motherboard"),
    }
    assert implied_instances(instances, [], tax) == []


def test_nothing_is_implied_when_nothing_references_the_class(tax):
    instances = {"chassis.01": InstanceRec("chassis.01", 64, "chassis")}
    assert implied_instances(instances, [], tax) == []


# --------------------------------------------------------------------------- #
# a host_class declaration is a reference too
# --------------------------------------------------------------------------- #
def _latches_only(desktop: int = 64) -> dict[str, InstanceRec]:
    """A desktop whose only claim on a motherboard is that it has board latches."""
    return {
        "chassis.01": InstanceRec("chassis.01", desktop, "chassis"),
        "ram_latch.01": InstanceRec("ram_latch.01", desktop, "ram_latch"),
        "cpu_socket_lever.01": InstanceRec(
            "cpu_socket_lever.01", desktop, "cpu_socket_lever"),
    }


def test_a_board_mounted_latch_references_the_board_it_rides_on(tax):
    made = implied_instances(_latches_only(), [], tax)
    assert [rec.key for rec in made] == ["motherboard.01"]
    assert "referenced by 2 instances" in made[0].attrs["note"]


def test_the_implied_board_then_becomes_the_latches_parent(tax):
    instances = _latches_only()
    for rec in implied_instances(instances, [], tax):
        instances[rec.key] = rec
    infer_relational_fields(instances, tax, [])
    for key in ("ram_latch.01", "cpu_socket_lever.01"):
        assert (instances[key].parent, instances[key].attached) \
            == ("motherboard.01", True), key
    assert unresolved_relations(instances, tax, []) == []


def test_a_latch_that_already_has_a_parent_is_not_a_reference(tax):
    """Only an *unresolved* row is evidence the desktop is missing the part."""
    instances = _latches_only()
    instances["ram_latch.01"].parent = "chassis.01"
    instances["cpu_socket_lever.01"].parent = "chassis.01"
    assert implied_instances(instances, [], tax) == []


def test_a_latch_draft_is_not_a_reference_either(tax):
    instances = {
        "ls:RAM Module Retention Clip#1": InstanceRec(
            "ls:RAM Module Retention Clip#1", 66, "ram_latch"),
    }
    assert implied_instances(instances, [], tax) == []


def test_a_host_on_a_connector_or_screw_class_is_counted_all_the_same(tax):
    """The host check is not an ``elif``: it has to survive the two older ones.

    ``screw`` matches the role branch first and that branch says no -- a psu
    screw names no motherboard. A class that one day declares both must still
    be counted once, not dropped and not twice.
    """
    hosted = replace(tax, host_classes={**tax.host_classes, "screw": "motherboard"})
    instances = {
        "screw.psu.01": InstanceRec("screw.psu.01", 64, "screw", attrs={"role": "psu"}),
    }
    assert referencing_instances(instances, "motherboard", hosted) == ["screw.psu.01"]
    # a screw whose *role* already names the class is still counted exactly once
    instances["screw.motherboard.01"] = InstanceRec(
        "screw.motherboard.01", 64, "screw", attrs={"role": "motherboard"})
    assert referencing_instances(instances, "motherboard", hosted) \
        == ["screw.motherboard.01", "screw.psu.01"]


def test_creating_the_implied_instance_is_idempotent(tax):
    instances = _d64_like()
    actions = _d64_actions()
    made = implied_instances(instances, actions, tax)
    for rec in made:
        instances[rec.key] = rec
    assert implied_instances(instances, actions, tax) == []


def test_the_normal_inference_then_resolves_everything_to_it(tax):
    instances = _d64_like()
    actions = _d64_actions()
    before = unresolved_relations(instances, tax, actions)
    assert len(before) == 4
    assert all(NO_CANDIDATE in line for line in before)

    for rec in implied_instances(instances, actions, tax):
        instances[rec.key] = rec
    infer_relational_fields(instances, tax, actions)

    assert unresolved_relations(instances, tax, actions) == []
    assert instances["connector.atx_24pin.01"].socket_host == "motherboard.01"
    assert instances["screw.motherboard.01"].fastens == "motherboard.01"


def test_an_empty_desktop_implies_nothing(tax):
    assert implied_instances({}, [], tax) == []


# --------------------------------------------------------------------------- #
# stage S1 says what it is and how to get rid of it
# --------------------------------------------------------------------------- #
def test_the_step_table_explains_the_implied_instance(tax):
    instances = _d64_like()
    for rec in implied_instances(instances, _d64_actions(), tax):
        instances[rec.key] = rec
    infer_relational_fields(instances, tax, _d64_actions())

    lines = list(unresolved_issues(instances, tax))
    assert len(lines) == 1
    assert lines[0].startswith("implied instance motherboard.01:")
    assert "never operated in the log" in lines[0]
    assert "keep it" in lines[0] and "delete it in the Instances tab" in lines[0]


def test_an_implied_instance_is_not_reported_as_an_orphan(tax):
    """It is *defined* as the instance no action names; saying so twice is noise."""
    instances = _d64_like()
    for rec in implied_instances(instances, _d64_actions(), tax):
        instances[rec.key] = rec
    lines = list(orphan_issues(instances, _d64_actions()))
    assert not any("motherboard.01" in line for line in lines)


# --------------------------------------------------------------------------- #
# import-logs creates them; D13 is the control
# --------------------------------------------------------------------------- #
def test_import_logs_implies_the_board_of_a_sheet_that_never_lifts_it(env, capsys):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    out = capsys.readouterr().out
    db = open_db(env)
    try:
        rec = db.instances(63).get("motherboard.01")
        assert rec is not None and rec.attrs.get(IMPLIED_ATTR) is True
        assert [o["kind"] for o in db.ops(63, "-")] == [OP_KIND]
        # ... and every motherboard-side connector now points at it
        hosts = {r.socket_host for r in db.instances(63).values() if r.cls == "connector"}
        assert "motherboard" not in hosts
    finally:
        db.close()
    assert "D63" in out and "1 implied instance" in out


def test_import_logs_implies_nothing_for_a_sheet_that_removes_the_board(env):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    db = open_db(env)
    try:
        rec = db.instances(13).get("motherboard.01")
        assert rec is not None
        assert not rec.attrs.get(IMPLIED_ATTR)
        assert db.ops(13, "-") == []
    finally:
        db.close()


def test_the_import_report_names_the_implied_instances(env):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    report = (env["cache"] / "import_logs_issues.md").read_text(encoding="utf-8")
    assert "implied instance motherboard.01" in report


# --------------------------------------------------------------------------- #
# infer-relations --add-implied (off by default)
# --------------------------------------------------------------------------- #
def _strip_implied(env: dict, desktop: int = 63) -> None:
    """Put the database back into the state the old importer left it in."""
    db = open_db(env)
    try:
        instances = db.instances(desktop)
        for key, rec in list(instances.items()):
            if is_implied(rec):
                db.delete_instance(desktop, key)
                continue
            if rec.cls == "connector" and rec.socket_host in instances:
                rec.socket_host = instances[rec.socket_host].cls
                db.upsert_instance(rec)
    finally:
        db.close()


def test_infer_relations_leaves_the_gap_alone_by_default(env):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    _strip_implied(env)
    assert run(env, "infer-relations") == EXIT_OK
    db = open_db(env)
    try:
        assert "motherboard.01" not in db.instances(63)
    finally:
        db.close()


def test_infer_relations_add_implied_fills_the_gap(env, capsys):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    _strip_implied(env)
    capsys.readouterr()
    assert run(env, "infer-relations", "--add-implied") == EXIT_OK
    out = capsys.readouterr().out
    db = open_db(env)
    try:
        rec = db.instances(63).get("motherboard.01")
        assert rec is not None and rec.attrs.get(IMPLIED_ATTR) is True
        assert OP_KIND in [o["kind"] for o in db.ops(63, "-")]
    finally:
        db.close()
    assert "implied instance motherboard.01" in out


def test_a_dry_run_reports_the_implied_instance_but_writes_nothing(env, capsys):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    _strip_implied(env)
    capsys.readouterr()
    assert run(env, "infer-relations", "--add-implied", "--dry-run") == EXIT_OK
    assert "implied instance motherboard.01" in capsys.readouterr().out
    db = open_db(env)
    try:
        assert "motherboard.01" not in db.instances(63)
    finally:
        db.close()
