"""Applying the relational-field heuristic to the database (spec 2.3 / 7.1-7.3).

Three layers are covered here:

* :func:`tda.core.graph_rules.infer_relational_fields` and its companion
  :func:`~tda.core.graph_rules.unresolved_relations` -- how a unique instance of
  a class is counted now that Label Studio drafts share the class names;
* ``import-logs``, which applies the heuristic inside each desktop's
  transaction and lists what it filled in the import report;
* ``infer-relations``, the one-off command that does the same to a database
  that was imported before the heuristic was wired in.

Everything runs against the throw-away ``paths.yaml`` of ``test_cli.py`` (whose
``env`` fixture is re-used verbatim) or against a ``tmp_path`` database, so no
test reads F: or touches the real annotations file.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from graph_scenes import inst
from test_cli import d13_steps, env, open_db, run  # noqa: F401  (re-used fixtures)

from tda.cli import EXIT_ERROR, EXIT_LOCKED, EXIT_OK
from tda.core.db import Db
from tda.core.graph_infer import (
    AMBIGUOUS,
    NO_CANDIDATE,
    UNRESOLVED,
    infer_relational_fields,
    unresolved_kind,
    unresolved_relations,
)
from tda.core.model import ActionRec, FrameKey, InstanceRec
from tda.core.states import needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import state_of
from tda.ui.steps_issues import unresolved_issues

#: D13's four captive CPU-cooler screws and the cooler they belong to.
COOLER = "cpu_cooler.fan.01"
COOLER_SCREWS = tuple(f"screw.cpu_cooler.{n:02d}" for n in (1, 2, 3, 4))
#: The step D13's sheet removes the cooler at.
COOLER_STEP = 13
#: ``op_log`` is scoped per (desktop, view); a relational field belongs to none.
OP_VIEW = "-"


@pytest.fixture
def tax():
    return load_taxonomy()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def strip_relations(db: Db, desktop: int) -> None:
    """Undo the heuristic, leaving the desktop as the old importer wrote it.

    ``infer-relations`` exists for a database that already exists, so the tests
    that exercise it first put one back into that state: no ``fastens``, no
    captive ``parent``, no latch ``of``, and ``socket_host`` holding a bare
    class name again.
    """
    instances = db.instances(desktop)
    for rec in instances.values():
        rec.fastens = None
        rec.parent = None
        rec.attached = False
        rec.attrs.pop("of", None)
        host = instances.get(rec.socket_host or "")
        if rec.cls == "connector" and host is not None:
            rec.socket_host = host.cls
        db.upsert_instance(rec)


def imported(env: dict) -> None:
    """``load-index`` + ``import-logs`` against the temporary workspace."""
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK


# --------------------------------------------------------------------------- #
# 1. the rule layer: Label Studio drafts must not break class uniqueness
# --------------------------------------------------------------------------- #
def _one_board_plus_a_draft() -> dict[str, InstanceRec]:
    """One real motherboard, one Label Studio draft of the same class."""
    recs = [
        inst("chassis", "chassis"),
        inst("motherboard.01", "motherboard"),
        inst("ls:Motherboard#1", "motherboard"),
        inst("connector.01", "connector", socket_host="motherboard"),
    ]
    return {r.key: r for r in recs}


def test_a_provisional_draft_does_not_make_a_class_ambiguous(tax):
    instances = _one_board_plus_a_draft()
    infer_relational_fields(instances, tax)
    assert instances["connector.01"].socket_host == "motherboard.01"


def test_a_provisional_draft_is_never_a_subject_of_the_heuristic(tax):
    instances = _one_board_plus_a_draft()
    instances["ls:Screw#1"] = inst(
        "ls:Screw#1", "screw", attrs={"role": "motherboard", "captive": True}
    )
    infer_relational_fields(instances, tax)
    draft = instances["ls:Screw#1"]
    assert (draft.fastens, draft.parent, draft.attached) == (None, None, False)


def test_two_real_instances_stay_unresolved_rather_than_guessed(tax):
    instances = _one_board_plus_a_draft()
    instances["motherboard.02"] = inst("motherboard.02", "motherboard")
    assert infer_relational_fields(instances, tax) == []
    assert instances["connector.01"].socket_host == "motherboard"
    lines = unresolved_relations(instances, tax)
    assert any("connector.01.socket_host" in line for line in lines)
    assert all(line.startswith(f"{UNRESOLVED} (") for line in lines)
    assert all(unresolved_kind(line) == AMBIGUOUS for line in lines)


def test_a_captive_screw_with_nothing_to_hang_on_is_unresolved(tax):
    instances = {
        "chassis": inst("chassis", "chassis"),
        "screw.cpu_cooler.01": inst(
            "screw.cpu_cooler.01", "screw", attrs={"role": "cpu_cooler", "captive": True}
        ),
    }
    infer_relational_fields(instances, tax)  # there is no cooler to point at
    lines = unresolved_relations(instances, tax)
    assert any("screw.cpu_cooler.01" in line and "captive" in line for line in lines)


def test_unresolved_is_silent_once_everything_is_filled(tax):
    instances = _one_board_plus_a_draft()
    infer_relational_fields(instances, tax)
    assert unresolved_relations(instances, tax) == []


# --------------------------------------------------------------------------- #
# 1b. two coolers: physical necessity in time breaks the tie
# --------------------------------------------------------------------------- #
FAN = "cpu_cooler.fan.01"
HEATSINK = "cpu_cooler.heatsink.01"
#: The machine the eight real two-cooler desktops all look like: the fan comes
#: off at step 10, the heatsink at 15.
FAN_STEP, HEATSINK_STEP = 10, 15


def _two_cooler_desktop() -> dict[str, InstanceRec]:
    """Five cooler screws and two candidates for every one of them."""
    recs = [
        inst("chassis", "chassis"),
        inst(FAN, "cpu_cooler", attrs={"kind": "fan"}, raw_names=["CPU fan"]),
        inst(HEATSINK, "cpu_cooler", attrs={"kind": "heatsink"}, raw_names=["Heatsink"]),
    ]
    for n, (step, name) in enumerate(
        [(6, "CPU fan screw 1"), (9, "CPU fan screw 2"),
         (11, "Heatsink screw 1"), (14, "Heatsink screw 2"),
         (HEATSINK_STEP + 3, "Spare screw")],
        start=1,
    ):
        recs.append(inst(
            f"screw.cpu_cooler.{n:02d}", "screw",
            attrs={"role": "cpu_cooler", "captive": True, "unscrewed_at": step},
            raw_names=[name],
        ))
    return {r.key: r for r in recs}


def _two_cooler_actions(instances: dict[str, InstanceRec]) -> list[ActionRec]:
    """One ``unscrew`` per screw at its own step, then the two removals."""
    out = [
        ActionRec(900, int(rec.attrs["unscrewed_at"]), 0, key, "unscrew")
        for key, rec in sorted(instances.items()) if rec.cls == "screw"
    ]
    out.append(ActionRec(900, FAN_STEP, 0, FAN, "remove"))
    out.append(ActionRec(900, HEATSINK_STEP, 0, HEATSINK, "remove"))
    return out


def test_a_screw_fastens_the_part_that_leaves_soonest_after_it(tax):
    instances = _two_cooler_desktop()
    infer_relational_fields(instances, tax, _two_cooler_actions(instances))
    # unscrewed at 6 and 9, before the fan goes at 10
    assert instances["screw.cpu_cooler.01"].fastens == FAN
    assert instances["screw.cpu_cooler.02"].fastens == FAN
    # unscrewed at 11 and 14: the fan is already gone, the heatsink goes at 15
    assert instances["screw.cpu_cooler.03"].fastens == HEATSINK
    assert instances["screw.cpu_cooler.04"].fastens == HEATSINK


def test_the_captive_parent_follows_the_resolved_fastens(tax):
    instances = _two_cooler_desktop()
    infer_relational_fields(instances, tax, _two_cooler_actions(instances))
    for key, part in (("screw.cpu_cooler.01", FAN), ("screw.cpu_cooler.03", HEATSINK)):
        rec = instances[key]
        assert (rec.parent, rec.attached) == (part, True), key


def test_a_screw_undone_after_both_removals_stays_unresolved(tax):
    instances = _two_cooler_desktop()
    actions = _two_cooler_actions(instances)
    infer_relational_fields(instances, tax, actions)
    late = instances["screw.cpu_cooler.05"]  # unscrewed at 18, after both parts left
    assert late.fastens is None
    assert late.parent is None
    lines = unresolved_relations(instances, tax, actions)
    assert any("screw.cpu_cooler.05" in line and "is removed after" in line
               for line in lines)
    assert all(unresolved_kind(line) == AMBIGUOUS for line in lines)


def test_the_clock_and_the_sheet_name_disagreeing_is_left_to_the_human(tax):
    instances = _two_cooler_desktop()
    # the sheet calls it a heatsink screw, but it is undone before the fan goes
    instances["screw.cpu_cooler.01"].raw_names = ["Heatsink screw 9"]
    actions = _two_cooler_actions(instances)
    infer_relational_fields(instances, tax, actions)
    assert instances["screw.cpu_cooler.01"].fastens is None
    lines = [
        line for line in unresolved_relations(instances, tax, actions)
        if "screw.cpu_cooler.01." in line
    ]
    assert len(lines) == 1
    assert "the two heuristics disagree" in lines[0]
    assert HEATSINK in lines[0] and FAN in lines[0]


def test_two_parts_removed_on_the_same_step_are_not_a_tie_break(tax):
    instances = _two_cooler_desktop()
    actions = [a for a in _two_cooler_actions(instances) if a.verb != "remove"]
    actions.append(ActionRec(900, FAN_STEP, 0, FAN, "remove"))
    actions.append(ActionRec(900, FAN_STEP, 1, HEATSINK, "remove"))
    infer_relational_fields(instances, tax, actions)
    assert instances["screw.cpu_cooler.01"].fastens is None
    assert any("are both removed at step" in line
               for line in unresolved_relations(instances, tax, actions))


def test_without_actions_the_tie_break_is_simply_not_attempted(tax):
    """The keyword is additive: leaving it out must change nothing else."""
    instances = _two_cooler_desktop()
    assert infer_relational_fields(instances, tax) == []
    lines = unresolved_relations(instances, tax)
    assert any("nothing ranks them" in line for line in lines)


def test_a_role_with_no_part_at_all_is_a_different_kind_of_unresolved(tax):
    instances = {
        "chassis": inst("chassis", "chassis"),
        "screw.card.01": inst("screw.card.01", "screw", attrs={"role": "card"}),
    }
    lines = unresolved_relations(instances, tax, [])
    assert lines and all(unresolved_kind(line) == NO_CANDIDATE for line in lines)
    assert any("names no part of this desktop" in line for line in lines)


def test_a_socket_host_class_with_no_instance_asks_for_one_to_be_created(tax):
    instances = {
        "chassis": inst("chassis", "chassis"),
        "connector.01": inst("connector.01", "connector", socket_host="motherboard"),
    }
    lines = unresolved_relations(instances, tax)
    assert len(lines) == 1
    assert unresolved_kind(lines[0]) == NO_CANDIDATE
    assert "has no instance on this desktop" in lines[0]

    issues = list(unresolved_issues(instances, tax))
    assert issues == [
        "unresolved socket host: class 'motherboard' has no instance on this "
        "desktop - add the instance in the Instances tab or leave it unresolved"
    ]


# --------------------------------------------------------------------------- #
# 2. import-logs applies it
# --------------------------------------------------------------------------- #
def test_import_logs_fills_the_captive_cooler_screws(env):
    imported(env)
    db = open_db(env)
    try:
        instances = db.instances(13)
        for key in COOLER_SCREWS:
            rec = instances[key]
            assert (rec.fastens, rec.parent, rec.attached) == (COOLER, COOLER, True), key
    finally:
        db.close()


def test_import_logs_resolves_socket_hosts_to_the_motherboard_instance(env):
    imported(env)
    db = open_db(env)
    try:
        assert db.instances(13)["connector.03"].socket_host == "motherboard.01"
    finally:
        db.close()


def test_the_cooler_screws_leave_the_chassis_with_the_cooler(env, tax):
    """The cascade of spec 3.3, read through the API the annotator's session uses."""
    imported(env)
    db = open_db(env)
    try:
        before = state_of(db, tax, 13, COOLER_STEP - 1)
        after = state_of(db, tax, 13, COOLER_STEP)
        later = state_of(db, tax, 13, COOLER_STEP + 1)
        assert before[COOLER].state != "removed"
        for key in COOLER_SCREWS:
            assert (before[key].state, before[key].placement) == ("loosened", "in_chassis"), key
            assert (after[key].state, after[key].placement) == ("removed", "on_bench"), key
            assert later[key].state == "removed", key
    finally:
        db.close()


def test_a_screw_that_left_inside_the_cooler_is_never_asked_for_again(env, tax):
    """User decision C7: 子零件随父零件一起消失 -- no mask, no bench box, no row."""
    from tda.core.truth_inputs import instances_of

    imported(env)
    db = open_db(env)
    try:
        instances = instances_of(db, 13)
        before = needs_geom(instances, state_of(db, tax, 13, COOLER_STEP - 1), tax)
        assert all(before[key] == "mask" for key in COOLER_SCREWS)  # still in the chassis
        for step in (COOLER_STEP, COOLER_STEP + 1, COOLER_STEP + 5):
            geom = needs_geom(instances, state_of(db, tax, 13, step), tax)
            assert all(key not in geom for key in COOLER_SCREWS), step
            assert COOLER not in geom or geom[COOLER] == "box"  # the parent still is
    finally:
        db.close()


def test_the_compiler_reports_no_missing_shape_for_a_screw_inside_its_parent(env, tax):
    """The frame the annotator actually opens must carry no question about them."""
    from tda.core.truth_inputs import gather

    imported(env)
    db = open_db(env)
    try:
        for step in (COOLER_STEP, COOLER_STEP + 1):
            needs = gather(db, tax, FrameKey(13, step, "scan")).needs
            assert all(key not in needs for key in COOLER_SCREWS), step
    finally:
        db.close()


def test_the_import_report_lists_what_the_heuristic_filled(env):
    imported(env)
    report = (env["cache"] / "import_logs_issues.md").read_text(encoding="utf-8")
    assert "inferred relational fields" in report
    assert "confirm in S1" in report
    assert "screw.cpu_cooler.01.parent = cpu_cooler.fan.01" in report


# --------------------------------------------------------------------------- #
# 3. the infer-relations command
# --------------------------------------------------------------------------- #
def test_infer_relations_fills_a_database_that_predates_the_heuristic(env, capsys):
    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
        assert db.instances(13)[COOLER_SCREWS[0]].parent is None
    finally:
        db.close()

    capsys.readouterr()
    assert run(env, "infer-relations", "--desktops", "13") == EXIT_OK
    assert "screw.cpu_cooler.01.fastens = cpu_cooler.fan.01" in capsys.readouterr().out

    db = open_db(env)
    try:
        rec = db.instances(13)[COOLER_SCREWS[0]]
        assert (rec.fastens, rec.parent, rec.attached) == (COOLER, COOLER, True)
        assert db.instances(13)["connector.03"].socket_host == "motherboard.01"
    finally:
        db.close()


def test_infer_relations_is_idempotent(env, capsys):
    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
    finally:
        db.close()
    assert run(env, "infer-relations", "--desktops", "13") == EXIT_OK
    db = open_db(env)
    try:
        first = len(db.ops(13, OP_VIEW))
    finally:
        db.close()
    assert first

    capsys.readouterr()
    assert run(env, "infer-relations", "--desktops", "13") == EXIT_OK
    assert "0 fills" in capsys.readouterr().out
    db = open_db(env)
    try:
        assert len(db.ops(13, OP_VIEW)) == first  # a second run writes nothing
    finally:
        db.close()


def test_infer_relations_dry_run_writes_nothing(env, capsys):
    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
    finally:
        db.close()
    before = Path(env["db_path"]).read_bytes()

    capsys.readouterr()
    assert run(env, "infer-relations", "--desktops", "13", "--dry-run") == EXIT_OK
    out = capsys.readouterr().out
    assert "screw.cpu_cooler.01.parent = cpu_cooler.fan.01" in out
    assert "dry run" in out.lower()
    assert Path(env["db_path"]).read_bytes() == before
    assert not list(Path(env["cfg"]["backup_dir"]).glob("tda_*.sqlite"))


def test_infer_relations_backs_the_database_up_first(env, capsys):
    imported(env)
    capsys.readouterr()
    assert run(env, "infer-relations", "--desktops", "13") == EXIT_OK
    assert "backed the database up first" in capsys.readouterr().out
    assert list(Path(env["cfg"]["backup_dir"]).glob("tda_*.sqlite"))


def test_infer_relations_never_overwrites_a_human_value(env):
    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
        rec = db.instances(13)[COOLER_SCREWS[0]]
        rec.fastens = "motherboard.01"  # a deliberate (if odd) human correction
        db.upsert_instance(rec)
    finally:
        db.close()

    assert run(env, "infer-relations", "--desktops", "13") == EXIT_OK
    db = open_db(env)
    try:
        rec = db.instances(13)[COOLER_SCREWS[0]]
        assert rec.fastens == "motherboard.01"
        assert rec.parent == "motherboard.01"  # the captive parent follows `fastens`
    finally:
        db.close()


def test_infer_relations_ignores_label_studio_drafts(env):
    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
        db.upsert_instance(InstanceRec(key="ls:Motherboard#1", desktop=13, cls="motherboard"))
    finally:
        db.close()

    assert run(env, "infer-relations", "--desktops", "13") == EXIT_OK
    db = open_db(env)
    try:
        instances = db.instances(13)
        assert instances["connector.03"].socket_host == "motherboard.01"
        draft = instances["ls:Motherboard#1"]
        assert (draft.parent, draft.fastens) == (None, None)
    finally:
        db.close()


def test_infer_relations_logs_every_change_with_its_old_value(env):
    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
    finally:
        db.close()
    assert run(env, "infer-relations", "--desktops", "13") == EXIT_OK

    db = open_db(env)
    try:
        ops = db.ops(13, OP_VIEW)
        assert ops and all(op["kind"] == "infer_relations" for op in ops)
        screw = [op for op in ops if op["payload"]["instance"] == COOLER_SCREWS[0]]
        assert len(screw) == 1
        assert screw[0]["payload"]["fields"]["fastens"] == COOLER
        assert screw[0]["inverse"]["fields"]["fastens"] is None
    finally:
        db.close()


def test_infer_relations_regenerates_the_stored_auto_events(env):
    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
        db.replace_events(13, [], auto_only=True)
    finally:
        db.close()
    assert run(env, "infer-relations", "--desktops", "13") == EXIT_OK

    db = open_db(env)
    try:
        assert any(
            e.target == COOLER_SCREWS[0] and e.step == COOLER_STEP and e.new == "removed"
            for e in db.events(13)
        )
    finally:
        db.close()


def test_infer_relations_refuses_while_another_annotator_holds_the_lock(env, capsys):
    imported(env)
    db = open_db(env)
    try:
        db.acquire_lock("chang")
    finally:
        db.close()

    capsys.readouterr()
    assert run(env, "infer-relations") == EXIT_LOCKED
    assert "chang" in capsys.readouterr().out
    # the refused run must not have taken the lock away from its holder
    assert "chang" in Path(env["db_path"] + ".lock").read_text(encoding="utf-8")


def test_one_failing_desktop_does_not_abort_the_rest(env, capsys, monkeypatch):
    import tda.cli_relations as CR

    real = CR.infer_relational_fields

    def flaky(instances, tax, actions=None):
        if any(rec.desktop == 13 for rec in instances.values()):
            raise RuntimeError("boom")
        return real(instances, tax, actions)

    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
        strip_relations(db, 77)
    finally:
        db.close()

    monkeypatch.setattr(CR, "infer_relational_fields", flaky)
    capsys.readouterr()
    assert run(env, "infer-relations") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "D13" in out and "boom" in out
    db = open_db(env)
    try:
        assert db.instances(13)[COOLER_SCREWS[0]].parent is None  # rolled back
        assert db.instances(77)["screw.motherboard.01"].fastens == "motherboard.01"
    finally:
        db.close()


def test_infer_relations_desktop_filter(env):
    imported(env)
    db = open_db(env)
    try:
        strip_relations(db, 13)
        strip_relations(db, 77)
    finally:
        db.close()

    assert run(env, "infer-relations", "--desktops", "77") == EXIT_OK
    db = open_db(env)
    try:
        assert db.instances(13)[COOLER_SCREWS[0]].parent is None
        assert db.instances(77)["screw.motherboard.01"].fastens == "motherboard.01"
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 4. what is left over becomes an S1 question
# --------------------------------------------------------------------------- #
def test_an_unresolved_socket_host_is_an_issue(tax):
    instances = _one_board_plus_a_draft()
    instances["motherboard.02"] = inst("motherboard.02", "motherboard")
    lines = list(unresolved_issues(instances, tax))
    assert any("unresolved socket host" in line and "connector.01" in line for line in lines)


def test_a_captive_screw_without_a_parent_is_an_issue(tax):
    instances = {
        "screw.cpu_cooler.01": inst(
            "screw.cpu_cooler.01", "screw", attrs={"role": "cpu_cooler", "captive": True}
        ),
    }
    lines = list(unresolved_issues(instances, tax))
    assert any("captive screw without parent" in line for line in lines)
    instances["screw.cpu_cooler.01"].parent = "cpu_cooler.fan.01"
    assert list(unresolved_issues(instances, tax)) == []


def test_the_step_table_asks_about_both_after_the_existing_questions(tmp_db_path, tax):
    """A desktop the heuristic never reached must put both questions on screen."""
    from steps_fixtures import seeded_db
    from tda.ui.steps_model import StepTableData

    db = seeded_db(tmp_db_path, tax, desktops=(13,))
    try:
        strip_relations(db, 13)
        data = StepTableData.load(db, 13, tax)
    finally:
        db.close()
    new = [
        i for i, line in enumerate(data.orphans)
        if "unresolved socket host" in line or "captive screw without parent" in line
    ]
    assert new
    # the new kinds sit at the very end, after every pre-existing one
    assert new == list(range(len(data.orphans) - len(new), len(data.orphans)))
    text = "\n".join(data.orphans)
    assert "unresolved socket host" in text and "captive screw without parent" in text
