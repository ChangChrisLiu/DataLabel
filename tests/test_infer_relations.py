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
from tda.core.graph_rules import infer_relational_fields, unresolved_relations
from tda.core.model import InstanceRec
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
    assert all(line.startswith("unresolved:") for line in lines)


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

    def flaky(instances, tax):
        if any(rec.desktop == 13 for rec in instances.values()):
            raise RuntimeError("boom")
        return real(instances, tax)

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
