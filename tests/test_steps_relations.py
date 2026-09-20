"""Staging constraint-graph edits with the rest of stage S1.

The Relations tab edits the same session the step table does: nothing reaches
the database until ``Apply``, and ``Apply`` is one transaction for the steps,
the actions, the instances, the pose re-cut *and* the edges -- so a failure
anywhere rolls all of it back. ``graph_version`` is re-stamped only when the
edge set really changed, and the op log records what was written.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from steps_fixtures import seeded_db

from tda.cli_graph import constraints_into_db
from tda.core.db import Db
from tda.core.graph import edges_from_db, graph_version
from tda.core.graph_edit import MANUAL, OVERRIDE, RULE
from tda.core.model import VIEWS, FrameKey, ShapeKeyframe, ShapePart
from tda.core.taxonomy import load_taxonomy
from tda.ui.steps_model import EditError, StepTableData

DESKTOP = 13
#: The first removal of the D13 log; the drive is still in the chassis and
#: everything else is still installed, so a `blocked_by` here has a choice.
FIRST_REMOVE = 10
#: ``remove cpu_cooler.fan.01``, by which time the drive is out of the machine.
COOLER_REMOVE = 13
DRIVE = "storage_drive.ssd.01"
BLOCK = ("blocked_by", "psu.01")
#: A class a new instance can be made of that ``remove`` applies to.
SPARE_CLASS = "expansion_card"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_path: Path, tax) -> Db:
    conn = seeded_db(tmp_path / "tda.sqlite", tax)
    constraints_into_db(conn, tax, {DESKTOP, 63})
    yield conn
    conn.close()


@pytest.fixture
def data(db, tax) -> StepTableData:
    return StepTableData.load(db, DESKTOP, tax)


def triples(edges) -> set[tuple[str, str, str]]:
    return {(e.type, e.target, e.blocker) for e in edges}


def add_block(data) -> None:
    data.relations.add(DRIVE, BLOCK[0], BLOCK[1], mode="physical_path",
                       note="机箱前部被电源挡住")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_the_session_loads_the_stored_edges(data):
    assert data.relations.rows
    assert {e.source for e in data.relations.rows} == {RULE}
    assert not data.relations.dirty


def test_the_pickers_offer_settled_instances_only(data):
    keys = data.relations.instance_keys()
    assert DRIVE in keys
    assert not [k for k in keys if k.startswith("ls:")]


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
def test_an_added_edge_is_staged_and_not_written(data, db):
    add_block(data)
    assert data.relations.dirty
    assert (BLOCK[0], DRIVE, BLOCK[1]) in triples(data.relations.rows)
    assert (BLOCK[0], DRIVE, BLOCK[1]) not in triples(edges_from_db(db, DESKTOP))


def test_a_refused_edge_raises_the_panel_error_type(data):
    with pytest.raises(EditError) as excinfo:
        data.relations.add(DRIVE, "blocked_by", DRIVE, mode="physical_path")
    assert "/" in str(excinfo.value)
    assert not data.relations.dirty


def test_a_rule_edge_cannot_be_removed_by_hand(data):
    rule = next(e for e in data.relations.rows if e.source == RULE)
    with pytest.raises(EditError):
        data.relations.remove(rule.target, rule.type, rule.blocker)


def test_a_staged_edge_may_name_an_instance_s1_has_only_just_created(data, db):
    key = data.change_target(FIRST_REMOVE, cls=SPARE_CLASS, attrs={})
    data.relations.add(DRIVE, "blocked_by", key, mode="physical_path")
    data.save(db)
    assert ("blocked_by", DRIVE, key) in triples(edges_from_db(db, DESKTOP))


def test_reverting_throws_the_staged_edges_away(data, db):
    add_block(data)
    data.relations.reload(db)
    assert not data.relations.dirty
    assert (BLOCK[0], DRIVE, BLOCK[1]) not in triples(data.relations.rows)


# --------------------------------------------------------------------------- #
# applying
# --------------------------------------------------------------------------- #
def test_apply_writes_the_edge_with_its_own_source(data, db):
    add_block(data)
    data.save(db)
    edge = next(e for e in edges_from_db(db, DESKTOP)
                if (e.type, e.target, e.blocker) == (BLOCK[0], DRIVE, BLOCK[1]))
    assert edge.source == MANUAL
    assert edge.mode == "physical_path"
    assert edge.reason == "机箱前部被电源挡住"
    assert not data.relations.dirty


def test_apply_restamps_the_graph_version(data, db):
    before = graph_version(db, DESKTOP)
    add_block(data)
    data.save(db)
    assert graph_version(db, DESKTOP) != before
    assert db.get_desktop(DESKTOP).get("graph_version") == graph_version(db, DESKTOP)


def test_an_apply_without_a_relation_edit_leaves_the_stamp_alone(data, db):
    from tda import pipeline as P

    P.merge_desktop_meta(db, DESKTOP, {"graph_version": "not-a-real-hash"})
    data.apply_edit(FIRST_REMOVE, "difficulty", 3)
    data.save(db)
    assert db.get_desktop(DESKTOP)["graph_version"] == "not-a-real-hash"


def test_apply_logs_the_operation(data, db):
    add_block(data)
    data.save(db)
    ops = [op for op in db.ops(DESKTOP, "-") if op["kind"] == "relations"]
    assert len(ops) == 1
    assert ops[0]["payload"]["added"] == 1


def test_removing_a_manual_edge_deletes_its_row(data, db):
    add_block(data)
    data.save(db)
    data.relations.remove(DRIVE, *BLOCK)
    data.save(db)
    assert (BLOCK[0], DRIVE, BLOCK[1]) not in triples(edges_from_db(db, DESKTOP))


def test_a_decision_is_written_as_an_override(data, db):
    rule = next(e for e in data.relations.rows if e.source == RULE)
    data.relations.decide(rule.target, rule.type, rule.blocker, "rejected")
    data.save(db)
    edge = next(e for e in edges_from_db(db, DESKTOP)
                if (e.type, e.target, e.blocker) == (rule.type, rule.target, rule.blocker))
    assert (edge.source, edge.status) == (OVERRIDE, "rejected")


# --------------------------------------------------------------------------- #
# one transaction, with everything else S1 writes
# --------------------------------------------------------------------------- #
def test_the_steps_and_the_edges_land_together(data, db):
    add_block(data)
    data.apply_edit(FIRST_REMOVE, "difficulty", 4)
    data.save(db)
    assert (BLOCK[0], DRIVE, BLOCK[1]) in triples(edges_from_db(db, DESKTOP))
    assert [a for a in db.actions(DESKTOP) if a.step == FIRST_REMOVE][0].difficulty == 4


N_STEPS = 42
CUT = 20


def _seed_views(db: Db, desktop: int = DESKTOP) -> None:
    """One frame row and one pose segment per view, plus a shape to be moved."""
    for view in VIEWS:
        for step in range(1, N_STEPS + 1):
            db.upsert_frame(FrameKey(desktop, step, view), f"s{step}.jpg",
                            {"hw": [64, 64]}, None)
        db.set_pose_segment(desktop, view, 1, 1, N_STEPS, N_STEPS, None, None)
    db.add_keyframe(ShapeKeyframe(
        id=None, instance="chassis", desktop=desktop, view="scan", pose_segment=1,
        anchor_step=N_STEPS, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", {"size": [64, 64], "counts": "0 8 4088"})]))


def test_an_apply_that_also_recuts_the_pose_segments(data, db):
    """B1: a step that becomes a ``reorient`` moves a boundary in every view."""
    _seed_views(db)
    add_block(data)
    data.apply_edit(CUT, "step_type", "reorient")

    data.save(db)

    assert data.recut == {view: 2 for view in VIEWS}
    assert (BLOCK[0], DRIVE, BLOCK[1]) in triples(edges_from_db(db, DESKTOP))


def test_a_failing_recut_rolls_the_edges_back_too(data, db, monkeypatch):
    _seed_views(db)
    add_block(data)
    data.apply_edit(CUT, "step_type", "reorient")
    before_steps = {s.step: s.step_type for s in db.steps(DESKTOP)}
    before_version = graph_version(db, DESKTOP)

    def explode(*_args, **_kwargs):
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(type(db), "apply_recut", explode)
    with pytest.raises(RuntimeError):
        data.save(db)
    monkeypatch.undo()

    assert (BLOCK[0], DRIVE, BLOCK[1]) not in triples(edges_from_db(db, DESKTOP))
    assert {s.step: s.step_type for s in db.steps(DESKTOP)} == before_steps
    assert graph_version(db, DESKTOP) == before_version
    assert data.relations.dirty, "the staged edit is still staged"


# --------------------------------------------------------------------------- #
# no dangling edge can ever be applied
# --------------------------------------------------------------------------- #
def deletable_key(data) -> str:
    """An instance nothing else in the session depends on."""
    key = data.change_target(FIRST_REMOVE, cls=SPARE_CLASS, attrs={})
    data.change_target(FIRST_REMOVE, target=DRIVE)   # free it again
    return key


def test_deleting_an_instance_a_staged_edge_names_is_refused(data, db):
    key = deletable_key(data)
    data.relations.add(DRIVE, "blocked_by", key, mode="physical_path")
    with pytest.raises(EditError) as excinfo:
        data.delete_instance(db, key)
    assert "blocked_by" in str(excinfo.value)
    assert key in data.instances


def test_deleting_an_instance_a_stored_edge_names_is_refused(data, db):
    key = deletable_key(data)
    data.relations.add(DRIVE, "blocked_by", key, mode="physical_path")
    data.save(db)
    with pytest.raises(EditError):
        data.delete_instance(db, key)


def test_an_instance_no_edge_names_still_deletes(data, db):
    key = deletable_key(data)
    data.delete_instance(db, key)
    assert key not in data.instances


def test_a_derived_edge_goes_with_the_instance_it_names(data, db):
    """A rule edge is a reading of the instance table, not somebody's decision.

    Since the S1 Apply derives them, every screw now has one, and refusing the
    delete over it would make "delete an instance nothing points at" impossible.
    """
    key = data.change_target(FIRST_REMOVE, cls="screw", attrs={"role": "motherboard"},
                             verb="unscrew")
    data.change_target(FIRST_REMOVE, target=DRIVE, verb="remove")
    data.apply_instance_edit(key, "fastens", "motherboard.01")
    data.save(db)
    assert [e for e in edges_from_db(db, DESKTOP) if key in (e.target, e.blocker)]

    data.delete_instance(db, key)

    assert key not in data.instances
    assert not [e for e in edges_from_db(db, DESKTOP) if key in (e.target, e.blocker)]


# --------------------------------------------------------------------------- #
# the violations list (the same replay `constraints --validate` prints)
# --------------------------------------------------------------------------- #
def fail_the_first_removal(data) -> None:
    data.apply_edit(FIRST_REMOVE, "result", "failed")
    data.apply_edit(FIRST_REMOVE, "failure_reason", "blocked_by_part")


def test_a_staged_failed_attempt_shows_up_as_a_hint(data):
    assert data.relations.violations() == []
    fail_the_first_removal(data)
    found = data.relations.violations()
    assert [(v.step, v.kind) for v in found] == [(FIRST_REMOVE, "hint")]


def test_the_right_blocked_by_clears_the_hint_before_apply(data, db):
    fail_the_first_removal(data)
    add_block(data)
    assert data.relations.violations() == []
    data.save(db)
    assert data.relations.violations() == []


def test_the_hint_stays_cleared_after_a_constraints_rerun(data, db, tax):
    fail_the_first_removal(data)
    add_block(data)
    data.save(db)
    run = constraints_into_db(db, tax, {DESKTOP}, validate=True)
    assert run.runs[0].violations == []
    assert (BLOCK[0], DRIVE, BLOCK[1]) in triples(edges_from_db(db, DESKTOP))


# --------------------------------------------------------------------------- #
# the S1 Apply re-derives this desktop's rule edges (round 1)
# --------------------------------------------------------------------------- #
SCREW = "screw.motherboard.01"
FASTENED = ("fastened_by", "motherboard.01", SCREW)


def test_the_staged_view_shows_the_rules_reading_the_staged_instances(data):
    assert FASTENED in triples(data.relations.rows)

    data.apply_instance_edit(SCREW, "fastens", "")

    assert FASTENED not in triples(data.relations.rows), \
        "the tab must not describe the instance table the annotator has just left"


def test_apply_writes_the_rederived_rule_edges(data, db):
    data.apply_instance_edit(SCREW, "fastens", "")
    data.save(db)
    assert FASTENED not in triples(edges_from_db(db, DESKTOP))


def test_a_new_relational_field_derives_its_edge_on_apply(data, db):
    data.apply_instance_edit(SCREW, "fastens", "cpu_cooler.fan.01")
    fresh = ("fastened_by", "cpu_cooler.fan.01", SCREW)
    assert fresh in triples(data.relations.rows)
    data.save(db)
    assert fresh in triples(edges_from_db(db, DESKTOP))


def test_a_decision_is_orphaned_when_its_rule_edge_goes(data, db):
    data.relations.decide("motherboard.01", "fastened_by", SCREW, "accepted")
    data.save(db)
    data.apply_instance_edit(SCREW, "fastens", "")

    staged = next(e for e in data.relations.rows
                  if (e.type, e.target, e.blocker) == FASTENED)
    assert staged.status == "accepted_orphan"
    data.save(db)
    stored = next(e for e in edges_from_db(db, DESKTOP)
                  if (e.type, e.target, e.blocker) == FASTENED)
    assert stored.status == "accepted_orphan"


def test_an_orphan_can_be_kept_as_a_manual_edge_through_the_session(data, db):
    data.relations.decide("motherboard.01", "fastened_by", SCREW, "accepted")
    data.save(db)
    data.apply_instance_edit(SCREW, "fastens", "")
    data.save(db)

    data.relations.adopt("motherboard.01", "fastened_by", SCREW, note="我看过，确实拧着")
    data.save(db)

    stored = next(e for e in edges_from_db(db, DESKTOP)
                  if (e.type, e.target, e.blocker) == FASTENED)
    assert (stored.source, stored.status) == (MANUAL, "accepted")
    assert stored.reason == "我看过，确实拧着"


def test_an_orphan_can_be_cleared_through_the_session(data, db):
    data.relations.decide("motherboard.01", "fastened_by", SCREW, "rejected")
    data.save(db)
    data.apply_instance_edit(SCREW, "fastens", "")
    data.save(db)

    data.relations.drop("motherboard.01", "fastened_by", SCREW)
    data.save(db)

    assert FASTENED not in triples(edges_from_db(db, DESKTOP))


def test_apply_restamps_the_cycle_count_with_the_version(data, db):
    add_block(data)
    data.save(db)
    meta = db.get_desktop(DESKTOP)
    assert meta["graph_cycles"] == 0
    assert meta["graph_edges"] == len(
        [e for e in edges_from_db(db, DESKTOP) if e.status == "proposed"]) + 1


def test_a_failing_derivation_rolls_the_whole_apply_back(data, db, monkeypatch):
    add_block(data)
    data.apply_edit(FIRST_REMOVE, "difficulty", 5)
    before = graph_version(db, DESKTOP)

    def explode(*_a, **_kw):
        raise RuntimeError("the rules blew up")

    monkeypatch.setattr("tda.ui.steps_relations.derive_edges", explode)
    with pytest.raises(RuntimeError):
        data.save(db)
    monkeypatch.undo()

    assert graph_version(db, DESKTOP) == before
    assert (BLOCK[0], DRIVE, BLOCK[1]) not in triples(edges_from_db(db, DESKTOP))
    assert [a for a in db.actions(DESKTOP) if a.step == FIRST_REMOVE][0].difficulty != 5


def test_the_violations_follow_the_rederived_rule_edges(data, db):
    """Clearing a `fastens` removes the constraint a breach was about."""
    data.apply_edit(FIRST_REMOVE, "target", "motherboard.01")
    data.apply_edit(FIRST_REMOVE, "verb", "remove")
    assert [v.kind for v in data.relations.violations()].count("breach") > 0
    for key in [k for k, rec in data.instances.items()
                if rec.cls == "screw" and rec.fastens == "motherboard.01"]:
        data.apply_instance_edit(key, "fastens", "")
    assert "fastened_by" not in " ".join(v.text for v in data.relations.violations())


def test_the_session_reports_the_cycles_of_the_staged_graph(data):
    assert data.relations.cycles() == []


def test_a_blocker_that_is_already_gone_does_not_clear_the_hint(data):
    """By step 13 the drive is out, so it explains nothing about the cooler."""
    data.apply_edit(COOLER_REMOVE, "result", "failed")
    data.relations.add("cpu_cooler.fan.01", "blocked_by", DRIVE, mode="physical_path")
    hints = [v for v in data.relations.violations() if v.kind == "hint"]
    assert [v.step for v in hints] == [COOLER_REMOVE]
