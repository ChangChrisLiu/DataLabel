"""One bad desktop must not decide what the other sixty-five get.

Three long-running commands share the same shape -- a loop over desktops, one
transaction each -- and this pins down what that has to mean:

* ``load-index`` survives a malformed entry, lists it and exits 1;
* ``infer-relations`` scans the **whole** selection for verified frames before
  it writes anything, rather than applying thirty-nine desktops and then
  refusing the fortieth, and says at the end how many failed and that the rest
  went through;
* a pose segment that had no reference step to begin with does not report that
  it "moved" one, because nothing was lost.

Plus the smallest of the lot: a latch whose ``of`` holds a stray space is
unanswered, not answered with a space.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_cli import d13_steps, env, make_index, open_db, run  # noqa: F401

from tda.cli import EXIT_ERROR, EXIT_OK
from tda.core.db import Db
from tda.core.graph_infer import infer_relational_fields
from tda.core.model import FrameKey, InstanceRec
from tda.core.taxonomy import load_taxonomy
from tda.pipeline import load_index_into_db, split_pose_segments


@pytest.fixture
def tax():
    return load_taxonomy()


# --------------------------------------------------------------------------- #
# load-index: one bad desktop, the rest of the run
# --------------------------------------------------------------------------- #
class _Exploding:
    """A DesktopIndex whose frames cannot be read."""

    desktop = 99
    n_steps = 3
    issues: list[str] = []
    missing: list = []

    @property
    def frames(self):
        raise ValueError("index entry for D99 is malformed")


def test_a_malformed_index_entry_does_not_end_the_run(env, capsys, d13_steps):
    index = {13: make_index(13, d13_steps), 99: _Exploding()}
    db = Db(env["db_path"])
    try:
        counts = load_index_into_db(db, index, log=print)
        assert counts["desktops"] == 1  # D13 landed
        assert len(counts["failed"]) == 1
        assert "D99" in counts["failed"][0] and "malformed" in counts["failed"][0]
        assert db.frames_for(13, "oak1")
        assert db.frames_for(99, "oak1") == []  # and nothing half-written
    finally:
        db.close()


def test_load_index_reports_the_failure_and_exits_one(env, capsys, monkeypatch, d13_steps):
    """The CLI has to make the failure visible to a shell, not only to a reader."""
    import tda.pipeline as P

    real = P.load_index

    def broken(path):
        index = real(path)
        index[99] = _Exploding()
        return index

    monkeypatch.setattr("tda.cli.load_index", broken)
    capsys.readouterr()
    assert run(env, "load-index") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "D99" in out and "malformed" in out
    assert "Traceback" not in out


# --------------------------------------------------------------------------- #
# a reference step that never existed did not "move"
# --------------------------------------------------------------------------- #
def test_a_segment_without_a_reference_step_reports_nothing(env):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    db = open_db(env)
    try:
        db.update_pose_segment(13, "scan", 1, ref_step=None)
        db.upsert_desktop(13, {"pose_issues": []})
        split_pose_segments(db, 13)
        issues = (db.get_desktop(13) or {}).get("pose_issues") or []
        assert not any("from None" in line for line in issues)
        assert issues == []
    finally:
        db.close()


def test_a_reference_step_that_really_moved_is_still_reported(env):
    """D77's sheet flips the chassis at step 3, so segment 1 ends at step 2."""
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    db = open_db(env)
    try:
        db.update_pose_segment(77, "scan", 1, ref_step=5)  # now outside [1-2]
        db.upsert_desktop(77, {"pose_issues": []})
        split_pose_segments(db, 77)
        issues = (db.get_desktop(77) or {}).get("pose_issues") or []
        assert any("moved the reference step from 5 to 2" in line for line in issues)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# infer-relations: pre-scan, and the closing summary
# --------------------------------------------------------------------------- #
def _verify_a_frame(db: Db, desktop: int, step: int = 20) -> None:
    db.put_compiled(
        FrameKey(desktop, step, "scan"), "cpu_cooler.fan.01", None, 0.0, "visible",
        "in_chassis", "verified", "hash", verified_by="chang",
        geom_type="box", box=(0, 0, 4, 4),
    )


def _strip(db: Db, desktop: int) -> None:
    for rec in db.instances(desktop).values():
        rec.fastens = None
        rec.parent = None
        rec.attached = False
        db.upsert_instance(rec)


def test_one_verified_desktop_refuses_the_whole_run_before_anything_is_written(
    env, capsys
):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    db = open_db(env)
    try:
        _strip(db, 13)
        _strip(db, 77)
        _verify_a_frame(db, 13)
    finally:
        db.close()
    capsys.readouterr()

    assert run(env, "infer-relations") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "D13" in out and "carry verified frames" in out
    db = open_db(env)
    try:  # D77 was NOT applied: the run stopped before it touched anything
        assert db.instances(77)["screw.motherboard.01"].fastens is None
        assert db.instances(13)["screw.cpu_cooler.01"].fastens is None
    finally:
        db.close()


def test_force_applies_every_desktop_including_the_verified_one(env, capsys):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    db = open_db(env)
    try:
        _strip(db, 13)
        _strip(db, 77)
        _verify_a_frame(db, 13)
    finally:
        db.close()

    assert run(env, "infer-relations", "--force") == EXIT_OK
    db = open_db(env)
    try:
        assert db.instances(77)["screw.motherboard.01"].fastens == "motherboard.01"
        assert db.instances(13)["screw.cpu_cooler.01"].fastens is not None
    finally:
        db.close()


def test_a_dry_run_still_visits_everything(env, capsys):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    db = open_db(env)
    try:
        _strip(db, 13)
        _verify_a_frame(db, 13)
    finally:
        db.close()
    capsys.readouterr()

    assert run(env, "infer-relations", "--dry-run") == EXIT_OK
    assert "carry verified frames" not in capsys.readouterr().out


def test_the_run_says_how_many_desktops_failed_and_that_the_rest_went_through(
    env, capsys, monkeypatch
):
    import tda.cli_relations as CR

    real = CR.infer_relational_fields

    def flaky(instances, tax, actions=None):
        if any(rec.desktop == 13 for rec in instances.values()):
            raise RuntimeError("boom")
        return real(instances, tax, actions)

    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    db = open_db(env)
    try:
        _strip(db, 13)
        _strip(db, 77)
    finally:
        db.close()

    monkeypatch.setattr(CR, "infer_relational_fields", flaky)
    capsys.readouterr()
    assert run(env, "infer-relations") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "1 desktops failed; every other desktop was applied" in out


def test_a_run_with_no_failures_says_nothing_about_them(env, capsys):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    capsys.readouterr()
    assert run(env, "infer-relations") == EXIT_OK
    assert "every other desktop was applied" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# a latch whose `of` is a stray space is still unanswered
# --------------------------------------------------------------------------- #
def test_a_whitespace_latch_of_counts_as_blank(tax):
    instances = {
        "ram_module.01": InstanceRec("ram_module.01", 1, "ram_module"),
        "ram_latch.01": InstanceRec("ram_latch.01", 1, "ram_latch", attrs={"of": "  "}),
    }
    infer_relational_fields(instances, tax)
    assert instances["ram_latch.01"].attrs["of"] == "ram_module.01"


def test_a_filled_latch_of_is_still_never_overwritten(tax):
    instances = {
        "ram_module.01": InstanceRec("ram_module.01", 1, "ram_module"),
        "ram_module.02": InstanceRec("ram_module.02", 1, "ram_module"),
        "ram_latch.01": InstanceRec("ram_latch.01", 1, "ram_latch",
                                    attrs={"of": "ram_module.02"}),
    }
    infer_relational_fields(instances, tax)
    assert instances["ram_latch.01"].attrs["of"] == "ram_module.02"
