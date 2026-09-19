"""Deleting is destructive, so the four ways it could go wrong are closed.

``import-logs --force`` removing an instance the sheet no longer names is right,
and it is the only thing in this tool that deletes annotator-visible data. So:

* it happens **only** under ``--force``, which is the only path that takes a
  safety backup first;
* a sheet that parses but is absurd -- a renamed header, an export truncated to
  its first rows -- refuses the desktop instead of emptying it;
* every drop is written to ``op_log`` with enough to put it back;
* a key that is kept because a human worked on it keeps whatever *it* points at.

A parse failure was already safe: the whole desktop rolls back.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from test_cli import LOG_FIXTURES, d13_steps, env, open_db, run  # noqa: F401
from test_force_drops import GHOST, imported, instances, reimport, seed_ghost

from tda.cli import EXIT_ERROR, EXIT_OK
from tda.core.db import Db
from tda.core.model import FrameKey, InstanceRec, ShapeKeyframe, ShapePart

DESKTOP = 13
SECOND_GHOST = "psu.03"


def sheet(env: dict, names: list[str]) -> None:
    """Rewrite D13's sheet with these step names, keeping the meta sidecar."""
    path = env["drive"] / f"desktop_{DESKTOP:02d}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Sequence Number", "Sequence Name", "Target Nest Group",
                         "Tool Utility", "Notes"])
        for i, name in enumerate(names, start=1):
            writer.writerow([i, name, "", "", ""])


def steps_of(env: dict) -> int:
    db = open_db(env)
    try:
        return len(db.steps(DESKTOP))
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 1. only under --force, which is the only path with a backup
# --------------------------------------------------------------------------- #
def test_an_import_without_force_drops_nothing(env, capsys):
    """A desktop with instances but no steps takes no backup, so it may not delete."""
    imported(env)
    db = open_db(env)
    try:
        db.replace_steps(DESKTOP, [], [])  # steps gone, instances still there
    finally:
        db.close()
    seed_ghost(env)
    capsys.readouterr()
    assert run(env, "import-logs", "--desktops", str(DESKTOP)) == EXIT_OK
    printed = capsys.readouterr().out
    assert GHOST in instances(env)
    assert "dropped instance" not in printed
    assert "re-run with --force to drop them" in printed


def test_the_forced_path_still_drops(env):
    imported(env)
    seed_ghost(env)
    reimport(env)
    assert GHOST not in instances(env)


# --------------------------------------------------------------------------- #
# 2. a sheet that parses but is absurd
# --------------------------------------------------------------------------- #
def test_a_truncated_sheet_refuses_the_desktop(env, capsys):
    """An export cut to its header: 0 steps, and 42 instances would go."""
    imported(env)
    before = sorted(instances(env))
    sheet(env, [])
    capsys.readouterr()
    assert reimport(env) == EXIT_ERROR
    printed = capsys.readouterr().out
    assert "refused" in printed
    assert sorted(instances(env)) == before  # nothing written at all
    assert steps_of(env) == env["n_steps_13"]


def test_the_refusal_says_what_it_compared_and_what_would_have_gone(env, capsys):
    imported(env)
    sheet(env, ["Initial Conditions", "CPU Fan Screw 1"])
    capsys.readouterr()
    reimport(env)
    printed = capsys.readouterr().out
    assert "--force-drop" in printed
    assert str(env["n_steps_13"]) in printed  # the old step count
    assert "At risk:" in printed  # and the keys themselves


def test_force_drop_goes_through_anyway(env):
    imported(env)
    sheet(env, ["Initial Conditions", "CPU Fan Screw 1"])
    assert run(env, "import-logs", "--desktops", str(DESKTOP), "--force",
               "--force-drop") == EXIT_OK
    assert steps_of(env) == 2


def test_a_sheet_that_drops_one_instance_is_not_refused(env):
    """The thirteen real desktops drop exactly one key and keep their steps."""
    imported(env)
    seed_ghost(env)
    assert reimport(env) == EXIT_OK
    assert GHOST not in instances(env)
    assert steps_of(env) == env["n_steps_13"]


def test_five_dropped_instances_are_allowed_and_six_are_not(env):
    imported(env)
    db = open_db(env)
    try:
        for i in range(2, 7):
            db.upsert_instance(InstanceRec(key=f"psu.{i:02d}", desktop=DESKTOP,
                                           cls="psu"))
    finally:
        db.close()
    assert reimport(env) == EXIT_OK  # five is the limit, not over it
    assert not [k for k in instances(env) if k.startswith("psu.0")
                and k != "psu.01"]


def test_more_than_five_dropped_instances_refuses(env, capsys):
    imported(env)
    db = open_db(env)
    try:
        for i in range(2, 8):  # six ghosts
            db.upsert_instance(InstanceRec(key=f"psu.{i:02d}", desktop=DESKTOP,
                                           cls="psu"))
    finally:
        db.close()
    capsys.readouterr()
    assert reimport(env) == EXIT_ERROR
    assert "refused" in capsys.readouterr().out
    assert "psu.07" in instances(env)


def test_a_first_import_is_never_refused(env):
    """Nothing to lose: the guard compares against a previous run, and there is none."""
    sheet(env, ["Initial Conditions"])
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs", "--desktops", str(DESKTOP)) == EXIT_OK


# --------------------------------------------------------------------------- #
# 3. every drop is undoable
# --------------------------------------------------------------------------- #
def test_a_drop_writes_an_op_log_row_that_could_put_it_back(env):
    imported(env)
    seed_ghost(env, mounted_on="chassis", attrs={"note": "seeded"})
    db = open_db(env)
    try:
        from tda.core.graph import Edge, edges_to_db
        edges_to_db(db, DESKTOP, [
            Edge("connected_to", GHOST, "connector.01", source="rule"),
        ])
    finally:
        db.close()
    reimport(env)
    db = open_db(env)
    try:
        rows = [r for r in db.ops(DESKTOP, "-") if r["kind"] == "dropped_instance"]
        assert len(rows) == 1
        payload, inverse = rows[0]["payload"], rows[0]["inverse"]
        assert payload["instance"] == GHOST
        assert payload["record"]["cls"] == "psu"
        assert payload["record"]["mounted_on"] == "chassis"
        assert any(r["target"] == GHOST for r in payload["relations"])
        # enough to recreate: the whole record comes back out of the inverse
        assert inverse["record"]["key"] == GHOST
        assert inverse["record"]["attrs"]["note"] == "seeded"
    finally:
        db.close()


def test_an_instance_that_is_kept_writes_no_op_log_row(env):
    imported(env)
    seed_ghost(env)
    db = open_db(env)
    try:
        db.add_keyframe(ShapeKeyframe(
            id=None, instance=GHOST, desktop=DESKTOP, view="scan",
            pose_segment=1, anchor_step=1,
            parts=[ShapePart(name="body", rle={"size": [4, 4], "counts": "a"})],
        ))
    finally:
        db.close()
    reimport(env)
    db = open_db(env)
    try:
        assert not [r for r in db.ops(DESKTOP, "-") if r["kind"] == "dropped_instance"]
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 4. a kept key keeps what it points at
# --------------------------------------------------------------------------- #
def test_a_kept_key_keeps_the_vanished_key_it_points_at(env):
    """``psu.02`` is frozen and mounted on ``psu.03``; neither may go."""
    imported(env)
    seed_ghost(env, mounted_on=SECOND_GHOST)
    db = open_db(env)
    try:
        db.upsert_instance(InstanceRec(key=SECOND_GHOST, desktop=DESKTOP, cls="psu"))
        db.add_keyframe(ShapeKeyframe(
            id=None, instance=GHOST, desktop=DESKTOP, view="scan",
            pose_segment=1, anchor_step=1,
            parts=[ShapePart(name="body", rle={"size": [4, 4], "counts": "a"})],
        ))
    finally:
        db.close()
    reimport(env)
    held = instances(env)
    assert GHOST in held  # the keyframe holds it
    assert SECOND_GHOST in held  # and it holds this one
    report = (env["cache"] / "import_logs_issues.md").read_text(encoding="utf-8")
    assert SECOND_GHOST in report


def test_no_surviving_instance_can_point_at_a_dropped_key(env):
    """The importer rewrote every relational field, so this cannot happen."""
    imported(env)
    seed_ghost(env)
    reimport(env)
    db = open_db(env)
    try:
        keys = set(db.instances(DESKTOP))
        for rec in db.instances(DESKTOP).values():
            for ref in (rec.parent, rec.mounted_on, rec.fastens, rec.socket_host):
                assert ref is None or ref in keys or not ref
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 5. an implied instance whose warrant is gone
# --------------------------------------------------------------------------- #
#: A sheet that only *references* a motherboard, and is big enough that losing
#: one instance is not itself implausible (the real desktops hold 35-98).
SCREW_ONLY = ["Initial Conditions"] + [f"Motherboard Screw {i}" for i in range(1, 7)]


def implied_board(env: dict) -> None:
    """Import a sheet that only *references* a motherboard, so one is implied."""
    sheet(env, SCREW_ONLY)
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs", "--desktops", str(DESKTOP)) == EXIT_OK
    assert instances(env)["motherboard.01"].attrs.get("implied")


def test_a_declined_implied_instance_is_dropped_with_its_own_reason(env, capsys):
    """S1 said no, so the next re-import must not quietly imply it again."""
    implied_board(env)
    db = open_db(env)
    try:
        db.decline_implied(DESKTOP, "motherboard")
    finally:
        db.close()
    capsys.readouterr()
    assert reimport(env) == EXIT_OK
    printed = capsys.readouterr().out
    assert "motherboard.01" not in instances(env)
    assert "implied instance no longer warranted" in printed
    assert "no longer in the sheet" not in printed


def test_a_real_part_replaces_the_implied_one_rather_than_dropping_it(env):
    """The sheet names it now: the same key is overwritten, not deleted."""
    implied_board(env)
    sheet(env, SCREW_ONLY + ["Motherboard"])
    assert reimport(env) == EXIT_OK
    held = instances(env)
    assert "motherboard.01" in held
    assert not held["motherboard.01"].attrs.get("implied")
    assert held["motherboard.01"].raw_names == ["Motherboard"]


# --------------------------------------------------------------------------- #
# 6. the run says so without anybody reading the report
# --------------------------------------------------------------------------- #
def test_the_run_counts_the_instances_it_kept_for_review(env, capsys):
    imported(env)
    seed_ghost(env)
    db = open_db(env)
    try:
        db.add_conflict(FrameKey(DESKTOP, 1, "scan"), GHOST, None, None, 3)
    finally:
        db.close()
    capsys.readouterr()
    reimport(env)
    assert "1 vanished instances kept for S1 review" in capsys.readouterr().out
