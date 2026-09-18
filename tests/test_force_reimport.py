"""``import-logs --force``: what it carries over, and what it refuses to touch.

A forced re-import replaces the step table wholesale. Two things have to survive
that honestly rather than quietly:

* the imported ``LS:`` note lines. They were matched by *step number* alone, so
  a re-exported sheet with one row inserted moved every note one step down the
  table without a word - and the warning still said "(carried over)". A note is
  now carried over only when the step number **and** the raw step name agree,
  and the ones that could not be are counted and reported.
* work a human has already frozen. ``infer-relations`` refuses a desktop with
  verified frames unless told twice; ``import-logs --force``, which destroys far
  more, did not. It does now, behind ``--force-verified``.

And what counts as a "verified frame" is a frame -- one ``(view, step)`` -- not
one compiled row, which is what made the same message claim 43 of them for a
desktop with two.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from test_cli import LOG_FIXTURES, d13_steps, env, open_db, run  # noqa: F401

from tda.cli import EXIT_ERROR, EXIT_OK
from tda.core.db import Db
from tda.core.model import FrameKey
from tda.pipeline_logs import carry_ls_notes
from tda.core.model import StepRec

LS_LINE = "LS: step_name=Remove PSU; complexity=3"


# --------------------------------------------------------------------------- #
# carry_ls_notes: number *and* name
# --------------------------------------------------------------------------- #
def _steps(*names: str, notes: str = "") -> list[StepRec]:
    return [StepRec(13, i, "normal", name, notes=notes)
            for i, name in enumerate(names, start=1)]


def test_a_note_is_carried_over_when_number_and_name_agree():
    previous = _steps("Initial", "CPU Fan Screw 1", notes=LS_LINE)
    fresh = _steps("Initial", "CPU Fan Screw 1")
    kept, dropped = carry_ls_notes(fresh, previous)
    assert (kept, dropped) == (2, 0)
    assert fresh[1].notes == LS_LINE


def test_a_renumbered_sheet_drops_the_note_instead_of_moving_it():
    """One row inserted: step 2 is a different operation now, so its note is not."""
    previous = _steps("Initial", "CPU Fan Screw 1", notes=LS_LINE)
    fresh = _steps("Initial", "Side Cover", "CPU Fan Screw 1")
    kept, dropped = carry_ls_notes(fresh, previous)
    assert kept == 1  # only step 1, whose name still matches
    assert dropped == 1
    assert LS_LINE not in (fresh[1].notes or "")
    assert LS_LINE not in (fresh[2].notes or "")


def test_a_note_past_the_end_of_the_new_sheet_is_dropped_and_counted():
    previous = _steps("Initial", "CPU Fan Screw 1", "Motherboard", notes=LS_LINE)
    fresh = _steps("Initial", "CPU Fan Screw 1")
    kept, dropped = carry_ls_notes(fresh, previous)
    assert (kept, dropped) == (2, 1)


def test_trailing_whitespace_in_a_sheet_name_is_not_a_renumbering():
    previous = _steps("Initial", "Heatsink cover ", notes=LS_LINE)
    fresh = _steps("Initial", "Heatsink cover")
    kept, dropped = carry_ls_notes(fresh, previous)
    assert (kept, dropped) == (2, 0)


def test_a_step_without_a_note_is_neither_kept_nor_dropped():
    previous = _steps("Initial", "CPU Fan Screw 1")
    fresh = _steps("Something else")
    assert carry_ls_notes(fresh, previous) == (0, 0)


# --------------------------------------------------------------------------- #
# ... and the run says so
# --------------------------------------------------------------------------- #
def _renumber_sheet(env: dict, desktop: int = 13) -> None:
    """Insert one row at the top of the exported sheet, renumbering the rest."""
    path = Path(env["drive"]) / f"desktop_{desktop:02d}.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    inserted = ["", "An extra row nobody logged before"] + [""] * (len(header) - 2)
    out = [header, inserted[: len(header)]] + body
    for i, row in enumerate(out[1:], start=1):
        row[0] = str(i)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(out)


def _note_every_step(env: dict, desktop: int = 13) -> int:
    db = open_db(env)
    try:
        steps = db.steps(desktop)
        for step in steps:
            step.notes = LS_LINE
        db.replace_steps(desktop, steps, db.actions(desktop))
        return len(steps)
    finally:
        db.close()


def _imported(env: dict) -> None:
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK


def test_a_renumbered_sheet_is_reported_per_desktop(env, capsys):
    _imported(env)
    _note_every_step(env)
    _renumber_sheet(env)
    capsys.readouterr()

    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    out = capsys.readouterr().out
    assert "LS notes could not be carried over (sheet renumbered)" in out
    assert "re-run import-ls" in out
    report = (env["cache"] / "import_logs_issues.md").read_text(encoding="utf-8")
    assert "could not be carried over (sheet renumbered)" in report


def test_an_unchanged_sheet_drops_nothing(env, capsys):
    _imported(env)
    _note_every_step(env)
    capsys.readouterr()

    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    out = capsys.readouterr().out
    assert "could not be carried over" not in out
    assert "run 'python -m tda.cli import-ls'" in out


def test_the_hint_is_gated_on_having_had_notes_not_on_survivors(env, capsys):
    """Every note is lost: that is exactly when re-running import-ls matters."""
    _imported(env)
    _note_every_step(env)
    _renumber_sheet(env)
    db = open_db(env)
    try:  # make sure nothing at all can survive: every name moved
        steps = db.steps(13)
        for step in steps:
            step.raw_name = f"gone {step.step}"
        db.replace_steps(13, steps, db.actions(13))
    finally:
        db.close()
    capsys.readouterr()

    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    out = capsys.readouterr().out
    assert "run 'python -m tda.cli import-ls'" in out


def test_a_desktop_that_never_had_notes_gets_no_hint(env, capsys):
    _imported(env)
    capsys.readouterr()
    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    assert "run 'python -m tda.cli import-ls'" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the verified-frame guard
# --------------------------------------------------------------------------- #
def _verify_rows(db: Db, desktop: int, frames: tuple[tuple[str, int], ...],
                 per_frame: int = 3) -> None:
    """Freeze ``per_frame`` compiled rows on each of ``frames``."""
    for view, step in frames:
        for n in range(per_frame):
            db.put_compiled(
                FrameKey(desktop, step, view), f"screw.motherboard.{n:02d}", None, 0.0,
                "visible", "in_chassis", "verified", "hash", verified_by="chang",
                geom_type="box", box=(0, 0, 4, 4),
            )


def test_force_refuses_a_desktop_that_carries_verified_frames(env, capsys):
    _imported(env)
    db = open_db(env)
    try:
        _verify_rows(db, 13, (("scan", 20), ("oak1", 21)))
        before = [s.raw_name for s in db.steps(13)]
    finally:
        db.close()
    capsys.readouterr()

    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "D13 has 2 verified frames" in out  # 2 frames, not 6 rows
    assert "--force-verified" in out
    db = open_db(env)
    try:
        assert [s.raw_name for s in db.steps(13)] == before
    finally:
        db.close()


def test_force_verified_goes_ahead(env, capsys):
    _imported(env)
    db = open_db(env)
    try:
        _verify_rows(db, 13, (("scan", 20),))
    finally:
        db.close()
    capsys.readouterr()

    assert run(env, "import-logs", "--desktops", "13", "--force",
               "--force-verified") == EXIT_OK
    assert "D13 has 1 verified frames" in capsys.readouterr().out


def test_a_desktop_without_verified_frames_is_never_refused(env, capsys):
    _imported(env)
    capsys.readouterr()
    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    assert "--force-verified" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# a verified frame is a frame
# --------------------------------------------------------------------------- #
def test_the_database_counts_verified_frames_not_verified_rows(env):
    _imported(env)
    db = open_db(env)
    try:
        _verify_rows(db, 13, (("scan", 20), ("scan", 21), ("oak1", 20)), per_frame=4)
        assert db.verified_frame_count(13) == 3
        assert db.verified_frames(13) == [("oak1", 20), ("scan", 20), ("scan", 21)]
        assert db.verified_frame_count(1) == 0
    finally:
        db.close()


def test_infer_relations_reports_frames_too(env, capsys):
    _imported(env)
    db = open_db(env)
    try:
        _verify_rows(db, 13, (("scan", 20), ("scan", 21)), per_frame=5)
    finally:
        db.close()
    capsys.readouterr()

    assert run(env, "infer-relations", "--desktops", "13") == EXIT_ERROR
    assert "D13 has 2 verified frames" in capsys.readouterr().out
