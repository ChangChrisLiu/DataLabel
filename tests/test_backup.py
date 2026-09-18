"""The safety copy every destructive command takes first (spec 3.5).

A backup nobody checked is worse than none: it makes the run that destroys the
step table look safe. So :meth:`tda.core.db.Db.backup` verifies what it wrote,
and every command that takes a *safety* copy turns any failure into one line and
exit 1 with nothing written and the single-user lock released.

The other half is ``backup --dest``: it may only name ``backup_dir`` or a folder
inside it, and on Windows that comparison has to survive a differently-cased or
``subst``-mapped spelling of the very same directory.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from test_cli import d13_steps, env, run  # noqa: F401  (re-used fixtures)

from tda.cli import EXIT_ERROR, EXIT_OK
from tda.core.db import Db
from tda.pipeline import backup_dest


# --------------------------------------------------------------------------- #
# Db.backup verifies what it wrote
# --------------------------------------------------------------------------- #
class _SilentlyDoesNothing:
    """A connection whose ``backup()`` reports success and writes no bytes.

    ``sqlite3.Connection`` is immutable, so the shim goes on the ``Db`` instance
    rather than on the class; everything else is forwarded to the real thing.
    """

    def __init__(self, conn):
        self._conn = conn

    def backup(self, target):
        return None

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_backup_raises_when_the_copy_did_not_land(tmp_path):
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        db.conn = _SilentlyDoesNothing(db.conn)
        with pytest.raises(OSError) as err:
            db.backup(str(tmp_path / "backups"))
        assert "empty" in str(err.value) or "not written" in str(err.value)
    finally:
        db.close()


def test_a_verified_backup_is_not_left_behind_as_an_empty_file(tmp_path):
    """An unusable copy must not sit in backup_dir looking like a real one."""
    db = Db(str(tmp_path / "tda.sqlite"))
    dest = tmp_path / "backups"
    try:
        db.conn = _SilentlyDoesNothing(db.conn)
        with pytest.raises(OSError):
            db.backup(str(dest))
    finally:
        db.close()
    assert list(dest.glob("tda_*.sqlite")) == []


def test_a_good_backup_has_content(tmp_path):
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        out = db.backup(str(tmp_path / "backups"))
    finally:
        db.close()
    assert os.path.getsize(out) > 0


# --------------------------------------------------------------------------- #
# --dest containment, Windows spellings included
# --------------------------------------------------------------------------- #
def test_dest_accepts_a_differently_cased_spelling_of_the_same_folder(tmp_path):
    """Neither folder exists yet - ``backup()`` creates them - so ``resolve()``
    cannot repair the casing from the disk and the comparison has to."""
    root = tmp_path / "Backups"
    paths = {"backup_dir": str(root)}
    assert backup_dest(paths, str(root / "weekly"))
    assert backup_dest(paths, str(root))
    if os.path.normcase("A") == os.path.normcase("a"):  # Windows / macOS
        assert backup_dest(paths, str(tmp_path / "backups" / "WEEKLY"))
        assert backup_dest(paths, str(tmp_path / "BACKUPS"))


def test_dest_outside_backup_dir_is_still_refused(tmp_path):
    root = tmp_path / "backups"
    root.mkdir()
    with pytest.raises(ValueError) as err:
        backup_dest({"backup_dir": str(root)}, str(tmp_path / "elsewhere"))
    assert "must be inside the configured backup_dir" in str(err.value)


def test_a_sibling_whose_name_merely_starts_with_backup_dir_is_refused(tmp_path):
    """``.../backups_old`` is not inside ``.../backups``, however it compares."""
    root = tmp_path / "backups"
    root.mkdir()
    with pytest.raises(ValueError):
        backup_dest({"backup_dir": str(root)}, str(tmp_path / "backups_old"))


def test_dest_needs_a_configured_backup_dir(tmp_path):
    with pytest.raises(ValueError) as err:
        backup_dest({}, str(tmp_path / "anywhere"))
    assert "backup_dir" in str(err.value)


# --------------------------------------------------------------------------- #
# every command that takes a safety copy fails the same way
# --------------------------------------------------------------------------- #
#: (argv, what the database must still be missing afterwards)
SAFETY_COMMANDS = (
    ("import-logs", "--force"),
    ("import-ls", "--purge-all", "--allow-missing-steps"),
    ("infer-relations",),
)


def _imported(env: dict) -> None:
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK


@pytest.mark.parametrize("argv", SAFETY_COMMANDS, ids=[a[0] for a in SAFETY_COMMANDS])
def test_a_failing_backup_stops_the_command_with_one_line(env, capsys, monkeypatch, argv):
    _imported(env)
    if argv[0] == "import-ls":
        argv = (*argv, "--export", str(Path(__file__).parent / "fixtures" / "ls_small.json"))
    capsys.readouterr()

    def boom(self, dest_dir):
        raise OSError("the backup volume is full")

    monkeypatch.setattr(Db, "backup", boom)
    assert run(env, *argv) == EXIT_ERROR
    out = capsys.readouterr().out
    assert f"[{argv[0]}] backup failed: the backup volume is full" in out
    assert "Traceback" not in out


@pytest.mark.parametrize("argv", SAFETY_COMMANDS, ids=[a[0] for a in SAFETY_COMMANDS])
def test_a_missing_backup_dir_is_the_same_one_line(env, capsys, monkeypatch, argv):
    """``ValueError`` from ``backup_dest`` must not read like a usage error."""
    _imported(env)
    if argv[0] == "import-ls":
        argv = (*argv, "--export", str(Path(__file__).parent / "fixtures" / "ls_small.json"))
    import yaml

    cfg = dict(env["cfg"])
    cfg.pop("backup_dir")
    Path(env["paths"]).write_text(yaml.safe_dump(cfg), encoding="utf-8")
    capsys.readouterr()

    assert run(env, *argv) == EXIT_ERROR
    out = capsys.readouterr().out
    assert f"[{argv[0]}] backup failed:" in out and "backup_dir" in out
    assert "Traceback" not in out


@pytest.mark.parametrize("argv", SAFETY_COMMANDS, ids=[a[0] for a in SAFETY_COMMANDS])
def test_a_sqlite_error_during_the_backup_is_caught_too(env, capsys, monkeypatch, argv):
    _imported(env)
    if argv[0] == "import-ls":
        argv = (*argv, "--export", str(Path(__file__).parent / "fixtures" / "ls_small.json"))
    capsys.readouterr()

    def boom(self, dest_dir):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(Db, "backup", boom)
    assert run(env, *argv) == EXIT_ERROR
    assert f"[{argv[0]}] backup failed: disk I/O error" in capsys.readouterr().out


def test_a_failed_backup_leaves_the_step_table_untouched(env, capsys, monkeypatch):
    _imported(env)
    db = Db(env["db_path"])
    try:
        before = [s.raw_name for s in db.steps(13)]
    finally:
        db.close()

    monkeypatch.setattr(Db, "backup", lambda self, dest: (_ for _ in ()).throw(
        OSError("no room")))
    assert run(env, "import-logs", "--force") == EXIT_ERROR
    db = Db(env["db_path"])
    try:
        assert [s.raw_name for s in db.steps(13)] == before
    finally:
        db.close()


def test_a_failed_backup_releases_the_lock(env, capsys, monkeypatch):
    _imported(env)
    monkeypatch.setattr(Db, "backup", lambda self, dest: (_ for _ in ()).throw(
        OSError("no room")))
    assert run(env, "infer-relations") == EXIT_ERROR
    # the lock is gone, so the next command runs normally rather than refusing
    assert run(env, "status") == EXIT_OK
