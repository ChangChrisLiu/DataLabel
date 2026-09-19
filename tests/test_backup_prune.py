"""Backups pile up until the drive is full, so something has to prune them.

Every ``--force``, every ``import-ls``, every ``infer-relations``, every
``load-index`` and every app exit adds an 18 MB copy to ``backup_dir`` on the
read-only-except-here F: drive, and nothing ever removed one.

Pruning a backup directory is the kind of code that deletes the wrong file, so
the rules are narrow and tested: only names this tool writes, only after the new
copy has been **verified**, never the newest copy of a recent day, and never
fatally.
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from tda.core.db_backup import BACKUP_RE, prune_backups


def _make(folder: Path, name: str, size: int = 16, age_days: float = 0.0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"x" * size)
    if age_days:
        stamp = time.time() - age_days * 86400
        os.utime(path, (stamp, stamp))
    return path


def _stamped(day: str, hour: int = 12, serial: str = "") -> str:
    """One backup name. ``day`` is a ``YYYYMMDD``, which is what the rule reads."""
    return f"tda_{day}_{hour:02d}0000{serial}.sqlite"


def _long_ago(n: int) -> list[str]:
    """``n`` ``YYYYMMDD`` days well outside the daily-retention window."""
    return [f"2025{month:02d}01" for month in range(1, n + 1)]


def _recent(n: int) -> list[str]:
    """The last ``n`` days, which the daily rule protects."""
    return [(date.today() - timedelta(days=d)).strftime("%Y%m%d")
            for d in range(n - 1, -1, -1)]


def _names(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir()}


# --------------------------------------------------------------------------- #
# which names are even candidates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [
    "tda_20260918_120000.sqlite",
    "tda_20260918_120000_1.sqlite",
    "tda_20260101_000000_12.sqlite",
])
def test_the_names_this_tool_writes_are_candidates(name: str):
    assert BACKUP_RE.match(name)


@pytest.mark.parametrize("name", [
    "tda_pre_v3_20260918.sqlite",       # somebody's hand-made checkpoint
    "tda.sqlite",
    "tda_20260918_120000.sqlite.bak",
    "tda_20260918.sqlite",              # no time
    "tda_20260918_120000_x.sqlite",     # not a serial
    "notes.txt",
])
def test_everything_else_is_not(name: str):
    assert not BACKUP_RE.match(name)


# --------------------------------------------------------------------------- #
# pruning
# --------------------------------------------------------------------------- #
def test_nothing_is_pruned_below_the_limit(tmp_path):
    for day in _long_ago(5):
        _make(tmp_path, _stamped(day))
    assert prune_backups(str(tmp_path), keep=10) == []
    assert len(_names(tmp_path)) == 5


def test_the_oldest_go_first(tmp_path):
    days = _long_ago(10)
    for day in days:
        _make(tmp_path, _stamped(day))
    removed = prune_backups(str(tmp_path), keep=4)
    assert len(removed) == 6
    assert sorted(_names(tmp_path)) == [_stamped(d) for d in days[-4:]]


def test_a_foreign_file_is_never_touched(tmp_path):
    _make(tmp_path, "tda_pre_v3_20260918.sqlite")
    _make(tmp_path, "notes.txt")
    for day in _long_ago(10):
        _make(tmp_path, _stamped(day))
    prune_backups(str(tmp_path), keep=2)
    assert "tda_pre_v3_20260918.sqlite" in _names(tmp_path)
    assert "notes.txt" in _names(tmp_path)


def test_the_newest_copy_of_each_recent_day_survives_the_limit(tmp_path):
    """Fourteen days of history is worth more than fourteen copies of today."""
    days = _recent(14)
    for day in days[:-1]:  # one a day for the last fortnight
        _make(tmp_path, _stamped(day))
    for hour in range(8):  # ... and eight from today
        _make(tmp_path, _stamped(days[-1], hour=hour))

    prune_backups(str(tmp_path), keep=4)
    kept = _names(tmp_path)
    # every day of the fortnight still has its newest copy
    assert {name[4:12] for name in kept} == set(days)
    # ... and today kept only its newest plus whatever the limit allowed
    assert len([n for n in kept if n[4:12] == days[-1]]) < 8


def test_an_old_days_extra_copies_do_go(tmp_path):
    first, second = _long_ago(2)
    for hour in range(6):
        _make(tmp_path, _stamped(first, hour=hour))
    _make(tmp_path, _stamped(second))
    removed = prune_backups(str(tmp_path), keep=2)
    assert len(removed) == 5
    assert len(_names(tmp_path)) == 2


def test_a_file_that_cannot_be_removed_is_not_fatal(tmp_path, monkeypatch):
    for day in _long_ago(5):
        _make(tmp_path, _stamped(day))

    def refuse(path):
        raise OSError("in use by another process")

    monkeypatch.setattr(os, "remove", refuse)
    assert prune_backups(str(tmp_path), keep=1) == []  # reported as removing nothing
    assert len(_names(tmp_path)) == 5


def test_a_missing_directory_is_not_fatal(tmp_path):
    assert prune_backups(str(tmp_path / "nope"), keep=3) == []


@pytest.mark.parametrize("keep", [0, -1, None])
def test_a_meaningless_limit_prunes_nothing(tmp_path, keep):
    for day in _long_ago(5):
        _make(tmp_path, _stamped(day))
    assert prune_backups(str(tmp_path), keep=keep) == []
    assert len(_names(tmp_path)) == 5


def test_only_plain_files_are_candidates(tmp_path):
    """A directory or a link that happens to carry a backup's name is left alone."""
    days = _long_ago(4)
    for day in days[1:]:
        _make(tmp_path, _stamped(day))
    (tmp_path / _stamped(days[0])).mkdir()                # oldest "name" is a folder
    target = tmp_path / "elsewhere.bin"
    target.write_bytes(b"not a backup")
    link = tmp_path / _stamped(days[0], hour=13)
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):                # no privilege on this Windows
        link = None
    removed = prune_backups(str(tmp_path), keep=2)
    assert (tmp_path / _stamped(days[0])).is_dir()
    assert target.exists()
    if link is not None:
        assert os.path.islink(link)
    assert removed == [_stamped(days[1])]                 # the oldest PLAIN file only
