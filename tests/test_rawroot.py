"""Where the raw dataset is *now*: one resolver for a drive letter that moves.

The raw data lives on an external drive whose letter Windows hands out at
plug-in time. Every ``frame.path`` was recorded as ``F:/PHD Data Backup/...``,
and on the day the drive came back as ``G:`` three of the four views went
blank and the exit backup created ``F:/PHD Data Backup/Desktop Dataset/
TDA_backups`` on a *different* drive that happened to be ``F:`` that day.

Every test here builds a fake layout under ``tmp_path`` -- one folder per
"drive" -- and monkeypatches the drive enumeration: nothing reads a real drive
and nothing is ever written outside ``tmp_path``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from tda.core import rawroot as RR

MARKER = RR.DEFAULT_MARKER


def make_drive(root: Path, *, dataset: bool = True, backups_only: bool = False) -> Path:
    """A fake drive: ``<root>/<marker>/{OAKD Capture,UGA DATA,Realsense Capture}``."""
    base = root / MARKER
    if dataset:
        for child in RR.RAW_CHILDREN:
            (base / child).mkdir(parents=True, exist_ok=True)
    if backups_only:
        (base / "TDA_backups").mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return base


def fake_drives(monkeypatch, drives: dict, labels: dict | None = None) -> None:
    """Make ``_drive_roots`` answer ``drives`` (letter -> folder) and nothing else."""
    monkeypatch.setattr(RR, "_drive_roots", lambda: [str(p) for p in drives.values()])
    labels = labels or {}
    by_root = {os.path.normcase(str(p)): labels.get(letter)
               for letter, p in drives.items()}
    monkeypatch.setattr(RR, "_volume_label",
                        lambda root: by_root.get(os.path.normcase(str(root))))


def paths_for(recorded: Path, **extra) -> dict:
    base = str(recorded / MARKER).replace("\\", "/")
    out = {
        "f_root": base,
        "scanner_root": f"{base}/UGA DATA",
        "oak_root": f"{base}/OAKD Capture/Desktop_Datacollection/DesktopData",
        "rs_root": f"{base}/Realsense Capture/Dataset Information/Exp",
        "raw_marker": MARKER,
    }
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# locating the root
# --------------------------------------------------------------------------- #
def test_the_recorded_root_is_used_when_it_still_holds_the_dataset(tmp_path, monkeypatch):
    f = tmp_path / "F_"
    make_drive(f)
    fake_drives(monkeypatch, {"F": f})
    raw = RR.locate(paths_for(f))
    assert raw.status == RR.FOUND
    assert raw.connected
    stored = f"{(f / MARKER).as_posix()}/UGA DATA/13/RGB11/P_0.png"
    assert raw.resolve(stored) == stored


def test_a_drive_that_moved_is_found_by_scanning_the_letters(tmp_path, monkeypatch):
    f, g = tmp_path / "F_", tmp_path / "G_"
    make_drive(f, dataset=False, backups_only=True)   # today's T9: only our backups
    make_drive(g)
    fake_drives(monkeypatch, {"F": f, "G": g})
    raw = RR.locate(paths_for(f))
    assert raw.status == RR.MOVED
    assert Path(raw.resolved) == g / MARKER
    stored = f"{(f / MARKER).as_posix()}/OAKD Capture/D13/001/x_rgb_12mp.jpg"
    assert Path(raw.resolve(stored)) == g / MARKER / "OAKD Capture" / "D13" / "001" / "x_rgb_12mp.jpg"
    # the one line a log gets, at INFO
    assert raw.log_line == (f"raw root: recorded {(f / MARKER).as_posix()} "
                            f"resolved {(g / MARKER).as_posix()}")
    assert raw.log_level == logging.INFO


def test_backslashes_and_case_do_not_hide_a_recorded_prefix(tmp_path, monkeypatch):
    f, g = tmp_path / "F_", tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, {"G": g})
    raw = RR.locate(paths_for(f))
    stored = str(f / MARKER).upper().replace("/", "\\") + "\\UGA DATA\\P_0.png"
    found = raw.resolve(stored)
    assert found is not None
    assert Path(found) == g / MARKER / "UGA DATA" / "P_0.png"


def test_a_marker_at_any_drive_root_is_recognised(tmp_path, monkeypatch):
    """A path recorded under *another* letter -- a re-built index on H: -- too."""
    g = tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, {"G": g})
    raw = RR.locate(paths_for(tmp_path / "F_"))
    assert raw.status == RR.MOVED
    found = raw.resolve(f"h:\\{MARKER}\\UGA DATA\\13\\P_0.png")
    assert Path(found) == g / MARKER / "UGA DATA" / "13" / "P_0.png"


def test_a_folder_with_only_backups_is_not_the_dataset(tmp_path, monkeypatch):
    f = tmp_path / "F_"
    make_drive(f, dataset=False, backups_only=True)
    fake_drives(monkeypatch, {"F": f})
    raw = RR.locate(paths_for(f))
    assert raw.status == RR.MISSING
    assert not raw.connected


def test_no_drive_with_the_dataset_means_not_connected(tmp_path, monkeypatch):
    f = tmp_path / "F_"
    f.mkdir()
    fake_drives(monkeypatch, {"F": f})
    raw = RR.locate(paths_for(f, raw_volume_label="Elements"))
    assert raw.status == RR.MISSING
    assert raw.resolve(f"{(f / MARKER).as_posix()}/UGA DATA/P_0.png") is None
    # a local file is not on the raw drive and is not affected by it
    local = str(tmp_path / "cache" / "scan" / "D13" / "s001.png")
    assert raw.resolve(local) == local
    assert "原始数据盘没连上" in raw.message and "raw data drive not found" in raw.message
    assert MARKER in raw.message and "Elements" in raw.message and "F5" in raw.message


def test_two_candidates_are_ambiguous_without_a_label(tmp_path, monkeypatch):
    g, h = tmp_path / "G_", tmp_path / "H_"
    make_drive(g)
    make_drive(h)
    fake_drives(monkeypatch, {"G": g, "H": h}, labels={"G": "Elements", "H": "Copy"})
    raw = RR.locate(paths_for(tmp_path / "F_"))
    assert raw.status == RR.AMBIGUOUS
    assert raw.resolved is None
    assert len(raw.candidates) == 2
    assert "raw_volume_label" in raw.message


def test_the_volume_label_breaks_a_tie(tmp_path, monkeypatch):
    g, h = tmp_path / "G_", tmp_path / "H_"
    make_drive(g)
    make_drive(h)
    fake_drives(monkeypatch, {"G": g, "H": h}, labels={"G": "Elements", "H": "Copy"})
    raw = RR.locate(paths_for(tmp_path / "F_", raw_volume_label="elements"))
    assert raw.status == RR.MOVED
    assert Path(raw.resolved) == g / MARKER


def test_a_configuration_without_raw_roots_changes_nothing(tmp_path, monkeypatch):
    """The test scenes' paths.yaml: no marker anywhere, so nothing is scanned."""
    monkeypatch.setattr(RR, "_drive_roots", lambda: pytest.fail("scanned the drives"))
    raw = RR.locate({"oak_root": str(tmp_path / "src" / "oak"),
                     "scanner_root": str(tmp_path / "src" / "scan")})
    assert raw.status == RR.UNCONFIGURED
    assert raw.resolve("F:/anything/at/all.png") == "F:/anything/at/all.png"


def test_the_module_resolver_is_identity_until_configured(tmp_path, monkeypatch):
    RR.reset()
    assert RR.current() is None
    assert RR.resolve_raw(f"F:/{MARKER}/UGA DATA/P_0.png") == f"F:/{MARKER}/UGA DATA/P_0.png"
    assert RR.resolve_raw(None) is None
    assert RR.resolve_raw("") is None
    g = tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, {"G": g})
    RR.configure(paths_for(tmp_path / "F_"))
    stored = f"{(tmp_path / 'F_' / MARKER).as_posix()}/UGA DATA/P_0.png"
    assert Path(RR.resolve_raw(stored)) == g / MARKER / "UGA DATA" / "P_0.png"
    # ... and a recorded letter the configuration never named is the same data
    assert Path(RR.resolve_raw(f"F:/{MARKER}/UGA DATA/P_0.png")) == \
        g / MARKER / "UGA DATA" / "P_0.png"
    RR.reset()
    assert RR.current() is None


def test_only_cheap_directory_checks_touch_the_drives(tmp_path, monkeypatch):
    """Scanning A:-Z: must be ``isdir`` and nothing else: no listing, no writes."""
    g = tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, {"G": g})
    monkeypatch.setattr(os, "listdir", lambda *a, **k: pytest.fail("listed a drive"))
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: pytest.fail("created a folder"))
    raw = RR.locate(paths_for(tmp_path / "F_"))
    assert raw.status == RR.MOVED


# --------------------------------------------------------------------------- #
# the backup target never creates a directory on the wrong volume
# --------------------------------------------------------------------------- #
def _entries(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def test_the_backup_goes_to_the_resolved_raw_root(tmp_path, monkeypatch):
    f, g = tmp_path / "F_", tmp_path / "G_"
    f.mkdir()
    make_drive(g)
    fake_drives(monkeypatch, {"F": f, "G": g})
    paths = paths_for(f, backup_dir=f"{(f / MARKER).as_posix()}/TDA_backups")
    RR.configure(paths)
    target = RR.backup_target(paths)
    assert Path(target) == g / MARKER / "TDA_backups"
    assert (g / MARKER / "TDA_backups").is_dir()
    assert _entries(f) == []          # nothing created on the drive that is F: today


def test_no_raw_drive_means_no_backup_and_no_folder(tmp_path, monkeypatch):
    f = tmp_path / "F_"
    f.mkdir()
    fake_drives(monkeypatch, {"F": f})
    paths = paths_for(f, backup_dir=f"{(f / MARKER).as_posix()}/TDA_backups")
    RR.configure(paths)
    with pytest.raises(RR.BackupUnavailable) as err:
        RR.backup_target(paths)
    assert "没有备份" in str(err.value) or "no backup" in str(err.value)
    assert _entries(f) == []


def test_a_backup_dir_outside_the_raw_roots_must_already_exist(tmp_path, monkeypatch):
    fake_drives(monkeypatch, {})
    paths = {"backup_dir": str(tmp_path / "local" / "backups")}
    with pytest.raises(RR.BackupUnavailable) as err:
        RR.backup_target(paths)
    assert "backups" in str(err.value)
    assert not (tmp_path / "local").exists()
    (tmp_path / "local" / "backups").mkdir(parents=True)
    assert Path(RR.backup_target(paths)) == tmp_path / "local" / "backups"


def test_only_the_leaf_is_ever_created_under_the_raw_root(tmp_path, monkeypatch):
    """A deeper configured target whose parent is missing is refused, not built."""
    g = tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, {"G": g})
    paths = paths_for(tmp_path / "F_",
                      backup_dir=f"{(tmp_path / 'F_' / MARKER).as_posix()}/TDA_backups/deep/er")
    RR.configure(paths)
    with pytest.raises(RR.BackupUnavailable):
        RR.backup_target(paths)
    assert not (g / MARKER / "TDA_backups").exists()


def test_db_backup_creates_at_most_the_leaf(tmp_path):
    from tda.core.db import Db

    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        with pytest.raises(OSError):
            db.backup(str(tmp_path / "missing" / "parent" / "backups"))
        assert not (tmp_path / "missing").exists()
        assert Path(db.backup(str(tmp_path / "backups"))).is_file()
    finally:
        db.close()
