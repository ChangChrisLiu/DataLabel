"""Every reader of a stored raw path goes through the one resolver (task U2a).

:mod:`tests.test_rawroot` pins the resolver itself; this file pins the places
that turn a stored ``frame.path`` / ``aux`` path into a file to open -- the
session's image cache, the size measurement, the local-cache builder, the
index builder, the backups -- and the window that has to *say* when the raw
drive is not there.

Every "drive" is a folder under ``tmp_path`` and the drive enumeration is
monkeypatched: nothing here reads or writes a real drive.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tda.core import rawroot as RR
from tda.core.db import Db
from tda.core.model import FrameKey

MARKER = RR.DEFAULT_MARKER
OAK = "OAKD Capture/Desktop_Datacollection/DesktopData"


def make_drive(root: Path) -> Path:
    base = root / MARKER
    for child in RR.RAW_CHILDREN:
        (base / child).mkdir(parents=True, exist_ok=True)
    return base


def fake_drives(monkeypatch, *roots: Path) -> None:
    monkeypatch.setattr(RR, "_drive_roots", lambda: [str(r) for r in roots])
    monkeypatch.setattr(RR, "_volume_label", lambda root: None)


def raw_paths(tmp_path: Path, **extra) -> dict:
    """A paths.yaml recorded on the fake ``F_`` drive, everything we write in tmp."""
    base = (tmp_path / "F_" / MARKER).as_posix()
    out = {
        "f_root": base,
        "scanner_root": f"{base}/UGA DATA",
        "oak_root": f"{base}/{OAK}",
        "rs_root": f"{base}/Realsense Capture/Dataset Information/Exp",
        "raw_marker": MARKER,
        "raw_volume_label": "Elements",
        "cache_dir": str(tmp_path / "cache"),
        "db_path": str(tmp_path / "annotations" / "tda.sqlite"),
        "backup_dir": f"{base}/TDA_backups",
        "raw_logs_dir": str(tmp_path / "raw_logs"),
        "app_dir": str(tmp_path / "state"),
    }
    out.update(extra)
    return out


def recorded(tmp_path: Path, rel: str) -> str:
    return f"{(tmp_path / 'F_' / MARKER).as_posix()}/{rel}"


def write_jpg(path: Path, hw=(48, 64), value: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((*hw, 3), value, np.uint8)
    img[10:30, 10:40] = 220
    assert cv2.imwrite(str(path), img)


# --------------------------------------------------------------------------- #
# the session's image cache
# --------------------------------------------------------------------------- #
def test_the_image_cache_reads_a_moved_drive(tmp_path, monkeypatch):
    from tda.ui.session_images import ImageCache

    g = tmp_path / "G_"
    make_drive(g)
    (tmp_path / "F_").mkdir()
    fake_drives(monkeypatch, tmp_path / "F_", g)
    rel = f"{OAK}/Desktop 13/Disassemble/Camera_1/001/x_rgb_12mp.jpg"
    write_jpg(g / MARKER / rel)
    RR.configure(raw_paths(tmp_path))

    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        key = FrameKey(13, 1, "oak1")
        db.upsert_frame(key, recorded(tmp_path, rel), {}, None)
        images = ImageCache(db, str(tmp_path / "cache"))
        assert Path(images.image_path(key)) == g / MARKER / rel
        assert images.get(key).shape == (48, 64, 3)
        assert images.why_unreadable(key) is None
        assert images.view_problem(13, "oak1") is None
        # the stored value is provenance: nothing rewrote it
        assert db.get_frame(key)["path"] == recorded(tmp_path, rel)
    finally:
        db.close()


def test_an_aux_cache_path_on_the_raw_drive_is_resolved_too(tmp_path, monkeypatch):
    """``rs`` stored ``aux.cache_path`` as an ``F:`` path -- and still showed nothing."""
    from tda.ui.session_images import ImageCache

    g = tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, g)
    rel = "Realsense Capture/Dataset Information/Exp/Desktop 13/Disassemble/041/original_color.png"
    write_jpg(g / MARKER / rel)
    RR.configure(raw_paths(tmp_path))
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        key = FrameKey(13, 41, "rs")
        db.upsert_frame(key, "", {"cache_path": recorded(tmp_path, rel)}, None)
        assert Path(ImageCache(db, str(tmp_path / "cache")).image_path(key)) == g / MARKER / rel
    finally:
        db.close()


def test_an_unplugged_drive_is_said_rather_than_drawn_as_nothing(tmp_path, monkeypatch):
    from tda.ui.session_images import ImageCache

    (tmp_path / "F_").mkdir()
    fake_drives(monkeypatch, tmp_path / "F_")
    RR.configure(raw_paths(tmp_path))
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        key = FrameKey(13, 1, "oak1")
        db.upsert_frame(key, recorded(tmp_path, f"{OAK}/x.jpg"), {}, None)
        db.upsert_frame(FrameKey(13, 2, "oak1"), "", {}, None, {"missing": True})
        images = ImageCache(db, str(tmp_path / "cache"))
        assert images.image_path(key) is None
        why = images.why_unreadable(key)
        assert "原始数据盘没连上" in why and "raw data drive not found" in why
        assert images.view_problem(13, "oak1") == why
        # a missing frame is not "unreadable": there is nothing to read
        assert images.why_unreadable(FrameKey(13, 2, "oak1")) is None
    finally:
        db.close()


def test_a_file_that_is_not_there_names_the_resolved_path(tmp_path, monkeypatch):
    from tda.ui.session_images import ImageCache

    g = tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, g)
    RR.configure(raw_paths(tmp_path))
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        key = FrameKey(13, 1, "oak1")
        db.upsert_frame(key, recorded(tmp_path, f"{OAK}/gone.jpg"), {}, None)
        why = ImageCache(db, str(tmp_path / "cache")).why_unreadable(key)
        assert why.startswith("这一视角的原图读不到：")
        assert (g / MARKER).as_posix() in why and "gone.jpg" in why
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the canvas size: measured off the moved drive, recorded as provenance
# --------------------------------------------------------------------------- #
def test_frame_hw_measures_the_moved_file_and_records_the_stored_spelling(tmp_path, monkeypatch):
    from tda.core.truth_inputs import HW_MEASURED, frame_hw

    g = tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, g)
    rel = f"{OAK}/Desktop 13/Disassemble/Camera_1/041/x_rgb_12mp.jpg"
    write_jpg(g / MARKER / rel, hw=(30, 40))
    RR.configure(raw_paths(tmp_path))
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        key = FrameKey(13, 41, "oak1")
        db.upsert_frame(key, recorded(tmp_path, rel), {}, None)
        assert frame_hw(db, key, None) == (30, 40)
        aux = db.get_frame(key)["aux"]
        assert aux["hw_source"] == HW_MEASURED
        assert aux["cache_path"] == recorded(tmp_path, rel)   # not today's letter
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the local cache builder
# --------------------------------------------------------------------------- #
def test_build_cache_copies_from_the_moved_drive_and_keeps_the_recorded_src(tmp_path, monkeypatch):
    from tda.core.cache import build_cache
    from tda.core.index import DesktopIndex, FrameFile

    g = tmp_path / "G_"
    make_drive(g)
    fake_drives(monkeypatch, g)
    rel = f"{OAK}/Desktop 13/Disassemble/Camera_1/001/x_rgb_12mp.jpg"
    write_jpg(g / MARKER / rel)
    RR.configure(raw_paths(tmp_path))
    key = FrameKey(13, 1, "oak1")
    index = {13: DesktopIndex(13, 1, {key: FrameFile(key=key, path=recorded(tmp_path, rel),
                                                    aux={}, ts=None)}, [], [])}
    stats = build_cache(index, str(tmp_path / "cache"), views=("oak1",))
    assert stats["failures"] == [] and stats["copied"] == 1
    manifest = json.loads((tmp_path / "cache" / "oak1" / "D13" / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["1"]["src"] == recorded(tmp_path, rel)
    again = build_cache(index, str(tmp_path / "cache"), views=("oak1",))
    assert again["skipped"] == 1          # the size check found the moved source too


# --------------------------------------------------------------------------- #
# the command line: build-index and backup
# --------------------------------------------------------------------------- #
def _cli_env(tmp_path: Path) -> str:
    cfg = raw_paths(tmp_path)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    Db(cfg["db_path"]).close()
    out = tmp_path / "paths.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(out)


def _entries(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_build_index_refuses_to_scan_an_unplugged_drive(tmp_path, monkeypatch, capsys):
    from tda.cli import EXIT_ERROR, main

    (tmp_path / "F_").mkdir()
    fake_drives(monkeypatch, tmp_path / "F_")
    paths = _cli_env(tmp_path)
    out = tmp_path / "cache" / "index.json"
    assert main(["--paths", paths, "build-index", "--desktops", "13",
                 "--out", str(out)]) == EXIT_ERROR
    text = capsys.readouterr().out
    assert "raw data drive not found" in text and "index.json was not touched" in text
    assert not out.exists()


def _oak_step(g: Path, step: int, stamp: str) -> None:
    """One OAK camera-1 capture of D13 on the fake drive ``g``."""
    folder = g / MARKER / OAK / "Desktop 13" / "Disassemble" / "Camera_1" / f"{step:03d}"
    write_jpg(folder / f"{stamp}_camera_1_rgb_12mp.jpg")
    write_jpg(folder / f"{stamp}_camera_1_rgb_aligned.png")


def _build(paths: str, out: Path, tmp_path: Path) -> int:
    from tda.cli import main

    return main(["--paths", paths, "build-index", "--desktops", "13",
                 "--out", str(out), "--report", str(tmp_path / "r.md")])


def _recorded_spelling(tmp_path: Path, value) -> bool:
    """Is every path in ``value`` spelled under the recorded ``F_`` root?"""
    values = value if isinstance(value, list) else [value]
    base = (tmp_path / "F_" / MARKER).as_posix()
    return all(str(v).replace("\\", "/").startswith(base + "/") for v in values)


def test_build_index_scans_where_the_drive_is_today(tmp_path, monkeypatch, capsys):
    """It reads G:, and it *records* F: -- the stored value is provenance."""
    from tda.cli import EXIT_OK
    from tda.core.index import load_index

    g = tmp_path / "G_"
    make_drive(g)
    (tmp_path / "F_").mkdir()
    fake_drives(monkeypatch, tmp_path / "F_", g)
    _oak_step(g, 1, "20250603_135553_600")
    paths = _cli_env(tmp_path)
    out = tmp_path / "cache" / "index.json"
    assert _build(paths, out, tmp_path) == EXIT_OK
    assert "raw root: recorded" in capsys.readouterr().out
    loaded = load_index(str(out))[13]
    oak = [f for k, f in loaded.frames.items() if k.view == "oak1"]
    assert oak, "the scan of the moved drive found the capture"
    for frame in oak:
        assert _recorded_spelling(tmp_path, frame.path), frame.path
        assert all(_recorded_spelling(tmp_path, v) for v in frame.aux.values()), frame.aux
        assert Path(RR.resolve_raw(frame.path)).is_file()    # ... and it is there today
    assert not any((g / MARKER).as_posix() in text.replace("\\", "/")
                   for text in loaded.issues), loaded.issues
    assert (g / MARKER).as_posix() not in out.read_text(encoding="utf-8")


def test_rebuilding_and_reloading_keeps_every_stored_path_recorded(tmp_path, monkeypatch):
    """The reviewer's run: build-index + load-index rewrote 168/168 D13 paths to G:."""
    from tda.cli import EXIT_OK, main

    g = tmp_path / "G_"
    make_drive(g)
    (tmp_path / "F_").mkdir()
    fake_drives(monkeypatch, tmp_path / "F_", g)
    _oak_step(g, 1, "20250603_135553_600")
    paths = _cli_env(tmp_path)
    out = tmp_path / "cache" / "index.json"
    assert _build(paths, out, tmp_path) == EXIT_OK
    assert main(["--paths", paths, "load-index", "--index", str(out)]) == EXIT_OK

    _oak_step(g, 2, "20250603_135633_273")           # a genuinely new capture
    assert _build(paths, out, tmp_path) == EXIT_OK
    assert main(["--paths", paths, "load-index", "--index", str(out)]) == EXIT_OK

    db = Db(str(tmp_path / "annotations" / "tda.sqlite"))
    try:
        rows = db.frames_for(13, "oak1")
        stored = {int(r["step"]): r for r in rows if r.get("path")}
        assert set(stored) == {1, 2}
        for step, row in stored.items():
            assert _recorded_spelling(tmp_path, row["path"]), (step, row["path"])
            for value in (row.get("aux") or {}).values():
                assert _recorded_spelling(tmp_path, value), (step, value)
        meta = db.get_desktop(13) or {}
        assert not any((g / MARKER).as_posix() in str(t).replace("\\", "/")
                       for t in meta.get("index_issues") or [])
    finally:
        db.close()


def test_the_backup_command_follows_the_drive(tmp_path, monkeypatch, capsys):
    from tda.cli import EXIT_OK, main

    g = tmp_path / "G_"
    make_drive(g)
    (tmp_path / "F_").mkdir()
    fake_drives(monkeypatch, tmp_path / "F_", g)
    paths = _cli_env(tmp_path)
    assert main(["--paths", paths, "backup"]) == EXIT_OK
    assert list((g / MARKER / "TDA_backups").glob("tda_*.sqlite"))
    assert _entries(tmp_path / "F_") == []


def test_no_raw_drive_means_the_backup_command_says_so_and_creates_nothing(
        tmp_path, monkeypatch, capsys):
    from tda.cli import EXIT_ERROR, main

    (tmp_path / "F_").mkdir()
    fake_drives(monkeypatch, tmp_path / "F_")
    paths = _cli_env(tmp_path)
    assert main(["--paths", paths, "backup"]) == EXIT_ERROR
    text = capsys.readouterr().out
    assert "[backup] backup failed:" in text and "no backup" in text
    assert _entries(tmp_path / "F_") == []


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _raw_window(tmp_path: Path, monkeypatch, plugged: bool):
    """The D13 scan scene, plus an ``oak1`` view whose files are on the raw drive."""
    from app_scene import StubSamQueue, make_db
    from tda.core.truth import TruthService
    from tda.ui.app import MainWindow
    from tda.ui.session import AnnotationSession

    g = tmp_path / "G_"
    (tmp_path / "F_").mkdir(exist_ok=True)
    if plugged:
        make_drive(g)
    fake_drives(monkeypatch, tmp_path / "F_", g)
    db, scene_paths, tax = make_db(tmp_path)
    for step in (1, 2):
        rel = f"{OAK}/Desktop 13/Disassemble/Camera_1/{step:03d}/x_rgb_12mp.jpg"
        write_jpg(g / MARKER / rel, hw=(64, 64))
        db.upsert_frame(FrameKey(13, step, "oak1"), recorded(tmp_path, rel), {}, None,
                        {"missing": False})
    paths = dict(scene_paths, **{k: v for k, v in raw_paths(tmp_path).items()
                                 if k not in ("cache_dir", "db_path")})
    RR.configure(paths)
    session = AnnotationSession(db, tax, TruthService(db, tax), scene_paths["cache_dir"],
                                "tester")
    session.open(13, "oak1")
    return MainWindow(session, paths, "tester", sam_queue=StubSamQueue()), g


def test_the_window_says_once_that_the_raw_drive_is_missing(qapp, tmp_path, monkeypatch):
    from app_scene import close_window

    import shutil

    win, g = _raw_window(tmp_path, monkeypatch, plugged=False)
    shutil.rmtree(g, ignore_errors=True)          # the drive is not there at all
    try:
        win.act_recheck_raw_data()
        assert not win.raw_root.connected
        assert win.raw_bar.isVisibleTo(win)
        assert "原始数据盘没连上" in win.raw_bar_text()
        assert "插上 Elements 盘后按 F5" in win.raw_bar_text()
        assert win.view_buttons["oak1"].property("unreadable") is True
        assert "no readable image" in win.view_buttons["oak1"].toolTip()
        assert win.view_buttons["scan"].property("unreadable") is False  # local cache
        assert win.stack.currentWidget() is win.placeholder_label
        assert "原始数据盘没连上" in win.placeholder_label.text()
    finally:
        close_window(win)


def test_f5_finds_the_drive_once_it_is_plugged_in(qapp, tmp_path, monkeypatch):
    from app_scene import close_window

    win, g = _raw_window(tmp_path, monkeypatch, plugged=False)
    try:
        assert win.raw_bar.isVisibleTo(win)
        assert win.canvas.image_rgb() is None or win.stack.currentWidget() is win.placeholder_label
        make_drive(g)                     # plugged in now
        win.act_refresh_all()
        assert win.raw_root.connected
        assert not win.raw_bar.isVisibleTo(win)
        assert win.stack.currentWidget() is win.canvas
        assert win.canvas.image_rgb().shape == (64, 64, 3)
        assert win.view_buttons["oak1"].property("unreadable") is False
        assert "raw data drive found" in win.status_message()
    finally:
        close_window(win)


def test_a_connected_moved_drive_shows_no_bar_and_logs_one_line(qapp, tmp_path, monkeypatch):
    from app_scene import close_window
    from tda.ui import app_support as S

    win, g = _raw_window(tmp_path, monkeypatch, plugged=True)
    try:
        assert win.raw_root.status == RR.MOVED
        assert not win.raw_bar.isVisibleTo(win)
        assert win.canvas.image_rgb().shape == (64, 64, 3)
        log = Path(S.log_path(win.paths)).read_text(encoding="utf-8")
        assert log.count("raw root: recorded") == 1
    finally:
        close_window(win)


def test_the_exit_backup_is_skipped_without_the_drive_and_closing_still_works(
        qapp, tmp_path, monkeypatch):
    from tda.ui import app_support as S

    win, _g = _raw_window(tmp_path, monkeypatch, plugged=False)
    win.close()
    assert win.closed is True
    assert "exit backup skipped" in win.status_message()
    assert _entries(tmp_path / "F_") == []        # no TDA_backups on the wrong drive
    # ... and the log says so: it used to be written after shutdown() had
    # already closed the log file, so it went nowhere
    log = Path(S.log_path(win.paths)).read_text(encoding="utf-8")
    assert "exit backup skipped" in log and "window closed" in log
