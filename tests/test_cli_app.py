"""The remaining CLI commands: app, check, build-cache, export-coco, export-vlm.

They all run against the temporary D13 workspace of :mod:`tests.app_scene`, so
nothing reads ``F:`` or the real database.  ``app`` is checked at the wiring
level only -- that the arguments reach :func:`tda.ui.app.main` and that its exit
code is passed through -- because launching a window belongs in the GUI tests.
"""
from __future__ import annotations

import json
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from app_scene import DESKTOP, LAST_STEP, VIEW, make_db, write_paths_yaml
from tda.cli import EXIT_ERROR, EXIT_LOCKED, EXIT_OK, main
from tda.core.db import Db
from tda.core.model import FrameKey


@pytest.fixture
def env(tmp_path: Path) -> dict:
    db, paths, _tax = make_db(tmp_path)
    db.close()
    return {"paths": write_paths_yaml(tmp_path), "cfg": paths, "tmp": tmp_path}


def run(env: dict, *argv: str) -> int:
    return main(["--paths", env["paths"], *argv])


def empty_workspace(tmp_path: Path) -> dict:
    """Frames and a pose segment, but nothing that needs geometry."""
    paths_yaml = write_paths_yaml(tmp_path)
    cfg_db = tmp_path / "annotations" / "tda.sqlite"
    cfg_db.parent.mkdir(parents=True, exist_ok=True)
    db = Db(str(cfg_db))
    db.upsert_desktop(99, {"brand": "Test"})
    for step in (1, 2):
        db.upsert_frame(FrameKey(99, step, VIEW), "", {"hw": [64, 64]}, None)
    db.set_pose_segment(99, VIEW, 1, 1, 2, 2, None, None)
    db.close()
    return {"paths": paths_yaml, "tmp": tmp_path}


# --------------------------------------------------------------------------- #
# the commands are registered
# --------------------------------------------------------------------------- #
def test_every_new_command_is_in_the_help():
    from tda.cli import build_parser

    text = build_parser().format_help()
    for command in ("app", "check", "build-cache", "export-coco", "export-vlm"):
        assert command in text


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def test_check_prints_zero_problems_on_a_clean_view(tmp_path, capsys):
    env = empty_workspace(tmp_path)
    code = main(["--paths", env["paths"], "check", "--desktop", "99", "--view", VIEW])
    out = capsys.readouterr().out
    assert "problems: 0" in out
    assert code == EXIT_OK


def test_check_counts_the_problems_and_exits_one(env, capsys):
    code = run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW)
    out = capsys.readouterr().out
    assert "problems: " in out
    count = int(out.split("problems: ")[1].split()[0])
    assert count > 0
    assert code == EXIT_ERROR


def test_check_refuses_while_another_annotator_holds_the_lock(env, capsys):
    db = Db(env["cfg"]["db_path"])
    db.acquire_lock("someone_else")
    db.close()
    try:
        code = run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW)
        assert code == EXIT_LOCKED
    finally:
        db = Db(env["cfg"]["db_path"])
        db.release_lock()
        db.close()


# --------------------------------------------------------------------------- #
# exports
# --------------------------------------------------------------------------- #
def test_export_coco_prints_a_summary_not_the_document(env, capsys):
    """``stats['images']`` is the image *list*: printing it dumped the whole COCO."""
    out = Path(env["tmp"]) / "coco_summary.json"
    assert run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out)) == EXIT_OK
    line = [l for l in capsys.readouterr().out.splitlines()
            if l.startswith("[export-coco]")][-1]
    assert len(line) < 200
    assert re.fullmatch(r"\[export-coco\] \d+ images, \d+ annotations -> .+", line), line


def test_export_vlm_prints_a_summary_not_the_records(env, capsys):
    out = Path(env["tmp"]) / "vlm_summary.jsonl"
    assert run(env, "export-vlm", "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out)) == EXIT_OK
    line = [l for l in capsys.readouterr().out.splitlines()
            if l.startswith("[export-vlm]")][-1]
    assert len(line) < 200 and "records" in line


def test_build_cache_caches_exactly_the_desktops_asked_for(env, monkeypatch):
    """``1-3,13`` used to collapse into ``--first 1 --last 13``: 13 machines."""
    runs: list[tuple[int, int]] = []

    def fake_main(argv):
        argv = list(argv)
        runs.append((int(argv[argv.index("--first") + 1]),
                     int(argv[argv.index("--last") + 1])))
        return 0

    monkeypatch.setattr("tda.core.cache.main", fake_main)
    assert run(env, "build-cache", "--desktops", "1-3,13") == EXIT_OK
    covered = {d for first, last in runs for d in range(first, last + 1)}
    assert covered == {1, 2, 3, 13}


def test_export_coco_writes_a_file(env, capsys):
    out = Path(env["tmp"]) / "coco.json"
    code = run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out))
    assert code == EXIT_OK
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "images" in payload and "categories" in payload
    assert str(out) in capsys.readouterr().out


def test_export_coco_honours_only_verified(env):
    out = Path(env["tmp"]) / "coco_all.json"
    assert run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out), "--no-only-verified") == EXIT_OK
    assert out.exists()


def test_export_vlm_writes_a_jsonl(env):
    out = Path(env["tmp"]) / "vlm.jsonl"
    assert run(env, "export-vlm", "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out)) == EXIT_OK
    assert out.exists()
    for line in out.read_text(encoding="utf-8").splitlines()[:3]:
        assert json.loads(line)["task"]


# --------------------------------------------------------------------------- #
# build-cache
# --------------------------------------------------------------------------- #
def test_build_cache_takes_the_same_desktop_spec_as_everything_else(env, monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr("tda.core.cache.main", lambda argv: seen.append(list(argv)) or 0)
    code = run(env, "build-cache", "--views", "scan", "--desktops", "13-15")
    assert code == EXIT_OK
    argv = seen[0]
    assert "--views" in argv and "scan" in argv and "--cache" in argv
    assert argv[argv.index("--first") + 1] == "13"
    assert argv[argv.index("--last") + 1] == "15"


def test_build_cache_still_accepts_first_and_last(env, monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr("tda.core.cache.main", lambda argv: seen.append(list(argv)) or 0)
    assert run(env, "build-cache", "--first", "13", "--last", "13") == EXIT_OK
    assert seen[0][seen[0].index("--last") + 1] == "13"


# --------------------------------------------------------------------------- #
# the truth is brought up to date before it is read
# --------------------------------------------------------------------------- #
def test_check_export_and_vlm_all_drain_the_pending_rechecks(env, monkeypatch):
    """One helper, three callers: a stale re-check must not reach an export."""
    from tda import cli_app

    calls: list[tuple] = []
    real = cli_app._prepare_truth

    def spy(db, tax, desktops, view, refresh=True):
        calls.append((tuple(desktops), view, refresh))
        return real(db, tax, desktops, view, refresh)

    monkeypatch.setattr(cli_app, "_prepare_truth", spy)
    out = Path(env["tmp"]) / "coco_prep.json"
    run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW)
    run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
        "--out", str(out))
    run(env, "export-vlm", "--desktops", str(DESKTOP), "--view", VIEW,
        "--out", str(Path(env["tmp"]) / "vlm_prep.jsonl"))
    assert [c[0] for c in calls] == [(DESKTOP,), (DESKTOP,), (DESKTOP,)]


def test_an_export_refuses_while_re_checks_are_pending(env, monkeypatch, capsys):
    from tda import cli_app

    monkeypatch.setattr(cli_app, "_pending_rechecks", lambda truth, d, v: [3, 4])
    out = Path(env["tmp"]) / "coco_blocked.json"
    code = run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out), "--no-refresh")
    assert code == EXIT_ERROR
    assert "pending" in capsys.readouterr().out
    assert not out.exists()


def test_the_exports_take_the_lock(env):
    db = Db(env["cfg"]["db_path"])
    db.acquire_lock("someone_else")
    db.close()
    try:
        out = Path(env["tmp"]) / "coco_locked.json"
        assert run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
                   "--out", str(out)) == EXIT_LOCKED
        assert run(env, "export-vlm", "--desktops", str(DESKTOP), "--view", VIEW,
                   "--out", str(Path(env["tmp"]) / "v.jsonl")) == EXIT_LOCKED
    finally:
        db = Db(env["cfg"]["db_path"])
        db.release_lock()
        db.close()


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
def test_app_passes_its_arguments_to_the_window_entry_point(env, monkeypatch):
    seen: dict = {}

    def fake_main(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr("tda.ui.app.main", fake_main)
    code = run(env, "app", "--desktop", str(DESKTOP), "--view", VIEW,
               "--annotator", "chang", "--step", str(LAST_STEP))
    assert code == EXIT_OK
    assert seen["desktop"] == DESKTOP and seen["view"] == VIEW
    assert seen["annotator"] == "chang" and seen["step"] == LAST_STEP
    assert seen["paths"] == env["paths"]


def test_app_defaults_to_the_last_frame_of_that_annotator(env, monkeypatch):
    """``--desktop``/``--view``/``--step`` are optional: resume where you were."""
    seen: dict = {}
    monkeypatch.setattr("tda.ui.app.main", lambda **kw: seen.update(kw) or 0)
    assert run(env, "app", "--annotator", "chang") == EXIT_OK
    assert seen["desktop"] is None and seen["view"] is None and seen["step"] is None


def test_resume_target_reads_the_ini_then_falls_back(qapp_or_none, tmp_path):
    from app_scene import make_paths
    from tda.ui import app_support as S
    from tda.ui.app_shell import resume_target

    db, paths, _tax = make_db(tmp_path)
    db.close()
    settings = S.make_settings(paths)
    settings.setValue("last/chang/desktop", DESKTOP)
    settings.setValue("last/chang/view", VIEW)
    settings.setValue("last/chang/step", 7)
    settings.sync()

    resumed = resume_target(paths, "chang", None, None, None, paths["db_path"])
    assert resumed == {"desktop": DESKTOP, "view": VIEW, "step": 7}
    fresh = resume_target(make_paths(tmp_path), "nobody", None, None, None,
                          paths["db_path"])
    assert fresh["desktop"] == DESKTOP and fresh["view"] == "scan"
    assert fresh["step"] is None
    explicit = resume_target(paths, "chang", 42, "rs", None, paths["db_path"])
    assert explicit == {"desktop": 42, "view": "rs", "step": None}


@pytest.fixture(scope="module")
def qapp_or_none():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_app_passes_the_exit_code_through(env, monkeypatch):
    monkeypatch.setattr("tda.ui.app.main", lambda **kwargs: 3)
    assert run(env, "app", "--desktop", str(DESKTOP), "--view", VIEW,
               "--annotator", "chang") == EXIT_LOCKED
