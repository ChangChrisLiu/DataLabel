"""The remaining CLI commands: app, check, build-cache, export-coco, export-vlm.

They all run against the temporary D13 workspace of :mod:`tests.app_scene`, so
nothing reads ``F:`` or the real database.  ``app`` is checked at the wiring
level only -- that the arguments reach :func:`tda.ui.app.main` and that its exit
code is passed through -- because launching a window belongs in the GUI tests.
"""
from __future__ import annotations

import json
import os

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
def test_build_cache_delegates_to_the_cache_module(env, monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr("tda.core.cache.main", lambda argv: seen.append(list(argv)) or 0)
    code = run(env, "build-cache", "--views", "scan", "--first", "13", "--last", "13")
    assert code == EXIT_OK
    assert seen and "--views" in seen[0] and "scan" in seen[0]
    assert "--cache" in seen[0]


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


def test_app_passes_the_exit_code_through(env, monkeypatch):
    monkeypatch.setattr("tda.ui.app.main", lambda **kwargs: 3)
    assert run(env, "app", "--desktop", str(DESKTOP), "--view", VIEW,
               "--annotator", "chang") == EXIT_LOCKED
