"""``check`` and the exports must not ignore a standing open conflict.

An open conflict is a frozen row the recompile disagrees with: somebody has to
look at it. ``check`` used to print the conflicts *this run* raised and say
nothing about the ones already in the queue, and the exports shipped over them.

The core half -- ``export_coco/export_vlm(..., allow_conflicts=False)`` and
``TruthService.open_conflicts`` -- belongs to worker F1, so the wiring here is
written against those names through a small adapter and exercised with stubs:
:func:`tda.cli_app._open_conflicts` falls back to ``Db.conflicts`` until the
service grows the method, and :func:`tda.cli_app._call_export` passes
``allow_conflicts`` only to an exporter that takes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from app_scene import DESKTOP, VIEW, make_db, write_paths_yaml
from tda.cli import EXIT_ERROR, EXIT_OK, main
from tda.core.db import Db
from tda.core.model import FrameKey


@pytest.fixture
def env(tmp_path: Path) -> dict:
    db, paths, _tax = make_db(tmp_path)
    db.close()
    return {"paths": write_paths_yaml(tmp_path), "cfg": paths, "tmp": tmp_path}


def run(env: dict, *argv: str) -> int:
    return main(["--paths", env["paths"], *argv])


def open_one_conflict(env: dict, step: int = 2) -> int:
    """Queue one open conflict on the scene's desktop/view; returns its id."""
    db = Db(env["cfg"]["db_path"])
    try:
        return db.add_conflict(FrameKey(DESKTOP, step, VIEW), "chassis", None, None, 17)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def test_check_reports_the_open_conflicts_it_found_waiting(env, capsys):
    open_one_conflict(env)
    code = run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW)
    out = capsys.readouterr().out
    assert "open conflicts: 1" in out
    assert code == EXIT_ERROR


def test_check_says_zero_open_conflicts_on_a_clean_desktop(env, capsys):
    run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW)
    assert "open conflicts: 0" in capsys.readouterr().out


def test_check_exits_one_for_an_open_conflict_even_with_no_problems(env, capsys, monkeypatch):
    from tda import cli_app

    real = cli_app._prepare_truth

    def clean(db, tax, desktops, view, refresh=True):
        stats = real(db, tax, desktops, view, refresh)
        stats["problems"] = []
        return stats

    monkeypatch.setattr(cli_app, "_prepare_truth", clean)
    open_one_conflict(env)
    code = run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW)
    out = capsys.readouterr().out
    assert "problems: 0" in out and "open conflicts: 1" in out
    assert code == EXIT_ERROR


def test_check_uses_the_truth_services_own_open_conflicts_when_it_has_one(env, monkeypatch, capsys):
    """F1's ``TruthService.open_conflicts`` wins over the raw table query."""
    from tda.core.truth import TruthService

    monkeypatch.setattr(TruthService, "open_conflicts",
                        lambda self, desktop, view: [{"id": 1}, {"id": 2}], raising=False)
    code = run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW)
    assert "open conflicts: 2" in capsys.readouterr().out
    assert code == EXIT_ERROR


# --------------------------------------------------------------------------- #
# the exports
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("command,out_name", [
    ("export-coco", "coco.json"),
    ("export-vlm", "vlm.jsonl"),
])
def test_an_export_refuses_while_a_conflict_is_open(env, capsys, command, out_name):
    open_one_conflict(env)
    out = Path(env["tmp"]) / out_name
    code = run(env, command, "--desktops", str(DESKTOP), "--view", VIEW, "--out", str(out))
    printed = capsys.readouterr().out
    assert code == EXIT_ERROR
    assert "1 open conflict" in printed
    assert "--allow-conflicts" in printed
    assert not out.exists()


@pytest.mark.parametrize("command,out_name", [
    ("export-coco", "coco_allowed.json"),
    ("export-vlm", "vlm_allowed.jsonl"),
])
def test_allow_conflicts_lets_the_export_through(env, command, out_name):
    open_one_conflict(env)
    out = Path(env["tmp"]) / out_name
    code = run(env, command, "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out), "--allow-conflicts")
    assert code == EXIT_OK
    assert out.exists()


def test_the_flag_reaches_an_exporter_that_takes_it(env, monkeypatch):
    """F1's signature: ``export_coco(..., allow_conflicts=False)``."""
    from tda.core.export import coco as coco_module

    seen: dict = {}

    def fake(db, tax, desktops, view, out, *, allow_conflicts=False, **kw):
        seen["allow_conflicts"] = allow_conflicts
        Path(out).write_text("{}", encoding="utf-8")
        return {"images": 0, "annotations": 0}

    monkeypatch.setattr(coco_module, "export_coco", fake)
    out = Path(env["tmp"]) / "coco_stub.json"
    assert run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out), "--allow-conflicts") == EXIT_OK
    assert seen["allow_conflicts"] is True


def test_an_exporter_without_the_keyword_still_runs(env, monkeypatch):
    """Today's signature: the adapter must not turn it into a TypeError."""
    from tda.core.export import coco as coco_module

    def old_style(db, tax, desktops, view, out, only_verified=True, roi_crop=False,
                  truth=None):
        Path(out).write_text("{}", encoding="utf-8")
        return {"images": 0, "annotations": 0}

    monkeypatch.setattr(coco_module, "export_coco", old_style)
    out = Path(env["tmp"]) / "coco_old.json"
    assert run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
               "--out", str(out)) == EXIT_OK


def test_a_type_error_from_inside_the_exporter_is_not_swallowed(env, monkeypatch):
    from tda.core.export import coco as coco_module

    def broken(db, tax, desktops, view, out, **kw):
        raise TypeError("something inside the exporter is wrong")

    monkeypatch.setattr(coco_module, "export_coco", broken)
    with pytest.raises(TypeError, match="inside the exporter"):
        run(env, "export-coco", "--desktops", str(DESKTOP), "--view", VIEW,
            "--out", str(Path(env["tmp"]) / "coco_boom.json"))
