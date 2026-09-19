"""Three things the end-to-end rehearsal on real data caught, in a child process.

* an export that found nothing still wrote the file and exited 0, so a day of
  annotation came back as an empty COCO and a zero-byte JSONL with no hint why;
* ``--paths`` was only accepted *before* the subcommand, so ``status --paths X``
  answered "unrecognized arguments";
* the exports spell it ``--view`` and ``build-cache`` spells it ``--views``, and
  each rejected the other.

They are checked through ``sys.executable`` because that is how the rehearsal
found them: the exit code and the terminal output are the whole interface here.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
import yaml
from test_cli_subprocess import output_lines, run_cli

from app_scene import DESKTOP, VIEW, freeze_mask, make_db, write_paths_yaml
from tda.core.db import Db
from tda.core.model import FrameKey


SECOND_VIEW = "oak1"


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """The seeded D13 workspace with verified frames in two views.

    ``app_scene`` alone confirms nothing, which is exactly the state that made
    an export write an empty file and exit 0 -- that is the ``bare`` fixture's
    job here. This one is the *successful* case the warning must stay out of.
    """
    db, _paths, _tax = make_db(tmp_path)
    try:
        for step in (1, 2):
            db.upsert_frame(FrameKey(DESKTOP, step, SECOND_VIEW), "",
                            {"hw": [64, 64]}, None)
        db.set_pose_segment(DESKTOP, SECOND_VIEW, 1, 1, 2, 2, None, None)
        freeze_mask(db, VIEW)
        freeze_mask(db, SECOND_VIEW)
    finally:
        db.close()
    write_paths_yaml(tmp_path)
    return tmp_path


@pytest.fixture
def bare(tmp_path: Path) -> Path:
    """Frames and a pose segment, but not one verified row anywhere."""
    cfg = {
        "cache_dir": str(tmp_path / "cache"),
        "db_path": str(tmp_path / "annotations" / "tda.sqlite"),
        "backup_dir": str(tmp_path / "backups"),
        "raw_logs_dir": str(tmp_path / "raw_logs"),
    }
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    db = Db(cfg["db_path"])
    try:
        db.upsert_desktop(99, {"brand": "Bare"})
        for step in (1, 2):
            db.upsert_frame(FrameKey(99, step, VIEW), "", {"hw": [64, 64]}, None)
        db.set_pose_segment(99, VIEW, 1, 1, 2, 2, None, None)
    finally:
        db.close()
    return tmp_path


# --------------------------------------------------------------------------- #
# (a) an export that found nothing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("command,name", [
    ("export-coco", "coco.json"),
    ("export-vlm", "vlm.jsonl"),
])
def test_an_export_with_nothing_to_write_warns_and_fails(bare, command, name):
    out = bare / name
    proc = run_cli(bare, ("-m", "tda.cli"), command, "--desktops", "99",
                   "--view", VIEW, "--out", str(out))
    printed = proc.stdout + proc.stderr
    assert "WARNING" in printed
    assert "nothing exported" in printed
    assert proc.returncode == 1, printed


def test_the_warning_offers_the_flag_that_would_have_helped(bare):
    """Only where it would: export-vlm already includes the unverified rows."""
    proc = run_cli(bare, ("-m", "tda.cli"), "export-coco", "--desktops", "99",
                   "--view", VIEW, "--out", str(bare / "coco.json"))
    assert "0 verified frames" in proc.stdout + proc.stderr
    assert "--no-only-verified" in proc.stdout + proc.stderr
    proc = run_cli(bare, ("-m", "tda.cli"), "export-vlm", "--desktops", "99",
                   "--view", VIEW, "--out", str(bare / "vlm.jsonl"))
    assert "no exportable frames" in proc.stdout + proc.stderr
    assert "--no-only-verified" not in proc.stdout + proc.stderr


@pytest.mark.parametrize("command,name", [
    ("export-coco", "coco.json"),
    ("export-vlm", "vlm.jsonl"),
])
def test_an_export_with_nothing_to_write_leaves_no_file_behind(bare, command, name):
    out = bare / name
    run_cli(bare, ("-m", "tda.cli"), command, "--desktops", "99", "--view", VIEW,
            "--out", str(out))
    assert not out.exists()


@pytest.mark.parametrize("command", ["export-coco", "export-vlm"])
def test_an_export_always_says_how_many_frames_it_considered(scene, command):
    proc = run_cli(scene, ("-m", "tda.cli"), command, "--desktops", str(DESKTOP),
                   "--view", VIEW, "--out", str(scene / f"{command}.out"))
    printed = proc.stdout + proc.stderr
    assert "verified" in printed and "not verified" in printed


def test_a_successful_export_still_exits_zero_and_keeps_its_file(scene):
    out = scene / "coco_ok.json"
    proc = run_cli(scene, ("-m", "tda.cli"), "export-coco", "--desktops", str(DESKTOP),
                   "--view", VIEW, "--out", str(out), "--allow-conflicts")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists() and out.stat().st_size > 2


# --------------------------------------------------------------------------- #
# (b) --paths on either side of the subcommand
# --------------------------------------------------------------------------- #
def test_paths_is_accepted_after_the_subcommand(scene):
    proc = run_cli(scene, ("-m", "tda.cli"), "status", "--paths", str(scene / "paths.yaml"))
    printed = proc.stdout + proc.stderr
    assert "unrecognized arguments" not in printed
    assert proc.returncode == 0, printed


def test_the_later_paths_wins(scene):
    """The global one is a default; the one next to the subcommand overrides it."""
    bad = scene / "bad"
    bad.mkdir()
    (bad / "paths.yaml").write_text("db_path: [unclosed\n", encoding="utf-8")
    proc = run_cli(bad, ("-m", "tda.cli"), "status", "--paths", str(scene / "paths.yaml"))
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
# (c) --view and --views everywhere
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spelling", ["--view", "--views"])
@pytest.mark.parametrize("command", ["export-coco", "export-vlm", "check"])
def test_both_spellings_are_accepted(scene, command, spelling):
    argv = [command, spelling, VIEW]
    argv += ["--desktop", str(DESKTOP)] if command == "check" \
        else ["--desktops", str(DESKTOP), "--out", str(scene / f"{command}{spelling}.out")]
    proc = run_cli(scene, ("-m", "tda.cli"), *argv)
    printed = proc.stdout + proc.stderr
    assert "unrecognized arguments" not in printed
    assert "invalid choice" not in printed


def test_build_cache_accepts_the_singular_spelling(scene):
    proc = run_cli(scene, ("-m", "tda.cli"), "build-cache", "--view", VIEW,
                   "--desktops", "999")
    assert "unrecognized arguments" not in proc.stdout + proc.stderr


def test_check_accepts_a_comma_separated_view_list(scene):
    proc = run_cli(scene, ("-m", "tda.cli"), "check", "--desktop", str(DESKTOP),
                   "--views", f"{VIEW},{SECOND_VIEW}")
    printed = proc.stdout + proc.stderr
    assert printed.count("problems: ") == 2, printed


def test_an_unknown_view_is_one_line_and_exit_one(scene):
    proc = run_cli(scene, ("-m", "tda.cli"), "check", "--desktop", str(DESKTOP),
                   "--views", "sideways")
    lines = output_lines(proc)
    assert "Traceback" not in proc.stdout + proc.stderr
    assert len(lines) == 1 and "sideways" in lines[0]
    assert proc.returncode == 1


def test_an_export_of_several_views_refuses_a_single_out(scene):
    proc = run_cli(scene, ("-m", "tda.cli"), "export-coco", "--desktops", str(DESKTOP),
                   "--views", f"{VIEW},{SECOND_VIEW}", "--out", str(scene / "one.json"))
    printed = proc.stdout + proc.stderr
    assert "--out" in printed
    assert proc.returncode == 1


def test_an_export_of_several_views_writes_one_file_each(scene):
    proc = run_cli(scene, ("-m", "tda.cli"), "export-coco", "--desktops", str(DESKTOP),
                   "--views", f"{VIEW},{SECOND_VIEW}", "--allow-conflicts")
    printed = proc.stdout + proc.stderr
    assert f"coco_{VIEW}.json" in printed and f"coco_{SECOND_VIEW}.json" in printed
