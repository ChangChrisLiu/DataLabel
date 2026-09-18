"""Tests for the command-line data pipeline (``tda.cli`` + ``tda.pipeline``).

Every test runs against a throw-away ``paths.yaml`` whose four roots point into
``tmp_path``, so no test reads F: or writes to the real database. The frame
index is a small synthetic :class:`~tda.core.index.DesktopIndex` written with
the real ``save_index``; the logs are the three verbatim fixtures in
``tests/fixtures/logs`` plus one synthesised sheet for the ``reorient`` case.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from tda.cli import main
from tda.core.db import Db
from tda.core.index import DesktopIndex, FrameFile, save_index
from tda.core.logs import read_desktop_csv
from tda.core.model import VIEWS, FrameKey, StepRec, StepType
from tda.pipeline import parse_desktops

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LOG_FIXTURES = FIXTURES / "logs"
LS_EXPORT = FIXTURES / "ls_small.json"
#: the desktops the ls_small.json fixture annotates
LS_DESKTOPS = (19, 24)
STEP_SECONDS = 90


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_index(
    desktop: int, n_steps: int, missing: tuple[tuple[int, str], ...] = ()
) -> DesktopIndex:
    """A synthetic index: every view of every step, minus ``missing``.

    Only ``oak1`` carries a timestamp (the source of the step durations), one
    every :data:`STEP_SECONDS`, exactly as the real OAK filename tokens do.
    """
    base = datetime(2025, 1, 1, 10, 0, 0)
    frames: dict[FrameKey, FrameFile] = {}
    for step in range(1, n_steps + 1):
        for view in VIEWS:
            if (step, view) in missing:
                continue
            key = FrameKey(desktop, step, view)
            aux = {"burst": [f"F:/fake/{view}/D{desktop}/s{step}_P0.png"]} if view == "scan" else {}
            frames[key] = FrameFile(
                key=key,
                path=f"F:/fake/{view}/D{desktop}/s{step}.png",
                aux=aux,
                ts=(base + timedelta(seconds=STEP_SECONDS * (step - 1))).isoformat()
                if view == "oak1"
                else None,
                src_step_dir=f"{step:03d}",
            )
    return DesktopIndex(
        desktop=desktop,
        n_steps=n_steps,
        frames=frames,
        missing=[FrameKey(desktop, step, view) for step, view in missing],
        issues=[f"scan: RGB{missing[0][0] if missing else 1}1 holds no P_k.png; skipped"],
    )


#: A sheet whose step 3 flips the chassis, and one whose last step does.
REORIENT_SHEET = [
    "Initial Conditions",
    "CPU Fan Screw 1",
    "change a direction",
    "Motherboard Screw 1",
    "Motherboard",
]
TRAILING_REORIENT_SHEET = [
    "Initial Conditions",
    "CPU Fan Screw 1",
    "Motherboard Screw 1",
    "Motherboard",
    "change a direction",
]


def write_reorient_log(
    directory: Path, desktop: int = 77, names: list[str] | None = None
) -> int:
    """Write a small sheet with the usual columns; returns its step count."""
    names = names or REORIENT_SHEET
    path = directory / f"desktop_{desktop:02d}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Sequence Number", "Sequence Name", "Target Nest Group",
                         "Tool Utility", "Notes"])
        for i, name in enumerate(names, start=1):
            writer.writerow([i, name, "", "Philips PH2" if "Screw" in name else "", ""])
    with open(directory / f"desktop_{desktop:02d}_meta.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Desktop ID", desktop])
        writer.writerow(["Desktop Brand", "Dell Optiplex 7020"])
        writer.writerow(["Desktop Size", "12*11*3"])
        writer.writerow(["Collection Date", "2025-06-03 00:00:00"])
    return len(names)


@pytest.fixture
def d13_steps() -> int:
    """How many step rows the real desktop 13 fixture yields."""
    rows, _meta = read_desktop_csv(LOG_FIXTURES / "desktop_13.csv")
    return len(rows)


@pytest.fixture
def env(tmp_path: Path, d13_steps: int) -> dict:
    """A complete tmp workspace: paths.yaml, index.json, a drive log folder."""
    cache = tmp_path / "cache"
    drive = tmp_path / "raw_logs" / "drive"
    drive.mkdir(parents=True)
    for name in ("desktop_01", "desktop_13", "desktop_63"):
        for suffix in ("", "_meta"):
            shutil.copy(LOG_FIXTURES / f"{name}{suffix}.csv", drive / f"{name}{suffix}.csv")
    n77 = write_reorient_log(drive)
    n78 = write_reorient_log(drive, 78, TRAILING_REORIENT_SHEET)

    index = {
        13: make_index(13, d13_steps, missing=((3, "scan"), (5, "rs"))),
        77: make_index(77, n77),
        # D78's sheet has one step more than the index: its trailing reorient was
        # never photographed, like the real D63
        78: make_index(78, n78 - 1),
    }
    cache.mkdir(parents=True)
    save_index(index, str(cache / "index.json"))

    cfg = {
        "oak_root": str(tmp_path / "src" / "oak"),
        "scanner_root": str(tmp_path / "src" / "scan"),
        "rs_root": str(tmp_path / "src" / "rs"),
        "cache_dir": str(cache),
        "db_path": str(tmp_path / "annotations" / "tda.sqlite"),
        "backup_dir": str(tmp_path / "backups"),
        "raw_logs_dir": str(tmp_path / "raw_logs"),
        "weights_dir": str(tmp_path / "weights"),
    }
    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {
        "paths": str(paths_yaml),
        "cfg": cfg,
        "db_path": cfg["db_path"],
        "cache": cache,
        "drive": drive,
        "n_steps_13": d13_steps,
        "n_steps_77": n77,
        "n_steps_78": n78,
    }


def run(env: dict, *argv: str) -> int:
    """Run the CLI against the tmp workspace."""
    return main(["--paths", env["paths"], *argv])


def open_db(env: dict) -> Db:
    return Db(env["db_path"])


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def test_parse_desktops_accepts_ranges_and_lists():
    assert parse_desktops("13") == {13}
    assert parse_desktops("1-4") == {1, 2, 3, 4}
    assert parse_desktops("1-3,13,60-61") == {1, 2, 3, 13, 60, 61}
    assert parse_desktops(None) is None
    with pytest.raises(ValueError):
        parse_desktops("4-1")
    with pytest.raises(ValueError):
        parse_desktops("abc")


# --------------------------------------------------------------------------- #
# load-index
# --------------------------------------------------------------------------- #
def test_load_index_creates_frames(env):
    assert run(env, "load-index") == 0
    db = open_db(env)
    try:
        frames = db.frames_for(13, "oak1")
        assert len(frames) == env["n_steps_13"]
        assert frames[0]["path"].endswith("oak1/D13/s1.png")
        assert frames[0]["ts"] == "2025-01-01T10:00:00"
        assert frames[0]["missing"] is False
        scan = {f["step"]: f for f in db.frames_for(13, "scan")}
        assert scan[3]["missing"] is True
        assert scan[1]["missing"] is False
        assert scan[1]["aux"]["burst"]
    finally:
        db.close()


def test_load_index_creates_pose_segment_one(env):
    assert run(env, "load-index") == 0
    db = open_db(env)
    try:
        for view in VIEWS:
            seg = db.pose_segment_for(FrameKey(13, 1, view))
            assert (seg["seg"], seg["start_step"], seg["end_step"]) == (1, 1, env["n_steps_13"])
            assert seg["ref_step"] == env["n_steps_13"]
            assert seg["corners"] is None
    finally:
        db.close()


def test_load_index_stores_issues_in_desktop_meta(env):
    assert run(env, "load-index") == 0
    db = open_db(env)
    try:
        meta = db.get_desktop(13)
        assert meta["index_issues"] == ["scan: RGB31 holds no P_k.png; skipped"]
        assert meta["index_n_steps"] == env["n_steps_13"]
    finally:
        db.close()


def test_load_index_desktop_filter(env):
    assert run(env, "load-index", "--desktops", "77") == 0
    db = open_db(env)
    try:
        assert db.frames_for(13, "oak1") == []
        assert len(db.frames_for(77, "oak1")) == env["n_steps_77"]
    finally:
        db.close()


def test_load_index_keeps_the_corners_a_human_clicked(env):
    """Re-running the loader must not throw away the chassis ROI of segment 1."""
    corners = [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]
    assert run(env, "load-index") == 0
    db = open_db(env)
    try:
        db.set_pose_segment(13, "scan", 1, 1, env["n_steps_13"], 7, corners, None)
    finally:
        db.close()

    assert run(env, "load-index") == 0
    db = open_db(env)
    try:
        seg = db.pose_segment_for(FrameKey(13, 1, "scan"))
        assert seg["corners"] == corners
        assert seg["ref_step"] == 7
    finally:
        db.close()


def test_load_index_is_idempotent(env):
    assert run(env, "load-index") == 0
    assert run(env, "load-index") == 0
    db = open_db(env)
    try:
        assert len(db.frames_for(13, "oak1")) == env["n_steps_13"]
        rows = db.conn.execute(
            "SELECT COUNT(*) AS n FROM pose_segment WHERE desktop=13"
        ).fetchone()["n"]
        assert rows == len(VIEWS)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# import-logs
# --------------------------------------------------------------------------- #
def test_import_logs_fills_steps_actions_instances_events(env, capsys):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    db = open_db(env)
    try:
        steps = db.steps(13)
        assert len(steps) == env["n_steps_13"]
        assert steps[0].step_type == StepType.INITIAL.value
        assert db.actions(13)
        instances = db.instances(13)
        assert "chassis" in instances and len(instances) > 5
        assert db.events(13)
        assert all(e.auto for e in db.events(13))
    finally:
        db.close()


def test_import_logs_writes_desktop_meta_and_report(env):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    db = open_db(env)
    try:
        meta = db.get_desktop(13)
        assert meta["brand"] == "Dell"
        assert meta["model_family"] == "Optiplex 7020"
        assert meta["date"] == "2025-06-03"
        assert meta["size"] == "12.25*11.5*3.75"
        # the index issues written by load-index survive the log import
        assert meta["index_issues"]
    finally:
        db.close()
    report = env["cache"] / "import_logs_issues.md"
    assert report.exists()
    assert "D13" in report.read_text(encoding="utf-8")


def test_import_logs_step_durations_from_index(env):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    db = open_db(env)
    try:
        steps = db.steps(13)
        assert steps[0].duration_s is None  # no predecessor
        assert steps[1].duration_s == pytest.approx(STEP_SECONDS)
        assert steps[-1].duration_s == pytest.approx(STEP_SECONDS)
    finally:
        db.close()


def test_import_logs_skips_existing_unless_forced(env, capsys):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    db = open_db(env)
    try:
        steps = db.steps(13)
        steps[1].notes = "hand written"
        db.replace_steps(13, steps, db.actions(13))
    finally:
        db.close()

    capsys.readouterr()
    assert run(env, "import-logs") == 0
    assert "skip" in capsys.readouterr().out.lower()
    db = open_db(env)
    try:
        assert db.steps(13)[1].notes == "hand written"
    finally:
        db.close()

    assert run(env, "import-logs", "--desktops", "13", "--force") == 0
    db = open_db(env)
    try:
        assert db.steps(13)[1].notes != "hand written"
    finally:
        db.close()


def test_import_logs_keeps_the_report_when_nothing_was_imported(env):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    report = env["cache"] / "import_logs_issues.md"
    before = report.read_text(encoding="utf-8")
    assert run(env, "import-logs") == 0  # everything is skipped this time
    assert report.read_text(encoding="utf-8") == before


def test_reorient_step_splits_pose_segments(env):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    db = open_db(env)
    try:
        assert db.steps(77)[2].step_type == StepType.REORIENT.value
        rows = db.conn.execute(
            "SELECT * FROM pose_segment WHERE desktop=77 AND view='scan' ORDER BY seg"
        ).fetchall()
        assert [(r["seg"], r["start_step"], r["end_step"]) for r in rows] == [
            (1, 1, 2),
            (2, 3, env["n_steps_77"]),
        ]
        assert rows[1]["ref_step"] == env["n_steps_77"]
        # a frame after the reorient resolves to the second segment
        assert db.pose_segment_for(FrameKey(77, 4, "scan"))["seg"] == 2
    finally:
        db.close()


def test_reorient_past_the_index_is_reported_not_split(env):
    """D63 in the real data: the sheet flips the chassis after the last photo."""
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    db = open_db(env)
    try:
        rows = db.conn.execute(
            "SELECT * FROM pose_segment WHERE desktop=78 AND view='scan' ORDER BY seg"
        ).fetchall()
        assert [(r["seg"], r["start_step"], r["end_step"]) for r in rows] == [
            (1, 1, env["n_steps_78"] - 1)
        ]
    finally:
        db.close()
    report = (env["cache"] / "import_logs_issues.md").read_text(encoding="utf-8")
    assert "past the index's last step" in report


def test_split_pose_segments_is_idempotent(env):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    assert run(env, "load-index") == 0  # re-running must not multiply segments
    db = open_db(env)
    try:
        n = db.conn.execute(
            "SELECT COUNT(*) AS n FROM pose_segment WHERE desktop=77 AND view='scan'"
        ).fetchone()["n"]
        assert n == 2
    finally:
        db.close()


def test_import_logs_without_index_still_works(env, tmp_path):
    """The index is optional: durations are simply left unset."""
    (env["cache"] / "index.json").unlink()
    assert run(env, "import-logs", "--desktops", "13") == 0
    db = open_db(env)
    try:
        steps = db.steps(13)
        assert len(steps) == env["n_steps_13"]
        assert all(s.duration_s is None for s in steps)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# import-ls
# --------------------------------------------------------------------------- #
def test_import_ls_refuses_before_steps_exist(env, capsys):
    assert run(env, "load-index") == 0
    code = run(env, "import-ls", "--export", str(LS_EXPORT))
    assert code == 2
    out = capsys.readouterr().out.lower()
    assert "step" in out and ("19" in out or "24" in out)


def test_import_ls_allow_missing_steps_override(env):
    assert run(env, "import-ls", "--export", str(LS_EXPORT), "--allow-missing-steps") == 0
    db = open_db(env)
    try:
        assert db.keyframes(19, "scan")
    finally:
        db.close()


def test_import_ls_runs_once_the_desktops_have_steps(env):
    db = open_db(env)
    try:
        for desktop in LS_DESKTOPS:
            db.replace_steps(
                desktop,
                [StepRec(desktop, k, StepType.NORMAL.value, f"row {k}") for k in range(1, 60)],
                [],
            )
    finally:
        db.close()
    assert run(env, "import-ls", "--export", str(LS_EXPORT)) == 0
    summary = json.loads((env["cache"] / "ls_import_summary.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 2
    db = open_db(env)
    try:
        assert db.keyframes(19, "scan")
        assert db.keyframes(24, "rs")
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# backup / status
# --------------------------------------------------------------------------- #
def test_backup_writes_a_file(env):
    assert run(env, "load-index") == 0
    assert run(env, "backup") == 0
    made = list(Path(env["cfg"]["backup_dir"]).glob("tda_*.sqlite"))
    assert len(made) == 1
    assert made[0].stat().st_size > 0


def test_status_prints_counts(env, capsys):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    capsys.readouterr()
    assert run(env, "status") == 0
    out = capsys.readouterr().out
    assert "D13" in out and "D77" in out
    assert str(env["n_steps_13"]) in out
    assert "scan" in out and "oak2" in out


def test_status_single_desktop(env, capsys):
    assert run(env, "load-index") == 0
    assert run(env, "import-logs") == 0
    capsys.readouterr()
    assert run(env, "status", "--desktop", "77") == 0
    out = capsys.readouterr().out
    assert "Desktop 77" in out
    assert "D13" not in out


def test_status_names_a_desktop_the_database_does_not_have(env, capsys):
    assert run(env, "load-index", "--desktops", "77") == 0
    capsys.readouterr()
    assert run(env, "status", "--desktop", "13") == 0
    assert "13 is not in the database" in capsys.readouterr().out


def test_incomplete_paths_yaml_is_an_error_not_a_traceback(tmp_path, capsys):
    broken = tmp_path / "paths.yaml"
    broken.write_text("cache_dir: " + str(tmp_path / "cache") + "\n", encoding="utf-8")
    assert main(["--paths", str(broken), "status"]) == 1
    assert "db_path" in capsys.readouterr().out


def test_db_override_wins_over_paths_yaml(env, tmp_path):
    other = tmp_path / "other.sqlite"
    assert main(["--paths", env["paths"], "--db", str(other), "load-index"]) == 0
    assert other.exists()
    assert not Path(env["db_path"]).exists()
