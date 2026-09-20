"""``python -m tda.cli pose-breaks`` (task B1 step 5).

The audit CSV is the real one's header with a handful of its rows, so the
import is tested against the file it will actually be pointed at:
``experiments_out/plan_b_probe/camera_moves/events.csv``.  Everything runs
against a throw-away ``paths.yaml`` in ``tmp_path``; nothing reads ``F:`` or the
annotator's database.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tda.cli import main
from tda.cli_common import EXIT_ERROR, EXIT_OK
from tda.cli_pose import accept_break, import_events, read_events, reject_break
from tda.core.db import Db
from tda.core.model import (
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StepRec,
    ZOrderRec,
)

#: The header of the real ``events.csv``, verbatim.
HEADER = ("view,desktop,step_from,step_to,kind,magnitude_px,rot_deg,scale,verdict,"
          "decided_by,n_inlier,n_match,iou_before,iou_after,iou_gain,chassis_px,"
          "chassis_area_ratio,reason,note")
ROWS = (
    "oak1,1,39,40,undetermined,,,,undetermined,auto,23,31,,,,,1.233,clustered,",
    "oak1,2,5,6,camera,8.6,0.219,0.99861,confirmed,tape_test,194,279,0.697,0.959,"
    "0.262,9.69,0.993,,",
    "oak1,36,18,19,camera,190.45,0.4,1.0,confirmed,tape_test,90,200,0.2,0.9,0.7,"
    "190.45,1.0,,tape rectangle grossly displaced in the zoom crop",
    "oak2,1,33,34,camera,14.09,0.1,1.0,confirmed,tape_test,100,200,0.3,0.9,0.6,"
    "14.09,1.0,,",
    "scan,63,31,32,chassis,964.75,90.0,1.0,chassis_manual,eye,10,200,0.1,0.8,0.7,"
    "964.75,1.0,,chassis rotated ~90 deg",
    "scan,36,40,41,camera,1.2,0.0,1.0,confirmed,tape_test,200,300,0.8,0.99,0.19,"
    "1.2,1.0,,",
    "oak1,3,4,5,chassis,50.0,0.0,1.0,rejected,eye,10,20,,,,50.0,1.0,,",
)
STEPS = 40
DESKTOP = 13
VIEW = "oak1"


@pytest.fixture
def events(tmp_path: Path) -> str:
    path = tmp_path / "events.csv"
    path.write_text("\n".join((HEADER, *ROWS)) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def env(tmp_path: Path) -> dict:
    """A paths.yaml plus a database with one desktop, one cut-able view."""
    db_path = tmp_path / "annotations" / "tda.sqlite"
    cfg = {"cache_dir": str(tmp_path / "cache"), "db_path": str(db_path),
           "backup_dir": str(tmp_path / "backups"), "backup_keep": 3}
    paths = tmp_path / "paths.yaml"
    paths.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    db = Db(str(db_path))
    db.upsert_desktop(DESKTOP, {"brand": "Dell"})
    db.upsert_instance(InstanceRec(key="psu.01", desktop=DESKTOP, cls="psu"))
    db.replace_steps(DESKTOP,
                     [StepRec(DESKTOP, k, "normal", f"row {k}")
                      for k in range(1, STEPS + 1)], [])
    for step in range(1, STEPS + 1):
        db.upsert_frame(FrameKey(DESKTOP, step, VIEW), f"s{step}.jpg",
                        {"hw": [64, 64]}, None)
    db.add_keyframe(ShapeKeyframe(
        id=None, instance="psu.01", desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=STEPS, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", {"size": [64, 64], "counts": "0 8 4088"})]))
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [("psu.01", "main")]))
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, STEPS, STEPS, None, None)
    db.close()
    return {"paths": str(paths), "db_path": str(db_path), "root": tmp_path}


def run(env: dict, *argv: str) -> int:
    return main(["--paths", env["paths"], *argv])


def open_db(env: dict) -> Db:
    return Db(env["db_path"])


# --------------------------------------------------------------------------- #
# reading the audit
# --------------------------------------------------------------------------- #
def test_only_confirmed_camera_and_chassis_rows_become_proposals(events: str):
    proposals, _skipped = read_events(events)

    assert ("oak1", 1, 40) not in [(p.view, p.desktop, p.step) for p in proposals]
    assert ("oak1", 3, 5) not in [(p.view, p.desktop, p.step) for p in proposals]


def test_a_chassis_event_is_proposed_for_all_four_views(events: str):
    proposals, _skipped = read_events(events)

    rotated = [p for p in proposals if p.desktop == 63]
    assert sorted(p.view for p in rotated) == ["oak1", "oak2", "rs", "scan"]
    assert {p.step for p in rotated} == {32}      # step_to, not step_from
    assert all(p.kind == "chassis" for p in rotated)


def test_a_camera_event_is_proposed_for_its_own_view_only(events: str):
    proposals, _skipped = read_events(events, min_px=0.0)

    moved = [p for p in proposals if p.desktop == 36 and p.kind == "camera"]
    assert sorted((p.view, p.step) for p in moved) == [("oak1", 19), ("scan", 41)]


def test_min_px_drops_the_small_movements(events: str):
    proposals, skipped = read_events(events, min_px=2.0)

    assert ("scan", 36, 41) not in [(p.view, p.desktop, p.step) for p in proposals]
    assert any("below --min-px 2" in line for line in skipped)
    assert len(read_events(events, min_px=0.0)[0]) > len(proposals)


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #
def test_a_dry_run_import_writes_nothing(env: dict, events: str):
    lines: list[str] = []
    db = open_db(env)
    try:
        run_out = import_events(db, events, dry_run=True, log=lines.append)
        assert db.pose_breaks(36) == [] and db.pose_breaks(63) == []
    finally:
        db.close()
    assert run_out.per_view()["oak1"] == 3      # D2, D36, and D63's rotation
    assert any("would propose" in line for line in lines)


def test_an_import_never_overwrites_a_decision(env: dict, events: str):
    db = open_db(env)
    try:
        import_events(db, events)
        db.set_pose_break_status(63, "scan", 32, "rejected")
        db.set_pose_break_status(36, "oak1", 19, "accepted")

        second = import_events(db, events)

        assert second.written == []
        assert len(second.known) == len(second.proposals)
        assert db.pose_break(63, "scan", 32)["status"] == "rejected"
        assert db.pose_break(36, "oak1", 19)["status"] == "accepted"
        assert db.pose_break(2, "oak1", 6)["source"] == "events.csv".join(("audit:", ""))
    finally:
        db.close()


def test_the_import_command_backs_up_before_it_writes(env: dict, events: str):
    assert run(env, "pose-breaks", "import", events) == EXIT_OK

    backups = list((env["root"] / "backups").glob("tda_*.sqlite"))
    assert len(backups) == 1
    db = open_db(env)
    try:
        assert db.pose_break(2, "oak1", 6)["status"] == "proposed"
    finally:
        db.close()


def test_a_dry_run_takes_no_backup(env: dict, events: str):
    assert run(env, "pose-breaks", "import", events, "--dry-run") == EXIT_OK
    assert not (env["root"] / "backups").exists()


def test_import_without_a_file_is_one_line_and_an_error(env: dict):
    assert run(env, "pose-breaks", "import") == EXIT_ERROR


def test_a_missing_events_file_is_reported_not_a_traceback(env: dict):
    assert run(env, "pose-breaks", "import", str(env["root"] / "nope.csv")) == EXIT_ERROR


# --------------------------------------------------------------------------- #
# accept / reject
# --------------------------------------------------------------------------- #
def test_accept_cuts_the_view_and_moves_the_rows(env: dict):
    db = open_db(env)
    try:
        db.add_pose_break(DESKTOP, VIEW, 19, status="proposed", kind="camera",
                          magnitude_px=190.45, source="audit:events.csv")

        out = accept_break(db, DESKTOP, VIEW, 19)

        assert out["changed"] is True
        assert [(r["seg"], r["start_step"], r["end_step"])
                for r in db.pose_segments(DESKTOP, VIEW)] == [(1, 1, 18), (2, 19, STEPS)]
        assert db.keyframes(DESKTOP, VIEW)[0].pose_segment == 2
        assert db.zorder(DESKTOP, VIEW, 1).order == [("psu.01", "main")]
        # 190 px is far past the carry threshold, so nothing was duplicated
        assert out["carried"] == []
    finally:
        db.close()


def test_a_small_movement_carries_the_shapes_across_by_default(env: dict):
    db = open_db(env)
    try:
        db.add_pose_break(DESKTOP, VIEW, 19, status="proposed", kind="camera",
                          magnitude_px=8.6, source="audit:events.csv")

        out = accept_break(db, DESKTOP, VIEW, 19)

        assert len(out["carried"]) == 1
        carried = [k for k in db.keyframes(DESKTOP, VIEW) if k.source == "carried"]
        assert (carried[0].anchor_step, carried[0].pose_segment) == (18, 1)
    finally:
        db.close()


def test_no_carry_overrides_the_default(env: dict):
    db = open_db(env)
    try:
        db.add_pose_break(DESKTOP, VIEW, 19, status="proposed", kind="camera",
                          magnitude_px=8.6, source="audit:events.csv")

        assert accept_break(db, DESKTOP, VIEW, 19, carry=False)["carried"] == []
    finally:
        db.close()


def test_reject_merges_the_segments_back(env: dict):
    db = open_db(env)
    try:
        db.add_pose_break(DESKTOP, VIEW, 19, status="proposed", kind="camera",
                          magnitude_px=8.6, source="audit:events.csv")
        accept_break(db, DESKTOP, VIEW, 19)

        out = reject_break(db, DESKTOP, VIEW, 19)

        assert [(r["seg"], r["start_step"], r["end_step"])
                for r in db.pose_segments(DESKTOP, VIEW)] == [(1, 1, STEPS)]
        assert [k for k in db.keyframes(DESKTOP, VIEW) if k.source == "carried"] == []
        assert db.pose_break(DESKTOP, VIEW, 19)["status"] == "rejected"
        assert out["changed"] is True
    finally:
        db.close()


def test_accepting_a_break_that_is_not_there_is_refused(env: dict):
    db = open_db(env)
    try:
        with pytest.raises(ValueError):
            accept_break(db, DESKTOP, VIEW, 19)
    finally:
        db.close()
    assert run(env, "pose-breaks", "accept", "--desktop", str(DESKTOP),
               "--view", VIEW, "--step", "19") == EXIT_ERROR


def test_accept_needs_a_desktop_a_view_and_a_step(env: dict):
    assert run(env, "pose-breaks", "accept", "--desktop", str(DESKTOP)) == EXIT_ERROR


def test_a_break_for_a_view_with_no_frames_is_stored_and_cuts_nothing(env: dict):
    db = open_db(env)
    try:
        db.add_pose_break(DESKTOP, "rs", 19, status="proposed", kind="chassis",
                          magnitude_px=964.75, source="audit:events.csv")

        out = accept_break(db, DESKTOP, "rs", 19)

        assert out["changed"] is False and out["ranges"] == []
        assert db.pose_break(DESKTOP, "rs", 19)["status"] == "accepted"
        assert db.pose_segments(DESKTOP, "rs") == []
    finally:
        db.close()


def test_a_break_at_step_one_or_past_the_last_step_cuts_nothing(env: dict):
    db = open_db(env)
    try:
        for step in (1, STEPS + 5):
            db.add_pose_break(DESKTOP, VIEW, step, status="proposed", kind="manual",
                              source="manual:anna")
            out = accept_break(db, DESKTOP, VIEW, step)
            assert out["changed"] is False
        assert [(r["seg"], r["start_step"], r["end_step"])
                for r in db.pose_segments(DESKTOP, VIEW)] == [(1, 1, STEPS)]
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def test_list_prints_every_stored_break(env: dict, events: str, capsys):
    assert run(env, "pose-breaks", "import", events) == EXIT_OK
    capsys.readouterr()

    assert run(env, "pose-breaks", "list") == EXIT_OK

    out = capsys.readouterr().out
    assert "D63 scan  step  32" in out
    assert "proposed" in out


def test_list_narrows_by_desktop_view_and_status(env: dict, events: str, capsys):
    assert run(env, "pose-breaks", "import", events) == EXIT_OK
    capsys.readouterr()

    assert run(env, "pose-breaks", "list", "--desktop", "63", "--view", "scan",
               "--status", "proposed") == EXIT_OK

    out = capsys.readouterr().out
    assert "1 breaks for D63, view scan, status proposed" in out
