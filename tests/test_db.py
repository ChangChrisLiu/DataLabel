"""Tests for the SQLite persistence layer (tda.core.db)."""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tda.core.db import Db
from tda.core.model import (
    ActionRec,
    FrameKey,
    FrameOverride,
    InstanceRec,
    OccluderMask,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    Similarity,
    StateEvent,
    StepRec,
    ZOrderRec,
)

RLE_A = {"size": [4, 4], "counts": "0 4 12"}
RLE_B = {"size": [4, 4], "counts": "4 8 4"}

EXPECTED_TABLES = {
    "meta",
    "desktop",
    "frame",
    "step",
    "action",
    "instance",
    "state_event",
    "pose_segment",
    "frame_transform",
    "shape_keyframe",
    "shape_part",
    "zorder",
    "pair_override",
    "occluder_mask",
    "frame_override",
    "instance_frame_flags",
    "compiled_mask",
    "conflict",
    "relation",
    "op_log",
}


@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    yield d
    d.close()


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows if not r[0].startswith("sqlite_")}


# --------------------------------------------------------------------------- schema


def test_schema_creation_is_idempotent_and_sets_pragmas(tmp_db_path: str):
    db = Db(tmp_db_path)
    assert _tables(db.conn) >= EXPECTED_TABLES
    assert db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "3"
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    db.close()

    again = Db(tmp_db_path)  # opening an existing db must not fail nor duplicate meta
    assert _tables(again.conn) >= EXPECTED_TABLES
    assert again.conn.execute("SELECT count(*) FROM meta WHERE key='schema_version'").fetchone()[0] == 1
    again.close()


def test_repository_is_qt_free():
    import tda.core.db as db_mod

    src = Path(db_mod.__file__).read_text(encoding="utf-8")
    assert "PySide6" not in src and "QtCore" not in src


# --------------------------------------------------------------------------- frames


def test_desktop_roundtrip_columns_and_meta_overflow(db: Db):
    db.upsert_desktop(13, {
        "brand": "Dell", "model_family": "OptiPlex 7010", "chassis_type": "sff",
        "notes": "试标机", "operator": "zzh", "psu_watt": 240,
    })
    got = db.get_desktop(13)
    assert got["id"] == 13
    assert got["brand"] == "Dell" and got["model_family"] == "OptiPlex 7010"
    assert got["chassis_type"] == "sff" and got["notes"] == "试标机"
    assert got["operator"] == "zzh" and got["psu_watt"] == 240  # overflow -> meta_json
    assert "split" not in got  # documented: NULL columns are omitted
    stored = db.conn.execute("SELECT meta_json, brand FROM desktop WHERE id=13").fetchone()
    assert json.loads(stored["meta_json"]) == {"operator": "zzh", "psu_watt": 240}
    assert "试标机" not in (stored["meta_json"] or "")  # known key stays a column

    db.upsert_desktop(13, {"brand": "Dell", "split": "train", "operator": "lab2"})
    again = db.get_desktop(13)
    assert again["split"] == "train" and again["operator"] == "lab2"
    assert "model_family" not in again  # full replacement clears dropped columns
    assert "psu_watt" not in again
    assert db.conn.execute("SELECT count(*) FROM desktop").fetchone()[0] == 1

    db.upsert_desktop(14, {})
    assert db.get_desktop(14) == {"id": 14}
    assert db.get_desktop(99) is None


def test_upsert_frame_get_frame_roundtrip(db: Db):
    db.upsert_desktop(13, {"brand": "Dell", "model_family": "OptiPlex 7010", "notes": "试标"})
    key = FrameKey(13, 2, "scan")
    db.upsert_frame(
        key,
        "F:/x/scan_002.png",
        {"width": 4000, "height": 3000},
        "2026-09-01T10:00:00",
        flags={"hand_or_tool_in_frame": True, "image_quality": "ok", "pose_segment": 1},
    )
    got = db.get_frame(key)
    assert got["path"] == "F:/x/scan_002.png"
    assert got["aux"] == {"width": 4000, "height": 3000}
    assert got["ts"] == "2026-09-01T10:00:00"
    assert got["hand_or_tool_in_frame"] is True
    assert got["image_quality"] == "ok"
    assert got["pose_segment"] == 1
    assert got["review_status"] == "unlabeled"
    assert db.get_frame(FrameKey(13, 99, "scan")) is None


def test_set_frame_flags_preserves_other_columns(db: Db):
    key = FrameKey(13, 2, "oak1")
    db.upsert_frame(key, "p.png", {}, None, flags={"image_quality": "blurry"})
    db.set_frame_flags(key, in_progress=True, bench_annotated=False, review_status="needs_review")
    got = db.get_frame(key)
    assert got["image_quality"] == "blurry"  # untouched
    assert got["in_progress"] is True
    assert got["bench_annotated"] is False
    assert got["review_status"] == "needs_review"
    assert got["path"] == "p.png"
    with pytest.raises(ValueError):
        db.set_frame_flags(key, not_a_flag=1)


def test_frames_for_is_ordered_and_view_scoped(db: Db):
    for step in (3, 1, 2):
        db.upsert_frame(FrameKey(13, step, "scan"), f"s{step}.png", {}, None)
    db.upsert_frame(FrameKey(13, 1, "rs"), "r1.png", {}, None)
    assert [f["step"] for f in db.frames_for(13, "scan")] == [1, 2, 3]
    assert [f["path"] for f in db.frames_for(13, "rs")] == ["r1.png"]


def test_upsert_frame_updates_existing_row(db: Db):
    key = FrameKey(13, 1, "scan")
    db.upsert_frame(key, "old.png", {"a": 1}, None, flags={"missing": True})
    db.upsert_frame(key, "new.png", {"a": 2}, "2026-09-02T00:00:00")
    got = db.get_frame(key)
    assert (got["path"], got["aux"], got["missing"]) == ("new.png", {"a": 2}, True)
    assert db.conn.execute("SELECT count(*) FROM frame").fetchone()[0] == 1


# --------------------------------------------------------------------------- steps / actions


def _steps_fixture() -> tuple[list[StepRec], list[ActionRec]]:
    steps = [
        StepRec(13, 2, "normal", "拆侧板螺丝", dupli=False, notes="", duration_s=42.5),
        StepRec(13, 1, "initial", "初始状态"),
        StepRec(13, 3, "failed", "拆散热器", dupli=True, notes="卡住"),
    ]
    actions = [
        ActionRec(13, 2, 0, "screw.side_panel.01", "unscrew", tool="ph2", direction="-z"),
        ActionRec(13, 2, 1, "panel.side.01", "remove"),
        ActionRec(
            13, 3, 0, "cooler.cpu.01", "remove", result="failed",
            failure_reason="fastener_stuck", difficulty=4,
        ),
    ]
    return steps, actions


def test_replace_steps_then_steps_ordered(db: Db):
    steps, actions = _steps_fixture()
    db.replace_steps(13, steps, actions)
    got = db.steps(13)
    assert [s.step for s in got] == [1, 2, 3]
    assert got[1].raw_name == "拆侧板螺丝" and got[1].duration_s == 42.5
    assert got[2].dupli is True and got[2].notes == "卡住"


def test_replace_steps_is_a_full_replacement(db: Db):
    steps, actions = _steps_fixture()
    db.replace_steps(13, steps, actions)
    db.replace_steps(13, [StepRec(13, 1, "initial", "only")], [])
    assert [s.step for s in db.steps(13)] == [1]
    assert db.actions(13) == []


def test_actions_roundtrip_and_step_filter(db: Db):
    steps, actions = _steps_fixture()
    db.replace_steps(13, steps, actions)
    assert [(a.step, a.idx) for a in db.actions(13)] == [(2, 0), (2, 1), (3, 0)]
    failed = db.actions(13, step=3)[0]
    assert failed.result == "failed"
    assert failed.failure_reason == "fastener_stuck"
    assert failed.difficulty == 4
    assert db.actions(13, step=2)[0].tool == "ph2"


# --------------------------------------------------------------------------- instances / events


def test_instance_roundtrip_preserves_every_field(db: Db):
    inst = InstanceRec(
        key="screw.cpu_cooler.03",
        desktop=13,
        cls="screw",
        attrs={"role": "cooler", "head": "ph2", "head_source": "image", "captive": True},
        parent="cooler.cpu.01",
        attached=True,
        mounted_on="mainboard.01",
        fastens="cooler.cpu.01",
        socket_host="mainboard.01",
        cable="cable:cpu_fan",
        slot_id="slot.cooler.03",
        group_id="grp.cooler_screws",
        group_order="opposite_pairs",
        removal_direction="-z",
        raw_names=["散热器螺丝3", "cooler screw 3"],
    )
    db.upsert_instance(inst)
    got = db.instances(13)["screw.cpu_cooler.03"]
    assert got == inst

    inst.attached = False
    inst.raw_names = ["renamed"]
    db.upsert_instance(inst)
    all_inst = db.instances(13)
    assert len(all_inst) == 1
    assert all_inst["screw.cpu_cooler.03"].attached is False
    assert all_inst["screw.cpu_cooler.03"].raw_names == ["renamed"]


def test_replace_events_auto_only_keeps_manual(db: Db):
    auto = StateEvent(13, 2, "panel.side.01", "state", "installed", "removed", evidence_view="scan")
    manual = StateEvent(13, 2, "cable:psu_harness", "state", "routed", "released", auto=False)
    db.replace_events(13, [auto, manual], auto_only=False)
    assert len(db.events(13)) == 2

    fresh = StateEvent(13, 3, "panel.side.01", "placement", "in_chassis", "on_bench", evidence_view="oak1")
    db.replace_events(13, [fresh], auto_only=True)
    got = db.events(13)
    assert [e.target for e in got] == ["cable:psu_harness", "panel.side.01"]
    assert got[1].evidence_view == "oak1" and got[1].auto is True
    assert got[0].auto is False

    db.replace_events(13, [], auto_only=False)
    assert db.events(13) == []


# --------------------------------------------------------------------------- geometry


def _kf(**kw) -> ShapeKeyframe:
    base = dict(
        id=None, instance="panel.side.01", desktop=13, view="scan", pose_segment=0,
        anchor_step=5, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", rle=dict(RLE_A), box=(1.0, 2.0, 3.0, 4.0))],
        amodal_complete=True, source="sam", draft_id=7, version=1,
        edit_count=2, edit_time_ms=1500,
    )
    base.update(kw)
    return ShapeKeyframe(**base)


def test_add_keyframe_roundtrip_all_fields(db: Db):
    kf = _kf()
    kid = db.add_keyframe(kf)
    assert isinstance(kid, int) and kid > 0
    got = db.keyframes(13, "scan")[0]
    assert got.id == kid
    assert got.version == 1
    assert (got.placement, got.geom_type, got.draft_id) == ("in_chassis", "mask", 7)
    assert (got.edit_count, got.edit_time_ms, got.source) == (2, 1500, "sam")
    assert got.amodal_complete is True
    assert got.parts[0].name == "main"
    assert got.parts[0].rle == RLE_A
    assert got.parts[0].box == (1.0, 2.0, 3.0, 4.0)


def test_update_keyframe_bumps_version_and_replaces_parts(db: Db):
    kf = _kf()
    db.add_keyframe(kf)
    kf.parts = [ShapePart("floor", rle=dict(RLE_B)), ShapePart("wall", box=(0.0, 0.0, 1.0, 1.0))]
    kf.edit_count = 3
    kf.placement = "on_bench"
    db.update_keyframe(kf)
    got = db.keyframes(13, "scan")[0]
    assert got.version == 2
    assert kf.version == 2
    assert [p.name for p in got.parts] == ["floor", "wall"]
    assert got.parts[0].rle == RLE_B and got.parts[0].box is None
    assert got.parts[1].rle is None
    assert got.placement == "on_bench" and got.edit_count == 3

    db.update_keyframe(kf)
    assert db.keyframes(13, "scan")[0].version == 3


def test_keyframes_filtered_by_instance_and_view(db: Db):
    db.add_keyframe(_kf())
    db.add_keyframe(_kf(instance="mainboard.01", anchor_step=9))
    db.add_keyframe(_kf(view="oak1"))
    assert len(db.keyframes(13, "scan")) == 2
    assert [k.instance for k in db.keyframes(13, "scan", "mainboard.01")] == ["mainboard.01"]
    assert len(db.keyframes(13, "oak1")) == 1


def test_zorder_default_is_empty_then_roundtrips(db: Db):
    empty = db.zorder(13, "scan", 0)
    assert isinstance(empty, ZOrderRec)
    assert empty.order == []
    assert (empty.desktop, empty.view, empty.pose_segment) == (13, "scan", 0)

    rec = ZOrderRec(13, "scan", 0, [("mainboard.01", "main"), ("cooler.cpu.01", "main")], version=4)
    db.set_zorder(rec)
    got = db.zorder(13, "scan", 0)
    assert got.order == [("mainboard.01", "main"), ("cooler.cpu.01", "main")]
    assert got.version == 4


def test_pair_overrides_roundtrip(db: Db):
    po = PairOverride(13, "scan", 0, "cable.psu.01", "mainboard.01")
    db.set_pair_override(po)
    db.set_pair_override(po)  # idempotent
    got = db.pair_overrides(13, "scan", 0)
    assert got == [po]
    assert db.pair_overrides(13, "scan", 1) == []


def test_occluders_and_frame_overrides(db: Db):
    key = FrameKey(13, 4, "scan")
    db.set_occluder(OccluderMask(key, "hand", dict(RLE_A)))
    db.set_occluder(OccluderMask(key, "tool", dict(RLE_B)))
    db.set_occluder(OccluderMask(key, "hand", dict(RLE_B)))  # replaces the hand layer
    got = {o.occluder_type: o.rle for o in db.occluders(key)}
    assert got == {"hand": RLE_B, "tool": RLE_B}
    assert db.occluders(FrameKey(13, 5, "scan")) == []

    db.set_frame_override(FrameOverride(key, "panel.side.01", visible_rle=dict(RLE_A), visibility=None))
    db.set_frame_override(FrameOverride(key, "mainboard.01", visibility="occluded_full"))
    fos = db.frame_overrides(key)
    assert set(fos) == {"panel.side.01", "mainboard.01"}
    assert fos["panel.side.01"].visible_rle == RLE_A
    assert fos["mainboard.01"].visible_rle is None
    assert fos["mainboard.01"].visibility == "occluded_full"
    assert fos["panel.side.01"].frame == key


def test_pose_segment_and_transform(db: Db):
    corners = [[0, 0], [10, 0], [10, 8], [0, 8]]
    homography = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    db.set_pose_segment(13, "scan", 0, 1, 5, 3, corners, homography)
    db.set_pose_segment(13, "scan", 1, 6, 9, 7, None, None)
    seg = db.pose_segment_for(FrameKey(13, 4, "scan"))
    assert seg["seg"] == 0 and seg["ref_step"] == 3
    assert seg["corners"] == corners and seg["homography"] == homography
    assert db.pose_segment_for(FrameKey(13, 7, "scan"))["seg"] == 1
    assert db.pose_segment_for(FrameKey(13, 99, "scan"))["seg"] == 0  # default segment

    key = FrameKey(13, 4, "scan")
    assert db.transform(key).is_identity()
    db.set_transform(key, Similarity(scale=1.5, theta=0.25, tx=-3.0, ty=4.0))
    t = db.transform(key)
    assert (t.scale, t.theta, t.tx, t.ty) == (1.5, 0.25, -3.0, 4.0)


# --------------------------------------------------------------------------- truth


def test_put_compiled_upserts_and_compiled_reads_back(db: Db):
    key = FrameKey(13, 4, "scan")
    db.put_compiled(key, "panel.side.01", dict(RLE_A), 0.12, "visible", "in_chassis", "auto", "h1")
    db.put_compiled(key, "mainboard.01", None, 1.0, "occluded_full", "in_chassis", "auto", "h2")
    rows = db.compiled(key)
    assert set(rows) == {"panel.side.01", "mainboard.01"}
    assert rows["panel.side.01"]["visible_rle"] == RLE_A
    assert rows["panel.side.01"]["occlusion_ratio"] == 0.12
    assert rows["panel.side.01"]["status"] == "auto"
    assert rows["panel.side.01"]["input_hash"] == "h1"
    assert rows["mainboard.01"]["visible_rle"] is None

    db.put_compiled(key, "panel.side.01", dict(RLE_B), 0.3, "occluded_partial", "in_chassis",
                    "verified", "h3", verified_by="zzh")
    row = db.compiled(key)["panel.side.01"]
    assert row["status"] == "verified" and row["verified_by"] == "zzh"
    assert row["verified_at"] is not None
    assert row["visible_rle"] == RLE_B
    assert db.conn.execute("SELECT count(*) FROM compiled_mask").fetchone()[0] == 2
    assert db.compiled(FrameKey(13, 5, "scan")) == {}


def test_conflict_queue_add_list_resolve(db: Db):
    key = FrameKey(13, 4, "scan")
    cid = db.add_conflict(key, "panel.side.01", dict(RLE_A), dict(RLE_B), 350)
    other = db.add_conflict(FrameKey(13, 6, "oak1"), "mainboard.01", None, dict(RLE_A), 21)
    assert isinstance(cid, int) and cid != other

    open_rows = db.conflicts(13)
    assert {r["id"] for r in open_rows} == {cid, other}
    row = next(r for r in open_rows if r["id"] == cid)
    assert row["old_rle"] == RLE_A and row["new_rle"] == RLE_B
    assert row["sym_diff_px"] == 350 and row["status"] == "open"
    assert [r["id"] for r in db.conflicts(13, view="oak1")] == [other]

    db.resolve_conflict(cid, "keep_old")
    assert [r["id"] for r in db.conflicts(13)] == [other]
    closed = next(r for r in db.conflicts(13, open_only=False) if r["id"] == cid)
    assert closed["status"] == "resolved" and closed["resolution"] == "keep_old"
    with pytest.raises(ValueError):
        db.resolve_conflict(other, "whatever")

    assert db.get_conflict(other)["instance"] == "mainboard.01"
    assert db.get_conflict(other)["new_rle"] == RLE_A
    assert db.get_conflict(cid)["resolution"] == "keep_old"
    assert db.get_conflict(999999) is None





def test_relations_roundtrip(db: Db):
    rid = db.add_relation(
        13, "blocked_by", "cooler.cpu.01", "panel.side.01",
        necessity="required", mode="physical_path", reason="手够不到", source="manual",
        evidence_step=3,
    )
    db.add_relation(13, "fastened_by", "panel.side.01", "screw.side_panel.01", source="template")
    rows = db.relations(13)
    assert len(rows) == 2
    first = next(r for r in rows if r["id"] == rid)
    assert first["type"] == "blocked_by"
    assert (first["target"], first["blocker"]) == ("cooler.cpu.01", "panel.side.01")
    assert first["mode"] == "physical_path" and first["necessity"] == "required"
    assert first["reason"] == "手够不到" and first["source"] == "manual"
    assert first["evidence_step"] == 3 and first["status"] == "active"
    assert db.relations(99) == []


# --------------------------------------------------------------------------- ops


def test_op_log_returns_newest_first_and_honours_limit(db: Db):
    ids = [db.log_op(13, "scan", "paint", {"i": i}, {"i": -i}, "zzh") for i in range(5)]
    rows = db.ops(13, "scan")
    assert [r["id"] for r in rows] == list(reversed(ids))
    assert rows[0]["kind"] == "paint"
    assert rows[0]["payload"] == {"i": 4} and rows[0]["inverse"] == {"i": -4}
    assert rows[0]["annotator"] == "zzh" and rows[0]["ts"]
    assert len(db.ops(13, "scan", limit=2)) == 2
    assert db.ops(13, "oak1") == []


# --------------------------------------------------------------------------- backup / lock


def test_backup_creates_readable_copy(db: Db, tmp_path: Path):
    db.upsert_desktop(13, {"brand": "Dell"})
    db.upsert_frame(FrameKey(13, 1, "scan"), "s1.png", {"w": 1}, None)
    dest = tmp_path / "backups" / "nested"  # must be created by backup()
    out = db.backup(str(dest))

    assert Path(out).exists()
    assert re.fullmatch(r"tda_\d{8}_\d{6}(_\d+)?\.sqlite", Path(out).name)
    copy = sqlite3.connect(out)
    try:
        assert _tables(copy) == _tables(db.conn)
        assert copy.execute("SELECT path FROM frame").fetchone()[0] == "s1.png"
    finally:
        copy.close()

    second = db.backup(str(dest))
    assert second != out and Path(second).exists()


def test_lock_blocks_other_annotator_but_not_self(db: Db, tmp_db_path: str):
    lock = Path(tmp_db_path + ".lock")
    db.acquire_lock("zzh")
    assert lock.exists()
    assert json.loads(lock.read_text(encoding="utf-8"))["annotator"] == "zzh"

    db.acquire_lock("zzh")  # same annotator re-acquires silently
    with pytest.raises(RuntimeError):
        db.acquire_lock("someone_else")

    db.release_lock()
    assert not lock.exists()
    db.acquire_lock("someone_else")  # free again
    db.release_lock()
    db.release_lock()  # idempotent


def test_stale_lock_is_taken_over(db: Db, tmp_db_path: str):
    lock = Path(tmp_db_path + ".lock")
    old = (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()
    lock.write_text(json.dumps({"annotator": "ghost", "ts": old}), encoding="utf-8")
    db.acquire_lock("zzh")
    assert json.loads(lock.read_text(encoding="utf-8"))["annotator"] == "zzh"
    db.release_lock()


@pytest.mark.parametrize("content", ['"zzh"', "[1, 2]", "42", "null", "not json at all", ""])
def test_malformed_lock_is_taken_over(tmp_db_path: str, content: str):
    db = Db(tmp_db_path)
    lock = Path(tmp_db_path + ".lock")
    lock.write_text(content, encoding="utf-8")
    db.acquire_lock("zzh")  # must not raise, whatever the file holds
    assert json.loads(lock.read_text(encoding="utf-8"))["annotator"] == "zzh"
    db.release_lock()
    db.close()


def test_close_is_idempotent(tmp_db_path: str):
    db = Db(tmp_db_path)
    db.close()
    db.close()
