"""Tests for the SQLite layer's connection-level behaviour (tda.core.dbconn).

Schema version 2 -- the ``compiled_mask`` geometry columns and the in-place
migration of an older file -- and the grouped-write transaction. The queries
themselves are covered in ``test_db.py``.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from tda.core.db import Db
from tda.core.model import FrameKey

RLE_A = {"size": [4, 4], "counts": "0 4 12"}
RLE_B = {"size": [4, 4], "counts": "4 8 4"}


@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    yield d
    d.close()


def test_put_compiled_keeps_box_geometry_apart_from_masks(db: Db):
    key = FrameKey(13, 4, "scan")
    db.put_compiled(key, "panel.side.01", dict(RLE_A), 0.0, "visible", "in_chassis", "auto", "h1")
    db.put_compiled(key, "screw.psu.01", None, 0.0, "visible", "on_bench", "auto", "h1",
                    geom_type="box", box=(2.0, 3.0, 12.0, 13.0))

    rows = db.compiled(key)
    assert rows["panel.side.01"]["geom_type"] == "mask"
    assert rows["panel.side.01"]["box"] is None
    assert rows["screw.psu.01"]["geom_type"] == "box"
    assert rows["screw.psu.01"]["visible_rle"] is None
    assert rows["screw.psu.01"]["box"] == [2.0, 3.0, 12.0, 13.0]

    db.delete_compiled(key, "screw.psu.01")
    assert set(db.compiled(key)) == {"panel.side.01"}
    db.delete_compiled(key, "screw.psu.01")  # deleting what is gone is a no-op


def test_schema_v1_database_is_migrated_in_place(tmp_db_path: str):
    old = sqlite3.connect(tmp_db_path)
    with old:
        old.execute("CREATE TABLE desktop (id INTEGER PRIMARY KEY)")
        old.execute(
            "CREATE TABLE compiled_mask (desktop INTEGER NOT NULL REFERENCES desktop(id), "
            "step INTEGER NOT NULL, view TEXT NOT NULL, instance TEXT NOT NULL, "
            "visible_rle_json TEXT, occlusion_ratio REAL, visibility TEXT, placement TEXT, "
            "status TEXT NOT NULL DEFAULT 'auto', input_hash TEXT, verified_by TEXT, "
            "verified_at TEXT, PRIMARY KEY (desktop, step, view, instance))"
        )
        old.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        old.execute("INSERT INTO meta(key, value) VALUES('schema_version', '1')")
        old.execute("INSERT INTO desktop(id) VALUES(13)")
        old.execute(
            "INSERT INTO compiled_mask(desktop, step, view, instance, visible_rle_json, status) "
            "VALUES(13, 4, 'scan', 'panel.side.01', ?, 'verified')",
            (json.dumps(RLE_A),),
        )
    old.close()

    db = Db(tmp_db_path)
    row = db.compiled(FrameKey(13, 4, "scan"))["panel.side.01"]
    assert row["visible_rle"] == RLE_A  # the old data survives
    assert row["status"] == "verified"
    assert row["geom_type"] == "mask" and row["box"] is None  # the new columns default
    assert db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "4"
    db.close()


# --------------------------------------------------------------------------- #
# schema version 4: the per-view pose breaks (task B1)
# --------------------------------------------------------------------------- #
def _stamp(path: str, version: str) -> None:
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (version,))
    conn.close()


def test_a_v3_database_gains_the_empty_pose_break_table(tmp_db_path: str):
    """An older file is upgraded by re-opening it; nothing in it is touched."""
    db = Db(tmp_db_path)
    db.set_pose_segment(13, "scan", 1, 1, 20, 20, None, None)
    db.conn.execute("DROP TABLE pose_break")
    db.conn.commit()
    db.close()
    _stamp(tmp_db_path, "3")

    db = Db(tmp_db_path)
    try:
        assert db.conn.execute(
            "SELECT count(*) FROM pose_break").fetchone()[0] == 0
        assert db.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "4"
        # the v3 data is still exactly where it was
        assert [(r["seg"], r["start_step"], r["end_step"])
                for r in db.pose_segments(13, "scan")] == [(1, 1, 20)]
    finally:
        db.close()


def test_a_v4_database_is_refused_by_a_build_that_knows_only_v3(tmp_db_path: str):
    """The other direction: v3 code must not silently re-stamp a v4 file."""
    from pathlib import Path

    db = Db(tmp_db_path)
    db.upsert_desktop(13, {})          # a break needs a machine that exists
    db.add_pose_break(13, "scan", 32, status="accepted", kind="chassis",
                      magnitude_px=964.75, source="audit:events.csv")
    db.close()

    older = Db.__new__(Db)          # a v3 build, spelled as the version it knows
    older.path = tmp_db_path
    older.conn = sqlite3.connect(tmp_db_path)
    older.conn.row_factory = sqlite3.Row
    older._tx_depth = 0
    try:
        with pytest.raises(RuntimeError) as err:
            older.init_schema(Path(tda_core_dir()) / "schema.sql", 3)
        assert "schema version 4" in str(err.value)
        stamped = older.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        assert stamped == "4"       # left exactly as the newer build wrote it
    finally:
        older.conn.close()


def tda_core_dir() -> str:
    import tda.core.db as _db

    return str(__import__("pathlib").Path(_db.__file__).parent)


def test_a_newer_schema_is_refused_instead_of_downgraded(tmp_db_path: str):
    db = Db(tmp_db_path)
    with db.conn:
        db.conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    db.close()

    with pytest.raises(RuntimeError) as err:
        Db(tmp_db_path)

    assert "99" in str(err.value)
    again = sqlite3.connect(tmp_db_path)
    stamped = again.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    again.close()
    assert stamped == "99"  # the newer file was left exactly as it was


def test_transaction_commits_once_and_rolls_back_on_error(tmp_db_path: str):
    db = Db(tmp_db_path)
    key = FrameKey(13, 4, "scan")

    with db.transaction():
        db.put_compiled(key, "a", dict(RLE_A), 0.0, "visible", "in_chassis", "auto", "h1")
        with db.transaction():  # re-entrant: the inner block must not commit
            db.put_compiled(key, "b", dict(RLE_B), 0.0, "visible", "in_chassis", "auto", "h1")
        onlooker = sqlite3.connect(tmp_db_path)
        assert onlooker.execute("SELECT count(*) FROM compiled_mask").fetchone()[0] == 0
        onlooker.close()
    onlooker = sqlite3.connect(tmp_db_path)
    assert onlooker.execute("SELECT count(*) FROM compiled_mask").fetchone()[0] == 2
    onlooker.close()

    with pytest.raises(RuntimeError):
        with db.transaction():
            db.put_compiled(key, "c", dict(RLE_A), 0.0, "visible", "in_chassis", "auto", "h2")
            db.delete_compiled(key, "a")
            raise RuntimeError("boom")
    assert set(db.compiled(key)) == {"a", "b"}  # the whole block was rolled back

    db.put_compiled(key, "d", None, 0.0, "visible", "in_chassis", "auto", "h1")
    assert set(db.compiled(key)) == {"a", "b", "d"}  # writes commit again afterwards
    db.close()
