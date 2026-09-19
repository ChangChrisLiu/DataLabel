"""A new implied instance changes what every frozen frame needs geometry for.

``infer-relations`` queues a desktop's verified frames for a re-check whenever
it touched something. An implied instance is not in ``changes`` -- it is a new
row, not a rewritten one -- so a desktop whose *only* change was the implied
motherboard queued nothing at all, while ``needs_geom`` had just gained an
instance on every single frame of it. The frozen rows would have gone on saying
a board is not part of the picture.
"""
from __future__ import annotations

import pytest
from test_cli import d13_steps, env, open_db, run  # noqa: F401  (re-used fixtures)

from tda.cli import EXIT_OK
from tda.core.db import Db
from tda.core.model import FrameKey, InstanceRec

DESKTOP = 63  # its sheet never lifts the motherboard out


def _imported(env: dict) -> None:
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK


def _freeze(db: Db, desktop: int, step: int, view: str = "scan") -> None:
    db.put_compiled(
        FrameKey(desktop, step, view), "chassis.01", None, 0.0, "visible",
        "in_chassis", "verified", "hash", verified_by="chang",
        geom_type="box", box=(0, 0, 4, 4),
    )


def _only_the_board_can_change(env: dict) -> None:
    """Leave D63 in the one state where the implied instance is the sole change.

    Every motherboard-side connector loses its reference, and a single screw
    whose ``role`` names the class - but whose ``fastens`` a human has already
    answered - is what implies the board. The heuristic then fills nothing, so
    ``changes`` is empty and only the new instance is left to notice.
    """
    db = open_db(env)
    try:
        for rec in db.instances(DESKTOP).values():
            if rec.cls == "connector" and (rec.socket_host or "").startswith("motherboard"):
                rec.socket_host = None
                db.upsert_instance(rec)
        db.delete_instance(DESKTOP, "motherboard.01")
        db.upsert_instance(InstanceRec(
            "screw.motherboard.01", DESKTOP, "screw",
            attrs={"role": "motherboard"}, fastens="chassis.01",
        ))
    finally:
        db.close()


def _queued(env: dict, desktop: int = DESKTOP) -> int:
    db = open_db(env)
    try:
        return int(db.conn.execute(
            "SELECT COUNT(*) FROM recheck_queue WHERE desktop=?", (desktop,)
        ).fetchone()[0])
    finally:
        db.close()


def test_an_implied_instance_alone_queues_the_frozen_frames(env, capsys):
    _imported(env)
    _only_the_board_can_change(env)
    db = open_db(env)
    try:
        _freeze(db, DESKTOP, 5)
        _freeze(db, DESKTOP, 6, "oak1")
    finally:
        db.close()
    assert _queued(env) == 0
    capsys.readouterr()

    assert run(env, "infer-relations", "--add-implied", "--force",
               "--desktops", str(DESKTOP)) == EXIT_OK
    out = capsys.readouterr().out
    assert "1 implied, 0 fills on 0 instances" in out  # the board is the only change
    db = open_db(env)
    try:
        assert db.instances(DESKTOP).get("motherboard.01") is not None
    finally:
        db.close()
    assert _queued(env) == 2


def test_a_desktop_that_changed_nothing_queues_nothing(env):
    _imported(env)
    db = open_db(env)
    try:
        _freeze(db, DESKTOP, 5)
    finally:
        db.close()
    # a second run fills nothing and implies nothing: the import already did
    assert run(env, "infer-relations", "--add-implied", "--force",
               "--desktops", str(DESKTOP)) == EXIT_OK
    assert _queued(env) == 0
