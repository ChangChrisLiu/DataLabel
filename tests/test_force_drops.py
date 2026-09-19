"""``import-logs --force`` removes an instance the re-import no longer produces.

The writer only ever upserted, so a key that used to exist and does not any more
simply stayed. That is harmless while the identity rule is stable and fatal the
moment it changes: the real database's thirteen PSU desktops were imported with
the old rule, so re-importing them merges ``psu.01`` and ``psu.02`` into one --
and leaves ``psu.02`` behind with no actions at all, ``installed`` in the chassis
for ever. The annotator would draw a ghost PSU on every frame,
``unique_of_class("psu")`` would stay ambiguous, and ``constraints`` would go on
writing 44 fewer ``connected_to`` edges across those desktops. The merge would
have been a no-op on real data.

So a vanished key is dropped -- but only when nothing of a human's is attached to
it. One that carries a keyframe, an override, a verified row, an open conflict, a
manual event, a hand-made relation or a z-order entry is kept and reported, for
stage S1 to merge or delete by hand.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from test_cli import LOG_FIXTURES, d13_steps, env, open_db, run  # noqa: F401

from tda.cli import EXIT_OK
from tda.core.db import Db
from tda.core.graph import Edge, edges_to_db
from tda.core.model import (
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StateEvent,
    ZOrderRec,
)

GHOST = "psu.02"
DESKTOP = 13


def seed_ghost(env: dict, **fields) -> None:
    """Add an instance the sheet does not name, as the old rule would have."""
    db = open_db(env)
    try:
        db.upsert_instance(InstanceRec(key=GHOST, desktop=DESKTOP, cls="psu", **fields))
    finally:
        db.close()


def imported(env: dict) -> None:
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs", "--desktops", str(DESKTOP)) == EXIT_OK


def reimport(env: dict, *extra: str) -> int:
    return run(env, "import-logs", "--desktops", str(DESKTOP), "--force", *extra)


def instances(env: dict) -> dict:
    db = open_db(env)
    try:
        return db.instances(DESKTOP)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the drop
# --------------------------------------------------------------------------- #
def test_an_instance_the_sheet_no_longer_names_is_dropped(env):
    imported(env)
    seed_ghost(env)
    assert GHOST in instances(env)
    reimport(env)
    assert GHOST not in instances(env)


def test_the_drop_is_printed_and_reported(env, capsys):
    imported(env)
    seed_ghost(env)
    capsys.readouterr()
    reimport(env)
    printed = capsys.readouterr().out
    assert f"dropped instance {GHOST}" in printed
    assert "no longer in the sheet" in printed
    report = (env["cache"] / "import_logs_issues.md").read_text(encoding="utf-8")
    assert f"dropped instance {GHOST}" in report


def test_nothing_is_left_dangling_behind_it(env):
    imported(env)
    seed_ghost(env)
    db = open_db(env)
    try:
        edges_to_db(db, DESKTOP, [
            Edge("connected_to", GHOST, "connector.01", reason="stale", source="rule"),
        ])
    finally:
        db.close()
    reimport(env)
    db = open_db(env)
    try:
        assert GHOST not in db.instances(DESKTOP)
        for rec in db.instances(DESKTOP).values():
            assert rec.parent != GHOST
            assert rec.fastens != GHOST
            assert rec.socket_host != GHOST
            assert rec.mounted_on != GHOST
        assert not [r for r in db.relations(DESKTOP)
                    if GHOST in (r["target"], r["blocker"])]
        assert not [e for e in db.events(DESKTOP) if e.target == GHOST]
    finally:
        db.close()


def test_dropping_is_idempotent(env):
    imported(env)
    seed_ghost(env)
    assert reimport(env) == EXIT_OK
    before = sorted(instances(env))
    assert reimport(env) == EXIT_OK
    assert sorted(instances(env)) == before


def test_a_desktop_with_nothing_to_drop_says_nothing(env, capsys):
    imported(env)
    capsys.readouterr()
    reimport(env)
    assert "dropped instance" not in capsys.readouterr().out


def test_a_provisional_label_studio_instance_is_never_dropped(env):
    imported(env)
    db = open_db(env)
    try:
        db.upsert_instance(InstanceRec(key="ls:Motherboard#1", desktop=DESKTOP,
                                       cls="motherboard"))
    finally:
        db.close()
    reimport(env)
    assert "ls:Motherboard#1" in instances(env)


def test_an_implied_instance_is_kept_because_it_is_re_created(env):
    """``add_implied_instances`` puts it back into the import, so it never vanishes."""
    imported(env)
    before = {k for k, r in instances(env).items() if r.cls == "motherboard"}
    reimport(env)
    assert {k for k, r in instances(env).items() if r.cls == "motherboard"} == before


# --------------------------------------------------------------------------- #
# what stops the drop
# --------------------------------------------------------------------------- #
def keep_cases() -> list:
    """One seeded reference per table that must block the deletion."""
    def keyframe(db):
        db.add_keyframe(ShapeKeyframe(
            id=None, instance=GHOST, desktop=DESKTOP, view="scan",
            pose_segment=1, anchor_step=1,
            parts=[ShapePart(name="body", rle={"size": [4, 4], "counts": "a"})],
        ))

    def verified(db):
        db.put_compiled(FrameKey(DESKTOP, 1, "scan"), GHOST, None, 0.0, "visible",
                        "in_chassis", "verified", "h", verified_by="chang",
                        geom_type="box", box=(0, 0, 4, 4))

    def conflict(db):
        db.add_conflict(FrameKey(DESKTOP, 1, "scan"), GHOST, None, None, 9)

    def manual_event(db):
        kept = [e for e in db.events(DESKTOP)]
        kept.append(StateEvent(DESKTOP, 2, GHOST, "state", "installed", "removed",
                               auto=False))
        db.replace_events(DESKTOP, kept, auto_only=False)

    def manual_relation(db):
        edges_to_db(db, DESKTOP, [
            Edge("blocked_by", GHOST, "chassis", reason="by hand", source="manual"),
        ])

    def zorder(db):
        db.set_zorder(ZOrderRec(desktop=DESKTOP, view="scan", pose_segment=1,
                                order=[(GHOST, "body")]))

    return [
        pytest.param(keyframe, "shape_keyframe", (), id="keyframe"),
        # a verified frame makes --force refuse outright, so this one needs the
        # second key as well: the drop must still not happen behind it
        pytest.param(verified, "compiled_mask", ("--force-verified",), id="verified"),
        pytest.param(conflict, "conflict", (), id="conflict"),
        pytest.param(manual_event, "state_event", (), id="manual-event"),
        pytest.param(manual_relation, "relation", (), id="manual-relation"),
        pytest.param(zorder, "zorder", (), id="zorder"),
    ]


@pytest.mark.parametrize("seed,table,extra", keep_cases())
def test_human_work_keeps_the_instance_and_raises_an_issue(env, seed, table, extra):
    imported(env)
    seed_ghost(env)
    db = open_db(env)
    try:
        seed(db)
    finally:
        db.close()
    reimport(env, *extra)
    assert GHOST in instances(env), f"{table} should have blocked the drop"
    report = (env["cache"] / "import_logs_issues.md").read_text(encoding="utf-8")
    assert f"{GHOST} vanished from the sheet" in report
    assert table in report
    assert "S1" in report


def test_a_rule_relation_does_not_keep_it_alive(env):
    """A derived edge is machinery, not a human's work: it goes with the key."""
    imported(env)
    seed_ghost(env)
    db = open_db(env)
    try:
        edges_to_db(db, DESKTOP, [
            Edge("connected_to", GHOST, "connector.01", source="rule"),
        ])
    finally:
        db.close()
    reimport(env)
    assert GHOST not in instances(env)
