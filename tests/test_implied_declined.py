"""Deleting an implied instance in S1 has to stick.

An implied instance is a *judgement* about what the dataset should contain, so
the annotator may overrule it -- and that is the one thing the old code could
not hear. ``import-logs`` always implies, and it runs again on every re-import,
so the deleted motherboard came back the next day with a mask on every frame.

The refusal is durable: the delete records the class in the desktop's
``implied_declined`` meta, in the same transaction, and both call sites pass it
to :func:`tda.core.implied.implied_instances`. ``infer-relations --add-implied
--reset-declined`` is how it is taken back.
"""
from __future__ import annotations

import pytest
from test_cli import d13_steps, env, open_db, run  # noqa: F401  (re-used fixtures)
from test_implied import _d64_actions, _d64_like, tax  # noqa: F401

from tda.cli import EXIT_OK
from tda.core.db import Db
from tda.core.implied import implied_instances, is_implied
from tda.core.model import InstanceRec
from tda.core.taxonomy import load_taxonomy
from tda.ui.steps_delete import delete_instance
from tda.ui.steps_issues import unresolved_issues
from tda.ui.steps_model import StepTableData

BOARD = "motherboard.01"


def _imported(env: dict) -> None:
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK


def _delete_board(env: dict, desktop: int = 63) -> None:
    db = open_db(env)
    try:
        delete_instance(StepTableData.load(db, desktop, load_taxonomy()), db, BOARD)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the pure function
# --------------------------------------------------------------------------- #
def test_a_declined_class_is_never_implied_again(tax):
    instances = _d64_like()
    assert implied_instances(instances, _d64_actions(), tax,
                             declined={"motherboard"}) == []


def test_declining_a_different_class_changes_nothing(tax):
    made = implied_instances(_d64_like(), _d64_actions(), tax, declined={"psu"})
    assert [rec.key for rec in made] == [BOARD]


# --------------------------------------------------------------------------- #
# the marker
# --------------------------------------------------------------------------- #
def test_the_database_remembers_a_declined_class(tmp_db_path):
    db = Db(tmp_db_path)
    try:
        assert db.declined_implied(63) == set()
        db.decline_implied(63, "motherboard")
        assert db.declined_implied(63) == {"motherboard"}
        db.decline_implied(63, "motherboard")  # idempotent
        assert db.declined_implied(63) == {"motherboard"}
        db.decline_implied(63, "psu")
        assert db.declined_implied(63) == {"motherboard", "psu"}
        assert db.declined_implied(13) == set()  # per desktop
        db.reset_declined_implied(63)
        assert db.declined_implied(63) == set()
    finally:
        db.close()


def test_declining_keeps_the_rest_of_the_desktop_meta(tmp_db_path):
    db = Db(tmp_db_path)
    try:
        db.upsert_desktop(63, {"brand": "Dell", "index_n_steps": 37})
        db.decline_implied(63, "motherboard")
        meta = db.get_desktop(63)
        assert meta["brand"] == "Dell" and meta["index_n_steps"] == 37
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# through the S1 delete
# --------------------------------------------------------------------------- #
def test_deleting_an_implied_instance_is_remembered(env):
    _imported(env)
    _delete_board(env)
    db = open_db(env)
    try:
        assert BOARD not in db.instances(63)
        assert db.declined_implied(63) == {"motherboard"}
    finally:
        db.close()


def test_deleting_an_ordinary_instance_declines_nothing(env):
    _imported(env)
    db = open_db(env)
    try:
        db.upsert_instance(InstanceRec("psu.09", 63, "psu"))  # no action names it
        data = StepTableData.load(db, 63, load_taxonomy())
        assert not is_implied(data.instances["psu.09"])
        delete_instance(data, db, "psu.09")
        assert "psu.09" not in db.instances(63)
        assert db.declined_implied(63) == set()
    finally:
        db.close()


def test_deleting_the_implied_board_hands_the_references_back(env):
    """The connectors go back to naming the class, as the importer wrote them."""
    _imported(env)
    _delete_board(env)
    db = open_db(env)
    try:
        hosts = {r.socket_host for r in db.instances(63).values() if r.cls == "connector"}
        assert "motherboard" in hosts  # not None: still unresolved, not forgotten
    finally:
        db.close()


def test_a_failed_delete_declines_nothing(env, monkeypatch):
    """The marker rides in the delete's own transaction, so it rolls back too."""
    _imported(env)
    db = open_db(env)
    try:
        data = StepTableData.load(db, 63, load_taxonomy())
        monkeypatch.setattr(Db, "delete_instance",
                            lambda self, d, k: (_ for _ in ()).throw(RuntimeError("no")))
        with pytest.raises(Exception):
            delete_instance(data, db, BOARD)
        assert db.declined_implied(63) == set()
        assert BOARD in db.instances(63)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# ... and it survives both ways back in
# --------------------------------------------------------------------------- #
def test_a_forced_reimport_does_not_bring_it_back(env):
    _imported(env)
    _delete_board(env)
    assert run(env, "import-logs", "--desktops", "63", "--force") == EXIT_OK
    db = open_db(env)
    try:
        assert BOARD not in db.instances(63)
        assert db.declined_implied(63) == {"motherboard"}
    finally:
        db.close()


def test_add_implied_does_not_bring_it_back(env):
    _imported(env)
    _delete_board(env)
    assert run(env, "infer-relations", "--add-implied", "--desktops", "63") == EXIT_OK
    db = open_db(env)
    try:
        assert BOARD not in db.instances(63)
    finally:
        db.close()


def test_reset_declined_brings_it_back(env):
    _imported(env)
    _delete_board(env)
    assert run(env, "infer-relations", "--add-implied", "--reset-declined",
               "--desktops", "63") == EXIT_OK
    db = open_db(env)
    try:
        assert BOARD in db.instances(63)
        assert db.declined_implied(63) == set()
    finally:
        db.close()


def test_reset_declined_only_touches_the_selected_desktops(env):
    _imported(env)
    db = open_db(env)
    try:
        db.decline_implied(13, "motherboard")
        db.decline_implied(63, "motherboard")
    finally:
        db.close()
    run(env, "infer-relations", "--add-implied", "--reset-declined", "--desktops", "63")
    db = open_db(env)
    try:
        assert db.declined_implied(13) == {"motherboard"}
        assert db.declined_implied(63) == set()
    finally:
        db.close()


def test_reset_declined_needs_add_implied(env, capsys):
    _imported(env)
    capsys.readouterr()
    assert run(env, "infer-relations", "--reset-declined", "--desktops", "63") != EXIT_OK
    assert "--add-implied" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# S1 says so before the annotator finds out the hard way
# --------------------------------------------------------------------------- #
def test_the_step_table_says_deleting_is_remembered(tax):
    instances = _d64_like()
    for rec in implied_instances(instances, _d64_actions(), tax):
        instances[rec.key] = rec
    line = next(text for text in unresolved_issues(instances, tax)
                if text.startswith("implied instance"))
    assert "remembered" in line
