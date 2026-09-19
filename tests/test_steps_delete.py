"""Deleting an instance in stage S1, and the issues that guard the instance table.

Removing an identity row is the one step-table command that cannot wait for
``Apply``: :meth:`~tda.ui.steps_model.StepTableData.save` only upserts, so a
deletion has to be written through. These tests pin that it is written through
*completely* -- the references other instances held to the deleted key are
cleared in the database as well, in the same transaction -- and that it is
refused whenever anything else still depends on the key.
"""
from __future__ import annotations

import pytest

from steps_fixtures import seeded_db
from tda.core.model import (
    FrameKey,
    InstanceRec,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    StateEvent,
)
from tda.core.taxonomy import load_taxonomy
from tda.ui.steps_model import EditError, StepTableData

ORPHAN = "screw.cpu_cooler.02"
NEIGHBOUR = "screw.cpu_cooler.01"


@pytest.fixture
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_db_path, tax):
    conn = seeded_db(tmp_db_path, tax)
    yield conn
    conn.close()


@pytest.fixture
def data(db, tax) -> StepTableData:
    return StepTableData.load(db, 13, tax)


@pytest.fixture
def orphaned(data) -> StepTableData:
    """D13 with step 4 retargeted, so ``screw.cpu_cooler.02`` is unreferenced."""
    data.change_target(4, target=NEIGHBOUR)
    return data


# --------------------------------------------------------------------------- #
# orphans
# --------------------------------------------------------------------------- #
def test_issues_flag_an_instance_no_action_targets_any_more(orphaned):
    assert orphaned.orphans == [
        f"no action references {ORPHAN} - delete it or retarget a step at it"
    ]
    assert orphaned.issues[-1] == orphaned.orphans[0]


def test_the_implicit_chassis_is_never_reported_as_an_orphan(data):
    assert "chassis" in data.instances
    assert not any("chassis" in text for text in data.orphans)


# --------------------------------------------------------------------------- #
# the happy path, all the way to the stored rows
# --------------------------------------------------------------------------- #
def test_delete_instance_drops_the_row_and_the_references_to_it(db, orphaned):
    orphaned.apply_instance_edit(NEIGHBOUR, "parent", ORPHAN)
    orphaned.delete_instance(db, ORPHAN)

    assert ORPHAN not in orphaned.instances
    assert orphaned.instances[NEIGHBOUR].parent is None
    # nothing points at the deleted key any more; what is left is the parent the
    # edit above overwrote, which the S1 list now asks about in its own right
    assert not any(ORPHAN in text for text in orphaned.orphans)
    assert orphaned.orphans == [
        f"captive screw without parent: {NEIGHBOUR} is captive but leaves the "
        f"chassis with nothing - name the part it stays in"
    ]
    assert ORPHAN not in db.instances(13)


def test_delete_instance_clears_the_reference_in_the_database_too(db, orphaned, tax):
    """Reverting after a delete must not bring a dangling pointer back."""
    orphaned.apply_instance_edit(NEIGHBOUR, "parent", ORPHAN)
    orphaned.apply_instance_edit(NEIGHBOUR, "mounted_on", ORPHAN)
    orphaned.save(db)
    orphaned.delete_instance(db, ORPHAN)

    reloaded = StepTableData.load(db, 13, tax)  # what Revert does
    assert ORPHAN not in reloaded.instances
    assert reloaded.instances[NEIGHBOUR].parent is None
    assert reloaded.instances[NEIGHBOUR].mounted_on is None
    assert not [
        (key, field)
        for key, inst in reloaded.instances.items()
        for field in ("parent", "mounted_on", "fastens", "socket_host")
        if getattr(inst, field) == ORPHAN
    ]
    # the only question left is the parent this test itself overwrote
    assert reloaded.issues  # an empty list would satisfy the `all` below for free
    assert not any("is not an instance" in text for text in reloaded.issues)
    assert all("captive screw without parent" in text for text in reloaded.issues)


def test_delete_instance_does_not_flush_the_neighbours_other_unsaved_edits(db, orphaned, tax):
    orphaned.apply_instance_edit(NEIGHBOUR, "parent", ORPHAN)
    orphaned.save(db)
    orphaned.apply_instance_edit(NEIGHBOUR, "head", "T15")  # unsaved on purpose
    orphaned.delete_instance(db, ORPHAN)

    stored = StepTableData.load(db, 13, tax).instances[NEIGHBOUR]
    assert stored.parent is None  # the dangling pointer was written away
    assert stored.attrs.get("head") is None  # the unrelated edit stayed unsaved
    assert orphaned.instances[NEIGHBOUR].attrs["head"] == "T15"  # still in memory


def test_delete_instance_drops_the_derived_state_events_of_the_key(db, orphaned):
    assert any(e.target == ORPHAN for e in db.events(13))
    orphaned.delete_instance(db, ORPHAN)
    assert not any(e.target == ORPHAN for e in db.events(13))
    assert db.events(13)  # the rest of the log is untouched


def test_delete_instance_rolls_back_and_leaves_memory_untouched_on_failure(
    db, orphaned, tax, monkeypatch
):
    orphaned.apply_instance_edit(NEIGHBOUR, "parent", ORPHAN)
    orphaned.save(db)
    monkeypatch.setattr(
        db, "upsert_instance", lambda inst: (_ for _ in ()).throw(RuntimeError("disk full"))
    )
    with pytest.raises(EditError) as err:
        orphaned.delete_instance(db, ORPHAN)
    assert "disk full" in str(err.value)

    assert ORPHAN in orphaned.instances  # memory untouched
    assert orphaned.instances[NEIGHBOUR].parent == ORPHAN
    monkeypatch.undo()
    reloaded = StepTableData.load(db, 13, tax)  # database untouched
    assert ORPHAN in reloaded.instances
    assert reloaded.instances[NEIGHBOUR].parent == ORPHAN


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_delete_instance_refuses_while_a_step_still_targets_it(db, data):
    with pytest.raises(EditError) as err:
        data.delete_instance(db, ORPHAN)
    assert "step(s) 4" in str(err.value)
    assert ORPHAN in data.instances


def test_delete_instance_refuses_while_a_keyframe_references_it(db, orphaned):
    db.add_keyframe(ShapeKeyframe(None, ORPHAN, 13, "oak1", 0, 9, parts=[ShapePart("main")]))
    with pytest.raises(EditError) as err:
        orphaned.delete_instance(db, ORPHAN)
    assert "oak1" in str(err.value)
    assert ORPHAN in orphaned.instances


def test_delete_instance_refuses_while_a_relation_references_it(db, orphaned):
    db.add_relation(13, "fastened_by", "cpu_cooler.fan.01", ORPHAN)
    with pytest.raises(EditError) as err:
        orphaned.delete_instance(db, ORPHAN)
    assert "fastened_by" in str(err.value)


def test_delete_instance_refuses_while_a_frame_override_references_it(db, orphaned):
    from tda.core.model import FrameOverride

    db.set_frame_override(FrameOverride(FrameKey(13, 4, "scan"), ORPHAN, visibility="visible"))
    with pytest.raises(EditError) as err:
        orphaned.delete_instance(db, ORPHAN)
    assert "frame_override" in str(err.value)


def test_delete_instance_refuses_while_a_pair_override_references_it(db, orphaned):
    db.set_pair_override(PairOverride(13, "scan", 0, NEIGHBOUR, ORPHAN))
    with pytest.raises(EditError) as err:
        orphaned.delete_instance(db, ORPHAN)
    assert "pair_override" in str(err.value)


def test_delete_instance_refuses_while_verified_truth_references_it(db, orphaned):
    """A frozen row is a signature. An ``auto`` row is a cache and goes with it."""
    db.put_compiled(FrameKey(13, 4, "scan"), ORPHAN, None, 0.0, "visible", "in_chassis",
                    "verified", "hash", verified_by="chang")
    with pytest.raises(EditError) as err:
        orphaned.delete_instance(db, ORPHAN)
    assert "compiled_mask" in str(err.value)


def test_delete_instance_clears_the_auto_truth_rows_instead_of_refusing(db, orphaned):
    db.put_compiled(FrameKey(13, 4, "scan"), ORPHAN, None, 0.0, "visible", "in_chassis",
                    "auto", "hash")
    orphaned.delete_instance(db, ORPHAN)
    assert ORPHAN not in db.compiled(FrameKey(13, 4, "scan"))


def test_delete_instance_refuses_while_an_open_conflict_references_it(db, orphaned):
    cid = db.add_conflict(FrameKey(13, 4, "scan"), ORPHAN, None, None, 40)
    with pytest.raises(EditError):
        orphaned.delete_instance(db, ORPHAN)

    db.resolve_conflict(cid, "keep_old")  # a closed conflict is history, not a blocker
    orphaned.delete_instance(db, ORPHAN)
    assert ORPHAN not in orphaned.instances


def test_delete_instance_refuses_while_a_hand_written_event_references_it(db, orphaned):
    manual = StateEvent(13, 4, ORPHAN, "state", "fastened", "loosened", auto=False)
    db.replace_events(13, [manual], auto_only=False)
    with pytest.raises(EditError) as err:
        orphaned.delete_instance(db, ORPHAN)
    assert "state_event" in str(err.value)


def test_delete_instance_refuses_the_chassis_and_unknown_keys(db, data):
    with pytest.raises(EditError):
        data.delete_instance(db, "chassis")
    with pytest.raises(EditError):
        data.delete_instance(db, "nope.01")


# --------------------------------------------------------------------------- #
# dangling references can never be silent
# --------------------------------------------------------------------------- #
def test_a_reference_to_a_key_that_is_not_an_instance_is_reported(db, tax):
    stored = db.instances(13)[NEIGHBOUR]
    stored.parent = "screw.cpu_cooler.99"
    db.upsert_instance(stored)

    issues = StepTableData.load(db, 13, tax).issues
    assert any(
        f"{NEIGHBOUR}.parent" in text and "screw.cpu_cooler.99" in text for text in issues
    )


def test_a_bare_taxonomy_class_name_is_not_a_dangling_reference(db, tax):
    """A class name in ``socket_host`` is unfinished work, never a broken pointer.

    ``logs.py`` writes ``socket_host='motherboard'`` for every motherboard-side
    connector, and the heuristic of spec 7.3 narrows it to ``motherboard.01``
    during the import -- but only while the desktop has exactly one motherboard.
    When it cannot (two boards, or none), the class name stays, and the step
    table must ask about it as *unresolved*, not report a dangling pointer.
    """
    stored = db.instances(13)["connector.03"]
    assert stored.socket_host == "motherboard.01"  # the import resolved it
    stored.socket_host = "motherboard"
    db.upsert_instance(stored)

    issues = StepTableData.load(db, 13, tax).issues
    assert not any("is not an instance" in text for text in issues)
    assert [text for text in issues if "unresolved socket host" in text] == [
        "unresolved socket host: connector.03.socket_host is still the class "
        "'motherboard' - name the instance it plugs into"
    ]


def test_an_instance_that_no_longer_exists_leaves_a_dangling_reference(db, tax):
    db.upsert_instance(InstanceRec(key="misc_part.01", desktop=13, cls="misc_part",
                                   mounted_on="gone.01"))
    issues = StepTableData.load(db, 13, tax).issues
    assert any("misc_part.01.mounted_on" in text and "gone.01" in text for text in issues)
