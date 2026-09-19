"""An instance must not become undeletable because it was once compiled.

``compiled_mask`` holds one row per instance the frame *needs*, and the compiler
writes one even for an instance with no geometry at all. So the moment any frame
of the desktop was compiled -- which the app does by itself, in the background --
every instance had ``compiled_mask`` rows naming it and S1's delete refused
forever. The implied motherboard is the worst case: it exists to be looked at
and said no to, and it was undeletable before the annotator ever saw it.

The rule is that an ``auto`` compiled row is a **cache**, not annotator work: it
is derived from the instance table and is re-derived on the next refresh. So
deleting an instance drops its ``auto`` rows and the affected frames' digests
(so they really are compiled again), and only a **verified** row -- a human's
signature - still refuses.
"""
from __future__ import annotations

import pytest

from tda.core.db import Db
from tda.core.model import (
    FrameKey,
    FrameOverride,
    InstanceRec,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    StateEvent,
)

DESKTOP = 13
VIEW = "scan"
PSU = "psu.01"


@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    d.upsert_desktop(DESKTOP, {})
    d.upsert_instance(InstanceRec(PSU, DESKTOP, "psu"))
    yield d
    d.close()


def _auto_row(db: Db, step: int, view: str = VIEW, instance: str = PSU) -> None:
    db.put_compiled(FrameKey(DESKTOP, step, view), instance, None, 0.0, "visible",
                    "in_chassis", "auto", "h1", geom_type="box", box=(0, 0, 4, 4))


def _verified_row(db: Db, step: int, view: str = VIEW) -> None:
    db.put_compiled(FrameKey(DESKTOP, step, view), PSU, None, 0.0, "visible",
                    "in_chassis", "verified", "h1", verified_by="chang",
                    geom_type="box", box=(0, 0, 4, 4))


def _compiled(db: Db, step: int, view: str = VIEW) -> dict:
    return db.compiled(FrameKey(DESKTOP, step, view))


# --------------------------------------------------------------------------- #
# what still refuses, and what no longer does
# --------------------------------------------------------------------------- #
def test_auto_rows_alone_are_not_a_reason_to_refuse(db: Db):
    _auto_row(db, 2)
    _auto_row(db, 3)
    assert db.instance_reference_counts(DESKTOP, PSU) == {}


def test_auto_rows_are_still_reported_for_information(db: Db):
    _auto_row(db, 2)
    _auto_row(db, 3, "oak1")
    assert db.instance_cache_counts(DESKTOP, PSU) == {"compiled_mask_auto": 2}
    assert db.instance_cache_counts(DESKTOP, "nothing.01") == {}


def test_a_verified_row_still_refuses(db: Db):
    _auto_row(db, 2)
    _verified_row(db, 3)
    counts = db.instance_reference_counts(DESKTOP, PSU)
    assert counts.get("compiled_mask") == 1  # the verified one only


@pytest.mark.parametrize("seed", ["override", "pair", "conflict", "manual_event", "zorder"])
def test_the_other_reasons_still_refuse(db: Db, seed: str):
    if seed == "override":
        db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 2, VIEW), PSU,
                                            visibility="visible"))
    elif seed == "pair":
        db.set_pair_override(PairOverride(DESKTOP, VIEW, 1, PSU, "chassis.01"))
    elif seed == "conflict":
        db.add_conflict(FrameKey(DESKTOP, 2, VIEW), PSU, None, None, 10)
    elif seed == "manual_event":
        db.replace_events(DESKTOP, [StateEvent(DESKTOP, 2, PSU, "state", "installed",
                                               "removed", auto=False)], auto_only=False)
    elif seed == "zorder":
        from tda.core.model import ZOrderRec

        db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(PSU, "main")]))
    assert db.instance_reference_counts(DESKTOP, PSU)


# --------------------------------------------------------------------------- #
# deleting clears the cache it invalidates
# --------------------------------------------------------------------------- #
def test_deleting_an_instance_drops_its_auto_rows_and_their_digests(db: Db):
    _auto_row(db, 2)
    _auto_row(db, 3, "oak1")
    _auto_row(db, 2, VIEW, "chassis.01")  # another instance of the same frame
    for key in (FrameKey(DESKTOP, 2, VIEW), FrameKey(DESKTOP, 3, "oak1")):
        db.set_frame_digest(key, "d", "v1", 1)

    db.delete_instance(DESKTOP, PSU)

    assert PSU not in _compiled(db, 2)
    assert PSU not in _compiled(db, 3, "oak1")
    assert "chassis.01" in _compiled(db, 2)  # the neighbour's row is untouched
    # ... and the frames will really be compiled again
    assert db.frame_digest(FrameKey(DESKTOP, 2, VIEW)) is None
    assert db.frame_digest(FrameKey(DESKTOP, 3, "oak1")) is None


def test_a_verified_row_is_never_dropped_by_the_delete(db: Db):
    """The guard refuses first; if a caller bypasses it, nothing frozen is lost."""
    _verified_row(db, 2)
    db.delete_instance(DESKTOP, PSU)
    assert PSU in _compiled(db, 2)


def test_an_untouched_frames_digest_survives(db: Db):
    _auto_row(db, 2)
    other = FrameKey(DESKTOP, 9, VIEW)
    db.set_frame_digest(other, "d", "v1", 1)
    db.delete_instance(DESKTOP, PSU)
    assert db.frame_digest(other) is not None


# --------------------------------------------------------------------------- #
# ... and the same through the S1 command
# --------------------------------------------------------------------------- #
def test_s1_can_delete_an_instance_a_background_compile_touched(db: Db):
    from tda.core.taxonomy import load_taxonomy
    from tda.ui.steps_delete import delete_instance
    from tda.ui.steps_model import StepTableData

    _auto_row(db, 2)
    _auto_row(db, 3)
    data = StepTableData.load(db, DESKTOP, load_taxonomy())
    delete_instance(data, db, PSU)
    assert PSU not in db.instances(DESKTOP)
    assert PSU not in _compiled(db, 2)


def test_s1_still_refuses_when_a_human_froze_a_row(db: Db):
    from tda.core.taxonomy import load_taxonomy
    from tda.ui.steps_delete import delete_instance
    from tda.ui.steps_values import EditError
    from tda.ui.steps_model import StepTableData

    _verified_row(db, 2)
    data = StepTableData.load(db, DESKTOP, load_taxonomy())
    with pytest.raises(EditError) as err:
        delete_instance(data, db, PSU)
    assert "compiled_mask" in str(err.value)
    assert PSU in db.instances(DESKTOP)


def test_s1_still_refuses_on_a_keyframe(db: Db):
    from tda.core.taxonomy import load_taxonomy
    from tda.ui.steps_delete import delete_instance
    from tda.ui.steps_values import EditError
    from tda.ui.steps_model import StepTableData

    _auto_row(db, 2)
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=PSU, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=2, parts=[ShapePart("main", box=(0, 0, 4, 4))], geom_type="box",
    ))
    data = StepTableData.load(db, DESKTOP, load_taxonomy())
    with pytest.raises(EditError):
        delete_instance(data, db, PSU)
    assert PSU in db.instances(DESKTOP)
