"""Tests for frozen truth rows: disagreements, the conflict queue and resolving it.

Compiling, refreshing and verifying are covered in ``test_truth.py``; the scene
both modules use is ``truth_scenes.py``.
"""
from __future__ import annotations

import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.model import FrameOverride, InstanceRec, ShapePart, StateEvent, ZOrderRec
from tda.core.truth import StaleConflictError
from truth_scenes import (
    BENCH_BOX,
    DESKTOP,
    FAN,
    FAN_RECT,
    PSU,
    SCREW,
    VIEW,
    Scene,
    build_scene,
    mask_kf,
    rect,
    replace_parts,
)


@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    yield d
    d.close()


@pytest.fixture
def scene(db: Db) -> Scene:
    return build_scene(db)


# --------------------------------------------------------------------------- #
# re-compilation against frozen rows
# --------------------------------------------------------------------------- #
def test_shifting_a_shape_updates_auto_rows_and_leaves_the_verified_one(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen = {inst: dict(row) for inst, row in scene.rows(2).items()}

    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(11, 10, 51, 50)))])
    out = scene.refresh_all()

    assert out["conflicts"] == 0
    assert out["updated"] == 4  # both instances of step 1 and step 3
    assert out["skipped"] == 2  # the two frozen rows of step 2
    assert scene.db.conflicts(DESKTOP) == []
    assert masks.bbox(masks.decode_rle(scene.row(1, PSU)["visible_rle"])) == (11, 10, 51, 50)
    assert scene.rows(2) == frozen  # byte for byte, including the input hash
    assert scene.review_status(2) == "verified"


def test_eroding_a_shape_conflicts_with_the_verified_row_only(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen_counts = scene.counts(2, PSU)

    # 40% off the PSU: 40x40 -> 40x24
    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 34)))])
    out = scene.refresh_all()

    assert out["conflicts"] == 1
    assert out["updated"] == 4  # steps 1 and 3 follow the new shape
    conflicts = scene.db.conflicts(DESKTOP)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert (conflict["instance"], conflict["step"], conflict["status"]) == (PSU, 2, "open")
    assert conflict["old_rle"]["counts"] == frozen_counts
    assert masks.bbox(masks.decode_rle(conflict["new_rle"])) == (10, 10, 50, 34)
    assert conflict["sym_diff_px"] > 100
    # the frozen row itself is untouched
    assert scene.counts(2, PSU) == frozen_counts
    assert scene.row(2, PSU)["status"] == "verified"
    assert masks.bbox(masks.decode_rle(scene.row(1, PSU)["visible_rle"])) == (10, 10, 50, 34)


def test_the_same_disagreement_is_queued_only_once(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 34)))])
    scene.svc.refresh(scene.key(2))

    out = scene.svc.refresh(scene.key(2))

    assert out["conflicts"] == 1  # still in disagreement
    assert len(scene.db.conflicts(DESKTOP)) == 1  # but not queued twice


def test_a_moved_bench_box_conflicts_only_beyond_two_pixels(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(3), "lin")

    replace_parts(scene, scene.bench_kf, [ShapePart("main", None, (3.0, 2.0, 13.0, 12.0))])
    assert scene.svc.refresh(scene.key(3))["conflicts"] == 0

    replace_parts(scene, scene.bench_kf, [ShapePart("main", None, (7.0, 2.0, 17.0, 12.0))])
    out = scene.svc.refresh(scene.key(3))

    assert out["conflicts"] == 1
    conflict = scene.db.conflicts(DESKTOP)[-1]
    assert conflict["instance"] == SCREW
    assert conflict["old_rle"] == {"box": list(BENCH_BOX)}
    assert conflict["new_rle"] == {"box": [7.0, 2.0, 17.0, 12.0]}
    assert conflict["sym_diff_px"] == 2 * 5 * 10  # two 5x10 slivers
    assert scene.row(3, SCREW)["box"] == list(BENCH_BOX)  # the frozen box is untouched


def test_an_instance_added_to_a_verified_frame_demotes_it(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen_counts = scene.counts(2, PSU)

    scene.db.upsert_instance(InstanceRec(key=FAN, desktop=DESKTOP, cls="case_fan"))
    scene.db.add_keyframe(mask_kf(FAN, FAN_RECT, anchor=3))
    scene.db.set_zorder(
        ZOrderRec(DESKTOP, VIEW, 1, [(PSU, "main"), (SCREW, "main"), (FAN, "main")], version=2)
    )
    out = scene.svc.refresh(scene.key(2))

    assert sorted(scene.rows(2)) == [FAN, PSU, SCREW]
    assert scene.row(2, FAN)["status"] == "auto"
    assert out["updated"] == 1 and out["conflicts"] == 0
    assert scene.review_status(2) == "needs_review"
    assert scene.counts(2, PSU) == frozen_counts  # the frozen rows stay frozen
    assert scene.row(2, PSU)["status"] == "verified"
    demotions = [op for op in scene.db.ops(DESKTOP, VIEW) if op["kind"] == "demote_frame"]
    assert FAN in demotions[0]["payload"]["reason"]


def test_an_instance_dropped_from_a_verified_frame_conflicts_and_demotes(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen_counts = scene.counts(2, SCREW)
    frozen_area = masks.area(masks.decode_rle(scene.row(2, SCREW)["visible_rle"]))

    # the screw was put down outside every view from step 2 on: no geometry
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, SCREW, "placement", "in_chassis", "elsewhere", auto=False)],
        auto_only=False,
    )
    out = scene.svc.refresh(scene.key(2))

    assert out["conflicts"] == 1
    conflict = scene.db.conflicts(DESKTOP)[0]
    assert conflict["instance"] == SCREW
    assert conflict["new_rle"] is None
    assert conflict["old_rle"]["counts"] == frozen_counts
    assert conflict["sym_diff_px"] == frozen_area
    assert scene.counts(2, SCREW) == frozen_counts  # kept, never deleted
    assert scene.row(2, SCREW)["status"] == "verified"
    assert scene.review_status(2) == "needs_review"

    # accepting that emptiness is what finally removes the row
    scene.svc.resolve_conflict(conflict["id"], "accept_new", "lin")
    assert sorted(scene.rows(2)) == [PSU]
    assert scene.db.conflicts(DESKTOP) == []


def test_an_instance_dropped_from_an_auto_frame_deletes_its_row(scene: Scene):
    scene.refresh_all()
    # a manual event wins over the recorded action within the same step
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 3, SCREW, "placement", "on_bench", "elsewhere", auto=False)],
        auto_only=False,
    )

    out = scene.svc.refresh(scene.key(3))

    assert sorted(scene.rows(3)) == [PSU]
    assert out["conflicts"] == 0
    assert out["updated"] == 2  # the PSU row rewritten, the screw row deleted
    assert scene.db.conflicts(DESKTOP) == []


# --------------------------------------------------------------------------- #
# conflict resolution
# --------------------------------------------------------------------------- #
def _conflicting_scene(scene: Scene) -> int:
    """Verify step 2, erode the PSU, refresh: returns the conflict's id."""
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 34)))])
    scene.svc.refresh(scene.key(2))
    return scene.db.conflicts(DESKTOP)[0]["id"]


def test_resolving_keep_old_pins_the_frame_so_nothing_comes_back(scene: Scene):
    cid = _conflicting_scene(scene)
    frozen_counts = scene.counts(2, PSU)

    scene.svc.resolve_conflict(cid, "keep_old", "lin")

    assert scene.db.conflicts(DESKTOP) == []  # closed
    assert scene.db.get_conflict(cid)["resolution"] == "keep_old"
    assert scene.counts(2, PSU) == frozen_counts
    assert scene.db.frame_overrides(scene.key(2))[PSU].visible_rle["counts"] == frozen_counts

    out = scene.svc.refresh(scene.key(2))  # the compiler now agrees with the frozen row
    assert out["conflicts"] == 0
    assert scene.db.conflicts(DESKTOP) == []
    assert scene.counts(2, PSU) == frozen_counts
    assert [op["kind"] for op in scene.db.ops(DESKTOP, VIEW)][0] == "resolve_conflict"


def test_resolving_accept_new_writes_the_new_mask_into_the_frozen_row(scene: Scene):
    cid = _conflicting_scene(scene)

    scene.svc.resolve_conflict(cid, "accept_new", "lin")

    row = scene.row(2, PSU)
    assert masks.bbox(masks.decode_rle(row["visible_rle"])) == (10, 10, 50, 34)
    assert row["status"] == "verified" and row["verified_by"] == "lin"
    assert scene.db.conflicts(DESKTOP) == []
    again = scene.svc.refresh(scene.key(2))
    assert {k: again[k] for k in ("updated", "conflicts", "skipped", "problems")} == {
        "updated": 0, "conflicts": 0, "skipped": 2, "problems": [],
    }


def test_resolving_accept_new_keeps_a_box_row_a_box(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(3), "lin")
    replace_parts(scene, scene.bench_kf, [ShapePart("main", None, (7.0, 2.0, 17.0, 12.0))])
    scene.svc.refresh(scene.key(3))
    cid = scene.db.conflicts(DESKTOP)[0]["id"]

    scene.svc.resolve_conflict(cid, "accept_new", "lin")

    row = scene.row(3, SCREW)
    assert row["geom_type"] == "box" and row["visible_rle"] is None
    assert row["box"] == [7.0, 2.0, 17.0, 12.0]
    assert row["status"] == "verified"
    assert scene.svc.refresh(scene.key(3))["conflicts"] == 0


def test_accept_new_writes_the_geometry_the_inputs_say_now(scene: Scene):
    cid = _conflicting_scene(scene)  # the conflict carries (10, 10, 50, 34)

    # a re-trace within tolerance: the conflict is still about this disagreement,
    # but the truth the inputs describe has moved by a pixel
    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(11, 10, 51, 34)))])
    scene.svc.resolve_conflict(cid, "accept_new", "lin")

    row = scene.row(2, PSU)
    assert masks.bbox(masks.decode_rle(row["visible_rle"])) == (11, 10, 51, 34)
    assert row["status"] == "verified"
    again = scene.svc.refresh(scene.key(2))
    assert {k: again[k] for k in ("updated", "conflicts", "skipped", "problems")} == {
        "updated": 0, "conflicts": 0, "skipped": 2, "problems": [],
    }


def test_accept_new_refuses_a_conflict_the_inputs_have_overtaken(scene: Scene):
    cid = _conflicting_scene(scene)  # the conflict carries (10, 10, 50, 34)
    frozen_counts = scene.counts(2, PSU)

    # the shape moved again since the conflict was queued: accepting the queued
    # value would confirm something nobody has seen
    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 26)))])
    with pytest.raises(StaleConflictError):
        scene.svc.resolve_conflict(cid, "accept_new", "lin")

    stale = scene.db.get_conflict(cid)
    assert stale["status"] == "resolved" and stale["resolution"] == "superseded"
    fresh = scene.db.conflicts(DESKTOP)
    assert len(fresh) == 1 and fresh[0]["id"] != cid
    assert masks.bbox(masks.decode_rle(fresh[0]["new_rle"])) == (10, 10, 50, 26)
    assert fresh[0]["old_rle"]["counts"] == frozen_counts
    assert scene.counts(2, PSU) == frozen_counts  # nothing was written into the row

    scene.svc.resolve_conflict(fresh[0]["id"], "accept_new", "lin")  # the fresh one works
    assert masks.bbox(masks.decode_rle(scene.row(2, PSU)["visible_rle"])) == (10, 10, 50, 26)
    assert scene.db.conflicts(DESKTOP) == []


def test_accept_new_is_superseded_when_the_instance_left_the_frame(scene: Scene):
    cid = _conflicting_scene(scene)  # the conflict carries a mask for the PSU
    frozen_counts = scene.counts(2, PSU)

    # the PSU is now recorded as being nowhere in view from step 2 on
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, PSU, "placement", "in_chassis", "elsewhere", auto=False)],
        auto_only=False,
    )
    with pytest.raises(StaleConflictError):
        scene.svc.resolve_conflict(cid, "accept_new", "lin")

    assert scene.db.get_conflict(cid)["resolution"] == "superseded"
    fresh = scene.db.conflicts(DESKTOP)
    assert len(fresh) == 1
    assert fresh[0]["new_rle"] is None  # the disagreement is now about its absence
    assert fresh[0]["old_rle"]["counts"] == frozen_counts
    assert scene.counts(2, PSU) == frozen_counts


def test_keep_old_merges_into_an_existing_frame_override(scene: Scene):
    scene.refresh_all()
    scene.db.set_frame_override(FrameOverride(scene.key(2), PSU, visibility="occluded_full"))
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen_counts = scene.counts(2, PSU)
    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 34)))])
    scene.svc.refresh(scene.key(2))
    cid = scene.db.conflicts(DESKTOP)[0]["id"]

    scene.svc.resolve_conflict(cid, "keep_old", "lin")

    override = scene.db.frame_overrides(scene.key(2))[PSU]
    assert override.visible_rle["counts"] == frozen_counts  # the pin
    assert override.visibility == "occluded_full"  # and what was there before
    assert scene.svc.refresh(scene.key(2))["conflicts"] == 0
    assert scene.row(2, PSU)["visibility"] == "occluded_full"


def test_keep_old_is_refused_when_the_instance_left_the_frame(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen_counts = scene.counts(2, SCREW)
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, SCREW, "placement", "in_chassis", "elsewhere", auto=False)],
        auto_only=False,
    )
    scene.svc.refresh(scene.key(2))
    conflict = scene.db.conflicts(DESKTOP)[0]

    with pytest.raises(ValueError) as err:
        scene.svc.resolve_conflict(conflict["id"], "keep_old", "lin")

    assert SCREW in str(err.value) and "state" in str(err.value)
    assert scene.db.get_conflict(conflict["id"])["status"] == "open"  # nothing settled
    assert SCREW not in scene.db.frame_overrides(scene.key(2))  # and nothing written
    assert scene.counts(2, SCREW) == frozen_counts
    assert [op["kind"] for op in scene.db.ops(DESKTOP, VIEW)][0] != "resolve_conflict"


def test_keep_old_pins_a_frozen_absence(scene: Scene):
    scene.refresh_all()
    scene.db.set_frame_override(FrameOverride(scene.key(2), PSU, visibility="out_of_view"))
    scene.svc.verify_frame(scene.key(2), "lin")
    assert scene.row(2, PSU)["visible_rle"] is None  # frozen as "not in this frame"

    scene.db.set_frame_override(FrameOverride(scene.key(2), PSU))  # the override is cleared
    scene.svc.refresh(scene.key(2))
    conflict = scene.db.conflicts(DESKTOP)[0]
    assert conflict["old_rle"] is None and conflict["new_rle"] is not None

    scene.svc.resolve_conflict(conflict["id"], "keep_old", "lin")

    override = scene.db.frame_overrides(scene.key(2))[PSU]
    assert override.visibility == "out_of_view"
    assert override.visible_rle is None
    assert scene.svc.refresh(scene.key(2))["conflicts"] == 0
    assert scene.row(2, PSU)["visible_rle"] is None


def test_resolve_conflict_refuses_a_conflict_that_is_already_closed(scene: Scene):
    cid = _conflicting_scene(scene)
    scene.svc.resolve_conflict(cid, "edited", "lin")

    with pytest.raises(ValueError) as err:
        scene.svc.resolve_conflict(cid, "keep_old", "lin")

    assert "resolved" in str(err.value)
    assert PSU not in scene.db.frame_overrides(scene.key(2))


def test_a_resolved_disagreement_can_be_raised_again(scene: Scene):
    cid = _conflicting_scene(scene)
    scene.svc.resolve_conflict(cid, "edited", "lin")  # closed without touching anything

    out = scene.svc.refresh(scene.key(2))

    assert out["conflicts"] == 1  # the inputs still disagree, and nothing pinned them
    open_now = scene.db.conflicts(DESKTOP)
    assert len(open_now) == 1 and open_now[0]["id"] != cid


def test_resolve_conflict_rejects_an_unknown_resolution(scene: Scene):
    cid = _conflicting_scene(scene)
    frozen_counts = scene.counts(2, PSU)

    with pytest.raises(ValueError):
        scene.svc.resolve_conflict(cid, "whatever", "lin")
    with pytest.raises(ValueError):
        scene.svc.resolve_conflict(cid, "superseded", "lin")  # the service's own, not a choice

    assert scene.db.get_conflict(cid)["status"] == "open"
    assert scene.counts(2, PSU) == frozen_counts
