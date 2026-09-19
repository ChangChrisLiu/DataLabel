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
from tda.core.truth_conflicts import payload_labels
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
# labels are part of the frozen truth too
# --------------------------------------------------------------------------- #
def _verified_with_new_visibility(scene: Scene) -> dict:
    """Confirm step 1, then press 2 on the PSU; returns the queued conflict."""
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(1), "lin")
    scene.db.set_frame_override(
        FrameOverride(scene.key(1), PSU, None, "occluded_partial")
    )
    scene.svc.refresh(scene.key(1))
    return scene.db.conflicts(DESKTOP)[0]


def test_a_label_change_on_a_verified_row_is_a_disagreement(scene: Scene):
    """The reproduction: the 1-7 shortcut on a confirmed frame did nothing.

    ``disagreement`` compared pixels only, so a changed ``visibility`` counted
    as "no disagreement": the refresh reported 0 updated and 0 conflicts, left
    the row saying ``visible`` and stamped the digest -- after which no pass
    ever looked at the frame again. ``visibility`` is exported per annotation
    and is the ground truth of a VLM task.
    """
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(1), "lin")
    scene.db.set_frame_override(
        FrameOverride(scene.key(1), PSU, None, "occluded_partial")
    )

    out = scene.svc.refresh(scene.key(1))

    assert out["conflicts"] == 1
    conflict = scene.db.conflicts(DESKTOP)[0]
    assert conflict["instance"] == PSU
    assert payload_labels(conflict["new_rle"]) == [
        {"field": "visibility", "old": "visible", "new": "occluded_partial"}
    ]
    assert scene.row(1, PSU)["visibility"] == "visible"  # frozen, not overwritten
    assert scene.db.frame_digest(scene.key(1)) is None  # and not claimed as done


def test_accepting_a_label_conflict_writes_the_new_label(scene: Scene):
    conflict = _verified_with_new_visibility(scene)

    scene.svc.resolve_conflict(conflict["id"], "accept_new", "lin")

    row = scene.row(1, PSU)
    assert row["visibility"] == "occluded_partial"
    assert row["status"] == "verified"
    assert scene.db.conflicts(DESKTOP) == []


def test_keeping_the_old_label_pins_it_on_this_frame(scene: Scene):
    conflict = _verified_with_new_visibility(scene)

    scene.svc.resolve_conflict(conflict["id"], "keep_old", "lin")

    override = scene.db.frame_overrides(scene.key(1))[PSU]
    assert override.visibility == "visible"
    scene.svc.refresh(scene.key(1))
    assert scene.row(1, PSU)["visibility"] == "visible"
    assert scene.db.conflicts(DESKTOP, open_only=True) == []


def test_removing_a_visibility_override_is_a_disagreement_too(scene: Scene):
    """Ctrl+Z on the 1-7 shortcut, on a frame somebody had already confirmed.

    The row then carries a label neither a human nor its own pixels stand
    behind -- and COCO writes that label beside a segmentation that contradicts
    it. The refresh saw no override to compare against, called it skipped and
    stamped the digest, so no pass ever looked again.
    """
    scene.refresh_all()
    scene.db.set_frame_override(
        FrameOverride(scene.key(1), PSU, None, "occluded_full")
    )
    scene.svc.refresh(scene.key(1))
    scene.svc.verify_frame(scene.key(1), "lin")
    assert scene.row(1, PSU)["visibility"] == "occluded_full"

    scene.db.delete_frame_override(scene.key(1), PSU)
    out = scene.svc.refresh(scene.key(1))

    assert out["conflicts"] == 1
    conflict = scene.db.conflicts(DESKTOP)[0]
    assert payload_labels(conflict["new_rle"]) == [
        {"field": "visibility", "old": "occluded_full", "new": "visible"}
    ]
    assert scene.row(1, PSU)["visibility"] == "occluded_full"  # still frozen
    assert scene.db.frame_digest(scene.key(1)) is None

    scene.svc.resolve_conflict(conflict["id"], "accept_new", "lin")
    assert scene.row(1, PSU)["visibility"] == "visible"  # what the pixels say


def test_keeping_a_removed_override_pins_it_again(scene: Scene):
    scene.refresh_all()
    scene.db.set_frame_override(
        FrameOverride(scene.key(1), PSU, None, "occluded_full")
    )
    scene.svc.refresh(scene.key(1))
    scene.svc.verify_frame(scene.key(1), "lin")
    scene.db.delete_frame_override(scene.key(1), PSU)
    scene.svc.refresh(scene.key(1))
    cid = scene.db.conflicts(DESKTOP)[0]["id"]

    scene.svc.resolve_conflict(cid, "keep_old", "lin")

    assert scene.db.frame_overrides(scene.key(1))[PSU].visibility == "occluded_full"
    scene.svc.refresh(scene.key(1))
    assert scene.row(1, PSU)["visibility"] == "occluded_full"
    assert scene.db.conflicts(DESKTOP, open_only=True) == []


def test_a_frozen_row_the_pixels_agree_with_is_not_re_examined(scene: Scene):
    """The "was it forced" test must not fire on an ordinary derived label."""
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(1), "lin")

    assert scene.svc.refresh(scene.key(1), ignore_digest=True)["conflicts"] == 0
    assert scene.db.conflicts(DESKTOP) == []


def test_keeping_the_old_placement_is_refused(scene: Scene):
    """Where a part is, is the step table's decision, not an override's."""
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, SCREW, "placement", "in_chassis", "on_bench", auto=False)],
        auto_only=False,
    )
    scene.svc.refresh(scene.key(2))
    conflict = scene.db.conflicts(DESKTOP)[0]

    with pytest.raises(ValueError, match="state log"):
        scene.svc.resolve_conflict(conflict["id"], "keep_old", "lin")

    assert scene.row(2, SCREW)["placement"] == "in_chassis"
    assert scene.db.conflicts(DESKTOP, open_only=True)


def test_a_conflict_with_unchanged_labels_and_pixels_is_still_skipped(scene: Scene):
    """The cheap path has to stay cheap: nothing moved, nothing is queued."""
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(1), "lin")

    out = scene.svc.refresh(scene.key(1), ignore_digest=True)

    assert out["conflicts"] == 0 and out["updated"] == 0
    assert out["skipped"] == 2
    assert scene.db.conflicts(DESKTOP) == []


# --------------------------------------------------------------------------- #
# verify_frame and the frozen-truth invariant
# --------------------------------------------------------------------------- #
def test_verify_frame_refuses_while_a_conflict_of_that_frame_is_open(scene: Scene):
    """The exact reproduction: Space in the review queue wiped a frozen row.

    Verify the frame; the screw then leaves it, so the refresh queues one
    conflict and demotes the frame. The annotator meets it in the
    ``needs_review`` queue and presses Space: ``verify_frame`` recompiled, found
    the frozen row missing from the compilation, **deleted** it and marked the
    frame verified again. The disagreement stayed open for ever and a human's
    signature was gone.
    """
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen = scene.counts(2, SCREW)
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, SCREW, "placement", "in_chassis", "elsewhere", auto=False)],
        auto_only=False,
    )
    assert scene.svc.refresh(scene.key(2))["conflicts"] == 1
    assert scene.review_status(2) == "needs_review"
    cid = scene.db.conflicts(DESKTOP)[0]["id"]

    with pytest.raises(ValueError) as refused:
        scene.svc.verify_frame(scene.key(2), "lin")

    assert str(cid) in str(refused.value)
    assert scene.counts(2, SCREW) == frozen  # the frozen row is untouched
    assert scene.review_status(2) == "needs_review"
    assert [c["id"] for c in scene.db.conflicts(DESKTOP)] == [cid]

    # settling it is what makes the frame confirmable again
    scene.svc.resolve_conflict(cid, "accept_new", "lin")
    scene.svc.verify_frame(scene.key(2), "lin")
    assert scene.review_status(2) == "verified"
    assert scene.db.conflicts(DESKTOP, open_only=True) == []


def test_verify_frame_queues_a_vanished_frozen_row_instead_of_deleting_it(scene: Scene):
    """A confirmed instance the inputs no longer contain is a disagreement.

    ``verify_frame`` dropped such a row outright -- ``delete_compiled`` ignored
    ``status`` -- so the one thing the compiler may never overwrite was removed
    by the confirmation itself.
    """
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen = scene.counts(2, SCREW)
    assert scene.db.conflicts(DESKTOP) == []

    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, SCREW, "placement", "in_chassis", "elsewhere", auto=False)],
        auto_only=False,
    )

    with pytest.raises(ValueError) as refused:
        scene.svc.verify_frame(scene.key(2), "lin")

    assert SCREW in str(refused.value)
    queued = scene.db.conflicts(DESKTOP)
    assert [c["instance"] for c in queued] == [SCREW]
    assert queued[0]["new_rle"] is None
    assert scene.counts(2, SCREW) == frozen
    assert scene.row(2, SCREW)["status"] == "verified"


def test_verify_frame_refuses_a_frozen_row_the_inputs_have_moved_under(scene: Scene):
    """The re-check has not run yet, and Space must not do its job for it.

    A frozen row that is still *in* the frame but no longer agrees with it was
    the one disagreement the gate did not look for: ``_put_row`` wrote the new
    geometry over the human's signature and ``_stamp`` then turned the queued
    re-check into a no-op, so the conflict was never raised at all.
    """
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen = scene.counts(2, PSU)
    replace_parts(scene, scene.psu_kf,
                  [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 34)))])
    # the sweeper has not drained it yet: the truth table still holds the old row
    scene.svc.queue_rechecks(DESKTOP, VIEW, [2])
    assert scene.db.conflicts(DESKTOP) == []

    with pytest.raises(ValueError, match=PSU):
        scene.svc.verify_frame(scene.key(2), "lin")

    assert scene.counts(2, PSU) == frozen
    assert scene.row(2, PSU)["status"] == "verified"
    assert [c["instance"] for c in scene.db.conflicts(DESKTOP)] == [PSU]
    assert scene.db.frame_digest(scene.key(2)) is None


def test_verify_frame_refuses_a_frozen_row_whose_label_moved(scene: Scene):
    """The same gate, for the half of the truth that is not pixels."""
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(1), "lin")
    scene.db.set_frame_override(
        FrameOverride(scene.key(1), PSU, None, "occluded_partial")
    )

    with pytest.raises(ValueError, match="visibility"):
        scene.svc.verify_frame(scene.key(1), "lin")

    assert scene.row(1, PSU)["visibility"] == "visible"
    assert [c["instance"] for c in scene.db.conflicts(DESKTOP)] == [PSU]


def test_verify_frame_leaves_an_agreeing_frozen_row_exactly_as_it_is(scene: Scene):
    """A re-trace within tolerance is the same annotation, so nothing is written.

    ``refresh`` skips such a row; the confirmation rewrote it -- new pixels, a
    new ``verified_by``/``verified_at`` -- so pressing Space a second time moved
    a signature the first Space had already made, and the stored geometry drifted
    one pixel per confirmation away from what the human actually confirmed.
    """
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen = {inst: dict(row) for inst, row in scene.rows(2).items()}

    # one pixel row off the PSU: inside the re-tracing tolerance of spec 3.4
    replace_parts(scene, scene.psu_kf,
                  [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 49)))])
    assert scene.svc.refresh(scene.key(2))["conflicts"] == 0

    scene.svc.verify_frame(scene.key(2), "bo")

    assert scene.rows(2) == frozen  # byte for byte, "lin" included
    assert scene.db.conflicts(DESKTOP) == []
    assert scene.review_status(2) == "verified"
    assert scene.db.frame_digest(scene.key(2)) is not None


def test_confirming_an_unchanged_frame_again_writes_nothing(scene: Scene, monkeypatch):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen = {inst: dict(row) for inst, row in scene.rows(2).items()}
    written: list = []
    monkeypatch.setattr(scene.db, "put_compiled",
                        lambda *a, **k: written.append(a))

    scene.svc.verify_frame(scene.key(2), "bo")

    assert written == []
    assert scene.rows(2) == frozen
    assert scene.db.frame_digest(scene.key(2)) is not None


def test_verify_frame_still_writes_the_auto_rows_and_the_new_instances(scene: Scene):
    """Only a *frozen* row is protected; everything else is the compiler's."""
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    scene.db.upsert_instance(InstanceRec(key=FAN, desktop=DESKTOP, cls="case_fan"))
    scene.db.add_keyframe(mask_kf(FAN, FAN_RECT, anchor=3))
    scene.svc.refresh(scene.key(2))
    assert scene.row(2, FAN)["status"] == "auto"

    scene.svc.verify_frame(scene.key(2), "bo")

    assert scene.row(2, FAN)["status"] == "verified"
    assert scene.row(2, FAN)["verified_by"] == "bo"
    assert scene.row(2, PSU)["verified_by"] == "lin"  # the first confirmation stands


def test_verify_frame_still_drops_an_auto_row_the_inputs_lost(scene: Scene):
    """Only a frozen row is a signature; an ``auto`` row is a cache."""
    scene.refresh_all()
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, SCREW, "placement", "in_chassis", "elsewhere", auto=False)],
        auto_only=False,
    )
    assert SCREW in scene.rows(2)  # the stale auto row is still there

    scene.svc.verify_frame(scene.key(2), "lin")

    assert sorted(scene.rows(2)) == [PSU]
    assert scene.db.conflicts(DESKTOP) == []
    assert scene.review_status(2) == "verified"


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


# --------------------------------------------------------------------------- #
# what a superseding message may claim
# --------------------------------------------------------------------------- #
def test_a_superseded_conflict_says_the_disagreement_was_queued_again(scene: Scene):
    cid = _conflicting_scene(scene)
    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 26)))])
    with pytest.raises(StaleConflictError) as err:
        scene.svc.resolve_conflict(cid, "accept_new", "lin")
    assert "was queued again" in str(err.value)
    assert len(scene.db.conflicts(DESKTOP)) == 1


def test_a_disagreement_the_dedup_suppressed_says_it_is_already_queued(scene: Scene):
    """The same disagreement is open twice over: nothing new was written."""
    cid = _conflicting_scene(scene)
    replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 26)))])
    scene.svc.refresh(scene.key(2))  # queues the new disagreement by itself
    open_before = len(scene.db.conflicts(DESKTOP, open_only=True))

    with pytest.raises(StaleConflictError) as err:
        scene.svc.resolve_conflict(cid, "accept_new", "lin")
    assert "is already queued" in str(err.value)
    assert "was queued again" not in str(err.value)
    # ... and the message is true: no second copy was inserted
    assert len(scene.db.conflicts(DESKTOP, open_only=True)) == open_before - 1
