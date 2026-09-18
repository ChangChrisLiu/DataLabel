"""Tests for the persisted truth table (tda.core.truth / tda.core.truth_inputs).

One synthetic scene backs almost every test: desktop 1, view ``scan``, a 64x64
image, three logical steps and two instances -- a PSU and the screw that
fastens it, which is unscrewed and taken out at step 3 (so it lies on the bench
there, tracked by a box). Shapes are rectangles encoded as COCO RLE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.model import (
    ActionRec,
    FrameKey,
    FrameOverride,
    InstanceRec,
    OccluderMask,
    ShapeKeyframe,
    ShapePart,
    Similarity,
    StateEvent,
    StepRec,
    ZOrderRec,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import VIEW_HW, frame_hw

DESKTOP = 1
VIEW = "scan"
HW = (64, 64)
PSU = "psu.01"
SCREW = "screw.psu.01"
FAN = "case_fan.01"

PSU_RECT = (10, 10, 50, 50)
SCREW_RECT = (14, 14, 22, 22)
FAN_RECT = (52, 52, 62, 62)
BENCH_BOX = (2.0, 2.0, 12.0, 12.0)


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """A filled rectangle ``[x0, x1) x [y0, y1)`` as a 64x64 bool mask."""
    m = np.zeros(HW, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def mask_kf(instance: str, box, anchor: int, placement: str = "in_chassis") -> ShapeKeyframe:
    return ShapeKeyframe(
        id=None,
        instance=instance,
        desktop=DESKTOP,
        view=VIEW,
        pose_segment=1,
        anchor_step=anchor,
        placement=placement,
        geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(*box)))],
    )


def box_kf(instance: str, box, anchor: int, placement: str = "on_bench") -> ShapeKeyframe:
    return ShapeKeyframe(
        id=None,
        instance=instance,
        desktop=DESKTOP,
        view=VIEW,
        pose_segment=1,
        anchor_step=anchor,
        placement=placement,
        geom_type="box",
        parts=[ShapePart("main", None, tuple(float(v) for v in box))],
    )


@dataclass
class Scene:
    """The seeded database plus the service and the keyframes under test."""

    db: Db
    tax: Taxonomy
    svc: TruthService
    psu_kf: ShapeKeyframe
    screw_kf: ShapeKeyframe
    bench_kf: ShapeKeyframe

    def key(self, step: int) -> FrameKey:
        return FrameKey(DESKTOP, step, VIEW)

    def refresh_all(self) -> dict:
        return self.svc.refresh_range(DESKTOP, VIEW, [1, 2, 3])

    def rows(self, step: int) -> dict[str, dict]:
        return self.db.compiled(self.key(step))

    def row(self, step: int, instance: str) -> dict:
        return self.rows(step)[instance]

    def counts(self, step: int, instance: str) -> str | None:
        rle = self.row(step, instance)["visible_rle"]
        return None if rle is None else rle["counts"]

    def review_status(self, step: int) -> str | None:
        return self.db.get_frame(self.key(step))["review_status"]

    def add_frame(self, step: int, hw=HW) -> None:
        aux = {} if hw is None else {"hw": [int(hw[0]), int(hw[1])]}
        self.db.upsert_frame(self.key(step), f"s{step:03d}.jpg", aux, None)


@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    yield d
    d.close()


@pytest.fixture
def scene(db: Db) -> Scene:
    tax = load_taxonomy()
    db.upsert_instance(InstanceRec(key=PSU, desktop=DESKTOP, cls="psu"))
    db.upsert_instance(
        InstanceRec(
            key=SCREW,
            desktop=DESKTOP,
            cls="screw",
            attrs={"role": "psu", "head": "PH2", "captive": False},
            parent=PSU,
            fastens=PSU,
        )
    )
    db.replace_steps(
        DESKTOP,
        [
            StepRec(DESKTOP, 1, "initial", "initial state"),
            StepRec(DESKTOP, 2, "normal", "loosen psu screw"),
            StepRec(DESKTOP, 3, "normal", "psu screw"),
        ],
        [ActionRec(DESKTOP, 3, 0, SCREW, "remove", tool="PH2")],
    )
    scene = Scene(
        db=db,
        tax=tax,
        svc=TruthService(db, tax),
        psu_kf=mask_kf(PSU, PSU_RECT, anchor=3),
        screw_kf=mask_kf(SCREW, SCREW_RECT, anchor=2),
        bench_kf=box_kf(SCREW, BENCH_BOX, anchor=3),
    )
    for step in (1, 2, 3):
        scene.add_frame(step)
    for kf in (scene.psu_kf, scene.screw_kf, scene.bench_kf):
        db.add_keyframe(kf)
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(PSU, "main"), (SCREW, "main")]))
    return scene


# --------------------------------------------------------------------------- #
# compile: input gathering
# --------------------------------------------------------------------------- #
def test_compile_gathers_state_keyframes_and_zorder(scene: Scene):
    out = scene.svc.compile(scene.key(1))

    assert out.problems == []
    assert sorted(out.instances) == [PSU, SCREW]
    psu, screw = out.instances[PSU], out.instances[SCREW]
    assert psu.placement == "in_chassis" and screw.placement == "in_chassis"
    # the screw sits on top of the PSU, so the PSU keeps a screw-shaped hole
    assert masks.area(psu.visible) == 40 * 40 - 8 * 8
    assert psu.occlusion_ratio == pytest.approx(64 / 1600)
    assert not (psu.visible & screw.visible).any()
    assert out.input_hash


def test_compile_follows_the_state_machine_onto_the_bench(scene: Scene):
    out = scene.svc.compile(scene.key(3))

    assert out.problems == []
    screw = out.instances[SCREW]
    assert screw.placement == "on_bench"
    assert screw.visible is None  # box geometry carries no mask
    assert screw.box == pytest.approx(BENCH_BOX)


def test_compile_uses_stored_events_when_the_db_has_them(scene: Scene):
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, PSU, "placement", "in_chassis", "on_bench", auto=False)],
        auto_only=False,
    )

    out = scene.svc.compile(scene.key(2))

    assert out.instances[PSU].placement == "on_bench"
    # the stored log replaces the derived one, so the screw never left step 3
    assert out.instances[SCREW].placement == "in_chassis"
    assert f"bench_missing:{PSU}" in out.problems  # no bench chain for the PSU


def test_compile_gathers_occluders_overrides_and_the_transform(scene: Scene):
    key = scene.key(1)
    scene.db.set_occluder(OccluderMask(key, "hand", masks.encode_rle(rect(0, 10, 64, 30))))
    scene.db.set_frame_override(FrameOverride(key, SCREW, visibility="occluded_full"))
    scene.db.set_transform(key, Similarity(tx=2.0))

    out = scene.svc.compile(key)

    psu = out.instances[PSU]
    assert masks.bbox(psu.visible) == (12, 30, 52, 50)  # shifted right, top cut away
    assert out.instances[SCREW].visibility == "occluded_full"


def test_compile_defaults_to_pose_segment_one_and_honours_a_stored_one(scene: Scene):
    assert scene.svc.compile(scene.key(1)).problems == []  # no pose row: segment 1

    scene.db.set_pose_segment(DESKTOP, VIEW, 2, 1, 3, 1, None, None)
    scene.db.set_frame_flags(scene.key(1), pose_segment=2)

    out = scene.svc.compile(scene.key(1))
    assert f"missing_shape:{PSU}" in out.problems  # the shapes belong to segment 1


def test_frame_hw_is_inferred_from_the_view_and_stored_back(scene: Scene):
    key = FrameKey(DESKTOP, 1, "rs")
    scene.db.upsert_frame(key, "s001.png", {}, None)

    assert frame_hw(scene.db, key) == VIEW_HW["rs"] == (720, 1280)
    assert scene.db.get_frame(key)["aux"]["hw"] == [720, 1280]
    assert frame_hw(scene.db, key) == (720, 1280)  # read back, not inferred again
    assert frame_hw(scene.db, FrameKey(DESKTOP, 1, "oak1")) == (3040, 4032)
    assert frame_hw(scene.db, scene.key(1)) == HW  # the scene stores its own


# --------------------------------------------------------------------------- #
# refresh
# --------------------------------------------------------------------------- #
def test_refresh_writes_one_auto_row_per_instance_and_step(scene: Scene):
    out = scene.refresh_all()

    assert out == {"updated": 6, "conflicts": 0, "skipped": 0, "problems": []}
    for step in (1, 2, 3):
        rows = scene.rows(step)
        assert sorted(rows) == [PSU, SCREW]
        for row in rows.values():
            assert row["status"] == "auto"
            assert row["verified_by"] is None and row["verified_at"] is None
            assert row["input_hash"]
    psu = scene.row(1, PSU)
    assert masks.area(masks.decode_rle(psu["visible_rle"])) == 40 * 40 - 8 * 8
    assert psu["occlusion_ratio"] == pytest.approx(64 / 1600)
    assert psu["visibility"] == "visible" and psu["placement"] == "in_chassis"
    # a bench box is stored as its rectangle, so the truth table stays readable
    bench = scene.row(3, SCREW)
    assert bench["placement"] == "on_bench"
    assert masks.bbox(masks.decode_rle(bench["visible_rle"])) == (2, 2, 12, 12)


def test_refresh_skips_every_row_when_the_input_hash_is_unchanged(scene: Scene, monkeypatch):
    scene.refresh_all()
    calls: list = []
    monkeypatch.setattr(scene.db, "put_compiled", lambda *a, **k: calls.append(a))

    out = scene.svc.refresh(scene.key(1))

    assert out == {"updated": 0, "conflicts": 0, "skipped": 2, "problems": []}
    assert calls == []


def test_refresh_reports_problems_without_refusing_to_write(scene: Scene):
    scene.add_frame(4)  # beyond every anchor: no shape applies any more

    out = scene.svc.refresh(FrameKey(DESKTOP, 4, VIEW))

    assert f"missing_shape:{PSU}" in out["problems"]
    assert f"bench_missing:{SCREW}" in out["problems"]
    assert out["updated"] == 2
    assert scene.row(4, PSU)["visible_rle"] is None
    assert scene.row(4, PSU)["visibility"] == "out_of_view"


def test_refresh_clears_bench_annotated_when_a_bench_shape_is_missing(scene: Scene):
    scene.svc.refresh(scene.key(3))
    assert scene.db.get_frame(scene.key(3))["bench_annotated"] is True

    scene.db.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (scene.bench_kf.id,))
    scene.db.conn.commit()
    out = scene.svc.refresh(scene.key(3))

    assert out["problems"] == [f"bench_missing:{SCREW}"]
    assert scene.db.get_frame(scene.key(3))["bench_annotated"] is False


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def test_verify_frame_freezes_every_row_and_marks_the_frame(scene: Scene):
    scene.refresh_all()

    scene.svc.verify_frame(scene.key(2), "lin")

    for row in scene.rows(2).values():
        assert row["status"] == "verified"
        assert row["verified_by"] == "lin"
        assert row["verified_at"]
    assert scene.review_status(2) == "verified"
    assert scene.review_status(1) == "unlabeled"
    kinds = [op["kind"] for op in scene.db.ops(DESKTOP, VIEW)]
    assert "verify_frame" in kinds


def test_verify_frame_refuses_a_blocking_problem(scene: Scene):
    scene.add_frame(4)
    key = FrameKey(DESKTOP, 4, VIEW)

    with pytest.raises(ValueError) as err:
        scene.svc.verify_frame(key, "lin")

    assert f"missing_shape:{PSU}" in str(err.value)
    assert scene.db.get_frame(key)["review_status"] != "verified"
    assert scene.db.compiled(key) == {}


def test_verify_frame_accepts_a_bench_warning(scene: Scene):
    scene.db.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (scene.bench_kf.id,))
    scene.db.conn.commit()

    scene.svc.verify_frame(scene.key(3), "lin")

    assert scene.review_status(3) == "verified"
    assert scene.row(3, SCREW)["status"] == "verified"


def test_demote_frame_marks_needs_review_and_logs_the_reason(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")

    scene.svc.demote_frame(scene.key(2), "z-order changed")

    assert scene.review_status(2) == "needs_review"
    op = scene.db.ops(DESKTOP, VIEW)[0]
    assert op["kind"] == "demote_frame"
    assert op["payload"]["reason"] == "z-order changed"
    assert op["payload"]["step"] == 2


# --------------------------------------------------------------------------- #
# re-compilation against frozen rows
# --------------------------------------------------------------------------- #
def _replace_parts(scene: Scene, kf: ShapeKeyframe, parts: list[ShapePart]) -> None:
    kf.parts = parts
    scene.db.update_keyframe(kf)


def test_shifting_a_shape_updates_auto_rows_and_leaves_the_verified_one(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    frozen = {inst: dict(row) for inst, row in scene.rows(2).items()}

    _replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(11, 10, 51, 50)))])
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
    _replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 34)))])
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
    _replace_parts(scene, scene.psu_kf, [ShapePart("main", masks.encode_rle(rect(10, 10, 50, 34)))])
    scene.svc.refresh(scene.key(2))

    out = scene.svc.refresh(scene.key(2))

    assert out["conflicts"] == 1  # still in disagreement
    assert len(scene.db.conflicts(DESKTOP)) == 1  # but not queued twice


def test_a_moved_bench_box_conflicts_only_beyond_two_pixels(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(3), "lin")
    frozen_counts = scene.counts(3, SCREW)

    _replace_parts(scene, scene.bench_kf, [ShapePart("main", None, (3.0, 2.0, 13.0, 12.0))])
    assert scene.svc.refresh(scene.key(3))["conflicts"] == 0

    _replace_parts(scene, scene.bench_kf, [ShapePart("main", None, (7.0, 2.0, 17.0, 12.0))])
    out = scene.svc.refresh(scene.key(3))

    assert out["conflicts"] == 1
    conflict = scene.db.conflicts(DESKTOP)[-1]
    assert conflict["instance"] == SCREW
    assert masks.bbox(masks.decode_rle(conflict["new_rle"])) == (7, 2, 17, 12)
    assert conflict["sym_diff_px"] == 2 * 5 * 10  # two 5x10 slivers
    assert scene.counts(3, SCREW) == frozen_counts


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


def test_an_instance_dropped_from_an_auto_frame_deletes_its_row(scene: Scene):
    scene.refresh_all()
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, SCREW, "placement", "in_chassis", "elsewhere", auto=False)],
        auto_only=False,
    )

    out = scene.svc.refresh(scene.key(3))

    assert sorted(scene.rows(3)) == [PSU]
    assert out["conflicts"] == 0
    assert out["updated"] == 2  # the PSU row rewritten, the screw row deleted
    assert scene.db.conflicts(DESKTOP) == []


# --------------------------------------------------------------------------- #
# ranges and keyframe reach
# --------------------------------------------------------------------------- #
def test_refresh_range_aggregates_every_step(scene: Scene):
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(2), "lin")
    scene.db.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (scene.bench_kf.id,))
    scene.db.conn.commit()

    out = scene.refresh_all()

    assert out["problems"] == [f"bench_missing:{SCREW}"]  # step 3 only
    assert out["updated"] == 2  # both rows of step 3
    assert out["skipped"] == 4  # steps 1 (unchanged) and 2 (frozen)
    assert out["conflicts"] == 0


def test_affected_steps_covers_the_reach_of_one_keyframe(scene: Scene):
    assert scene.svc.affected_steps(DESKTOP, VIEW, PSU, scene.psu_kf) == [1, 2, 3]
    # the screw is on the bench at step 3, so its chassis chain stops at step 2
    assert scene.svc.affected_steps(DESKTOP, VIEW, SCREW, scene.screw_kf) == [1, 2]
    assert scene.svc.affected_steps(DESKTOP, VIEW, SCREW, scene.bench_kf) == [3]


def test_affected_steps_splits_a_two_keyframe_chain(scene: Scene):
    early = mask_kf(PSU, (12, 12, 20, 20), anchor=1)
    scene.db.add_keyframe(early)

    assert scene.svc.affected_steps(DESKTOP, VIEW, PSU, early) == [1]
    assert scene.svc.affected_steps(DESKTOP, VIEW, PSU, scene.psu_kf) == [2, 3]


def test_affected_steps_is_limited_to_the_pose_segment(scene: Scene):
    # the chassis was turned over before step 3: its shapes are drawn anew there
    scene.db.set_pose_segment(DESKTOP, VIEW, 1, 1, 2, 2, None, None)
    scene.db.set_pose_segment(DESKTOP, VIEW, 2, 3, 3, 3, None, None)

    assert scene.svc.affected_steps(DESKTOP, VIEW, PSU, scene.psu_kf) == [1, 2]
