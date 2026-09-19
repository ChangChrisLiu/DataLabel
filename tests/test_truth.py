"""Tests for the persisted truth table: compiling, refreshing and verifying.

The frozen-row comparison, the conflict queue and its resolutions live in
``test_truth_conflicts.py``; the scene both modules use is ``truth_scenes.py``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.model import FrameKey, FrameOverride, OccluderMask, Similarity, StateEvent
from tda.core.truth_inputs import VIEW_HW, frame_hw, state_of
from truth_scenes import (
    BENCH_BOX,
    DESKTOP,
    HW,
    PSU,
    SCREW,
    VIEW,
    Scene,
    build_scene,
    mask_kf,
    rect,
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


def test_manual_events_are_merged_on_top_of_the_derived_ones(scene: Scene):
    # a hand-written note: the PSU was pushed aside at step 1
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 1, PSU, "state", "installed", "displaced", auto=False)],
        auto_only=False,
    )

    state = state_of(scene.db, scene.tax, DESKTOP, 3)

    assert state[PSU].state == "displaced"  # the manual event survives
    assert state[SCREW].state == "removed"  # and the recorded action still applies
    out = scene.svc.compile(scene.key(3))
    assert out.instances[PSU].placement == "in_chassis"
    assert out.instances[SCREW].placement == "on_bench"


def test_a_manual_removal_still_cascades_to_attached_children(scene: Scene):
    """The screw follows the PSU out, and is then asked about no further.

    It left *inside* its parent, so it is neither in the chassis nor a thing of
    its own in the staging area: the snapshot still says ``removed/on_bench``,
    but no geometry is wanted and no row is compiled (user decision C7).
    """
    scene.db.replace_events(
        DESKTOP,
        [StateEvent(DESKTOP, 2, PSU, "state", "installed", "removed", auto=False)],
        auto_only=False,
    )

    state = state_of(scene.db, scene.tax, DESKTOP, 2)
    out = scene.svc.compile(scene.key(2))

    assert (state[SCREW].state, state[SCREW].placement) == ("removed", "on_bench")
    assert PSU not in out.instances  # removed: it needs no geometry in the chassis
    assert SCREW not in out.instances  # and neither does what went out inside it
    assert not [p for p in out.problems if SCREW in p]


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
    aux = scene.db.get_frame(key)["aux"]
    assert aux["hw"] == [720, 1280] and aux["hw_source"] == "inferred"
    assert frame_hw(scene.db, key) == (720, 1280)  # read back, not inferred again
    assert frame_hw(scene.db, FrameKey(DESKTOP, 1, "oak1")) == (3040, 4032)
    assert frame_hw(scene.db, scene.key(1)) == HW  # the scene stores its own


def test_frame_hw_measures_the_cached_image_and_supersedes_a_guess(scene: Scene, tmp_path: Path):
    key = FrameKey(DESKTOP, 2, "rs")
    image = tmp_path / "s002.png"
    scene.db.upsert_frame(key, str(image), {}, None)

    assert frame_hw(scene.db, key) == (720, 1280)  # no file yet: the view's nominal size
    assert scene.db.get_frame(key)["aux"]["hw_source"] == "inferred"

    cv2.imwrite(str(image), np.zeros((12, 34, 3), np.uint8))
    assert frame_hw(scene.db, key) == (12, 34)  # measured beats inferred
    aux = scene.db.get_frame(key)["aux"]
    assert aux["hw"] == [12, 34] and aux["hw_source"] == "measured"

    image.unlink()
    assert frame_hw(scene.db, key) == (12, 34)  # measured once, then trusted


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
    assert psu["geom_type"] == "mask" and psu["box"] is None
    assert masks.area(masks.decode_rle(psu["visible_rle"])) == 40 * 40 - 8 * 8
    assert psu["occlusion_ratio"] == pytest.approx(64 / 1600)
    assert psu["visibility"] == "visible" and psu["placement"] == "in_chassis"
    # a bench part is a box row: its rectangle, no mask
    bench = scene.row(3, SCREW)
    assert bench["placement"] == "on_bench"
    assert bench["geom_type"] == "box"
    assert bench["visible_rle"] is None
    assert bench["box"] == list(BENCH_BOX)


def test_refresh_skips_every_row_when_the_input_hash_is_unchanged(scene: Scene, monkeypatch):
    scene.refresh_all()
    calls: list = []
    monkeypatch.setattr(scene.db, "put_compiled", lambda *a, **k: calls.append(a))

    out = scene.svc.refresh(scene.key(1))

    assert {k: out[k] for k in ("updated", "conflicts", "skipped", "problems")} == {
        "updated": 0, "conflicts": 0, "skipped": 2, "problems": [],
    }
    # a frame whose digest still matches is not compiled at all any more
    assert out["compiled"] is None
    assert scene.svc.refresh(scene.key(1), want_compiled=True)["compiled"].key == scene.key(1)
    assert calls == []


# --------------------------------------------------------------------------- #
# where the pixels are
# --------------------------------------------------------------------------- #
def test_frame_hw_reads_the_local_cache_and_never_the_source_drive(
    scene: Scene, tmp_path: Path, monkeypatch
):
    """The first compile of a view used to decode 12 MP images off F: (spec 2.4).

    ``_image_path`` looked for ``aux["cache_path"]``, which nothing writes, so
    it fell through to the frame's own path -- the read-only source drive, one
    full-resolution read per frame, and the nominal size when F: is detached.
    """
    from tda.core.cache import cached_image_path

    key = FrameKey(DESKTOP, 9, VIEW)
    scene.db.upsert_frame(key, "F:/raw/scan/D01/s009.png", {}, None)
    cached = Path(cached_image_path(str(tmp_path), key))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cached), np.zeros((12, 34, 3), np.uint8))

    read: list[str] = []
    real = cv2.imread
    monkeypatch.setattr(cv2, "imread",
                        lambda path, *a, **k: (read.append(str(path)), real(path, *a, **k))[1])

    assert frame_hw(scene.db, key, cache_dir=str(tmp_path)) == (12, 34)

    assert [Path(p) for p in read] == [cached]
    assert not [p for p in read if p.upper().startswith("F:")]
    assert scene.db.get_frame(key)["aux"]["hw_source"] == "measured"


def test_a_truth_service_carries_the_cache_directory_into_every_compile(
    scene: Scene, tmp_path: Path
):
    from tda.core.cache import cached_image_path
    from tda.core.truth import TruthService

    key = FrameKey(DESKTOP, 9, VIEW)
    scene.db.upsert_frame(key, "F:/raw/scan/D01/s009.png", {}, None)
    cached = Path(cached_image_path(str(tmp_path), key))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cached), np.zeros((12, 34, 3), np.uint8))

    TruthService(scene.db, scene.tax, cache_dir=str(tmp_path)).compile(key)

    assert tuple(scene.db.get_frame(key)["aux"]["hw"]) == (12, 34)


def test_a_temporary_database_never_borrows_the_configured_cache(tmp_path: Path):
    """A cache belongs to the annotations it was built for, and to no other.

    The default is resolved through the database's own path, so a copy under
    ``.cache/tmp`` -- or a test's temp file -- gets ``None`` rather than the
    machine's real cache, whose D01 images would be measured into it.
    """
    from tda.core.cache import configured_cache_dir

    assert configured_cache_dir(str(tmp_path / "scratch.sqlite")) is None
    assert configured_cache_dir(None) is None


def test_affected_steps_asks_the_same_need_set_the_compiler_does(scene: Scene):
    """A view with no staging area is not asked for bench boxes -- everywhere.

    ``affected_steps`` drives the canvas' "affects N frames" strip; it used to
    count frames the compiler does not even put the instance in.
    """
    assert scene.svc.affected_steps(DESKTOP, VIEW, SCREW, scene.bench_kf) == [3]

    scene.db.set_pose_segment_bench_roi(DESKTOP, VIEW, 1, None)

    assert scene.svc.affected_steps(DESKTOP, VIEW, SCREW, scene.bench_kf) == []
    assert scene.svc.affected_steps(DESKTOP, VIEW, PSU, scene.psu_kf) == [1, 2, 3]


def test_an_ignore_step_is_never_compiled_or_stamped(scene: Scene):
    """Spec 6.6: an ``ignore`` step describes no moment of the teardown.

    The exports already skipped it; the truth table did not, so it was compiled,
    stamped and confirmable -- and then dropped on the way out.
    """
    from tda.core.truth_inputs import annotatable_steps

    steps = scene.db.steps(DESKTOP)
    steps[1].step_type = "ignore"
    scene.db.replace_steps(DESKTOP, steps, scene.db.actions(DESKTOP))

    assert annotatable_steps(scene.db, DESKTOP, VIEW, [1, 2, 3]) == [1, 3]

    scene.svc.ensure_fresh(DESKTOP, VIEW)

    assert scene.rows(2) == {}
    assert scene.db.frame_digest(scene.key(2)) is None
    assert scene.rows(1) and scene.rows(3)


def test_input_digest_ignores_whether_an_rle_carries_str_or_bytes_counts(scene: Scene):
    """pycocotools hands out ``bytes`` counts; a digest that saw them never matched."""
    from tda.core.truth_fresh import digest_of
    from tda.core.truth_inputs import gather

    key = scene.key(1)
    inputs = gather(scene.db, scene.tax, key)
    rle = masks.encode_rle(rect(0, 0, 8, 8))
    raw = {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")}

    def digest(payload) -> str:
        inputs.occluders = [OccluderMask(key, "hand", dict(payload))]
        inputs.frame_overrides = {PSU: FrameOverride(key, PSU, dict(payload), "visible")}
        return digest_of(inputs, "1")

    assert digest(raw) == digest(rle)


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


def test_verify_frame_writes_all_or_nothing(scene: Scene, monkeypatch):
    scene.refresh_all()

    def boom(*args, **kwargs):
        raise RuntimeError("the op log is down")

    monkeypatch.setattr(scene.db, "log_op", boom)
    with pytest.raises(RuntimeError):
        scene.svc.verify_frame(scene.key(2), "lin")

    assert [row["status"] for row in scene.rows(2).values()] == ["auto", "auto"]
    assert all(row["verified_by"] is None for row in scene.rows(2).values())
    assert scene.review_status(2) != "verified"


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
