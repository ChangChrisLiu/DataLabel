"""Offscreen tests for :class:`tda.ui.session.AnnotationSession` (spec 4.2-4.4).

The scene is the real D13 step sheet (``tests/fixtures/logs/desktop_13.csv``)
imported with :func:`tda.core.logs.import_log` and truncated to its first
:data:`LAST_STEP` logical steps, plus one synthetic 64x64 frame per step written
into a temporary cache directory.  The database is therefore exactly what S0/S1
would leave behind, and the session is driven through the public
:class:`tda.ui.session_api.SessionLike` surface only.

One thing the importer cannot know is filled in the way the S1 step-table review
would (spec 4.1): the four captive CPU-cooler screws are marked ``attached`` to
the cooler, so that removing the cooler at step 13 takes them out with it.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tda.core import masks
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.db import Db
from tda.core.logs import import_log, read_desktop_csv
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.states import needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import instances_of, state_of
from tda.ui import session_api as api
from tda.ui.commands import edit_editing_mask_op
from tda.ui.session import AnnotationSession
from tda.ui.session_edit import suggest_scope

FIXTURES = Path(__file__).parent / "fixtures" / "logs"

DESKTOP = 13
VIEW = "scan"
HW = (64, 64)
#: The sheet has 42 steps; the first 14 carry every transition the tests need
#: (cover, four captive screws, SSD, cage, fan connector, the cooler itself)
#: and keep a full-scene recompile to a fraction of a second.
LAST_STEP = 14
COOLER = "cpu_cooler.fan.01"
SCREWS = tuple(f"screw.cpu_cooler.{i:02d}" for i in (1, 2, 3, 4))
CHASSIS = "chassis"
FAN_CONNECTOR = "connector.fan.01"


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """A filled ``[x0, x1) x [y0, y1)`` rectangle as a 64x64 bool mask."""
    mask = np.zeros(HW, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def cell(index: int) -> np.ndarray:
    """A small rectangle in cell ``index`` of an 8x8 grid over the frame.

    Distinct, non-overlapping shapes keep every instance visible, so seeded
    scenes produce no ``empty_visible`` noise.
    """
    col, row = index % 8, (index // 8) % 8
    x0, y0 = col * 8 + 1, row * 8 + 1
    return rect(x0, y0, x0 + 6, y0 + 6)


def _write_frames(db: Db, cache_dir: Path, steps, missing=()) -> None:
    """One synthetic 64x64 PNG per step, plus its ``frame`` row."""
    for step in steps:
        key = FrameKey(DESKTOP, step, VIEW)
        path = cache_path(str(cache_dir), key, VIEW_EXT[VIEW])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img = np.full((*HW, 3), 20 + step, dtype=np.uint8)
        img[8:24, 8:24] = 200
        cv2.imwrite(path, img)
        aux = {"cache_path": path, "hw": [HW[0], HW[1]]}
        flags = {"missing": True} if step in missing else None
        db.upsert_frame(key, path, aux, None, flags)


def seed_db(db: Db, tax, cache_dir: Path, last_step: int = LAST_STEP, missing=()) -> None:
    """Import the D13 sheet, truncate it, and write the frames."""
    rows, meta = read_desktop_csv(FIXTURES / f"desktop_{DESKTOP}.csv")
    imported = import_log(DESKTOP, rows, meta, tax)
    db.replace_steps(
        DESKTOP,
        [s for s in imported.steps if s.step <= last_step],
        [a for a in imported.actions if a.step <= last_step],
    )
    for key, rec in imported.instances.items():
        if key in SCREWS:  # what the S1 review records: captive, rides out with the cooler
            rec.parent = COOLER
            rec.attached = True
            rec.fastens = COOLER
        db.upsert_instance(rec)
    _write_frames(db, cache_dir, range(1, last_step + 1), missing=missing)


def make_session(tmp_path: Path, last_step: int = LAST_STEP, missing=()) -> AnnotationSession:
    """A session opened on the seeded D13 scanner view."""
    tax = load_taxonomy()
    db = Db(str(tmp_path / "tda.sqlite"))
    cache_dir = tmp_path / "cache"
    seed_db(db, tax, cache_dir, last_step=last_step, missing=missing)
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), "tester")
    session.open(DESKTOP, VIEW)
    return session


def chassis_instances(session: AnnotationSession, step: int) -> list[str]:
    """Instances that need a chassis mask at ``step``, in a stable order."""
    db, tax = session.db, session.tax
    insts = instances_of(db, DESKTOP)
    state = state_of(db, tax, DESKTOP, step)
    return sorted(
        key
        for key, kind in needs_geom(insts, state, tax).items()
        if kind == "mask" and state[key].placement == "in_chassis"
    )


def seed_shapes(session: AnnotationSession, step: int, skip=()) -> None:
    """Give every chassis instance of ``step`` its own rectangle, cheaply.

    Written straight to the database (one refresh at the end) rather than
    through ``commit_edit``, so a test that needs a fully drawn frame does not
    pay for one refresh sweep per instance.  The anchor is the last logical step
    of the scene, which selects the keyframe for every step where the instance
    is still in the chassis.
    """
    db = session.db
    keys = [k for k in chassis_instances(session, step) if k not in skip]
    order: list[tuple[str, str]] = []
    for index, key in enumerate(keys):
        db.add_keyframe(
            ShapeKeyframe(
                id=None,
                instance=key,
                desktop=DESKTOP,
                view=VIEW,
                pose_segment=1,
                anchor_step=LAST_STEP,
                placement="in_chassis",
                geom_type="mask",
                parts=[ShapePart("main", masks.encode_rle(cell(index)))],
            )
        )
        order.append((key, "main"))
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, order))
    session.refresh_all()


def draw(session: AnnotationSession, instance: str, mask: np.ndarray, scope: str) -> dict:
    """``begin_edit`` + paint + ``commit_edit`` in one call."""
    session.begin_edit(instance)
    session.set_editing_mask(mask)
    return session.commit_edit(scope)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    return make_session(tmp_path)


# --------------------------------------------------------------------------- #
# the protocol
# --------------------------------------------------------------------------- #
def test_session_satisfies_the_panel_protocol(session):
    assert isinstance(session, api.SessionLike)


# --------------------------------------------------------------------------- #
# open / navigation
# --------------------------------------------------------------------------- #
def test_open_starts_at_the_last_step(session):
    assert session.steps() == list(range(1, LAST_STEP + 1))
    assert session.current() == FrameKey(DESKTOP, LAST_STEP, VIEW)


def test_open_skips_missing_frames_but_keeps_the_logical_steps(qapp, tmp_path):
    session = make_session(tmp_path, missing=(LAST_STEP,))
    assert session.steps() == list(range(1, LAST_STEP + 1))  # the logical step stays
    assert session.current().step == LAST_STEP - 1  # ... but it is not annotated
    assert session.frame_status(LAST_STEP) == api.STATUS_MISSING


def test_prev_and_next_walk_the_available_steps(qapp, tmp_path):
    session = make_session(tmp_path, missing=(12,))
    session.goto(13)
    session.prev()
    assert session.current().step == 11  # 12 has no image
    session.next()
    assert session.current().step == 13
    session.goto(1)
    session.prev()
    assert session.current().step == 1  # nothing before the first step


def test_goto_emits_the_frame_changed_signal(session):
    seen = []
    session.sigFrameChanged.connect(seen.append)
    session.goto(10)
    assert seen == [FrameKey(DESKTOP, 10, VIEW)]


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def test_image_is_rgb_from_the_cache_and_is_reused(session):
    session.goto(10)
    img = session.image()
    assert img.shape == (*HW, 3)
    # cv2 wrote BGR with a constant colour, so R and B differ only by the fill
    assert img[0, 0, 0] == 30 and img[0, 0, 2] == 30
    assert session.image() is img  # the LRU hands back the very same array


def test_flash_compare_returns_the_previous_step(session):
    session.goto(10)
    assert np.array_equal(session.flash_compare(), session.image_at(9))


def test_image_cache_is_bounded(session):
    for step in range(1, LAST_STEP + 1):
        session.goto(step)
        session.image()
    assert len(session.image_cache) <= 8


def test_thumb_path_points_at_the_cached_image(session):
    path = session.thumb_path(7)
    assert path is not None and Path(path).exists()


# --------------------------------------------------------------------------- #
# task card (spec 4.2)
# --------------------------------------------------------------------------- #
def test_task_card_at_step_13_adds_the_cooler_and_its_captive_screws(session):
    session.goto(13)
    card = session.task_card()
    adds = {item["instance"] for item in card if item["kind"] == api.KIND_ADD_SHAPE}
    assert COOLER in adds
    assert set(SCREWS) <= adds
    assert all(item["done"] is False for item in card if item["kind"] == api.KIND_ADD_SHAPE)
    assert card[0]["instance"] == COOLER  # the parent leads, its children follow


def test_task_card_marks_an_item_done_once_the_shape_exists(session):
    session.goto(13)
    seed_shapes(session, 12)
    card = {item["instance"]: item for item in session.task_card()}
    assert card[COOLER]["done"] is True
    assert card[SCREWS[0]]["done"] is True


def test_task_card_splits_a_keyframe_for_a_latch_and_a_connector(session):
    session.goto(7)  # step 7 opens the SSD drive latch
    kinds = {item["instance"]: item["kind"] for item in session.task_card()}
    assert kinds["drive_latch.01"] == api.KIND_SPLIT_KEYFRAME

    session.goto(12)  # step 12 unplugs the CPU-fan connector
    kinds = {item["instance"]: item["kind"] for item in session.task_card()}
    assert kinds[FAN_CONNECTOR] == api.KIND_SPLIT_KEYFRAME


def test_task_card_is_state_only_for_an_unscrewed_screw(session):
    session.goto(3)  # step 3 unscrews the first CPU-cooler screw
    kinds = {item["instance"]: item["kind"] for item in session.task_card()}
    assert kinds[SCREWS[0]] == api.KIND_STATE_ONLY


def test_task_card_of_the_start_frame_lists_every_instance_needing_geometry(session):
    card = session.task_card()  # the session opened on the start frame
    listed = {item["instance"] for item in card}
    assert CHASSIS in listed
    assert COOLER in listed  # on the bench at step 14: it needs a box
    assert all(item["done"] is False for item in card)


def test_task_card_of_the_first_step_is_a_single_confirmation(session):
    session.goto(1)
    card = session.task_card()
    assert [item["kind"] for item in card] == [api.KIND_CONFIRM]


def test_task_card_says_the_bench_box_ends_here(session):
    session.commit_box(COOLER, (2.0, 2.0, 12.0, 12.0))  # the start frame draws the box
    session.goto(13)
    kinds = [item["kind"] for item in session.task_card() if item["instance"] == COOLER]
    assert api.KIND_REMOVE_BENCH_BOX in kinds


# --------------------------------------------------------------------------- #
# commit_edit (spec 3.3 / 4.3)
# --------------------------------------------------------------------------- #
def test_keyframe_commit_anchors_at_the_last_step_that_needs_geometry(session):
    session.goto(10)
    result = draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)

    kfs = session.db.keyframes(DESKTOP, VIEW, COOLER)
    assert len(kfs) == 1
    # the cooler is removed at step 13, so its chassis chain ends at step 12
    assert kfs[0].anchor_step == 12
    assert kfs[0].placement == "in_chassis"
    assert result["affected"] == list(range(1, 13))
    for step in result["affected"]:
        assert COOLER in session.db.compiled(FrameKey(DESKTOP, step, VIEW))


def test_keyframe_commit_updates_the_selected_keyframe_and_bumps_the_version(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(5), api.SCOPE_KEYFRAME)

    kfs = session.db.keyframes(DESKTOP, VIEW, COOLER)
    assert len(kfs) == 1  # still one keyframe: the existing one was re-traced
    assert kfs[0].version == 2
    assert np.array_equal(masks.decode_rle(kfs[0].parts[0].rle), cell(5))


def test_a_new_instance_goes_on_top_of_the_zorder(session):
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    order = session.db.zorder(DESKTOP, VIEW, 1).order
    assert order[-1] == (COOLER, "main")  # the order runs bottom -> top


def test_split_in_reverse_anchors_the_new_keyframe_at_the_current_step(session):
    session.goto(12)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    session.goto(8)
    draw(session, COOLER, cell(3), api.SCOPE_SPLIT)

    anchors = sorted(kf.anchor_step for kf in session.db.keyframes(DESKTOP, VIEW, COOLER))
    assert anchors == [8, 12]
    at_8 = session.db.compiled(FrameKey(DESKTOP, 8, VIEW))[COOLER]
    at_9 = session.db.compiled(FrameKey(DESKTOP, 9, VIEW))[COOLER]
    assert masks.decode_rle(at_8["visible_rle"]).sum() == cell(3).sum()
    assert at_8["visible_rle"]["counts"] != at_9["visible_rle"]["counts"]


def test_split_forward_moves_the_old_anchor_back(session):
    session.goto(12)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    session.goto(8)
    session.begin_edit(COOLER)
    session.set_editing_mask(cell(3))
    session.commit_edit(api.SCOPE_SPLIT, direction="forward")

    anchors = sorted(kf.anchor_step for kf in session.db.keyframes(DESKTOP, VIEW, COOLER))
    assert anchors == [7, 12]  # the old shape now ends at k-1, the new one inherits


def test_frame_override_touches_only_this_frame(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    result = draw(session, COOLER, cell(6), api.SCOPE_FRAME_OVERRIDE)

    assert result["affected"] == [10]
    assert COOLER in session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW))
    assert session.db.frame_overrides(FrameKey(DESKTOP, 9, VIEW)) == {}
    at_10 = session.db.compiled(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    at_9 = session.db.compiled(FrameKey(DESKTOP, 9, VIEW))[COOLER]
    assert np.array_equal(masks.decode_rle(at_10["visible_rle"]), cell(6))
    assert np.array_equal(masks.decode_rle(at_9["visible_rle"]), cell(0))


def test_commit_logs_a_source_level_operation(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    ops = session.db.ops(DESKTOP, VIEW)
    assert ops[0]["kind"] == "commit_keyframe"
    assert ops[0]["annotator"] == "tester"
    assert ops[0]["payload"]["instance"] == COOLER


def test_commit_box_writes_a_bench_rectangle(session):
    session.goto(14)  # the cooler is on the bench from step 13 on
    result = session.commit_box(COOLER, (2.0, 2.0, 12.0, 12.0))
    kfs = [kf for kf in session.db.keyframes(DESKTOP, VIEW, COOLER) if kf.geom_type == "box"]
    assert len(kfs) == 1
    assert kfs[0].placement == "on_bench"
    assert kfs[0].anchor_step == LAST_STEP
    assert result["affected"] == [13, 14]


# --------------------------------------------------------------------------- #
# scope suggestion (spec 4.3)
# --------------------------------------------------------------------------- #
def test_suggest_scope_proposes_a_zorder_change_inside_another_shape(session):
    session.goto(10)
    draw(session, CHASSIS, rect(0, 0, 40, 40), api.SCOPE_KEYFRAME)
    draw(session, COOLER, rect(50, 50, 60, 60), api.SCOPE_KEYFRAME)
    compiled = session.compiled()
    assert suggest_scope(compiled, COOLER, rect(4, 4, 12, 12)) == f"zorder:{CHASSIS}"
    assert suggest_scope(compiled, COOLER, rect(44, 44, 48, 48)) == api.SCOPE_KEYFRAME


def test_commit_with_a_zorder_scope_writes_a_pair_override(session):
    session.goto(10)
    draw(session, CHASSIS, rect(0, 0, 40, 40), api.SCOPE_KEYFRAME)
    draw(session, COOLER, rect(50, 50, 60, 60), api.SCOPE_KEYFRAME)
    session.begin_edit(COOLER)
    session.set_editing_mask(rect(4, 4, 12, 12))
    session.commit_edit(f"zorder:{CHASSIS}")

    pairs = session.db.pair_overrides(DESKTOP, VIEW, 1)
    assert [(p.above, p.below) for p in pairs] == [(COOLER, CHASSIS)]


# --------------------------------------------------------------------------- #
# instance list, visibility, z-order
# --------------------------------------------------------------------------- #
def test_instance_rows_are_top_first_with_the_fields_the_panel_reads(session):
    session.goto(10)
    seed_shapes(session, 10)
    rows = session.instance_rows()
    assert rows
    assert set(rows[0]) == {"key", "cls", "state", "placement", "visibility", "z", "hidden"}
    assert [r["z"] for r in rows] == sorted((r["z"] for r in rows), reverse=True)


def test_set_visibility_writes_a_frame_override(session):
    session.goto(10)
    seed_shapes(session, 10)
    session.set_visibility(COOLER, "motion_blur")
    override = session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert override.visibility == "motion_blur"
    row = next(r for r in session.instance_rows() if r["key"] == COOLER)
    assert row["visibility"] == "motion_blur"


def test_set_visibility_keeps_an_existing_mask_override(session):
    session.goto(10)
    seed_shapes(session, 10)
    draw(session, COOLER, cell(20), api.SCOPE_FRAME_OVERRIDE)
    session.set_visibility(COOLER, "motion_blur")
    override = session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert override.visibility == "motion_blur"
    assert np.array_equal(masks.decode_rle(override.visible_rle), cell(20))


def test_set_hidden_is_a_view_setting_and_is_not_persisted(session):
    session.goto(10)
    seed_shapes(session, 10)
    session.set_hidden(COOLER, True)
    assert next(r for r in session.instance_rows() if r["key"] == COOLER)["hidden"] is True
    assert session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW)) == {}
    assert session.db.ops(DESKTOP, VIEW) == []


def test_set_zorder_move_puts_one_instance_above_another(session):
    session.goto(10)
    seed_shapes(session, 10)
    rows = session.instance_rows()
    bottom, above_of = rows[-1]["key"], rows[-2]["key"]
    session.set_zorder_move(bottom, above_of)
    order = [inst for inst, _ in session.db.zorder(DESKTOP, VIEW, 1).order]
    assert order.index(bottom) == order.index(above_of) + 1


# --------------------------------------------------------------------------- #
# confirming a frame (spec 4.2 step 5)
# --------------------------------------------------------------------------- #
def test_confirm_frame_refuses_a_frame_with_a_missing_shape(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    problems: list[list[str]] = []
    session.sigProblems.connect(problems.append)

    assert session.confirm_frame() is False
    assert problems and f"missing_shape:{CHASSIS}" in problems[0]
    assert session.current().step == 12  # it did not move on
    assert session.frame_status(12) != api.STATUS_VERIFIED


def test_confirm_frame_accepts_a_complete_frame_and_steps_back(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    assert session.confirm_frame() is False
    draw(session, CHASSIS, cell(60), api.SCOPE_KEYFRAME)

    assert session.confirm_frame() is True
    assert session.frame_status(12) == api.STATUS_VERIFIED
    assert session.current().step == 11  # reverse order: the next frame is k-1


# --------------------------------------------------------------------------- #
# frame status and queues (spec 3.4 / 4.4)
# --------------------------------------------------------------------------- #
def test_frame_status_reports_unlabeled_then_auto(session):
    assert session.frame_status(10) == api.STATUS_UNLABELED
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert session.frame_status(10) == api.STATUS_AUTO


def test_a_conflicting_edit_on_a_verified_frame_is_queued(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    draw(session, CHASSIS, cell(60), api.SCOPE_KEYFRAME)
    assert session.confirm_frame() is True
    assert session.frame_status(12) == api.STATUS_VERIFIED

    session.goto(12)
    result = draw(session, CHASSIS, rect(30, 30, 60, 60), api.SCOPE_KEYFRAME)

    assert result["conflicts"] >= 1
    assert session.frame_status(12) == api.STATUS_CONFLICT
    queued = session.queues()[api.QUEUE_CONFLICTS]
    assert [entry["step"] for entry in queued] == [12]
    assert queued[0]["instance"] == CHASSIS
    assert queued[0]["sym_diff_px"] > 0


def test_resolve_conflict_closes_the_queue_entry(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    draw(session, CHASSIS, cell(60), api.SCOPE_KEYFRAME)
    session.confirm_frame()
    session.goto(12)
    draw(session, CHASSIS, rect(30, 30, 60, 60), api.SCOPE_KEYFRAME)

    cid = session.queues()[api.QUEUE_CONFLICTS][0]["id"]
    session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW)
    assert session.queues()[api.QUEUE_CONFLICTS] == []
    assert session.frame_status(12) == api.STATUS_VERIFIED


def test_queues_report_missing_shapes_and_needs_review(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    missing = session.queues()[api.QUEUE_MISSING_SHAPE]
    assert {entry["instance"] for entry in missing if entry["step"] == 10} >= {CHASSIS}
    assert session.queues()[api.QUEUE_UNEXPLAINED] == []

    session.truth.demote_frame(FrameKey(DESKTOP, 9, VIEW), "test")
    assert [e["step"] for e in session.queues()[api.QUEUE_NEEDS_REVIEW]] == [9]
    assert session.frame_status(9) == api.STATUS_NEEDS_REVIEW


# --------------------------------------------------------------------------- #
# undo (spec 4.6)
# --------------------------------------------------------------------------- #
def test_undo_of_a_commit_restores_the_previous_shape(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(5), api.SCOPE_KEYFRAME)

    assert session.undo() is True
    kf = session.db.keyframes(DESKTOP, VIEW, COOLER)[0]
    assert np.array_equal(masks.decode_rle(kf.parts[0].rle), cell(0))
    row = session.db.compiled(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert np.array_equal(masks.decode_rle(row["visible_rle"]), cell(0))

    assert session.redo() is True
    kf = session.db.keyframes(DESKTOP, VIEW, COOLER)[0]
    assert np.array_equal(masks.decode_rle(kf.parts[0].rle), cell(5))


def test_undo_of_the_first_commit_takes_the_shape_out_of_every_frame(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert session.undo() is True

    assert session.db.keyframes(DESKTOP, VIEW, COOLER) == []
    # the instance still needs geometry here, so the truth row stays -- but with
    # nothing in it, and the frame is back to reporting a missing shape
    row = session.db.compiled(FrameKey(DESKTOP, 10, VIEW))[COOLER]
    assert row["visible_rle"] is None
    assert f"missing_shape:{COOLER}" in session.compiled().problems


def test_undo_restores_a_frame_override_and_a_zorder(session):
    session.goto(10)
    seed_shapes(session, 10)
    before = list(session.db.zorder(DESKTOP, VIEW, 1).order)
    rows = session.instance_rows()
    session.set_zorder_move(rows[-1]["key"], rows[-2]["key"])
    assert session.undo() is True
    assert session.db.zorder(DESKTOP, VIEW, 1).order == before

    draw(session, COOLER, cell(20), api.SCOPE_FRAME_OVERRIDE)
    assert session.undo() is True
    assert session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW)) == {}


def test_undo_of_a_brush_stroke_restores_the_editing_layer(session):
    session.goto(10)
    session.begin_edit(COOLER)
    before = np.zeros(HW, dtype=bool)
    session.set_editing_mask(before)
    session.undo_stack.push(edit_editing_mask_op(COOLER, before, cell(2)), apply=True)

    assert np.array_equal(session.editing_mask(), cell(2))
    assert session.undo() is True
    assert not session.editing_mask().any()  # the stroke never touched the database
    assert session.db.keyframes(DESKTOP, VIEW, COOLER) == []


def test_undo_restores_an_occluder(session):
    session.goto(10)
    seed_shapes(session, 10)
    session.commit_occluder(rect(0, 0, 20, 20), "hand")
    assert len(session.db.occluders(FrameKey(DESKTOP, 10, VIEW))) == 1
    assert session.undo() is True
    assert session.db.occluders(FrameKey(DESKTOP, 10, VIEW)) == []


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def test_save_and_close_clear_the_dirty_flag(session):
    dirty: list[bool] = []
    session.sigDirty.connect(dirty.append)
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert dirty[-1] is True
    session.save()
    assert dirty[-1] is False
    session.close()
    assert session.steps() == []
