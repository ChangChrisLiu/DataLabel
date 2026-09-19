"""A Label Studio draft is draft material, not an instance of the machine.

``tda.core.ls_import`` writes 414 provisional instances ``ls:<label>#<n>`` with
a real taxonomy class and 11,191 draft keyframes: the team's old annotations of
fourteen desktops, kept so that a later "adopt draft" tool can turn them into
real instances. Until somebody does that in S1 they are not parts of anything.

They carry no actions, so the state machine has them ``installed`` and
``in_chassis`` for ever -- which made every one of them a missing chassis shape
on every frame of those desktops, and ``verify_frame`` refused every frame of
D13/scan on 42 blocking problems nobody could act on. The annotator could never
press Space.

This module pins the ruling down at both choke points: the geometry one
(:func:`tda.core.states.needs_geom`) and the export one
(:meth:`tda.core.export.coco.DesktopCtx.cls_of`), plus what S1 may do with a
draft and what the session may not.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.export import export_coco, export_vlm
from tda.core.model import FrameKey, InstanceRec, ShapeKeyframe, ShapePart, is_provisional
from tda.core.states import initial_state, needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.ui import session_api as api
from tda.ui.session_api import SessionRefusal
from session_scene import DESKTOP as S_DESKTOP, VIEW as S_VIEW, make_session, seed_shapes
from truth_scenes import DESKTOP, PSU, SCREW, VIEW, Scene, build_scene, rect

#: What the importer writes: a real taxonomy class on a provisional key.
DRAFT = "ls:Motherboard#1"
DRAFT_TWO = "ls:PSU#2"
LS_SOURCE = "labelstudio"


# --------------------------------------------------------------------------- #
# scenes
# --------------------------------------------------------------------------- #
@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    yield d
    d.close()


def add_draft(db: Db, desktop: int, view: str, key: str, cls: str, step: int,
              box=(0, 0, 8, 8)) -> int:
    """One provisional instance plus the draft keyframe the importer gives it."""
    db.upsert_instance(InstanceRec(key=key, desktop=desktop, cls=cls,
                                   raw_names=[key.split(":", 1)[1]]))
    return db.add_keyframe(ShapeKeyframe(
        id=None, instance=key, desktop=desktop, view=view, pose_segment=0,
        anchor_step=step, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(*box)))],
        amodal_complete=False, source=LS_SOURCE, draft_id=4242,
    ))


@pytest.fixture
def drafted(db: Db) -> Scene:
    """The truth scene with two Label Studio drafts imported on top of it."""
    scene = build_scene(db)
    add_draft(db, DESKTOP, VIEW, DRAFT, "motherboard", 1)
    add_draft(db, DESKTOP, VIEW, DRAFT_TWO, "psu", 1, box=(20, 20, 28, 28))
    return scene


# --------------------------------------------------------------------------- #
# the predicate
# --------------------------------------------------------------------------- #
def test_is_provisional_is_one_predicate_in_one_place():
    from tda.core import graph_infer, model

    assert model.is_provisional("ls:Motherboard#1")
    assert not model.is_provisional("motherboard.01")
    assert not model.is_provisional("")
    # graph_infer keeps its name working, but no longer owns the answer
    assert graph_infer.is_provisional is model.is_provisional
    assert graph_infer.LS_PREFIX == model.LS_PREFIX


def test_needs_geom_never_asks_for_a_draft_shape():
    tax = load_taxonomy()
    instances = {
        "psu.01": InstanceRec(key="psu.01", desktop=1, cls="psu"),
        DRAFT: InstanceRec(key=DRAFT, desktop=1, cls="motherboard"),
    }
    geom = needs_geom(instances, initial_state(instances, tax), tax)
    assert geom == {"psu.01": "mask"}


# --------------------------------------------------------------------------- #
# the truth table
# --------------------------------------------------------------------------- #
def test_a_draft_is_never_a_missing_shape_and_the_frame_verifies(drafted: Scene):
    out = drafted.svc.refresh(drafted.key(1))

    assert not [p for p in out["problems"] if ":ls:" in p]
    assert DRAFT not in drafted.rows(1) and DRAFT_TWO not in drafted.rows(1)
    assert sorted(drafted.rows(1)) == [PSU, SCREW]

    drafted.svc.verify_frame(drafted.key(1), "tester")  # this used to raise
    assert drafted.review_status(1) == "verified"


def test_importing_drafts_does_not_move_a_single_input_digest(db: Db):
    """A re-import must not queue every frozen frame of fourteen desktops."""
    scene = build_scene(db)
    scene.refresh_all()
    before = {step: db.frame_digest(scene.key(step))["digest"] for step in (1, 2, 3)}

    add_draft(db, DESKTOP, VIEW, DRAFT, "motherboard", 1)
    add_draft(db, DESKTOP, VIEW, DRAFT_TWO, "psu", 2, box=(20, 20, 28, 28))

    assert {step: scene.svc.inputs_digest(scene.key(step)) for step in (1, 2, 3)} == before


def test_the_draft_keyframes_stay_in_the_database(drafted: Scene):
    """A later "adopt draft" tool needs the pixels the team already traced."""
    drafted.svc.refresh_range(DESKTOP, VIEW, [1, 2, 3])
    kfs = [kf for kf in drafted.db.keyframes(DESKTOP, VIEW) if kf.source == LS_SOURCE]
    assert len(kfs) == 2
    assert {kf.instance for kf in kfs} == {DRAFT, DRAFT_TWO}


# --------------------------------------------------------------------------- #
# the exports
# --------------------------------------------------------------------------- #
def test_no_export_ever_names_a_provisional_key(drafted: Scene, tmp_path: Path):
    drafted.svc.refresh_range(DESKTOP, VIEW, [1, 2, 3])
    # a draft row that predates the fix is still in the table: the export choke
    # point has to drop it, not rely on the compiler never having written it
    drafted.db.put_compiled(drafted.key(1), DRAFT, masks.encode_rle(rect(0, 0, 8, 8)),
                            0.0, "visible", "in_chassis", "verified", "h",
                            verified_by="tester")

    # such a row is one the compiler will never produce again, so the view now
    # carries a standing conflict as well: that is what `allow_conflicts` is for
    doc = export_coco(drafted.db, drafted.tax, [DESKTOP], VIEW,
                      str(tmp_path / "c.json"), only_verified=False,
                      include_boxes=True, allow_conflicts=True)
    keys = [a["attributes"]["instance_key"] for a in doc["annotations"]]
    assert keys and not [k for k in keys if is_provisional(k)]

    out = tmp_path / "v.jsonl"
    export_vlm(drafted.db, drafted.tax, [DESKTOP], VIEW, str(out),
               allow_conflicts=True)
    assert out.read_text(encoding="utf-8")
    assert "ls:" not in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# S1: a draft may be deleted, with its own draft keyframes
# --------------------------------------------------------------------------- #
def test_s1_deletes_a_draft_together_with_its_draft_keyframes(db: Db):
    from tda.ui.steps_delete import check_deletable, delete_instance
    from tda.ui.steps_model import StepTableData

    scene = build_scene(db)
    kf_id = add_draft(db, DESKTOP, VIEW, DRAFT, "motherboard", 1)
    data = StepTableData.load(db, DESKTOP, scene.tax)

    check_deletable(data, db, DRAFT)  # the draft keyframe must not block it
    delete_instance(data, db, DRAFT)

    assert DRAFT not in db.instances(DESKTOP)
    assert not [kf for kf in db.keyframes(DESKTOP, VIEW) if kf.id == kf_id]
    # ... and the real instances are untouched
    assert sorted(db.instances(DESKTOP)) == [PSU, SCREW]


def test_the_step_table_says_drafts_are_reference_material_once(db: Db):
    """D13 got 63 lines telling the annotator to delete or retarget a draft.

    They are not orphaned parts of the machine: nothing will ever reference
    them, because they are the team's old tracings waiting to be adopted. One
    collapsed line per desktop, last, says so without burying the real work.
    """
    from tda.ui.steps_model import StepTableData

    scene = build_scene(db)
    for n in (1, 2, 3):
        add_draft(db, DESKTOP, VIEW, f"ls:Motherboard#{n}", "motherboard", 1,
                  box=(n, n, n + 4, n + 4))
    data = StepTableData.load(db, DESKTOP, scene.tax)

    named = [line for line in data.orphans if "ls:" in line]
    assert named == [
        "3 Label Studio drafts (ls:*) on this desktop - they are reference "
        "material and need no action"
    ]
    assert data.orphans[-1] == named[0]  # lowest priority, after the real work
    assert not [line for line in data.orphans if "retarget" in line and "ls:" in line]


def test_a_desktop_without_drafts_says_nothing_about_them(db: Db):
    from tda.ui.steps_model import StepTableData

    scene = build_scene(db)
    data = StepTableData.load(db, DESKTOP, scene.tax)
    assert not [line for line in data.orphans if "Label Studio" in line]


def test_s1_still_refuses_a_real_instance_that_has_shapes(db: Db):
    from tda.ui.steps_delete import check_deletable
    from tda.ui.steps_model import StepTableData
    from tda.ui.steps_values import EditError

    scene = build_scene(db)
    data = StepTableData.load(db, DESKTOP, scene.tax)
    with pytest.raises(EditError):
        check_deletable(data, db, PSU)


def test_a_draft_with_a_hand_drawn_shape_is_not_deletable(db: Db):
    """Only *draft* keyframes go with the draft: a manual one is somebody's work."""
    from tda.ui.steps_delete import check_deletable
    from tda.ui.steps_model import StepTableData
    from tda.ui.steps_values import EditError

    scene = build_scene(db)
    add_draft(db, DESKTOP, VIEW, DRAFT, "motherboard", 1)
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=DRAFT, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=1, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(0, 0, 4, 4)))],
        source="manual",
    ))
    data = StepTableData.load(db, DESKTOP, scene.tax)
    with pytest.raises(EditError):
        check_deletable(data, db, DRAFT)


# --------------------------------------------------------------------------- #
# the session: no card, no rows, no editing
# --------------------------------------------------------------------------- #
@pytest.fixture
def session(tmp_path: Path):
    made = make_session(tmp_path)
    add_draft(made.db, S_DESKTOP, S_VIEW, DRAFT, "motherboard", 1)
    made.review.invalidate()
    yield made
    made.close(force=True)


def test_the_start_card_lists_no_drafts(session):
    session.goto(1)
    listed = [item["instance"] for item in session.task_card()]
    assert listed and not [key for key in listed if is_provisional(key)]


def test_the_missing_shape_queue_lists_no_drafts(session):
    queued = session.queues()[api.QUEUE_MISSING_SHAPE]
    assert queued and not [row for row in queued if is_provisional(row["instance"])]


def test_instance_rows_never_show_a_draft(session):
    session.goto(10)
    seed_shapes(session, 10)
    rows = [row["key"] for row in session.instance_rows()]
    assert rows and not [key for key in rows if is_provisional(key)]


def test_begin_edit_refuses_a_draft(session):
    """A draft keyframe is the team's record of what they traced, not a layer."""
    session.goto(10)
    with pytest.raises(SessionRefusal) as refused:
        session.begin_edit(DRAFT)
    assert "Label Studio" in str(refused.value)
    assert session.editing_instance is None
