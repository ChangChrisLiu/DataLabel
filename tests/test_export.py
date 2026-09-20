"""Tests for the rehearsal exports: COCO instance segmentation + VLM JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

from tda.core.db import Db
from tda.core.export import export_coco, export_vlm
from tda.core.export.coco import cache_rel_path, categories, image_id, load_ctx
from tda.core.masks import encode_rle
from tda.core.model import (
    ActionRec,
    FrameKey,
    FrameOverride,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StateEvent,
    StepRec,
)
from tda.core.taxonomy import load_taxonomy

DESKTOP = 1
VIEW = "scan"
HW = (64, 64)
PSU = "psu.01"
SCREW = "screw.motherboard.03"
GHOST = "ls:mystery#1"  # a provisional draft key with no instance record
#: The pose segment `truth_inputs.pose_segment_of` reports when none is recorded.
SEGMENT = 1
#: The staging area this view can see, without which no bench row exists.
BENCH_ROI = (0, 0, 64, 64)
#: The psu's rectangle once it lies on the bench: x0, y0, x1, y1.
BENCH_BOX = [2.0, 3.0, 12.0, 15.0]


def _rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(HW, bool)
    m[y0:y1, x0:x1] = True
    return m


#: bbox [x=5, y=10, w=20, h=20], area 400
PSU_MASK = _rect(10, 30, 5, 25)
#: bbox [x=50, y=40, w=8, h=8], area 64
SCREW_MASK = _rect(40, 48, 50, 58)
#: x0, y0, x1, y1 -- contains both masks, 40x32 pixels
ROI = [4, 8, 44, 40]


def _steps() -> list[StepRec]:
    return [
        StepRec(DESKTOP, 1, "initial", "initial"),
        StepRec(DESKTOP, 2, "normal", "motherboard screw 3"),
        StepRec(DESKTOP, 3, "normal", "psu"),
    ]


def _actions() -> list[ActionRec]:
    return [
        ActionRec(DESKTOP, 2, 0, SCREW, "unscrew", tool="PH2", direction="+Z"),
        ActionRec(DESKTOP, 3, 0, PSU, "remove", tool="hand", direction="+Z"),
    ]


@pytest.fixture
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_db_path: str, tax):
    """One desktop, scan view, 3 steps, a psu and one motherboard screw.

    Step 2 unscrews the (non-captive) screw, step 3 removes the psu, so the
    state machine gives: step 1 ``installed``/``fastened``, step 2 the screw is
    ``removed``, step 3 the psu is ``removed`` and ``on_bench``.
    """
    d = Db(tmp_db_path)
    d.upsert_desktop(DESKTOP, {"brand": "dell", "split": "train"})
    d.upsert_instance(InstanceRec(key=PSU, desktop=DESKTOP, cls="psu"))
    d.upsert_instance(
        InstanceRec(
            key=SCREW, desktop=DESKTOP, cls="screw",
            attrs={"role": "motherboard", "head": "PH2", "captive": False},
        )
    )
    d.replace_steps(DESKTOP, _steps(), _actions())
    for step in (1, 2, 3):
        d.upsert_frame(
            FrameKey(DESKTOP, step, VIEW),
            f"F:/scan/019/{step:03d}/P_0.png",
            {"hw": [64, 64]},
            "2025-05-31T10:00:00",
            flags={"review_status": "confirmed" if step == 1 else "unlabeled"},
        )
    psu_rle, screw_rle = encode_rle(PSU_MASK), encode_rle(SCREW_MASK)
    d.put_compiled(FrameKey(DESKTOP, 1, VIEW), PSU, psu_rle, 0.0, "visible",
                   "in_chassis", "verified", "h1", verified_by="tester")
    d.put_compiled(FrameKey(DESKTOP, 1, VIEW), SCREW, screw_rle, 0.25, "occluded_partial",
                   "in_chassis", "verified", "h1", verified_by="tester")
    d.put_compiled(FrameKey(DESKTOP, 2, VIEW), PSU, psu_rle, 0.0, "visible",
                   "in_chassis", "auto", "h2")
    # step 3: the psu lies on the bench -- a box row, no visible RLE.
    d.put_compiled(FrameKey(DESKTOP, 3, VIEW), PSU, None, 0.0, "visible",
                   "on_bench", "auto", "h3", geom_type="box", box=BENCH_BOX)
    d.add_keyframe(ShapeKeyframe(
        id=None, instance=PSU, desktop=DESKTOP, view=VIEW, pose_segment=SEGMENT,
        anchor_step=2, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", rle=psu_rle)], amodal_complete=False,
    ))
    # The screw deliberately has no keyframe of its own (several tests are about
    # what an annotation without one reports). Its frozen row at step 1 still
    # has to follow from *some* input, or the export's own `ensure_fresh`
    # queues a conflict about a mask nobody drew -- and an unsettled
    # disagreement is a view no export may publish. A frame override is exactly
    # that input: geometry for this one frame, and no keyframe.
    d.set_frame_override(FrameOverride(FrameKey(DESKTOP, 1, VIEW), SCREW,
                                       screw_rle, "occluded_partial"))
    d.add_keyframe(ShapeKeyframe(
        id=None, instance=PSU, desktop=DESKTOP, view=VIEW, pose_segment=SEGMENT,
        anchor_step=3, placement="on_bench", geom_type="box",
        parts=[ShapePart("main", box=tuple(BENCH_BOX))], amodal_complete=True,
    ))
    # this scene is about the bench row at step 3, so its view is one that can
    # see the staging area: without an ROI an on_bench instance is not part of
    # the frame at all (spec 3.3 step 2)
    d.set_pose_segment(DESKTOP, VIEW, SEGMENT, 1, 3, 1, None, None)
    d.set_pose_segment_bench_roi(DESKTOP, VIEW, SEGMENT, BENCH_ROI)
    yield d
    d.close()


def _with_roi(d: Db) -> None:
    """Give the view a pose segment carrying an ROI (no Db writer for roi_json yet)."""
    d.set_pose_segment(DESKTOP, VIEW, SEGMENT, 1, 3, 1, None, None)
    with d.conn:
        d.conn.execute(
            "UPDATE pose_segment SET roi_json=? WHERE desktop=? AND view=? AND seg=?",
            (json.dumps(ROI), DESKTOP, VIEW, SEGMENT),
        )


def _ann_of(doc: dict, step: int, instance: str) -> dict:
    by_id = {im["id"]: im for im in doc["images"]}
    for ann in doc["annotations"]:
        if (by_id[ann["image_id"]]["extra"]["step"] == step
                and ann["attributes"]["instance_key"] == instance):
            return ann
    raise AssertionError(f"no annotation for {instance} at step {step}")


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
def test_coco_loads_in_pycocotools_with_23_categories(db, tax, tmp_path: Path):
    out = tmp_path / "coco.json"
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(out), only_verified=False)

    coco = COCO(str(out))
    assert len(coco.getCatIds()) == 23
    assert sorted(coco.getCatIds()) == list(range(1, 24))
    assert coco.cats[1]["name"] == next(iter(tax.classes))
    assert coco.cats[1]["supercategory"] == "structure"
    assert len(coco.getImgIds()) == len(doc["images"])
    assert len(coco.getAnnIds()) == len(doc["annotations"])


def test_categories_follow_taxonomy_order(tax):
    cats = categories(tax)
    assert [c["name"] for c in cats] == list(tax.classes)
    assert [c["id"] for c in cats] == list(range(1, 24))
    assert {c["supercategory"] for c in cats} == {
        "structure", "part", "fastener", "latch", "interface"
    }


def test_only_verified_filters_rows_and_images(db, tax, tmp_path: Path):
    strict = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "gold.json"))
    assert [im["extra"]["step"] for im in strict["images"]] == [1]
    assert len(strict["annotations"]) == 2
    assert {a["attributes"]["verified"] for a in strict["annotations"]} == {True}

    full = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "all.json"),
                       only_verified=False)
    assert [im["extra"]["step"] for im in full["images"]] == [1, 2, 3]
    assert len(full["annotations"]) == 3  # the bench row carries no mask
    assert {a["attributes"]["verified"] for a in full["annotations"]} == {True, False}


def test_images_carry_ids_paths_and_extra(db, tax, tmp_path: Path):
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"), only_verified=False)
    first = doc["images"][0]
    assert first["id"] == image_id(DESKTOP, 1, VIEW) == 100010
    assert first["file_name"] == cache_rel_path(DESKTOP, 1, VIEW) == "scan/D01/s001.png"
    assert (first["width"], first["height"]) == (64, 64)
    assert first["extra"] == {"desktop": 1, "step": 1, "view": "scan",
                              "review_status": "confirmed"}


def test_annotation_geometry_and_attributes(db, tax, tmp_path: Path):
    out = tmp_path / "c.json"
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(out), only_verified=False)
    psu = _ann_of(doc, 1, PSU)

    assert psu["bbox"] == [5, 10, 20, 20]
    assert psu["area"] == 400
    assert psu["iscrowd"] == 0
    assert psu["category_id"] == list(tax.classes).index("psu") + 1
    assert psu["segmentation"]["size"] == [64, 64]
    assert psu["attributes"] == {
        "instance_key": PSU, "state": "installed", "placement": "in_chassis",
        "visibility": "visible", "occlusion_ratio": 0.0, "amodal_complete": False,
        "implied": False, "tier": "gold", "verified": True,
    }

    screw = _ann_of(doc, 1, SCREW)
    assert screw["bbox"] == [50, 40, 8, 8]
    assert screw["area"] == 64
    assert screw["attributes"]["state"] == "fastened"
    assert screw["attributes"]["amodal_complete"] is None  # no keyframe for the screw

    # the segmentation really is the seeded mask
    coco = COCO(str(out))
    assert np.array_equal(coco.annToMask(coco.anns[psu["id"]]).astype(bool), PSU_MASK)


def test_state_follows_the_action_log(db, tax, tmp_path: Path):
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"), only_verified=False)
    assert _ann_of(doc, 2, PSU)["attributes"]["state"] == "installed"
    with pytest.raises(AssertionError):  # the bench row carries no mask
        _ann_of(doc, 3, PSU)
    boxed = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "b.json"),
                        only_verified=False, include_boxes=True)
    bench = _ann_of(boxed, 3, PSU)
    assert bench["attributes"]["state"] == "removed"
    assert bench["attributes"]["placement"] == "on_bench"


def test_include_boxes_emits_bbox_only_rows(db, tax, tmp_path: Path):
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      only_verified=False, include_boxes=True)
    assert len(doc["annotations"]) == 4
    bench = _ann_of(doc, 3, PSU)
    assert bench["segmentation"] == []
    assert bench["bbox"] == [2.0, 3.0, 10.0, 12.0]  # the compiled row's own box
    assert bench["area"] == 120.0
    assert bench["attributes"]["amodal_complete"] is True


def test_roi_crop_shifts_coordinates(db, tax, tmp_path: Path):
    _with_roi(db)
    out = tmp_path / "roi.json"
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(out), only_verified=False, roi_crop=True)

    image = doc["images"][0]
    assert image["roi"] == ROI
    assert (image["width"], image["height"]) == (40, 32)
    psu = _ann_of(doc, 1, PSU)
    assert psu["bbox"] == [1, 2, 20, 20]
    assert psu["area"] == 400
    assert psu["segmentation"]["size"] == [32, 40]
    COCO(str(out))  # still a valid COCO file


def test_coco_file_matches_the_returned_document(db, tax, tmp_path: Path):
    out = tmp_path / "c.json"
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(out), only_verified=False)
    assert json.loads(out.read_text(encoding="utf-8")) == doc
    assert doc["info"]["view"] == VIEW and doc["info"]["desktops"] == [DESKTOP]


# --------------------------------------------------------------------------- #
# a board-mounted latch leaves the chassis inside the motherboard
# --------------------------------------------------------------------------- #
BOARD, LATCH = "motherboard.01", "ram_latch.01"
BOARD_MASK = _rect(0, 40, 0, 40)
LATCH_MASK = _rect(4, 12, 4, 12)


@pytest.fixture
def board_db(tmp_db_path: str, tax):
    """A board with one RAM latch on it: the clip opens at 2, the board goes at 3.

    The taxonomy's ``host_class`` is what the S1 heuristic turns into this
    ``parent``/``attached`` pair, and the spec-3.3 cascade then takes the latch
    out with the board. This view has no staging-area ROI, so once a part is on
    the bench it is in nobody's picture.
    """
    d = Db(tmp_db_path)
    d.upsert_desktop(DESKTOP, {"brand": "dell", "split": "train"})
    d.upsert_instance(InstanceRec(key=BOARD, desktop=DESKTOP, cls="motherboard"))
    d.upsert_instance(InstanceRec(key=LATCH, desktop=DESKTOP, cls="ram_latch",
                                  parent=BOARD, attached=True))
    d.replace_steps(
        DESKTOP,
        [StepRec(DESKTOP, 1, "initial", "initial"),
         StepRec(DESKTOP, 2, "normal", "ram clip 1"),
         StepRec(DESKTOP, 3, "normal", "motherboard")],
        [ActionRec(DESKTOP, 2, 0, LATCH, "open", tool="hand"),
         ActionRec(DESKTOP, 3, 0, BOARD, "remove", tool="hand", direction="+Z")],
    )
    for step in (1, 2, 3):
        d.upsert_frame(FrameKey(DESKTOP, step, VIEW), f"F:/scan/019/{step:03d}/P_0.png",
                       {"hw": [64, 64]}, "2025-05-31T10:00:00",
                       flags={"review_status": "unlabeled"})
    for instance, mask in ((BOARD, BOARD_MASK), (LATCH, LATCH_MASK)):
        d.add_keyframe(ShapeKeyframe(
            id=None, instance=instance, desktop=DESKTOP, view=VIEW,
            pose_segment=SEGMENT, anchor_step=3, placement="in_chassis",
            geom_type="mask", parts=[ShapePart("main", rle=encode_rle(mask))],
        ))
    yield d
    d.close()


def _ann_keys(doc: dict, step: int) -> set[str]:
    by_id = {im["id"]: im for im in doc["images"]}
    return {ann["attributes"]["instance_key"] for ann in doc["annotations"]
            if by_id[ann["image_id"]]["extra"]["step"] == step}


def test_coco_stops_exporting_a_latch_once_its_board_is_out(board_db, tax, tmp_path: Path):
    doc = export_coco(board_db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      only_verified=False, include_boxes=True)
    assert _ann_keys(doc, 1) == {BOARD, LATCH}
    assert _ann_of(doc, 2, LATCH)["attributes"]["state"] == "open"
    # step 3: the board is on the bench and this view has no staging area, and
    # the latch went out inside it -- neither is anybody's work here
    assert _ann_keys(doc, 3) == set()


def test_the_vlm_export_asks_about_the_latch_only_while_it_is_there(
    board_db, tax, tmp_path: Path
):
    """Both classes still have more than one state, so V2 keeps asking (spec 8)."""
    _confirm(board_db, tax, (1, 2, 3))
    out = tmp_path / "vlm.jsonl"
    export_vlm(board_db, tax, [DESKTOP], VIEW, str(out), only_verified=False)
    states = {(r["step"], r["answer"]["state"]) for r in _records(out)
              if r["task"] == "V2" and r["id"].endswith(f"state-{LATCH}")}
    assert states == {(1, "closed"), (2, "open")}


# --------------------------------------------------------------------------- #
# VLM
# --------------------------------------------------------------------------- #
def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _confirm(d: Db, tax, steps) -> None:
    """Sign the given frames off the way pressing Space does.

    The VLM export asks a *perception* question only about a frame a human has
    confirmed (spec 8.2: the answer is the truth table, and an unconfirmed truth
    table is a cache). Setting ``review_status`` by hand is not the same thing
    and must not be -- the truth service demotes a confirmed frame whose
    instance set has changed -- so the fixtures go through the service.
    """
    from tda.core.truth import TruthService

    service = TruthService(d, tax)
    service.refresh_range(DESKTOP, VIEW, steps)
    for step in steps:
        service.verify_frame(FrameKey(DESKTOP, step, VIEW), "tester")


def test_vlm_writes_at_least_one_record_per_task(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    summary = export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    records = _records(out)

    assert summary["records"] == len(records)
    assert all(summary["by_task"][t] >= 1 for t in ("V1", "V2", "V3"))
    assert len({r["id"] for r in records}) == len(records)
    for rec in records:
        assert set(rec) >= {"id", "task", "desktop", "step", "view", "images",
                            "question", "answer", "evidence", "rationale", "tier",
                            "verified", "graph_version"}
        assert rec["tier"] == "gold" and isinstance(rec["verified"], bool)
        assert rec["desktop"] == DESKTOP and rec["view"] == VIEW
        assert rec["graph_version"] is None
        assert rec["question"] and isinstance(rec["question"], str)
        assert json.loads(json.dumps(rec["answer"])) == rec["answer"]
        assert all(img.startswith("scan/D01/s") for img in rec["images"])


def test_vlm_v1_lists_the_visible_components(db, tax, tmp_path: Path):
    _confirm(db, tax, (2, 3))  # step 1 is confirmed by the fixture's frozen rows
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    first = [r for r in _records(out) if r["step"] == 1]
    assert len(first) == 1
    components = first[0]["answer"]["components"]
    assert {c["instance"] for c in components} == {PSU, SCREW}
    psu = next(c for c in components if c["instance"] == PSU)
    assert psu["class"] == "psu" and psu["bbox"] == [5, 10, 20, 20]
    assert psu["placement"] == "in_chassis"
    assert first[0]["evidence"]["bboxes"][SCREW] == [50, 40, 8, 8]

    # spec 8.2: the bench part at step 3 is a box row, and V1 must still list it
    assert [r["step"] for r in _records(out)] == [1, 2, 3]
    bench = [r for r in _records(out) if r["step"] == 3][0]["answer"]["components"]
    assert bench == [{"class": "psu", "instance": PSU, "bbox": [2.0, 3.0, 10.0, 12.0],
                      "placement": "on_bench"}]


def test_vlm_v2_asks_states_and_counts(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V2",))
    records = _records(out)
    states = [r for r in records if "state" in r["answer"]]
    counts = [r for r in records if "count" in r["answer"]]

    screw = next(r for r in states if r["evidence"]["instances"] == [SCREW])
    assert screw["step"] == 1
    assert screw["answer"] == {"state": "fastened"}
    assert "Motherboard screw 3" in screw["question"]
    assert len(counts) == 1 and counts[0]["step"] == 1
    assert counts[0]["answer"] == {"count": 1}
    assert "motherboard screws" in counts[0]["question"]
    assert "fastened" in counts[0]["question"]


def test_vlm_v3_reads_the_action_between_two_frames(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V3",))
    records = _records(out)
    assert [r["step"] for r in records] == [2, 3]

    unscrew = records[0]
    assert unscrew["images"] == ["scan/D01/s001.png", "scan/D01/s002.png"]
    assert unscrew["answer"] == {
        "verb": "unscrew", "target_class": "screw",
        "target_instance": SCREW, "tool": "PH2",
    }
    assert unscrew["evidence"]["bboxes"][SCREW] == [50, 40, 8, 8]
    assert records[1]["answer"]["verb"] == "remove"
    assert records[1]["answer"]["tool"] == "hand"


def test_vlm_output_is_reproducible(db, tax, tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(a))
    export_vlm(db, tax, [DESKTOP], VIEW, str(b))
    assert a.read_bytes() == b.read_bytes()


def test_vlm_tasks_argument_selects_the_rows(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    summary = export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V3",))
    assert summary["by_task"] == {"V3": 2}
    assert {r["task"] for r in _records(out)} == {"V3"}


def test_vlm_only_verified_keeps_confirmed_records(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), only_verified=True)
    records = _records(out)
    assert records and {r["verified"] for r in records} == {True}
    # step 1 is the only verified frame; the V3 pair (1, 2) still reads it
    assert {r["step"] for r in records} == {1, 2}


def test_rationale_never_observes_without_a_box(db, tax, tmp_path: Path):
    """An instance the frame cannot localise is `propagate_state`, not `observe`.

    A second screw nobody has drawn joins the desktop, which is what demotes the
    confirmed frame to ``needs_review`` -- so what is left here is the planning
    half of the export, and the invariant has to hold there too. The V2 counting
    case that used to live here is in ``test_vlm_tasks.py``, on a scene whose
    frames can actually be signed off.
    """
    db.upsert_instance(InstanceRec(
        key="screw.motherboard.04", desktop=DESKTOP, cls="screw",
        attrs={"role": "motherboard", "head": "PH2", "captive": False},
    ))
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    records = _records(out)

    assert records
    for rec in records:
        for step in rec["rationale"]["steps"]:
            if step["op"] == "observe":
                assert step["evidence"]["bbox"], rec["id"]
                assert step["evidence"]["visibility"], rec["id"]


# --------------------------------------------------------------------------- #
# inputs shared by both exports
# --------------------------------------------------------------------------- #
def test_manual_events_do_not_erase_the_action_log(db, tax, tmp_path: Path):
    """A hand-written event corrects the derived log; it never replaces it."""
    db.replace_events(DESKTOP, [StateEvent(DESKTOP, 1, SCREW, "state", "fastened",
                                           "loosened", auto=False)])
    ctx = load_ctx(db, tax, DESKTOP, VIEW)
    assert ctx.state_at(1)[SCREW].state == "loosened"  # the correction lands
    assert ctx.state_at(2)[SCREW].state == "removed"   # the unscrew at step 2 survives
    assert ctx.state_at(3)[PSU].state == "removed"

    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"), only_verified=False)
    assert _ann_of(doc, 1, SCREW)["attributes"]["state"] == "loosened"


def test_amodal_complete_reads_the_frames_pose_segment(db, tax, tmp_path: Path):
    """A shape drawn in another pose segment must not answer for this frame."""
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=SCREW, desktop=DESKTOP, view=VIEW, pose_segment=SEGMENT + 1,
        anchor_step=1, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", rle=encode_rle(SCREW_MASK))], amodal_complete=True,
    ))
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"), only_verified=False)
    assert _ann_of(doc, 1, SCREW)["attributes"]["amodal_complete"] is None
    assert _ann_of(doc, 1, PSU)["attributes"]["amodal_complete"] is False


def test_unknown_instance_keys_are_skipped(db, tax, tmp_path: Path):
    """A provisional draft key has no taxonomy class: skip it instead of crashing.

    Such a row is by definition one the compiler will not produce again, so it
    is also a standing conflict -- ``allow_conflicts`` is what lets the export
    look at it at all.
    """
    db.put_compiled(FrameKey(DESKTOP, 1, VIEW), GHOST, encode_rle(SCREW_MASK), 0.0,
                    "visible", "in_chassis", "verified", "h1", verified_by="tester")
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      only_verified=False, allow_conflicts=True)
    assert GHOST not in {a["attributes"]["instance_key"] for a in doc["annotations"]}

    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), allow_conflicts=True)
    assert GHOST not in out.read_text(encoding="utf-8")


def test_unknown_action_target_produces_no_v3_record(db, tax, tmp_path: Path):
    db.replace_steps(
        DESKTOP,
        [*_steps(), StepRec(DESKTOP, 4, "normal", "mystery part")],
        [*_actions(), ActionRec(DESKTOP, 4, 0, GHOST, "remove", tool="hand")],
    )
    db.upsert_frame(FrameKey(DESKTOP, 4, VIEW), "F:/scan/019/004/P_0.png",
                    {"hw": [64, 64]}, None)
    db.put_compiled(FrameKey(DESKTOP, 4, VIEW), PSU, None, 0.0, "visible", "on_bench",
                    "auto", "h4", geom_type="box", box=BENCH_BOX)

    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V3",))
    assert [r["step"] for r in _records(out)] == [2, 3]


def test_step_types_gate_both_exports(db, tax, tmp_path: Path):
    """`ignore` frames are never exported and `dupli`/`initial` never end a V3 pair."""
    db.replace_steps(
        DESKTOP,
        [
            StepRec(DESKTOP, 1, "initial", "initial"),
            StepRec(DESKTOP, 2, "dupli", "motherboard screw 3", dupli=True),
            StepRec(DESKTOP, 3, "ignore", "calibration shot"),
        ],
        _actions(),
    )
    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      only_verified=False, include_boxes=True)
    assert [im["extra"]["step"] for im in doc["images"]] == [1, 2]

    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    assert {r["step"] for r in _records(out)} == {1, 2}
    assert not [r for r in _records(out) if r["task"] == "V3"]


def test_an_empty_mask_measures_as_no_box_at_all():
    """An instance with nothing visible is not a detection.

    The bbox is read straight off the run lengths now, and pycocotools measures
    an empty RLE as ``[0, 0, 0, 0]`` rather than refusing -- which is a box of
    zero size, not the absence of one, and COCO must not be told otherwise.
    """
    from tda.core.export.coco import mask_bbox_xywh

    assert mask_bbox_xywh(encode_rle(np.zeros(HW, dtype=bool))) is None
    assert mask_bbox_xywh(None) is None

    real = np.zeros(HW, dtype=bool)
    real[4:9, 2:8] = True
    assert mask_bbox_xywh(encode_rle(real)) == [2, 4, 6, 5]


def test_the_measurements_match_a_decoded_mask():
    """Reading them off the run lengths has to give the same numbers as before."""
    from tda.core import masks
    from tda.core.export.coco import bbox_xywh, mask_bbox_xywh

    shape = np.zeros(HW, dtype=bool)
    shape[10:30, 5:25] = True
    shape[12:14, 40:44] = True  # a second blob, so the box is not the first one
    rle = encode_rle(shape)

    assert mask_bbox_xywh(rle) == bbox_xywh(masks.bbox(masks.decode_rle(rle)))
    assert masks.rle_area(rle) == masks.area(masks.decode_rle(rle))


# --------------------------------------------------------------------------- #
# standing conflicts
# --------------------------------------------------------------------------- #
def _queue_conflict(d: Db, step: int = 1) -> int:
    """One open disagreement about ``step``, the way a re-check would leave it."""
    return d.add_conflict(FrameKey(DESKTOP, step, VIEW), PSU,
                          encode_rle(PSU_MASK), encode_rle(SCREW_MASK), 400)


# --------------------------------------------------------------------------- #
# a draft must not reach an export by any road
# --------------------------------------------------------------------------- #
DRAFT_SCREW = "ls:Screw#7"


def _add_draft_screw(d: Db) -> None:
    """A draft carrying the *same* class and role as the real screw.

    ``ls:Motherboard#1`` never reaches the counting path; a screw does, because
    the count is read off the instance table rather than off the frame.
    """
    d.upsert_instance(InstanceRec(
        key=DRAFT_SCREW, desktop=DESKTOP, cls="screw",
        attrs={"role": "motherboard", "head": "PH2", "captive": False},
        raw_names=["Motherboard Screw"],
    ))


def test_a_draft_screw_is_counted_by_nothing_and_named_by_nothing(
    db, tax, tmp_path: Path
):
    """The one loop that walked the instance table without the choke point.

    "How many motherboard screws are still fastened?" is answered off the state
    machine, so a draft the compiler never puts in a frame still entered the
    count -- 2 where the truth is 1, and 1 after the real screw came out where
    the truth is 0 -- and ``ls:Screw#7`` appeared in the rationale as a
    ``propagate_state`` step.
    """
    _add_draft_screw(db)
    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    records = _records(out)

    counts = {r["step"]: r["answer"]["count"] for r in records
              if r["task"] == "V2" and "count" in r["answer"]}
    # only step 1 can be asked (the question needs a screw this view can point
    # at), and the answer there is the one real motherboard screw, not two
    assert counts == {1: 1}
    assert "ls:" not in out.read_text(encoding="utf-8")


def test_no_road_through_either_export_reaches_a_draft(db, tax, tmp_path: Path):
    """V1, V2 states, V2 counts, V3 and COCO, with a draft of a real class."""
    _add_draft_screw(db)

    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      only_verified=False, include_boxes=True)
    assert "ls:" not in json.dumps(doc)

    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    assert {r["task"] for r in _records(out)} >= {"V1", "V2", "V3"}
    assert "ls:" not in out.read_text(encoding="utf-8")


def test_the_desktop_context_drops_drafts_however_it_is_built(db, tax):
    """The filtering is the type's, not one call site's."""
    _add_draft_screw(db)
    ctx = load_ctx(db, tax, DESKTOP, VIEW)

    assert DRAFT_SCREW not in ctx.instances
    assert DRAFT_SCREW not in ctx.state_at(1)
    assert ctx.cls_of(DRAFT_SCREW) is None
    assert SCREW in ctx.instances
    assert sorted(ctx.drafts) == [DRAFT_SCREW]


def test_the_vlm_export_stamps_the_graph_version_when_the_tool_can_say(
    db, tax, tmp_path: Path, monkeypatch
):
    """The field has always been in the schema so it could be filled in later."""
    from tda.core import graph as graph_mod
    from tda.core.export.vlm import graph_version_of

    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    assert {r["graph_version"] for r in _records(out)} == {None}

    seen: list[int] = []
    monkeypatch.setattr(graph_mod, "graph_version",
                        lambda _db, desktop: (seen.append(desktop), "abc123")[1],
                        raising=False)
    assert graph_version_of(db, DESKTOP) == "abc123"

    export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    assert {r["graph_version"] for r in _records(out)} == {"abc123"}
    assert seen == [DESKTOP, DESKTOP]


def test_open_conflicts_is_the_services_own_question(db, tax):
    from tda.core.truth import TruthService

    service = TruthService(db, tax)
    assert service.open_conflicts(DESKTOP, VIEW) == []

    cid = _queue_conflict(db)

    assert [c["id"] for c in service.open_conflicts(DESKTOP, VIEW)] == [cid]
    assert service.open_conflicts(DESKTOP, "oak1") == []
    assert service.ensure_fresh(DESKTOP, VIEW)["open_conflicts"] == 1
    db.resolve_conflict(cid, "accept_new")
    assert service.ensure_fresh(DESKTOP, VIEW)["open_conflicts"] == 0


def test_both_exports_refuse_a_view_with_a_standing_conflict(db, tax, tmp_path: Path):
    """A disagreement nobody settled is not a truth anybody may publish."""
    _queue_conflict(db)

    with pytest.raises(RuntimeError, match="allow_conflicts"):
        export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                    only_verified=False)
    with pytest.raises(RuntimeError, match="allow_conflicts"):
        export_vlm(db, tax, [DESKTOP], VIEW, str(tmp_path / "v.jsonl"))
    assert not (tmp_path / "c.json").exists()


def test_allow_conflicts_exports_those_frames_as_unverified(db, tax, tmp_path: Path):
    _queue_conflict(db, step=1)  # step 1 is the confirmed frame

    doc = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      only_verified=False, allow_conflicts=True)

    by_step = {im["extra"]["step"]: im for im in doc["images"]}
    assert by_step[1]["verified"] is False  # confirmed, but still argued about
    assert doc["info"]["allow_conflicts"] is True
    assert doc["info"]["open_conflicts"] == 1

    # ... and so does every annotation of that frame: a row on a disputed frame
    # is not a confirmed row, whatever its own status column still says
    disputed = by_step[1]["id"]
    on_one = [a for a in doc["annotations"] if a["image_id"] == disputed]
    assert on_one and not [a for a in on_one if a["attributes"]["verified"]]
    elsewhere = [a for a in doc["annotations"] if a["image_id"] != disputed]
    assert elsewhere  # the rest of the view is untouched

    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), allow_conflicts=True)
    at_one = [r for r in _records(out) if r["step"] == 1]
    assert at_one and not [r for r in at_one if r["verified"]]
