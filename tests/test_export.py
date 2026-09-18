"""Tests for the rehearsal exports: COCO instance segmentation + VLM JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

from tda.core.db import Db
from tda.core.export import export_coco, export_vlm
from tda.core.export.coco import cache_rel_path, categories, image_id
from tda.core.masks import encode_rle
from tda.core.model import (
    ActionRec,
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StepRec,
)
from tda.core.taxonomy import load_taxonomy

DESKTOP = 1
VIEW = "scan"
HW = (64, 64)
PSU = "psu.01"
SCREW = "screw.motherboard.03"


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
    d.replace_steps(
        DESKTOP,
        [
            StepRec(DESKTOP, 1, "initial", "initial"),
            StepRec(DESKTOP, 2, "normal", "motherboard screw 3"),
            StepRec(DESKTOP, 3, "normal", "psu"),
        ],
        [
            ActionRec(DESKTOP, 2, 0, SCREW, "unscrew", tool="PH2", direction="+Z"),
            ActionRec(DESKTOP, 3, 0, PSU, "remove", tool="hand", direction="+Z"),
        ],
    )
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
    # step 3: the psu lies on the bench -- box geometry only, so no visible RLE.
    d.put_compiled(FrameKey(DESKTOP, 3, VIEW), PSU, None, 0.0, "visible",
                   "on_bench", "auto", "h3")
    d.add_keyframe(ShapeKeyframe(
        id=None, instance=PSU, desktop=DESKTOP, view=VIEW, pose_segment=0, anchor_step=2,
        placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", rle=psu_rle)], amodal_complete=False,
    ))
    d.add_keyframe(ShapeKeyframe(
        id=None, instance=PSU, desktop=DESKTOP, view=VIEW, pose_segment=0, anchor_step=3,
        placement="on_bench", geom_type="box",
        parts=[ShapePart("main", box=(2.0, 3.0, 12.0, 15.0))], amodal_complete=True,
    ))
    yield d
    d.close()


def _with_roi(d: Db) -> None:
    """Give the view a pose segment carrying an ROI (no Db writer for roi_json yet)."""
    d.set_pose_segment(DESKTOP, VIEW, 0, 1, 3, 1, None, None)
    with d.conn:
        d.conn.execute(
            "UPDATE pose_segment SET roi_json=? WHERE desktop=? AND view=? AND seg=?",
            (json.dumps(ROI), DESKTOP, VIEW, 0),
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
    assert {a["attributes"]["quality"] for a in strict["annotations"]} == {"gold"}

    full = export_coco(db, tax, [DESKTOP], VIEW, str(tmp_path / "all.json"),
                       only_verified=False)
    assert [im["extra"]["step"] for im in full["images"]] == [1, 2, 3]
    assert len(full["annotations"]) == 3  # the bench row carries no mask
    assert {a["attributes"]["quality"] for a in full["annotations"]} == {"gold", "auto"}


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
        "quality": "gold",
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
    assert bench["bbox"] == [2.0, 3.0, 10.0, 12.0]
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
# VLM
# --------------------------------------------------------------------------- #
def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_vlm_writes_at_least_one_record_per_task(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    summary = export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    records = _records(out)

    assert summary["records"] == len(records)
    assert all(summary["by_task"][t] >= 1 for t in ("V1", "V2", "V3"))
    assert len({r["id"] for r in records}) == len(records)
    for rec in records:
        assert set(rec) >= {"id", "task", "desktop", "step", "view", "images",
                            "question", "answer", "evidence", "graph_version"}
        assert rec["desktop"] == DESKTOP and rec["view"] == VIEW
        assert rec["graph_version"] is None
        assert rec["question"] and isinstance(rec["question"], str)
        assert json.loads(json.dumps(rec["answer"])) == rec["answer"]
        assert all(img.startswith("scan/D01/s") for img in rec["images"])


def test_vlm_v1_lists_the_visible_components(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    first = [r for r in _records(out) if r["step"] == 1]
    assert len(first) == 1
    components = first[0]["answer"]["components"]
    assert {c["instance"] for c in components} == {PSU, SCREW}
    psu = next(c for c in components if c["instance"] == PSU)
    assert psu["class"] == "psu" and psu["bbox"] == [5, 10, 20, 20]
    assert first[0]["evidence"]["bboxes"][SCREW] == [50, 40, 8, 8]
    # step 3 holds a bench row without a mask: nothing to point at
    assert [r["step"] for r in _records(out)] == [1, 2]


def test_vlm_v2_asks_states_and_counts(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V2",))
    records = _records(out)
    states = [r for r in records if "state" in r["answer"]]
    counts = [r for r in records if "count" in r["answer"]]

    screw = next(r for r in states if r["evidence"]["instances"] == [SCREW])
    assert screw["step"] == 1
    assert screw["answer"] == {"state": "fastened"}
    assert "motherboard screw 3" in screw["question"]
    assert len(counts) == 1 and counts[0]["step"] == 1
    assert counts[0]["answer"] == {"count": 1}
    assert counts[0]["question"] == "How many motherboard screws are still fastened?"


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
