"""``tier`` and ``verified``: two fields, because they are two facts.

The exports carried one ``quality`` field holding ``gold``/``silver``/``auto``,
which collided with the spec's own vocabulary: there ``gold``/``silver``/
``bronze`` is the **view's** tier (spec §8.1 -- scanner and OAK1 gold, OAK2
silver, RealSense bronze) and confirmation is a separate thing entirely. A
consumer filtering on ``quality == "gold"`` would have got "a human checked
this" from a word that means "this view is annotated to the highest standard".

So every record and every annotation now carries both: ``tier`` from the view
(one table in ``configs/taxonomy.yaml``, never a literal in code) and
``verified``, a bool -- the frame confirmed and every evidence row with it.
``only_verified`` filters on ``verified``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_export import (  # noqa: F401  (re-used fixtures and constants)
    DESKTOP,
    PSU,
    PSU_MASK,
    SCREW,
    VIEW,
    _records,
    db,
    tax,
)

from tda.core.export import export_coco, export_vlm
from tda.core.export.coco import image_id, view_tier
from tda.core.masks import encode_rle
from tda.core.model import FrameKey, InstanceRec
from tda.core.taxonomy import load_taxonomy


# --------------------------------------------------------------------------- #
# the tier is the view's, and it comes from the config
# --------------------------------------------------------------------------- #
def test_every_view_has_a_tier_in_the_taxonomy():
    tax = load_taxonomy()
    assert tax.view_tiers == {"scan": "gold", "oak1": "gold",
                              "oak2": "silver", "rs": "bronze"}


@pytest.mark.parametrize("view,want", [
    ("scan", "gold"), ("oak1", "gold"), ("oak2", "silver"), ("rs", "bronze"),
])
def test_view_tier_reads_the_table(view: str, want: str):
    assert view_tier(view, load_taxonomy()) == want


def test_an_unknown_view_is_a_clear_error_not_a_silent_none():
    """A view the configuration does not describe is a configuration problem.

    It used to come back as ``None`` and travel into every record of the export
    as ``"tier": null`` -- a whole view's worth of data with no tier on it, and
    nothing said.
    """
    with pytest.raises(KeyError) as err:
        view_tier("webcam", load_taxonomy())
    assert "webcam" in str(err.value) and "view_tiers" in str(err.value)


def test_a_new_view_fails_the_export_rather_than_silently_untiered(db, tax, tmp_path: Path):
    with pytest.raises(KeyError):
        export_vlm(db, tax, [DESKTOP], "webcam", str(tmp_path / "v.jsonl"))


# --------------------------------------------------------------------------- #
# VLM
# --------------------------------------------------------------------------- #
def test_every_vlm_record_carries_the_views_tier(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    records = _records(out)
    assert records
    assert {r["tier"] for r in records} == {"gold"}  # VIEW is "scan"
    assert all("quality" not in r for r in records)


def test_a_fully_verified_frame_is_verified(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    first = [r for r in _records(out) if r["step"] == 1]
    assert first and first[0]["verified"] is True


def test_one_unverified_row_takes_the_confirmation_off_the_frame(db, tax, tmp_path: Path):
    db.upsert_instance(InstanceRec(key="chassis.01", desktop=DESKTOP, cls="chassis"))
    db.put_compiled(FrameKey(DESKTOP, 1, VIEW), "chassis.01", encode_rle(PSU_MASK),
                    0.0, "visible", "in_chassis", "auto", "h9")
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    first = [r for r in _records(out) if r["step"] == 1]
    assert first and first[0]["verified"] is False
    assert first[0]["tier"] == "gold"  # the tier is about the view, not the review


def test_an_answer_with_no_evidence_rows_follows_its_frame():
    """"How many screws are still fastened?" when the answer is none.

    It is read off the state machine, not off a box, so it cites no rows at all
    -- and "every one of no rows is verified" is true. Requiring a non-empty
    evidence list dropped every such fact out of a verified export, which is the
    opposite of what those exports are for.
    """
    from tda.core.export.vlm import _verified

    assert _verified([], True) is True
    assert _verified([], False) is False
    assert _verified([{"status": "verified"}], True) is True
    assert _verified([{"status": "auto"}], True) is False


def test_a_counting_answer_of_zero_is_exported_as_verified(db, tax, tmp_path: Path):
    """The whole frame is signed off; the count of visible screws is 0."""
    key = FrameKey(DESKTOP, 2, VIEW)
    db.put_compiled(key, SCREW, encode_rle(PSU_MASK), 0.0, "visible", "in_chassis",
                    "verified", "h2", verified_by="tester")
    db.upsert_frame(key, "F:/scan/019/002/P_0.png", {"hw": [64, 64]},
                    "2025-05-31T10:00:00", flags={"review_status": "verified"})
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V2",))
    counts = [r for r in _records(out) if r["step"] == 2 and "count" in r["answer"]]
    assert counts and counts[0]["answer"]["count"] == 0
    assert counts[0]["evidence"]["instances"] == []
    assert counts[0]["verified"] is True


def test_only_verified_keeps_exactly_the_confirmed_records(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), only_verified=True)
    records = _records(out)
    assert records and all(r["verified"] is True for r in records)
    assert {r["step"] for r in records} == {1, 2}


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
def _annotations(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["annotations"]


def _images(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["images"]


def test_every_coco_image_says_whether_the_frame_was_confirmed(db, tax, tmp_path: Path):
    """A consumer picking whole frames should not have to read every annotation."""
    out = tmp_path / "coco.json"
    export_coco(db, tax, [DESKTOP], VIEW, str(out))
    images = {img["id"]: img for img in _images(out)}
    assert images
    for img in images.values():
        assert isinstance(img["verified"], bool)
        assert "review_status" in img["extra"]  # kept for whoever reads it
    assert images[image_id(DESKTOP, 1, VIEW)]["verified"] is True
    assert {img["tier"] for img in images.values()} == {"gold"}


def test_a_coco_image_is_unverified_once_any_of_its_rows_is(db, tax, tmp_path: Path):
    db.upsert_instance(InstanceRec(key="chassis.01", desktop=DESKTOP, cls="chassis"))
    db.put_compiled(FrameKey(DESKTOP, 1, VIEW), "chassis.01", encode_rle(PSU_MASK),
                    0.0, "visible", "in_chassis", "auto", "h9")
    out = tmp_path / "coco.json"
    export_coco(db, tax, [DESKTOP], VIEW, str(out))
    images = {img["id"]: img for img in _images(out)}
    assert images[image_id(DESKTOP, 1, VIEW)]["verified"] is False


def test_every_coco_annotation_carries_the_tier_and_the_confirmation(
    db, tax, tmp_path: Path
):
    out = tmp_path / "coco.json"
    export_coco(db, tax, [DESKTOP], VIEW, str(out))
    rows = _annotations(out)
    assert rows
    for row in rows:
        attrs = row["attributes"]
        assert attrs["tier"] == "gold"
        assert isinstance(attrs["verified"], bool)
        assert "quality" not in attrs
    # step 1 is the frame this scene signed off
    assert all(a["attributes"]["verified"] for a in rows
               if a["image_id"] == image_id(DESKTOP, 1, VIEW))


def test_an_unconfirmed_row_is_not_verified(db, tax, tmp_path: Path):
    """`only_verified=False` exports the derived rows too, and says what they are."""
    out = tmp_path / "coco.json"
    export_coco(db, tax, [DESKTOP], VIEW, str(out))
    verified_ids = {image_id(DESKTOP, 1, VIEW)}
    for row in _annotations(out):
        assert row["attributes"]["verified"] is (row["image_id"] in verified_ids)


def test_a_coco_annotation_says_whether_its_instance_was_implied(db, tax, tmp_path: Path):
    db.upsert_instance(InstanceRec(key=PSU, desktop=DESKTOP, cls="psu",
                                   attrs={"implied": True}))
    out = tmp_path / "coco.json"
    export_coco(db, tax, [DESKTOP], VIEW, str(out))
    by_instance = {a["attributes"]["instance_key"]: a["attributes"]
                   for a in _annotations(out)}
    assert by_instance[PSU]["implied"] is True
    assert by_instance[SCREW]["implied"] is False
