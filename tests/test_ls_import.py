"""Tests for the Label Studio draft importer (tda.core.ls_import)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tda.core.db import Db
from tda.core.ls_import import (
    LS_LABEL_MAP_PATH,
    NATIVE_HW,
    import_ls_export,
    load_label_map,
    ls_ellipse_to_mask,
    ls_polygon_to_mask,
    ls_rect_to_mask,
    ls_reference_masks,
    parse_task_image,
)
from tda.core.masks import area, bbox, decode_rle
from tda.core.model import FrameKey
from tda.core.taxonomy import load_taxonomy

FIXTURE = Path(__file__).parent / "fixtures" / "ls_small.json"

# The 40 distinct labels used by the real export, with their occurrence counts.
EXPORT_LABELS = {
    "Motherboard Screw": 2227,
    "Case to Motherboard Connector": 2050,
    "RAM Module Retention Clip (open)": 1519,
    "RAM Module Retention Clip (closed)": 1017,
    "Motherboard": 528,
    "PSU to Motherboard Connector": 492,
    "RAM Module": 439,
    "SATA Connector": 380,
    "Target": 359,
    "Power Supply Unit (PSU)": 356,
    "CPU Socket Retention Bracket": 291,
    "Chassis Screw Cover on Case": 275,
    "PSU Retention Bracket": 263,
    "Additional Card Upper Cover": 224,
    "CPU Chip": 203,
    "Heatsink Screw": 189,
    "CPU Cooling Fan Screw": 152,
    "Optical Drive": 141,
    "Drive Power Connector": 132,
    "GPU": 128,
    "CPU Fan Connector": 123,
    "CPU Cooling Fan": 96,
    "Heatsink": 80,
    "Storage Drive Retention Bracket": 77,
    "Solid State Drive (SSD)": 60,
    "Connector": 53,
    "Frame Screw (NRS)": 32,
    "CPU Fan Retention Bracket": 27,
    "Storage Drive Retention Bracket Locker (open)": 17,
    "Screw Locker (Big Screw)": 16,
    "PH2 Screw": 12,
    "HDD Retention Bracket": 12,
    "Chassis Screw Cover": 12,
    "Storage Drive Retention Bracket Locker (closed)": 8,
    "T15 Screw": 7,
    "Hard Disk Drive (HDD)": 6,
    "Screw Locker": 6,
    "RAM Module Slot": 3,
    "Frame Screw": 3,
    "Motherboard PCB": 1,
}

# Labels the map is expected to drop (they are no removable instance).
DROPPED_LABELS = {"RAM Module Slot", "Motherboard PCB"}


# --------------------------------------------------------------------------- #
# module layout
# --------------------------------------------------------------------------- #
def test_the_brief_interface_is_reachable_from_ls_import():
    import tda.core.ls_import as mod

    for name in ("parse_task_image", "ls_polygon_to_mask", "import_ls_export",
                 "ls_reference_masks"):
        assert callable(getattr(mod, name)), name
        assert name in mod.__all__, name


def test_the_export_reader_stays_database_free():
    """`ls_export` is the half that must keep working without a database."""
    src = (Path(__file__).parents[1] / "tda" / "core" / "ls_export.py").read_text(
        encoding="utf-8")
    for forbidden in ("from tda.core.db", "import sqlite3", "from tda.core.index"):
        assert forbidden not in src, forbidden


# --------------------------------------------------------------------------- #
# parse_task_image
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, expected",
    [
        # the four patterns named in the brief
        ("3e99d760-Rs_13_42.png", (13, 42, "rs")),
        ("b419c37c-Front_OAK_13_42.jpg", (13, 42, "oak1")),
        ("7972a2f3-Side_OAK_24_39.jpg", (24, 39, "oak2")),
        ("a6f035f4-33_39.png", (33, 39, "scan")),
        # patterns that also occur in the real export
        ("f0f0f0f0-ctrl_desktop_data_10__42.png", (10, 42, "scan")),
        ("00112233-Front_OAK_Align_33_29.png", (33, 29, "oak1")),
        ("44556677-Side_OAK_Align_13_42.png", (13, 42, "oak2")),
        # leading zeros in the step token
        ("71d57f86-Front_OAK_24_05.jpg", (24, 5, "oak1")),
        # no upload prefix at all
        ("Rs_13_42.png", (13, 42, "rs")),
        # a full storage URL, not just the basename
        ("/storage-data/uploaded/?filepath=upload/195079/e5a70a47-Rs_24_35.png", (24, 35, "rs")),
        # unusable: carries no desktop id
        ("P_0.png", None),
        ("P_1.png", None),
        # not an image name at all
        ("", None),
        ("Metadata", None),
        ("13_42.txt", None),
    ],
)
def test_parse_task_image(name, expected):
    assert parse_task_image(name) == expected


# --------------------------------------------------------------------------- #
# percent -> pixel rasterisation
# --------------------------------------------------------------------------- #
def test_polygon_percent_to_pixels_area():
    """A 10% x 10% square of a 1600x1600 frame covers ~160x160 = 25,600 px."""
    square = [[20.0, 30.0], [30.0, 30.0], [30.0, 40.0], [20.0, 40.0]]
    mask = ls_polygon_to_mask(square, NATIVE_HW["scan"])
    assert mask.shape == (1600, 1600)
    assert mask.dtype == bool
    assert area(mask) == pytest.approx(25600, rel=0.02)
    assert bbox(mask) == pytest.approx((320, 480, 481, 641), abs=2)


def test_polygon_percent_is_resolution_independent():
    """The same percent polygon covers the same *fraction* of any frame.

    The tolerance is the boundary pixel that ``fillPoly`` includes on each
    edge: on the 720-px-tall RealSense frame a 10% band is 72 px, so one extra
    row and column is already 2% of the area.
    """
    square = [[20.0, 30.0], [30.0, 30.0], [30.0, 40.0], [20.0, 40.0]]
    for view, (h, w) in NATIVE_HW.items():
        mask = ls_polygon_to_mask(square, NATIVE_HW[view])
        frac = area(mask) / (h * w)
        assert frac == pytest.approx(0.01, rel=0.03), view
        # every percent coordinate is rounded to its own pixel, independently
        assert bbox(mask) == (
            round(0.2 * w), round(0.3 * h), round(0.3 * w) + 1, round(0.4 * h) + 1
        ), view


def test_polygon_degenerate_is_empty():
    assert area(ls_polygon_to_mask([[1.0, 1.0], [2.0, 2.0]], (100, 100))) == 0
    assert area(ls_polygon_to_mask([], (100, 100))) == 0


def test_ellipse_percent_to_pixels_area():
    """radiusX/radiusY are percents of width/height: pi*r_x*r_y pixels."""
    value = {"x": 50.0, "y": 50.0, "radiusX": 10.0, "radiusY": 5.0, "rotation": 0}
    mask = ls_ellipse_to_mask(value, NATIVE_HW["scan"])
    assert area(mask) == pytest.approx(math.pi * 160 * 80, rel=0.02)
    x0, y0, x1, y1 = bbox(mask)
    assert (x0 + x1) / 2 == pytest.approx(800, abs=3)
    assert (y0 + y1) / 2 == pytest.approx(800, abs=3)
    assert x1 - x0 == pytest.approx(320, abs=3)
    assert y1 - y0 == pytest.approx(160, abs=3)


def test_ellipse_small_radius_still_marks_a_pixel():
    value = {"x": 50.0, "y": 50.0, "radiusX": 0.001, "radiusY": 0.001, "rotation": 0}
    assert area(ls_ellipse_to_mask(value, (100, 100))) >= 1


def test_rect_percent_to_pixels():
    value = {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "rotation": 0}
    mask = ls_rect_to_mask(value, (1000, 2000))  # h=1000, w=2000
    assert bbox(mask) == pytest.approx((200, 200, 801, 601), abs=2)


def test_rect_rotation_is_applied_about_the_corner():
    """A 90 deg rotation swaps the extents and keeps the anchor corner."""
    upright = {"x": 20.0, "y": 20.0, "width": 40.0, "height": 10.0, "rotation": 0}
    turned = {"x": 20.0, "y": 20.0, "width": 40.0, "height": 10.0, "rotation": 90}
    bu = bbox(ls_rect_to_mask(upright, (1000, 1000)))
    bt = bbox(ls_rect_to_mask(turned, (1000, 1000)))
    assert (bu[2] - bu[0], bu[3] - bu[1]) == pytest.approx((401, 101), abs=3)
    assert (bt[2] - bt[0], bt[3] - bt[1]) == pytest.approx((101, 401), abs=3)
    assert area(ls_rect_to_mask(turned, (1000, 1000))) == pytest.approx(400 * 100, rel=0.03)


# --------------------------------------------------------------------------- #
# configs/ls_label_map.yaml
# --------------------------------------------------------------------------- #
def test_label_map_covers_every_export_label():
    mapping = load_label_map()
    assert set(mapping) == set(EXPORT_LABELS), "every LS label needs an explicit entry"
    for label in DROPPED_LABELS:
        assert mapping[label] is None, f"{label} must be mapped to null"


def test_label_map_entries_validate_against_the_taxonomy():
    tax = load_taxonomy()
    mapping = load_label_map()
    for label, entry in mapping.items():
        if entry is None:
            continue
        if entry.special is not None:
            assert entry.special == "action_target_hint"
            assert entry.cls is None
            continue
        assert entry.cls in tax.classes, f"{label} -> unknown class {entry.cls!r}"
        declared = tax.classes[entry.cls].get("attrs") or {}
        for key, value in entry.attrs.items():
            assert key in declared, f"{label}: {entry.cls} has no attr {key!r}"
            allowed = declared[key]
            if allowed:
                assert value in allowed, f"{label}: {key}={value!r} not in {allowed}"
        if entry.state is not None:
            assert entry.state in tax.states_of(entry.cls)


def test_label_map_rejects_an_unknown_class(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nlabels:\n  X:\n    cls: not_a_class\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_class"):
        load_label_map(str(bad))


def test_label_map_path_points_at_the_repo_config():
    assert Path(LS_LABEL_MAP_PATH).name == "ls_label_map.yaml"
    assert Path(LS_LABEL_MAP_PATH).exists()


# --------------------------------------------------------------------------- #
# import_ls_export on the 2-task fixture
# --------------------------------------------------------------------------- #
SCAN_FRAME = FrameKey(19, 52, "scan")
RS_FRAME = FrameKey(24, 35, "rs")


@pytest.fixture
def imported(tmp_db_path):
    db = Db(tmp_db_path)
    summary = import_ls_export(str(FIXTURE), db, load_taxonomy(), {})
    yield db, summary
    db.close()


def test_import_creates_labelstudio_keyframes(imported):
    db, summary = imported
    scan = db.keyframes(19, "scan")
    rs = db.keyframes(24, "rs")
    assert scan and rs
    for kf in scan + rs:
        assert kf.source == "labelstudio"
        assert kf.geom_type == "mask"
        assert kf.placement == "in_chassis"
        assert [p.name for p in kf.parts] == ["main"]
        assert kf.parts[0].rle is not None
    # 13 polygons + 1 ellipse on the scanner frame; the realsense frame has 9
    # polygons of which one is the "Target" hint (not an instance).
    assert len(scan) == 14
    assert len(rs) == 8
    assert {kf.anchor_step for kf in scan} == {52}
    assert {kf.anchor_step for kf in rs} == {35}


def test_import_rasterises_at_native_resolution(imported):
    db, _ = imported
    for kf in db.keyframes(19, "scan"):
        assert kf.parts[0].rle["size"] == [1600, 1600]
    for kf in db.keyframes(24, "rs"):
        assert kf.parts[0].rle["size"] == [720, 1280]


def test_import_uses_provisional_keys_numbered_left_to_right(imported):
    db, _ = imported
    kfs = [kf for kf in db.keyframes(19, "scan")
           if kf.instance.startswith("ls:Case to Motherboard Connector#")]
    assert {kf.instance for kf in kfs} == {
        "ls:Case to Motherboard Connector#1",
        "ls:Case to Motherboard Connector#2",
        "ls:Case to Motherboard Connector#3",
    }
    by_key = {kf.instance: bbox(decode_rle(kf.parts[0].rle))[0] for kf in kfs}
    xs = [by_key[f"ls:Case to Motherboard Connector#{n}"] for n in (1, 2, 3)]
    assert xs == sorted(xs), "numbering must run left to right by bbox x"


def test_import_writes_instance_rows_with_mapped_class_and_attrs(imported):
    db, _ = imported
    insts = db.instances(19)
    assert insts["ls:Motherboard Screw#1"].cls == "screw"
    assert insts["ls:Motherboard Screw#1"].attrs["role"] == "motherboard"
    assert insts["ls:Motherboard Screw#1"].raw_names == ["Motherboard Screw"]
    clip = insts["ls:RAM Module Retention Clip (closed)#1"]
    assert clip.cls == "ram_latch"
    assert clip.attrs["state"] == "closed"
    locker = insts["ls:Screw Locker (Big Screw)#1"]
    assert locker.cls == "screw"
    assert locker.attrs["role"] == "other"


def test_import_stores_text_results_in_step_notes(imported):
    db, _ = imported
    notes = {s.step: s.notes for s in db.steps(24)}
    assert 35 in notes
    line = notes[35]
    assert line.startswith("LS: ") or "\nLS: " in line
    assert "task_desc_value=Unscrew Motherboard screw" in line
    assert "seq_name_value=Unscrew Motherboard T15 Screw" in line
    assert "tool_utility_value=T15 Screwdriver" in line
    assert "disassembly_operation=Unscrew" in line
    step = next(s for s in db.steps(24) if s.step == 35)
    assert step.step_type == "normal"


def test_import_records_the_target_hint_as_a_bbox_in_notes(imported):
    db, _ = imported
    notes = next(s.notes for s in db.steps(24) if s.step == 35)
    marker = "action_target_hint="
    assert marker in notes
    payload = notes.split(marker, 1)[1]
    payload = payload.split("; ", 1)[0].split("\n", 1)[0]
    hints = json.loads(payload)
    assert isinstance(hints, list) and hints
    hint = hints[0]
    assert hint["view"] == "rs"
    x0, y0, x1, y1 = hint["bbox"]
    assert 0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720
    # the Target region is never turned into an instance
    assert not any(k.startswith("ls:Target#") for k in db.instances(24))


def test_import_stores_relations_as_proposed_edges(imported):
    db, summary = imported
    rels = db.relations(24)
    assert len(rels) == 3
    assert summary["relations"] == 3
    for r in rels:
        assert r["source"] == "labelstudio"
        assert r["status"] == "proposed"
        assert r["type"] == "partner_of"
        assert r["target"].startswith("ls:Motherboard Screw#")
        assert r["blocker"].startswith("ls:Motherboard Screw#")
        assert r["evidence_step"] == 35
    assert db.relations(19) == []


def test_import_summary_reports_per_view_and_per_desktop_counts(imported):
    _, summary = imported
    assert summary["frames"] == 2
    assert summary["keyframes"] == 22
    assert summary["tasks_total"] == 2
    assert summary["by_view"]["scan"]["frames"] == 1
    assert summary["by_view"]["scan"]["keyframes"] == 14
    assert summary["by_view"]["rs"]["keyframes"] == 8
    assert summary["by_desktop"][19]["keyframes"] == 14
    assert summary["by_desktop"][24]["keyframes"] == 8
    assert summary["by_desktop"][24]["relations"] == 3
    assert summary["skipped_labels"] == {}
    assert summary["target_hints"] == 1
    assert summary["frames_not_in_index"] == 2
    # one provisional key can span several steps, so the row counts are the
    # distinct rows actually written, not the export items consumed
    assert summary["instances"] == 22
    assert summary["instance_rows"] == 22
    assert summary["relations"] == 3
    assert summary["relation_edges"] == 3


def test_import_is_idempotent(tmp_db_path):
    db = Db(tmp_db_path)
    first = import_ls_export(str(FIXTURE), db, load_taxonomy(), {})
    second = import_ls_export(str(FIXTURE), db, load_taxonomy(), {})
    assert first["keyframes"] == second["keyframes"]
    assert len(db.keyframes(19, "scan")) == 14
    assert len(db.relations(24)) == 3
    notes = next(s.notes for s in db.steps(24) if s.step == 35)
    assert notes.count("LS: ") == 1
    db.close()


def test_import_preserves_pre_existing_step_notes(tmp_db_path):
    from tda.core.model import ActionRec, StepRec

    db = Db(tmp_db_path)
    db.replace_steps(
        24,
        [StepRec(24, 35, "normal", "Unscrew Motherboard T15 Screw 4", notes="hand written")],
        [ActionRec(24, 35, 0, "screw.motherboard.01", "unscrew")],
    )
    import_ls_export(str(FIXTURE), db, load_taxonomy(), {})
    step = next(s for s in db.steps(24) if s.step == 35)
    assert step.notes.startswith("hand written")
    assert "LS: " in step.notes
    assert step.raw_name == "Unscrew Motherboard T15 Screw 4"
    assert len(db.actions(24, 35)) == 1, "existing actions must survive the notes rewrite"
    db.close()


def test_import_flags_the_frames_it_touched(imported):
    db, _ = imported
    assert db.get_frame(SCAN_FRAME)["review_status"] == "ls_draft"
    assert db.get_frame(RS_FRAME)["review_status"] == "ls_draft"


def test_import_uses_the_index_for_frame_paths_and_step_range(tmp_db_path):
    from tda.core.index import DesktopIndex, FrameFile

    ff = FrameFile(key=SCAN_FRAME, path="F:/scan/019/052/P_0.png", aux={"crop": "c.png"},
                   ts="2025-01-01T00:00:00", src_step_dir="052")
    index = {19: DesktopIndex(desktop=19, n_steps=55, frames={SCAN_FRAME: ff})}
    db = Db(tmp_db_path)
    summary = import_ls_export(str(FIXTURE), db, load_taxonomy(), index)
    assert db.get_frame(SCAN_FRAME)["path"] == "F:/scan/019/052/P_0.png"
    assert summary["frames_not_in_index"] == 1  # only the realsense frame is unknown
    db.close()


def test_import_skips_a_task_whose_upload_has_a_different_framing(tmp_path):
    """The `_Align_` OAK uploads are 1280x800, i.e. a different crop: skip them."""
    payload = {
        "exported_at": "2026-09-17T00:00:00Z",
        "source": "test",
        "projects": [{
            "id": 1, "title": "OAK_front_camera", "workspace": "Desktop_33",
            "label_config": "", "tasks": [{
                "id": 9, "data": {"image": "upload/1/00112233-Front_OAK_Align_33_29.png"},
                "annotations": [{"id": 5, "result": [
                    {"id": "aaa", "type": "polygon", "from_name": "polygon",
                     "to_name": "image1", "original_width": 1280, "original_height": 800,
                     "image_rotation": 0,
                     "value": {"closed": True,
                               "points": [[10, 10], [20, 10], [20, 20], [10, 20]]}},
                    {"id": "aaa", "type": "labels", "from_name": "valuable_component_labels",
                     "to_name": "image1", "original_width": 1280, "original_height": 800,
                     "image_rotation": 0, "value": {"labels": ["Motherboard"]}},
                ]}],
            }],
        }],
    }
    src = tmp_path / "align.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    db = Db(str(tmp_path / "db.sqlite"))
    summary = import_ls_export(str(src), db, load_taxonomy(), {})
    assert summary["keyframes"] == 0
    assert summary["skipped_framing"] == 1
    assert summary["skipped_framing_labels"] == 1
    assert db.keyframes(33, "oak1") == []
    db.close()


def test_import_skips_an_align_upload_that_has_no_geometry_to_measure(tmp_path):
    """One `_Align_` task carries only text, so the name must be enough to skip it."""
    payload = {
        "exported_at": "x", "source": "test",
        "projects": [{
            "id": 1, "title": "OAK_side_camera", "workspace": "Desktop_13",
            "tasks": [{
                "id": 9, "data": {"image": "upload/1/44556677-Side_OAK_Align_13_42.png"},
                "annotations": [{"id": 5, "result": [
                    {"id": "t1", "type": "textarea", "from_name": "seq_number_value",
                     "to_name": "seq_number", "value": {"text": ["42"]}},
                ]}],
            }],
        }],
    }
    src = tmp_path / "align_text.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    db = Db(str(tmp_path / "db.sqlite"))
    summary = import_ls_export(str(src), db, load_taxonomy(), {})
    assert summary["skipped_framing"] == 1
    assert summary["frames"] == 0
    assert db.steps(13) == []
    db.close()


def test_import_skips_metadata_projects_and_unparseable_images(tmp_path):
    payload = {
        "exported_at": "x", "source": "test",
        "projects": [
            {"id": 1, "title": "Metadata", "workspace": "Desktop_03", "tasks": [
                {"id": 1, "data": {"$undefined$": True},
                 "annotations": [{"id": 1, "result": []}]}]},
            {"id": 2, "title": "Scanner_camera", "workspace": "Desktop_04", "tasks": [
                {"id": 2, "data": {"image": "upload/2/aabbccdd-P_0.png"},
                 "annotations": [{"id": 2, "result": []}]}]},
        ],
    }
    src = tmp_path / "meta.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    db = Db(str(tmp_path / "db.sqlite"))
    summary = import_ls_export(str(src), db, load_taxonomy(), {})
    assert summary["frames"] == 0
    assert summary["skipped_metadata_projects"] == 1
    assert summary["skipped_unparseable"] == 1
    db.close()


def _scan_task(task_id: int, step: int, label: str) -> dict:
    return {
        "id": task_id, "data": {"image": f"upload/1/aabbccdd-19_{step}.png"},
        "annotations": [{"id": task_id, "result": [
            {"id": f"g{task_id}", "type": "polygon", "from_name": "polygon",
             "to_name": "image1", "original_width": 1600, "original_height": 1600,
             "value": {"points": [[10, 10], [20, 10], [20, 20], [10, 20]]}},
            {"id": f"g{task_id}", "type": "labels",
             "from_name": "valuable_component_labels", "to_name": "image1",
             "original_width": 1600, "original_height": 1600,
             "value": {"labels": [label]}},
        ]}],
    }


def test_one_provisional_key_spans_the_steps_it_appears_on(tmp_path):
    """The same label at the same rank is one instance row with many keyframes."""
    payload = {
        "exported_at": "x", "source": "test",
        "projects": [{
            "id": 1, "title": "Scanner_camera", "workspace": "Desktop_19",
            "tasks": [_scan_task(1, 7, "Motherboard"), _scan_task(2, 8, "Motherboard")],
        }],
    }
    src = tmp_path / "two_steps.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    db = Db(str(tmp_path / "db.sqlite"))
    summary = import_ls_export(str(src), db, load_taxonomy(), {})
    assert summary["keyframes"] == 2
    assert summary["instances"] == 2  # export items consumed
    assert summary["instance_rows"] == 1  # rows written
    kfs = db.keyframes(19, "scan", "ls:Motherboard#1")
    assert sorted(kf.anchor_step for kf in kfs) == [7, 8]
    assert len({kf.draft_id for kf in kfs}) == 2, "each shape keeps its own LS result id"
    db.close()


def test_import_counts_labels_it_dropped(tmp_path):
    payload = {
        "exported_at": "x", "source": "test",
        "projects": [{
            "id": 1, "title": "Scanner_camera", "workspace": "Desktop_19",
            "tasks": [{
                "id": 1, "data": {"image": "upload/1/aabbccdd-19_7.png"},
                "annotations": [{"id": 1, "result": [
                    {"id": "b1", "type": "polygon", "from_name": "polygon",
                     "to_name": "image1", "original_width": 1600, "original_height": 1600,
                     "value": {"points": [[10, 10], [20, 10], [20, 20]]}},
                    {"id": "b1", "type": "labels", "from_name": "valuable_component_labels",
                     "to_name": "image1", "original_width": 1600, "original_height": 1600,
                     "value": {"labels": ["RAM Module Slot"]}},
                ]}],
            }],
        }],
    }
    src = tmp_path / "drop.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    db = Db(str(tmp_path / "db.sqlite"))
    summary = import_ls_export(str(src), db, load_taxonomy(), {})
    assert summary["keyframes"] == 0
    assert summary["skipped_labels"] == {"RAM Module Slot": 1}
    db.close()


# --------------------------------------------------------------------------- #
# ls_reference_masks
# --------------------------------------------------------------------------- #
def test_reference_masks_yields_frame_label_and_mask():
    out = list(ls_reference_masks(str(FIXTURE), "scan"))
    assert len(out) == 14
    for key, label, mask in out:
        assert key == SCAN_FRAME
        assert label in EXPORT_LABELS
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert mask.shape == NATIVE_HW["scan"]
        assert mask.any()
    assert sum(1 for _, lab, _ in out if lab == "RAM Module Retention Clip (closed)") == 5


def test_reference_masks_keeps_the_target_label_for_the_other_view():
    out = list(ls_reference_masks(str(FIXTURE), "rs"))
    assert len(out) == 9
    assert sum(1 for _, lab, _ in out if lab == "Target") == 1
    assert all(k == RS_FRAME for k, _, _ in out)


def test_reference_masks_needs_no_db_and_rejects_an_unknown_view():
    with pytest.raises(ValueError, match="view"):
        list(ls_reference_masks(str(FIXTURE), "oak3"))
