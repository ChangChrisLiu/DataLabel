"""VLM export: what ``gold`` promises, and what a label reads like.

``quality`` used to be read off the single compiled row an answer happened to
cite, so a frame with one verified screw exported "gold" answers about an
otherwise unreviewed image. Gold now means the *frame* is verified -- the truth
service's own notion: a human signed the frame off, or every compiled row of it
is frozen -- **and** every row the answer was read off is. Everything else is
``silver``; nothing the truth table produces is unreviewed enough to be worse.

And the questions are read by people (and by a model that answers in words), so
``instance_label`` says "PSU 1" and "RAM module 2", not "psu 1". The structured
fields keep the machine ids.
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

from tda.core.export import export_vlm
from tda.core.export.vlm import instance_label
from tda.core.masks import encode_rle
from tda.core.model import FrameKey, InstanceRec
from tda.core.taxonomy import Taxonomy, load_taxonomy


# --------------------------------------------------------------------------- #
# wording
# --------------------------------------------------------------------------- #
CASES = (
    ("psu.01", "psu", {}, "PSU 1"),
    ("cpu_cooler.01", "cpu_cooler", {}, "CPU cooler 1"),
    ("ram_module.02", "ram_module", {}, "RAM module 2"),
    ("motherboard.01", "motherboard", {}, "Motherboard 1"),
    ("storage_drive.01", "storage_drive", {}, "Storage drive 1"),
    ("cpu_socket_lever.01", "cpu_socket_lever", {}, "CPU socket lever 1"),
    ("screw.motherboard.03", "screw", {"role": "motherboard"}, "Motherboard screw 3"),
    ("expansion_card.01", "expansion_card", {"kind": "gpu"}, "GPU expansion card 1"),
    ("cpu_cooler.fan.02", "cpu_cooler", {"kind": "fan"}, "Fan CPU cooler 2"),
)


@pytest.mark.parametrize("key,cls,attrs,want", CASES, ids=[c[3] for c in CASES])
def test_a_label_reads_the_way_a_person_says_it(key, cls, attrs, want):
    tax = load_taxonomy()
    rec = InstanceRec(key=key, desktop=1, cls=cls, attrs=dict(attrs))
    assert instance_label(key, rec, tax) == want


def test_a_label_without_a_record_is_still_the_key():
    assert instance_label("ls:Motherboard#1", None) == "ls:Motherboard#1"


def test_a_taxonomy_label_wins_over_the_derived_one():
    """``taxonomy.yaml`` has no ``label`` field today; if it gains one, it rules."""
    shared = load_taxonomy()
    # a copy: load_taxonomy() is cached and hands every caller the same object
    own = Taxonomy(
        classes={**shared.classes, "psu": {**shared.classes["psu"],
                                           "label": "power supply"}},
        verbs=shared.verbs, tools=shared.tools,
    )
    rec = InstanceRec(key="psu.01", desktop=1, cls="psu")
    assert instance_label("psu.01", rec, own) == "Power supply 1"


def test_the_question_carries_the_readable_label(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V2",))
    states = [r for r in _records(out) if "-state-" in r["id"]]
    assert states
    assert any("PSU 1" in r["question"] for r in states)


def test_the_structured_fields_keep_the_ids(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    components = [c for r in _records(out) for c in r["answer"]["components"]]
    assert components
    assert {c["instance"] for c in components} <= {PSU, SCREW}
    assert {c["class"] for c in components} <= {"psu", "screw"}


# --------------------------------------------------------------------------- #
# gold
# --------------------------------------------------------------------------- #
def test_a_fully_verified_frame_is_gold(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    first = [r for r in _records(out) if r["step"] == 1]
    assert first and first[0]["quality"] == "gold"


def test_one_unverified_row_takes_the_gold_off_the_whole_frame(db, tax, tmp_path: Path):
    db.put_compiled(FrameKey(DESKTOP, 1, VIEW), "chassis.01", encode_rle(PSU_MASK),
                    0.0, "visible", "in_chassis", "auto", "h9")
    db.upsert_instance(InstanceRec(key="chassis.01", desktop=DESKTOP, cls="chassis"))
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    first = [r for r in _records(out) if r["step"] == 1]
    assert first and first[0]["quality"] == "silver"


def test_the_frames_own_verified_flag_is_enough(db, tax, tmp_path: Path):
    """A frame a human signed off is verified even where a row still says auto."""
    db.upsert_frame(FrameKey(DESKTOP, 2, VIEW), "F:/scan/019/002/P_0.png",
                    {"hw": [64, 64]}, "2025-05-31T10:00:00",
                    flags={"review_status": "verified"})
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    second = [r for r in _records(out) if r["step"] == 2]
    # the frame is verified, but the row the answer was read off is not
    assert second and second[0]["quality"] == "silver"


def test_the_non_gold_grade_is_silver(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out))
    grades = {r["quality"] for r in _records(out)}
    assert grades <= {"gold", "silver"}
    assert "auto" not in grades


def test_only_verified_still_keeps_exactly_the_gold_records(db, tax, tmp_path: Path):
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, tax, [DESKTOP], VIEW, str(out), only_verified=True)
    records = _records(out)
    assert records and {r["quality"] for r in records} == {"gold"}
