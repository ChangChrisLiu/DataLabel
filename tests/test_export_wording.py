"""VLM export: what a question reads like.

The questions are read by people -- and answered by a model in words -- so
``instance_label`` says "PSU 1" and "RAM module 2", not "psu 1". The structured
fields keep the machine ids.

What ``tier`` and ``verified`` mean is pinned next door in
``test_export_tier.py``.
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
