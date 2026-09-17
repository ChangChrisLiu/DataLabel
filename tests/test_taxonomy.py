"""Tests for the taxonomy vocabulary layer (spec section 6)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tda.core.taxonomy import load_taxonomy, map_tool, parse_raw_name

REPO_ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("CPU fan screw 3", "", dict(cls="screw", role="cpu_cooler", n=3)),
    ("Heatsink screw 2", "", dict(cls="screw", role="cpu_cooler", n=2)),
    ("Motherboard screw 6", "", dict(cls="screw", role="motherboard", n=6)),
    ("RAM clip 1", "", dict(cls="ram_latch", n=1)),
    ("RAM1", "", dict(cls="ram_module", n=1)),
    ("CPU locker", "", dict(cls="cpu_socket_lever")),
    ("Power module", "", dict(cls="psu")),
    ("Case - motherboard connector 2", "", dict(cls="connector", kind="front_panel", n=2)),
    ("Power module - motherboard connector 1", "", dict(cls="connector", cable_owner="psu", n=1)),
    ("Connector 3", "Power Supply Unit (PSU)", dict(cls="connector", cable_owner="psu", n=3)),
    ("SSD SATA connector", "", dict(cls="connector", kind="sata_data")),
    ("Optical drive power connector", "", dict(cls="connector", kind="sata_power")),
    ("SSD shield", "", dict(cls="drive_cage")),
    ("SSD shield locker", "", dict(cls="drive_latch")),
    ("Case cover for motherboard screws", "", dict(cls="cover", of="motherboard_screws")),
    ("Initial Conditions", "", dict(step_type_hint="initial")),
    ("dupli", "", dict(step_type_hint="dupli")),
    ("change a direction", "", dict(step_type_hint="reorient")),
    ("remove cable from cable locker", "", dict(virtual="cable", verb="release")),
    ("Try to remove the power module", "", dict(cls="psu", attempt=True)),
    ("Motherboard screws 1/2/3", "", dict(cls="screw", role="motherboard", multi=True)),
    ("Open power module", "", dict(cls="psu", verb="displace")),
    ("Heatsink locker base", "", dict(cls="cooler_bracket")),
    ("all components", "", dict(step_type_hint="ignore")),
]


@pytest.mark.parametrize("raw,nest,exp", CASES)
def test_parse_raw_name(raw, nest, exp):
    p = parse_raw_name(raw, nest)
    for k, v in exp.items():
        if k == "n":
            assert p.instance_no == v
        elif k in ("cls", "virtual", "verb", "multi", "attempt", "step_type_hint"):
            assert getattr(p, k) == v
        else:
            assert p.attrs.get(k) == v


def test_taxonomy_states_and_verbs():
    t = load_taxonomy()
    assert set(t.classes) == {
        "chassis", "drive_cage", "cover", "cooler_bracket", "motherboard", "cpu",
        "cpu_cooler", "ram_module", "psu", "storage_drive", "optical_drive",
        "expansion_card", "case_fan", "misc_part", "screw", "ram_latch",
        "cpu_socket_lever", "psu_latch", "drive_latch", "card_latch", "cooler_latch",
        "cable_clip", "connector",
    }
    assert t.states_of("screw") == ["fastened", "loosened", "removed"]
    assert t.apply_verb("screw", {"captive": True}, "unscrew") == ("state", "loosened")
    assert t.apply_verb("screw", {"captive": False}, "unscrew") == ("state", "removed")
    assert t.apply_verb("connector", {}, "disconnect") == ("state", "unplugged")
    assert t.needs_mask("connector", "unplugged", "in_chassis") is False
    assert t.needs_mask("screw", "loosened", "in_chassis") is True
    assert t.needs_mask("psu", "removed", "on_bench") is True
    assert t.needs_mask("psu", "removed", "elsewhere") is False


@pytest.mark.parametrize(
    "raw,exp",
    [
        ("Philips PH2", "PH2"),
        ("PH 2", "PH2"),
        ("Philip PH2", "PH2"),
        ("T15", "T15"),
        ("Screw Driver T15", "T15"),
        ("Hand", "hand"),
        ("", "none"),
        ("PH1", "PH1"),
        ("Philips  PH2", "PH2"),
        ("PH3", "PH3"),
        ("T20", "T20"),
        ("Torx T15", "T15"),
        ("no idea", "none"),
    ],
)
def test_map_tool(raw, exp):
    assert map_tool(raw) == exp


# --- taxonomy.yaml structural invariants -------------------------------------


def test_every_class_default_state_is_one_of_its_states():
    t = load_taxonomy()
    for cls in t.classes:
        assert t.default_state(cls) in t.states_of(cls), cls


def test_needs_mask_covers_every_state_of_every_class():
    t = load_taxonomy()
    for cls in t.classes:
        table = t.classes[cls]["needs_mask"]
        assert set(table) == set(t.states_of(cls)), cls


def test_verbs_and_tools_match_spec():
    t = load_taxonomy()
    assert set(t.verbs) == {
        "unscrew", "disconnect", "open", "release", "remove", "displace", "reorient",
    }
    assert t.tools == ["PH1", "PH2", "PH3", "T15", "T20", "hand", "none"]


def test_apply_verb_returns_none_for_inapplicable_and_for_reorient():
    t = load_taxonomy()
    assert t.apply_verb("screw", {}, "disconnect") is None
    assert t.apply_verb("chassis", {}, "reorient") is None
    assert t.apply_verb("cover", {}, "open") == ("state", "open")
    assert t.apply_verb("psu", {}, "displace") == ("state", "displaced")
    assert t.apply_verb("cable_clip", {}, "release") == ("state", "open")
    assert t.apply_verb("cable", {}, "release") == ("state", "released")


def test_needs_mask_on_bench_is_true_except_chassis():
    t = load_taxonomy()
    assert t.needs_mask("screw", "removed", "on_bench") is True
    assert t.needs_mask("chassis", "present", "on_bench") is False


# --- taxonomy_map.yaml -------------------------------------------------------


def test_map_covers_all_canon_groups():
    # raw_logs/drive/vocab_target_canon.csv holds 53 groups (the brief says 52;
    # the csv is authoritative and its 53 groups cover all 2,830 step rows).
    path = REPO_ROOT / "configs" / "taxonomy_map.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    groups = {r["canon_group"] for r in cfg["rules"] if r.get("canon_group")}
    assert len(groups) == 53


def test_qualifier_and_multi_instances_are_captured():
    p = parse_raw_name("Optical drive - motherboard connector 2 (SATA)", "")
    assert p.cls == "connector"
    assert p.instance_no == 2
    assert p.attrs["qualifier"] == "SATA"
    q = parse_raw_name("Heatsink screw 1&2&3", "")
    assert q.multi is True
    assert q.attrs["instance_nos"] == [1, 2, 3]
    assert q.instance_no == 1


def test_leading_number_is_a_count_not_an_index():
    p = parse_raw_name("2 Case - motherboard connector", "")
    assert p.cls == "connector"
    assert p.multi is True
    assert p.instance_no is None


def test_all_and_half_mean_several_targets():
    assert parse_raw_name("All RAM clips", "").multi is True
    assert parse_raw_name("Half of the screws on the motherboard", "").multi is True


def test_capture_marker_carries_no_instance():
    p = parse_raw_name("replacement of 31 (dupli)", "")
    assert p.step_type_hint == "dupli"
    assert p.instance_no is None


def test_prefix_verb_is_remapped_per_class():
    assert parse_raw_name("Remove RAM cover", "").verb == "remove"          # cover
    assert parse_raw_name("Remove GPU", "").verb == "remove"                # part
    assert parse_raw_name("remove SSD locker shield", "").verb == "open"    # latch
    assert parse_raw_name("Remove heatsink broken screw 1", "").verb == "unscrew"


def test_unknown_name_falls_back():
    p = parse_raw_name("完全不存在的步骤名", "")
    assert p.matched_rule == "fallback"
    assert p.cls == ""
    assert p.canon_group == ""
