"""Tests for the state machine and geometry policy (spec 3.3 steps 1-2, 6.2-6.3).

The fixtures are hand-made instance/action lists -- a chassis, a cooler cover,
a CPU cooler held by four captive screws, a RAM module with two latches, a fan
connector, and a PSU with one non-captive screw -- so the expectations here do
not depend on the step-table importer.
"""
from __future__ import annotations

import pytest

from tda.core.model import ActionRec, InstanceRec, StateEvent
from tda.core.states import (
    InstState,
    diff_states,
    events_from_actions,
    initial_state,
    needs_geom,
    state_at,
    validate_events,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy

DESKTOP = 7
COOLER = "cpu_cooler.01"
COOLER_SCREWS = tuple(f"screw.cpu_cooler.{i:02d}" for i in range(1, 5))
CABLE = "cable:fan_harness"


def _inst(key: str, cls: str, **kw) -> InstanceRec:
    return InstanceRec(key=key, desktop=DESKTOP, cls=cls, **kw)


def _act(step: int, idx: int, target: str, verb: str, **kw) -> ActionRec:
    return ActionRec(desktop=DESKTOP, step=step, idx=idx, target=target, verb=verb, **kw)


def _ev(step: int, target: str, attr: str, old: str, new: str, **kw) -> StateEvent:
    return StateEvent(
        desktop=DESKTOP, step=step, target=target, attr=attr, old=old, new=new, **kw
    )


def _tuples(events: list[StateEvent]) -> list[tuple]:
    return [(e.step, e.target, e.attr, e.old, e.new) for e in events]


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return load_taxonomy()


@pytest.fixture
def instances() -> dict[str, InstanceRec]:
    recs = [
        _inst("chassis.01", "chassis"),
        _inst("cover.01", "cover", attrs={"of": "cpu_cooler"}),
        _inst(COOLER, "cpu_cooler", attrs={"kind": "heatsink_fan"}),
        *[
            _inst(
                key,
                "screw",
                attrs={"role": "cpu_cooler", "head": "PH2", "captive": True},
                parent=COOLER,
                attached=True,
                fastens=COOLER,
            )
            for key in COOLER_SCREWS
        ],
        _inst("ram_module.01", "ram_module"),
        _inst("ram_latch.01", "ram_latch", attrs={"of": "ram_module.01"}),
        _inst("ram_latch.02", "ram_latch", attrs={"of": "ram_module.01"}),
        _inst(
            "connector.01",
            "connector",
            attrs={"kind": "fan"},
            socket_host="motherboard.01",
            cable=CABLE,
        ),
        _inst("psu.01", "psu"),
        _inst(
            "screw.psu.01",
            "screw",
            attrs={"role": "psu", "head": "PH2", "captive": False},
            fastens="psu.01",
        ),
    ]
    return {r.key: r for r in recs}


@pytest.fixture
def cooler_actions() -> list[ActionRec]:
    """Loosen the four captive screws at step 12, lift the cooler at step 13."""
    return [
        *[_act(12, i, key, "unscrew", tool="PH2") for i, key in enumerate(COOLER_SCREWS)],
        _act(13, 0, COOLER, "remove", tool="hand", direction="+Z"),
    ]


# --------------------------------------------------------------------------- #
# initial_state
# --------------------------------------------------------------------------- #
def test_initial_state_uses_class_defaults_and_in_chassis(instances, tax):
    fs = initial_state(instances, tax)
    assert set(fs) == set(instances)
    assert all(st.placement == "in_chassis" for st in fs.values())
    assert fs["chassis.01"] == InstState(state="present", placement="in_chassis")
    assert fs["cover.01"].state == "closed"
    assert fs[COOLER].state == "installed"
    assert fs[COOLER_SCREWS[0]].state == "fastened"
    assert fs["ram_latch.01"].state == "closed"
    assert fs["connector.01"].state == "plugged"
    assert fs["psu.01"].state == "installed"


def test_initial_state_is_independent_per_call(instances, tax):
    first = initial_state(instances, tax)
    first["psu.01"].state = "removed"
    assert initial_state(instances, tax)["psu.01"].state == "installed"


# --------------------------------------------------------------------------- #
# events_from_actions
# --------------------------------------------------------------------------- #
def test_unscrew_captive_screw_only_loosens(instances, tax):
    events = events_from_actions(instances, [_act(4, 0, COOLER_SCREWS[0], "unscrew")], tax)
    assert _tuples(events) == [(4, COOLER_SCREWS[0], "state", "fastened", "loosened")]
    assert state_at(instances, events, 4, tax)[COOLER_SCREWS[0]] == InstState(
        state="loosened", placement="in_chassis"
    )


def test_unscrew_non_captive_screw_removes_it_to_the_bench(instances, tax):
    events = events_from_actions(instances, [_act(4, 0, "screw.psu.01", "unscrew")], tax)
    assert _tuples(events) == [
        (4, "screw.psu.01", "state", "fastened", "removed"),
        (4, "screw.psu.01", "placement", "in_chassis", "on_bench"),
    ]


def test_failed_action_emits_no_events(instances, tax):
    actions = [
        _act(4, 0, COOLER_SCREWS[0], "unscrew", result="failed", failure_reason="stripped head"),
        _act(5, 0, COOLER, "remove", result="failed", failure_reason="still fastened"),
    ]
    assert events_from_actions(instances, actions, tax) == []


def test_remove_part_cascades_to_attached_children_at_the_same_step(
    instances, cooler_actions, tax
):
    events = events_from_actions(instances, cooler_actions, tax)
    at_12 = [e for e in events if e.step == 12]
    at_13 = [e for e in events if e.step == 13]

    assert _tuples(at_12) == [(12, key, "state", "fastened", "loosened") for key in COOLER_SCREWS]
    assert _tuples(at_13) == [
        (13, COOLER, "state", "installed", "removed"),
        (13, COOLER, "placement", "in_chassis", "on_bench"),
        *[
            ev
            for key in COOLER_SCREWS
            for ev in (
                (13, key, "state", "loosened", "removed"),
                (13, key, "placement", "in_chassis", "on_bench"),
            )
        ],
    ]
    assert all(e.auto for e in at_13)
    assert all(e.desktop == DESKTOP for e in events)


def test_cascade_skips_children_that_are_not_attached(instances, tax):
    instances["screw.psu.01"].parent = "psu.01"
    instances["screw.psu.01"].attached = False
    events = events_from_actions(instances, [_act(20, 0, "psu.01", "remove")], tax)
    assert _tuples(events) == [
        (20, "psu.01", "state", "installed", "removed"),
        (20, "psu.01", "placement", "in_chassis", "on_bench"),
    ]


def test_cascade_is_transitive_through_attached_grandchildren(tax):
    instances = {
        r.key: r
        for r in (
            _inst(COOLER, "cpu_cooler", attrs={"kind": "heatsink"}),
            _inst("cooler_bracket.01", "cooler_bracket", parent=COOLER, attached=True),
            _inst(
                "screw.cooler_bracket.01",
                "screw",
                attrs={"role": "cooler_bracket", "captive": True},
                parent="cooler_bracket.01",
                attached=True,
            ),
        )
    }
    events = events_from_actions(instances, [_act(9, 0, COOLER, "remove")], tax)
    fs = state_at(instances, events, 9, tax)
    assert fs["cooler_bracket.01"] == InstState(state="removed", placement="on_bench")
    assert fs["screw.cooler_bracket.01"] == InstState(state="removed", placement="on_bench")


def test_remove_connector_changes_state_only(instances, tax):
    events = events_from_actions(instances, [_act(6, 0, "connector.01", "remove")], tax)
    assert _tuples(events) == [(6, "connector.01", "state", "plugged", "removed")]


def test_disconnect_then_remove_connector(instances, tax):
    actions = [_act(5, 0, "connector.01", "disconnect"), _act(6, 0, "connector.01", "remove")]
    assert _tuples(events_from_actions(instances, actions, tax)) == [
        (5, "connector.01", "state", "plugged", "unplugged"),
        (6, "connector.01", "state", "unplugged", "removed"),
    ]


def test_release_of_a_virtual_cable_is_recorded_but_not_an_instance(instances, tax):
    events = events_from_actions(instances, [_act(3, 0, CABLE, "release")], tax)
    assert _tuples(events) == [(3, CABLE, "state", "routed", "released")]
    fs = state_at(instances, events, 3, tax)
    assert CABLE not in fs
    assert CABLE not in needs_geom(instances, fs, tax)


def test_unknown_or_virtual_targets_are_ignored(instances, tax):
    actions = [
        _act(3, 0, "phantom.99", "remove"),
        _act(3, 1, CABLE, "remove"),  # "remove" does not apply to a cable node
    ]
    assert events_from_actions(instances, actions, tax) == []


def test_actions_are_processed_in_step_then_idx_order(instances, tax):
    actions = [
        _act(13, 0, COOLER, "remove"),
        _act(12, 1, COOLER_SCREWS[1], "unscrew"),
        _act(12, 0, COOLER_SCREWS[0], "unscrew"),
    ]
    events = events_from_actions(instances, actions, tax)
    assert [(e.step, e.target, e.old, e.new) for e in events[:3]] == [
        (12, COOLER_SCREWS[0], "fastened", "loosened"),
        (12, COOLER_SCREWS[1], "fastened", "loosened"),
        (13, COOLER, "installed", "removed"),
    ]


def test_displace_keeps_the_part_in_the_chassis(instances, tax):
    events = events_from_actions(instances, [_act(8, 0, "psu.01", "displace")], tax)
    assert _tuples(events) == [(8, "psu.01", "state", "installed", "displaced")]
    assert state_at(instances, events, 8, tax)["psu.01"].placement == "in_chassis"


def test_verbs_without_a_state_effect_emit_nothing(instances, tax):
    actions = [
        _act(2, 0, "chassis.01", "reorient", direction="rotate"),
        _act(3, 0, "ram_latch.01", "open"),
        _act(4, 0, "ram_latch.01", "open"),  # already open -> no second event
    ]
    assert _tuples(events_from_actions(instances, actions, tax)) == [
        (3, "ram_latch.01", "state", "closed", "open")
    ]


# --------------------------------------------------------------------------- #
# state_at
# --------------------------------------------------------------------------- #
def test_state_at_before_the_removal_step(instances, cooler_actions, tax):
    events = events_from_actions(instances, cooler_actions, tax)
    fs = state_at(instances, events, 12, tax)
    assert fs[COOLER] == InstState(state="installed", placement="in_chassis")
    assert fs[COOLER_SCREWS[0]] == InstState(state="loosened", placement="in_chassis")


def test_state_at_on_and_after_the_removal_step(instances, cooler_actions, tax):
    events = events_from_actions(instances, cooler_actions, tax)
    for step in (13, 40):
        fs = state_at(instances, events, step, tax)
        assert fs[COOLER] == InstState(state="removed", placement="on_bench")
        assert all(
            fs[key] == InstState(state="removed", placement="on_bench") for key in COOLER_SCREWS
        )


def test_state_at_step_zero_is_the_initial_state(instances, cooler_actions, tax):
    events = events_from_actions(instances, cooler_actions, tax)
    assert state_at(instances, events, 0, tax) == initial_state(instances, tax)


# --------------------------------------------------------------------------- #
# needs_geom
# --------------------------------------------------------------------------- #
def test_needs_geom_initially_asks_for_a_mask_everywhere(instances, tax):
    geom = needs_geom(instances, initial_state(instances, tax), tax)
    assert geom == {key: "mask" for key in instances}


def test_needs_geom_skips_unplugged_connectors_and_boxes_the_bench(instances, tax):
    actions = [
        _act(5, 0, "connector.01", "disconnect"),
        _act(6, 0, "screw.psu.01", "unscrew"),
        _act(7, 0, "psu.01", "remove"),
    ]
    events = events_from_actions(instances, actions, tax)
    geom = needs_geom(instances, state_at(instances, events, 7, tax), tax)
    assert "connector.01" not in geom
    assert geom["psu.01"] == "box"
    assert geom["screw.psu.01"] == "box"
    assert geom["chassis.01"] == "mask"
    assert geom["ram_latch.01"] == "mask"


def test_needs_geom_skips_removed_connectors_and_elsewhere(instances, tax):
    fs = initial_state(instances, tax)
    fs["connector.01"] = InstState(state="removed", placement="in_chassis")
    fs["psu.01"] = InstState(state="removed", placement="elsewhere")
    geom = needs_geom(instances, fs, tax)
    assert "connector.01" not in geom
    assert "psu.01" not in geom


def test_needs_geom_ignores_keys_without_an_instance(instances, tax):
    fs = initial_state(instances, tax)
    fs["phantom.99"] = InstState(state="installed", placement="in_chassis")
    assert "phantom.99" not in needs_geom(instances, fs, tax)


# --------------------------------------------------------------------------- #
# diff_states
# --------------------------------------------------------------------------- #
def test_diff_states_reports_state_and_placement_changes(instances, cooler_actions, tax):
    events = events_from_actions(instances, cooler_actions, tax)
    before = state_at(instances, events, 12, tax)
    after = state_at(instances, events, 13, tax)
    diff = diff_states(before, after)
    assert (COOLER, "state", "installed", "removed") in diff
    assert (COOLER, "placement", "in_chassis", "on_bench") in diff
    assert (COOLER_SCREWS[3], "state", "loosened", "removed") in diff
    assert len(diff) == 10  # 5 instances x (state + placement)
    assert diff == sorted(diff, key=lambda row: row[0])


def test_diff_states_of_equal_frames_is_empty(instances, tax):
    fs = initial_state(instances, tax)
    assert diff_states(fs, initial_state(instances, tax)) == []
    assert diff_states(fs, fs) == []


# --------------------------------------------------------------------------- #
# validate_events
# --------------------------------------------------------------------------- #
def test_validate_events_accepts_every_legal_transition(instances, tax):
    screw, latch, psu, ram = COOLER_SCREWS[0], "ram_latch.01", "psu.01", "ram_module.01"
    events = [
        _ev(2, screw, "state", "fastened", "loosened"),
        _ev(3, screw, "state", "loosened", "removed"),
        _ev(3, screw, "placement", "in_chassis", "on_bench"),
        _ev(4, "screw.psu.01", "state", "fastened", "removed"),
        _ev(4, "screw.psu.01", "placement", "in_chassis", "on_bench"),
        _ev(5, latch, "state", "closed", "open"),
        _ev(6, latch, "state", "open", "closed", auto=False),  # a manual reversal is legal
        _ev(7, "cover.01", "state", "closed", "open"),
        _ev(8, "cover.01", "state", "open", "removed"),
        _ev(8, "cover.01", "placement", "in_chassis", "on_bench"),
        _ev(9, "connector.01", "state", "plugged", "unplugged"),
        _ev(10, "connector.01", "state", "unplugged", "removed"),
        _ev(11, psu, "state", "installed", "displaced"),
        _ev(12, psu, "state", "displaced", "removed"),
        _ev(12, psu, "placement", "in_chassis", "on_bench"),
        _ev(13, ram, "state", "installed", "removed"),
        _ev(13, ram, "placement", "in_chassis", "elsewhere"),
        _ev(14, CABLE, "state", "routed", "released"),
    ]
    assert validate_events(instances, events, tax) == []


def test_validate_events_flags_transitions_out_of_removed(instances, tax):
    screw = COOLER_SCREWS[0]
    events = [
        _ev(11, screw, "state", "fastened", "removed"),
        _ev(12, screw, "state", "removed", "fastened"),
    ]
    problems = validate_events(instances, events, tax)
    assert len(problems) == 1
    assert problems[0].startswith(f"{screw}: removed → fastened at step 12: ")
    assert "removed" in problems[0].rsplit(": ", 1)[1]


def test_validate_events_flags_an_old_value_that_does_not_match(instances, tax):
    events = [
        _ev(4, "ram_latch.01", "state", "closed", "open"),
        _ev(5, "ram_latch.01", "state", "closed", "open"),
    ]
    problems = validate_events(instances, events, tax)
    assert len(problems) == 1
    assert problems[0].startswith("ram_latch.01: closed → open at step 5: ")
    assert "open" in problems[0].rsplit(": ", 1)[1]


def test_validate_events_flags_unknown_states_and_placements(instances, tax):
    events = [
        _ev(3, "ram_latch.01", "state", "closed", "unplugged"),
        _ev(4, "chassis.01", "state", "present", "removed"),
        _ev(5, "psu.01", "placement", "in_chassis", "in_the_bin"),
    ]
    problems = validate_events(instances, events, tax)
    assert len(problems) == 3
    assert "unplugged" in problems[0] and "ram_latch" in problems[0]
    assert "chassis.01" in problems[1]
    assert "in_the_bin" in problems[2]


def test_validate_events_flags_a_second_release_of_the_same_cable(instances, tax):
    events = [
        _ev(3, CABLE, "state", "routed", "released"),
        _ev(9, CABLE, "state", "routed", "released"),
    ]
    problems = validate_events(instances, events, tax)
    assert len(problems) == 1
    assert problems[0].startswith(f"{CABLE}: routed → released at step 9: ")


def test_validate_events_ignores_targets_it_does_not_know(instances, tax):
    assert validate_events(instances, [_ev(3, "phantom.99", "state", "a", "b")], tax) == []


def test_generated_events_validate_clean(instances, cooler_actions, tax):
    actions = [
        *cooler_actions,
        _act(3, 0, CABLE, "release"),
        _act(5, 0, "connector.01", "disconnect"),
        _act(6, 0, "cover.01", "open"),
        _act(7, 0, "cover.01", "remove"),
        _act(9, 0, "ram_latch.01", "open"),
        _act(9, 1, "ram_latch.02", "open"),
        _act(10, 0, "ram_module.01", "remove"),
        _act(14, 0, "screw.psu.01", "unscrew"),
        _act(15, 0, "psu.01", "displace"),
        _act(16, 0, "psu.01", "remove"),
    ]
    events = events_from_actions(instances, actions, tax)
    assert validate_events(instances, events, tax) == []
