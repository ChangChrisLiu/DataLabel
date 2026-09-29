"""Latches that leave the machine: carried out on their host, or taken out (U2a).

Two facts about D13 the vocabulary could not say:

* ``drive_latch.01`` sits on the drive cage (``parent=drive_cage.01,
  attached=True``), so the spec-3.3 cascade puts it in state ``removed`` when the
  cage comes out -- and saving warned "'removed' is not a state of class
  'drive_latch'";
* at step 33 the PSU latch is physically **taken out** (it is in the frames),
  and ``remove`` did not apply to ``psu_latch``.

So every latch-group class that can leave gets a ``removed`` state that needs
no mask, and the four brackets/latches that can be taken out get ``remove``.
``ram_latch`` and ``cpu_socket_lever`` are moulded into the board: they leave
only with it, never by a verb of their own.
"""
from __future__ import annotations

import pytest

from tda.core.export.vlm_tasks import state_is_askable
from tda.core.graph_plan import clearing_actions
from tda.core.graph_rules import Edge, verb_applies
from tda.core.model import ActionRec, InstanceRec
from tda.core.states import (
    events_from_actions,
    needs_geom,
    state_at,
    validate_events,
)
from tda.core.taxonomy import load_taxonomy

DESKTOP = 13
#: Every latch-group class that can leave the machine.
LEAVING = ("drive_latch", "psu_latch", "card_latch", "cooler_latch", "cable_clip")
#: ... of which these can be taken out by hand (removable brackets/latches).
REMOVABLE = ("psu_latch", "drive_latch", "card_latch", "cooler_latch")
#: Moulded into the motherboard: ``removed`` only ever by the cascade.
MOULDED = ("ram_latch", "cpu_socket_lever")


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def _inst(key: str, cls: str, **kw) -> InstanceRec:
    return InstanceRec(key=key, desktop=DESKTOP, cls=cls, **kw)


def _act(step: int, target: str, verb: str) -> ActionRec:
    return ActionRec(desktop=DESKTOP, step=step, idx=0, target=target, verb=verb)


# --------------------------------------------------------------------------- #
# the vocabulary
# --------------------------------------------------------------------------- #
def test_the_latch_group_is_exactly_the_classes_ruled_on(tax):
    """A latch class added later has to be ruled on too, not silently left out."""
    assert {c for c in tax.classes if tax.group_of(c) == "latch"} == {*LEAVING, *MOULDED}


@pytest.mark.parametrize("cls", [*LEAVING, *MOULDED])
def test_every_latch_has_a_removed_state_that_needs_no_mask(tax, cls):
    assert "removed" in tax.states_of(cls)
    assert tax.needs_mask(cls, "removed", "in_chassis") is False
    assert tax.needs_mask(cls, "closed", "in_chassis") is True
    assert tax.needs_mask(cls, "open", "in_chassis") is True
    assert tax.default_state(cls) == "closed"


@pytest.mark.parametrize("cls", REMOVABLE)
def test_a_removable_latch_can_be_taken_out(tax, cls):
    assert cls in tax.verbs["remove"]["applies_to"]
    assert tax.apply_verb(cls, {}, "remove") == ("state", "removed")
    for current in ("closed", "open"):
        assert verb_applies(tax, cls, {}, "remove", current)
    assert not verb_applies(tax, cls, {}, "remove", "removed")


@pytest.mark.parametrize("cls", [*MOULDED, "cable_clip"])
def test_moulded_latches_and_clips_are_never_removed_by_a_verb(tax, cls):
    assert cls not in tax.verbs["remove"]["applies_to"]
    assert tax.apply_verb(cls, {}, "remove") is None


def test_opening_is_still_how_a_latch_is_opened(tax):
    for cls in (*LEAVING, *MOULDED):
        assert cls in tax.verbs["open"]["applies_to"]
        assert tax.apply_verb(cls, {}, "open") == ("state", "open")


# --------------------------------------------------------------------------- #
# the state machine: D13's two facts
# --------------------------------------------------------------------------- #
def test_a_drive_latch_leaves_with_its_cage_without_a_warning(tax):
    instances = {r.key: r for r in (
        _inst("chassis.01", "chassis"),
        _inst("drive_cage.01", "drive_cage"),
        _inst("drive_latch.01", "drive_latch", parent="drive_cage.01", attached=True),
    )}
    actions = [_act(10, "drive_latch.01", "open"), _act(11, "drive_cage.01", "remove")]
    events = events_from_actions(instances, actions, tax)
    assert validate_events(instances, events, tax) == []   # no "'removed' is not a state"
    fs = state_at(instances, events, 11, tax)
    assert fs["drive_latch.01"].state == "removed"
    assert "drive_latch.01" not in needs_geom(instances, fs, tax)
    before = state_at(instances, events, 10, tax)
    assert needs_geom(instances, before, tax)["drive_latch.01"] == "mask"


def test_a_psu_latch_taken_out_is_gone_from_the_next_frames_task(tax):
    instances = {r.key: r for r in (
        _inst("chassis.01", "chassis"),
        _inst("psu.01", "psu"),
        _inst("psu_latch.01", "psu_latch"),
    )}
    actions = [_act(20, "psu_latch.01", "open"), _act(33, "psu_latch.01", "remove")]
    events = events_from_actions(instances, actions, tax)
    assert validate_events(instances, events, tax) == []
    after = state_at(instances, events, 33, tax)
    assert after["psu_latch.01"].state == "removed"
    # out of the chassis: no mask; on the bench like every part taken out, so
    # only a view that sees the bench asks for its box
    assert "psu_latch.01" not in needs_geom(instances, after, tax, bench_roi=None)
    assert needs_geom(instances, after, tax, bench_roi=[0, 0, 32, 32])["psu_latch.01"] == "box"
    assert needs_geom(instances, state_at(instances, events, 32, tax), tax,
                      bench_roi=None)["psu_latch.01"] == "mask"


# --------------------------------------------------------------------------- #
# the constraint graph: removing a latch clears what it locks
# --------------------------------------------------------------------------- #
def test_a_removable_latch_offers_remove_as_the_last_way_to_clear_it(tax):
    instances = {"psu.01": _inst("psu.01", "psu"),
                 "psu_latch.01": _inst("psu_latch.01", "psu_latch")}
    edge = Edge("locked_by", "psu.01", "psu_latch.01")
    assert clearing_actions(edge, instances, {"psu.01": "installed",
                                              "psu_latch.01": "closed"}, tax) == [
        ("open", "psu_latch.01"), ("remove", "psu_latch.01")]


def test_a_moulded_latch_is_still_cleared_by_opening_only(tax):
    instances = {"ram_module.01": _inst("ram_module.01", "ram_module"),
                 "ram_latch.01": _inst("ram_latch.01", "ram_latch")}
    edge = Edge("locked_by", "ram_module.01", "ram_latch.01")
    assert clearing_actions(edge, instances, {"ram_module.01": "installed",
                                              "ram_latch.01": "closed"}, tax) == [
        ("open", "ram_latch.01")]


# --------------------------------------------------------------------------- #
# the VLM export: the state question still asks, and knows the new answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [*LEAVING, *MOULDED])
def test_a_latch_state_is_still_a_question_and_removed_is_an_allowed_answer(tax, cls):
    assert state_is_askable(tax, cls)
    assert "removed" in tax.states_of(cls)
