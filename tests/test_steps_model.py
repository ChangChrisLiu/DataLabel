"""Qt-free tests for the S1 step-table view-model (:mod:`tda.ui.steps_model`).

Every test seeds a temporary database the way stage S0 does -- import one of the
exported log fixtures and write the drafts -- then drives the view-model exactly
as the panel does: edit, validate, save, reload.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from steps_fixtures import seeded_db
from tda.core.db import Db
from tda.core.taxonomy import load_taxonomy
from tda.ui.steps_model import LS_NOTE_PREFIX, EditError, StepTableData, thumb_path


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_db_path, tax) -> Db:
    conn = seeded_db(tmp_db_path, tax)
    yield conn
    conn.close()


@pytest.fixture
def data(db, tax) -> StepTableData:
    return StepTableData.load(db, 13, tax)


@pytest.fixture
def compound(db, tax) -> StepTableData:
    return StepTableData.load(db, 63, tax)


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def test_load_builds_one_row_per_step(data):
    assert len(data.rows) == 42
    assert [r.number for r in data.rows] == list(range(1, 43))
    first = data.row(1)
    assert first.step_type == "initial"
    assert first.actions == []  # a capture marker drafts no action


def test_load_carries_the_parsed_action_of_each_row(data):
    row = data.row(3)
    assert row.raw_name == "CPU Fan Screw 1"
    assert len(row.actions) == 1
    action = row.actions[0]
    assert (action.target, action.verb, action.tool) == ("screw.cpu_cooler.01", "unscrew", "PH2")


def test_load_carries_the_instance_table(data):
    assert data.instances["screw.cpu_cooler.01"].cls == "screw"
    assert data.instances["chassis"].cls == "chassis"


# --------------------------------------------------------------------------- #
# apply_edit -- accepted
# --------------------------------------------------------------------------- #
def test_apply_edit_changes_the_verb(data):
    data.apply_edit(3, "verb", "remove")
    assert data.row(3).actions[0].verb == "remove"


def test_apply_edit_changes_the_tool(data):
    data.apply_edit(3, "tool", "PH1")
    assert data.row(3).actions[0].tool == "PH1"


def test_apply_edit_changes_direction_difficulty_and_notes(data):
    data.apply_edit(3, "direction", "+Z")
    data.apply_edit(3, "difficulty", 4)
    data.apply_edit(3, "notes", "very tight")
    row = data.row(3)
    assert row.actions[0].direction == "+Z"
    assert row.actions[0].difficulty == 4
    assert row.notes == "very tight"


def test_apply_edit_records_a_failed_attempt(data):
    data.apply_edit(3, "result", "failed")
    data.apply_edit(3, "failure_reason", "fastener_stuck")
    assert data.row(3).actions[0].result == "failed"
    assert data.row(3).actions[0].failure_reason == "fastener_stuck"


def test_apply_edit_changes_the_step_type(data):
    data.apply_edit(11, "step_type", "auxiliary")
    assert data.row(11).step_type == "auxiliary"


def test_apply_edit_clears_difficulty_and_failure_reason(data):
    data.apply_edit(3, "difficulty", 2)
    data.apply_edit(3, "difficulty", None)
    data.apply_edit(3, "failure_reason", "")
    assert data.row(3).actions[0].difficulty is None
    assert data.row(3).actions[0].failure_reason is None


# --------------------------------------------------------------------------- #
# apply_edit -- rejected
# --------------------------------------------------------------------------- #
def test_apply_edit_rejects_a_verb_that_does_not_apply_to_the_class(data):
    with pytest.raises(EditError) as err:
        data.apply_edit(3, "verb", "disconnect")
    assert "screw" in str(err.value)
    assert data.row(3).actions[0].verb == "unscrew"  # unchanged


def test_apply_edit_refuses_to_remove_a_board_mounted_latch(data):
    """A latch reaches ``removed`` by riding out inside the board, never by a verb."""
    for step, cls in ((14, "ram_latch"), (20, "cpu_socket_lever")):
        with pytest.raises(EditError) as err:
            data.apply_edit(step, "verb", "remove")
        assert cls in str(err.value)
        assert data.row(step).actions[0].verb == "open"  # unchanged


def test_apply_edit_rejects_an_unknown_verb(data):
    with pytest.raises(EditError):
        data.apply_edit(3, "verb", "unbolt")


def test_apply_edit_rejects_an_unknown_tool(data):
    with pytest.raises(EditError):
        data.apply_edit(3, "tool", "hammer")


def test_apply_edit_rejects_an_unknown_direction(data):
    with pytest.raises(EditError):
        data.apply_edit(3, "direction", "sideways")


@pytest.mark.parametrize("value", [0, 6, "high"])
def test_apply_edit_rejects_an_out_of_range_difficulty(data, value):
    with pytest.raises(EditError):
        data.apply_edit(3, "difficulty", value)


def test_apply_edit_rejects_an_unknown_failure_reason(data):
    with pytest.raises(EditError):
        data.apply_edit(3, "failure_reason", "gremlins")


def test_apply_edit_rejects_an_unknown_step_type(data):
    with pytest.raises(EditError):
        data.apply_edit(3, "step_type", "maybe")


def test_apply_edit_rejects_an_unknown_field(data):
    with pytest.raises(EditError):
        data.apply_edit(3, "colour", "red")


def test_apply_edit_rejects_an_action_field_on_a_row_without_actions(data):
    with pytest.raises(EditError):
        data.apply_edit(1, "verb", "remove")


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #
def test_add_action_appends_with_the_next_index(data):
    added = data.add_action(3)
    row = data.row(3)
    assert len(row.actions) == 2
    assert [a.idx for a in row.actions] == [0, 1]
    assert added.target == row.actions[0].target  # cloned from the first action


def test_apply_edit_reaches_the_second_action(data):
    data.add_action(3)
    data.apply_edit(3, "tool", "hand", action_idx=1)
    assert [a.tool for a in data.row(3).actions] == ["PH2", "hand"]


def test_remove_action_renumbers_the_rest(data):
    data.add_action(3)
    data.add_action(3)
    data.remove_action(3, 0)
    row = data.row(3)
    assert [a.idx for a in row.actions] == [0, 1]


def test_remove_action_rejects_an_unknown_index(data):
    with pytest.raises(EditError):
        data.remove_action(3, 7)


# --------------------------------------------------------------------------- #
# change_target
# --------------------------------------------------------------------------- #
def test_change_target_to_an_existing_instance(data):
    key = data.change_target(4, target="screw.cpu_cooler.01")
    assert key == "screw.cpu_cooler.01"
    assert data.row(4).actions[0].target == "screw.cpu_cooler.01"


def test_change_target_rejects_an_unknown_instance(data):
    with pytest.raises(EditError):
        data.change_target(4, target="screw.cpu_cooler.99")


def test_change_target_creates_a_new_instance_with_the_next_ordinal(data):
    key = data.change_target(42, cls="screw", attrs={"role": "motherboard"})
    assert key == "screw.motherboard.07"  # six exist already
    assert data.instances[key].cls == "screw"
    assert data.instances[key].attrs["role"] == "motherboard"
    assert data.row(42).actions[0].target == key


def test_change_target_starts_a_new_class_at_ordinal_one(data):
    key = data.change_target(42, cls="expansion_card", attrs={"kind": "gpu"})
    assert key == "expansion_card.gpu.01"


def test_change_target_rejects_an_unknown_class(data):
    with pytest.raises(EditError):
        data.change_target(42, cls="flux_capacitor")


def test_change_target_rejects_a_role_outside_the_class_vocabulary(data):
    with pytest.raises(EditError) as err:
        data.change_target(42, cls="screw", attrs={"role": "mainboard"})
    assert "motherboard" in str(err.value)  # the message lists what is allowed
    assert not any(k.startswith("screw.mainboard") for k in data.instances)


def test_change_target_rejects_a_kind_outside_the_class_vocabulary(data):
    with pytest.raises(EditError):
        data.change_target(42, cls="expansion_card", attrs={"kind": "soundcard"})


def test_change_target_rejects_a_discriminator_the_class_does_not_have(data):
    with pytest.raises(EditError) as err:
        data.change_target(42, cls="motherboard", attrs={"kind": "atx"})
    assert "kind" in str(err.value)


def test_change_target_accepts_a_free_form_discriminator(data):
    """``ram_latch.of`` is an open list, so the taxonomy states no vocabulary."""
    key = data.change_target(17, cls="ram_latch", attrs={"of": "ram_module.01"})
    assert key == "ram_latch.05"


def test_change_target_accepts_a_virtual_cable_node(data):
    key = data.change_target(42, target="cable:psu", verb="release")
    assert key == "cable:psu"
    assert data.row(42).actions[0].verb == "release"
    assert "cable:psu" not in data.instances  # virtual nodes carry no instance row


def test_change_target_validates_the_new_verb_against_the_new_class(data):
    with pytest.raises(EditError):
        data.change_target(42, target="cable:psu", verb="remove")
    assert data.row(42).actions[0].target == "motherboard.01"  # unchanged


def test_change_target_rejects_a_target_the_current_verb_cannot_reach(data):
    """Step 3 is an ``unscrew``; only a screw can be unscrewed."""
    with pytest.raises(EditError) as err:
        data.change_target(3, target="motherboard.01")
    assert "unscrew" in str(err.value)
    assert data.row(3).actions[0].target == "screw.cpu_cooler.01"  # unchanged


# --------------------------------------------------------------------------- #
# split_compound
# --------------------------------------------------------------------------- #
def test_split_compound_creates_n_actions_and_n_instances(compound):
    row = compound.row(5)
    assert row.step_type == "compound"
    assert row.actions[0].target.endswith("?")

    # D63 step 4 already unscrewed `screw.cpu_cooler.01`, so the three parts of
    # "Heatsink screw 1&2&3" continue that numbering rather than restarting it.
    keys = compound.split_compound(5, 3)
    assert keys == ["screw.cpu_cooler.02", "screw.cpu_cooler.03", "screw.cpu_cooler.04"]
    row = compound.row(5)
    assert row.step_type == "normal"
    assert [a.idx for a in row.actions] == [0, 1, 2]
    assert [a.target for a in row.actions] == keys
    assert all(a.verb == "unscrew" for a in row.actions)
    for key in keys:
        assert compound.instances[key].cls == "screw"
        assert compound.instances[key].raw_names == ["Heatsink screw 1&2&3"]


def test_split_compound_rejects_a_non_compound_row(data):
    with pytest.raises(EditError):
        data.split_compound(3, 2)


def test_split_compound_leaves_the_row_untouched_when_it_fails(compound, monkeypatch):
    """A split refused half-way must not leave the row headless."""
    before_actions = list(compound.row(5).actions)
    before_keys = set(compound.instances)
    calls = {"n": 0}
    real = compound._build_instance

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise EditError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(compound, "_build_instance", flaky)
    with pytest.raises(EditError):
        compound.split_compound(5, 3)

    assert compound.row(5).actions == before_actions
    assert compound.row(5).step_type == "compound"
    assert set(compound.instances) == before_keys


def test_split_compound_rejects_fewer_than_two(compound):
    with pytest.raises(EditError):
        compound.split_compound(5, 1)


# --------------------------------------------------------------------------- #
# instances tab
# --------------------------------------------------------------------------- #
def test_instance_edit_sets_the_relational_attributes(data):
    data.apply_instance_edit("screw.cpu_cooler.01", "parent", "cpu_cooler.fan.01")
    data.apply_instance_edit("screw.cpu_cooler.01", "attached", True)
    data.apply_instance_edit("screw.cpu_cooler.01", "fastens", "cpu_cooler.fan.01")
    data.apply_instance_edit("screw.cpu_cooler.01", "mounted_on", "motherboard.01")
    inst = data.instances["screw.cpu_cooler.01"]
    assert inst.parent == "cpu_cooler.fan.01"
    assert inst.attached is True
    assert inst.fastens == "cpu_cooler.fan.01"
    assert inst.mounted_on == "motherboard.01"


def test_instance_edit_sets_head_captive_group_order_and_direction(data):
    data.apply_instance_edit("screw.cpu_cooler.01", "head", "T15")
    data.apply_instance_edit("screw.cpu_cooler.01", "captive", False)
    data.apply_instance_edit("screw.cpu_cooler.01", "group_order", "opposite_pairs")
    data.apply_instance_edit("screw.cpu_cooler.01", "removal_direction", "+Z")
    inst = data.instances["screw.cpu_cooler.01"]
    assert inst.attrs["head"] == "T15"
    assert inst.attrs["captive"] is False
    assert inst.group_order == "opposite_pairs"
    assert inst.removal_direction == "+Z"


def test_instance_edit_sets_connector_socket_host_and_cable(data):
    data.apply_instance_edit("connector.01", "socket_host", "storage_drive.ssd.01")
    data.apply_instance_edit("connector.01", "cable", "cable:psu")
    inst = data.instances["connector.01"]
    assert inst.socket_host == "storage_drive.ssd.01"
    assert inst.cable == "cable:psu"


def test_instance_edit_clears_a_relation_with_an_empty_value(data):
    data.apply_instance_edit("screw.cpu_cooler.01", "parent", "cpu_cooler.fan.01")
    data.apply_instance_edit("screw.cpu_cooler.01", "parent", "")
    assert data.instances["screw.cpu_cooler.01"].parent is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("parent", "nope.01"),
        ("mounted_on", "nope.01"),
        ("head", "PH9"),
        ("group_order", "random"),
        ("removal_direction", "sideways"),
        ("cable", "psu"),  # a virtual cable node id must carry the prefix
        ("colour", "red"),
    ],
)
def test_instance_edit_rejects_invalid_values(data, field, value):
    with pytest.raises(EditError):
        data.apply_instance_edit("screw.cpu_cooler.01", field, value)


def test_instance_edit_rejects_fastens_on_a_non_screw(data):
    with pytest.raises(EditError):
        data.apply_instance_edit("cpu_cooler.fan.01", "fastens", "motherboard.01")


def test_instance_edit_rejects_socket_host_on_a_non_connector(data):
    with pytest.raises(EditError):
        data.apply_instance_edit("screw.cpu_cooler.01", "socket_host", "motherboard.01")


def test_instance_edit_rejects_a_parent_cycle(data):
    data.apply_instance_edit("screw.cpu_cooler.01", "parent", "cpu_cooler.fan.01")
    with pytest.raises(EditError):
        data.apply_instance_edit("cpu_cooler.fan.01", "parent", "screw.cpu_cooler.01")
    with pytest.raises(EditError):
        data.apply_instance_edit("cpu_cooler.fan.01", "parent", "cpu_cooler.fan.01")


def test_instance_edit_rejects_an_unknown_instance(data):
    with pytest.raises(EditError):
        data.apply_instance_edit("nope.01", "parent", "chassis")


# --------------------------------------------------------------------------- #
# issues
# --------------------------------------------------------------------------- #
def test_issues_flag_unresolved_targets_and_compound_rows(compound):
    text = "\n".join(compound.issues)
    assert "step 5" in text
    assert compound.row(5).issues  # the row carries its own copy


def test_a_clean_desktop_has_no_issues(data):
    assert data.issues == []


def test_issues_flag_a_failed_step_without_a_reason(data):
    data.apply_edit(3, "result", "failed")
    assert any("failure_reason" in i for i in data.row(3).issues)


# --------------------------------------------------------------------------- #
# notes: the imported Label Studio lines are preserved
# --------------------------------------------------------------------------- #
def test_ls_note_prefix_matches_the_label_studio_importer():
    from tda.core import ls_import

    assert LS_NOTE_PREFIX == ls_import.NOTES_PREFIX


def test_notes_hide_the_imported_ls_line_from_the_editor(db, data, tax):
    data.row(3).step.notes = "tight\nLS: difficulty=hard; target=[1]"
    assert data.row(3).notes == "tight"
    assert data.row(3).ls_notes == ["LS: difficulty=hard; target=[1]"]


def test_editing_notes_preserves_the_imported_ls_line(db, data, tax):
    data.row(3).step.notes = "tight\nLS: difficulty=hard"
    data.apply_edit(3, "notes", "stripped head")
    assert data.row(3).notes == "stripped head"
    assert data.row(3).step.notes == "stripped head\nLS: difficulty=hard"

    data.save(db)
    assert StepTableData.load(db, 13, tax).row(3).ls_notes == ["LS: difficulty=hard"]


def test_clearing_notes_keeps_the_imported_ls_line(data):
    data.row(3).step.notes = "tight\nLS: difficulty=hard"
    data.apply_edit(3, "notes", "")
    assert data.row(3).notes == ""
    assert data.row(3).step.notes == "LS: difficulty=hard"


# --------------------------------------------------------------------------- #
# save / reload
# --------------------------------------------------------------------------- #
def test_save_writes_steps_actions_and_instances(db, data, tax):
    data.apply_edit(3, "tool", "PH1")
    data.apply_edit(3, "difficulty", 5)
    data.apply_edit(11, "step_type", "auxiliary")
    data.apply_instance_edit("screw.cpu_cooler.01", "head", "T15")
    assert data.save(db) == []

    back = StepTableData.load(db, 13, tax)
    assert back.row(3).actions[0].tool == "PH1"
    assert back.row(3).actions[0].difficulty == 5
    assert back.row(11).step_type == "auxiliary"
    assert back.instances["screw.cpu_cooler.01"].attrs["head"] == "T15"


def test_save_regenerates_the_auto_events(db, data, tax):
    data.apply_instance_edit("screw.cpu_cooler.01", "parent", "cpu_cooler.fan.01")
    data.apply_instance_edit("screw.cpu_cooler.01", "attached", True)
    data.save(db)

    cascade = [
        e for e in db.events(13)
        if e.target == "screw.cpu_cooler.01" and e.step == 13 and e.attr == "state"
    ]
    assert len(cascade) == 1
    assert (cascade[0].old, cascade[0].new, cascade[0].auto) == ("loosened", "removed", True)


def test_save_keeps_hand_written_events(db, data, tax):
    from tda.core.model import StateEvent

    manual = StateEvent(13, 2, "chassis", "state", "present", "present", auto=False)
    db.replace_events(13, [manual], auto_only=False)
    StepTableData.load(db, 13, tax).save(db)

    kept = [e for e in db.events(13) if not e.auto]
    assert len(kept) == 1
    assert any(e.auto for e in db.events(13))  # the auto log was rebuilt too


def test_save_returns_validation_messages(db, data, tax):
    from tda.core.model import StateEvent

    # A hand-written event that contradicts the compiled trajectory.
    bad = StateEvent(13, 2, "screw.cpu_cooler.01", "state", "removed", "fastened", auto=False)
    db.replace_events(13, [bad], auto_only=False)
    fresh = StepTableData.load(db, 13, tax)
    messages = fresh.save(db)
    assert messages
    assert any("screw.cpu_cooler.01" in m for m in messages)
    assert fresh.messages == messages


def test_save_after_split_compound_persists_the_new_instances(db, compound, tax):
    keys = compound.split_compound(5, 3)
    compound.save(db)

    back = StepTableData.load(db, 63, tax)
    assert [a.target for a in back.row(5).actions] == keys
    assert all(k in back.instances for k in keys)


def test_save_is_atomic(db, data, tax, monkeypatch):
    """A failure part-way through save() must leave the database untouched."""
    data.apply_edit(3, "tool", "PH1")
    data.apply_edit(11, "step_type", "auxiliary")

    calls = {"n": 0}
    real = db.upsert_instance

    def flaky(inst):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("disk full")
        return real(inst)

    monkeypatch.setattr(db, "upsert_instance", flaky)
    with pytest.raises(RuntimeError):
        data.save(db)

    back = StepTableData.load(db, 13, tax)
    assert back.row(3).actions[0].tool == "PH2"
    assert back.row(11).step_type == "normal"
    assert len(back.instances) == len(data.instances)
    assert db.events(13)  # the auto log was not wiped either


def test_reload_discards_unsaved_edits(db, data, tax):
    data.apply_edit(3, "tool", "PH1")
    assert StepTableData.load(db, 13, tax).row(3).actions[0].tool == "PH2"


# --------------------------------------------------------------------------- #
# thumbnails
# --------------------------------------------------------------------------- #
def test_thumb_path_returns_none_when_the_cache_file_is_missing(tmp_path):
    assert thumb_path(tmp_path, 13, 4) is None


def test_thumb_path_finds_the_cached_scan_frame(tmp_path):
    target = tmp_path / "scan" / "D13" / "s004.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert thumb_path(tmp_path, 13, 4) == target
    assert thumb_path(tmp_path, 13, 4, view="oak1") is None


def test_thumb_path_tolerates_a_missing_cache_root():
    assert thumb_path("D:/DataSet/cache/definitely-not-here", 13, 4) is None


def test_thumb_path_of_step_zero_is_none(tmp_path):
    """There is no frame before step 1, so the "before" cell stays empty."""
    assert thumb_path(tmp_path, 13, 0) is None


def test_thumb_path_agrees_with_the_cache_module(tmp_path):
    """Drift guard: the panel must look where :mod:`tda.core.cache` writes."""
    from tda.core.cache import cache_path
    from tda.core.model import FrameKey

    key = FrameKey(13, 4, "scan")
    expected = Path(cache_path(tmp_path, key, "png"))
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"x")
    assert thumb_path(tmp_path, key.desktop, key.step, key.view) == expected
