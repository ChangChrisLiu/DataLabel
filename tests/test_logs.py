"""Tests for the Drive log importer (spec sections 2.3 and 3.2).

The three fixtures are verbatim copies of the real exports:

* ``desktop_13`` -- clean Dell/Optiplex log, contiguous Sequence Numbers.
* ``desktop_63`` -- Sequence Numbers 11-13 are repeated; several compound rows.
* ``desktop_01`` -- two ``dupli`` marker rows and nest-group-only connectors.

Synthetic row lists cover the rules no fixture happens to exercise
(``ignore``/``auxiliary``/``failed`` steps, skipped Sequence Numbers,
tool-less screws, fallback names).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tda.core.log_report import main, operation_counts
from tda.core.logs import (
    _attr_conflicts,
    _identity,
    _merge_attrs,
    import_log,
    instance_key,
    iter_desktop_csvs,
    read_desktop_csv,
)
from tda.core.taxonomy import load_taxonomy

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"

COLUMNS = (
    "Sequence Number",
    "Sequence Name",
    "Target Class Label",
    "Target Nest Group",
    "Detailed Task Description",
    "Complexity",
    "Human Operation Description",
    "Pre-request Steps",
    "Operation Time (s)",
    "Tool Utility",
    "Deliverable",
    "Notes",
)


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def fixture_import(desktop: int, tax):
    """Import one fixture; returns (LogImport, raw rows)."""
    rows, meta = read_desktop_csv(FIXTURES / f"desktop_{desktop:02d}.csv")
    return import_log(desktop, rows, meta, tax), rows


def row(name: str, seq: int, **kw) -> dict:
    """One synthetic sheet row with all exported columns present."""
    out = {c: "" for c in COLUMNS}
    out["Sequence Number"] = str(seq)
    out["Sequence Name"] = name
    for key, value in kw.items():
        out[key.replace("_", " ")] = value
    return out


def synth(names, meta=None, desktop=99, tax=None, **kw):
    """Import a list of raw names as consecutive rows.

    Row 1 is always the ``initial`` step, so a leading "Initial Conditions" row
    is prepended unless the caller already supplied one.
    """
    if names[0] != "Initial Conditions":
        names = ["Initial Conditions", *names]
    rows = [row(n, i + 1, **kw) for i, n in enumerate(names)]
    return import_log(desktop, rows, meta or {}, tax)


def actions_of(li, step: int) -> list:
    return [a for a in li.actions if a.step == step]


# --------------------------------------------------------------------------- #
# instance_key
# --------------------------------------------------------------------------- #
def test_instance_key():
    assert instance_key("screw", {"role": "cpu_cooler"}, 3) == "screw.cpu_cooler.03"
    assert instance_key("psu", {}, 1) == "psu.01"


def test_instance_key_discriminators():
    assert instance_key("connector", {"kind": "front_panel"}, 2) == "connector.front_panel.02"
    assert instance_key("ram_latch", {}, 1) == "ram_latch.01"
    assert instance_key("chassis", {}, 1) == "chassis"
    # role wins over kind, and cable_owner never enters the key.
    assert instance_key("connector", {"kind": "fan", "cable_owner": "cpu_fan"}, 4) == (
        "connector.fan.04"
    )


# --------------------------------------------------------------------------- #
# read_desktop_csv
# --------------------------------------------------------------------------- #
def test_read_desktop_csv_rows_and_meta():
    rows, meta = read_desktop_csv(FIXTURES / "desktop_13.csv")
    assert len(rows) == 42  # the 18 number-only trailing rows are dropped
    assert set(COLUMNS) <= set(rows[0])
    assert rows[0]["Sequence Name"] == "Initial Conditions"
    assert rows[2]["Tool Utility"] == "Philips PH2"
    assert meta["brand_model_raw"] == "Optiplex 7020"
    assert meta["size_raw"] == "12.25*11.5*3.75"
    assert meta["collection_date"] == "2025-06-03"
    assert meta["notes"] == ""


def test_read_desktop_csv_two_table_layout(tmp_path):
    """The steps table and the metadata table may share one exported sheet."""
    path = tmp_path / "desktop_77.csv"
    path.write_text(
        "Sequence Number,Sequence Name,Tool Utility,Notes\n"
        "1,Initial Conditions,,\n"
        "2,CPU fan screw 1,Philips PH2,tight\n"
        ",,,\n"
        "Desktop ID,77,,\n"
        "Desktop Brand,Dell Optiplex 7020,,\n"
        "Desktop Size,12*11*4,,\n"
        "Collection Date,2025-06-03 00:00:00,,\n"
        ",Side screws are T15,,\n",
        encoding="utf-8",
    )
    rows, meta = read_desktop_csv(path)
    assert [r["Sequence Name"] for r in rows] == ["Initial Conditions", "CPU fan screw 1"]
    assert rows[1]["Notes"] == "tight"
    assert meta["brand_model_raw"] == "Dell Optiplex 7020"
    assert meta["size_raw"] == "12*11*4"
    assert meta["collection_date"] == "2025-06-03"
    assert meta["notes"] == "Side screws are T15"


# --------------------------------------------------------------------------- #
# The three fixture imports
# --------------------------------------------------------------------------- #
def test_import_d13_counts(tax):
    rows, meta = read_desktop_csv(FIXTURES / "desktop_13.csv")
    li = import_log(13, rows, meta, tax)
    assert len(li.steps) == 42 and li.steps[0].step_type == "initial"
    screws = [k for k in li.instances if k.startswith("screw.cpu_cooler.")]
    assert screws == [
        "screw.cpu_cooler.01",
        "screw.cpu_cooler.02",
        "screw.cpu_cooler.03",
        "screw.cpu_cooler.04",
    ]
    # Row 3 is "CPU Fan Screw 1" -> first-operated cpu_cooler screw -> .01;
    # row 4 is "CPU Fan Screw 2" -> .02 (the brief's expectation, one row down).
    first = actions_of(li, 3)[0]
    assert first.target == "screw.cpu_cooler.01"
    a = actions_of(li, 4)[0]
    assert a.target == "screw.cpu_cooler.02" and a.verb == "unscrew" and a.tool == "PH2"
    assert "chassis" in li.instances


def test_import_d63_duplicate_seq_numbers_reported(tax):
    rows, meta = read_desktop_csv(FIXTURES / "desktop_63.csv")
    li = import_log(63, rows, meta, tax)
    assert any("Sequence Number" in s for s in li.issues)
    assert li.steps[-1].step == len(rows)


def test_import_d01_dupli_and_connectors(tax):
    rows, meta = read_desktop_csv(FIXTURES / "desktop_01.csv")
    li = import_log(1, rows, meta, tax)
    dupli = [s for s in li.steps if s.step_type == "dupli"]
    assert len(dupli) == 2 and all(not actions_of(li, s.step) for s in dupli)
    c = actions_of(li, 14)[0]  # "Connector 1" nest=PSU
    assert li.instances[c.target].cls == "connector" and li.instances[c.target].cable == "cable:psu"


# --------------------------------------------------------------------------- #
# Logical steps vs the sheet's own numbers
# --------------------------------------------------------------------------- #
def test_logical_steps_are_row_order_not_sequence_number(tax):
    li, rows = fixture_import(63, tax)
    assert [s.step for s in li.steps] == list(range(1, len(rows) + 1))
    # Sheet row 14 restarts the Sequence Number at 11; the logical step is 14.
    assert rows[13]["Sequence Number"] == "11"
    assert li.steps[13].step == 14
    assert li.steps[13].raw_name == "Open cover case for CPU fan"
    dups = [s for s in li.issues if "duplicated" in s]
    assert len(dups) == 3 and all("Sequence Number" in s for s in dups)


def test_skipped_sequence_number_reported(tax):
    rows = [row("Initial Conditions", 1), row("CPU fan screw 1", 2), row("Motherboard", 4)]
    li = import_log(56, rows, {}, tax)
    assert any("Sequence Number 3" in s and "skipped" in s for s in li.issues)
    assert [s.step for s in li.steps] == [1, 2, 3]


def test_blank_sequence_number_reported(tax):
    li = import_log(99, [row("Initial Conditions", 1), row("CPU", "")], {}, tax)
    assert any("Sequence Number" in s and "blank" in s for s in li.issues)


# --------------------------------------------------------------------------- #
# Step types
# --------------------------------------------------------------------------- #
def test_step_types(tax):
    li = synth(
        [
            "Initial Conditions",
            "dupli",
            "Change a direction",
            "all components (final layout)",
            "moving cable",
            "try to remove the power module",
            "CPU and half of the screws on the motherboard",
            "Motherboard screw 1",
        ],
        tax=tax,
    )
    assert [s.step_type for s in li.steps] == [
        "initial",
        "dupli",
        "reorient",
        "ignore",
        "auxiliary",
        "failed",
        "compound",
        "normal",
    ]
    assert li.steps[1].dupli is True and li.steps[0].dupli is False
    # initial / dupli / ignore draft no actions at all.
    for step in (1, 2, 4):
        assert actions_of(li, step) == []
    assert actions_of(li, 3)[0].target == "chassis"
    assert actions_of(li, 3)[0].verb == "reorient"


def test_compound_from_and_in_name(tax):
    li = synth(["Initial Conditions", "Case fan and SATA connector"], tax=tax)
    assert li.steps[1].step_type == "compound"


def test_failed_attempt_action(tax):
    li = synth(["Initial Conditions", "try to remove the power module"], tax=tax)
    a = actions_of(li, 2)[0]
    assert a.result == "failed" and a.failure_reason is None
    assert a.target == "psu.01" and a.verb == "remove"


def test_notes_and_duration(tax):
    li = synth(["Initial Conditions", "Motherboard"], tax=tax, Notes="glued")
    assert li.steps[1].notes == "glued"
    assert all(s.duration_s is None for s in li.steps)


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #
def test_instance_ordinal_follows_first_operation(tax):
    """Ordinals count first operations; the sheet's own numbers never leak in.

    The third row repeats "RAM clip 3" with the same verb, so it is a third
    instance -- two physically different latches are never merged just because
    the annotator reused a number.
    """
    li = synth(["RAM clip 3", "RAM clip 1", "RAM clip 3"], tax=tax)
    assert [a.target for a in li.actions] == [
        "ram_latch.01",
        "ram_latch.02",
        "ram_latch.03",
    ]
    assert li.instances["ram_latch.01"].attrs["sheet_no"] == 3
    assert li.instances["ram_latch.02"].attrs["sheet_no"] == 1
    assert li.instances["ram_latch.03"].raw_names == ["RAM clip 3"]


def test_unnumbered_repeats_are_separate_instances(tax):
    """D61 style: "Case - motherboard connector" seven times = seven parts."""
    li = synth(["Case - motherboard connector"] * 7, tax=tax)
    targets = [a.target for a in li.actions]
    assert targets == [f"connector.front_panel.{i:02d}" for i in range(1, 8)]
    assert len(set(targets)) == 7
    assert all(len(li.instances[t].raw_names) == 1 for t in targets)


def test_restarted_numbering_series_are_separate_instances(tax):
    """D03 style: "CPU fan screw 1..5" then "Heatsink screw 1..4" = 9 screws."""
    names = [f"CPU fan screw {i}" for i in range(1, 6)]
    names += [f"Heatsink screw {i}" for i in range(1, 5)]
    li = synth(names, tax=tax)
    screws = [k for k in li.instances if k.startswith("screw.cpu_cooler.")]
    assert len(screws) == 9
    assert screws[-1] == "screw.cpu_cooler.09"
    assert [a.target for a in li.actions] == screws


def test_same_number_different_cable_owner_is_a_new_instance(tax):
    """D03 style: two "Connector 1" rows whose cables differ are two parts.

    The cable owner is part of the reuse identity, so neither row can absorb
    the other and no attribute value is dropped.
    """
    li = synth(
        ["Connector 1 (HDD - Motherboard)", "Connector 1 (Power Module - Motherboard)"],
        tax=tax,
    )
    first, second = (a.target for a in li.actions)
    assert first != second
    assert li.instances[first].cable == "cable:storage_drive"
    assert li.instances[second].cable == "cable:psu"
    assert not any("reused instance" in s for s in li.issues)


def test_same_number_different_of_is_a_new_instance(tax):
    """A RAM cover is never the heatsink cover, whatever number they share."""
    li = synth(["Open RAM cover 1", "Remove heatsink cover 1"], tax=tax)
    assert [a.target for a in li.actions] == ["cover.01", "cover.02"]
    assert li.instances["cover.01"].attrs["of"] == "ram"
    assert li.instances["cover.02"].attrs["of"] == "cpu_cooler"
    assert not any("reused instance" in s for s in li.issues)
    assert not any("conflicts" in s for s in li.issues)


def test_reuse_needs_number_matching_attrs_and_a_new_verb(tax):
    """Open then remove the same numbered cover = one instance, reported."""
    li = synth(["Open RAM cover 1", "Remove RAM cover 1"], tax=tax)
    assert [a.target for a in li.actions] == ["cover.01", "cover.01"]
    assert [a.verb for a in li.actions] == ["open", "remove"]
    inst = li.instances["cover.01"]
    assert inst.raw_names == ["Open RAM cover 1", "Remove RAM cover 1"]
    assert inst.attrs["of"] == "ram"
    assert any(
        "reused instance cover.01" in s and "prior verbs open" in s for s in li.issues
    )
    assert not any("conflicts" in s for s in li.issues)
    # A third row with a verb already applied starts a new instance instead.
    li = synth(["Open RAM cover 1", "Remove RAM cover 1", "Remove RAM cover 1"], tax=tax)
    assert [a.target for a in li.actions] == ["cover.01", "cover.01", "cover.02"]


def test_identity_covers_every_attribute_that_names_the_part(tax):
    """Role, kind, cable owner, qualifier and ``of`` all split an identity."""
    base = {"role": "motherboard", "cable_owner": "psu", "qualifier": "left", "of": "ram"}
    same = _identity("screw", "motherboard", 1, dict(base))
    assert same == _identity("screw", "motherboard", 1, dict(base, kind="ignored"))
    for attr in ("cable_owner", "qualifier", "of"):
        assert _identity("screw", "motherboard", 1, dict(base, **{attr: "other"})) != same
    assert _identity("screw", "cpu_cooler", 1, dict(base)) != same
    assert _identity("screw", "motherboard", 2, dict(base)) != same


def test_attribute_conflict_on_reuse_is_reported():
    """The reuse-time guard for attributes the identity does not cover.

    No pair of today's ``taxonomy_map`` rules can reach this (every attribute
    that names a part is part of the identity), so the guard is tested at its
    own seam: it exists for later rules and for S1's edits, and it must report
    rather than drop.
    """
    stored = {"of": "ram", "name": "front_io_module", "sheet_no": 1, "captive": True}
    new = {"of": "ram", "name": "rear_io_module", "captive": False, "slot": "a"}
    assert _attr_conflicts(stored, new) == [("name", "front_io_module", "rear_io_module")]
    _merge_attrs(stored, new)
    assert stored["name"] == "front_io_module"  # the stored value wins
    assert stored["captive"] is True  # derived attributes are never merged
    assert stored["slot"] == "a"  # a genuinely new attribute is kept


def test_no_instance_is_operated_with_the_same_verb_twice(tax):
    """The invariant guard, on all three real fixtures."""
    imports = [fixture_import(n, tax)[0] for n in (1, 13, 63)]
    multi, repeat = operation_counts(imports)
    assert repeat == 0
    assert multi == 0  # no fixture row legitimately re-operates a part
    assert not any("INVARIANT" in s for li in imports for s in li.issues)


def test_invariant_guard_fires_when_an_instance_repeats_a_verb(tax):
    li = synth(["Open RAM cover 1", "Remove RAM cover 1"], tax=tax)
    li.actions.append(replace(li.actions[0], step=9))  # simulate a merge bug
    assert operation_counts([li]) == (1, 1)


def test_chassis_instance_always_exists(tax):
    li = synth(["Initial Conditions"], tax=tax)
    chassis = li.instances["chassis"]
    assert chassis.cls == "chassis" and chassis.desktop == 99


def test_captive_heuristic_dell(tax):
    li, _ = fixture_import(13, tax)  # "Optiplex 7020"
    screw = li.instances["screw.cpu_cooler.01"]
    assert screw.attrs["captive"] is True
    assert screw.attrs["captive_source"] == "heuristic"
    assert li.instances["screw.motherboard.01"].attrs["captive"] is False


def test_captive_heuristic_non_dell(tax):
    li, _ = fixture_import(1, tax)  # "HP EliteDesk 800 G2 TWR"
    screw = li.instances["screw.cpu_cooler.01"]
    assert screw.attrs["captive"] is False
    assert screw.attrs["captive_source"] == "heuristic"


def test_connector_socket_host_and_cable(tax):
    li, _ = fixture_import(13, tax)
    mb = li.instances[actions_of(li, 28)[0].target]  # "SSD - motherboard connector 1"
    assert mb.socket_host == "motherboard" and mb.cable == "cable:storage_drive"
    li63, _ = fixture_import(63, tax)
    sata = li63.instances[actions_of(li63, 9)[0].target]  # "SSD - SATA connector"
    assert sata.socket_host == "storage_drive.ssd.01"


def test_connector_socket_host_falls_back_to_motherboard(tax):
    li = synth(["SSD - SATA connector"], tax=tax)  # no storage_drive instance
    assert li.instances["connector.sata_data.01"].socket_host == "motherboard"


def test_raw_names_and_attrs_kept(tax):
    li, _ = fixture_import(13, tax)
    inst = li.instances["screw.cpu_cooler.03"]
    assert inst.raw_names == ["CPU Fan Screw 3"]
    assert inst.attrs["role"] == "cpu_cooler" and inst.desktop == 13


# --------------------------------------------------------------------------- #
# Verbs and tools
# --------------------------------------------------------------------------- #
def test_verb_defaults_by_class(tax):
    li = synth(
        [
            "Motherboard screw 1",
            "Power module - motherboard connector 1",
            "RAM clip 1",
            "CPU locker",
            "CPU Fan Cover",
            "cable locker",
            "Motherboard",
        ],
        tax=tax,
    )
    assert [a.verb for a in li.actions] == [
        "unscrew",
        "disconnect",
        "open",
        "open",
        "open",
        "release",
        "remove",
    ]


def test_verb_hint_wins_over_class_default(tax):
    li = synth(["Remove RAM cover", "Open additional card cover"], tax=tax)
    assert [a.verb for a in li.actions] == ["remove", "open"]


def test_tool_mapping(tax):
    li = synth(["Motherboard screw 1"], tax=tax, Tool_Utility="Screw Driver T15")
    assert li.actions[0].tool == "T15"
    li = synth(["Motherboard screw 1"], tax=tax, Tool_Utility="Philips PH2")
    assert li.actions[0].tool == "PH2"
    li = synth(["Motherboard screw 1"], tax=tax, Tool_Utility="Hand")
    assert li.actions[0].tool == "hand"
    li = synth(["Motherboard"], tax=tax)
    assert li.actions[0].tool == "none"


def test_screw_without_tool_is_unknown_and_reported(tax):
    li = synth(["Initial Conditions", "Motherboard screw 1"], tax=tax)
    a = actions_of(li, 2)[0]
    assert a.tool == "unknown"
    assert any("tool" in s and "step 2" in s for s in li.issues)


# --------------------------------------------------------------------------- #
# Issues
# --------------------------------------------------------------------------- #
def test_compound_row_gets_placeholder_target_and_issue(tax):
    li, _ = fixture_import(63, tax)
    a = actions_of(li, 5)[0]  # "Heatsink screw 1&2&3"
    assert a.target == "screw.cpu_cooler.?"
    assert a.target not in li.instances
    assert any("step 5" in s and "manual" in s for s in li.issues)
    # The compound rows with no class at all still get a flagged placeholder.
    b = actions_of(li, 8)[0]  # "Power module connector and 2 CPU fan connector"
    assert b.target == "connector.?"


def test_auxiliary_row_without_target_class_reported(tax):
    li = synth(["Move CPU fan to reveal connector"], tax=tax)
    assert li.steps[1].step_type == "auxiliary"
    a = actions_of(li, 2)[0]
    assert a.target == "?" and a.verb == "displace"
    assert any("step 2" in s and "no target class" in s for s in li.issues)


def test_fallback_name_reported(tax):
    li = synth(["Initial Conditions", "Frobnicate the widget"], tax=tax)
    assert any("step 2" in s and "fallback" in s for s in li.issues)
    assert actions_of(li, 2)[0].target == "?"


def test_issue_strings_are_prefixed_with_desktop(tax):
    li, _ = fixture_import(63, tax)
    assert li.issues and all(s.startswith("D63 ") for s in li.issues)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _tiny_log(path: Path, extra: str = "") -> None:
    path.write_text(
        "Sequence Number,Sequence Name,Tool Utility\n1,Initial Conditions,\n" + extra,
        encoding="utf-8",
    )


def test_iter_desktop_csvs_skips_sidecars_and_d28_copies(tmp_path):
    for name in (
        "desktop_01.csv",
        "desktop_01_meta.csv",
        "desktop_28_1E9qmg.csv",
        "desktop_28_14CRZX.csv",
        "desktop_meta.csv",
        "notes.txt",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert [(d, p.name) for d, p in iter_desktop_csvs(tmp_path)] == [
        (1, "desktop_01.csv"),
        (28, "desktop_28_1E9qmg.csv"),
    ]


def test_cli_writes_report(tmp_path, capsys):
    _tiny_log(tmp_path / "desktop_01.csv", "2,Motherboard screw 1,\n")
    (tmp_path / "desktop_meta.csv").write_text(
        "desktop_id,n_logged_steps\n1,2\n", encoding="utf-8"
    )
    assert main(["--all", str(tmp_path)]) == 0
    report = (tmp_path / "import_issues.md").read_text(encoding="utf-8")
    assert "# Drive log import issues" in report
    assert "| ok |" in report and "tool set to 'unknown'" in report
    assert "imported 1 desktops" in capsys.readouterr().out


def test_cli_fails_when_the_invariant_is_broken(tmp_path, capsys, monkeypatch):
    _tiny_log(tmp_path / "desktop_01.csv", "2,Motherboard,\n")
    monkeypatch.setattr("tda.core.log_report.operation_counts", lambda imports: (1, 1))
    assert main(["--all", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "1 with a repeated verb" in out and "INVARIANT" in out


def test_cli_reports_step_count_mismatch(tmp_path, capsys):
    _tiny_log(tmp_path / "desktop_01.csv", "2,Motherboard,\n")
    (tmp_path / "desktop_meta.csv").write_text(
        "desktop_id,n_logged_steps\n1,3\n", encoding="utf-8"
    )
    assert main(["--all", str(tmp_path), "--out", str(tmp_path / "issues.md")]) == 1
    assert "mismatch D01: 2 rows vs n_logged_steps 3" in capsys.readouterr().out
    assert "MISMATCH" in (tmp_path / "issues.md").read_text(encoding="utf-8")
