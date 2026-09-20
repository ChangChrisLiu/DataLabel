"""Offscreen tests for the Relations tab (:mod:`tda.ui.panels.relations`).

Stage S6 inside the S1 panel: the edge table, the add row, the decisions about
a rule edge, and the spec 7.4 violations list that updates while the edits are
still staged.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from steps_fixtures import seeded_db

from tda.cli_graph import constraints_into_db
from tda.core.db import Db
from tda.core.graph import edges_from_db
from tda.core.graph_edit import MANUAL, OVERRIDE, RULE
from tda.core.taxonomy import load_taxonomy
from tda.ui.panels.relations import RELATION_COLUMNS, STEP_ROLE
from tda.ui.panels.steptable import StepTablePanel

DESKTOP = 13
FIRST_REMOVE = 10
DRIVE = "storage_drive.ssd.01"
PSU = "psu.01"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_db_path, tax) -> Db:
    conn = seeded_db(tmp_db_path, tax)
    constraints_into_db(conn, tax, {DESKTOP})
    yield conn
    conn.close()


@pytest.fixture
def panel(qapp, db, tmp_path, tax) -> StepTablePanel:
    widget = StepTablePanel(db, DESKTOP, taxonomy=tax, cache_dir=tmp_path / "cache")
    yield widget
    widget.deleteLater()


@pytest.fixture
def tab(panel):
    return panel.relations_tab


def arm(tab, target=DRIVE, kind="blocked_by", blocker=PSU, note="") -> None:
    """Fill the add row the way the annotator would, without a dialog."""
    tab.target_box.setCurrentText(target)
    tab.kind_box.setCurrentText(kind)
    tab.blocker_box.setCurrentText(blocker)
    tab.note_edit.setText(note)


def triples(edges) -> set[tuple[str, str, str]]:
    return {(e.type, e.target, e.blocker) for e in edges}


def errors(tab) -> list[str]:
    seen: list[str] = []
    tab.sigError.connect(seen.append)
    return seen


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_the_panel_has_a_third_tab(panel):
    assert panel.tabs.count() == 3
    assert panel.tabs.tabText(2) == "Relations"


def test_the_table_shows_every_stored_edge_with_its_source(tab, db):
    assert tab.model.rowCount() == len(edges_from_db(db, DESKTOP))
    assert tab.model.columnCount() == len(RELATION_COLUMNS)
    sources = {tab.model.index(row, 0).data()
               for row in range(tab.model.rowCount())}
    assert sources == {RULE}


def test_the_table_is_read_only(tab):
    index = tab.model.index(0, 2)
    assert not (tab.model.flags(index) & Qt.ItemIsEditable)


def test_the_pickers_are_filled_with_the_desktop_s_instances(tab):
    keys = [tab.target_box.itemText(i) for i in range(tab.target_box.count())]
    assert DRIVE in keys and PSU in keys
    assert not [k for k in keys if k.startswith("ls:")]


def test_the_blocker_picker_also_offers_the_cable_nodes(tab):
    keys = [tab.blocker_box.itemText(i) for i in range(tab.blocker_box.count())]
    assert [k for k in keys if k.startswith("cable:")]


def test_only_blocked_by_offers_a_mode(tab):
    tab.kind_box.setCurrentText("covered_by")
    assert not tab.mode_box.isEnabled()
    assert tab.mode_box.currentText() == ""
    tab.kind_box.setCurrentText("blocked_by")
    assert tab.mode_box.isEnabled()
    assert tab.mode_box.currentText() == "physical_path"


# --------------------------------------------------------------------------- #
# adding
# --------------------------------------------------------------------------- #
def test_adding_an_edge_stages_it_and_shows_it(tab, db):
    before = tab.model.rowCount()
    changed = []
    tab.sigChanged.connect(lambda: changed.append(1))
    arm(tab, note="电源挡住硬盘")

    tab.add_edge()

    assert tab.model.rowCount() == before + 1
    assert changed == [1]
    row = tab.model.row_of(DRIVE, "blocked_by", PSU)
    assert row >= 0
    assert tab.model.index(row, 0).data() == MANUAL
    assert tab.model.index(row, 7).data() == "电源挡住硬盘"
    assert ("blocked_by", DRIVE, PSU) not in triples(edges_from_db(db, DESKTOP))


def test_a_refused_edge_says_why_and_stages_nothing(tab):
    seen = errors(tab)
    arm(tab, target=DRIVE, blocker=DRIVE)
    before = tab.model.rowCount()

    tab.add_edge()

    assert seen and "/" in seen[0]
    assert tab.model.rowCount() == before
    assert not tab.data.relations.dirty


def test_a_duplicate_is_refused_with_the_other_edge_s_source(tab):
    seen = errors(tab)
    edge = tab.model.edge_at(0)
    arm(tab, target=edge.target, kind=edge.type, blocker=edge.blocker)

    tab.add_edge()

    assert seen and RULE in seen[0]


def test_the_note_is_cleared_after_a_successful_add(tab):
    arm(tab, note="电源挡住硬盘")
    tab.add_edge()
    assert tab.note_edit.text() == ""


# --------------------------------------------------------------------------- #
# the context menu
# --------------------------------------------------------------------------- #
def labels(menu) -> list[str]:
    return [a.text() for a in menu.actions()]


def test_a_rule_edge_offers_accept_and_reject_but_not_remove(tab):
    menu = tab.edge_menu(0)
    assert any("Accept" in text for text in labels(menu))
    assert any("Reject" in text for text in labels(menu))
    assert not any("Remove" in text for text in labels(menu))


def test_a_manual_edge_offers_remove(tab):
    arm(tab)
    tab.add_edge()
    row = tab.model.row_of(DRIVE, "blocked_by", PSU)
    menu = tab.edge_menu(row)
    assert labels(menu) == ["Remove this manual edge"]


def test_removing_a_manual_edge_takes_it_off_the_table(tab):
    arm(tab)
    tab.add_edge()
    before = tab.model.rowCount()
    tab.remove_edge(DRIVE, "blocked_by", PSU)
    assert tab.model.rowCount() == before - 1
    assert tab.model.row_of(DRIVE, "blocked_by", PSU) == -1


def test_rejecting_a_rule_edge_marks_it_without_dropping_the_row(tab):
    edge = tab.model.edge_at(0)
    before = tab.model.rowCount()

    tab.decide(edge.target, edge.type, edge.blocker, "rejected")

    row = tab.model.row_of(edge.target, edge.type, edge.blocker)
    assert tab.model.rowCount() == before
    assert tab.model.index(row, 0).data() == OVERRIDE
    assert tab.model.index(row, 6).data() == "rejected"


def test_a_decided_edge_offers_to_take_the_decision_back(tab):
    edge = tab.model.edge_at(0)
    tab.decide(edge.target, edge.type, edge.blocker, "rejected")
    row = tab.model.row_of(edge.target, edge.type, edge.blocker)
    assert "Clear the decision" in labels(tab.edge_menu(row))
    tab.decide(edge.target, edge.type, edge.blocker, "proposed")
    row = tab.model.row_of(edge.target, edge.type, edge.blocker)
    assert tab.model.index(row, 0).data() == RULE


# --------------------------------------------------------------------------- #
# Apply / Revert
# --------------------------------------------------------------------------- #
def test_apply_writes_the_staged_edge(tab, panel, db):
    arm(tab, note="电源挡住硬盘")
    tab.add_edge()

    panel.apply()

    assert ("blocked_by", DRIVE, PSU) in triples(edges_from_db(db, DESKTOP))
    assert not panel.data.relations.dirty


def test_revert_discards_the_staged_edge(tab, panel, db):
    arm(tab)
    tab.add_edge()

    panel.revert()

    assert panel.relations_tab.model.row_of(DRIVE, "blocked_by", PSU) == -1
    assert ("blocked_by", DRIVE, PSU) not in triples(edges_from_db(db, DESKTOP))


def test_a_staged_edge_puts_a_line_in_the_status(tab, panel):
    arm(tab)
    tab.add_edge()
    assert "Apply" in panel.status.text()


# --------------------------------------------------------------------------- #
# the violations list
# --------------------------------------------------------------------------- #
def fail_the_first_removal(panel) -> None:
    panel.data.apply_edit(FIRST_REMOVE, "result", "failed")
    panel.data.apply_edit(FIRST_REMOVE, "failure_reason", "blocked_by_part")
    panel.relations_tab.refresh()


def test_the_violations_list_is_empty_on_a_clean_log(tab):
    assert tab.violations.count() == 0


def test_a_staged_failed_attempt_shows_up_in_the_list(tab, panel):
    fail_the_first_removal(panel)
    assert tab.violations.count() == 1
    assert "missing edge?" in tab.violations.item(0).text()
    assert tab.violations.item(0).data(STEP_ROLE) == FIRST_REMOVE


def test_the_right_blocked_by_clears_the_line_before_apply(tab, panel, db):
    fail_the_first_removal(panel)
    arm(tab)

    tab.add_edge()

    assert tab.violations.count() == 0
    panel.apply()
    assert tab.violations.count() == 0
    assert ("blocked_by", DRIVE, PSU) in triples(edges_from_db(db, DESKTOP))


def test_an_s1_edit_refreshes_the_violations_too(tab, panel):
    """The replay is about the staged actions, and a result is a staged action."""
    column = [c.field for c in panel.steps_model.columns].index("result")
    row = panel.steps_model.first_row_of(FIRST_REMOVE)
    panel.steps_model.setData(panel.steps_model.index(row, column), "failed",
                              Qt.EditRole)
    assert tab.violations.count() == 1


def test_double_clicking_a_violation_jumps_the_steps_table(tab, panel):
    fail_the_first_removal(panel)
    seen: list[int] = []
    tab.sigGoToStep.connect(seen.append)

    tab._on_violation_activated(tab.violations.item(0))

    assert seen == [FIRST_REMOVE]
    assert panel.tabs.currentWidget() is panel.steps_view
    assert panel.steps_view.currentIndex().row() == \
        panel.steps_model.first_row_of(FIRST_REMOVE)


def test_the_jump_never_touches_the_session(tab, panel):
    """The gate stays the window's: this moves a table, not the frame."""
    fail_the_first_removal(panel)
    assert not hasattr(panel, "session")
    tab._on_violation_activated(tab.violations.item(0))
    assert panel.data.desktop == DESKTOP
