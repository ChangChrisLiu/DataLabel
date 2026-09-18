"""Offscreen tests for the S1 step-table panel (:mod:`tda.ui.panels.steptable`).

Runs on the ``offscreen`` Qt platform plugin (see ``conftest.py``); the env var
is set at import time as well because the ``QApplication`` lives in a session
fixture that may be built before the function-scoped autouse fixture runs.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox

from tda.core.db import Db
from tda.core.logs import import_log, read_desktop_csv
from tda.core.states import events_from_actions
from tda.core.taxonomy import load_taxonomy
from tda.ui.panels.steptable import (
    INSTANCE_COLUMNS,
    STEP_COLUMNS,
    InstanceTableModel,
    StepTableModel,
    StepTablePanel,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_db_path, tax) -> Db:
    conn = Db(tmp_db_path)
    for desktop in (13, 63):
        rows, meta = read_desktop_csv(FIXTURES / f"desktop_{desktop:02d}.csv")
        imp = import_log(desktop, rows, meta, tax)
        conn.upsert_desktop(desktop, {"brand": meta.get("brand_model_raw") or ""})
        conn.replace_steps(desktop, imp.steps, imp.actions)
        for inst in imp.instances.values():
            conn.upsert_instance(inst)
        conn.replace_events(desktop, events_from_actions(imp.instances, imp.actions, tax))
    yield conn
    conn.close()


@pytest.fixture
def panel(qapp, db, tmp_path, tax) -> StepTablePanel:
    widget = StepTablePanel(db, 13, taxonomy=tax, cache_dir=tmp_path / "cache")
    yield widget
    widget.deleteLater()


def _column(model, title: str) -> int:
    return [c.title for c in model.columns].index(title)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_panel_builds_with_two_tabs(panel):
    assert panel.tabs.count() == 2
    assert panel.tabs.tabText(0) == "Steps"
    assert panel.tabs.tabText(1) == "Instances"


def test_step_model_row_and_column_counts(panel):
    model = panel.steps_model
    assert model.rowCount() == 42
    assert model.columnCount() == len(STEP_COLUMNS)
    assert model.rowCount(model.index(0, 0)) == 0  # flat, not a tree


def test_instance_model_row_and_column_counts(panel):
    model = panel.instances_model
    assert model.rowCount() == len(panel.data.instances)
    assert model.columnCount() == len(INSTANCE_COLUMNS)


def test_the_panel_uses_the_published_model_classes(panel):
    assert isinstance(panel.steps_model, StepTableModel)
    assert isinstance(panel.instances_model, InstanceTableModel)


def test_step_model_headers_are_the_column_titles(panel):
    model = panel.steps_model
    titles = [
        model.headerData(c, Qt.Horizontal, Qt.DisplayRole) for c in range(model.columnCount())
    ]
    assert titles == [c.title for c in STEP_COLUMNS]


def test_step_model_shows_the_parsed_draft(panel):
    model = panel.steps_model
    row = 2  # logical step 3
    assert model.index(row, _column(model, "Step")).data() == 3
    assert model.index(row, _column(model, "Raw name")).data() == "CPU Fan Screw 1"
    assert model.index(row, _column(model, "Target")).data() == "screw.cpu_cooler.01"
    assert model.index(row, _column(model, "Verb")).data() == "unscrew"
    assert model.index(row, _column(model, "Tool")).data() == "PH2"


def test_editable_columns_carry_the_edit_flag(panel):
    model = panel.steps_model
    editable = model.index(2, _column(model, "Verb"))
    read_only = model.index(2, _column(model, "Step"))
    assert model.flags(editable) & Qt.ItemIsEditable
    assert not (model.flags(read_only) & Qt.ItemIsEditable)


# --------------------------------------------------------------------------- #
# editing through the model
# --------------------------------------------------------------------------- #
def test_set_data_updates_the_view_model(panel):
    model = panel.steps_model
    index = model.index(2, _column(model, "Tool"))
    assert model.setData(index, "PH1", Qt.EditRole)
    assert panel.data.row(3).actions[0].tool == "PH1"
    assert index.data() == "PH1"


def test_set_data_rejects_an_invalid_edit_and_reports_it(panel):
    model = panel.steps_model
    index = model.index(2, _column(model, "Verb"))
    errors = []
    model.sigError.connect(errors.append)
    assert not model.setData(index, "disconnect", Qt.EditRole)
    assert panel.data.row(3).actions[0].verb == "unscrew"
    assert errors and "screw" in errors[0]


def test_set_data_edits_the_step_notes(panel):
    model = panel.steps_model
    index = model.index(2, _column(model, "Notes"))
    assert model.setData(index, "stripped head", Qt.EditRole)
    assert panel.data.row(3).notes == "stripped head"


def test_set_data_edits_the_difficulty(panel):
    model = panel.steps_model
    index = model.index(2, _column(model, "Difficulty"))
    assert model.setData(index, 4, Qt.EditRole)
    assert panel.data.row(3).actions[0].difficulty == 4


def test_instance_model_edits_a_relational_attribute(panel):
    model = panel.instances_model
    row = model.keys.index("screw.cpu_cooler.01")
    index = model.index(row, _column(model, "Parent"))
    assert model.setData(index, "cpu_cooler.fan.01", Qt.EditRole)
    assert panel.data.instances["screw.cpu_cooler.01"].parent == "cpu_cooler.fan.01"


def test_instance_model_rejects_an_invalid_attribute(panel):
    model = panel.instances_model
    row = model.keys.index("screw.cpu_cooler.01")
    index = model.index(row, _column(model, "Head"))
    errors = []
    model.sigError.connect(errors.append)
    assert not model.setData(index, "PH9", Qt.EditRole)
    assert errors


def test_instance_model_toggles_attached_with_a_check_box(panel):
    model = panel.instances_model
    row = model.keys.index("screw.cpu_cooler.01")
    index = model.index(row, _column(model, "Attached"))
    assert model.setData(index, Qt.Checked, Qt.CheckStateRole)
    assert panel.data.instances["screw.cpu_cooler.01"].attached is True


# --------------------------------------------------------------------------- #
# delegates
# --------------------------------------------------------------------------- #
def test_verb_column_edits_with_a_combo_box(panel, qapp):
    model = panel.steps_model
    column = _column(model, "Verb")
    delegate = panel.steps_view.itemDelegateForColumn(column)
    editor = delegate.createEditor(panel.steps_view, None, model.index(2, column))
    assert isinstance(editor, QComboBox)
    assert "unscrew" in [editor.itemText(i) for i in range(editor.count())]
    editor.deleteLater()


def test_difficulty_column_edits_with_a_spin_box(panel):
    model = panel.steps_model
    column = _column(model, "Difficulty")
    delegate = panel.steps_view.itemDelegateForColumn(column)
    editor = delegate.createEditor(panel.steps_view, None, model.index(2, column))
    assert isinstance(editor, QSpinBox)
    assert (editor.minimum(), editor.maximum()) == (0, 5)
    editor.deleteLater()


# --------------------------------------------------------------------------- #
# thumbnails
# --------------------------------------------------------------------------- #
def test_thumbnail_cells_are_empty_when_the_cache_is_missing(panel):
    model = panel.steps_model
    for title in ("Before (k-1)", "After (k)"):
        index = model.index(2, _column(model, title))
        assert model.data(index, Qt.DecorationRole) is None


def test_thumbnail_cells_load_a_cached_frame(qapp, db, tmp_path, tax):
    cache = tmp_path / "cache"
    folder = cache / "scan" / "D13"
    folder.mkdir(parents=True)
    QImage(120, 90, QImage.Format_RGB32).save(str(folder / "s003.png"))

    widget = StepTablePanel(db, 13, taxonomy=tax, cache_dir=cache)
    model = widget.steps_model
    index = model.index(2, _column(model, "After (k)"))
    pixmap = model.data(index, Qt.DecorationRole)
    assert pixmap is not None
    assert max(pixmap.width(), pixmap.height()) == 96
    widget.deleteLater()


# --------------------------------------------------------------------------- #
# issues, apply and revert
# --------------------------------------------------------------------------- #
def test_issue_list_is_empty_for_a_clean_desktop(panel):
    assert panel.issues.count() == 0


def test_issue_list_shows_the_compound_rows(qapp, db, tmp_path, tax):
    widget = StepTablePanel(db, 63, taxonomy=tax, cache_dir=tmp_path / "cache")
    assert widget.issues.count() > 0
    assert "step 5" in widget.issues.item(0).text()
    widget.deleteLater()


def test_issue_list_grows_when_an_edit_creates_an_issue(panel):
    """The list is re-derived after every accepted edit, not only on load."""
    model = panel.steps_model
    assert panel.issues.count() == 0
    assert model.setData(model.index(2, _column(model, "Result")), "failed", Qt.EditRole)
    texts = [panel.issues.item(i).text() for i in range(panel.issues.count())]
    assert any("failure_reason" in t for t in texts)

    assert model.setData(
        model.index(2, _column(model, "Failure reason")), "fastener_stuck", Qt.EditRole
    )
    assert panel.issues.count() == 0


def test_issue_list_reports_an_orphaned_instance_after_a_retarget(panel):
    model = panel.steps_model
    index = model.index(3, _column(model, "Target"))  # step 4
    assert model.setData(index, "screw.cpu_cooler.01", Qt.EditRole)
    texts = [panel.issues.item(i).text() for i in range(panel.issues.count())]
    assert any("no action references screw.cpu_cooler.02" in t for t in texts)


def test_instance_issue_clears_once_the_orphan_is_deleted(panel):
    model = panel.steps_model
    assert model.setData(
        model.index(3, _column(model, "Target")), "screw.cpu_cooler.01", Qt.EditRole
    )
    panel.delete_instance("screw.cpu_cooler.02")
    assert panel.issues.count() == 0
    assert "screw.cpu_cooler.02" not in panel.data.instances


def test_deleting_a_referenced_instance_is_refused_in_the_status_line(panel):
    panel.delete_instance("screw.cpu_cooler.02")  # step 4 still targets it
    assert "Rejected" in panel.status.text()
    assert "screw.cpu_cooler.02" in panel.data.instances


# --------------------------------------------------------------------------- #
# per-row vocabularies
# --------------------------------------------------------------------------- #
def test_verb_combo_only_offers_the_verbs_of_the_row_class(panel):
    model = panel.steps_model
    column = _column(model, "Verb")
    delegate = panel.steps_view.itemDelegateForColumn(column)

    screw = delegate.values_for(model.index(2, column))  # step 3: a screw
    assert screw == ["remove", "unscrew"]
    connector = delegate.values_for(model.index(7, column))  # step 8: a connector
    assert connector == ["disconnect", "remove"]


def test_verb_combo_falls_back_to_every_verb_for_an_unresolved_target(qapp, db, tmp_path, tax):
    widget = StepTablePanel(db, 63, taxonomy=tax, cache_dir=tmp_path / "cache")
    model = widget.steps_model
    column = _column(model, "Verb")
    delegate = widget.steps_view.itemDelegateForColumn(column)
    row = model.first_row_of(19)  # 'SATA 1 and SATA 2' -> connector.?
    assert "disconnect" in delegate.values_for(model.index(row, column))
    widget.deleteLater()


def test_instance_attributes_that_do_not_apply_are_not_editable(panel):
    model = panel.instances_model
    connector = model.keys.index("connector.01")
    screw = model.keys.index("screw.cpu_cooler.01")

    head = _column(model, "Head")
    assert not (model.flags(model.index(connector, head)) & Qt.ItemIsEditable)
    assert model.flags(model.index(screw, head)) & Qt.ItemIsEditable
    assert model.index(connector, head).data() is None

    socket = _column(model, "Socket host")
    assert not (model.flags(model.index(screw, socket)) & Qt.ItemIsEditable)
    assert model.flags(model.index(connector, socket)) & Qt.ItemIsEditable

    captive = _column(model, "Captive")
    assert not (model.flags(model.index(connector, captive)) & Qt.ItemIsUserCheckable)
    assert not model.setData(model.index(connector, captive), Qt.Checked, Qt.CheckStateRole)


# --------------------------------------------------------------------------- #
# multi-action rows and the context menus
# --------------------------------------------------------------------------- #
def test_step_level_cells_are_blank_on_a_continuation_row(panel):
    model = panel.steps_model
    panel.add_action(3)
    first = model.first_row_of(3)
    raw = _column(model, "Raw name")
    assert model.index(first, raw).data() == "CPU Fan Screw 1"
    assert model.index(first + 1, raw).data() == ""
    assert not (model.flags(model.index(first + 1, raw)) & Qt.ItemIsEditable)
    assert model.flags(model.index(first + 1, _column(model, "Tool"))) & Qt.ItemIsEditable


def test_editing_a_continuation_row_reaches_the_second_action(panel):
    model = panel.steps_model
    panel.add_action(3)
    first = model.first_row_of(3)
    assert model.setData(model.index(first + 1, _column(model, "Tool")), "hand", Qt.EditRole)
    assert [a.tool for a in panel.data.row(3).actions] == ["PH2", "hand"]


def test_remove_action_shrinks_the_table(panel):
    model = panel.steps_model
    panel.add_action(3)
    rows = model.rowCount()
    panel.remove_action(3, 1)
    assert model.rowCount() == rows - 1
    assert len(panel.data.row(3).actions) == 1


def test_steps_context_menu_offers_the_structural_commands(panel):
    menu = panel.steps_menu(2)
    labels = [a.text() for a in menu.actions()]
    assert labels == [
        "Add action", "Remove action", "Split compound into N...",
        "Retarget to new instance...",
    ]
    assert menu.actions()[1].isEnabled()  # step 3 has an action to remove
    menu.deleteLater()


def test_steps_context_menu_disables_remove_on_a_marker_row(panel):
    menu = panel.steps_menu(0)  # step 1 is the `initial` marker: no action
    assert not menu.actions()[1].isEnabled()
    menu.deleteLater()


def test_steps_context_menu_is_empty_outside_the_table(panel):
    assert panel.steps_menu(-1).isEmpty()


def test_split_menu_entry_asks_for_the_count(qapp, db, tmp_path, tax, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    widget = StepTablePanel(db, 63, taxonomy=tax, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(lambda *a, **k: (3, True)))
    row = widget.steps_model.first_row_of(5)
    widget.steps_menu(row).actions()[2].trigger()
    assert len(widget.data.row(5).actions) == 3
    widget.deleteLater()


def _fake_get_item(monkeypatch, answers: list[str], asked: list):
    """Answer successive ``QInputDialog.getItem`` calls, recording the choices."""
    from PySide6.QtWidgets import QInputDialog

    replies = iter(answers)

    def fake(parent, title, label, items, current=0, editable=False):
        asked.append((label, list(items)))
        return next(replies), True

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(fake))
    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **k: pytest.fail("the retarget dialog must not ask for free text")),
    )


def test_retarget_menu_entry_offers_the_class_vocabulary(panel, monkeypatch):
    asked: list = []
    _fake_get_item(monkeypatch, ["screw", "motherboard"], asked)
    panel.steps_menu(2).actions()[3].trigger()

    assert panel.data.row(3).actions[0].target == "screw.motherboard.07"
    assert "screw.motherboard.07" in panel.instances_model.keys
    assert asked[0][0] == "Taxonomy class:"
    assert asked[1][0] == "Role:"
    assert asked[1][1] == ["motherboard", "cpu_cooler", "cooler_bracket", "drive",
                           "optical_drive", "card", "psu", "other"]


def test_retarget_menu_entry_skips_the_vocabulary_for_a_class_without_one(panel, monkeypatch):
    asked: list = []
    _fake_get_item(monkeypatch, ["motherboard"], asked)
    panel.steps_menu(41).actions()[3].trigger()  # step 42

    assert len(asked) == 1  # a motherboard has no role and no kind
    assert panel.data.row(42).actions[0].target == "motherboard.02"


def test_retarget_refuses_a_discriminator_outside_the_vocabulary(panel):
    panel.retarget_new(3, "screw", "mainboard")
    assert "Rejected" in panel.status.text()
    assert panel.data.row(3).actions[0].target == "screw.cpu_cooler.01"


def test_a_command_reports_an_unexpected_exception(panel, monkeypatch):
    monkeypatch.setattr(
        panel.data, "add_action",
        lambda step: (_ for _ in ()).throw(RuntimeError("the disk went away")),
    )
    panel.add_action(3)
    assert "RuntimeError" in panel.status.text()
    assert "the disk went away" in panel.status.text()


def test_the_issues_cell_of_a_split_row_repaints_from_a_later_action(panel):
    """Step-level cells live on the first row, so the whole step must repaint."""
    model = panel.steps_model
    panel.add_action(3)
    first = model.first_row_of(3)
    spans: list[tuple[int, int]] = []
    model.dataChanged.connect(lambda tl, br, *_: spans.append((tl.row(), br.row())))

    assert model.setData(model.index(first + 1, _column(model, "Result")), "failed", Qt.EditRole)
    assert any(top <= first <= bottom for top, bottom in spans)
    assert "failure_reason" in model.index(first, _column(model, "Issues")).data()


def test_instances_context_menu_deletes_the_row(panel):
    model = panel.instances_model
    steps = panel.steps_model
    assert steps.setData(
        steps.index(3, _column(steps, "Target")), "screw.cpu_cooler.01", Qt.EditRole
    )
    menu = panel.instances_menu(model.keys.index("screw.cpu_cooler.02"))
    assert menu.actions()[0].text() == "Delete screw.cpu_cooler.02"
    menu.actions()[0].trigger()
    assert "screw.cpu_cooler.02" not in panel.data.instances
    menu.deleteLater()


# --------------------------------------------------------------------------- #
# notes
# --------------------------------------------------------------------------- #
def test_notes_cell_hides_and_preserves_the_label_studio_line(panel):
    model = panel.steps_model
    panel.data.row(3).step.notes = "tight\nLS: difficulty=hard"
    column = _column(model, "Notes")
    assert model.index(2, column).data() == "tight"
    assert "LS: difficulty=hard" in model.data(model.index(2, column), Qt.ToolTipRole)

    assert model.setData(model.index(2, column), "stripped head", Qt.EditRole)
    assert panel.data.row(3).step.notes == "stripped head\nLS: difficulty=hard"


def test_apply_saves_and_emits_sig_saved(panel, db, tax):
    from tda.ui.steps_model import StepTableData

    model = panel.steps_model
    assert model.setData(model.index(2, _column(model, "Tool")), "PH1", Qt.EditRole)

    saved: list[int] = []
    panel.sigSaved.connect(saved.append)
    panel.apply()

    assert saved == [13]
    assert StepTableData.load(db, 13, tax).row(3).actions[0].tool == "PH1"


def test_apply_reports_a_failed_save_instead_of_raising(panel, db, monkeypatch):
    saved: list[int] = []
    panel.sigSaved.connect(saved.append)
    monkeypatch.setattr(
        panel.data, "save", lambda _db: (_ for _ in ()).throw(RuntimeError("disk full"))
    )
    panel.apply()
    assert "disk full" in panel.status.text()
    assert saved == []


def test_revert_drops_unsaved_edits(panel):
    model = panel.steps_model
    assert model.setData(model.index(2, _column(model, "Tool")), "PH1", Qt.EditRole)
    panel.revert()
    assert panel.data.row(3).actions[0].tool == "PH2"
    assert panel.steps_model.index(2, _column(panel.steps_model, "Tool")).data() == "PH2"


def test_panel_paints_every_row(qapp, db, tmp_path, tax):
    """Render the whole table offscreen: delegates and thumbnails must not raise."""
    cache = tmp_path / "cache"
    folder = cache / "scan" / "D13"
    folder.mkdir(parents=True)
    for step in (2, 3):
        QImage(60, 40, QImage.Format_RGB32).save(str(folder / f"s{step:03d}.png"))

    widget = StepTablePanel(db, 13, taxonomy=tax, cache_dir=cache)
    widget.resize(1400, 700)
    widget.show()
    widget.steps_view.scrollToBottom()
    qapp.processEvents()
    assert not widget.steps_view.grab().isNull()
    widget.tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert not widget.instances_view.grab().isNull()
    widget.deleteLater()


def test_split_step_refreshes_both_tables(qapp, db, tmp_path, tax):
    widget = StepTablePanel(db, 63, taxonomy=tax, cache_dir=tmp_path / "cache")
    model = widget.steps_model
    instances_before = widget.instances_model.rowCount()
    rows_before = model.rowCount()
    widget.split_step(5, 3)

    assert widget.instances_model.rowCount() == instances_before + 3
    assert model.rowCount() == rows_before + 2  # one row per action, not one per step
    assert not any("step 5" in widget.issues.item(i).text() for i in range(widget.issues.count()))

    # Every action of the split row is visible; none is hidden behind the first.
    target = _column(model, "Target")
    first = model.first_row_of(5)
    assert [model.index(first + i, target).data() for i in range(3)] == [
        "screw.cpu_cooler.02", "screw.cpu_cooler.03", "screw.cpu_cooler.04",
    ]
    step_column = _column(model, "Step")
    assert [model.index(first + i, step_column).data() for i in range(3)] == [5, "5.2", "5.3"]
    widget.deleteLater()


def test_split_step_reports_a_refused_split(qapp, db, tmp_path, tax):
    widget = StepTablePanel(db, 13, taxonomy=tax, cache_dir=tmp_path / "cache")
    widget.split_step(3, 2)  # step 3 is not a compound row
    assert "Rejected" in widget.status.text()
    widget.deleteLater()


def test_set_desktop_reloads_the_panel(panel):
    panel.set_desktop(63)
    assert panel.desktop == 63
    assert panel.steps_model.rowCount() == 37
    assert panel.issues.count() > 0
