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


def test_apply_saves_and_emits_sig_saved(panel, db, tax):
    from tda.ui.steps_model import StepTableData

    model = panel.steps_model
    assert model.setData(model.index(2, _column(model, "Tool")), "PH1", Qt.EditRole)

    saved: list[int] = []
    panel.sigSaved.connect(saved.append)
    panel.apply()

    assert saved == [13]
    assert StepTableData.load(db, 13, tax).row(3).actions[0].tool == "PH1"


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
    before = widget.instances_model.rowCount()
    widget.split_step(5, 3)
    assert widget.instances_model.rowCount() == before + 3
    assert not any("step 5" in widget.issues.item(i).text() for i in range(widget.issues.count()))

    # The row now holds three actions: the display says so, the editor does not.
    index = model.index(4, _column(model, "Target"))
    assert model.data(index, Qt.DisplayRole) == "screw.cpu_cooler.02 (+2)"
    assert model.data(index, Qt.EditRole) == "screw.cpu_cooler.02"
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
