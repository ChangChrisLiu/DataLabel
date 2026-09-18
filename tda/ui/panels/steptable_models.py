"""Qt table models and item delegates behind the S1 step-table panel.

Split out of :mod:`tda.ui.panels.steptable` so the panel file stays about
layout and commands while this one stays about presenting
:class:`~tda.ui.steps_model.StepTableData` to Qt.

Two models, both editing through the view-model (never through the records):

* :class:`StepTableModel` -- one view row per **action**, so a step that was
  split into several actions shows all of them instead of hiding all but the
  first. The step-level cells (number, type, raw name, thumbnails, notes,
  issues) are filled and editable on the step's first row only.
* :class:`InstanceTableModel` -- one row per instance. Cells that do not apply
  to the row's class (a screw head on a connector, a socket host on a screw)
  are shown empty and are not editable.

Both refuse an edit by returning ``False`` from ``setData`` and emitting
``sigError`` with the message :class:`~tda.ui.steps_model.EditError` carried.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtGui import QPixmap

from PySide6.QtWidgets import QComboBox, QSpinBox, QStyledItemDelegate

from tda.core.taxonomy import Taxonomy
from tda.ui.steps_model import StepTableData
from tda.ui.steps_values import (
    DIFFICULTY_MAX,
    FAILURE_REASONS,
    GROUP_ORDERS,
    RESULTS,
    SCREW_HEADS,
    STEP_TYPES,
    EditError,
    thumb_path,
)

__all__ = [
    "BLANK",
    "INSTANCE_COLUMNS",
    "STEP_COLUMNS",
    "THUMB_PX",
    "THUMB_VIEW",
    "Column",
    "ComboDelegate",
    "DifficultyDelegate",
    "InstanceTableModel",
    "StepTableModel",
]

#: Edge length of a step thumbnail, in device-independent pixels (spec 4.1).
THUMB_PX = 96
#: The view whose frames the step table shows.
THUMB_VIEW = "scan"
BLANK = ""

Choices = Callable[[Taxonomy], Sequence[str]]


@dataclass(frozen=True)
class Column:
    """One table column: its header, the view-model field and its editor."""

    title: str
    field: str
    #: ``label`` (read-only), ``text``, ``combo``, ``spin``, ``check``, ``thumb``.
    kind: str = "text"
    choices: Optional[Choices] = None
    #: ``thumb`` columns only: the step offset of the frame to show.
    offset: int = 0
    #: Steps table only: ``step`` cells belong to the row, ``action`` cells to
    #: one of its actions.
    scope: str = "action"


def _optional(values: Sequence[str]) -> Choices:
    """A combo whose first entry clears the field."""
    return lambda tax: [BLANK, *values]


STEP_COLUMNS: tuple[Column, ...] = (
    Column("Step", "number", "label", scope="step"),
    Column("Type", "step_type", "combo", lambda tax: STEP_TYPES, scope="step"),
    Column("Raw name", "raw_name", "label", scope="step"),
    Column("Before (k-1)", BLANK, "thumb", offset=-1, scope="step"),
    Column("After (k)", BLANK, "thumb", offset=0, scope="step"),
    Column("Target", "target", "text"),
    Column("Verb", "verb", "combo", lambda tax: sorted(tax.verbs)),
    Column("Tool", "tool", "combo", lambda tax: list(tax.tools)),
    Column("Direction", "direction", "combo", lambda tax: list(tax.directions)),
    Column("Result", "result", "combo", lambda tax: list(RESULTS)),
    Column("Failure reason", "failure_reason", "combo", _optional(FAILURE_REASONS)),
    Column("Difficulty", "difficulty", "spin"),
    Column("Notes", "notes", "text", scope="step"),
    Column("Issues", BLANK, "label", scope="step"),
)

INSTANCE_COLUMNS: tuple[Column, ...] = (
    Column("Key", "key", "label"),
    Column("Class", "cls", "label"),
    Column("Parent", "parent", "text"),
    Column("Attached", "attached", "check"),
    Column("Mounted on", "mounted_on", "text"),
    Column("Fastens", "fastens", "text"),
    Column("Socket host", "socket_host", "text"),
    Column("Cable", "cable", "text"),
    Column("Head", "head", "combo", _optional(SCREW_HEADS)),
    Column("Captive", "captive", "check"),
    Column("Group order", "group_order", "combo", lambda tax: list(GROUP_ORDERS)),
    Column("Removal direction", "removal_direction", "combo",
           lambda tax: [BLANK, *tax.directions]),
    Column("Raw names", "raw_names", "label"),
)

#: Instance fields only one class carries (spec 7.1); elsewhere they stay blank.
CLASS_ONLY_FIELDS = {
    "fastens": "screw",
    "head": "screw",
    "captive": "screw",
    "socket_host": "connector",
}

_EDITABLE = frozenset({"text", "combo", "spin"})


# --------------------------------------------------------------------------- #
# delegates
# --------------------------------------------------------------------------- #
class ComboDelegate(QStyledItemDelegate):
    """Drop-down editor whose entries the model narrows down per row.

    The static list is the column's whole vocabulary; ``choices_for`` on the
    model trims it to what is legal for the row being edited (the verbs that
    apply to that target's class, for instance).
    """

    def __init__(self, values: Sequence[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.values = list(values)

    def values_for(self, index: QModelIndex) -> list[str]:
        model = index.model()
        narrowed = model.choices_for(index) if hasattr(model, "choices_for") else None
        return list(narrowed) if narrowed is not None else list(self.values)

    def createEditor(self, parent, option, index) -> QComboBox:  # noqa: N802 - Qt API
        editor = QComboBox(parent)
        editor.addItems(self.values_for(index))
        return editor

    def setEditorData(self, editor: QComboBox, index) -> None:  # noqa: N802 - Qt API
        text = str(index.data(Qt.EditRole) or BLANK)
        position = editor.findText(text)
        editor.setCurrentIndex(position if position >= 0 else 0)

    def setModelData(self, editor: QComboBox, model, index) -> None:  # noqa: N802 - Qt API
        model.setData(index, editor.currentText(), Qt.EditRole)


class DifficultyDelegate(QStyledItemDelegate):
    """Spin box for ``Action.difficulty``; ``0`` means "not recorded"."""

    def createEditor(self, parent, option, index) -> QSpinBox:  # noqa: N802 - Qt API
        editor = QSpinBox(parent)
        editor.setRange(0, DIFFICULTY_MAX)
        editor.setSpecialValueText(BLANK)
        return editor

    def setEditorData(self, editor: QSpinBox, index) -> None:  # noqa: N802 - Qt API
        editor.setValue(int(index.data(Qt.EditRole) or 0))

    def setModelData(self, editor: QSpinBox, model, index) -> None:  # noqa: N802 - Qt API
        value = editor.value()
        model.setData(index, value if value else None, Qt.EditRole)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class _TableModel(QAbstractTableModel):
    """Shared plumbing: columns, headers, the error channel, choice narrowing."""

    #: Emitted with the message of an edit :class:`EditError` refused.
    sigError = Signal(str)

    columns: tuple[Column, ...] = ()

    def __init__(self, data: StepTableData, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.data_model = data

    def set_data(self, data: StepTableData) -> None:
        """Swap in a freshly loaded session (reload / desktop change)."""
        self.beginResetModel()
        self.data_model = data
        self._rebuild()
        self.endResetModel()

    def refresh_structure(self) -> None:
        """Re-read the row layout after actions or instances were added/removed."""
        self.beginResetModel()
        self._rebuild()
        self.endResetModel()

    def _rebuild(self) -> None:
        """Recompute whatever maps view rows onto the view-model."""

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802 - Qt API
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return self.columns[section].title

    def choices_for(self, index: QModelIndex) -> Optional[list[str]]:
        """The combo entries legal for this cell, or ``None`` for "all of them"."""
        return None

    def _refuse(self, error: EditError) -> bool:
        self.sigError.emit(str(error))
        return False

    def _changed(self, index: QModelIndex) -> bool:
        """Repaint the edited row (its issues column moves with it)."""
        row = index.row()
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))
        return True


class StepTableModel(_TableModel):
    """One view row per action, grouped under the step it belongs to."""

    columns = STEP_COLUMNS

    def __init__(
        self,
        data: StepTableData,
        cache_dir: str | Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(data, parent)
        self.cache_dir = Path(cache_dir)
        self._thumbs: dict[tuple[int, int], Optional[QPixmap]] = {}
        #: view row -> (index into ``data.rows``, action index within that step)
        self._map: list[tuple[int, int]] = []
        self._rebuild()

    def set_data(self, data: StepTableData) -> None:
        self._thumbs.clear()
        super().set_data(data)

    def _rebuild(self) -> None:
        self._map = [
            (position, action_idx)
            for position, row in enumerate(self.data_model.rows)
            for action_idx in range(max(len(row.actions), 1))
        ]

    # -- row addressing ----------------------------------------------------- #
    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._map)

    def step_at(self, view_row: int) -> Optional[int]:
        """The logical step shown on a view row, or ``None`` when out of range."""
        if not 0 <= view_row < len(self._map):
            return None
        return self.data_model.rows[self._map[view_row][0]].number

    def action_at(self, view_row: int) -> int:
        """The index within its step of the action shown on a view row."""
        return self._map[view_row][1] if 0 <= view_row < len(self._map) else 0

    def first_row_of(self, step: int) -> int:
        """The first view row of a logical step, or ``-1``."""
        for view_row, (position, action_idx) in enumerate(self._map):
            if action_idx == 0 and self.data_model.rows[position].number == step:
                return view_row
        return -1

    def _at(self, index: QModelIndex):
        position, action_idx = self._map[index.row()]
        return self.data_model.rows[position], action_idx

    # -- Qt ----------------------------------------------------------------- #
    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        column = self.columns[index.column()]
        if column.kind not in _EDITABLE:
            return base
        if column.scope == "step" and self.action_at(index.row()) > 0:
            return base  # a continuation row carries no step-level cell
        return base | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, action_idx = self._at(index)
        column = self.columns[index.column()]
        if role == Qt.DecorationRole:
            if column.kind != "thumb" or action_idx > 0:
                return None
            return self._thumb(row.number + column.offset)
        if role == Qt.ToolTipRole:
            return "\n".join([*row.issues, *row.ls_notes]) or None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self._value(row, action_idx, column, decorate=role == Qt.DisplayRole)
        return None

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:  # noqa: N802
        if not index.isValid() or role != Qt.EditRole:
            return False
        column = self.columns[index.column()]
        if not (self.flags(index) & Qt.ItemIsEditable):
            return False
        row, action_idx = self._at(index)
        try:
            self.data_model.apply_edit(row.number, column.field, value, action_idx=action_idx)
        except EditError as error:
            return self._refuse(error)
        return self._changed(index)

    def choices_for(self, index: QModelIndex) -> Optional[list[str]]:
        """Narrow the verb list to the verbs that apply to this target's class."""
        if self.columns[index.column()].field != "verb":
            return None
        row, action_idx = self._at(index)
        action = row.action(action_idx)
        if action is None:
            return None
        cls = self.data_model.class_of(action.target)
        if cls is None:
            return None
        applicable = [v for v, spec in self.data_model.tax.verbs.items()
                      if cls in spec["applies_to"]]
        return sorted(applicable) or None

    # -- rendering ---------------------------------------------------------- #
    @staticmethod
    def _value(row, action_idx: int, column: Column, decorate: bool = False) -> Any:
        """The cell's value; step-level cells are drawn on the first row only."""
        if column.scope == "step":
            if column.title == "Step":
                return row.number if action_idx == 0 else f"{row.number}.{action_idx + 1}"
            if action_idx > 0:
                return BLANK
            if column.title == "Issues":
                return "; ".join(row.issues)
            if column.kind == "thumb":
                return None
            return getattr(row, column.field)
        action = row.action(action_idx)
        if action is None:
            return None
        value = getattr(action, column.field)
        return BLANK if value is None else value

    def _thumb(self, step: int) -> Optional[QPixmap]:
        """The cached scanner frame of ``step``, scaled and remembered."""
        key = (self.data_model.desktop, step)
        if key not in self._thumbs:
            path = thumb_path(self.cache_dir, self.data_model.desktop, step, THUMB_VIEW)
            pixmap = QPixmap(str(path)) if path is not None else QPixmap()
            self._thumbs[key] = (
                None
                if pixmap.isNull()
                else pixmap.scaled(
                    THUMB_PX, THUMB_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        return self._thumbs[key]


class InstanceTableModel(_TableModel):
    """One row per instance, with the relational attributes of spec 7.1."""

    columns = INSTANCE_COLUMNS

    def __init__(self, data: StepTableData, parent: QObject | None = None) -> None:
        super().__init__(data, parent)
        self.keys: list[str] = data.instance_keys()

    def _rebuild(self) -> None:
        self.keys = self.data_model.instance_keys()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self.keys)

    def key_at(self, view_row: int) -> Optional[str]:
        """The instance key shown on a view row, or ``None`` when out of range."""
        return self.keys[view_row] if 0 <= view_row < len(self.keys) else None

    def _applies(self, index: QModelIndex) -> bool:
        """Does this column's field exist on the row's class at all?"""
        required = CLASS_ONLY_FIELDS.get(self.columns[index.column()].field)
        if required is None:
            return True
        return self.data_model.instances[self.keys[index.row()]].cls == required

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        kind = self.columns[index.column()].kind
        if not self._applies(index):
            return base  # e.g. a screw head on a connector: nothing to type there
        if kind == "check":
            return base | Qt.ItemIsUserCheckable
        if kind in _EDITABLE:
            return base | Qt.ItemIsEditable
        return base

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        inst = self.data_model.instances[self.keys[index.row()]]
        column = self.columns[index.column()]
        if role == Qt.CheckStateRole:
            if column.kind != "check" or not self._applies(index):
                return None
            return Qt.Checked if self._value(inst, column) else Qt.Unchecked
        if role in (Qt.DisplayRole, Qt.EditRole):
            if column.kind == "check" or not self._applies(index):
                return None
            return self._value(inst, column)
        return None

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:  # noqa: N802
        if not index.isValid() or not self._applies(index):
            return False
        column = self.columns[index.column()]
        checking = role == Qt.CheckStateRole and column.kind == "check"
        if not checking and (role != Qt.EditRole or column.kind not in _EDITABLE):
            return False
        if checking:
            value = Qt.CheckState(value) == Qt.Checked
        try:
            self.data_model.apply_instance_edit(self.keys[index.row()], column.field, value)
        except EditError as error:
            return self._refuse(error)
        return self._changed(index)

    @staticmethod
    def _value(inst, column: Column) -> Any:
        if column.field in ("head", "captive"):
            return inst.attrs.get(column.field, BLANK if column.field == "head" else False)
        value = getattr(inst, column.field)
        if column.field == "raw_names":
            return " | ".join(value or [])
        return BLANK if value is None else value
