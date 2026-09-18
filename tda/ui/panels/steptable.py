"""Stage S1 panel: "步骤与实例核对" -- review the imported log of one desktop.

Two tables behind a :class:`QTabWidget` (spec 4.1):

* **Steps** -- one row per logical step with the scanner thumbnails of frame
  k-1 and k, the parsed target, verb, tool, direction, result, failure reason,
  difficulty and the operator's notes;
* **Instances** -- the desktop's instance table with the relational attributes
  of spec 7.1 (parent/attached, mounted_on, fastens, socket_host, cable, screw
  head and captive flag, group order, removal direction).

Below them sits the list of open questions, re-derived from the drafts on every
edit. ``Apply`` writes everything back and recompiles the automatic state-event
log (emitting :attr:`StepTablePanel.sigSaved`); ``Revert`` reloads from the
database. All editing logic and validation lives in :mod:`tda.ui.steps_model`;
this module only maps it onto Qt models, delegates and widgets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tda.core.db import Db
from tda.core.taxonomy import Taxonomy, load_taxonomy
from tda.ui.steps_model import (
    DIFFICULTY_MAX,
    FAILURE_REASONS,
    GROUP_ORDERS,
    RESULTS,
    SCREW_HEADS,
    STEP_TYPES,
    EditError,
    StepTableData,
    thumb_path,
)

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


def _optional(values: Sequence[str]) -> Choices:
    """A combo whose first entry clears the field."""
    return lambda tax: [BLANK, *values]


STEP_COLUMNS: tuple[Column, ...] = (
    Column("Step", "number", "label"),
    Column("Type", "step_type", "combo", lambda tax: STEP_TYPES),
    Column("Raw name", "raw_name", "label"),
    Column("Before (k-1)", BLANK, "thumb", offset=-1),
    Column("After (k)", BLANK, "thumb", offset=0),
    Column("Target", "target", "text"),
    Column("Verb", "verb", "combo", lambda tax: sorted(tax.verbs)),
    Column("Tool", "tool", "combo", lambda tax: list(tax.tools)),
    Column("Direction", "direction", "combo", lambda tax: list(tax.directions)),
    Column("Result", "result", "combo", lambda tax: list(RESULTS)),
    Column("Failure reason", "failure_reason", "combo", _optional(FAILURE_REASONS)),
    Column("Difficulty", "difficulty", "spin"),
    Column("Notes", "notes", "text"),
    Column("Issues", BLANK, "label"),
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

_EDITABLE = frozenset({"text", "combo", "spin"})


# --------------------------------------------------------------------------- #
# delegates
# --------------------------------------------------------------------------- #
class ComboDelegate(QStyledItemDelegate):
    """Drop-down editor over a fixed vocabulary (verbs, tools, directions...)."""

    def __init__(self, values: Sequence[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.values = list(values)

    def createEditor(self, parent, option, index) -> QComboBox:  # noqa: N802 - Qt API
        editor = QComboBox(parent)
        editor.addItems(self.values)
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
    """Shared plumbing: columns, headers, header data and the error channel."""

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
        self.endResetModel()

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802 - Qt API
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return self.columns[section].title

    def _refuse(self, error: EditError) -> bool:
        self.sigError.emit(str(error))
        return False

    def _changed(self, index: QModelIndex) -> bool:
        """Repaint the edited row (its issues column moves with it)."""
        row = index.row()
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))
        return True


class StepTableModel(_TableModel):
    """One row per logical step; edits go through :class:`StepTableData`."""

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

    def set_data(self, data: StepTableData) -> None:
        self._thumbs.clear()
        super().set_data(data)

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self.data_model.rows)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return Qt.NoItemFlags
        if self.columns[index.column()].kind in _EDITABLE:
            return base | Qt.ItemIsEditable
        return base

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self.data_model.rows[index.row()]
        column = self.columns[index.column()]
        if role == Qt.DecorationRole:
            return self._thumb(row.number + column.offset) if column.kind == "thumb" else None
        if role == Qt.ToolTipRole:
            return "\n".join(row.issues) or None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self._value(row, column, decorate=role == Qt.DisplayRole)
        return None

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:  # noqa: N802
        if not index.isValid() or role != Qt.EditRole:
            return False
        column = self.columns[index.column()]
        if column.kind not in _EDITABLE:
            return False
        try:
            self.data_model.apply_edit(
                self.data_model.rows[index.row()].number, column.field, value
            )
        except EditError as error:
            return self._refuse(error)
        return self._changed(index)

    # -- rendering --------------------------------------------------------- #
    @staticmethod
    def _value(row, column: Column, decorate: bool = False) -> Any:
        """The cell's value; the table edits the row's *first* action.

        A row that holds several actions (a split compound row) shows how many
        more there are behind the first one, but only in the display text --
        the editor must still see the plain stored value.
        """
        if column.title == "Issues":
            return "; ".join(row.issues)
        if column.kind == "thumb":
            return None
        if column.field in ("number", "step_type", "raw_name", "notes"):
            return getattr(row, column.field)
        action = row.action(0)
        if action is None:
            return None
        value = getattr(action, column.field)
        if decorate and column.field == "target" and len(row.actions) > 1:
            return f"{value} (+{len(row.actions) - 1})"
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

    def set_data(self, data: StepTableData) -> None:
        self.beginResetModel()
        self.data_model = data
        self.keys = data.instance_keys()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self.keys)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return Qt.NoItemFlags
        kind = self.columns[index.column()].kind
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
            if column.kind != "check":
                return None
            return Qt.Checked if self._value(inst, column) else Qt.Unchecked
        if role in (Qt.DisplayRole, Qt.EditRole):
            return None if column.kind == "check" else self._value(inst, column)
        return None

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:  # noqa: N802
        if not index.isValid():
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


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
class StepTablePanel(QWidget):
    """Stage S1: review one desktop's step table and instance table."""

    #: Emitted with the desktop id after ``Apply`` wrote the session back.
    sigSaved = Signal(int)

    def __init__(
        self,
        db: Db,
        desktop: int,
        taxonomy: Taxonomy | None = None,
        cache_dir: str | Path = "D:/DataSet/cache",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.desktop = desktop
        self.tax = taxonomy or load_taxonomy()
        self.cache_dir = Path(cache_dir)
        self.data = StepTableData.load(db, desktop, self.tax)

        self.steps_model = StepTableModel(self.data, self.cache_dir, self)
        self.instances_model = InstanceTableModel(self.data, self)
        self.steps_view = self._table(self.steps_model, STEP_COLUMNS, THUMB_PX + 8)
        self.instances_view = self._table(self.instances_model, INSTANCE_COLUMNS)
        self.issues = QListWidget(self)
        self.tabs = QTabWidget(self)
        self._build()
        for model in (self.steps_model, self.instances_model):
            model.sigError.connect(self._show_error)
        self._refresh_issues()

    # -- construction ------------------------------------------------------ #
    def _build(self) -> None:
        self.tabs.addTab(self.steps_view, "Steps")
        self.tabs.addTab(self.instances_view, "Instances")
        self.apply_button = QPushButton("Apply", self)
        self.revert_button = QPushButton("Revert", self)
        self.apply_button.clicked.connect(self.apply)
        self.revert_button.clicked.connect(self.revert)
        self.status = QLabel(BLANK, self)
        self.status.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.revert_button)
        buttons.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, 3)
        layout.addWidget(QLabel("Open questions", self))
        layout.addWidget(self.issues, 1)
        layout.addLayout(buttons)

    def _table(
        self, model: _TableModel, columns: Sequence[Column], row_height: int = 0
    ) -> QTableView:
        view = QTableView(self)
        view.setModel(model)
        view.setSelectionBehavior(QAbstractItemView.SelectItems)
        view.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        view.verticalHeader().setVisible(False)
        if row_height:
            view.verticalHeader().setDefaultSectionSize(row_height)
        for index, column in enumerate(columns):
            delegate = self._delegate(column, view)
            if delegate is not None:
                view.setItemDelegateForColumn(index, delegate)
        return view

    def _delegate(self, column: Column, parent: QWidget) -> Optional[QStyledItemDelegate]:
        if column.kind == "combo" and column.choices is not None:
            return ComboDelegate(column.choices(self.tax), parent)
        if column.kind == "spin":
            return DifficultyDelegate(parent)
        return None

    # -- actions ----------------------------------------------------------- #
    def set_desktop(self, desktop: int) -> None:
        """Load another desktop (or reload this one), dropping unsaved edits."""
        self.desktop = desktop
        self.data = StepTableData.load(self.db, desktop, self.tax)
        self.steps_model.set_data(self.data)
        self.instances_model.set_data(self.data)
        self._refresh_issues()

    def revert(self) -> None:
        """Throw away every unsaved edit and reload from the database."""
        self.set_desktop(self.desktop)
        self.status.setText("Reverted to the stored step table.")

    def apply(self) -> None:
        """Write the session back, recompile the auto events and report."""
        messages = self.data.save(self.db)
        self._refresh_issues()
        self.status.setText(
            f"Saved D{self.desktop:02d}: {len(self.data.issues)} open question(s), "
            f"{len(messages)} state warning(s)."
        )
        self.sigSaved.emit(self.desktop)

    def split_step(self, step: int, n: int) -> None:
        """Split a compound row into ``n`` actions/instances (spec 2.3)."""
        try:
            self.data.split_compound(step, n)
        except EditError as error:
            self._show_error(str(error))
            return
        self.steps_model.set_data(self.data)
        self.instances_model.set_data(self.data)
        self._refresh_issues()

    # -- feedback ---------------------------------------------------------- #
    def _refresh_issues(self) -> None:
        self.issues.clear()
        self.issues.addItems(self.data.issues)
        self.issues.addItems(f"state event: {m}" for m in self.data.messages)

    def _show_error(self, message: str) -> None:
        self.status.setText(f"Rejected: {message}")
