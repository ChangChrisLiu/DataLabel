"""Stage S6 tab: the constraint graph of one desktop (spec 7, spec 4.1 S6).

The third tab of the S1 panel. It shows every edge of the desktop with the one
thing that decides what may be done to it -- where it came from:

* ``rule`` -- derived from the instance attributes on every ``constraints``
  run. It cannot be edited or deleted here, because the next run would undo
  that silently; ``Accept`` / ``Reject`` record the spec 7.3 decision instead,
  and the re-run keeps it;
* ``manual`` -- the annotator's own edge, mostly the ``blocked_by`` a failed
  attempt needs (spec 7.3 item 3). It is never touched by a re-run and it is
  the only kind that can be removed here;
* ``override`` -- a decision about a rule edge;
* anything else (``labelstudio``) -- imported rows of another kind, shown for
  completeness and left alone.

Below the table sits the spec 7.4 replay of the **staged** session: the same
lines ``constraints --validate`` prints, so adding the right ``blocked_by``
makes a failed attempt's "missing edge?" disappear before ``Apply``, not
after it. Double-clicking a line jumps the Steps table to that step -- inside
the panel only, so the window's uncommitted-edit gate is not involved.

Nothing here writes: every edit is staged in
:class:`~tda.ui.steps_relations.RelationsData` and written by the panel's one
``Apply``, in one transaction with the rest of S1.
"""
from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from tda.core.graph import HARD_TYPES, NECESSITY_ORDER
from tda.core.graph_edit import MANUAL, OVERRIDE, RULE, is_orphan
from tda.core.graph_rules import BLOCKED_GATES, BLOCKED_MODES
from tda.ui.panels.steptable_models import BLANK, Column
from tda.ui.steps_model import EditError, StepTableData
from tda.ui.steps_relations import edge_sort_key

__all__ = ["RELATION_COLUMNS", "RelationTableModel", "RelationsTab"]

#: The type a new edge starts from: the one the rules never derive (spec 7.3).
DEFAULT_KIND = "blocked_by"
#: ``blocked_by`` is the only type that carries a mode; this is its default.
DEFAULT_MODE = BLOCKED_MODES[0]

#: Prefix the note box is prefilled with when an orphaned decision is kept as a
#: manual edge: the rule's own wording was written by the machine, and the row
#: is about to claim a human wrote it.
WAS_RULE = "was rule: "

#: What each mode stops, in the annotator's words -- the table of
#: :data:`~tda.core.graph_rules.BLOCKED_GATES` read out loud.
MODE_TOOLTIP = "blocked_by 的方式决定它挡住哪些动作 / what this block stops:\n" + "\n".join(
    f"  {mode}: {', '.join(sorted(BLOCKED_GATES[mode]))}" for mode in BLOCKED_MODES
)

RELATION_COLUMNS: tuple[Column, ...] = (
    Column("Source", "source", "label"),
    Column("Kind", "type", "label"),
    Column("From (target)", "target", "label"),
    Column("To (blocker)", "blocker", "label"),
    Column("Necessity", "necessity", "label"),
    Column("Mode", "mode", "label"),
    Column("Status", "status", "label"),
    Column("Note", "reason", "label"),
)

#: What the violations list stores on each row so a double-click can jump.
STEP_ROLE = Qt.UserRole + 1

#: The Status column, which must never elide (``accepted_orphan``).
STATUS_COLUMN = [c.field for c in RELATION_COLUMNS].index("status")


class RelationTableModel(QAbstractTableModel):
    """One row per constraint edge, read-only: the edits are commands."""

    columns = RELATION_COLUMNS

    def __init__(self, data: StepTableData, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.data_model = data
        self.edges = data.relations.rows

    def set_data(self, data: StepTableData) -> None:
        """Swap in a freshly loaded session (reload / desktop change)."""
        self.beginResetModel()
        self.data_model = data
        self.edges = data.relations.rows
        self.endResetModel()

    def refresh_structure(self) -> None:
        """Re-read the staged edge list after an edit."""
        self.beginResetModel()
        self.edges = self.data_model.relations.rows
        self.endResetModel()

    def edge_at(self, view_row: int):
        """The edge shown on a view row, or ``None`` when out of range."""
        return self.edges[view_row] if 0 <= view_row < len(self.edges) else None

    def row_of(self, target: str, kind: str, blocker: str) -> int:
        """Where an edge sits in the table, or ``-1``."""
        for row, edge in enumerate(self.edges):
            if (edge.type, edge.target, edge.blocker) == (kind, target, blocker):
                return row
        return -1

    # -- Qt ----------------------------------------------------------------- #
    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self.edges)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return self.columns[section].title

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        edge = self.edge_at(index.row()) if index.isValid() else None
        if edge is None:
            return None
        if role == Qt.ToolTipRole:
            # the keys are long and the columns are not: the whole edge, in
            # words, on every cell of the row
            return f"{edge.label()}\n{edge.source} / {edge.status}" + (
                f"\n{edge.reason}" if edge.reason else "")
        if role in (Qt.DisplayRole, Qt.EditRole):
            value = getattr(edge, self.columns[index.column()].field)
            return BLANK if value is None else str(value)
        return None


class RelationsTab(QWidget):
    """The constraint editor: the edge table, the add row and the violations."""

    #: A refused edit, with the bilingual reason; the panel shows it.
    sigError = Signal(str)
    #: Something was staged (or un-staged): the panel is dirty.
    sigChanged = Signal()
    #: A violation was double-clicked; the panel jumps its Steps table there.
    sigGoToStep = Signal(int)

    def __init__(self, data: StepTableData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self.model = RelationTableModel(data, self)
        self.view = QTableView(self)
        self.violations = QListWidget(self)
        self.cycles_label = QLabel(BLANK, self)
        self.target_box = QComboBox(self)
        self.kind_box = QComboBox(self)
        self.blocker_box = QComboBox(self)
        self.necessity_box = QComboBox(self)
        self.mode_box = QComboBox(self)
        self.note_edit = QLineEdit(self)
        self.add_button = QPushButton("Add edge", self)
        self._build()
        self._connect()
        self.refresh()

    # -- construction ------------------------------------------------------- #
    def _build(self) -> None:
        self.view.setModel(self.model)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)  # the Note column
        # `accepted_orphan` is the longest thing a status ever says, and eliding
        # it to "accepted..." would hide the one word that matters
        self.view.horizontalHeader().setSectionResizeMode(
            STATUS_COLUMN, QHeaderView.ResizeToContents)
        self.view.verticalHeader().setVisible(False)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.cycles_label.setWordWrap(True)

        self.kind_box.addItems(HARD_TYPES)
        self.kind_box.setCurrentText(DEFAULT_KIND)
        self.necessity_box.addItems(NECESSITY_ORDER)
        self.mode_box.addItems([BLANK, *BLOCKED_MODES])
        self.mode_box.setCurrentText(DEFAULT_MODE)
        self.mode_box.setToolTip(MODE_TOOLTIP)
        self.note_edit.setPlaceholderText("为什么 / why (kept as the edge's reason)")

        row = QHBoxLayout()
        for label, widget in (("From", self.target_box), ("Kind", self.kind_box),
                              ("To", self.blocker_box), ("Necessity", self.necessity_box),
                              ("Mode", self.mode_box)):
            row.addWidget(QLabel(label, self))
            row.addWidget(widget)
        row.addWidget(self.note_edit, 1)
        row.addWidget(self.add_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 3)
        layout.addLayout(row)
        layout.addWidget(QLabel("Constraint violations (spec 7.4)", self))
        layout.addWidget(self.violations, 1)
        layout.addWidget(self.cycles_label)

    def _connect(self) -> None:
        self.add_button.clicked.connect(self.add_edge)
        self.kind_box.currentTextChanged.connect(self._on_kind_changed)
        self.target_box.currentTextChanged.connect(self._on_target_changed)
        self.view.customContextMenuRequested.connect(self._context_menu)
        self.violations.itemDoubleClicked.connect(self._on_violation_activated)

    # -- loading ------------------------------------------------------------ #
    def set_data(self, data: StepTableData) -> None:
        """Show another session (desktop change, or ``Revert``)."""
        self.data = data
        self.model.set_data(data)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the derived edges, the pickers, the violations and the cycles."""
        self.model.refresh_structure()
        self._fill_pickers()
        self._fill_violations()
        self._fill_cycles()

    def _fill_pickers(self) -> None:
        """Offer the settled instances only: a draft carries no constraints."""
        for box, keys in ((self.target_box, self.data.relations.instance_keys()),
                          (self.blocker_box, self.data.relations.blocker_keys())):
            current = box.currentText()
            blocked = box.blockSignals(True)
            box.clear()
            box.addItems(keys)
            if current in keys:
                box.setCurrentText(current)
            box.blockSignals(blocked)
        self._on_target_changed(self.target_box.currentText())

    def _fill_violations(self) -> None:
        self.violations.clear()
        for violation in self.data.relations.violations():
            item = QListWidgetItem(violation.text)
            item.setData(STEP_ROLE, int(violation.step))
            self.violations.addItem(item)

    def _fill_cycles(self) -> None:
        """Spec 7.4: no deadlock of actions. Say so either way.

        No single edit can create one -- each is checked -- but two edits either
        side of an S1 correction can, and a database written elsewhere can
        already hold one. A deadlocked graph is one the planner cannot answer
        for, so it is loud here and the ``Apply`` refuses it.
        """
        cycles = self.data.relations.cycles()
        if not cycles:
            self.cycles_label.setStyleSheet("")
            self.cycles_label.setText("Deadlocks: none (spec 7.4 satisfied)")
            return
        named = "; ".join(deadlock.label() for deadlock in cycles)
        self.cycles_label.setStyleSheet("color: #b00020; font-weight: bold;")
        self.cycles_label.setText(
            f"动作死锁 / DEADLOCK, which spec 7.4 forbids and Apply refuses: {named}")

    # -- commands ----------------------------------------------------------- #
    def add_edge(self) -> None:
        """Stage the edge the add row describes."""
        mode = self.mode_box.currentText() or None
        target, kind = self.target_box.currentText(), self.kind_box.currentText()
        blocker = self.blocker_box.currentText()
        self._command(lambda: self.data.relations.add(
            target, kind, blocker, necessity=self.necessity_box.currentText(),
            mode=mode, note=self.note_edit.text().strip(),
        ), target, kind, blocker)

    def remove_edge(self, target: str, kind: str, blocker: str) -> None:
        """Stage the removal of a manual edge."""
        self._command(lambda: self.data.relations.remove(target, kind, blocker))

    def decide(self, target: str, kind: str, blocker: str, decision: str) -> None:
        """Stage the spec 7.3 decision about a rule edge."""
        self._command(lambda: self.data.relations.decide(target, kind, blocker, decision),
                      target, kind, blocker)

    def adopt(self, target: str, kind: str, blocker: str) -> None:
        """Keep an orphaned decision as a manual edge of the annotator's own.

        The row is about to say a human wrote it, so it needs a human's reason:
        the first call prefills the note box with the rule's old machine-written
        wording, marked ``was rule:``, and asks for a word; the second call --
        once something is in the box -- does it.
        """
        note = self.note_edit.text().strip()
        if not note:
            edge = self.model.edge_at(self.model.row_of(target, kind, blocker))
            self.note_edit.setText(f"{WAS_RULE}{edge.reason if edge else ''}")
            self.note_edit.setFocus()
            self.show_edge(target, kind, blocker)
            self.sigError.emit(
                "先写一句你自己的理由，再选一次「保留为手动边」 / say why you believe "
                "it, then choose 'Keep as a manual edge' again -- the rule's "
                "wording was the machine's"
            )
            return
        self._command(lambda: self.data.relations.adopt(target, kind, blocker, note),
                      target, kind, blocker)

    def drop(self, target: str, kind: str, blocker: str) -> None:
        """Clear an orphaned decision: its rule edge is not derived any more."""
        self._command(lambda: self.data.relations.drop(target, kind, blocker))

    def _command(self, run, *show) -> None:
        """Run one staged edit; a refusal is shown and changes nothing.

        ``show`` is the ``(target, kind, blocker)`` of the row the edit is
        about, which is selected and scrolled to afterwards: on a table of
        forty edges the one that just changed is otherwise anybody's guess.
        """
        try:
            run()
        except EditError as error:
            self.sigError.emit(str(error))
            return
        except Exception as error:  # noqa: BLE001 - a bug must not take the tab down
            self.sigError.emit(f"{type(error).__name__}: {error}")
            return
        self.note_edit.clear()
        self.refresh()
        if show:
            self.show_edge(*show)
        self.sigChanged.emit()

    def show_edge(self, target: str, kind: str, blocker: str) -> None:
        """Select one edge's row and bring it into view."""
        row = self.model.row_of(target, kind, blocker)
        if row < 0:
            return
        self.view.selectRow(row)
        self.view.scrollTo(self.model.index(row, 0))

    # -- context menu ------------------------------------------------------- #
    def edge_menu(self, view_row: int) -> QMenu:
        """The table's context menu for one row (not shown yet)."""
        menu = QMenu(self.view)
        edge = self.model.edge_at(view_row)
        if edge is None:
            return menu
        triple = (edge.target, edge.type, edge.blocker)
        if edge.source == MANUAL:
            menu.addAction("Remove this manual edge",
                           lambda: self.remove_edge(*triple))
            return menu
        if edge.source not in (RULE, OVERRIDE):
            return menu  # an imported row of another kind: not ours to decide
        if is_orphan(edge):
            # the rules stopped deriving the edge this decision was about, so
            # there is nothing left to accept or reject
            menu.addAction("Keep as a manual edge", lambda: self.adopt(*triple))
            menu.addAction("Clear the decision", lambda: self.drop(*triple))
            return menu
        if edge.status != "accepted":
            menu.addAction("Accept this rule edge",
                           lambda: self.decide(*triple, "accepted"))
        if edge.status != "rejected":
            menu.addAction("Reject this rule edge",
                           lambda: self.decide(*triple, "rejected"))
        if edge.source == OVERRIDE:
            menu.addAction("Clear the decision",
                           lambda: self.decide(*triple, "proposed"))
        return menu

    def _context_menu(self, pos) -> None:
        menu = self.edge_menu(self.view.indexAt(pos).row())
        if not menu.isEmpty():
            menu.exec(self.view.viewport().mapToGlobal(pos))

    # -- violations --------------------------------------------------------- #
    def _on_kind_changed(self, kind: str) -> None:
        """Only ``blocked_by`` carries a mode (spec 7.1), so only it offers one."""
        blocked = kind == DEFAULT_KIND
        self.mode_box.setEnabled(blocked)
        self.mode_box.setCurrentText(DEFAULT_MODE if blocked else BLANK)

    def _on_target_changed(self, target: str) -> None:
        """Never offer the target as its own blocker: a self edge is refused."""
        if not target or self.blocker_box.currentText() != target:
            return
        for row in range(self.blocker_box.count()):
            if self.blocker_box.itemText(row) != target:
                self.blocker_box.setCurrentIndex(row)
                return

    def _on_violation_activated(self, item: QListWidgetItem) -> None:
        step = item.data(STEP_ROLE)
        if step:
            self.sigGoToStep.emit(int(step))
