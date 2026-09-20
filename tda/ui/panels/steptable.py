"""Stage S1 panel: "步骤与实例核对" -- review the imported log of one desktop.

Two tables behind a :class:`QTabWidget` (spec 4.1):

* **Steps** -- one row per action, grouped under its logical step, with the
  scanner thumbnails of frames k-1 and k, the parsed target, verb, tool,
  direction, result, failure reason, difficulty and the operator's notes;
* **Instances** -- the desktop's instance table with the relational attributes
  of spec 7.1 (parent/attached, mounted_on, fastens, socket_host, cable, screw
  head and captive flag, group order, removal direction).

Below them sits the list of open questions, re-derived after every edit. The
context menus resolve them in place: split a compound row into N actions, add
or remove an action, retarget a step at a new instance, delete an instance
nothing points at any more. ``Apply`` writes everything back in one transaction
and recompiles the automatic state-event log (emitting
:attr:`StepTablePanel.sigSaved`); ``Revert`` reloads from the database.

The models and delegates live in :mod:`tda.ui.panels.steptable_models` and all
editing logic in :mod:`tda.ui.steps_model`; this module is layout and commands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tda.core.db import Db
from tda.core.taxonomy import Taxonomy, load_taxonomy
from tda.ui.panels.steptable_models import (
    BLANK,
    INSTANCE_COLUMNS,
    STEP_COLUMNS,
    THUMB_PX,
    THUMB_VIEW,
    Column,
    ComboDelegate,
    DifficultyDelegate,
    InstanceTableModel,
    StepTableModel,
)
from tda.ui.steps_model import EditError, StepTableData, thumb_path
from tda.ui.steps_values import DISCRIMINATORS

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
    "StepTablePanel",
    "thumb_path",
]

#: How many targets a compound row is split into unless the annotator says else.
DEFAULT_SPLIT = 2


def _default_cache_dir() -> str:
    """``cache_dir`` from ``configs/paths.yaml``, falling back to the checkout."""
    from tda import pipeline as P

    try:
        return str(P.require(P.load_paths(P.DEFAULT_PATHS_PATH), "cache_dir"))
    except Exception:  # noqa: BLE001 - no config: stay inside the repository
        return str(Path(__file__).resolve().parents[3] / "cache")


class StepTablePanel(QWidget):
    """Stage S1: review one desktop's step table and instance table."""

    #: Emitted with the desktop id after ``Apply`` wrote the session back.
    sigSaved = Signal(int)

    def __init__(
        self,
        db: Db,
        desktop: int,
        taxonomy: Taxonomy | None = None,
        cache_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.desktop = desktop
        self.tax = taxonomy or load_taxonomy()
        # No literal path here: the cache directory is configuration, and a
        # default baked into the panel would write to this machine's real cache
        # from any caller (or test) that forgot to pass one.
        self.cache_dir = Path(cache_dir if cache_dir else _default_cache_dir())
        self.data = StepTableData.load(db, desktop, self.tax)

        self.steps_model = StepTableModel(self.data, self.cache_dir, self)
        self.instances_model = InstanceTableModel(self.data, self)
        self.steps_view = self._table(self.steps_model, STEP_COLUMNS, THUMB_PX + 8)
        self.instances_view = self._table(self.instances_model, INSTANCE_COLUMNS)
        self.issues = QListWidget(self)
        self.tabs = QTabWidget(self)
        self._build()
        self._connect()
        self._refresh_issues()

    # -- construction ------------------------------------------------------ #
    def _build(self) -> None:
        self.tabs.addTab(self.steps_view, "Steps")
        self.tabs.addTab(self.instances_view, "Instances")
        self.apply_button = QPushButton("Apply", self)
        self.revert_button = QPushButton("Revert", self)
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

    def _connect(self) -> None:
        self.apply_button.clicked.connect(self.apply)
        self.revert_button.clicked.connect(self.revert)
        for model in (self.steps_model, self.instances_model):
            model.sigError.connect(self._show_error)
            # Any accepted edit can create or settle an open question.
            model.dataChanged.connect(self._on_data_changed)
            model.modelReset.connect(self._refresh_issues)
        for view, handler in (
            (self.steps_view, self._steps_context_menu),
            (self.instances_view, self._instances_context_menu),
        ):
            view.setContextMenuPolicy(Qt.CustomContextMenu)
            view.customContextMenuRequested.connect(handler)

    def _table(
        self, model, columns: Sequence[Column], row_height: int = 0
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

    # -- loading ----------------------------------------------------------- #
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
        """Write the session back, recompile the auto events and report.

        The save is one transaction, so a failure leaves the database as it
        was; the reason lands in the status line instead of escaping into Qt.
        """
        try:
            messages = self.data.save(self.db)
        except Exception as error:  # a failed save must not take the panel down
            self._show_error(f"could not save D{self.desktop:02d}: {error}")
            return
        self._refresh_issues()
        # A step that became (or stopped being) a `reorient` moved a pose
        # boundary in every view, and every shape of those views is anchored to
        # one. The annotator has to be told on the Apply, not by a renumbered
        # segment turning up under them later.
        recut = ""
        if self.data.recut:
            cut = ", ".join(f"{view} {n}" for view, n in sorted(self.data.recut.items()))
            recut = f" Pose segments re-cut: {cut}."
        self.status.setText(
            f"Saved D{self.desktop:02d}: {len(self.data.issues)} open question(s), "
            f"{len(messages)} state warning(s).{recut}"
        )
        self.sigSaved.emit(self.desktop)

    # -- commands ---------------------------------------------------------- #
    def split_step(self, step: int, n: int) -> None:
        """Split a compound row into ``n`` actions/instances (spec 2.3)."""
        self._command(lambda: self.data.split_compound(step, n),
                      f"Split step {step} into {n} actions.")

    def add_action(self, step: int) -> None:
        """Append one more action to a step."""
        self._command(lambda: self.data.add_action(step), f"Added an action to step {step}.")

    def remove_action(self, step: int, action_idx: int) -> None:
        """Drop one action of a step."""
        self._command(lambda: self.data.remove_action(step, action_idx),
                      f"Removed action {action_idx + 1} of step {step}.")

    def retarget_new(self, step: int, cls: str, disc: str = BLANK, action_idx: int = 0) -> None:
        """Point a step at a brand-new instance of ``cls``.

        ``disc`` is the class's discriminator -- ``role`` for a screw, ``kind``
        for everything else that has one -- and decides which ordinal run the
        new key continues. The view-model checks it against the class's
        vocabulary, so a value from anywhere but the dialog is checked too.
        """
        attrs = {}
        if disc:
            name, _vocabulary = self.data.discriminator_of(cls)
            attrs[name or DISCRIMINATORS[0]] = disc
        self._command(
            lambda: self.data.change_target(step, action_idx=action_idx, cls=cls, attrs=attrs),
            f"Retargeted step {step}.",
        )

    def delete_instance(self, key: str) -> None:
        """Delete an instance nothing references any more."""
        self._command(lambda: self.data.delete_instance(self.db, key), f"Deleted {key}.")

    def _command(self, run, done: str) -> None:
        """Run one structural command, then rebuild both tables and the issues.

        A refused command shows its reason; anything unexpected shows its type
        as well, and neither takes the panel down.
        """
        try:
            run()
        except EditError as error:
            self._show_error(str(error))
            return
        except Exception as error:
            self._show_error(f"{type(error).__name__}: {error}")
            return
        self.steps_model.refresh_structure()
        self.instances_model.refresh_structure()
        self._refresh_issues()
        self.status.setText(done)

    # -- context menus ----------------------------------------------------- #
    def steps_menu(self, view_row: int) -> QMenu:
        """The steps-table context menu for one view row (not shown yet)."""
        menu = QMenu(self.steps_view)
        step = self.steps_model.step_at(view_row)
        if step is None:
            return menu
        action_idx = self.steps_model.action_at(view_row)
        row = self.data.row(step)

        menu.addAction("Add action", lambda: self.add_action(step))
        remove = menu.addAction(
            "Remove action", lambda: self.remove_action(step, action_idx)
        )
        remove.setEnabled(bool(row.actions))
        split = menu.addAction("Split compound into N...", lambda: self._prompt_split(step))
        split.setEnabled(len(row.actions) == 1)
        menu.addAction(
            "Retarget to new instance...", lambda: self._prompt_retarget(step, action_idx)
        )
        return menu

    def instances_menu(self, view_row: int) -> QMenu:
        """The instances-table context menu for one view row (not shown yet)."""
        menu = QMenu(self.instances_view)
        key = self.instances_model.key_at(view_row)
        if key is not None:
            menu.addAction(f"Delete {key}", lambda: self.delete_instance(key))
        return menu

    def _steps_context_menu(self, pos) -> None:
        menu = self.steps_menu(self.steps_view.indexAt(pos).row())
        if not menu.isEmpty():
            menu.exec(self.steps_view.viewport().mapToGlobal(pos))

    def _instances_context_menu(self, pos) -> None:
        menu = self.instances_menu(self.instances_view.indexAt(pos).row())
        if not menu.isEmpty():
            menu.exec(self.instances_view.viewport().mapToGlobal(pos))

    def _prompt_split(self, step: int) -> None:
        count, ok = QInputDialog.getInt(
            self, "Split compound row", f"How many targets does step {step} name?",
            DEFAULT_SPLIT, 2, 99,
        )
        if ok:
            self.split_step(step, count)

    def _prompt_retarget(self, step: int, action_idx: int) -> None:
        """Ask for the class, then for its discriminator from a closed list.

        A class that carries no ``role``/``kind`` -- a motherboard, a CPU --
        skips the second question instead of inviting free text that would
        silently start a parallel ordinal run.
        """
        cls, ok = QInputDialog.getItem(
            self, "New instance", "Taxonomy class:", sorted(self.tax.classes), 0, False
        )
        if not ok or not cls:
            return
        name, vocabulary = self.data.discriminator_of(cls)
        disc = BLANK
        if vocabulary:
            disc, ok = QInputDialog.getItem(
                self, "New instance", f"{name.capitalize()}:", vocabulary, 0, False
            )
            if not ok:
                return
        self.retarget_new(step, cls, disc, action_idx)

    # -- feedback ---------------------------------------------------------- #
    def _on_data_changed(self, *_args) -> None:
        self._refresh_issues()

    def _refresh_issues(self) -> None:
        self.issues.clear()
        self.issues.addItems(self.data.issues)
        self.issues.addItems(f"state event: {m}" for m in self.data.messages)

    def _show_error(self, message: str) -> None:
        self.status.setText(f"Rejected: {message}")
