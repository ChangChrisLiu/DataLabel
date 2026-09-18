"""Instance list dock: the layers of the current frame, top-most first.

One row per instance of the compiled frame (spec 4.5, right-hand column): its
palette colour -- the same one the canvas overlay paints it with -- key, class,
state, placement, frame-level visibility (spec 6.2) and a hidden checkbox.

Reordering rows is how the annotator fixes the z-order: the "Move up"/"Move
down" buttons translate a row move into one ``set_zorder_move(instance,
above_of)`` call, and the table is then rebuilt from the session, so the panel
never holds an order of its own.  Buttons rather than drag-and-drop on purpose:
a keyboard-reachable, single-call gesture is easier to undo and to test, and
the pairwise constraint the session stores (spec 3.3) is exactly "A above B".

Keys, on the table or on the panel: ``Ctrl+Up``/``Ctrl+Down`` reorder (the bare
arrows stay with the table's row navigation), ``H`` toggles hidden, ``V``
cycles the visibility values and ``1``-``7`` set one of them directly.

Every mutating gesture is followed by a re-read of ``instance_rows()``, so the
table can never drift from the session's state -- hitting ``H`` after clicking
the checkbox sends the value the session actually holds.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tda.ui import session_api as api
from tda.ui.canvas.overlay import palette_color

__all__ = ["InstanceListPanel"]

KEY_ROLE = int(Qt.ItemDataRole.UserRole)

#: Visibility values are long (``occluded_partial``); the column shows the
#: initials and the row's tooltip spells the value out.
_SHORT_VIS = {"visible": "vis", "occluded_partial": "occ-p", "occluded_full": "occ-f",
              "out_of_view": "out", "too_small": "tiny", "visible_tiny": "v-tiny",
              "motion_blur": "blur"}


def _short(value: str) -> str:
    return _SHORT_VIS.get(value, value[:5])

#: Number keys ``1``-``7`` (spec 6.2 order, see :data:`api.VISIBILITY_VALUES`).
_NUMBER_KEYS = (
    Qt.Key.Key_1,
    Qt.Key.Key_2,
    Qt.Key.Key_3,
    Qt.Key.Key_4,
    Qt.Key.Key_5,
    Qt.Key.Key_6,
    Qt.Key.Key_7,
)


class InstanceListPanel(QWidget):
    """Table of the current frame's instances with z-order and visibility edits."""

    #: An instance was double-clicked: the canvas should start editing it.
    sigRequestEdit = Signal(str)

    #: Four columns, not seven.  Class and placement are in every row's tooltip
    #: instead: the instance key already names the class (``screw.cpu_cooler.01``)
    #: and a 90-part machine needs the dock narrow far more than it needs them
    #: spelled out -- with all seven the table demanded 495 px of dock width.
    COLUMNS: tuple[str, ...] = ("Color", "Instance", "State", "Vis", "Hidden")

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        self._rows: list[dict] = []
        self._loading = False

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        self._table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self._table.setMinimumWidth(180)
        self._table.setWordWrap(False)
        self._table.setTextElideMode(Qt.TextElideMode.ElideLeft)  # keep the ordinal
        self._apply_column_widths()
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.itemDoubleClicked.connect(self._on_item_double_clicked)

        self.up_button = QPushButton("▲ Ctrl+↑")
        self.down_button = QPushButton("▼ Ctrl+↓")
        self.up_button.setToolTip("Move up: put the selected instance above the one over it")
        self.down_button.setToolTip("Move down: put it below the one under it")
        for button in (self.up_button, self.down_button):
            button.setMinimumWidth(1)
        self.up_button.clicked.connect(self.move_up)
        self.down_button.clicked.connect(self.move_down)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(4, 0, 4, 4)
        buttons.addWidget(self.up_button)
        buttons.addWidget(self.down_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        heading = QLabel("Instances — top layer first")
        heading.setToolTip("The frame's instances, top-most layer at the top")
        heading.setMinimumWidth(1)   # a caption must not set the dock's width
        layout.addWidget(heading)
        layout.addWidget(self._table, 1)
        layout.addLayout(buttons)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if session is not None:
            self.set_session(session)

    # -- wiring -------------------------------------------------------------
    def set_session(self, session: Optional[api.SessionLike]) -> None:
        """Attach a session (or ``None``) and rebuild the table."""
        if self._session is not None:
            self._session.sigFrameChanged.disconnect(self._on_frame_changed)
        self._session = session
        if session is not None:
            session.sigFrameChanged.connect(self._on_frame_changed)
        self.refresh()

    def table(self) -> QTableWidget:
        """The underlying table, for the main window's layout and for tests."""
        return self._table

    # -- content ------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the table from ``session.instance_rows()``, keeping the selection.

        The two columns that size themselves to their contents are switched off
        while the rows are written and switched back on once: with them live,
        ``setItem`` re-measures every section on every cell, which on a 90-part
        machine cost ~380 ms per frame change -- more than decoding the 12 MP
        frame itself.
        """
        keep = self.selected_instance()
        self._rows = self._session.instance_rows() if self._session is not None else []
        self._loading = True
        header = self._table.horizontalHeader()
        self._table.setUpdatesEnabled(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        try:
            self._table.clearContents()
            self._table.setRowCount(len(self._rows))
            for row, data in enumerate(self._rows):
                key = str(data.get("key", ""))
                tooltip = (f"{key}\nclass: {data.get('cls', '')}\n"
                           f"placement: {data.get('placement', '')}\n"
                           f"visibility: {data.get('visibility', '')}\nz {data.get('z', '')}")
                swatch = QTableWidgetItem("")
                swatch.setBackground(QBrush(QColor(*palette_color(key))))
                cells = [
                    swatch,
                    QTableWidgetItem(key),
                    QTableWidgetItem(str(data.get("state", ""))),
                    QTableWidgetItem(_short(str(data.get("visibility", "")))),
                ]
                hidden = QTableWidgetItem("")
                hidden.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                hidden.setCheckState(
                    Qt.CheckState.Checked
                    if data.get("hidden", False)
                    else Qt.CheckState.Unchecked
                )
                cells.append(hidden)
                for col, item in enumerate(cells):
                    item.setData(KEY_ROLE, key)
                    item.setToolTip(tooltip)
                    self._table.setItem(row, col, item)
        finally:
            self._loading = False
            self._apply_column_widths()
            self._table.setUpdatesEnabled(True)
        # A removed part has no compiled row any more, so the table shrinks as
        # the teardown proceeds: a selection that is gone must clear rather than
        # leave H / V / 1-7 acting on a key this frame no longer has.
        if keep is None or not self.select_instance(keep):
            self._table.setCurrentCell(-1, -1)

    #: Column widths in pixels; ``None`` means "take what is left" (the key).
    WIDTHS: tuple[Optional[int], ...] = (22, None, 78, 46, 46)

    def _apply_column_widths(self) -> None:
        """Fixed widths, not "resize to contents".

        An instance key is long, and a table that sizes itself to its contents
        reports that length as the dock's preferred width -- which is how the
        canvas ended up with less than half the window.  The key column takes
        whatever is left instead, and the table scrolls when the dock is narrow.
        """
        header = self._table.horizontalHeader()
        for column, width in enumerate(self.WIDTHS):
            if width is None:
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
                continue
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self._table.setColumnWidth(column, width)

    def rows(self) -> list[dict]:
        """The row dicts currently displayed, top layer first."""
        return [dict(r) for r in self._rows]

    # -- selection ----------------------------------------------------------
    def selected_instance(self) -> Optional[str]:
        """Key of the selected row, or ``None``."""
        row = self._table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return str(self._rows[row].get("key", ""))

    def select_instance(self, instance: str) -> bool:
        """Select the row of ``instance``; ``False`` when it is not in the table."""
        for row, data in enumerate(self._rows):
            if data.get("key") == instance:
                self._table.setCurrentCell(row, 1)
                return True
        return False

    # -- actions ------------------------------------------------------------
    def move_up(self) -> None:
        """Move the selected instance one layer up (above its upper neighbour)."""
        row = self._table.currentRow()
        if self._session is None or row <= 0 or row >= len(self._rows):
            return
        self._session.set_zorder_move(
            str(self._rows[row]["key"]), str(self._rows[row - 1]["key"])
        )
        self.refresh()

    def move_down(self) -> None:
        """Move the selected instance one layer down.

        Expressed as "the lower neighbour goes above it", which is the same
        pairwise fact with the arguments the session takes.
        """
        row = self._table.currentRow()
        if self._session is None or row < 0 or row >= len(self._rows) - 1:
            return
        self._session.set_zorder_move(
            str(self._rows[row + 1]["key"]), str(self._rows[row]["key"])
        )
        self.refresh()

    def toggle_hidden(self) -> None:
        """Flip the hidden flag of the selected instance (``H``)."""
        row = self._table.currentRow()
        if self._session is None or row < 0 or row >= len(self._rows):
            return
        data = self._rows[row]
        self._session.set_hidden(str(data["key"]), not bool(data.get("hidden", False)))
        self.refresh()

    def cycle_visibility(self) -> None:
        """Set the next visibility value on the selected instance (``V``)."""
        row = self._table.currentRow()
        if self._session is None or row < 0 or row >= len(self._rows):
            return
        current = str(self._rows[row].get("visibility", api.VISIBILITY_VALUES[0]))
        try:
            index = api.VISIBILITY_VALUES.index(current)
        except ValueError:
            index = -1
        nxt = api.VISIBILITY_VALUES[(index + 1) % len(api.VISIBILITY_VALUES)]
        self.set_visibility(nxt)

    def set_visibility(self, vis: str) -> None:
        """Set ``vis`` on the selected instance (number keys ``1``-``7``)."""
        instance = self.selected_instance()
        if self._session is None or instance is None:
            return
        self._session.set_visibility(instance, vis)
        self.refresh()

    # -- keys ---------------------------------------------------------------
    # There is exactly one key map, and it is not here: the main window's
    # ``tda.ui.app_actions.ACTIONS`` table owns every binding, per mode, and
    # calls the plain methods above.  A second table in the panel is how ``Ctrl+K``
    # on this list came to raise out of ``keyPressEvent`` in Steps mode and how
    # ``Enter`` in Review committed with keyframe scope past ``act_commit``.


    # -- slots --------------------------------------------------------------
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or self._session is None:
            return
        if item.column() != self.COLUMNS.index("Hidden"):
            return
        # Read the item *before* refreshing: the rebuild below deletes it.
        key = str(item.data(KEY_ROLE))
        hidden = item.checkState() == Qt.CheckState.Checked
        self._session.set_hidden(key, hidden)
        self.refresh()

    def _on_item_double_clicked(self, item: QTableWidgetItem) -> None:
        """Report the request only; ``begin_edit`` belongs to the main window.

        Starting the edit here made switching instance unrefusable -- the
        previous instance's uncommitted pixels were dropped before anybody could
        ask about them.
        """
        instance = str(item.data(KEY_ROLE))
        if instance:
            self.sigRequestEdit.emit(instance)

    def _on_frame_changed(self, _key: object) -> None:
        self.refresh()
