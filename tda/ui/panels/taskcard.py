"""Task card dock: what stepping from frame k to k-1 requires (spec 4.2).

The session turns the state diff between two steps into instructions -- draw
this part back into the chassis, split that keyframe, only flip this state --
and this panel is their checklist: done items are struck through, the first
open one is highlighted and is what the four buttons act on.

The panel decides nothing.  Activating an item only emits
:attr:`TaskCardPanel.sigRequestEdit`; the main window is what calls
``begin_edit``, because it is the one that can refuse -- switching instance
while pixels are uncommitted has to be answerable.  The buttons are the four
commit/confirm calls of spec 4.3, with their keyboard equivalents handled here
so they work while the list has focus.  When ``confirm_frame`` refuses, the
problems that came with ``sigProblems`` are shown instead of any local check.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeyEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tda.ui import session_api as api

__all__ = ["TaskCardPanel", "KIND_ICONS"]

INSTANCE_ROLE = int(Qt.ItemDataRole.UserRole)

#: One glyph per task kind (spec 4.2), so the list can be skimmed vertically.
KIND_ICONS: dict[str, str] = {
    api.KIND_ADD_SHAPE: "✚",  # heavy greek cross: draw a new shape
    api.KIND_SPLIT_KEYFRAME: "✂",  # scissors: split the keyframe
    api.KIND_STATE_ONLY: "≡",  # identical to: shape unchanged
    api.KIND_REMOVE_BENCH_BOX: "⌫",  # erase: the bench box ends here
    api.KIND_CONFIRM: "✔",  # check: nothing to draw
}

#: Rows whose work is already done.
DONE_COLOR = QColor(128, 128, 132)


class TaskCardPanel(QWidget):
    """The per-frame instruction list with the commit and confirm actions."""

    #: An item was activated: the canvas should start editing this instance.
    sigRequestEdit = Signal(str)

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        self._problems: list[str] = []

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setAlternatingRowColors(True)
        self._list.itemActivated.connect(self._on_item_activated)
        self._list.installEventFilter(self)

        self.commit_button = QPushButton("Commit edit (Enter)")
        self.override_button = QPushButton("Commit as frame override (Alt+Enter)")
        self.split_button = QPushButton("Split keyframe (Ctrl+K)")
        self.confirm_button = QPushButton("Confirm frame (Space)")
        self.commit_button.clicked.connect(lambda: self.commit(api.SCOPE_KEYFRAME))
        self.override_button.clicked.connect(
            lambda: self.commit(api.SCOPE_FRAME_OVERRIDE)
        )
        self.split_button.clicked.connect(lambda: self.commit(api.SCOPE_SPLIT))
        self.confirm_button.clicked.connect(self.confirm)

        self._problems_label = QLabel("Problems")
        self._problems_list = QListWidget()
        self._problems_list.setMaximumHeight(90)
        self._problems_label.setVisible(False)
        self._problems_list.setVisible(False)

        buttons = QGridLayout()
        buttons.setContentsMargins(4, 0, 4, 4)
        buttons.addWidget(self.commit_button, 0, 0)
        buttons.addWidget(self.override_button, 0, 1)
        buttons.addWidget(self.split_button, 1, 0)
        buttons.addWidget(self.confirm_button, 1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(QLabel("Task card"))
        layout.addWidget(self._list, 1)
        layout.addWidget(self._problems_label)
        layout.addWidget(self._problems_list)
        layout.addLayout(buttons)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if session is not None:
            self.set_session(session)

    # -- wiring -------------------------------------------------------------
    def set_session(self, session: Optional[api.SessionLike]) -> None:
        """Attach a session (or ``None``) and rebuild the card."""
        if self._session is not None:
            self._session.sigFrameChanged.disconnect(self._on_frame_changed)
            self._session.sigProblems.disconnect(self._on_problems)
        self._session = session
        if session is not None:
            session.sigFrameChanged.connect(self._on_frame_changed)
            session.sigProblems.connect(self._on_problems)
        self._hide_problems()
        self.refresh()

    def list_widget(self) -> QListWidget:
        """The underlying list, for the main window's layout and for tests."""
        return self._list

    # -- content ------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the card from ``session.task_card()``."""
        self._list.clear()
        rows = self._session.task_card() if self._session is not None else []
        first_open = -1
        for i, row in enumerate(rows):
            kind = str(row.get("kind", api.KIND_CONFIRM))
            icon = KIND_ICONS.get(kind, "•")
            instance = str(row.get("instance", ""))
            item = QListWidgetItem(f"{icon}  {instance} — {row.get('text', '')}")
            item.setData(INSTANCE_ROLE, instance)
            item.setToolTip(kind)
            font = item.font()
            done = bool(row.get("done", False))
            if done:
                font.setStrikeOut(True)
                item.setForeground(QBrush(DONE_COLOR))
            elif first_open < 0:
                first_open = i
                font.setBold(True)
            item.setFont(font)
            self._list.addItem(item)
        self._list.setCurrentRow(first_open)

    def current_index(self) -> int:
        """Row of the first item that is not done, or ``-1`` when all are."""
        for row in range(self._list.count()):
            if not self._list.item(row).font().strikeOut():
                return row
        return -1

    def current_instance(self) -> Optional[str]:
        """Instance of the highlighted row, or ``None``."""
        item = self._list.currentItem()
        return None if item is None else str(item.data(INSTANCE_ROLE))

    # -- actions ------------------------------------------------------------
    def commit(self, scope: str) -> None:
        """Commit the current edit with one of :data:`api.COMMIT_SCOPES`."""
        if self._session is not None:
            self._session.commit_edit(scope)

    def confirm(self) -> bool:
        """Confirm the frame; on refusal show the problems the session sent.

        Only the problems that arrived during *this* call are shown: a refusal
        always comes with a fresh ``sigProblems`` (see :class:`api.SessionLike`),
        and showing an older list would attribute another frame's problems to
        this one.
        """
        if self._session is None:
            return False
        self._problems = []
        ok = self._session.confirm_frame()
        if ok:
            self._hide_problems()
        else:
            self._show_problems(self._problems)
        return ok

    def problems(self) -> list[str]:
        """The problems currently on display (empty when none are shown)."""
        if not self._problems_list.isVisibleTo(self):
            return []
        return [
            self._problems_list.item(i).text()
            for i in range(self._problems_list.count())
        ]

    def problems_visible(self) -> bool:
        """Whether the problem list is on display."""
        return self._problems_list.isVisibleTo(self)

    # -- keys ---------------------------------------------------------------
    def handle_key(self, event: QKeyEvent) -> bool:
        """Spec 4.3 keys; ``True`` when the event was consumed."""
        key = event.key()
        mods = event.modifiers()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if mods & Qt.KeyboardModifier.AltModifier:
                self.commit(api.SCOPE_FRAME_OVERRIDE)
            else:
                self.commit(api.SCOPE_KEYFRAME)
            return True
        if key == Qt.Key.Key_K and mods & Qt.KeyboardModifier.ControlModifier:
            self.commit(api.SCOPE_SPLIT)
            return True
        if key == Qt.Key.Key_Space:
            self.confirm()
            return True
        return False

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: D102 - Qt override
        if self.handle_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: D102
        # The list would otherwise swallow Enter (activation) and Space
        # (selection toggle) while it has the focus.
        if obj is self._list and event.type() == QEvent.Type.KeyPress:
            if self.handle_key(event):
                return True
        return super().eventFilter(obj, event)

    # -- slots --------------------------------------------------------------
    def _on_item_activated(self, item: QListWidgetItem) -> None:
        """Report the request; the window decides whether the edit may start.

        The panel used to call ``begin_edit`` itself and then emit, which made
        it impossible to refuse: by the time the window heard about it the
        previous instance's uncommitted pixels were already gone.
        """
        instance = str(item.data(INSTANCE_ROLE))
        if instance:
            self.sigRequestEdit.emit(instance)

    def _on_frame_changed(self, _key: object) -> None:
        self._problems = []
        self._hide_problems()
        self.refresh()

    def _on_problems(self, problems: list) -> None:
        self._problems = [str(p) for p in problems]

    def _show_problems(self, problems: list[str]) -> None:
        self._problems_list.clear()
        for problem in problems:
            self._problems_list.addItem(problem)
        visible = bool(problems)
        self._problems_label.setVisible(visible)
        self._problems_list.setVisible(visible)

    def _hide_problems(self) -> None:
        self._problems_list.clear()
        self._problems_label.setVisible(False)
        self._problems_list.setVisible(False)
