"""Task card dock: the work that belongs to the frame on screen (spec 4.2).

The session diffs this frame against the neighbour the annotator came from and
turns the result into instructions -- draw this part back into the chassis, box
that one in the staging area, split this keyframe, only flip that state -- and
this panel is their checklist: done items are struck through, the first open one
is highlighted and is what the four buttons act on.

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

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
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
    api.KIND_ADD_BENCH_BOX: "▭",  # rectangle: box it in the staging area
    api.KIND_REMOVE_BENCH_BOX: "⌫",  # erase: the bench box ends here
    api.KIND_CONFIRM: "✔",  # check: nothing to draw
}

#: Rows whose work is already done.
DONE_COLOR = QColor(128, 128, 132)

#: :attr:`TaskCardPanel.sigCommit` payload meaning "ask the session".
SUGGESTED = ""


class TaskCardPanel(QWidget):
    """The per-frame instruction list with the commit and confirm actions."""

    #: An item was activated: the canvas should start editing this instance.
    sigRequestEdit = Signal(str)
    #: A commit button was pressed; the payload is the scope it asks for, or
    #: :data:`SUGGESTED` for "whatever the session suggests" (the plain Commit).
    sigCommit = Signal(str)
    #: The confirm button was pressed.
    sigConfirm = Signal()

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        self._problems: list[str] = []

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setAlternatingRowColors(True)
        # An instruction is a sentence; the dock's width is not negotiable by it
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setMinimumWidth(160)
        self._list.itemActivated.connect(self._on_item_activated)

        # Short captions in a 2x2 grid, full sentences in the tooltips: laid out
        # in a row with their long names the four buttons asked for 747 px of
        # dock width -- "Commit as frame override (Alt+Enter)" alone is 446 --
        # and the canvas was left with less than half the window.
        self.commit_button = self._button("Commit  ⏎", "Commit the edit (Enter)")
        self.override_button = self._button(
            "This frame  Alt+⏎", "Commit as a frame override (Alt+Enter)")
        self.split_button = self._button(
            "Split  Ctrl+K", "Split the keyframe here (Ctrl+K)")
        self.confirm_button = self._button(
            "Confirm  Space", "Confirm the frame and step back (Space)")
        # The buttons **report**; they do not act.  Calling the session from
        # here made "Confirm" step the frame back over an uncommitted layer --
        # the window never heard about the click, so nothing checked and nothing
        # was said -- and made "Commit" mean ``keyframe`` while the same label's
        # key asked the session what the edit meant.
        self.commit_button.clicked.connect(lambda: self.sigCommit.emit(SUGGESTED))
        self.override_button.clicked.connect(
            lambda: self.sigCommit.emit(api.SCOPE_FRAME_OVERRIDE)
        )
        self.split_button.clicked.connect(lambda: self.sigCommit.emit(api.SCOPE_SPLIT))
        self.confirm_button.clicked.connect(self.sigConfirm.emit)

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

    @staticmethod
    def _button(caption: str, tooltip: str) -> QPushButton:
        """A button that shows its key, explains itself, and stays narrow."""
        button = QPushButton(caption)
        button.setToolTip(tooltip)
        button.setMinimumWidth(1)
        button.setSizePolicy(QSizePolicy.Policy.Ignored,
                             QSizePolicy.Policy.Fixed)
        return button

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
    # There is exactly one key map, and it is not here: the main window's
    # ``tda.ui.app_actions.ACTIONS`` table owns every binding, per mode, and
    # calls the plain methods above.  A second table in the panel is how ``Ctrl+K``
    # on this list came to raise out of ``keyPressEvent`` in Steps mode and how
    # ``Enter`` in Review committed with keyframe scope past ``act_commit``.


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
