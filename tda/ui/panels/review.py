"""Review dock: the four work queues of review mode (spec 4.4).

Conflicts, needs-review, missing shapes and unexplained differences each get a
tab whose label carries the queue's size, so the remaining work is visible
without opening anything.  Activating an entry opens its frame in the canvas
(``session.goto``); ``Enter`` confirms the frame that is open and ``R`` marks
it for rework, which the panel only reports (:attr:`ReviewPanel.sigRework`) --
what rework means is the main window's business.

A conflict is resolved with the two buttons, which pass the entry's id and one
of :data:`api.RESOLUTIONS` to the session; the queues are then re-read, never
patched locally.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tda.ui import session_api as api

__all__ = ["ReviewPanel", "QUEUE_TITLES"]

STEP_ROLE = int(Qt.ItemDataRole.UserRole)
CID_ROLE = int(Qt.ItemDataRole.UserRole) + 1

#: Tab captions, in :data:`api.QUEUE_NAMES` order.
QUEUE_TITLES: dict[str, str] = {
    api.QUEUE_CONFLICTS: "Conflicts",
    api.QUEUE_NEEDS_REVIEW: "Needs review",
    api.QUEUE_MISSING_SHAPE: "Missing shape",
    api.QUEUE_UNEXPLAINED: "Unexplained",
}


def _entry_text(queue: str, entry: dict) -> str:
    """One line for a queue entry."""
    step = entry.get("step", "?")
    if queue == api.QUEUE_CONFLICTS:
        return (
            f"Step {step} — {entry.get('instance', '')} "
            f"(Δ {entry.get('sym_diff_px', 0)} px)"
        )
    if queue == api.QUEUE_MISSING_SHAPE:
        return f"Step {step} — {entry.get('instance', '')}"
    return f"Step {step}"


class ReviewPanel(QWidget):
    """Queue tabs with the accept / rework / resolve actions of review mode."""

    #: The annotator marked a step for rework.
    sigRework = Signal(int)

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        self._problems: list[str] = []

        self._tabs = QTabWidget()
        self._lists: dict[str, QListWidget] = {}
        for queue in api.QUEUE_NAMES:
            lw = QListWidget()
            lw.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            lw.itemActivated.connect(self._on_item_activated)
            self._lists[queue] = lw
            self._tabs.addTab(lw, QUEUE_TITLES[queue])
        self._tabs.currentChanged.connect(lambda _i: self._sync_buttons())

        self.keep_old_button = QPushButton("Keep old")
        self.accept_new_button = QPushButton("Accept new")
        self.keep_old_button.setToolTip("Discard the conflicting edit")
        self.accept_new_button.setToolTip("Take the edit and re-freeze the frames")
        self.keep_old_button.clicked.connect(
            lambda: self.resolve_selected(api.RESOLVE_KEEP_OLD)
        )
        self.accept_new_button.clicked.connect(
            lambda: self.resolve_selected(api.RESOLVE_ACCEPT_NEW)
        )

        self._problems_label = QLabel("Problems")
        self._problems_list = QListWidget()
        self._problems_list.setMaximumHeight(90)
        self._problems_label.setVisible(False)
        self._problems_list.setVisible(False)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(4, 0, 4, 4)
        buttons.addWidget(self.keep_old_button)
        buttons.addWidget(self.accept_new_button)
        buttons.addStretch(1)
        buttons.addWidget(QLabel("Enter: confirm frame   R: rework"))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._problems_label)
        layout.addWidget(self._problems_list)
        layout.addLayout(buttons)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if session is not None:
            self.set_session(session)

    # -- wiring -------------------------------------------------------------
    def set_session(self, session: Optional[api.SessionLike]) -> None:
        """Attach a session (or ``None``) and re-read the queues."""
        if self._session is not None:
            self._session.sigFrameChanged.disconnect(self._on_frame_changed)
            self._session.sigProblems.disconnect(self._on_problems)
        self._session = session
        if session is not None:
            session.sigFrameChanged.connect(self._on_frame_changed)
            session.sigProblems.connect(self._on_problems)
        self.refresh()

    def tabs(self) -> QTabWidget:
        """The tab widget, for the main window's layout and for tests."""
        return self._tabs

    def list_for(self, queue: str) -> QListWidget:
        """The list of one queue of :data:`api.QUEUE_NAMES`."""
        return self._lists[queue]

    def queue_at(self, index: int) -> str:
        """Name of the queue shown in tab ``index``."""
        return api.QUEUE_NAMES[index]

    def current_queue(self) -> str:
        """Name of the queue currently on top."""
        return self.queue_at(max(0, self._tabs.currentIndex()))

    # -- content ------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read ``session.queues()`` into the four lists."""
        queues = self._session.queues() if self._session is not None else {}
        for index, queue in enumerate(api.QUEUE_NAMES):
            entries = list(queues.get(queue, []))
            lw = self._lists[queue]
            keep = lw.currentRow()
            lw.clear()
            for entry in entries:
                item = QListWidgetItem(_entry_text(queue, entry))
                item.setData(STEP_ROLE, int(entry.get("step", -1)))
                if entry.get("id") is not None:
                    item.setData(CID_ROLE, int(entry["id"]))
                lw.addItem(item)
            if 0 <= keep < lw.count():
                lw.setCurrentRow(keep)
            self._tabs.setTabText(index, f"{QUEUE_TITLES[queue]} ({len(entries)})")
        self._sync_buttons()

    def problems(self) -> list[str]:
        """Problems of the last refused confirmation (empty after a good one)."""
        return list(self._problems)

    # -- actions ------------------------------------------------------------
    def selected_step(self) -> Optional[int]:
        """Step of the selected entry, falling back to the frame being shown."""
        item = self._lists[self.current_queue()].currentItem()
        if item is not None:
            return int(item.data(STEP_ROLE))
        if self._session is not None:
            return int(self._session.current().step)
        return None

    def open_selected(self) -> None:
        """Open the selected entry's frame in the canvas."""
        item = self._lists[self.current_queue()].currentItem()
        if item is not None:
            self._goto(item)

    def resolve_selected(self, resolution: str) -> None:
        """Resolve the selected conflict with ``keep_old`` or ``accept_new``."""
        if self._session is None:
            return
        item = self._lists[api.QUEUE_CONFLICTS].currentItem()
        if item is None:
            return
        cid = item.data(CID_ROLE)
        if cid is None:
            return
        self._session.resolve_conflict(int(cid), resolution)
        self.refresh()

    def confirm(self) -> bool:
        """Accept the frame (``Enter``); on refusal keep the session's problems."""
        if self._session is None:
            return False
        self._problems = []
        ok = self._session.confirm_frame()
        self._show_problems([] if ok else self._problems)
        return ok

    def rework(self) -> None:
        """Mark the selected step for rework (``R``)."""
        step = self.selected_step()
        if step is not None:
            self.sigRework.emit(int(step))

    # -- keys ---------------------------------------------------------------
    # There is exactly one key map, and it is not here: the main window's
    # ``tda.ui.app_actions.ACTIONS`` table owns every binding, per mode, and
    # calls the plain methods above.  A second table in the panel is how ``Ctrl+K``
    # on this list came to raise out of ``keyPressEvent`` in Steps mode and how
    # ``Enter`` in Review committed with keyframe scope past ``act_commit``.


    # -- slots --------------------------------------------------------------
    def _on_item_activated(self, item: QListWidgetItem) -> None:
        self._goto(item)

    def _goto(self, item: QListWidgetItem) -> None:
        step = item.data(STEP_ROLE)
        if self._session is not None and step is not None and int(step) >= 0:
            self._session.goto(int(step))

    def _on_frame_changed(self, _key: object) -> None:
        self.refresh()

    def _on_problems(self, problems: list) -> None:
        self._problems = [str(p) for p in problems]

    def _sync_buttons(self) -> None:
        is_conflicts = self.current_queue() == api.QUEUE_CONFLICTS
        self.keep_old_button.setEnabled(is_conflicts)
        self.accept_new_button.setEnabled(is_conflicts)

    def _show_problems(self, problems: list[str]) -> None:
        self._problems_list.clear()
        for problem in problems:
            self._problems_list.addItem(problem)
        visible = bool(problems)
        self._problems_label.setVisible(visible)
        self._problems_list.setVisible(visible)
