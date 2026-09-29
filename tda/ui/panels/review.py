"""Review dock: the four work queues of review mode (spec 4.4).

Conflicts, needs-review, missing shapes and unexplained differences each get a
tab whose label carries the queue's size, so the remaining work is visible
without opening anything.  Activating an entry asks for its frame; ``Enter``
confirms the frame that is open and ``R`` marks it for rework.

The panel decides nothing: the lists **report**
(:attr:`ReviewPanel.sigOpenStep`, :attr:`ReviewPanel.sigRework`) and the window
acts.  Opening another frame can be refused, and a resolution has three
possible outcomes; neither answer is the panel's to give.  The queues are then
re-read, never patched locally.

It has **no buttons** of its own (task U2b round 1b): the "Keep old K" / "Take
new N" pair duplicated the tool palette's 保留旧的 / 采用新的, and two places for
one thing is the confusion the second trial reported.  The panel says which
conflict is selected (:meth:`ReviewPanel.selected_conflict`); the palette's
buttons and the keys resolve it through the window.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tda.ui import session_api as api
from tda.ui.panels import session_is_open

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
        # A disagreement about a *label* moves no pixels: "visibility: visible
        # -> occluded_partial" is 0 differing pixels, and the row read "(Δ 0
        # px)" -- an entry somebody has to settle, described as nothing having
        # changed.  The queue says what disagrees; the pixel count is its
        # fallback, and is all an entry queued by an older session carries.
        what = entry.get("summary") or f"{entry.get('sym_diff_px', 0)} px differ"
        return f"Step {step} — {entry.get('instance', '')} ({what})"
    if queue == api.QUEUE_MISSING_SHAPE:
        return f"Step {step} — {entry.get('instance', '')}"
    return f"Step {step}"


class ReviewPanel(QWidget):
    """Queue tabs of review mode; the actions are the window's (keys, palette)."""

    #: The annotator marked a step for rework.
    sigRework = Signal(int)
    #: A queue entry was activated: the window should open that step -- through
    #: the gate.  The panel used to call ``session.goto`` itself, un-forced, and
    #: an uncommitted layer then made it raise out of a Qt slot.
    sigOpenStep = Signal(int)

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        self._problems: list[str] = []
        #: The queue list whose row was activated and not yet answered for.
        self._activated: Optional[QListWidget] = None

        self._tabs = QTabWidget()
        self._lists: dict[str, QListWidget] = {}
        for queue in api.QUEUE_NAMES:
            lw = QListWidget()
            lw.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            lw.itemActivated.connect(self._on_item_activated)
            self._lists[queue] = lw
            self._tabs.addTab(lw, QUEUE_TITLES[queue])

        self._problems_label = QLabel("Problems")
        self._problems_list = QListWidget()
        self._problems_list.setMaximumHeight(90)
        self._problems_label.setVisible(False)
        self._problems_list.setVisible(False)

        # The keys live on the tool palette, in the tooltip and in the cheat
        # sheet, not in a 384 px label that decides how wide the dock has to be.
        self.setToolTip(
            "Enter: open the selected entry's frame and accept it\n"
            "R: rework it in Annotate mode\n"
            "K: keep the frozen shape    N: take the edit\n"
            "(左边工具面板上的「接受这一帧 / 返工 / 保留旧的 / 采用新的」是同样的键)"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._problems_label)
        layout.addWidget(self._problems_list)

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
        queues = self._session.queues() if session_is_open(self._session) else {}
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

    def problems(self) -> list[str]:
        """Problems of the last refused confirmation (empty after a good one)."""
        return list(self._problems)

    # -- actions ------------------------------------------------------------
    def selected_step(self) -> Optional[int]:
        """Step of the selected entry, falling back to the frame being shown."""
        item = self._lists[self.current_queue()].currentItem()
        if item is not None:
            return int(item.data(STEP_ROLE))
        if session_is_open(self._session):
            return int(self._session.current().step)
        return None

    def selected_conflict(self) -> Optional[int]:
        """Id of the selected conflict, or ``None``."""
        item = self._lists[api.QUEUE_CONFLICTS].currentItem()
        cid = None if item is None else item.data(CID_ROLE)
        return None if cid is None else int(cid)

    def select_conflict(self, cid: int) -> bool:
        """Put the Conflicts tab on top with conflict ``cid`` selected.

        What a click on a conflict line of the task card asks for (U2e), so
        that ``K`` / ``N`` act on that one.  ``False`` when the queue does not
        list it (it was settled meanwhile).
        """
        lw = self._lists[api.QUEUE_CONFLICTS]
        self._tabs.setCurrentIndex(list(api.QUEUE_NAMES).index(api.QUEUE_CONFLICTS))
        for row in range(lw.count()):
            if lw.item(row).data(CID_ROLE) is not None and int(lw.item(row).data(CID_ROLE)) == int(cid):
                lw.setCurrentRow(row)
                return True
        return False

    def select_current_step(self) -> None:
        """Put the highlight back on the frame that is open, after a refusal.

        Qt selects the row before the click is delivered, so a refused move left
        the queue pointing at a step the canvas is not showing.  Only the list
        the annotator actually activated is touched: restoring all four cleared
        a conflict they had picked in another tab, and the ``K`` / ``N`` keys
        then had nothing to act on.
        """
        lw, self._activated = self._activated, None
        if lw is None or not session_is_open(self._session):
            return
        step = int(self._session.current().step)
        rows = [r for r in range(lw.count())
                if int(lw.item(r).data(STEP_ROLE)) == step]
        lw.setCurrentRow(rows[0] if rows else -1)

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
        step = item.data(STEP_ROLE)
        if step is None or int(step) < 0:
            return
        # Remember which list asked, so that a refusal puts back that one and
        # leaves the other three tabs' selections alone.
        self._activated = item.listWidget()
        self.sigOpenStep.emit(int(step))

    def _on_frame_changed(self, _key: object) -> None:
        self._activated = None      # the move happened: there is nothing to put back
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
