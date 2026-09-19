"""Timeline dock: one thumbnail per logical step, newest first (spec 4.5).

The list runs *downwards in annotation order*: because a teardown is annotated
in reverse (spec 4.2), the last step of the machine is the first row.  A toggle
flips the list into chronological order for reading.

Each row carries a coloured bar telling the frame's status at a glance -- grey
unlabeled, yellow auto, green verified, red conflict or needs-review, and a
hatched bar for a step this view has no image for.  Thumbnails are read from
the cache only when their row is actually on screen and are kept as ``QPixmap``
afterwards, so opening a 120-step machine costs no disk I/O.

The panel never decides anything: a click only reports
(:attr:`TimelinePanel.sigOpenStep`) and the highlight follows ``sigFrameChanged``
no matter who moved the frame -- including when the window refuses the move.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from tda.ui import session_api as api
from tda.ui.panels import session_is_open

__all__ = ["TimelinePanel", "STATUS_COLORS", "status_brush"]

#: Item data roles.
STEP_ROLE = int(Qt.ItemDataRole.UserRole)
STATUS_ROLE = int(Qt.ItemDataRole.UserRole) + 1

#: Status colours (spec 4.5).  ``needs_review`` shares the conflict red: both
#: mean "a human must look at this frame again".
STATUS_COLORS: dict[str, QColor] = {
    api.STATUS_UNLABELED: QColor(136, 136, 136),
    api.STATUS_AUTO: QColor(230, 190, 60),
    api.STATUS_VERIFIED: QColor(64, 172, 90),
    api.STATUS_NEEDS_REVIEW: QColor(214, 72, 64),
    api.STATUS_CONFLICT: QColor(214, 72, 64),
    # a frozen frame waiting to be re-checked: amber, between "confirmed" and
    # "somebody has to look at this" -- it is not yet known which it is
    api.STATUS_RECHECK: QColor(224, 140, 48),
    api.STATUS_MISSING: QColor(150, 150, 156),
}


def status_brush(status: str) -> QBrush:
    """Brush for a frame status; ``missing`` is hatched, everything else solid."""
    color = STATUS_COLORS.get(status, STATUS_COLORS[api.STATUS_UNLABELED])
    if status == api.STATUS_MISSING:
        return QBrush(color, Qt.BrushStyle.BDiagPattern)
    return QBrush(color, Qt.BrushStyle.SolidPattern)


class _StatusBarDelegate(QStyledItemDelegate):
    """Paints the status bar down the left edge of each row."""

    def __init__(self, width: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._width = width

    def paint(self, painter, option, index) -> None:  # noqa: D102 - Qt override
        super().paint(painter, option, index)
        status = index.data(STATUS_ROLE)
        if not status:
            return
        rect = option.rect
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.fillRect(
            rect.left() + 1, rect.top() + 2, self._width, rect.height() - 4,
            status_brush(str(status)),
        )
        painter.restore()


class TimelinePanel(QWidget):
    """Vertical step timeline with thumbnails, status bars and an order toggle."""

    #: A row was clicked: the window should open that step -- through the gate.
    #: The panel used to call ``session.goto`` itself, un-forced, so an
    #: uncommitted layer made it raise ``SessionRefusal`` out of a Qt slot the
    #: moment anybody forgot to intercept the connection.
    sigOpenStep = Signal(int)

    #: Long edge of a thumbnail, in pixels (task 12 brief).
    THUMB_SIZE = 96
    #: Width of the status bar drawn at the left of a row.
    BAR_WIDTH = 6
    #: Extra rows loaded above and below the visible range.
    PREFETCH = 2
    #: Thumbnails decoded per event-loop turn (see ``ensure_visible_thumbs``).
    THUMBS_PER_TICK = 2

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        # Keyed by (desktop, view, step): step numbers repeat across machines
        # and the four views show different images of the same step.
        self._thumbs: dict[tuple[int, str, int], QPixmap] = {}
        self._context: Optional[tuple[int, str]] = None
        self._placeholder_pm: Optional[QPixmap] = None
        self._descending = True
        self._syncing = False

        self.order_toggle = QCheckBox("Oldest ↑")
        self.order_toggle.setToolTip(
            "Oldest first.  Off: newest step first, the order frames are annotated in."
        )
        self.order_toggle.setMinimumWidth(1)
        self.order_toggle.toggled.connect(self._on_order_toggled)

        self._list = QListWidget()
        self._list.setIconSize(QSize(self.THUMB_SIZE, self.THUMB_SIZE))
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setUniformItemSizes(True)
        self._list.setItemDelegate(_StatusBarDelegate(self.BAR_WIDTH, self._list))
        # The dock decides how wide it is; a 96 px thumbnail plus its label must
        # not be the thing that sets it.  Narrower than a thumbnail the list
        # simply scrolls sideways.
        self._list.setMinimumWidth(1)
        self._list.setHorizontalScrollMode(
            QListWidget.ScrollMode.ScrollPerPixel
        )
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.verticalScrollBar().valueChanged.connect(
            lambda _v: self.ensure_visible_thumbs()
        )

        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        header.addWidget(QLabel("Steps"))
        header.addStretch(1)
        header.addWidget(self.order_toggle)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self._list, 1)

        if session is not None:
            self.set_session(session)

    # -- wiring -------------------------------------------------------------
    def set_session(self, session: Optional[api.SessionLike]) -> None:
        """Attach a session (or ``None``) and rebuild the list."""
        if self._session is not None:
            self._session.sigFrameChanged.disconnect(self._on_frame_changed)
        self._session = session
        self._thumbs.clear()
        self._context = None
        if session is not None:
            session.sigFrameChanged.connect(self._on_frame_changed)
        self.refresh()

    def list_widget(self) -> QListWidget:
        """The underlying list, for the main window's layout and for tests."""
        return self._list

    # -- content ------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild every row from ``session.steps()``."""
        self._context = self._current_context()
        self._syncing = True
        try:
            self._list.clear()
            for step in self._ordered_steps():
                item = QListWidgetItem(f"Step {step}")
                item.setData(STEP_ROLE, step)
                item.setData(STATUS_ROLE, self._status(step))
                item.setSizeHint(
                    QSize(self.THUMB_SIZE * 2, self.THUMB_SIZE + 2 * self.BAR_WIDTH)
                )
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )
                pm = self._thumbs.get(self._cache_key(step))
                if pm is not None:
                    item.setIcon(QIcon(pm))
                self._list.addItem(item)
            self._select_current()
        finally:
            self._syncing = False
        if self.isVisible():
            self.ensure_visible_thumbs()

    def item_steps(self) -> list[int]:
        """The steps in display order."""
        return [
            int(self._list.item(row).data(STEP_ROLE))
            for row in range(self._list.count())
        ]

    def current_step(self) -> Optional[int]:
        """Step of the highlighted row, or ``None`` when nothing is selected."""
        item = self._list.currentItem()
        return None if item is None else int(item.data(STEP_ROLE))

    def refresh_statuses(self) -> None:
        """Re-read every row's status colour without rebuilding the list.

        Cheap -- it asks the session for one status per row and repaints -- and
        it is the only way a frame *other than the open one* ever changes
        colour: a commit that reaches a verified frame queues a re-check, the
        sweeper then turns it into a conflict, and until this ran the row still
        said what it said when the annotator last stood on it.
        """
        self._refresh_statuses()

    def select_current_step(self) -> None:
        """Put the highlight back on the frame that is actually open.

        Qt selects the row under the mouse *before* the click reaches anybody,
        so when the window refuses the move the list is left pointing at a frame
        the canvas is not showing.  The window calls this on a refusal.
        """
        self._select_current()

    def step_brush(self, step: int) -> QBrush:
        """The brush the status bar of ``step`` is painted with."""
        item = self._item_for(step)
        status = api.STATUS_UNLABELED if item is None else str(item.data(STATUS_ROLE))
        return status_brush(status)

    # -- order --------------------------------------------------------------
    def is_descending(self) -> bool:
        """``True`` when the newest step is on top (the annotation order)."""
        return self._descending

    def set_descending(self, descending: bool) -> None:
        """Set the row order; the highlighted step is kept."""
        if bool(descending) == self._descending:
            return
        self._descending = bool(descending)
        blocked = self.order_toggle.blockSignals(True)
        self.order_toggle.setChecked(not self._descending)
        self.order_toggle.blockSignals(blocked)
        self.refresh()

    # -- thumbnails ---------------------------------------------------------
    def thumbnail(self, step: int) -> Optional[QPixmap]:
        """The step's thumbnail, read from the cache directory at most once.

        A step with no cached image (or an unreadable one) gets the shared
        placeholder pixmap, which is stored under the step as well so the path
        is never looked up twice.  The cache is keyed by desktop and view, so
        switching either one shows that view's images rather than the ones the
        previous view had under the same step numbers.
        """
        cache_key = self._cache_key(step)
        if cache_key in self._thumbs:
            return self._thumbs[cache_key]
        if not session_is_open(self._session):
            return None
        path = self._session.thumb_path(step)
        pm: Optional[QPixmap] = None
        if path:
            loaded = QPixmap(str(path))
            if not loaded.isNull():
                pm = loaded.scaled(
                    self.THUMB_SIZE,
                    self.THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        if pm is None:
            pm = self._placeholder()
        self._thumbs[cache_key] = pm
        item = self._item_for(step)
        if item is not None:
            item.setIcon(QIcon(pm))
        return pm

    def ensure_visible_thumbs(self) -> None:
        """Load the thumbnails of the rows currently on screen (plus a margin).

        At most :data:`THUMBS_PER_TICK` of them per call; the rest are picked up
        from a zero-timer, so the GUI thread goes back to the annotator between
        batches.  Jumping to an unvisited part of the list scrolled a whole
        screenful of *new* rows into view, and decoding and scaling all of them
        synchronously is a large part of why a timeline click cost three times
        what ``PgDn`` costs (313 ms against 114 ms measured).
        """
        rows = self._visible_rows()
        if not rows:
            return
        first = max(0, min(rows) - self.PREFETCH)
        last = min(self._list.count() - 1, max(rows) + self.PREFETCH)
        loaded = 0
        for row in range(first, last + 1):
            item = self._list.item(row)
            if item is None:
                continue
            if self._cache_key(int(item.data(STEP_ROLE))) in self._thumbs:
                continue
            if loaded >= self.THUMBS_PER_TICK:
                QTimer.singleShot(0, self.ensure_visible_thumbs)
                return
            self.thumbnail(int(item.data(STEP_ROLE)))
            loaded += 1

    # -- Qt overrides -------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().showEvent(event)
        self.ensure_visible_thumbs()

    def resizeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().resizeEvent(event)
        if self.isVisible():
            self.ensure_visible_thumbs()

    # -- internals ----------------------------------------------------------
    def _ordered_steps(self) -> list[int]:
        if not session_is_open(self._session):
            return []
        steps = sorted(self._session.steps())
        return list(reversed(steps)) if self._descending else steps

    def _current_context(self) -> Optional[tuple[int, str]]:
        """``(desktop, view)`` of the open frame, which the cache is keyed by."""
        if not session_is_open(self._session):
            return None
        key = self._session.current()
        return (key.desktop, key.view)

    def _cache_key(self, step: int) -> tuple[int, str, int]:
        desktop, view = self._context or (-1, "")
        return (desktop, view, step)

    def _status(self, step: int) -> str:
        if not session_is_open(self._session):
            return api.STATUS_UNLABELED
        return self._session.frame_status(step)

    def _item_for(self, step: int) -> Optional[QListWidgetItem]:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if int(item.data(STEP_ROLE)) == step:
                return item
        return None

    def _select_current(self) -> None:
        if not session_is_open(self._session):
            return
        item = self._item_for(self._session.current().step)
        if item is None:
            return
        self._list.setCurrentItem(item)
        self._list.scrollToItem(item)

    def _refresh_statuses(self) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            item.setData(STATUS_ROLE, self._status(int(item.data(STEP_ROLE))))
        self._list.viewport().update()

    def _visible_rows(self) -> list[int]:
        viewport = self._list.viewport().rect()
        if viewport.isEmpty() or self._list.count() == 0:
            return []
        rows: list[int] = []
        y = viewport.top()
        while y <= viewport.bottom():
            index = self._list.indexAt(QPoint(viewport.left() + self.BAR_WIDTH, y))
            if not index.isValid():
                break
            rows.append(index.row())
            rect = self._list.visualRect(index)
            step = max(1, rect.height())
            y = rect.bottom() + 1 if rect.bottom() >= y else y + step
        return rows

    # -- slots --------------------------------------------------------------
    def _on_order_toggled(self, checked: bool) -> None:
        self._descending = not checked
        self.refresh()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if self._session is None or self._syncing:
            return
        self.sigOpenStep.emit(int(item.data(STEP_ROLE)))

    def _on_frame_changed(self, _key: object) -> None:
        if self._current_context() != self._context:
            # Another machine or another view: the steps, their statuses and
            # their thumbnails all belong to a different list.
            self.refresh()
            return
        self._syncing = True
        try:
            self._refresh_statuses()
            self._select_current()
        finally:
            self._syncing = False
        if self.isVisible():
            self.ensure_visible_thumbs()

    def _placeholder(self) -> QPixmap:
        if self._placeholder_pm is None:
            pm = QPixmap(self.THUMB_SIZE, self.THUMB_SIZE * 3 // 4)
            pm.fill(QColor(58, 58, 62))
            self._placeholder_pm = pm
        return self._placeholder_pm
