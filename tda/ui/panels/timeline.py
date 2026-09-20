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

import threading
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QImageReader, QPixmap
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

__all__ = ["BREAK_COLOR", "BREAK_ROLE", "TimelinePanel", "STATUS_COLORS",
           "read_thumb", "status_brush"]


def read_thumb(path: str, size: int) -> Optional[QImage]:
    """One image decoded **at row size**, or ``None`` when it cannot be read.

    ``session.thumb_path`` falls back to the frame itself when the offline
    thumbnail pass has not run, and on an OAK view that frame is 4032x3040.
    ``QImageReader`` is told the size that is wanted, so a JPEG is decoded
    scaled by its own codec: 63 ms becomes 16 ms, measured on a real oak1
    frame.  (A PNG has no scaled decode, so a scanner frame still costs a full
    one -- which is why this runs on a thread of its own.)

    No ``QPixmap``: that is a GUI-thread type, and the point of this function
    is that it can be called from anywhere.
    """
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    wanted = reader.size()
    if wanted.isValid() and (wanted.width() > size or wanted.height() > size):
        wanted.scale(size, size, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(wanted)
    image = reader.read()
    return None if image.isNull() else image


class _ThumbReader(QObject):
    """Reads the timeline's pictures on a thread of its own.

    One thread, a newest-first stack of requests and a queued signal back:
    scrolling fast asks for a screenful of rows and then scrolls past them, so
    the row the annotator is looking at *now* is the one worth reading next.
    A request for a row that has already been answered is simply dropped by
    the panel when it lands.
    """

    #: ``(cache_key, QImage | None)``, delivered on the GUI thread.
    sigThumb = Signal(object)

    def __init__(self, size: int, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._size = int(size)
        self._lock = threading.Condition()
        self._queue: list[tuple] = []
        self._stopped = False
        #: Started by the first request, not by the constructor: a panel that
        #: is built and never shown -- which is most of them in a test run --
        #: has nothing to read, and a thread per panel adds up.
        self._thread: Optional[threading.Thread] = None

    def ask(self, cache_key: tuple, path: str) -> None:
        """Queue one picture; the newest request is read first."""
        with self._lock:
            if self._stopped:
                return
            self._queue.append((cache_key, str(path)))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run,
                                                name="tda-thumbs", daemon=True)
                self._thread.start()
            self._lock.notify()

    def running(self) -> bool:
        """Is the reader's thread alive?  (Lifecycle tests.)"""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def forget(self) -> None:
        """Drop everything still queued (the view or the machine changed)."""
        with self._lock:
            self._queue.clear()

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._stopped:
                    self._lock.wait()
                if self._stopped:
                    return
                cache_key, path = self._queue.pop()
            try:
                image = read_thumb(path, self._size)
            except Exception:  # noqa: BLE001 - a worker must never crash Qt
                image = None
            self.sigThumb.emit((cache_key, image))

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._stopped = True
            self._queue.clear()
            self._lock.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

#: Item data roles.
STEP_ROLE = int(Qt.ItemDataRole.UserRole)
STATUS_ROLE = int(Qt.ItemDataRole.UserRole) + 1
#: ``True`` on the row a pose segment **starts** at (spec 2.5): the shapes do
#: not carry across it, so where the boundary sits is worth seeing while
#: scrolling rather than only in the split dialog.
BREAK_ROLE = int(Qt.ItemDataRole.UserRole) + 2

#: Colour of that boundary mark.
BREAK_COLOR = QColor(120, 170, 235)

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
    """Paints the status bar down the left edge of each row, and the break marks."""

    #: Height of the line drawn across a row that starts a pose segment.
    BREAK_HEIGHT = 2

    def __init__(self, width: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._width = width

    def paint(self, painter, option, index) -> None:  # noqa: D102 - Qt override
        super().paint(painter, option, index)
        rect = option.rect
        status = index.data(STATUS_ROLE)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        if status:
            painter.fillRect(
                rect.left() + 1, rect.top() + 2, self._width, rect.height() - 4,
                status_brush(str(status)),
            )
        if index.data(BREAK_ROLE):
            # across the whole row, at the edge the new segment begins at: the
            # rows are in annotation order, so "before this step" is the top
            painter.fillRect(rect.left(), rect.top(), rect.width(),
                             self.BREAK_HEIGHT,
                             QBrush(BREAK_COLOR, Qt.BrushStyle.SolidPattern))
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

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        # Keyed by (desktop, view, step): step numbers repeat across machines
        # and the four views show different images of the same step.
        self._thumbs: dict[tuple[int, str, int], QPixmap] = {}
        #: Rows the reader has been asked for and has not answered yet.
        self._asked: set[tuple[int, str, int]] = set()
        self._reader = _ThumbReader(self.THUMB_SIZE, self)
        self._reader.sigThumb.connect(self._on_thumb,
                                      Qt.ConnectionType.QueuedConnection)
        self._context: Optional[tuple[int, str]] = None
        self._placeholder_pm: Optional[QPixmap] = None
        self._descending = True
        self._syncing = False
        #: Steps this view starts a pose segment at (spec 2.5), drawn as a mark.
        self._breaks: set[int] = set()

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
        self._asked.clear()
        self._reader.forget()
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
                item.setData(BREAK_ROLE, step in self._breaks)
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

    def set_break_steps(self, steps) -> None:
        """The steps of the current view that start a pose segment.

        The window hands these in on every frame change, because a re-cut can
        add or remove one without the step list changing at all.  Repainting is
        enough -- the rows themselves do not move.
        """
        wanted = {int(s) for s in steps or ()}
        if wanted == self._breaks:
            return
        self._breaks = wanted
        for row in range(self._list.count()):
            item = self._list.item(row)
            item.setData(BREAK_ROLE, int(item.data(STEP_ROLE)) in wanted)
        self._list.viewport().update()

    def break_steps(self) -> list[int]:
        """The steps currently marked as a segment start, ascending."""
        return sorted(self._breaks)

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
        pm = self._load_thumb(path) if path else None
        if pm is None:
            pm = self._placeholder()
        self._thumbs[cache_key] = pm
        item = self._item_for(step)
        if item is not None:
            item.setIcon(QIcon(pm))
        return pm

    def _load_thumb(self, path: str) -> Optional[QPixmap]:
        """Read one image at row size, here and now (see :func:`read_thumb`)."""
        image = read_thumb(path, self.THUMB_SIZE)
        return None if image is None else self._as_pixmap(image)

    def _as_pixmap(self, image: QImage) -> QPixmap:
        """A row-sized pixmap from a decoded image (GUI thread only)."""
        loaded = QPixmap.fromImage(image)
        if loaded.width() <= self.THUMB_SIZE and loaded.height() <= self.THUMB_SIZE:
            return loaded
        return loaded.scaled(
            self.THUMB_SIZE,
            self.THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def ensure_visible_thumbs(self) -> None:
        """Ask for the pictures of the rows on screen (plus a margin), off-thread.

        The rows are asked for, not read: jumping to an unvisited part of the
        list scrolls a screenful of *new* rows into view, and without the
        offline thumbnail pass -- which ``oak1`` and ``oak2`` have never had --
        each of those is a 12 MP decode.  Reading two per event-loop turn was
        still 118 ms of a 12 MP timeline click, spent on pictures the size of a
        postage stamp while the annotator waited for the frame.  The reader has
        its own thread and the icons appear as they land; the GUI thread only
        asks where each file is.
        """
        rows = self._visible_rows()
        if not rows or not session_is_open(self._session):
            return
        first = max(0, min(rows) - self.PREFETCH)
        last = min(self._list.count() - 1, max(rows) + self.PREFETCH)
        for row in range(first, last + 1):
            item = self._list.item(row)
            if item is None:
                continue
            step = int(item.data(STEP_ROLE))
            cache_key = self._cache_key(step)
            if cache_key in self._thumbs or cache_key in self._asked:
                continue
            path = self._session.thumb_path(step)
            if not path:
                # nothing to read: the placeholder is the answer, and caching
                # it is what stops the path being looked up on every scroll
                self._thumbs[cache_key] = self._placeholder()
                item.setIcon(QIcon(self._thumbs[cache_key]))
                continue
            self._asked.add(cache_key)
            self._reader.ask(cache_key, path)

    def _on_thumb(self, payload: object) -> None:
        """One picture came back from the reader (GUI thread)."""
        cache_key, image = payload  # type: ignore[misc]
        self._asked.discard(cache_key)
        if cache_key in self._thumbs:
            return
        pm = self._placeholder() if image is None else self._as_pixmap(image)
        self._thumbs[cache_key] = pm
        if self._context is not None and cache_key[:2] == self._context:
            item = self._item_for(cache_key[2])
            if item is not None:
                item.setIcon(QIcon(pm))

    def shutdown(self) -> None:
        """Stop the thumbnail reader; the panel is done with its thread.

        Called by the window on its way out: a daemon thread that outlives the
        window is exactly the leak ``scripts/mvp_smoke.py`` checks for.
        """
        self._reader.shutdown()

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
