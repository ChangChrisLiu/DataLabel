"""The window around the annotation: widgets, status bar, lifecycle, entry point.

Mixed into :class:`tda.ui.app.MainWindow`.  Nothing here knows what an
annotation is -- it builds the top bar, the docks and the status bar, writes the
window state to an INI file on ``D:``, and owns the way in and the way out:

* **in** -- :func:`take_lock` (the single-user lock of spec 3.5) and
  :func:`main`, which refuses with exit code 3 and touches nothing else when
  somebody is already annotating;
* **out** -- ``session.close()`` (which joins the session's own worker), then a
  backup that is *verified* rather than assumed, then the lock.  A failed backup
  warns and still lets the annotator leave: refusing to close would cost them
  the rest of the evening and gain nothing, since every commit is already on
  disk.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDockWidget,
    QLabel,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tda.core.model import VIEWS
from tda.ui import app_compat as compat
from tda.ui.canvas.overlay import LabelOverlay
from tda.ui.canvas.view import ImageCanvas
from tda.ui.panels.instances import InstanceListPanel
from tda.ui.panels.review import ReviewPanel
from tda.ui.panels.taskcard import TaskCardPanel
from tda.ui.panels.timeline import TimelinePanel

__all__ = ["MODE_TITLES", "ShellMixin", "main", "take_lock"]

#: Tab caption -> mode name, in the order the tabs appear.
MODE_TITLES: tuple[tuple[str, str], ...] = (
    ("Steps", "steps"), ("Annotate", "annotate"), ("Review", "review")
)


def take_lock(db, annotator: str) -> Optional[str]:
    """Take the single-user lock; returns a message when somebody else holds it."""
    try:
        db.acquire_lock(annotator)
    except RuntimeError as exc:
        return str(exc)
    return None


class ShellMixin:
    """Widgets, the status bar and the window's life from open to close."""

    # ------------------------------------------------------------------ build
    def _build_central(self) -> None:
        """Canvas, missing-frame placeholder and the room the bars live in."""
        self.canvas = ImageCanvas()
        self.overlay: Optional[LabelOverlay] = None
        self.placeholder_label = QLabel(
            "no image for this step in this view\n本视图在该步骤没有图像"
        )
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.placeholder_label)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.stack, 1)
        self._central_layout = layout
        self.setCentralWidget(central)

    def _build_top_bar(self) -> None:
        """Desktop combo with a done/total counter, four views, three modes."""
        bar = QToolBar("Top")
        bar.setMovable(False)
        bar.setObjectName("top_bar")
        self.addToolBar(bar)

        self.desktop_combo = QComboBox()
        for desktop in self.db.desktop_ids():
            self.desktop_combo.addItem(self._desktop_text(desktop), desktop)
        index = self.desktop_combo.findData(self.session.desktop)
        if index >= 0:
            self.desktop_combo.setCurrentIndex(index)
        self.desktop_combo.currentIndexChanged.connect(self._on_desktop_chosen)
        bar.addWidget(self.desktop_combo)
        bar.addSeparator()

        self.view_buttons: dict[str, QToolButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for i, view in enumerate(VIEWS):
            button = QToolButton()
            button.setText(f"{view} (F{i + 1})")
            button.setCheckable(True)
            button.setChecked(view == self.session.view)
            button.clicked.connect(lambda _c=False, v=view: self.act_set_view(v))
            group.addButton(button)
            self.view_buttons[view] = button
            bar.addWidget(button)
        bar.addSeparator()

        self.mode_tabs = QTabBar()
        for title, _mode in MODE_TITLES:
            self.mode_tabs.addTab(title)
        self.mode_tabs.setCurrentIndex(1)
        self.mode_tabs.currentChanged.connect(self._on_mode_tab)
        bar.addWidget(self.mode_tabs)

    def _build_docks(self) -> None:
        """Timeline on the left; task card + instance list and review on the right."""
        self.timeline = TimelinePanel(self.session)
        self.task_card = TaskCardPanel(self.session)
        self.instances = InstanceListPanel(self.session)
        self.review = ReviewPanel(self.session)

        self.timeline_dock = self._dock("Timeline", self.timeline,
                                        Qt.DockWidgetArea.LeftDockWidgetArea)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.task_card)
        right.addWidget(self.instances)
        self.right_dock = self._dock("Frame", right,
                                     Qt.DockWidgetArea.RightDockWidgetArea)
        self.review_dock = self._dock("Review", self.review,
                                      Qt.DockWidgetArea.RightDockWidgetArea)
        self.review_dock.hide()

        self.task_card.sigRequestEdit.connect(self.on_request_edit)
        self.instances.sigRequestEdit.connect(self.on_request_edit)
        self.review.sigRework.connect(self.on_rework)
        self._rewire_review_buttons()

    def _dock(self, title: str, widget: QWidget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower()}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_status_bar(self) -> None:
        self.zoom_label = QLabel("100%")
        self.frame_label = QLabel("")
        self.tool_label = QLabel("")
        self.sam_label = QLabel("")
        self.hint_label = QLabel("")
        bar = self.statusBar()
        for label in (self.zoom_label, self.frame_label, self.tool_label,
                      self.sam_label):
            bar.addPermanentWidget(label)
        bar.addWidget(self.hint_label, 1)

    def _connect_session(self) -> None:
        """Every session signal the window reacts to, when the session has it."""
        self.session.sigFrameChanged.connect(self._on_frame_changed)
        self.session.sigProblems.connect(self._on_problems)
        self.session.sigDirty.connect(self._on_dirty)
        signal = compat.editing_changed_signal(self.session)
        if signal is not None:
            signal.connect(self._on_editing_changed)
        for name, slot in (("sigQueuesChanged", self._on_queues_changed),
                           ("sigSweepProgress", self._on_sweep_progress)):
            found = getattr(self.session, name, None)
            if found is not None:
                found.connect(slot)

    # ----------------------------------------------------------- status bar
    def update_status(self) -> None:
        """Rewrite the four permanent status labels from the current state."""
        self.zoom_label.setText(f"{self.canvas.zoom_factor() * 100:.0f}%")
        self.frame_label.setText(self._frame_text())
        radius = getattr(self.active_tool, "radius", None)
        suffix = "" if radius is None else f" r{radius}"
        self.tool_label.setText(f"{self._tool_name}{suffix}")
        self.sam_label.setText(self.sam_status_text())

    def _frame_text(self) -> str:
        if not compat.is_open(self.session):
            return "no frame"
        key = self.session.current()
        steps = self.session.steps()
        total = max(steps) if steps else key.step
        status = self.session.frame_status(key.step)
        return f"D{key.desktop} · {key.view} · step {key.step}/{total} · {status}"

    def _desktop_text(self, desktop: int) -> str:
        """``D13 Dell Optiplex [12/38]`` -- the done/total counter of this view."""
        meta = self.db.get_desktop(desktop) or {}
        done = self.db.count_per_view("verified").get((desktop, self.session.view), 0)
        total = self.db.count_per_view("frames").get((desktop, self.session.view), 0)
        brand = str(meta.get("brand") or "")
        return f"D{desktop} {brand} [{done}/{total}]".replace("  ", " ")

    def status_message(self) -> str:
        """The last transient line shown in the status bar."""
        return self._message

    def report(self, text: str) -> None:
        """Show a transient line (a hint, a count, a refusal)."""
        self._message = str(text)
        self.hint_label.setText(self._message)

    def last_error_message(self) -> str:
        """The last error shown, which a later hint does not erase."""
        return self._last_error

    def report_error(self, text: str) -> None:
        """Show an error line and log it."""
        self.logger.error("%s", text)
        self._last_error = str(text)
        self.report(str(text))

    def report_exception(self, exc: BaseException, where: str = "") -> None:
        """What :func:`tda.ui.app_support.guard` calls; never raises itself.

        The rollback matters as much as the message: a slot that failed halfway
        through a write would otherwise leave the connection inside a
        transaction, and every later write would join it.
        """
        self.logger.exception("exception in %s: %s", where, exc)
        try:
            if self.db.conn.in_transaction:
                self.db.conn.rollback()
        except Exception:  # pragma: no cover - a rollback failure is terminal
            self.logger.exception("rollback after %s failed", where)
        self.report(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------- lifecycle
    def save_window_state(self) -> None:
        """Geometry, dock layout and the frame this annotator was last on."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        prefix = f"last/{self.annotator}"
        self.settings.setValue(f"{prefix}/desktop", int(self.session.desktop or 0))
        self.settings.setValue(f"{prefix}/view", str(self.session.view))
        if compat.is_open(self.session):
            self.settings.setValue(f"{prefix}/step", int(self.session.current().step))
        self.settings.sync()

    def restore_window_state(self) -> None:
        """Put the window back where the annotator left it (if it was ever there)."""
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self.settings.value("state")
        if state is not None:
            self.restoreState(state)

    def last_frame_for(self, annotator: str) -> dict:
        """The desktop/view/step this annotator last had open, as far as it is known."""
        prefix = f"last/{annotator}"
        out: dict = {}
        for name, cast in (("desktop", int), ("view", str), ("step", int)):
            value = self.settings.value(f"{prefix}/{name}")
            if value not in (None, ""):
                try:
                    out[name] = cast(value)
                except (TypeError, ValueError):
                    pass
        return out

    def shutdown(self) -> None:
        """Stop the threads and detach; the database is the caller's business."""
        if self.closed:
            return
        self.closed = True
        try:
            QApplication.instance().removeEventFilter(self)
        except RuntimeError:  # pragma: no cover - the app is already gone
            pass
        self.shutdown_assist()
        self._detach_tool()
        for signal, slot in ((self.session.sigFrameChanged, self._on_frame_changed),
                             (self.session.sigProblems, self._on_problems),
                             (self.session.sigDirty, self._on_dirty)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):  # pragma: no cover
                pass
        if self._cheat_sheet is not None:
            self._cheat_sheet.close()
        sys.excepthook = self._previous_hook

    def closeEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.save_window_state()
        try:
            self.session.save()
            compat.close_session(self.session)  # joins the session's sweeper
        except Exception as exc:  # noqa: BLE001 - closing must never hang
            self.report_error(f"closing the session failed: {exc}")
        self._backup_on_exit()
        try:
            self.db.release_lock()
        except Exception as exc:  # noqa: BLE001
            self.report_error(f"releasing the lock failed: {exc}")
        self.shutdown()
        event.accept()

    def _backup_on_exit(self) -> None:
        """Back the database up and *verify* it; a failure warns, never blocks."""
        dest = self.paths.get("backup_dir")
        if not dest:
            self.report_error("no backup_dir in paths.yaml: no exit backup was made")
            return
        try:
            out = Path(self.db.backup(str(dest)))
            if not out.exists() or out.stat().st_size <= 0:
                raise OSError(f"{out} is empty")
        except Exception as exc:  # noqa: BLE001 - the annotator still gets to leave
            self.report_error(f"exit backup failed: {exc}")
            return
        self.report(f"backup written to {out}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(paths: str = "configs/paths.yaml", desktop: int = 13, view: str = "scan",
         annotator: str = "", step: Optional[int] = None, db: Optional[str] = None,
         *, exec_: bool = True) -> int:
    """Open the annotator on one desktop/view; returns the process exit code.

    ``3`` means somebody else holds the single-user lock, and nothing at all was
    touched -- not the database, not the cache, not the settings.  SAM is loaded
    only after the window is on screen, so the first frame does not wait for it.
    """
    from tda import pipeline as P
    from tda.core.taxonomy import load_taxonomy
    from tda.core.truth import TruthService
    from tda.ui.app import MainWindow
    from tda.ui.session import AnnotationSession

    config = P.load_paths(paths)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    database = P.open_db(config, db)
    held = take_lock(database, annotator or "annotator")
    if held is not None:
        if exec_:
            QMessageBox.critical(None, "Database locked", f"{held}\n\n"
                                 "Close the other annotator and try again.")
        database.close()
        return 3

    tax = load_taxonomy()
    session = AnnotationSession(database, tax, TruthService(database, tax),
                                config.get("cache_dir", ""), annotator or "annotator")
    session.open(int(desktop), str(view))
    if step is not None:
        session.goto(int(step))
    window = MainWindow(session, config, annotator or "annotator")
    window.resize(1600, 1000)
    window.show()
    window.start_sam()
    if not exec_:
        window.close()
        database.close()
        return 0
    code = int(app.exec())
    database.close()
    return code
