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
    QAbstractScrollArea,
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

from tda.core.db import acquire_lock_file, release_lock_file
from tda.core.model import VIEWS
from tda.ui import app_compat as compat
from tda.ui import app_support as S
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
#: Used when this annotator has never opened the window on this machine.
DEFAULT_WINDOW_SIZE = (1600, 1000)
#: Fractions of the window width the docks get by default; the canvas keeps the
#: remaining ~67 %, which at 1920 px is 1280 px for a 12 MP frame.
TIMELINE_FRACTION = 0.11
RIGHT_FRACTION = 0.22


def take_lock(target, annotator: str) -> Optional[str]:
    """Take the single-user lock; returns a message when somebody else holds it.

    ``target`` is a database **path** in the application (the lock has to be
    taken before the file is opened, because opening it replays the schema) and
    may still be a ``Db`` for callers that already have one.
    """
    try:
        if hasattr(target, "acquire_lock"):
            target.acquire_lock(annotator)
        else:
            acquire_lock_file(str(target), annotator)
    except RuntimeError as exc:
        return str(exc)
    return None


def confirm_discard_dialog(parent, why: str) -> bool:
    """Ask before throwing unsaved S1 edits away."""
    answer = QMessageBox.question(
        parent, "Unsaved changes", f"{why}\nLeave the step table anyway?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


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
        counts = self._view_counts()  # two aggregate queries, not two per desktop
        for desktop in self.db.desktop_ids():
            self.desktop_combo.addItem(self._desktop_text(desktop, counts), desktop)
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
        # The dock decides its own width; the table scrolls inside it.
        self.instances.table().setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )

        self.timeline_dock = self._dock("Timeline", self.timeline,
                                        Qt.DockWidgetArea.LeftDockWidgetArea)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.task_card)
        right.addWidget(self.instances)
        # The card is a short checklist, the instance table is long: without
        # this they open at half the dock each and the table shows four rows.
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 2)
        self.right_dock = self._dock("Frame", right,
                                     Qt.DockWidgetArea.RightDockWidgetArea)
        self.review_dock = self._dock("Review", self.review,
                                      Qt.DockWidgetArea.RightDockWidgetArea)
        self.review_dock.hide()

        self.task_card.sigRequestEdit.connect(self.on_request_edit)
        self.instances.sigRequestEdit.connect(self.on_request_edit)
        self.review.sigRework.connect(self.on_rework)
        self._rewire_panels()

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
                           ("sigSweepProgress", self._on_sweep_progress),
                           ("sigSweepError", self._on_sweep_error)):
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

    def _view_counts(self) -> tuple[dict, dict]:
        """``(verified, frames)`` per ``(desktop, view)``; two whole-table scans.

        Read once and passed around: at 66 desktops, asking per row turned the
        combo box into 132 aggregate queries and 0.7 s of the start-up.
        """
        return self.db.count_per_view("verified"), self.db.count_per_view("frames")

    def _desktop_text(self, desktop: int, counts: Optional[tuple] = None) -> str:
        """``D13 Dell Optiplex [12/38]`` -- the done/total counter of this view."""
        meta = self.db.get_desktop(desktop) or {}
        verified, frames = counts if counts is not None else self._view_counts()
        done = verified.get((desktop, self.session.view), 0)
        total = frames.get((desktop, self.session.view), 0)
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
        """Put the window back where the annotator left it, or lay it out sensibly.

        The default matters: with the docks at their natural sizes the canvas got
        less than half the window, and the instance table (seven auto-sized
        columns, one of them a full instance key) pushed its dock wider still.
        """
        geometry = self.settings.value("geometry")
        state = self.settings.value("state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(*DEFAULT_WINDOW_SIZE)
        if state is not None:
            self.restoreState(state)
        else:
            self.apply_default_layout()

    def apply_default_layout(self) -> None:
        """Timeline ~11 %, the right docks ~22 %, the canvas the rest."""
        width = max(self.width(), DEFAULT_WINDOW_SIZE[0])
        self.resizeDocks(
            [self.timeline_dock, self.right_dock, self.review_dock],
            [int(width * TIMELINE_FRACTION), int(width * RIGHT_FRACTION),
             int(width * RIGHT_FRACTION)],
            Qt.Orientation.Horizontal,
        )

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
        """Stop every thread and detach; the database is the caller's business.

        Order matters: the assist threads (the SAM queue, the SAM loader and the
        difference worker) are stopped **first**, because each of them can
        deliver into the window, and a result landing after the session has been
        closed is an exception out of a Qt slot with nothing left to catch it.
        """
        if self.closed:
            return
        self.closed = True
        self.shutdown_assist()
        self._detach_tool()
        try:
            QApplication.instance().removeEventFilter(self)
        except RuntimeError:  # pragma: no cover - the app is already gone
            pass
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
        S.close_logger(self.logger)

    def closeEvent(self, event) -> None:  # noqa: D102 - Qt override
        if not self._settle_uncommitted_edit():
            event.ignore()
            return
        self.flush_sidecar()
        self.save_window_state()
        self.shutdown()                       # threads first, then the database
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
        event.accept()

    def _settle_uncommitted_edit(self) -> bool:
        """Ask what to do with an edit in progress; ``False`` cancels the close.

        The one place a modal question is right: the annotator is leaving, so
        there is no gesture to interrupt, and the alternative is silently
        throwing away work or silently writing something they never approved.
        """
        if not self.has_uncommitted_edit():
            return True
        answer = QMessageBox.question(
            self, "Uncommitted edit",
            f"{getattr(self.session, 'editing_instance', 'an instance')} has "
            f"changes that were never committed.\n未提交的修改：提交、放弃，还是留下？",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            self.report("close cancelled: the edit is still open")
            return False
        if answer == QMessageBox.StandardButton.Save:
            self.act_commit()
            if self.has_uncommitted_edit():   # the session refused it
                self.report_error("the edit could not be committed; close cancelled")
                return False
            return True
        self.act_clear_edit()
        return True

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
def resume_target(config: dict, annotator: str, desktop: Optional[int],
                  view: Optional[str], step: Optional[int], db_path: str) -> dict:
    """Which frame to open: what was asked for, else where this annotator left off.

    ``--desktop``/``--view``/``--step`` are optional so that the usual launch is
    ``python -m tda.cli app --annotator chang`` and it carries on where the last
    session stopped.  Nothing here opens the database: the last frame comes from
    the INI file, and the fallback (lowest desktop with frames, scanner view) is
    read from the same file the caller already has.
    """
    settings = S.make_settings(config)
    prefix = f"last/{annotator}"

    def stored(name: str, cast):
        value = settings.value(f"{prefix}/{name}")
        if value in (None, ""):
            return None
        try:
            return cast(value)
        except (TypeError, ValueError):
            return None

    target = {
        "desktop": desktop if desktop is not None else stored("desktop", int),
        "view": view if view is not None else stored("view", str),
        "step": step if step is not None else (
            stored("step", int) if desktop is None else None
        ),
    }
    if not target["view"]:
        target["view"] = VIEWS[0]
    if target["desktop"] is None:
        target["desktop"] = _lowest_desktop_with_frames(db_path)
    return target


def _lowest_desktop_with_frames(db_path: str) -> int:
    """The first machine that has any frames at all; 1 when nothing is loaded."""
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return 1
    try:
        row = conn.execute("SELECT MIN(desktop) FROM frame").fetchone()
    except sqlite3.Error:
        return 1
    finally:
        conn.close()
    return int(row[0]) if row and row[0] is not None else 1


def main(paths: str = "configs/paths.yaml", desktop: Optional[int] = None,
         view: Optional[str] = None, annotator: str = "",
         step: Optional[int] = None, db: Optional[str] = None,
         *, exec_: bool = True) -> int:
    """Open the annotator on one desktop/view; returns the process exit code.

    ``3`` means somebody else holds the single-user lock, and nothing at all was
    touched -- not the database, not the cache, not the settings.  The lock is
    taken **before** the database is opened, because ``Db.__init__`` replays the
    schema and the migrations: a refused launch has to leave the file byte for
    byte as it was.  SAM is loaded only after the window is on screen, so the
    first frame does not wait for it.
    """
    from tda import pipeline as P
    from tda.core.taxonomy import load_taxonomy
    from tda.core.truth import TruthService
    from tda.ui import app as app_module
    from tda.ui.session import AnnotationSession

    config = P.load_paths(paths)
    who = annotator or "annotator"
    db_path = db or P.require(config, "db_path")
    app = QApplication.instance() or QApplication(sys.argv[:1])

    held = take_lock(db_path, who)
    if held is not None:
        if exec_:
            QMessageBox.critical(None, "Database locked", f"{held}\n\n"
                                 "Close the other annotator and try again.")
        return 3

    database = None
    try:
        target = resume_target(config, who, desktop, view, step, db_path)
        database = P.open_db(config, db_path)
        tax = load_taxonomy()
        session = AnnotationSession(database, tax, TruthService(database, tax),
                                    config.get("cache_dir", ""), who)
        session.open(int(target["desktop"]), str(target["view"]))
        if target["step"] is not None:
            session.goto(int(target["step"]))
        window = app_module.MainWindow(session, config, who)
        window.show()
        window.start_sam()
        if not exec_:
            window.close()
            return 0
        return int(app.exec())
    except BaseException:
        release_lock_file(db_path)   # nothing was opened for this annotator to keep
        raise
    finally:
        if database is not None:
            database.close()
