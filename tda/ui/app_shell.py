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
from tda.ui.app_status import (  # re-exported: this was their home
    KNOWN_FAILURES,
    StatusMixin,
    explain_exception,
)
from tda.ui.canvas.overlay import LabelOverlay
from tda.ui.canvas.view import ImageCanvas
from tda.ui.panels.instances import InstanceListPanel
from tda.ui.panels.review import ReviewPanel
from tda.ui.panels.taskcard import TaskCardPanel
from tda.ui.panels.timeline import TimelinePanel

__all__ = ["KNOWN_FAILURES", "MODE_TITLES", "ShellMixin", "StatusMixin",
           "explain_exception", "main", "take_lock"]

#: Tab caption -> mode name, in the order the tabs appear.
MODE_TITLES: tuple[tuple[str, str], ...] = (
    ("Steps", "steps"), ("Annotate", "annotate"), ("Review", "review")
)
#: Used when this annotator has never opened the window on this machine.
DEFAULT_WINDOW_SIZE = (1600, 1000)
#: Fractions of the window width the docks are *asked* for by default.  Qt
#: clamps ``resizeDocks`` to each dock's minimum size hint, so these are only
#: honoured because the panels inside them are built to shrink -- short button
#: captions, four columns, elided rows.  The acceptance test in
#: ``tests/test_app.py`` measures what actually comes out, not what is asked.
TIMELINE_FRACTION = 0.11
RIGHT_FRACTION = 0.20
#: A restored layout that leaves the canvas less than this is not one anybody
#: chose: it is a dock that grew once and was saved.
MIN_CANVAS_FRACTION = 0.5
#: How long that warning stays put against an ordinary status line. The
#: ROI measurement answers within about half a second of the window opening.
LAYOUT_RESET_HOLD_MS = 4000


def _read_last_frame(settings, annotator: str) -> dict:
    """``{"desktop", "view", "step"}`` of one annotator's last frame, as stored.

    One reader for the INI, used both by the running window
    (:meth:`ShellMixin.last_frame_for`) and by :func:`resume_target` before a
    window exists -- two copies of this is how they would drift.
    """
    prefix = f"last/{annotator}"
    out: dict = {}
    for name, cast in (("desktop", int), ("view", str), ("step", int)):
        value = settings.value(f"{prefix}/{name}")
        if value in (None, ""):
            continue
        try:
            out[name] = cast(value)
        except (TypeError, ValueError):
            pass
    return out


def last_frame_in(paths: dict, annotator: str) -> dict:
    """:func:`_read_last_frame` for callers that have paths but no window."""
    return _read_last_frame(S.make_settings(paths), annotator)


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


#: Failures the annotator can do something about, in words they can act on.
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
        # Nothing in the toolbar may hold the keyboard.  A click on the chooser
        # or a view button left the focus there, and ``blocks_shortcuts`` -- which
        # is right about a combo the annotator may be typing in -- then switched
        # every shortcut off with no cue at all until they clicked the canvas.
        self.desktop_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setCheckable(True)
            button.setChecked(view == self.session.view)
            button.clicked.connect(lambda _c=False, v=view: self.act_set_view(v))
            group.addButton(button)
            self.view_buttons[view] = button
            bar.addWidget(button)
        bar.addSeparator()

        self.mode_tabs = QTabBar()
        self.mode_tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        # The compiled frame holds only what is *in* the picture, so what has
        # already been taken out has to come from the frame's state.
        self.instances.set_removed_source(lambda: compat.removed_rows(self.session))
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

    def showEvent(self, event) -> None:  # noqa: D102 - Qt override
        # The only place the guard may run.  Here in ``restore_window_state`` it
        # measured a canvas that has never been laid out -- the ``QWidget``
        # default of 640 px -- against a window ``restoreGeometry`` had just made
        # 1920 px wide, so it fired at every launch and threw away the layout the
        # annotator had chosen, blaming them for it in the status bar.  Once, so
        # that a deliberate later drag is left alone.
        super().showEvent(event)
        if not getattr(self, "_layout_checked", False):
            self._layout_checked = True
            if self.settings.value("state") is not None:
                self._reject_a_starved_canvas()

    def _reject_a_starved_canvas(self) -> None:
        """Throw a saved layout away when it leaves the canvas too little.

        A dock that grew once -- a panel with a wide label, a drag -- is written
        to the INI by ``saveState`` and then follows the annotator to every later
        session.  A layout that gives the frame less than half the window is not
        one anybody chose on purpose.
        """
        width = self.width() or DEFAULT_WINDOW_SIZE[0]
        # Measure the canvas rather than subtracting the docks: margins and
        # splitters are the difference between "just over half" and "under".
        canvas = self.canvas.width() or (width - self.timeline_dock.width()
                                         - self.right_dock.width())
        if canvas < MIN_CANVAS_FRACTION * width:
            self.apply_default_layout()
            # Held: the chassis ROI is measured on a worker started while the
            # window was being built, and its answer -- "drag the chassis box"
            # -- used to arrive a few hundred milliseconds later and take this
            # off the screen. Somebody whose layout was just thrown away has to
            # be told why.
            self.report("the saved dock layout left the canvas too small; "
                        "it was reset", hold_ms=LAYOUT_RESET_HOLD_MS)

    def apply_default_layout(self) -> None:
        """Timeline ~11 %, the right docks ~20 %, the canvas the rest."""
        width = self.width() or DEFAULT_WINDOW_SIZE[0]
        self.resizeDocks(
            [self.timeline_dock, self.right_dock, self.review_dock],
            [int(width * TIMELINE_FRACTION), int(width * RIGHT_FRACTION),
             int(width * RIGHT_FRACTION)],
            Qt.Orientation.Horizontal,
        )

    def last_frame_for(self, annotator: str) -> dict:
        """The desktop/view/step this annotator last had open, as far as it is known."""
        return _read_last_frame(self.settings, annotator)

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
        self.timeline.shutdown()   # the thumbnail reader has a thread of its own
        self._detach_tool()
        try:
            QApplication.instance().removeEventFilter(self)
        except RuntimeError:  # pragma: no cover - the app is already gone
            pass
        # Every signal _connect_session took, including the optional ones: a
        # sweeper that is still running would otherwise deliver into a window
        # that has let go of its session.
        for name, slot in (("sigFrameChanged", self._on_frame_changed),
                           ("sigProblems", self._on_problems),
                           ("sigDirty", self._on_dirty),
                           ("sigEditingChanged", self._on_editing_changed),
                           ("sigQueuesChanged", self._on_queues_changed),
                           ("sigSweepProgress", self._on_sweep_progress),
                           ("sigSweepError", self._on_sweep_error)):
            signal = getattr(self.session, name, None)
            if signal is None:
                continue
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):  # pragma: no cover - never connected
                pass
        if self._cheat_sheet is not None:
            self._cheat_sheet.close()
        self.logger.info("window closed (%s)", self.annotator)
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
            # Commit with the session's own suggestion directly: the scope bar
            # is a non-modal conversation and there is nobody left to have it
            # with, so "Save" must be able to settle a layering or split
            # suggestion too, not only a plain keyframe.
            self.commit_with_suggested_scope()
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
            keep = self._backup_keep()
            out = Path(self.db.backup(str(dest), keep))
            if not out.exists() or out.stat().st_size <= 0:
                raise OSError(f"{out} is empty")
        except Exception as exc:  # noqa: BLE001 - the annotator still gets to leave
            self.report_error(f"exit backup failed: {exc}")
            return
        self.report(f"backup written to {out}")

    def _backup_keep(self):
        """``backup_keep`` of paths.yaml, so exit backups are pruned like the CLI's.

        A value that will not parse must not stand between the annotator and the
        door: the copy is still made, nothing is pruned, and the status bar says why.
        """
        from tda import pipeline as P

        try:
            return P.backup_keep(self.paths)
        except ValueError as exc:
            self.report_error(f"{exc}; this exit backup was not pruned")
            return None


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
    last = last_frame_in(config, annotator)   # the one reader of the INI
    target = {
        "desktop": desktop if desktop is not None else last.get("desktop"),
        "view": view if view is not None else last.get("view"),
        "step": step if step is not None else (
            last.get("step") if desktop is None else None
        ),
    }
    if not target["view"]:
        target["view"] = VIEWS[0]
    if target["desktop"] is None:
        target["desktop"] = _lowest_desktop_with_frames(db_path)
    asked = {"desktop": desktop is not None, "view": view is not None,
             "step": step is not None}
    return _validated(target, db_path, asked)


def _validated(target: dict, db_path: str, asked: dict) -> dict:
    """Fall back to something that exists; a stale INI must not stop the launch.

    The INI is written from whatever was last on screen, so a view with no
    frames could be stored as "last view" -- and the next launch then died in
    ``MainWindow.__init__`` (the timeline asks the session for a frame that is
    not there) and the app would not start until somebody edited the file.
    Every part of the target that came from the INI is checked against the
    database here and replaced when it is not there.

    What the annotator asked for on the command line is **not** second-guessed:
    ``--desktop 42`` opens 42, and if it has no frames the window says so
    (spec 4.5) rather than opening something else without being asked.
    """
    rows = _frame_index(db_path)
    if not rows:
        return target
    desktop = target["desktop"]
    if not asked["desktop"] and (desktop is None or int(desktop) not in rows):
        desktop = min(rows)
        target["step"] = None      # a step of another machine means nothing
    target["desktop"] = int(desktop)
    views = rows.get(int(desktop), {})
    if not asked["view"] and views and target["view"] not in views:
        target["view"] = next((v for v in VIEWS if v in views), sorted(views)[0])
        target["step"] = None
    steps = views.get(target["view"], set())
    if not asked["step"] and target["step"] is not None and int(target["step"]) not in steps:
        target["step"] = None      # the session opens on its own starting frame
    return target


def _frame_index(db_path: str) -> dict[int, dict[str, set[int]]]:
    """``{desktop: {view: {steps}}}`` straight from ``frame``, read-only.

    One query, no ``Db`` (which replays the schema): this runs before the
    database is opened for writing, and before the lock is anybody's.
    """
    import sqlite3

    index: dict[int, dict[str, set[int]]] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return index
    try:
        for desktop, view, step in conn.execute(
            "SELECT desktop, view, step FROM frame"
        ):
            index.setdefault(int(desktop), {}).setdefault(str(view), set()).add(int(step))
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return index


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
