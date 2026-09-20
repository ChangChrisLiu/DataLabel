"""The annotator's main window: it wires, it does not decide (spec 4.5, 10.1).

Everything on screen already knows how to work on its own -- the canvas, the
tools, the dock panels, the session.  This module puts them in one window, gives
them a keyboard (:mod:`tda.ui.app_actions`), a status bar and a lifecycle, and
gets out of the way.  Any logic that looks like annotation belongs in the
session; anything that looks like a panel belongs in the panel.

What the window owns is exactly what no single part can:

* **the frame.**  A frame change repaints the canvas and the overlay, re-stamps
  the SAM tools with the new frame identity and keeps the zoom across frames of
  one pose segment (the annotator is comparing the same spot) while resetting it
  to the ROI when the segment changes.
* **the lifecycle.**  The single-user lock on the way in; ``session.close()``, a
  verified backup and the lock release on the way out; window geometry in an INI
  file under ``D:/DataSet/.cache`` -- never the registry.
* **robustness.**  Every slot is wrapped by :func:`tda.ui.app_support.guard`, so
  a failure becomes a log entry and one status line instead of an exception
  escaping into the Qt event loop with the database mid-transaction.

The editing layer lives in :mod:`tda.ui.app_edit` and the moment it is written
in :mod:`tda.ui.app_commit`; the ROI, the bench box, the review verdicts and
crash safety are in :mod:`tda.ui.app_roi`, the model assist in
:mod:`tda.ui.app_assist`, the Label Studio draft ghost in
:mod:`tda.ui.app_adopt` and the widgets, the status bar and the lifecycle in
:mod:`tda.ui.app_shell`; all of them are mixed in below.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.app_adopt import AdoptMixin
from tda.ui.app_assist import AssistMixin
from tda.ui.app_commit import CommitMixin
from tda.ui.app_edit import EditMixin
from tda.ui.app_keys import FLASH_UNNAMED, KeysMixin
from tda.ui.app_pose import PoseMixin
from tda.ui.app_roi import RoiMixin
from tda.ui.app_shell import (
    MODE_TITLES,
    ShellMixin,
    StatusMixin,
    confirm_discard_dialog,
    main,
    take_lock,
)
from tda.ui.app_view import (
    CANDIDATES_DROPPED,
    GRID_OFF,
    OPACITY_STEP,
    ToolsMixin,
)
from tda.ui.canvas.overlay import LabelOverlay

__all__ = ["MainWindow", "main", "take_lock"]

# Re-exported so that ``from tda.ui.app import ...`` keeps working wherever it
# already did; the definitions live with the code that uses them.
__all__ += ["CANDIDATES_DROPPED", "FLASH_UNNAMED", "GRID_OFF", "OPACITY_STEP"]


class MainWindow(EditMixin, CommitMixin, RoiMixin, AdoptMixin, PoseMixin, AssistMixin,
                 KeysMixin,
                 ToolsMixin, StatusMixin, ShellMixin, QMainWindow):
    """One annotator, one desktop/view, three modes."""

    def __init__(self, session, paths: dict, annotator: str, *,
                 sam_queue: Any = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.session = session
        self.paths = dict(paths or {})
        self.annotator = str(annotator)
        self.db = session.db
        self.closed = False
        self.mode = A.MODE_ANNOTATE
        self._message = ""
        self._last_error = ""
        self._segment: Optional[int] = None
        self._steps_dirty = False
        self._steps_panel: Optional[QWidget] = None
        self._cheat_sheet: Optional[QWidget] = None
        self.tools_enabled = True
        self._tool_name = "brush"
        #: The neighbour step the canvas is showing while ``Tab`` is held, or
        #: ``None``.  One place: every guard reads this and nothing else.
        self._flashing: Optional[int] = None
        self.review_refreshes = 0
        #: Steps whose background re-check gave up; ``F5`` retries them.
        self.sweep_failures: set[int] = set()

        self.logger = S.get_logger(self.paths)
        self.settings = S.make_settings(self.paths)
        self.sidecar = S.EditSidecar(self.paths, self.annotator)
        self._previous_hook = S.install_excepthook(self.logger, self.report_error)

        self._build_central()
        self._build_top_bar()
        self._build_docks()
        self._build_status_bar()
        self._init_edit()
        self._init_adopt()
        self._init_pose()
        self._init_assist(sam_queue)
        self._connect_session()

        self.setWindowTitle(f"Teardown Annotator — {self.annotator}")
        self.logger.info("window open: annotator=%s db=%s", self.annotator,
                         self.paths.get("db_path", ""))
        self.restore_window_state()
        QApplication.instance().installEventFilter(self)
        for name, why in compat.ADAPTED.items():
            self.logger.info("session gap adapted: %s (%s)", name, why)
        self.render_frame()

    # ------------------------------------------------------- lazy step table
    @property
    def steps_panel(self) -> QWidget:
        """The S1 step-table panel, built the first time Steps mode is opened."""
        if self._steps_panel is None:
            from tda.ui.panels.steptable import StepTablePanel

            panel = StepTablePanel(self.db, int(self.session.desktop),
                                   self.session.tax, self.paths.get("cache_dir", ""))
            panel.sigSaved.connect(self.on_steps_saved)
            for model in (panel.steps_model, panel.instances_model):
                model.dataChanged.connect(self._mark_steps_dirty)
            # A staged constraint edge is an unsaved step-table edit like any
            # other: the one gate has to know about it, or leaving Steps mode
            # would drop it without asking.
            panel.relations_tab.sigChanged.connect(self._mark_steps_dirty)
            self._steps_panel = panel
            self.stack.addWidget(panel)
        return self._steps_panel

    # ------------------------------------------------------------ frame paint
    @S.guard
    def _on_frame_changed(self, _key: object) -> None:
        self.render_frame()

    def render_frame(self) -> None:
        """Repaint everything that belongs to the frame the session is on."""
        # A hint is about the frame it was said on.  "step 14 is not complete:
        # 61 problems" stayed on screen two frames later, where it was simply
        # untrue.
        self.report("")
        # Whatever was being compared, this is not it; the tools are re-armed
        # below (or detached, when the frame has no image).
        self._flashing = None
        for tool in (self.sam_point, self.sam_box):
            tool.paused = False
        if not compat.is_open(self.session):
            self._render_nothing()
            return
        key = self.session.current()
        image = self.session.image()
        segment = self._pose_segment(key)
        keep = (self._segment == segment and self.canvas.image_rgb() is not None)
        zoom, centre = self.canvas.zoom_factor(), self._canvas_centre()

        self.tools_enabled = image is not None
        self.clear_prompt_box()  # before _attach_tool re-arms a SAM tool
        if image is None:
            # Steps mode is about the imported log, not about the picture: a
            # view with no image for this step must not take the table away.
            if self.mode != A.MODE_STEPS:
                self.stack.setCurrentWidget(self.placeholder_label)
            self._detach_tool()
            # The placeholder is in the middle of the window and easy to miss
            # while reading the task card; the status bar says which step.
            self.report(f"step {key.step}: 本视图没有图像 / no image in this view")
        else:
            self.stack.setCurrentWidget(
                self.steps_panel if self.mode == A.MODE_STEPS else self.canvas
            )
            self._ensure_overlay(image.shape[:2])
            self.canvas.set_image(image)
            # The view is put back *before* anything is composited.  The
            # overlay only paints what is about to be drawn, and ``set_image``
            # leaves the canvas fitted to the whole frame: compositing there
            # and then zooming back to the ROI paid for all 12 MP to show a
            # sixth of it, on every frame change.
            self._restore_view(keep, zoom, centre)
            # Edit layer first, committed masks second: one composite per frame.
            self._sync_editing_layer(repaint=False)
            self.refresh_overlay()
            self._attach_tool()
        self._segment = segment

        # One line per frame the annotator actually arrives on: with the
        # commits below it reconstructs a whole session, and it is the only
        # thing a 0-byte log file was not doing.
        self.logger.info("frame D%s/%s step %s (%s)", key.desktop, key.view,
                         key.step, self.session.frame_status(key.step))

        self.on_frame_changed_edit(key)
        self.on_frame_changed_pose(key)
        self.on_frame_changed_assist(key)
        self.update_status()

    def _render_nothing(self) -> None:
        """Show that there is nothing open, rather than the last thing there was.

        A desktop/view with no frame rows leaves the session closed.  Returning
        early here left the *previous* view's image, masks, instance list and
        timeline on screen with a tool armed over them: everything the annotator
        could see was about a frame they were no longer on.
        """
        self.tools_enabled = False
        self._segment = None
        self._detach_tool()
        self.stack.setCurrentWidget(self.placeholder_label)
        for panel in (self.timeline, self.task_card, self.instances, self.review):
            panel.refresh()
        self.update_status()

    def _pose_segment(self, key: FrameKey) -> Optional[int]:
        row = self.db.pose_segment_for(key) or {}
        return None if row.get("seg") is None else int(row["seg"])

    def _canvas_centre(self) -> tuple[float, float]:
        rect = self.canvas.viewport().rect()
        point = self.canvas.mapToScene(rect.center())
        return (point.x(), point.y())

    def _restore_view(self, keep: bool, zoom: float, centre) -> None:
        """Keep the zoom inside a pose segment, reset to the ROI across one."""
        if keep and zoom > 0:
            self.canvas.set_zoom(zoom)
            self.canvas.center_on(centre)
            return
        roi = self.roi()
        if roi is not None:
            self.canvas.zoom_to(roi)
        else:
            self.canvas.fit_image()

    def _ensure_overlay(self, hw: tuple[int, int]) -> None:
        if self.overlay is None or self.overlay.hw != tuple(hw):
            self.overlay = LabelOverlay(tuple(hw))
            self.canvas.set_overlay(self.overlay)
            for tool in self._all_tools():
                tool.overlay = self.overlay

    def refresh_overlay(self) -> None:
        """Repaint the committed masks of the frame (hidden instances left out).

        The windows travel with the masks: they are what lets the overlay
        repaint the one layer that moved, and what makes a second call with the
        same compiled frame -- which every commit makes, once here and once
        when the session re-announces the frame -- cost nothing at all.
        """
        if self.overlay is None:
            return
        masks, order, windows = compat.overlay_layers(self.session)
        self.overlay.set_instances(masks, order, windows=windows)
        self.canvas.refresh()


    # ----------------------------------------------------------------- modes
    @S.guard
    def set_mode(self, mode: str) -> None:
        """Switch between Steps, Annotate and Review."""
        if mode == self.mode:
            return
        if not self.can_leave_edit():
            self._sync_mode_tab()
            return
        if self.mode == A.MODE_STEPS and self._steps_dirty:
            if not self.confirm_discard("The step table has unsaved edits."):
                self._sync_mode_tab()
                return
            self._steps_dirty = False
        self.mode = mode
        self.disarm_bench()
        self.forget_draft_ghost()   # the canvas it was offered on is going away
        # ... and so is the frame a ``Shift+C`` alternate was an offer about.
        # Review arms no tool at all and Steps shows a table; coming back to
        # Annotate has to come back to the difference map's own box.
        self.reset_prompt_rank()
        if mode == A.MODE_STEPS:
            self.stack.setCurrentWidget(self.steps_panel)
        else:
            self.stack.setCurrentWidget(
                self.canvas if self.tools_enabled else self.placeholder_label
            )
        self.review_dock.setVisible(mode == A.MODE_REVIEW)
        if mode == A.MODE_REVIEW:
            self.review_dock.raise_()
            self.review.refresh()
        self._attach_tool()          # Review arms nothing; the others re-arm
        self._sync_mode_tab()
        self.focus_canvas()
        self.update_status()

    def _sync_mode_tab(self) -> None:
        index = [m for _t, m in MODE_TITLES].index(self.mode)
        if self.mode_tabs.currentIndex() != index:
            blocked = self.mode_tabs.blockSignals(True)
            self.mode_tabs.setCurrentIndex(index)
            self.mode_tabs.blockSignals(blocked)

    @S.guard
    def _on_mode_tab(self, index: int) -> None:
        self.set_mode(MODE_TITLES[index][1])

    def confirm_discard(self, why: str) -> bool:
        """Ask before throwing unsaved S1 edits away."""
        return confirm_discard_dialog(self, why)

    def _mark_steps_dirty(self, *_args) -> None:
        self._steps_dirty = True

    @S.guard
    def on_steps_saved(self, desktop: int) -> None:
        """Re-open the session on the same frame: instances and events changed."""
        step = self.session.current().step if compat.is_open(self.session) else None

        def reopen() -> None:
            self._steps_dirty = False
            # Apply re-cuts the pose segments when a step became (or stopped
            # being) a `reorient`, so the pieces on screen may be new ones.
            self.reset_roi_proposals()
            self.session.open(int(desktop), self.session.view, force=True)
            if step is not None and step in self.session.steps():
                self.session.goto(step, force=True)
            self.report(f"D{desktop}: step table saved, session reloaded")

        self.leave_frame(reopen)

    # ------------------------------------------------------- top-bar actions
    @S.guard
    def act_set_view(self, view: str) -> None:
        """Show another camera of the same machine."""
        if view == self.session.view:
            return
        if not self._has_frames(self.session.desktop, view):
            self.report(f"{view}: 这台机器没有这个视图的帧 / no frames in this view")
            self._sync_view_buttons()
            return
        step = self.session.current().step if compat.is_open(self.session) else None

        def switch() -> None:
            self.session.open(int(self.session.desktop), view, force=True)
            if step is not None and step in self.session.steps():
                self.session.goto(step, force=True)
            self._segment = None
            self.render_frame()
            self.refresh_desktop_counts()   # the count is per view

        self.leave_frame(switch)
        self._sync_view_buttons()
        self.focus_canvas()

    def _sync_view_buttons(self) -> None:
        """Check the button of the view that is actually open."""
        for name, button in self.view_buttons.items():
            button.setChecked(name == self.session.view)

    def focus_canvas(self) -> None:
        """Put the keyboard back where the annotator works.

        The toolbar widgets take no focus at all now, but a mode switch or a
        dock click can still leave it in a panel; after choosing something from
        the top bar the next key belongs to the canvas.
        """
        if self.stack.currentWidget() is self.canvas:
            self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)

    def _has_frames(self, desktop: Optional[int], view: str) -> bool:
        """Does this desktop/view have any frame rows at all?

        Opening one that has none left the *previous* view's image, masks and
        timeline on screen with a brush armed over them, and then saved that
        view as "last view" -- after which the next launch died in ``__init__``.
        Both moves are refused instead, with a line that says why.
        """
        if desktop is None:
            return False
        try:
            return bool(self.db.frames_for(int(desktop), str(view)))
        except Exception:  # noqa: BLE001 - a read that fails is not a reason to move
            return False

    @S.guard
    def _on_desktop_chosen(self, index: int) -> None:
        desktop = self.desktop_combo.itemData(index)
        if desktop is not None and int(desktop) != int(self.session.desktop):
            self.act_set_desktop(int(desktop))

    @S.guard
    def act_set_desktop(self, desktop: int) -> None:
        """Open another machine in the current view."""
        if not self._has_frames(int(desktop), self.session.view):
            self.report(f"D{desktop}: 这个视图没有帧 / no frames in this view")
            self._sync_desktop_combo()
            return

        def switch() -> None:
            self.session.open(int(desktop), self.session.view, force=True)
            if self._steps_panel is not None:
                self._steps_panel.set_desktop(int(desktop))
            self._segment = None
            self.render_frame()

        if not self.leave_frame(switch):
            self._sync_desktop_combo()

    def _sync_desktop_combo(self) -> None:
        """Point the chooser back at the machine that is open."""
        index = self.desktop_combo.findData(int(self.session.desktop))
        if index >= 0 and index != self.desktop_combo.currentIndex():
            blocked = self.desktop_combo.blockSignals(True)
            self.desktop_combo.setCurrentIndex(index)
            self.desktop_combo.blockSignals(blocked)

    # ------------------------------------------------------ navigation slots
    @S.guard
    def act_step(self, delta: int) -> None:
        """``PgDn`` goes to k-1 (the reverse-order "forward"), ``PgUp`` to k+1."""
        self.leave_frame(lambda: self.session.prev(force=True) if delta < 0
                         else self.session.next(force=True))

    @S.guard
    def act_step_edge(self, which: str) -> None:
        steps = self.session.steps()
        if steps:
            self.leave_frame(lambda: self.session.goto(
                min(steps) if which == "first" else max(steps), force=True))

    @S.guard
    def timeline_goto(self, step: int) -> None:
        """A click in the timeline (or a review entry): one guarded ``goto``.

        The panels call ``session.goto`` themselves, which would walk straight
        past the uncommitted-edit rule, so the window takes their activation
        signals over instead of letting them through.  Qt has already moved the
        selection by the time the click arrives, so a refusal has to put it
        back: a list pointing at a frame that is not on the canvas is a lie the
        annotator will act on.
        """
        if not self.leave_frame(lambda: self.session.goto(int(step), force=True)):
            self.timeline.select_current_step()
            self.review.select_current_step()
    # -------------------------------------------------------- session slots
    @S.guard
    def act_save(self) -> None:
        """``Ctrl+S``: save what is on screen.

        In Steps mode that is the step table, not the session: it used to save
        the session and say "saved" while the annotator's row edits were still
        sitting in the panel's model -- and "saved" is exactly the word that
        stops somebody pressing Apply.  Same path as the button, same message.
        """
        if self.mode == A.MODE_STEPS:
            self.steps_panel.apply()
            self.save_window_state()
            self.report(self.steps_panel.status.text() or "step table saved")
            return
        self.session.save()
        self.save_window_state()
        self.report("saved")

    @S.guard
    def act_refresh_all(self) -> None:
        """``F5``: recompile **this frame** and drain the re-check queue.

        It used to recompile the whole view, which on a 120-step machine froze
        the window for tens of seconds inside a Qt slot.  The choice made here
        is to keep ``F5`` instant and leave the batch to the command line --
        ``python -m tda.cli check --desktop N --view V`` does exactly that, with
        a lock, a progress line and an exit code.  A worker thread was the other
        option and was rejected: the session, its truth service and its
        ``sqlite3`` connection all belong to the GUI thread, and handing a
        second connection to a background recompile would have put two writers
        on one database for a convenience nobody asked for.
        """
        if not self.can_leave_edit():
            return
        key = self.session.current()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            stats = self.session.truth.refresh(key) or {}
        finally:
            QApplication.restoreOverrideCursor()
        # Only *enqueue* the backlog: draining it here would block the window
        # for as long as the queue is deep, and the sweeper is already the
        # thing that drains it. The parked frames are what needs the nudge.
        queued = compat.retry_rechecks(self.session)
        self.sweep_failures.clear()
        self.session.goto(key.step, force=True)   # re-read what the recompile changed
        self.review.refresh()
        # "queued" is only true when something is going to drain the queue.
        draining = bool(getattr(self.session, "sweeper_enabled", True))
        retry = f", {queued} re-check(s) queued" if queued and draining else ""
        # Two counts, because `conflicts` is what *this* recompile queued and
        # the truth table deliberately refuses to queue a disagreement twice:
        # on the second F5 a frame somebody is still arguing about reported
        # "0 conflicts". `open` is the same number the export and `cli check`
        # gates read, narrowed to this frame.
        self.report(f"step {key.step} recompiled: {stats.get('updated', 0)} rows, "
                    f"{stats.get('conflicts', 0)} new conflicts, "
                    f"{compat.open_conflicts(self.session, key)} open{retry} — use "
                    f"'python -m tda.cli check' for the whole view")

    @S.guard
    def _on_sweep_progress(self, done: int, total: int, failed: int = 0) -> None:
        """The session's background re-check of frozen frames is making headway."""
        suffix = f", {failed} failed — F5 retries" if failed else ""
        self.report(f"re-checking verified frames: {done}/{total}{suffix}")
        # Each verdict may repaint a row the annotator is not standing on, and
        # move the machine chooser's [done/total].
        self.timeline.refresh_statuses()
        self.refresh_desktop_counts()

    @S.guard
    def _on_sweep_error(self, step: int, text: str) -> None:
        """A background re-check gave up on a frame; it stays queued.

        The sweeper parks a frame after three failures rather than spinning on
        it, so without this the annotator would never learn that one frame of
        the machine is quietly out of date.
        """
        self.sweep_failures.add(int(step))
        self.report_error(f"re-check of step {step} failed: {text} — press F5 "
                          f"(or run 'python -m tda.cli check') to try again")

    @S.guard
    def _on_queues_changed(self) -> None:
        """A queue moved: the review panel and the timeline colours both follow.

        The timeline only ever repainted on a frame change, so a frame that
        became a conflict while the annotator worked two steps away kept its
        old colour until they happened to visit it.
        """
        self.review.refresh()
        self.timeline.refresh_statuses()

    @S.guard
    def _on_problems(self, problems: list) -> None:
        """The compiler has something to say about the open frame.

        The count and where to read them; the list itself is the task card's
        problems pane.  Joining them into the status bar put 2,550 characters
        into a one-line label.
        """
        if problems:
            self.report(f"{len(problems)} problem(s) — 见任务卡 / see the task card")

    @S.guard
    def _on_dirty(self, dirty: bool) -> None:
        mark = " *" if dirty else ""
        self.setWindowTitle(f"Teardown Annotator — {self.annotator}{mark}")

