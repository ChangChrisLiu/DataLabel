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
:mod:`tda.ui.app_assist` and the widgets, the status bar and the lifecycle in
:mod:`tda.ui.app_shell`; all of them are mixed in below.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.app_assist import AssistMixin
from tda.ui.app_commit import CommitMixin
from tda.ui.app_edit import EditMixin
from tda.ui.app_roi import RoiMixin
from tda.ui.app_shell import (
    MODE_TITLES,
    ShellMixin,
    confirm_discard_dialog,
    main,
    take_lock,
)
from tda.ui.canvas.overlay import LabelOverlay

__all__ = ["MainWindow", "main", "take_lock"]

#: How much one ``,``/``.`` press moves the overlay alpha.
OPACITY_STEP = 20
#: Zoom the pixel grid is disabled at (the canvas draws it above ``GRID_ZOOM``).
GRID_OFF = 1e9
#: ``_flashing`` when a neighbour is on screen but its step number is unknown.
FLASH_UNNAMED = -1


class MainWindow(EditMixin, CommitMixin, RoiMixin, AssistMixin, ShellMixin, QMainWindow):
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
        self._init_assist(sam_queue)
        self._connect_session()

        self.setWindowTitle(f"Teardown Annotator — {self.annotator}")
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
            self._steps_panel = panel
            self.stack.addWidget(panel)
        return self._steps_panel

    # ------------------------------------------------------------ frame paint
    @S.guard
    def _on_frame_changed(self, _key: object) -> None:
        self.render_frame()

    def render_frame(self) -> None:
        """Repaint everything that belongs to the frame the session is on."""
        # Whatever was being compared, this is not it; the tools are re-armed
        # below (or detached, when the frame has no image).
        self._flashing = None
        for tool in (self.sam_point, self.sam_box):
            tool.paused = False
        if not compat.is_open(self.session):
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
        else:
            self.stack.setCurrentWidget(
                self.steps_panel if self.mode == A.MODE_STEPS else self.canvas
            )
            self._ensure_overlay(image.shape[:2])
            self.canvas.set_image(image)
            # Edit layer first, committed masks second: one composite per frame.
            self._sync_editing_layer(repaint=False)
            self.refresh_overlay()
            self._restore_view(keep, zoom, centre)
            self._attach_tool()
        self._segment = segment

        self.on_frame_changed_edit(key)
        self.on_frame_changed_assist(key)
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
        """Repaint the committed masks of the frame (hidden instances left out)."""
        if self.overlay is None:
            return
        masks, order = compat.overlay_layers(self.session)
        self.overlay.set_instances(masks, order)
        self.canvas.refresh()

    # -------------------------------------------------------------- keyboard
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: D102
        kind = event.type()
        if self.closed:
            return False  # a window on its way out must not eat anybody's keys
        if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            focus = QApplication.focusWidget()
            if focus is None or focus is self or self.isAncestorOf(focus):
                if self.handle_key(event):
                    return True
        return super().eventFilter(obj, event)

    def handle_key(self, event) -> bool:
        """Run the action bound to ``event``; ``True`` when it was consumed.

        An auto-repeat of a bound key is **consumed but not fired**: holding
        ``Tab`` for the flash compare used to let the repeats through to Qt's
        focus chain, which walked the focus into a combo box -- after which
        :func:`~tda.ui.app_actions.blocks_shortcuts` switched the whole keyboard
        off until the annotator clicked somewhere. An auto-repeat of a key that
        is *not* bound is left alone, so ordinary widgets keep their repeats.
        """
        if not self._shortcut_context_ok():
            return False
        focus = self._focus_widget()
        if A.blocks_shortcuts(focus) or A.navigates_a_list(focus, event.key()):
            return False
        action = A.action_for(event.key(), event.modifiers(), self.mode)
        if action is None:
            # Any other key is the annotator moving on: a flash that is still up
            # because its release went missing ends here.
            if event.type() == QEvent.Type.KeyPress:
                self.end_flash()
            return False
        if not action.hold and event.type() == QEvent.Type.KeyPress:
            self.end_flash()
        if event.isAutoRepeat():
            return True
        pressed = event.type() == QEvent.Type.KeyPress
        if action.hold:
            self.dispatch(action, pressed)
        elif pressed:
            self.dispatch(action)
        return True

    def _shortcut_context_ok(self) -> bool:
        """Are the window's shortcuts live at all right now?

        Not while a modal dialog is up, and not while the focus sits in another
        **visible** window of ours -- the cheat sheet is a child dialog, so
        without this its ``Esc`` would also discard the edit underneath it.

        The visibility check matters: Qt keeps the application focus on a widget
        of a window that has been closed but not yet deleted, so a torn-down
        window would otherwise switch off the keyboard of the one that replaced
        it -- which is exactly what made a whole suite fail when another suite
        had run first.
        """
        if QApplication.activeModalWidget() is not None:
            return False
        focus = QApplication.focusWidget()
        if focus is None:
            return True
        other = focus.window()
        return other is self or not other.isVisible()

    def _focus_widget(self) -> Optional[QWidget]:
        """The focused widget *of this window*, or ``None``.

        ``QApplication.focusWidget()`` is authoritative while the window is
        active; when it is not (or nothing has been shown yet) the window's own
        ``focusWidget()`` still knows which child last took the focus, which is
        what makes the text-field guard work before the first activation.
        """
        focus = QApplication.focusWidget()
        if focus is not None and not (focus is self or self.isAncestorOf(focus)):
            focus = None
        return focus if focus is not None else self.focusWidget()

    def dispatch(self, action: A.Action, *extra) -> None:
        """Call the window slot an action names."""
        slot = getattr(self, action.slot, None)
        if slot is None:
            self.report_error(f"no slot {action.slot!r} for {action.name}")
            return
        slot(*(tuple(extra) if action.hold else tuple(action.args)))

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
        step = self.session.current().step if compat.is_open(self.session) else None

        def switch() -> None:
            self.session.open(int(self.session.desktop), view, force=True)
            if step is not None and step in self.session.steps():
                self.session.goto(step, force=True)
            self._segment = None
            self.render_frame()

        self.leave_frame(switch)
        for name, button in self.view_buttons.items():
            button.setChecked(name == self.session.view)

    @S.guard
    def _on_desktop_chosen(self, index: int) -> None:
        desktop = self.desktop_combo.itemData(index)
        if desktop is not None and int(desktop) != int(self.session.desktop):
            self.act_set_desktop(int(desktop))

    @S.guard
    def act_set_desktop(self, desktop: int) -> None:
        """Open another machine in the current view."""
        def switch() -> None:
            self.session.open(int(desktop), self.session.view, force=True)
            if self._steps_panel is not None:
                self._steps_panel.set_desktop(int(desktop))
            self._segment = None
            self.render_frame()

        if not self.leave_frame(switch):
            index = self.desktop_combo.findData(int(self.session.desktop))
            if index >= 0:
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

    @S.guard
    def act_flash_compare(self, pressed: bool, other: bool = False) -> None:
        """Hold ``Tab`` to see the neighbour frame without moving the view.

        The neighbour is the frame the task card is written against -- the one
        the annotator came from, ``j + 1`` in reverse order.  ``Shift+Tab``
        shows the other side instead.

        While it is held the canvas is showing a frame that is **not** the one
        being annotated, so nothing may be drawn on it: every tool is detached
        and the SAM tools refuse to prompt.  A brush stroke or a SAM click on
        the flashed image asked about the neighbour's pixels and wrote the
        answer into this frame's layer -- and in reverse order the part the card
        asks for is *absent* in j+1, so the mask was confidently wrong.
        """
        if not pressed:
            self.end_flash()
            return
        if self._flashing is not None or not compat.is_open(self.session):
            return
        image = compat.flash_image(self.session, other=other)
        if image is None:
            return
        # The step shown, so the status bar can name it; ``FLASH_UNNAMED`` when
        # the adapter cannot say which one it handed back.
        step = compat.flash_step(self.session, other=other)
        self._flashing = FLASH_UNNAMED if step is None else int(step)
        self._pause_tools(True)
        self._show_image(image)
        self.update_status()

    def end_flash(self) -> None:
        """Put the frame back on the canvas; safe to call at any time.

        Called from everywhere a release might never arrive: the key release,
        the window losing focus (``Alt+Tab`` while holding ``Tab`` is the one
        the reviewer hit), any other key, and every frame change.  Without it
        the canvas stayed on the neighbour's image with this frame's overlay and
        status, and every tool stayed live over it.
        """
        if self._flashing is None:
            return
        self._flashing = None
        self._pause_tools(False)
        image = self.session.image() if compat.is_open(self.session) else None
        if image is not None:
            self._show_image(image)
        self.update_status()

    def is_flashing(self) -> bool:
        """Is the canvas showing a neighbour frame rather than the open one?"""
        return self._flashing is not None

    def _show_image(self, image) -> None:
        """Swap the picture under the overlay, keeping zoom and centre."""
        zoom, centre = self.canvas.zoom_factor(), self._canvas_centre()
        self.canvas.set_image(image)
        self.canvas.set_zoom(zoom)
        self.canvas.center_on(centre)
        self.canvas.refresh()

    def _pause_tools(self, paused: bool) -> None:
        """Make every tool inert, or arm the chosen one again."""
        for tool in (self.sam_point, self.sam_box):
            tool.paused = bool(paused)
        if paused:
            self._detach_tool()
        else:
            self._attach_tool()

    def event(self, ev) -> bool:  # noqa: D102 - Qt override
        # A lost key release (Alt+Tab, a focus steal, a system dialog) would
        # otherwise leave the canvas stuck on the neighbour for good.  Qt
        # delivers the deactivation here, not through ``changeEvent``.
        kind = ev.type()
        if kind in (QEvent.Type.WindowDeactivate, QEvent.Type.FocusOut) or (
            kind == QEvent.Type.ActivationChange and not self.isActiveWindow()
        ):
            self.end_flash()
        return super().event(ev)

    @S.guard
    def act_flash_other(self, pressed: bool) -> None:
        """``Shift+Tab``: flash the frame on the *other* side of this one."""
        self.act_flash_compare(pressed, other=True)

    # ----------------------------------------------------------- tool slots
    def _all_tools(self) -> tuple:
        return (self.brush, self.eraser, self.occluder, self.sam_point,
                self.sam_box, self.roi_tool, self.bench_tool)

    @S.guard
    def act_tool(self, name: str) -> None:
        """Arm one tool; the SAM tools refuse when no model is loaded."""
        if name in ("sam_point", "sam_box") and not self.sam_available:
            self.report(f"SAM is unavailable: {self.sam_reason}")
            return
        if name != "bench_box":
            self.disarm_bench()   # the arm belongs to the box tool, not to the brush
        self.cancel_roi_edit()
        self._tool_name = name
        self._attach_tool()
        self.update_status()

    @property
    def active_tool(self):
        """The tool receiving the canvas mouse signals, or ``None`` in Review.

        Review mode is **read-only on the canvas**: an edit begun there could
        not be settled (``Enter`` and ``Esc`` belong to Annotate mode and the
        mode switch is blocked by the very layer it would create), so no tool is
        armed and ``R`` takes the frame into Annotate mode instead.
        """
        if self.mode == A.MODE_REVIEW:
            return None
        return self._tool_for(self._tool_name)

    def _tool_for(self, name: str):
        return {
            "brush": self.brush, "eraser": self.eraser, "occluder": self.occluder,
            "sam_point": self.sam_point, "sam_box": self.sam_box,
            "bench_box": self.bench_tool, "roi": self.roi_tool,
        }.get(name, self.brush)

    def _attach_tool(self) -> None:
        """Exactly one tool listens to the canvas; a SAM tool is re-armed after.

        In Review mode none is: the canvas is there to look at the frame a queue
        entry points to, not to edit it.
        """
        wanted = (None if self.mode == A.MODE_REVIEW
                  else self._tool_for("roi" if self.roi_editing else self._tool_name))
        for tool in self._all_tools():
            if tool is not wanted:
                tool.detach()
        if self.tools_enabled and wanted is not None:
            wanted.attach()
            if wanted in (self.sam_point, self.sam_box):
                self.rearm_sam()

    def _detach_tool(self) -> None:
        for tool in self._all_tools():
            tool.detach()

    @S.guard
    def act_radius(self, delta: int) -> None:
        """``[`` / ``]``: every pixel tool shares one radius."""
        radius = max(0, self.brush.radius + int(delta))
        for tool in (self.brush, self.eraser, self.occluder):
            tool.set_radius(radius)
        self.update_status()

    # --------------------------------------------------------- display slots
    @S.guard
    def act_toggle_overlays(self) -> None:
        if self.overlay is not None:
            self.overlay.visible = not self.overlay.visible
            self.canvas.refresh()

    @S.guard
    def act_toggle_outline(self) -> None:
        self.canvas.overlay_outline = not self.canvas.overlay_outline
        self.canvas.refresh()

    @S.guard
    def act_opacity(self, delta: int) -> None:
        alpha = self.canvas.overlay_alpha + int(delta) * OPACITY_STEP
        self.canvas.overlay_alpha = int(min(255, max(0, alpha)))
        self.canvas.refresh()
        self.report(f"overlay opacity {self.canvas.overlay_alpha}/255")

    @S.guard
    def act_toggle_grid(self) -> None:
        """The canvas draws the grid above ``GRID_ZOOM``; this parks the threshold."""
        default = type(self.canvas).GRID_ZOOM
        self.canvas.GRID_ZOOM = GRID_OFF if self.canvas.GRID_ZOOM == default else default
        self.canvas.viewport().update()
        self.report("pixel grid " + ("off" if self.canvas.GRID_ZOOM > 100 else "on"))

    @S.guard
    def act_fit_image(self) -> None:
        self.canvas.fit_image()
        self.update_status()

    @S.guard
    def act_fit_roi(self) -> None:
        roi = self.roi()
        self.canvas.zoom_to(roi) if roi is not None else self.canvas.fit_image()
        self.update_status()

    @S.guard
    def act_cheat_sheet(self) -> None:
        """The ``?`` / ``F12`` sheet, generated from the same table as the guide."""
        from PySide6.QtWidgets import QDialog, QTextBrowser

        if self._cheat_sheet is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("快捷键 / Shortcuts")
            browser = QTextBrowser(dialog)
            browser.setHtml(A.cheat_sheet_html())
            layout = QVBoxLayout(dialog)
            layout.addWidget(browser)
            dialog.resize(520, 640)
            self._cheat_sheet = dialog
        self._cheat_sheet.show()
        self._cheat_sheet.raise_()

    # -------------------------------------------------------- session slots
    @S.guard
    def act_save(self) -> None:
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
        self.report(f"step {key.step} recompiled: {stats.get('updated', 0)} rows, "
                    f"{stats.get('conflicts', 0)} conflicts{retry} — use "
                    f"'python -m tda.cli check' for the whole view")

    @S.guard
    def _on_sweep_progress(self, done: int, total: int, failed: int = 0) -> None:
        """The session's background re-check of frozen frames is making headway."""
        suffix = f", {failed} failed — F5 retries" if failed else ""
        self.report(f"re-checking verified frames: {done}/{total}{suffix}")

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
        self.review.refresh()

    @S.guard
    def _on_problems(self, problems: list) -> None:
        if problems:
            self.report(f"{len(problems)} problem(s): {'; '.join(str(p) for p in problems)}")

    @S.guard
    def _on_dirty(self, dirty: bool) -> None:
        mark = " *" if dirty else ""
        self.setWindowTitle(f"Teardown Annotator — {self.annotator}{mark}")

