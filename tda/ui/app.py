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

The editing, ROI, review and crash-safety flows live in :mod:`tda.ui.app_edit`,
the model assist in :mod:`tda.ui.app_assist` and the widgets, the status bar and
the lifecycle in :mod:`tda.ui.app_shell`; all three are mixed in below.
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
from tda.ui.app_edit import EditMixin
from tda.ui.app_shell import MODE_TITLES, ShellMixin, main, take_lock
from tda.ui.canvas.overlay import LabelOverlay

__all__ = ["MainWindow", "main", "take_lock"]

#: How much one ``,``/``.`` press moves the overlay alpha.
OPACITY_STEP = 20
#: Zoom the pixel grid is disabled at (the canvas draws it above ``GRID_ZOOM``).
GRID_OFF = 1e9


class MainWindow(EditMixin, AssistMixin, ShellMixin, QMainWindow):
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
        self.review_refreshes = 0

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
        if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            focus = QApplication.focusWidget()
            if focus is None or focus is self or self.isAncestorOf(focus):
                if self.handle_key(event):
                    return True
        return super().eventFilter(obj, event)

    def handle_key(self, event) -> bool:
        """Run the action bound to ``event``; ``True`` when it was consumed."""
        if event.isAutoRepeat():
            return False
        if A.blocks_shortcuts(self._focus_widget()):
            return False
        action = A.action_for(event.key(), event.modifiers(), self.mode)
        if action is None:
            return False
        pressed = event.type() == QEvent.Type.KeyPress
        if action.hold:
            self.dispatch(action, pressed)
        elif pressed:
            self.dispatch(action)
        return True

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
        if self.mode == A.MODE_STEPS and self._steps_dirty:
            if not self.confirm_discard("The step table has unsaved edits."):
                self._sync_mode_tab()
                return
            self._steps_dirty = False
        self.mode = mode
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
        """Ask before throwing unsaved S1 edits away (overridden in tests)."""
        answer = QMessageBox.question(
            self, "Unsaved changes", f"{why}\nLeave the step table anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _mark_steps_dirty(self, *_args) -> None:
        self._steps_dirty = True

    @S.guard
    def on_steps_saved(self, desktop: int) -> None:
        """Re-open the session on the same frame: instances and events changed."""
        self._steps_dirty = False
        step = self.session.current().step if compat.is_open(self.session) else None
        self.session.open(int(desktop), self.session.view)
        if step is not None:
            self.session.goto(step)
        self.report(f"D{desktop}: step table saved, session reloaded")

    # ------------------------------------------------------- top-bar actions
    @S.guard
    def act_set_view(self, view: str) -> None:
        """Show another camera of the same machine."""
        if view == self.session.view:
            return
        step = self.session.current().step if compat.is_open(self.session) else None
        self.session.open(int(self.session.desktop), view)
        if step is not None and step in self.session.steps():
            self.session.goto(step)
        for name, button in self.view_buttons.items():
            button.setChecked(name == view)
        self._segment = None
        self.render_frame()

    @S.guard
    def _on_desktop_chosen(self, index: int) -> None:
        desktop = self.desktop_combo.itemData(index)
        if desktop is not None and int(desktop) != int(self.session.desktop):
            self.act_set_desktop(int(desktop))

    @S.guard
    def act_set_desktop(self, desktop: int) -> None:
        """Open another machine in the current view."""
        self.session.open(int(desktop), self.session.view)
        if self._steps_panel is not None:
            self._steps_panel.set_desktop(int(desktop))
        self._segment = None
        self.render_frame()

    # ------------------------------------------------------ navigation slots
    @S.guard
    def act_step(self, delta: int) -> None:
        """``PgDn`` goes to k-1 (the reverse-order "forward"), ``PgUp`` to k+1."""
        self.session.prev() if delta < 0 else self.session.next()

    @S.guard
    def act_step_edge(self, which: str) -> None:
        steps = self.session.steps()
        if steps:
            self.session.goto(min(steps) if which == "first" else max(steps))

    @S.guard
    def act_flash_compare(self, pressed: bool) -> None:
        """Hold ``Tab`` to see step k-1 in place of k, without moving the view."""
        image = self.session.flash_compare() if pressed else self.session.image()
        if image is None:
            return
        zoom, centre = self.canvas.zoom_factor(), self._canvas_centre()
        self.canvas.set_image(image)
        self.canvas.set_zoom(zoom)
        self.canvas.center_on(centre)
        self.canvas.refresh()

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
        self.cancel_roi_edit()
        self._tool_name = name
        self._attach_tool()
        self.update_status()

    @property
    def active_tool(self):
        """The tool currently receiving the canvas mouse signals."""
        return self._tool_for(self._tool_name)

    def _tool_for(self, name: str):
        return {
            "brush": self.brush, "eraser": self.eraser, "occluder": self.occluder,
            "sam_point": self.sam_point, "sam_box": self.sam_box,
            "bench_box": self.bench_tool, "roi": self.roi_tool,
        }.get(name, self.brush)

    def _attach_tool(self) -> None:
        """Exactly one tool listens to the canvas; a SAM tool is re-armed after."""
        wanted = self._tool_for("roi" if self.roi_editing else self._tool_name)
        for tool in self._all_tools():
            if tool is not wanted:
                tool.detach()
        if self.tools_enabled:
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
        """Recompile the whole view (seconds); the cursor says so meanwhile."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            stats = self.session.refresh_all() or {}
            QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
        self.review.refresh()
        self.report(f"recompiled: {stats.get('updated', 0)} rows, "
                    f"{stats.get('conflicts', 0)} conflicts")

    def _on_sweep_progress(self, done: int, total: int) -> None:
        self.report(f"re-checking verified frames: {done}/{total}")

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

