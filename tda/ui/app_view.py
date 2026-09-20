"""Arming a tool and everything that only changes how the frame is *shown*.

Mixed into :class:`tda.ui.app.MainWindow`.  Nothing here writes an annotation:
one tool listens to the canvas at a time (in Review mode none does), the
overlay's opacity, outline, visibility and the pixel grid are view settings,
and the zoom actions are the two the keyboard offers beside the wheel.
"""
from __future__ import annotations

from tda.ui import app_actions as A
from tda.ui import app_support as S

__all__ = ["GRID_OFF", "OPACITY_STEP", "CANDIDATES_DROPPED", "ToolsMixin"]

#: How much one ``,``/``.`` press moves the overlay alpha.
OPACITY_STEP = 20
#: Zoom the pixel grid is disabled at (the canvas draws it above ``GRID_ZOOM``).
GRID_OFF = 1e9
#: Said when a tool switch cancels a SAM prompt that still had proposals.
CANDIDATES_DROPPED = ("切到画笔会丢弃其余候选 / switching tool discards the other "
                      "SAM candidates")


class ToolsMixin:
    """The armed tool, the overlay's appearance and the zoom actions."""


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
        losing = self._candidate_tool()
        if (name not in ("sam_point", "sam_box") and losing is not None
                and losing.candidate_count > 1):
            # ``detach()`` cancels the prompt, which takes the other proposals
            # with it.  That is the right thing to do -- they belong to a tool
            # that is no longer listening -- but it has to be said, or ``C``
            # simply stops working after a detour through the brush.  With one
            # proposal there is nothing to walk and nothing to say.
            self.report(CANDIDATES_DROPPED)
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
        """``A``: all overlays off -- and with them anything waiting for a key.

        A draft ghost that is merely *invisible* still owns ``Enter``, so the
        next press would adopt pixels nobody can see.  Hiding the layers ends
        that offer instead.
        """
        self.forget_draft_ghost()
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
        from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

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

