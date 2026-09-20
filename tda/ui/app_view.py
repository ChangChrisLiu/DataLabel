"""Arming a tool and everything that only changes how the frame is *shown*.

Mixed into :class:`tda.ui.app.MainWindow`.  Nothing here writes an annotation:
one tool listens to the canvas at a time (in Review mode none does), the
overlay's opacity, outline, visibility and the pixel grid are view settings,
and the zoom actions are the two the keyboard offers beside the wheel.
"""
from __future__ import annotations

from typing import Optional

from tda.ui import app_actions as A
from tda.ui import app_support as S
from tda.ui.canvas.overlay import EDIT_RGB, OCCLUDER_RGB
from tda.ui.canvas.view import ToolCursor

__all__ = ["GRID_OFF", "OPACITY_STEP", "CANDIDATES_DROPPED", "TOOL_LABELS",
           "ToolsMixin", "tool_label_text"]

#: How much one ``,``/``.`` press moves the overlay alpha.
OPACITY_STEP = 20
#: Zoom the pixel grid is disabled at (the canvas draws it above ``GRID_ZOOM``).
GRID_OFF = 1e9
#: Said when a tool switch cancels a SAM prompt that still had proposals.
CANDIDATES_DROPPED = ("切到画笔会丢弃其余候选 / switching tool discards the other "
                      "SAM candidates")

#: Short bilingual name of each tool for the status bar.  Deliberately *not* the
#: ``label_zh`` of :data:`tda.ui.app_actions.ACTIONS` -- those are sentences
#: ("画笔（加像素）") and this is a badge that has to stay one glance wide.  The
#: **key** still comes from ``ACTIONS`` (:func:`tda.ui.app_actions.tool_key`),
#: so there is still one key map.
TOOL_LABELS: dict[str, tuple[str, str]] = {
    "brush": ("画笔", "Brush"),
    "eraser": ("橡皮擦", "Eraser"),
    "sam_point": ("SAM 点提示", "SAM point"),
    "sam_box": ("SAM 框选", "SAM box"),
    "occluder": ("遮挡层画笔", "Occluder"),
    "bench_box": ("台面框", "Bench box"),
    "roi": ("ROI 机箱范围框", "ROI box"),
}
#: Shown in place of a tool in the two states where the canvas takes no edits.
NO_TOOL_ZH = "只看不改"
NO_TOOL_EN = "read-only"


#: The ring is drawn, but at the minimum size rather than the stroke's: the
#: shape still says "brush", the size does not (round 2, I2).
RING_SMALL_NOTE = "（光标未按比例 / cursor not to scale）"
#: No ring could be made at all -- the stroke is wider than a cursor may be --
#: so the cursor is a crosshair and this number is the only size on screen.
RING_LARGE_NOTE = "（笔刷太大，光标画不出 / brush too large for a cursor）"
#: How :func:`tool_label_text` is told which of the two applies.
RING_NOTES = {"small": RING_SMALL_NOTE, "large": RING_LARGE_NOTE}


def tool_label_text(name: str, radius=None, armed: bool = True,
                    scale: str = "") -> tuple[str, str]:
    """``(what the status bar shows, what its tooltip says)`` for one tool.

    ``工具：SAM 框选 X`` rather than ``sam_box``: the annotator reads Chinese,
    the key is what they press to get back to it, and both used to be missing.
    ``r=<n>`` is always shown for a tool that has a radius, and ``scale``
    (``""``, ``"small"``, ``"large"``) adds the clause that says the cursor is
    not the size of the stroke -- so the number is never the only warning and
    the cursor is never quietly lying.
    """
    if not armed:
        return (f"工具：{NO_TOOL_ZH} / {NO_TOOL_EN}", NO_TOOL_EN)
    zh, en = TOOL_LABELS.get(name, (name, name))
    key = A.tool_key(name)
    suffix = "" if radius is None else f" r={int(radius)}"
    note = RING_NOTES.get(scale, "") if radius is not None else ""
    shown = f"工具：{zh}{(' ' + key) if key else ''}{suffix}{note}"
    tip = f"{en}{(' (' + key + ')') if key else ''}{suffix}{note}"
    return (shown, tip)


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
        self.note_tool_switch(name, via="key")
        self._tool_name = name
        self._attach_tool()
        self.update_status()

    def note_tool_switch(self, name: str, via: str) -> None:
        """One log line per tool switch: what to, from what, and how.

        The trial's log could not say whether ``B`` had taken effect, because a
        tool switch left no trace at all -- so four box prompts and four brush
        strokes looked exactly alike afterwards (task U1, ruling R3).
        """
        if name == self._tool_name:
            return
        self.logger.info("tool %s -> %s via %s", self._tool_name, name, via)

    @property
    def active_tool(self):
        """The tool receiving the canvas mouse signals, or ``None`` in Review.

        Review mode is **read-only on the canvas**: an edit begun there could
        not be settled (``Enter`` and ``Esc`` belong to Annotate mode and the
        mode switch is blocked by the very layer it would create), so no tool is
        armed and ``R`` takes the frame into Annotate mode instead.

        While the chassis rectangle is up the ROI tool is the one attached, and
        this used to answer with the brush -- a second place saying the wrong
        thing about which tool is armed (task U1, ruling R1).
        """
        if self.mode == A.MODE_REVIEW:
            return None
        return self._tool_for("roi" if self.roi_editing else self._tool_name)

    def _tool_for(self, name: str):
        return {
            "brush": self.brush, "eraser": self.eraser, "occluder": self.occluder,
            "sam_point": self.sam_point, "sam_box": self.sam_box,
            "bench_box": self.bench_tool, "roi": self.roi_tool,
        }.get(name, self.brush)

    def armed_tool_name(self) -> Optional[str]:
        """Which tool the canvas is answering to, or ``None`` when none is.

        ``_tool_name`` is what the annotator last *asked* for; this is what is
        actually live, which is not the same thing while the ROI rectangle is
        up, in Review mode, on a frame with no image, or while ``Tab`` holds
        the neighbour on screen.
        """
        if self.mode == A.MODE_REVIEW or not self.tools_enabled:
            return None
        if self.is_flashing():
            return None
        return "roi" if self.roi_editing else self._tool_name

    def _tool_cursor_spec(self) -> ToolCursor:
        """What the cursor over the canvas should be right now (ruling R1)."""
        name = self.armed_tool_name()
        if name is None:
            # Review mode and a flashed neighbour are both "not this frame":
            # the press will do nothing and the cursor says so before it.
            return ToolCursor("forbidden" if self.is_flashing()
                              or self.mode == A.MODE_REVIEW else "arrow")
        if name == "brush":
            return ToolCursor("circle", EDIT_RGB, self.brush.radius)
        if name == "eraser":
            # Dashed and white: told apart from the brush without colour alone.
            return ToolCursor("circle", (245, 245, 245), self.eraser.radius,
                              dashed=True)
        if name == "occluder":
            return ToolCursor("circle", OCCLUDER_RGB, self.occluder.radius)
        return ToolCursor("cross")

    def sync_tool_cursor(self) -> None:
        """Put the armed tool's cursor on the canvas."""
        self.canvas.set_tool_cursor(self._tool_cursor_spec())

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
        self.sync_tool_cursor()

    def _detach_tool(self) -> None:
        for tool in self._all_tools():
            tool.detach()
        self.sync_tool_cursor()

    @S.guard
    def act_radius(self, delta: int) -> None:
        """``[`` / ``]``: every pixel tool shares one radius."""
        radius = max(0, self.brush.radius + int(delta))
        for tool in (self.brush, self.eraser, self.occluder):
            tool.set_radius(radius)
        self.sync_tool_cursor()   # the ring is the size of the stroke
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
        # The stored ROI's thin outline is a layer like the others; the
        # rectangle *being edited* is not -- it is a question waiting for an
        # answer, and hiding it would be hiding the answer.
        self.canvas.roi_outline_visible = not self.canvas.roi_outline_visible
        self.canvas.viewport().update()
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
        # ``F`` is the annotator asking for the move a mid-edit store put off.
        self._roi_fit_pending = False
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

