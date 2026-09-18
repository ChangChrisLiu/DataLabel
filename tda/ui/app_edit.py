"""The editing half of the main window: strokes, scopes, ROI, review, crash safety.

Mixed into :class:`tda.ui.app.MainWindow`.  It is the part of the wiring that
touches an annotation, so every path here follows the same two rules: the
session decides (the window never writes to the database itself except for the
pose-segment ROI, which is a view setting the session has no opinion on), and no
exception reaches the Qt event loop.

Three flows are worth reading as a whole:

* **A stroke.**  A tool paints into the overlay's edit layer and reports the
  dirty rect; the window turns the tool's ``stroke_before`` snapshot plus the
  layer into one undoable op on the session and writes the layer to the crash
  sidecar.  That is the only thing a crash can lose, and it is on disk before
  the mouse button is up long enough to notice.
* **A commit.**  ``Enter`` asks the session what scope the edit means.  A plain
  ``keyframe`` is written immediately; anything else (a layering statement, a
  split) opens a small non-modal bar showing the suggestion and its two
  alternatives, because a modal dialog every few seconds is unusable at six
  hours a day.
* **The ROI.**  The first open of a pose segment with no stored ROI proposes
  ``suggest_roi`` as a draggable rectangle.  ``Enter`` stores it on the segment,
  and it then feeds the difference map and the ``F`` key.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from tda.core import masks as _masks
from tda.core.cache import suggest_roi
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui.canvas.tools import BrushTool, EraserTool, OccluderTool, Tool

#: Where the timeline and the review panel keep a row's step number.
_STEP_ROLE = int(Qt.ItemDataRole.UserRole)

__all__ = ["BLOCK_HINT", "BoxDragTool", "EditMixin", "DESPECKLE_MIN_PX",
           "NO_INSTANCE_HINT"]

#: Components smaller than this are specks (``Shift+D``); spec 9.1's floor.
DESPECKLE_MIN_PX = 16
#: A drag shorter than this on either side is a click, not a box.
MIN_BOX_PX = 2.0
#: How long the editing layer may sit unsaved before it reaches the sidecar.
SIDECAR_DEBOUNCE_MS = 300
#: Shown instead of losing an uncommitted edit.  Not a dialog: at six hours a
#: day a modal box on every mis-press is worse than the mistake it prevents.
BLOCK_HINT = ("未提交的修改：Enter 提交 / Esc 放弃  "
              "(uncommitted edit: Enter to commit, Esc to discard)")
#: Shown when a stroke lands with no instance to put it on.
NO_INSTANCE_HINT = ("先在任务卡或实例表里选一个实例  "
                    "(pick an instance in the task card or the instance list first)")
#: Task kinds a stroke may adopt on its own: both of them mean "draw pixels".
_DRAWABLE = (api.KIND_ADD_SHAPE, api.KIND_SPLIT_KEYFRAME)


class BoxDragTool(Tool):
    """Drag a rectangle on the canvas; used for the ROI and for bench boxes.

    It writes nothing: the box is reported and whoever armed the tool decides
    what it means -- the pose segment's ROI, or the staging-area box of a part
    that is now on the bench (spec 4.2 S4).
    """

    sigBox = Signal(object)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.box: Optional[tuple[float, float, float, float]] = None
        self._start: Optional[tuple[float, float]] = None

    def on_press(self, x: float, y: float, ev: Any) -> None:
        self._start = (float(x), float(y))

    def on_move(self, x: float, y: float, ev: Any) -> None:
        if self._start is None:
            return
        self.box = self._norm(x, y)
        if self.canvas is not None:
            self.canvas.set_rubber_band(self.box)

    def on_release(self, x: float, y: float, ev: Any) -> None:
        if self._start is None:
            return
        box = self._norm(x, y)
        self._start = None
        if box[2] - box[0] < MIN_BOX_PX or box[3] - box[1] < MIN_BOX_PX:
            return
        self.box = box
        self.sigBox.emit(box)

    def _norm(self, x: float, y: float) -> tuple[float, float, float, float]:
        sx, sy = self._start or (x, y)
        return (min(sx, x), min(sy, y), max(sx, x), max(sy, y))


class _Bar(QWidget):
    """A one-line non-modal bar under the canvas (scope suggestion, restore offer)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(self.label, 1)
        self.buttons = layout
        self.hide()

    def add_button(self, text: str, slot) -> QPushButton:
        """A button whose ``clicked(bool)`` never reaches a no-argument slot."""
        button = QPushButton(text, self)
        button.clicked.connect(lambda _checked=False: slot())
        self.buttons.addWidget(button)
        return button

    def show_text(self, text: str) -> None:
        self.label.setText(text)
        self.show()


class EditMixin:
    """Editing, edit scopes, the ROI, review handling and the crash sidecar."""

    # ------------------------------------------------------------------ setup
    def _init_edit(self) -> None:
        self.brush = BrushTool(self.canvas, None)
        self.eraser = EraserTool(self.canvas, None)
        self.occluder = OccluderTool(self.canvas, None)
        self.roi_tool = BoxDragTool(self.canvas, None)
        self.bench_tool = BoxDragTool(self.canvas, None)
        for tool in (self.brush, self.eraser, self.occluder):
            tool.sigStroke.connect(self.on_stroke)
        self.roi_tool.sigBox.connect(self.on_roi_box)
        self.bench_tool.sigBox.connect(self.on_bench_box)
        # Connected before any tool is attached, so the window sees a press
        # first and can adopt an instance for it (or mark it as doomed).
        self.canvas.sigMousePress.connect(self._on_canvas_press)
        self._paint_blocked = False
        self._blocked_layer: Optional[np.ndarray] = None

        self.refusal = compat.session_refusal()
        self.sidecar_writes = 0
        self._sidecar_pending: Optional[tuple] = None
        self._sidecar_timer = QTimer(self)
        self._sidecar_timer.setSingleShot(True)
        self._sidecar_timer.setInterval(SIDECAR_DEBOUNCE_MS)
        self._sidecar_timer.timeout.connect(self.flush_sidecar)

        self.roi_editing = False
        self.roi_draft: Optional[tuple] = None
        #: ``(desktop, view, segment)`` already proposed in this run.  Asking
        #: again on every frame of a segment would steal the tool -- and with
        #: it the pending SAM prompt -- from an annotator who declined once.
        self._roi_asked: set[tuple] = set()
        self._pending_scope: Optional[str] = None
        self._restore_offer: Optional[dict] = None

        self.scope_bar = _Bar(self)
        self.scope_bar.add_button("Enter 接受", self.act_commit)
        self.scope_bar.add_button("Alt+Enter 仅本帧", self.act_commit_override)
        self.scope_bar.add_button("Ctrl+K 从此拆分", self.act_commit_split)
        self.restore_bar = _Bar(self)
        self.restore_bar.add_button("恢复 Restore", self.restore_pending)
        self.restore_bar.add_button("丢弃 Discard", self.discard_pending)
        self._central_layout.addWidget(self.scope_bar)
        self._central_layout.addWidget(self.restore_bar)

    def _rewire_panels(self) -> None:
        """Take over the panel gestures the window has to be able to refuse.

        Three of them reach the session directly, and all three have an answer
        only the window can give: a conflict resolution has three verdicts, a
        timeline click may not leave an uncommitted edit behind, and a refused
        reorder must not escape a Qt slot as an exception.  The panels keep
        their logic; the window only intercepts the connection.
        """
        for button, resolution in ((self.review.keep_old_button, api.RESOLVE_KEEP_OLD),
                                   (self.review.accept_new_button,
                                    api.RESOLVE_ACCEPT_NEW)):
            self._reconnect(button.clicked,
                            lambda _c=False, r=resolution: self.resolve_selected(r))
        self._reconnect(self.timeline.list_widget().itemClicked,
                        lambda item: self.timeline_goto(int(item.data(_STEP_ROLE))))
        for queue in api.QUEUE_NAMES:
            self._reconnect(self.review.list_for(queue).itemActivated,
                            lambda item: self.timeline_goto(int(item.data(_STEP_ROLE))))
        self._reconnect(self.instances.up_button.clicked,
                        lambda _c=False: self.move_instance(-1))
        self._reconnect(self.instances.down_button.clicked,
                        lambda _c=False: self.move_instance(+1))
        # The panel's key handler calls these by name, so shadow them on the
        # instance: re-wiring the buttons alone would leave Ctrl+Up unguarded.
        self._panel_move = lambda d: (self.instances.__class__.move_up(self.instances)
                                      if d < 0 else
                                      self.instances.__class__.move_down(self.instances))
        self.instances.move_up = lambda: self.move_instance(-1)
        self.instances.move_down = lambda: self.move_instance(+1)

    @staticmethod
    def _reconnect(signal, slot) -> None:
        """Replace whatever a panel connected to one of its own signals."""
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):  # pragma: no cover - nothing was connected
            pass
        signal.connect(slot)

    # ------------------------------------------------------------ frame hook
    def on_frame_changed_edit(self, key) -> None:
        """What the editing half has to do when the frame changes."""
        self._pending_scope = None
        self.scope_bar.hide()
        if self.session.image() is not None:
            segment = (int(key.desktop), str(key.view), self._pose_segment(key))
            if self.roi() is not None:
                self._roi_asked.add(segment)
                if self.roi_editing:
                    self.cancel_roi_edit()
            elif segment not in self._roi_asked and not self.roi_editing:
                self._roi_asked.add(segment)
                self.start_roi_edit()
        self._offer_restore(key)

    def _sync_editing_layer(self, repaint: bool = True) -> None:
        """Keep the overlay's edit layer in step with the session's."""
        if self.overlay is None:
            return
        instance = getattr(self.session, "editing_instance", None)
        mask = self.session.editing_mask() if instance else None
        if instance and mask is not None and tuple(mask.shape) == self.overlay.hw:
            self.overlay.set_editing(instance, mask)
        else:
            self.overlay.clear_editing()
        if repaint:
            self.canvas.refresh()

    # --------------------------------------------------- never lose an edit
    def has_uncommitted_edit(self) -> bool:
        """Is there an editing layer whose pixels differ from what was loaded?"""
        return compat.layer_changed(self.session)

    def can_leave_edit(self) -> bool:
        """``True`` when it is safe to change frame, instance, view or mode.

        An uncommitted edit blocks *every* way out with one hint rather than a
        dialog: the annotator is mid-gesture, ``Enter`` and ``Esc`` are both one
        key away, and auto-committing something they never approved is the one
        outcome that cannot be undone by reading the screen.
        """
        if not self.has_uncommitted_edit():
            return True
        self.report(BLOCK_HINT)
        return False

    @S.guard
    def _on_canvas_press(self, x: float, y: float, ev: object) -> None:
        """Runs before the armed tool sees the press (spec 4.3).

        With no instance being edited a stroke has nowhere to go.  When the task
        card is pointing at something to draw, that is adopted and the stroke
        lands where the annotator meant it to; otherwise the press is marked and
        :meth:`on_stroke` puts the layer back exactly as it was, so the canvas
        never shows pixels that are about to be thrown away.
        """
        self._paint_blocked = False
        self._blocked_layer = None
        if self.roi_editing or self._tool_name not in (
                "brush", "eraser", "sam_point", "sam_box"):
            return
        if getattr(self.session, "editing_instance", None) is not None:
            return
        adopted = self._adoptable_instance()
        if adopted is not None:
            self.on_request_edit(adopted)
            return
        self._paint_blocked = True
        self._blocked_layer = (None if self.overlay is None
                               else self.overlay.editing.copy())
        self.report(NO_INSTANCE_HINT)

    def _adoptable_instance(self) -> Optional[str]:
        """The task-card item a stroke may adopt: an open ``add_shape``/split."""
        rows = self.session.task_card()
        index = self.task_card.current_index()
        if 0 <= index < len(rows):
            row = rows[index]
            if not row.get("done") and row.get("kind") in _DRAWABLE and row.get("instance"):
                return str(row["instance"])
        for row in rows:
            if not row.get("done") and row.get("kind") in _DRAWABLE and row.get("instance"):
                return str(row["instance"])
        return None

    # ----------------------------------------------------------------- edits
    @S.guard
    def on_request_edit(self, instance: str) -> None:
        """A panel asked for an instance to be edited (task card / instance list).

        The window owns ``begin_edit``: the panels only report the gesture, so
        that switching instance while pixels are uncommitted can be refused in
        one place instead of three.
        """
        if getattr(self.session, "editing_instance", None) == instance:
            return
        if not self.can_leave_edit():
            return
        self.cancel_roi_edit()
        try:
            self.session.begin_edit(instance)
        except self.refusal as refused:
            self.report_error(f"refused: {refused}")
            return
        self._sync_editing_layer()
        self.set_sam_instance(instance)
        self._attach_tool()
        self.arm_prompt_box_for(instance)
        self._offer_restore(self.session.current(), instance)
        self.report(f"editing {instance}")

    @S.guard
    def on_stroke(self, rect: object) -> None:
        """One finished pixel stroke: one undoable op, one debounced sidecar write."""
        tool = self.sender() or self.active_tool
        if tool is self.occluder:
            self.session.commit_occluder(self.overlay.occluder_layer(
                self.occluder.occluder_type), self.occluder.occluder_type)
            return
        instance = getattr(self.session, "editing_instance", None)
        if self._paint_blocked or instance is None or self.overlay is None:
            self._revert_blocked_stroke()
            return
        compat.push_stroke(self.session, instance,
                           getattr(tool, "stroke_before", None), self.overlay.editing)
        self.queue_sidecar(self.session.current(), instance, self.overlay.editing)
        self.update_status()

    def _revert_blocked_stroke(self) -> None:
        """Undo a stroke that had no instance to belong to."""
        if not self._paint_blocked or self.overlay is None:
            return
        self._paint_blocked = False
        layer, self._blocked_layer = self._blocked_layer, None
        if layer is not None:
            self.overlay.set_editing(self.overlay.editing_instance or "", layer)
            self.overlay.clear_editing()
            self.canvas.refresh()
        self.report(NO_INSTANCE_HINT)

    # -------------------------------------------------------- crash sidecar
    def queue_sidecar(self, key, instance: str, mask: np.ndarray) -> None:
        """Schedule a sidecar write; repeated strokes coalesce into one.

        The reviewer measured 50-105 ms for one write at 4032x3040, which is a
        visible stutter at the end of every stroke.  The mask is snapshotted
        immediately (it is the window's buffer and the next stroke changes it)
        and written once the annotator pauses.
        """
        self._sidecar_pending = (key, str(instance), np.array(mask, dtype=bool, copy=True))
        self._sidecar_timer.start()

    @S.guard
    def flush_sidecar(self) -> None:
        """Write the pending editing layer now (the debounce timer, or on close)."""
        self._sidecar_timer.stop()
        pending, self._sidecar_pending = self._sidecar_pending, None
        if pending is None:
            return
        key, instance, mask = pending
        self.sidecar.save(key, instance, mask)
        self.sidecar_writes += 1

    def drop_sidecar(self, key, instance: Optional[str]) -> None:
        """Forget a layer that has been committed or abandoned."""
        self._sidecar_timer.stop()
        self._sidecar_pending = None
        if instance is not None:
            self.sidecar.clear(key, instance)

    def set_editing_mask(self, mask: np.ndarray, undoable: bool = False) -> None:
        """Replace the editing layer everywhere it is held at once."""
        instance = getattr(self.session, "editing_instance", None)
        before = None if self.overlay is None else self.overlay.editing.copy()
        mask = np.asarray(mask, dtype=bool)
        if undoable and instance is not None:
            compat.push_stroke(self.session, instance, before, mask)
            self.queue_sidecar(self.session.current(), instance, mask)
        else:
            self.session.set_editing_mask(mask)
        if self.overlay is not None and instance is not None:
            self.overlay.set_editing(instance, mask)
            self.canvas.refresh()

    @S.guard
    def act_fill_holes(self) -> None:
        """``Shift+F``: close the interior of the editing layer, one undoable op."""
        mask = self.session.editing_mask()
        if mask is None:
            self.report("nothing is being edited")
            return
        self.set_editing_mask(_masks.fill_holes(mask), undoable=True)
        self.report("filled the holes")

    @S.guard
    def act_despeckle(self) -> None:
        """``Shift+D``: drop components under :data:`DESPECKLE_MIN_PX` pixels."""
        mask = self.session.editing_mask()
        if mask is None:
            self.report("nothing is being edited")
            return
        self.set_editing_mask(
            _masks.remove_small_components(mask, DESPECKLE_MIN_PX), undoable=True
        )
        self.report(f"removed components under {DESPECKLE_MIN_PX} px")

    # --------------------------------------------------------------- commits
    @S.guard
    def act_commit(self) -> None:
        """``Enter``: accept the pending scope, commit the edit, or store the ROI."""
        if self._pending_scope is not None:
            self._commit(self._pending_scope)
            return
        if getattr(self.session, "editing_instance", None) is not None \
                and self.session.editing_mask() is not None:
            scope = self.session.suggest_scope()
            if scope == api.SCOPE_KEYFRAME:
                self._commit(scope)
            else:
                self._offer_scope(scope)
            return
        if self.roi_editing:
            self.accept_roi()

    @S.guard
    def act_commit_override(self) -> None:
        """``Alt+Enter``: this frame only."""
        self._commit(api.SCOPE_FRAME_OVERRIDE)

    @S.guard
    def act_commit_split(self) -> None:
        """``Ctrl+K``: a new shape version from this step on."""
        self._commit(api.SCOPE_SPLIT)

    def scope_bar_text(self) -> str:
        """What the non-modal scope bar is currently saying."""
        return self.scope_bar.label.text()

    def _offer_scope(self, scope: str) -> None:
        """Show the suggestion and its alternatives without blocking the canvas."""
        self._pending_scope = scope
        # A zorder statement changes a pairwise constraint, not a run of frames,
        # and the session cannot say how far it reaches: leave the count out
        # rather than print a number that means something else.
        counts = None if scope.startswith("zorder:") else compat.preview(self.session, scope)
        detail = ""
        if counts:
            detail = (f"，影响 {len(counts.get('steps', []))} 帧 / 将产生 "
                      f"{len(counts.get('verified_steps', []))} 个冲突")
        self.scope_bar.show_text(f"建议范围 {scope}{detail}")
        self.report(f"suggested scope: {scope}")

    def _commit(self, scope: str) -> None:
        """Write the edit, or show why the session will not take it.

        A refusal -- "this part is on the bench, use the bench box" -- is an
        ordinary answer, not a failure: it keeps the editing layer so the
        annotator can press ``R`` and draw the box instead, and it does not go
        through the exception path, which would log a traceback for something
        the annotator simply has to read.
        """
        instance = getattr(self.session, "editing_instance", None)
        if instance is None:
            self.report("nothing is being edited")
            return
        key = self.session.current()
        try:
            result = self.session.commit_edit(scope) or {}
        except ValueError as refused:
            self._pending_scope = None
            self.scope_bar.hide()
            self.report_error(f"refused: {refused}")
            return
        self._pending_scope = None
        self.scope_bar.hide()
        self.session.clear_edit()
        self.drop_sidecar(key, instance)
        self.set_sam_instance(None)
        self._sync_editing_layer()
        self.refresh_overlay()
        self.re_explain()          # the part is drawn now: its blob is explained
        self.report(f"committed ({scope}): {result.get('changed', '')}".strip())

    @S.guard
    def act_clear_edit(self) -> None:
        """``Esc``: drop the editing layer, or abandon the ROI rectangle."""
        instance = getattr(self.session, "editing_instance", None)
        if instance is not None:
            key = self.session.current()
            self.session.clear_edit()
            self.drop_sidecar(key, instance)
            self._restore_offer = None
            self.restore_bar.hide()
            self._pending_scope = None
            self.scope_bar.hide()
            self.set_sam_instance(None)
            self._sync_editing_layer()
            self.report("edit discarded")
            return
        if self.roi_editing:
            self.cancel_roi_edit()
            self.report("ROI unchanged")

    @S.guard
    def act_confirm(self) -> bool:
        """``Space``: verify the frame; on refusal the problems are shown, not a dialog.

        The unexplained differences are re-derived *here*, against what the frame
        holds now, and only what is still unexplained goes to the review queue.
        If the comparison has not come back yet it is finished synchronously
        under a short cap; if even that fails the step is recorded as "not
        analysed" rather than as "nothing unexplained", which would quietly
        claim the frame had been checked.
        """
        step = self.session.current().step
        blobs = self.unexplained_at_confirm()
        ok = self.task_card.confirm()
        if ok:
            self.hand_over_unexplained(step, blobs)
            self.report(f"step {step} confirmed")
        else:
            problems = self.task_card.problems()
            self.report(f"step {step} is not complete: {'; '.join(problems) or 'see the task card'}")
        return bool(ok)

    # ------------------------------------------------------------------ undo
    @S.guard
    def act_undo(self) -> None:
        self._history(self.session.undo(), "undo")

    @S.guard
    def act_redo(self) -> None:
        self._history(self.session.redo(), "redo")

    def _history(self, moved: bool, what: str) -> None:
        if not moved:
            self.report(f"nothing to {what}")
            return
        self.refresh_overlay()
        self._sync_editing_layer()
        self.update_status()
        self.report(what)

    @S.guard
    def _on_editing_changed(self, mask: object) -> None:
        """The session moved the editing layer behind our back (undo/redo)."""
        self._sync_editing_layer()

    # ------------------------------------------------------------ visibility
    @S.guard
    def act_set_visibility(self, value: str) -> None:
        self.instances.set_visibility(value)
        self.refresh_overlay()

    @S.guard
    def act_cycle_visibility(self) -> None:
        self.instances.cycle_visibility()
        self.refresh_overlay()

    @S.guard
    def act_toggle_hidden(self) -> None:
        self.instances.toggle_hidden()
        self.refresh_overlay()

    # ------------------------------------------------------------------- ROI
    def roi(self) -> Optional[tuple]:
        """The stored ROI of the open frame's pose segment, or ``None``."""
        if not compat.is_open(self.session):
            return None
        row = self.db.pose_segment_for(self.session.current()) or {}
        stored = row.get("roi")
        if not stored:
            return None
        return tuple(int(v) for v in stored)

    @S.guard
    def act_edit_roi(self) -> None:
        """``Shift+R``: draw the chassis rectangle again."""
        self.start_roi_edit()

    def start_roi_edit(self) -> None:
        """Propose a rectangle and let the annotator drag it (``Enter`` accepts)."""
        image = self.session.image()
        if image is None:
            return
        stored = self.roi()
        self.roi_draft = tuple(int(v) for v in (
            stored if stored is not None else suggest_roi(image, self.session.view)
        ))
        self.roi_editing = True
        # Arm the tool first: detaching a SAM tool clears the rubber band, so
        # painting the draft before the swap would erase it again.
        self._attach_tool()
        self.canvas.set_rubber_band(self.roi_draft)
        self.report("拖动框选机箱范围，Enter 确认 / drag the chassis box, Enter to accept")

    @S.guard
    def on_roi_box(self, box: object) -> None:
        """The ROI tool finished a drag; the rectangle is checked, not trusted."""
        from tda.core.db_pose import clean_roi

        hw = None if self.overlay is None else self.overlay.hw
        try:
            self.roi_draft = tuple(clean_roi(list(box), hw))  # type: ignore[arg-type]
        except ValueError as refused:
            self.report(f"that rectangle is not usable: {refused}")
            self.canvas.set_rubber_band(self.roi_draft)
            return
        self.canvas.set_rubber_band(self.roi_draft)

    @S.guard
    def accept_roi(self) -> None:
        """Store the rectangle on the pose segment and zoom to it."""
        key = self.session.current()
        segment = self._pose_segment(key)
        if self.roi_draft is None or segment is None:
            self.cancel_roi_edit()
            return
        hw = None if self.overlay is None else self.overlay.hw
        accepted = tuple(self.db.set_pose_segment_roi(
            int(key.desktop), str(key.view), int(segment), list(self.roi_draft),
            annotator=self.annotator, hw=hw,
        ))
        self.roi_draft = accepted
        self.roi_editing = False
        self.canvas.set_rubber_band(None)
        self._attach_tool()
        self.canvas.zoom_to(accepted)
        self.update_status()
        self.request_assist()
        self.report(f"ROI stored: {accepted}")

    def cancel_roi_edit(self) -> None:
        """Leave ROI editing without storing anything."""
        if not self.roi_editing:
            return
        self.roi_editing = False
        self.roi_draft = self.roi()
        self.canvas.set_rubber_band(None)
        self._attach_tool()

    # ------------------------------------------------------------ bench box
    @S.guard
    def on_bench_box(self, box: object) -> None:
        """``R`` + a drag: the staging-area rectangle of a part on the bench."""
        instance = (self.instances.selected_instance()
                    or self.task_card.current_instance())
        if not instance:
            self.report("select the instance the bench box belongs to first")
            return
        self.canvas.set_rubber_band(None)
        try:
            self.session.commit_box(instance, [float(v) for v in box])  # type: ignore[misc]
        except ValueError as refused:
            self.report_error(f"refused: {refused}")
            return
        self.refresh_overlay()
        self.report(f"bench box stored for {instance}")

    def act_move_instance(self, direction: int) -> None:
        """``Ctrl+Up`` / ``Ctrl+Down`` from the window's keyboard."""
        self.move_instance(direction)

    @S.guard
    def move_instance(self, direction: int) -> None:
        """``Ctrl+Up``/``Ctrl+Down``: one layer up or down, refusals included.

        The panel's own slots are wrapped rather than only its buttons, because
        its key handler calls them straight and a refused reorder would then
        escape as an exception from a Qt slot.
        """
        try:
            self._panel_move(direction)
        except ValueError as refused:
            self.report_error(f"refused: {refused}")
            return
        self.refresh_overlay()

    # ---------------------------------------------------------------- review
    @S.guard
    def resolve_selected(self, resolution: str) -> None:
        """Resolve the conflict the review panel has selected."""
        item = self.review.list_for(api.QUEUE_CONFLICTS).currentItem()
        cid = None if item is None else item.data(int(Qt.ItemDataRole.UserRole) + 1)
        if cid is None:
            self.report("select a conflict first")
            return
        self.resolve_conflict(int(cid), resolution)

    @S.guard
    def resolve_conflict(self, cid: int, resolution: str) -> None:
        """One conflict, with the three verdicts told apart (spec 4.4)."""
        verdict, text = compat.resolve_conflict(self.session, cid, resolution)
        self.review.refresh()
        self.review_refreshes += 1
        if verdict == compat.SUPERSEDED:
            self.report(f"conflict {cid}: superseded and re-queued (已被覆盖，重新入队)"
                        f"{' — ' + text if text else ''}")
        elif verdict == compat.REFUSED:
            self.report(f"conflict {cid} refused: {text}")
        else:
            self.report(f"conflict {cid}: {resolution}")

    @S.guard
    def on_rework(self, step: int) -> None:
        """``R`` in review mode: send the frame back to the annotator."""
        self.session.goto(int(step))
        self.set_mode("annotate")
        self.report(f"step {step} reopened for rework")

    # ---------------------------------------------------------- crash safety
    def pending_restore(self) -> Optional[dict]:
        """The uncommitted layer a previous run left on this frame, or ``None``."""
        return self._restore_offer

    def _offer_restore(self, key, instance: Optional[str] = None) -> None:
        """Offer a recovered layer for this frame (or for one instance of it).

        Called on every frame change *and* on every ``begin_edit``: an edit
        abandoned on an instance the annotator comes back to later is exactly
        the case a frame-change-only offer never caught.  A sidecar naming an
        instance the frame no longer has is deleted rather than offered.
        """
        hw = None if self.overlay is None else self.overlay.hw
        if instance is not None:
            candidates = [c for c in [self.sidecar.pending_for(key, instance, hw)]
                          if c is not None]
        else:
            candidates = self.sidecar.entries_for(key, hw)
        found = None
        for entry in candidates:
            # A sidecar for an instance this frame no longer has is not an offer
            # anybody can accept: drop it rather than keep asking about it.
            if not self._instance_exists(entry["instance"]):
                self.sidecar.drop(entry)
                continue
            if found is None:
                found = entry
        if found is None or (instance is None
                             and getattr(self.session, "editing_instance", None) is not None):
            self._restore_offer = None
            self.restore_bar.hide()
            return
        self._restore_offer = {"instance": found["instance"], "mask": found["mask"],
                               "key": found["key"]}
        self.restore_bar.show_text(
            f"上次未提交的编辑（{found['instance']}）可以恢复 / "
            f"an uncommitted edit of {found['instance']} was found"
        )

    def _instance_exists(self, instance: str) -> bool:
        """Is this instance part of the frame at all (compiled or on the card)?"""
        if instance in self.session.compiled().instances:
            return True
        return any(row.get("instance") == instance for row in self.session.task_card())

    @S.guard
    def restore_pending(self) -> None:
        """Put the recovered layer back into the editing layer."""
        offer, self._restore_offer = self._restore_offer, None
        self.restore_bar.hide()
        if offer is None:
            return
        try:
            self.session.begin_edit(offer["instance"])
        except self.refusal as refused:
            self.report_error(f"refused: {refused}")
            return
        self.set_sam_instance(offer["instance"])
        self.set_editing_mask(offer["mask"])
        self._attach_tool()
        self.report(f"restored the uncommitted edit of {offer['instance']}")

    @S.guard
    def discard_pending(self) -> None:
        """Throw the recovered layer away."""
        offer, self._restore_offer = self._restore_offer, None
        self.restore_bar.hide()
        if offer is not None:
            self.sidecar.clear(offer["key"], offer["instance"])
        self.report("the recovered edit was discarded")
