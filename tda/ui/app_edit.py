"""The editing half of the main window: strokes, edit scopes, never losing work.

Mixed into :class:`tda.ui.app.MainWindow` next to
:class:`tda.ui.app_roi.RoiMixin`.  It is the part of the wiring that touches a
pixel edit, so every path here follows the same two rules: the session decides,
and no exception reaches the Qt event loop.

Three flows are worth reading as a whole:

* **A stroke.**  A tool paints into the overlay's edit layer and reports the
  dirty rect; the window turns the tool's ``stroke_before`` snapshot plus the
  layer into one undoable op on the session and queues the layer for the crash
  sidecar.  That layer is the only thing a crash can lose.
* **A commit.**  ``Enter`` asks the session what scope the edit means.  A plain
  ``keyframe`` is written immediately; anything else (a layering statement, a
  split) opens a small non-modal bar showing the suggestion and its two
  alternatives, because a modal dialog every few seconds is unusable at six
  hours a day.
* **Leaving.**  An editing layer with uncommitted pixels blocks every way out of
  the frame, the instance and the mode, with one hint.  A stroke that lands with
  no instance to hold it adopts the task card's open item or is reverted.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt

from tda.core import masks as _masks
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui.app_widgets import MIN_BOX_PX, Bar, BoxDragTool
from tda.ui.canvas.tools import BrushTool, EraserTool, OccluderTool

#: Where the timeline and the review panel keep a row's step number.
_STEP_ROLE = int(Qt.ItemDataRole.UserRole)

__all__ = ["BLOCK_HINT", "BoxDragTool", "EditMixin", "DESPECKLE_MIN_PX",
           "NO_INSTANCE_HINT", "REVIEW_READ_ONLY"]

#: Components smaller than this are specks (``Shift+D``); spec 9.1's floor.
DESPECKLE_MIN_PX = 16
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
#: A part here is boxed with ``R``, not painted.
_ON_BENCH = "on_bench"
#: Shown when the canvas is clicked in Review mode, which is read-only.
REVIEW_READ_ONLY = ("按 R 返工：切到标注模式处理这一帧  "
                    "(press R to rework: Review mode only shows the frame)")


def _is_right(ev: Any) -> bool:
    """Was this press the right button?  Stubs may not answer at all."""
    button = getattr(ev, "button", None)
    try:
        return button is not None and button() == Qt.MouseButton.RightButton
    except TypeError:  # pragma: no cover - a stub without a callable button
        return False


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
        #: The instance an ``add_bench_box`` card item armed the box tool for.
        self.bench_instance: Optional[str] = None

        self.scope_bar = Bar(self)
        self.scope_bar.add_button("Enter 接受", self.act_commit)
        self.scope_bar.add_button("Alt+Enter 仅本帧", self.act_commit_override)
        self.scope_bar.add_button("Ctrl+K 从此拆分", self.act_commit_split)
        self.restore_bar = Bar(self)
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
        self.instances.sigReorder.connect(self.move_instance)
        self.instances.sigHiddenToggled.connect(self.on_hidden_toggled)
        self.task_card.sigCommit.connect(self.on_panel_commit)
        self.task_card.sigConfirm.connect(self.act_confirm)
        self._panel_move = lambda d: (self.instances.move_up() if d < 0
                                      else self.instances.move_down())

    @S.guard
    def on_panel_commit(self, scope: str) -> None:
        """A commit button was pressed; run the action its key runs.

        ``""`` is the plain "Commit" button, which means the same as ``Enter``:
        ask the session what the edit is, and show the scope bar when the answer
        is not a plain keyframe.  Hard-coding ``keyframe`` here was one label
        with two meanings.
        """
        if scope == api.SCOPE_FRAME_OVERRIDE:
            self.act_commit_override()
        elif scope == api.SCOPE_SPLIT:
            self.act_commit_split()
        else:
            self.act_commit()

    @S.guard
    def on_hidden_toggled(self, instance: str, hidden: bool) -> None:
        """The instance list's hidden checkbox: a view setting, applied here.

        The panel used to call the session itself, which left the canvas showing
        a layer the table said was hidden until something else repainted it.
        """
        self.session.set_hidden(str(instance), bool(hidden))
        self.instances.refresh()
        self.refresh_overlay()

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

        Blocking also **flushes the sidecar debounce**: until it is written the
        layer exists only in this process, which is exactly the 300 ms window a
        crash would take it in -- and the annotator has just been told to stop.
        """
        if not self.has_uncommitted_edit():
            return True
        self.flush_sidecar()
        self.report(BLOCK_HINT)
        return False

    def leave_frame(self, move) -> bool:
        """The one gate every navigation-like gesture goes through.

        ``move`` is called only when there is nothing uncommitted to lose; it
        may move the frame, the view, the desktop, the mode or the session
        itself.  Routing all of them through one function is the point: the
        parametrised test in ``tests/test_app_safety.py`` walks this list, so a
        new way out that forgets the gate fails there rather than silently
        throwing away somebody's afternoon.
        """
        if not self.can_leave_edit():
            return False
        try:
            move()
        except self.refusal as refused:      # the session's own backstop
            self.report_error(f"refused: {refused}")
            return False
        return True

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
        if self.mode == "review":
            self.report(REVIEW_READ_ONLY)
            return
        if self._is_right_button(ev):
            return          # the right button is a negative SAM point, not an edit
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

    @staticmethod
    def _is_right_button(ev: Any) -> bool:
        return _is_right(ev)

    def _is_bench_item(self, instance: str) -> bool:
        """Does the card ask for a staging-area box rather than a mask?"""
        want = getattr(api, "KIND_ADD_BENCH_BOX", None)
        if want is None:
            return False
        return any(row.get("instance") == instance and row.get("kind") == want
                   for row in self.session.task_card())

    @S.guard
    def begin_bench_box(self, instance: str) -> None:
        """Arm the box tool for a part on the bench: no mask layer at all.

        A bench part's geometry is a rectangle, so beginning a mask edit for it
        only ends in a refusal at commit time -- with the annotator's strokes
        already on screen.  The tool is armed instead and the drag writes
        ``commit_box`` directly.
        """
        self.cancel_roi_edit()
        self.bench_instance = str(instance)
        self._tool_name = "bench_box"
        self.set_sam_instance(None)
        self._attach_tool()
        self.update_status()
        self.report(f"{instance}: 拖一个台面框（R）/ drag the staging-area box for it")

    def _adoptable_instance(self) -> Optional[str]:
        """The task-card item a stroke may adopt: an open ``add_shape``/split.

        A part that is on the bench is skipped even when the card asks for it:
        its geometry is a rectangle drawn with ``R``, so adopting it for a brush
        stroke would only end in a refusal the annotator did not ask for.
        """
        bench = {str(row.get("key")) for row in self.session.instance_rows()
                 if row.get("placement") == _ON_BENCH}
        rows = self.session.task_card()
        index = self.task_card.current_index()
        ordered = ([rows[index]] if 0 <= index < len(rows) else []) + list(rows)
        for row in ordered:
            instance = str(row.get("instance") or "")
            if (instance and not row.get("done") and row.get("kind") in _DRAWABLE
                    and instance not in bench):
                return instance
        return None

    # ----------------------------------------------------------------- edits
    @S.guard
    def on_request_edit(self, instance: str) -> None:
        """A panel asked for an instance to be edited (task card / instance list).

        The window owns ``begin_edit``: the panels only report the gesture, so
        that switching instance while pixels are uncommitted can be refused in
        one place instead of three.
        """
        if self.mode != "annotate":
            # Qt activates a list row on Enter, and the task card is still
            # mounted in Steps mode with the canvas hidden: an edit begun there
            # is one nobody can see, finish or discard.
            self.report("切到标注模式再编辑 / switch to Annotate mode to edit")
            return
        if getattr(self.session, "editing_instance", None) == instance:
            return
        if not self.can_leave_edit():
            return
        if self._is_bench_item(instance):
            self.begin_bench_box(instance)
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
    def commit_with_suggested_scope(self) -> None:
        """Commit straight away with whatever the session suggests, no bar.

        The close dialog's "Save": there is no non-modal conversation left to
        have, so a layering or split suggestion is written as suggested rather
        than parked behind a bar nobody will read.
        """
        if getattr(self.session, "editing_instance", None) is None:
            return
        self._pending_scope = None
        self._commit(self.session.suggest_scope())

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
        # A layering statement reaches frames too -- the session answers for a
        # pair override the same way it answers for pixels -- so the strip says
        # so for every scope now.
        counts = compat.preview(self.session, scope)
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
        # No re_explain() here: committing re-renders the frame, which clears
        # assist_result, so a re-split would run against nothing.  The real one
        # happens at confirm time, where the answer is actually used.
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
        if not self.can_leave_edit():
            return False      # confirming steps the frame back: same gate
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

