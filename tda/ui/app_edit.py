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
* **A commit.**  ``Enter`` asks the session what scope the edit means; the scope
  bar and everything that writes are next door in :mod:`tda.ui.app_commit`.
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
from tda.ui import app_priors
from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui.app_widgets import Bar, BoxDragTool, RoiBoxTool
from tda.ui.canvas.tools import BrushTool, EraserTool, OccluderTool

__all__ = ["BLOCK_HINT", "BoxDragTool", "EditMixin", "DESPECKLE_MIN_PX",
           "FLASH_HINT", "NO_INSTANCE_HINT", "REVIEW_READ_ONLY"]

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
#: How long the flash hint stays put against an ordinary status line.
FLASH_HINT_HOLD_MS = 2000
#: Shown when the canvas is clicked while another frame is flashed over it.
FLASH_HINT = ("松开 Tab 再操作：屏幕上是对照帧  "
              "(release Tab first: the canvas is showing the other frame)")
#: Shown, and kept on screen, when the crash sidecar cannot be written.
SIDECAR_BROKEN = ("崩溃保护已失效：{why} —— 请尽快提交，崩溃会丢失当前图层 "
                  "(the crash sidecar cannot be written)")


def _px(mask) -> int:
    """How many pixels a boolean layer holds; 0.26 ms at 12 MP, against 4.9."""
    return 0 if mask is None else int(np.count_nonzero(mask))


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
        self.roi_tool = RoiBoxTool(self.canvas, None)
        self.bench_tool = BoxDragTool(self.canvas, None)
        for tool in (self.brush, self.eraser, self.occluder):
            tool.sigStroke.connect(self.on_stroke)
        self.roi_tool.sigBox.connect(self.on_roi_box)
        self.roi_tool.sigPreview.connect(self.on_roi_preview)
        self.bench_tool.sigBox.connect(self.on_bench_box)
        # Connected before any tool is attached, so the window sees a press
        # first and can adopt an instance for it (or mark it as doomed).
        self.canvas.sigMousePress.connect(self._on_canvas_press)
        # The wheel zooms without going through any action, so the percentage
        # in the status bar has to follow the canvas rather than the keyboard.
        self.canvas.sigZoomChanged.connect(self._on_zoom_changed)
        self._paint_blocked = False
        self._blocked_layer: Optional[np.ndarray] = None

        self.refusal = compat.session_refusal()
        self.sidecar_writes = 0
        self._sidecar_pending: Optional[tuple] = None
        #: Set once the sidecar cannot be written, and kept: it is the crash
        #: protection, so it outranks every other hint until the run ends.
        self._sidecar_broken = ""
        #: ``(desktop, view, step, instance)`` this window has written a crash
        #: copy for.  A net-zero gesture may delete its own file; one left by
        #: an earlier run belongs to whoever answers the restore offer.
        self._sidecar_written: set[tuple] = set()
        self._sidecar_timer = QTimer(self)
        self._sidecar_timer.setSingleShot(True)
        self._sidecar_timer.setInterval(SIDECAR_DEBOUNCE_MS)
        self._sidecar_timer.timeout.connect(self.flush_sidecar)

        self.roi_editing = False
        self.roi_draft: Optional[tuple] = None
        #: Segments whose ROI proposal the annotator has **dismissed** -- keyed
        #: by ``(desktop, view, seg, start_step, end_step)``.  Asking again on
        #: every frame of a segment would steal the tool, and with it the
        #: pending SAM prompt, from somebody who declined once.
        #:
        #: The step range is in the key on purpose: this used to record every
        #: segment it *asked* about, so a pose re-cut that left the earlier
        #: piece without an ROI -- which is exactly what a large camera move is
        #: supposed to do (task B1) -- was never asked about, because that piece
        #: usually keeps segment number 1 and 1 had "already been asked".  Those
        #: frames then diffed over the whole image and prompted SAM with an
        #: unbounded box, silently.  A re-cut always moves a range, so the key
        #: cannot survive one.
        self._roi_dismissed: set[tuple] = set()
        #: The segment whose ROI question nobody has answered yet, or ``None``.
        #: Dismissing the *rectangle* (``Esc``, picking a tool, starting an
        #: edit) stops it popping up on every frame -- that is what
        #: :attr:`_roi_dismissed` is for -- but it does not answer the
        #: question, and the annotator's first trial ended with the rectangle
        #: gone, no ROI stored, and nothing on screen saying so. While this is
        #: set the ROI bar keeps a one-line reminder up (ruling U-ROI-3).
        self._roi_pending: Optional[tuple] = None
        #: The last rectangle offered or dragged for :attr:`_roi_pending`, kept
        #: across a dismissal so that "确认建议框" has something to store.
        self._roi_proposal: Optional[tuple] = None
        self._pending_scope: Optional[str] = None
        self._restore_offer: Optional[dict] = None
        #: The instance an ``add_bench_box`` card item armed the box tool for.
        self.bench_instance: Optional[str] = None

        #: The area warning waiting for a second ``Enter``:
        #: ``(scope, text, facts)``, the facts going into the commit's op log.
        self._pending_warning: Optional[tuple] = None
        #: Those facts, once the second ``Enter`` has answered the warning, for
        #: exactly the one commit that follows.
        self._override_facts: Optional[dict] = None
        self.priors = app_priors.load_priors()
        self.warn_bar = Bar(self)
        self.warn_bar.add_button("Enter 仍然提交", self.act_commit)
        self.warn_bar.add_button("Esc 回去改", self.act_clear_edit)

        self.scope_bar = Bar(self)
        self.scope_bar.add_button("Enter 接受", self.act_commit)
        self.scope_bar.add_button("Alt+Enter 仅本帧", self.act_commit_override)
        self.scope_bar.add_button("Ctrl+K 从此拆分", self.act_commit_split)
        self.restore_bar = Bar(self)
        self.restore_bar.add_button("恢复 Restore", self.restore_pending)
        self.restore_bar.add_button("丢弃 Discard", self.discard_pending)
        # The ROI bar carries both halves of the question -- the rectangle
        # while it is up, and the compact reminder afterwards -- because they
        # are one conversation and two bars would be two things to dismiss.
        # Non-modal like its neighbours: ``Enter`` and ``Esc`` belong to the
        # rectangle itself through the layered ownership chain in
        # ``act_commit``/``act_clear_edit``, never to a button's focus.
        self.roi_bar = Bar(self)
        self._roi_buttons = {
            "save": self.roi_bar.add_button("保存 Enter", self.act_commit),
            "skip": self.roi_bar.add_button("跳过 Esc", self.act_clear_edit),
            "accept": self.roi_bar.add_button("确认建议框 Use it",
                                              self.act_accept_roi_proposal),
            "redraw": self.roi_bar.add_button("重画 Shift+R", self.act_edit_roi),
            "none": self.roi_bar.add_button("不用 ROI No ROI", self.act_no_roi),
        }
        for button in self._roi_buttons.values():
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._central_layout.addWidget(self.warn_bar)
        self._central_layout.addWidget(self.scope_bar)
        self._central_layout.addWidget(self.restore_bar)
        self._central_layout.addWidget(self.roi_bar)

    @S.guard
    def _on_zoom_changed(self, _factor: float) -> None:
        """The canvas zoomed by any route; the percentage follows it.

        Guarded like every other slot the window connects: an exception here
        would go straight into the Qt event loop.
        """
        self.update_status()

    def _rewire_panels(self) -> None:
        """Every panel gesture the window has to be able to refuse, in one place.

        The panels report; the window acts.  Each of these has an answer only
        the window can give: a conflict resolution has three verdicts, opening
        another frame may not leave an uncommitted edit behind, and a refused
        reorder must not escape a Qt slot as an exception.  Nothing is
        *intercepted* any more -- every one of them is a signal the panel emits,
        so a re-wire that is forgotten loses a gesture instead of letting one
        through.
        """
        self.review.sigResolve.connect(self.resolve_selected)
        self.review.sigOpenStep.connect(self.timeline_goto)
        self.timeline.sigOpenStep.connect(self.timeline_goto)
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

    # ------------------------------------------------------------ frame hook
    def on_frame_changed_edit(self, key) -> None:
        """What the editing half has to do when the frame changes.

        The ROI question is asked from the *stored* state, never from a memory
        of having asked: a segment with no rectangle gets the proposal unless
        this annotator has dismissed it for that segment (see
        :attr:`_roi_dismissed`).
        """
        self._pending_scope = None
        self.scope_bar.hide()
        self.disarm_bench()
        # A draft ghost is about one instance on one frame.  ``_sync_editing_layer``
        # covers the frames that repaint; this covers the ones that do not,
        # including a view with no image at this step.
        self.forget_draft_ghost()
        if self.session.image() is not None:
            if self.roi() is not None:
                if self.roi_editing:
                    self.cancel_roi_edit()
            elif self.roi_key() not in self._roi_dismissed and not self.roi_editing:
                self.start_roi_edit()
        self.on_frame_changed_roi()
        self._offer_restore(key)

    def roi_key(self) -> Optional[tuple]:
        """What a dismissed ROI proposal is remembered by, or ``None``.

        ``(desktop, view, seg, start_step, end_step)``: the range is what makes
        the memory self-invalidating.  A pose re-cut only ever writes when a
        range or a number moves (:meth:`tda.core.db.Db.apply_recut`), so a
        segment that has just lost its ROI to one cannot be mistaken for the
        segment somebody declined ten minutes ago.
        """
        if not compat.is_open(self.session):
            return None
        key = self.session.current()
        row = self.db.pose_segment_for(key) or {}
        return (int(key.desktop), str(key.view), row.get("seg"),
                row.get("start_step"), row.get("end_step"))

    def reset_roi_proposals(self) -> None:
        """Forget every dismissal: the segments themselves have just changed.

        Called by whatever re-opens the session on the same frame after a
        structural edit -- a pose re-cut, an S1 Apply -- so the annotator is
        asked about the pieces that came out of it.

        A measurement in flight is abandoned with them. It was taken for a
        segment whose range has just moved, so it is an answer about something
        that no longer exists; the five-tuple in
        :meth:`~tda.ui.app_roi.RoiMixin._roi_segment_key` would reject it
        anyway, and cancelling says so at the moment it becomes true.
        """
        self._roi_dismissed.clear()
        self._roi_awaiting = False
        self._roi_pending = None
        self._roi_proposal = None
        self.roi_proposer.cancel()

    def _sync_editing_layer(self, repaint: bool = True) -> None:
        """Keep the overlay's edit layer in step with the session's.

        This is the **one** place the layer is replaced from outside the SAM
        tool -- a commit, ``Esc``, an undo, a restored sidecar all end here --
        so it is where the half-built prompt is dropped.  Keeping the points
        made the next click refine a layer their result no longer had anything
        to do with.  A draft ghost, and the note saying the layer came from a
        draft, describe the same vanished layer and go with them.

        The **area warning** goes too, and this is the place that catches the
        one an undo or a redo would otherwise leave standing: it is an offer
        about a specific mask ("press Enter again and I will write these 2,116
        pixels"), and ``Ctrl+Z`` back to a hundred of them made the next Enter
        commit a hundred pixels while logging an override of two thousand.
        """
        self.reset_sam_prompt()
        self.forget_draft_ghost()
        self._invalidate_area_warning()
        if self.overlay is None:
            return
        instance = getattr(self.session, "editing_instance", None)
        mask = self.session.editing_mask() if instance else None
        was_px = _px(self.overlay.editing)
        if instance and mask is not None and tuple(mask.shape) == self.overlay.hw:
            # Somebody outside the tools replaced the layer -- a commit, Esc, an
            # undo, a restored sidecar.  Whatever it was, the annotator has to
            # be able to find it in the log next to the strokes it replaced.
            self.note_layer_size("layer replaced", instance, was_px, _px(mask))
            self.overlay.set_editing(instance, mask)
        else:
            if was_px:
                self.note_layer_size("layer cleared", instance, was_px, 0)
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
        if self._sidecar_broken:
            # The worse of the two messages wins: "press Enter" is advice,
            # "your work is no longer being protected" is news.
            self.report(self._sidecar_broken)
            return False
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
        # Whatever the move was -- frame, view, desktop, session -- the bench arm
        # was about the frame that has just been left.  ``on_frame_changed_edit``
        # covers the moves that repaint; this covers the ones that do not,
        # including a view with no image at this step.
        self.disarm_bench()
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
        # A click on the canvas is the annotator saying "I am working here":
        # the next letter key has to be a shortcut, not something a panel that
        # still holds the focus swallows (ruling R1).
        self.canvas.setFocus(Qt.FocusReason.MouseFocusReason)
        if self.is_flashing():
            # The tools are detached while flashing, so nothing is going to
            # paint; this is only here to say why the click did nothing.  Held,
            # because the difference map answers a few milliseconds later and
            # used to take it straight off the screen.
            self.report(FLASH_HINT, hold_ms=FLASH_HINT_HOLD_MS)
            return
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

    def disarm_bench(self) -> None:
        """Forget which instance the box tool was armed for.

        The arm is a statement about *this* frame -- "the rectangle you draw
        next belongs to that part, here" -- and it used to be cleared only by a
        successful drag.  It therefore survived frame, view, desktop and mode
        changes, every tool switch and every ``begin_edit``, and the next
        rectangle drawn anywhere was filed under a part the annotator had armed
        minutes and several frames ago.  Every one of those gestures calls this.
        """
        self.bench_instance = None

    @S.guard
    def begin_bench_box(self, instance: str) -> None:
        """Arm the box tool for a part on the bench: no mask layer at all.

        A bench part's geometry is a rectangle, so beginning a mask edit for it
        only ends in a refusal at commit time -- with the annotator's strokes
        already on screen.  The tool is armed instead and the drag writes
        ``commit_box`` directly.
        """
        self.cancel_roi_edit()
        # This branch of ``on_request_edit`` never reaches _sync_editing_layer,
        # so a ghost offered for the previous instance would still be on screen.
        self.forget_draft_ghost()
        self.bench_instance = str(instance)
        self.note_tool_switch("bench_box", via="task card")
        self._tool_name = "bench_box"
        self.set_sam_instance(None)
        self._attach_tool()
        self.focus_canvas()
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
            self.task_card.select_instance(
                getattr(self.session, "editing_instance", None) or "")
            return
        self.disarm_bench()
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
        # The panel was activated with a double-click, so it has the keyboard;
        # give it straight back, or the next ``B`` is a list keystroke.
        self.focus_canvas()
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
        self._invalidate_area_warning()
        before = getattr(tool, "stroke_before", None)
        self.note_layer_change("stroke", instance, before, self.overlay.editing)
        compat.push_stroke(self.session, instance, before, self.overlay.editing)
        self.queue_sidecar(self.session.current(), instance, self.overlay.editing)
        self.update_status()

    def note_layer_change(self, why: str, instance: Optional[str],
                          before, after) -> None:
        """One INFO line per change of the editing layer (ruling R3).

        The trial's log held four SAM prompts and nothing else, so there was no
        way to tell a brush stroke from a box drag afterwards -- which is
        exactly the question the annotator's report turned on.  One line, never
        a pixel dump: the tool, the instance, what went in and out, and the
        layer's size before and after.

        Counted with :func:`numpy.count_nonzero` and **one** temporary rather
        than the obvious ``(after & ~before).sum()``: at 12 MP that spelling is
        10.6 ms against 3, and this runs inside B7's window-gesture budget.
        """
        after_px = _px(after)
        if before is None:
            added, removed, was = after_px, 0, 0
        else:
            both = _px(np.asarray(before, dtype=bool)
                       & np.asarray(after, dtype=bool))
            was = _px(before)
            added, removed = after_px - both, was - both
        self.logger.info("%s tool=%s instance=%s +%d/-%d px layer %d -> %d",
                         why, self._tool_name, instance, added, removed,
                         was, after_px)

    def note_layer_size(self, why: str, instance: Optional[str],
                        was_px: int, now_px: int) -> None:
        """The same line for a replacement, where the diff is not worth 12 MP."""
        self.logger.info("%s tool=%s instance=%s layer %d -> %d px",
                         why, self._tool_name, instance, int(was_px), int(now_px))

    def _invalidate_area_warning(self) -> None:
        """Take back a size warning whose mask has just changed underneath it.

        The warning is an offer -- "press Enter again and I will write this
        1.5 Mpx screw" -- and the second Enter logs *the facts of the mask it
        was raised on*.  Painting in between made those facts describe a mask
        nobody was warned about, and the override rode on a commit the
        annotator never confirmed.  So any change of the layer ends the
        conversation and the next Enter asks again.
        """
        if self._pending_warning is None:
            return
        self._pending_warning = None
        self.warn_bar.hide()

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
        if not self.has_uncommitted_edit():
            # A layer back at what ``begin_edit`` loaded -- an undo inside the
            # debounce window, a net-zero brush-then-erase, a stroke inside a
            # shape that is already committed -- is not work to protect.  It is
            # not enough to skip the write: the *pending* one has to be dropped
            # and the file **this window wrote for this edit** deleted, or the
            # annotator is offered a "restore" of pixels they took back
            # (measured: 197 px).
            #
            # Only that file.  A sidecar left by an earlier run is somebody
            # else's unfinished work: it goes away when the annotator answers
            # the restore offer or commits the instance, never because their
            # first gesture on the frame happened to cancel itself out.
            self.drop_sidecar(key, instance)
            return
        self._sidecar_pending = (key, str(instance), np.array(mask, dtype=bool, copy=True))
        self._sidecar_timer.start()

    @S.guard
    def flush_sidecar(self) -> None:
        """Write the pending editing layer now (the debounce timer, or on close).

        A failure here is the **crash protection** failing, which is worse news
        than anything else on the status bar: it is said once, in a plain
        sentence, and it stays there -- ``BLOCK_HINT`` used to overwrite it on
        the very next gesture, so the annotator was told to press Enter and
        never told that a crash would now cost them the layer.
        """
        self._sidecar_timer.stop()
        pending, self._sidecar_pending = self._sidecar_pending, None
        if pending is None:
            return
        key, instance, mask = pending
        try:
            # The adopted drafts go with the pixels: a crash takes the undo
            # history, which is where the provenance otherwise lives.
            self.sidecar.save(key, instance, mask,
                              adopted=self.pending_adoptions(key, instance))
        except Exception as exc:  # noqa: BLE001 - reported, never raised at a stroke
            self._sidecar_broken = SIDECAR_BROKEN.format(why=exc)
            self.logger.error("sidecar write failed: %s", exc)
            self.report_error(self._sidecar_broken)
            return
        self.sidecar_writes += 1
        self._sidecar_written.add(self._sidecar_id(key, instance))

    def _sidecar_id(self, key, instance: Optional[str]) -> tuple:
        """What identifies one crash copy: the frame and the instance."""
        return (int(key.desktop), str(key.view), int(key.step), str(instance))

    def _wrote_sidecar_for(self, key, instance: Optional[str]) -> bool:
        """Did **this** window write the stored copy of that frame's instance?"""
        return self._sidecar_id(key, instance) in self._sidecar_written

    def drop_sidecar(self, key, instance: Optional[str], *,
                     foreign_ok: bool = False) -> None:
        """Forget a layer that has been committed or abandoned.

        The pending write and the debounce always go.  The **file** only goes
        when this window wrote it, unless ``foreign_ok`` says the caller is one
        of the three gestures that may speak for a copy left by an earlier run:

        * ``Restore`` -- the annotator took it back into the layer;
        * ``Discard`` -- they threw it away;
        * committing that instance -- what it held is now in the database.

        ``Esc`` is not one of them, which is what this argument exists for: it
        discards *the layer on screen*, and a crash copy that is still being
        offered belongs to a question nobody has answered yet.  Deleting it
        there took the previous session's work away with no answer at all.
        """
        self._sidecar_timer.stop()
        self._sidecar_pending = None
        if instance is None:
            return
        if foreign_ok or self._wrote_sidecar_for(key, instance):
            self.sidecar.clear(key, instance)
            self._sidecar_written.discard(self._sidecar_id(key, instance))

    def set_editing_mask(self, mask: np.ndarray, undoable: bool = False,
                         adopted: Optional[dict] = None) -> None:
        """Replace the editing layer everywhere it is held at once.

        ``adopted`` marks the change as an adopted Label Studio draft, which
        travels on the undo entry and into the crash sidecar.
        """
        instance = getattr(self.session, "editing_instance", None)
        before = None if self.overlay is None else self.overlay.editing.copy()
        mask = np.asarray(mask, dtype=bool)
        # The pixels the area warning was computed on are gone, so the answer
        # to it is gone with them (see ``_invalidate_area_warning``).
        self._invalidate_area_warning()
        if undoable and instance is not None:
            compat.push_stroke(self.session, instance, before, mask, adopted)
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

