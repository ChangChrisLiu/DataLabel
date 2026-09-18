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
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from tda.core import masks as _masks
from tda.core.cache import suggest_roi
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui.canvas.tools import BrushTool, EraserTool, OccluderTool, Tool

__all__ = ["BoxDragTool", "EditMixin", "DESPECKLE_MIN_PX"]

#: Components smaller than this are specks (``Shift+D``); spec 9.1's floor.
DESPECKLE_MIN_PX = 16
#: A drag shorter than this on either side is a click, not a box.
MIN_BOX_PX = 2.0


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

    def _rewire_review_buttons(self) -> None:
        """Route the review panel's two buttons through the window.

        The panel calls ``session.resolve_conflict`` directly, which may raise;
        the window has to tell "superseded and re-queued" from "refused", so it
        takes the connection over rather than letting an exception out of a slot.
        """
        for button, resolution in ((self.review.keep_old_button, api.RESOLVE_KEEP_OLD),
                                   (self.review.accept_new_button,
                                    api.RESOLVE_ACCEPT_NEW)):
            try:
                button.clicked.disconnect()
            except (RuntimeError, TypeError):  # pragma: no cover
                pass
            button.clicked.connect(
                lambda _checked=False, r=resolution: self.resolve_selected(r)
            )

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

    # ----------------------------------------------------------------- edits
    @S.guard
    def on_request_edit(self, instance: str) -> None:
        """A panel asked for an instance to be edited (task card / instance list)."""
        self.cancel_roi_edit()
        if getattr(self.session, "editing_instance", None) != instance:
            self.session.begin_edit(instance)
        self._sync_editing_layer()
        self.set_sam_instance(instance)
        self._attach_tool()
        self.report(f"editing {instance}")

    @S.guard
    def on_stroke(self, rect: object) -> None:
        """One finished pixel stroke: one undoable op, one sidecar write."""
        tool = self.sender() or self.active_tool
        if tool is self.occluder:
            self.session.commit_occluder(self.overlay.occluder_layer(
                self.occluder.occluder_type), self.occluder.occluder_type)
            return
        instance = getattr(self.session, "editing_instance", None)
        if instance is None or self.overlay is None:
            return
        compat.push_stroke(self.session, instance,
                           getattr(tool, "stroke_before", None), self.overlay.editing)
        self.sidecar.save(self.session.current(), instance, self.overlay.editing)
        self.update_status()

    def set_editing_mask(self, mask: np.ndarray, undoable: bool = False) -> None:
        """Replace the editing layer everywhere it is held at once."""
        instance = getattr(self.session, "editing_instance", None)
        before = None if self.overlay is None else self.overlay.editing.copy()
        mask = np.asarray(mask, dtype=bool)
        if undoable and instance is not None:
            compat.push_stroke(self.session, instance, before, mask)
            self.sidecar.save(self.session.current(), instance, mask)
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
        if getattr(self.session, "editing_instance", None) is None:
            self.report("nothing is being edited")
            return
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
        self.sidecar.clear()
        self.set_sam_instance(None)
        self._sync_editing_layer()
        self.refresh_overlay()
        self.report(f"committed ({scope}): {result.get('changed', '')}".strip())

    @S.guard
    def act_clear_edit(self) -> None:
        """``Esc``: drop the editing layer, or abandon the ROI rectangle."""
        if getattr(self.session, "editing_instance", None) is not None:
            self.session.clear_edit()
            self.sidecar.clear()
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
        """``Space``: verify the frame; on refusal the problems are shown, not a dialog."""
        step = self.session.current().step
        blobs = self.unexplained_boxes()
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

    def on_roi_box(self, box: object) -> None:
        """The ROI tool finished a drag."""
        self.roi_draft = tuple(int(round(float(v))) for v in box)  # type: ignore[misc]

    @S.guard
    def accept_roi(self) -> None:
        """Store the rectangle on the pose segment and zoom to it."""
        key = self.session.current()
        segment = self._pose_segment(key)
        if self.roi_draft is None or segment is None:
            self.cancel_roi_edit()
            return
        self.db.set_pose_segment_roi(int(key.desktop), str(key.view), int(segment),
                                     list(self.roi_draft))
        accepted = self.roi_draft
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
        self.session.commit_box(instance, [float(v) for v in box])  # type: ignore[misc]
        self.canvas.set_rubber_band(None)
        self.report(f"bench box stored for {instance}")

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

    def _offer_restore(self, key) -> None:
        hw = None if self.overlay is None else self.overlay.hw
        found = self.sidecar.pending_for(key, hw)
        if found is None or getattr(self.session, "editing_instance", None) is not None:
            self._restore_offer = None
            self.restore_bar.hide()
            return
        self._restore_offer = {"instance": found["instance"], "mask": found["mask"]}
        self.restore_bar.show_text(
            f"上次未提交的编辑（{found['instance']}）可以恢复 / "
            f"an uncommitted edit of {found['instance']} was found"
        )

    @S.guard
    def restore_pending(self) -> None:
        """Put the recovered layer back into the editing layer."""
        offer, self._restore_offer = self._restore_offer, None
        self.restore_bar.hide()
        if offer is None:
            return
        self.session.begin_edit(offer["instance"])
        self.set_editing_mask(offer["mask"])
        self._attach_tool()
        self.report(f"restored the uncommitted edit of {offer['instance']}")

    @S.guard
    def discard_pending(self) -> None:
        """Throw the recovered layer away."""
        self._restore_offer = None
        self.restore_bar.hide()
        self.sidecar.clear()
        self.report("the recovered edit was discarded")
