"""The ROI, the bench box, the review verdicts and the crash sidecar offer.

Mixed into :class:`tda.ui.app.MainWindow` next to
:class:`tda.ui.app_edit.EditMixin`.  What these four have in common is that none
of them is a pixel edit: they are the rectangle that frames the chassis, the
rectangle that stands for a part on the bench, the three answers a queued
conflict can get, and the offer to take back an edit a crash interrupted.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt

from tda.core.cache import suggest_roi
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import session_api as api

__all__ = ["RoiMixin"]


class RoiMixin:
    """The chassis rectangle, bench boxes, conflict verdicts, crash recovery."""

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
