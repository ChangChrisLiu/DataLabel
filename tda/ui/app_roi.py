"""The ROI, the bench box, the review verdicts and the crash sidecar offer.

Mixed into :class:`tda.ui.app.MainWindow` next to
:class:`tda.ui.app_edit.EditMixin`.  What these four have in common is that none
of them is a pixel edit: they are the rectangle that frames the chassis, the
rectangle that stands for a part on the bench, the three answers a queued
conflict can get, and the offer to take back an edit a crash interrupted.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt

from tda.core.cache import ROI_SAMPLE_FRAMES, suggest_roi, suggest_roi_over
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import session_api as api

__all__ = ["RoiMixin", "as_bgr"]

#: What :meth:`SessionLike.instance_rows` calls a part lying in the staging area.
ON_BENCH = "on_bench"
#: Shown when the detector returns the whole frame, which means "not found".
NO_CHASSIS_FOUND = ("未能自动找到机箱：请拖一个框 / could not find the chassis: "
                    "drag a box around it (Enter stores it)")
#: Shown when Enter arrives before the segment has been measured.
ROI_STILL_MEASURING = ("还在找机箱，稍等或直接拖框 / still looking for the chassis "
                       "-- wait a moment, or drag a box yourself")


def as_bgr(rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """The session hands out **RGB**; :mod:`tda.core.cache` measures **BGR**.

    The two channel orders had never been reconciled, and the detector was fed
    the session's array as it came. It is a colour detector: the yellow tape
    square read as BGR is a cyan blob, ``inRange`` finds no square, and the
    stage that needs one returns nothing. On the scanner that was invisible --
    the second stage, "whatever is not the scan bed", does not look at hue and
    carried every frame -- and on an OAK frame, where the tape square *is* the
    region the detector may look in, it meant every proposal was the whole
    frame, which is what the controller measured as ``roi: []``.
    """
    if rgb is None or getattr(rgb, "ndim", 0) != 3 or rgb.shape[2] != 3:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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
        """``Shift+R``: draw the chassis rectangle again -- through the gate.

        Arming the ROI tool takes ``Enter`` and ``Esc`` away from an uncommitted
        editing layer and gives them to a rectangle, so it is a way out of the
        edit like every other one and is refused the same way.
        """
        if not self.can_leave_edit():
            return
        self.start_roi_edit()

    def start_roi_edit(self) -> None:
        """Propose a rectangle and let the annotator drag it (``Enter`` accepts).

        The proposal is measured on **three frames of the pose segment** -- its
        first, its middle and its last that have an image -- and the union of
        what they say is what is offered (:func:`tda.core.cache.suggest_roi_over`).
        Inside a pose segment the machine does not move, so all three are
        measurements of the same rectangle, and a per-frame box can only be too
        small: segmentation clips where contrast fails, which by the late steps
        is most of an emptied chassis. Measuring the frame that happened to be
        open was worse still -- that is the *last* step, a bright empty box
        where both scanner strategies give up.

        It is **asynchronous**. Three decodes and three detections are 165 ms on
        a 12 MP OAK segment and 465 ms on a scanner one, which is not something
        to spend before the first paint, so the tool is armed at once and the
        rectangle arrives when :meth:`_on_roi_proposed` is called -- unless the
        annotator has already dragged one, which always wins.
        """
        image = self.session.image()
        if image is None:
            return
        stored = self.roi()
        self._roi_dragged = False
        self._roi_wanted = self._roi_segment_key()
        if stored is not None:
            self.roi_draft = tuple(int(v) for v in stored)
        else:
            self.roi_draft = None
        self.roi_editing = True
        # Arm the tool first: detaching a SAM tool clears the rubber band, so
        # painting the draft before the swap would erase it again.
        self._attach_tool()
        self.canvas.set_rubber_band(self.roi_draft)
        if stored is not None:
            self.report("拖动框选机箱范围，Enter 确认 / "
                        "drag the chassis box, Enter to accept")
            return
        self.report("正在寻找机箱…可直接拖框 / looking for the chassis -- "
                    "drag a box any time")
        self.roi_proposer.request(self._roi_wanted, self.session.view,
                                  self._roi_sample_paths())

    def _roi_segment_key(self) -> tuple:
        """What a pending proposal is *about*, so a stale one can be dropped."""
        key = self.session.current()
        return (int(key.desktop), str(key.view), self._pose_segment(key))

    def _roi_sample_steps(self) -> list[int]:
        """The segment's first, middle and last annotatable steps.

        Both ends matter and for opposite reasons: the first frame holds the
        whole machine, which is the box; the last holds an empty chassis whose
        dark floor reads as bench, which is where a single-frame measurement
        clips. The middle is the tie-breaker when one of the two is spoiled by
        a hand or an arm over the bench.
        """
        if not compat.is_open(self.session):
            return []
        row = self.db.pose_segment_for(self.session.current()) or {}
        steps = sorted(int(s) for s in self.session.steps())
        start, end = row.get("start_step"), row.get("end_step")
        if start is not None:
            steps = [s for s in steps if s >= int(start)]
        if end is not None:
            steps = [s for s in steps if s <= int(end)]
        if len(steps) <= ROI_SAMPLE_FRAMES:
            return steps
        return [steps[0], steps[len(steps) // 2], steps[-1]]

    def _roi_sample_paths(self) -> list[str]:
        """Where those frames' pixels are; the worker reads them itself."""
        out: list[str] = []
        for step in self._roi_sample_steps():
            found = self.session.image_path(step)
            if found:
                out.append(str(found))
        return out

    def _on_roi_proposed(self, payload: object) -> None:
        """A measurement came back: show it, unless it is no longer wanted.

        Three ways it can be stale and all three are ordinary: the annotator
        left the segment, accepted or cancelled the ROI edit, or dragged their
        own rectangle while the worker was reading three 12 MP frames. A
        rectangle a human drew is never replaced by one a detector measured.
        """
        if not isinstance(payload, dict) or not self.roi_editing:
            return
        if payload.get("segment") != getattr(self, "_roi_wanted", None):
            return
        if getattr(self, "_roi_dragged", False):
            return
        box = payload.get("box")
        image = self.session.image() if compat.is_open(self.session) else None
        if box is None or image is None:
            self.report(NO_CHASSIS_FOUND)
            return
        self.roi_draft = self._scaled_box(box, 1.0, image.shape[:2])
        self.canvas.set_rubber_band(self.roi_draft)
        if self._is_whole_frame(self.roi_draft):
            self.report(NO_CHASSIS_FOUND)
        else:
            self.report("拖动框选机箱范围，Enter 确认 / "
                        "drag the chassis box, Enter to accept")

    def wait_for_roi_proposal(self, timeout: float = 20.0) -> bool:
        """Block until the pending measurement has been delivered (tests, smoke).

        The annotator never waits for this -- that is the point of it being on a
        worker -- but a test that wants to look at the rectangle, and the smoke
        run that reports it, have to.
        """
        from PySide6.QtWidgets import QApplication

        drained = self.roi_proposer.wait(timeout)
        QApplication.processEvents()
        return drained

    def _roi_reference_image(self, fallback):
        """``(image, scale)`` a synchronous proposal would be measured on.

        Kept for :meth:`propose_roi_now`, the path with no event loop behind it.
        """
        if not compat.is_open(self.session):
            return fallback, 1.0
        for step in self._roi_sample_steps():
            image = self.session.image_at(step)
            if image is not None:
                return image, 1.0
        return fallback, 1.0

    def propose_roi_now(self) -> Optional[tuple]:
        """The segment's proposal, measured here and now on the GUI thread.

        The synchronous answer, for a caller with no event loop to deliver the
        asynchronous one. It reads the same frames through the session's image
        cache, so it also has to undo the session's channel order -- which is
        the one thing :func:`tda.ui.app_roi_worker.measure_paths` never has to
        do, since ``cv2.imread`` already gives BGR.
        """
        images = []
        for step in self._roi_sample_steps():
            found = self.session.image_at(step)
            if found is not None:
                images.append(as_bgr(found))
        if not images:
            return None
        return tuple(int(v) for v in suggest_roi_over(images, self.session.view))

    @staticmethod
    def _scaled_box(box, scale: float, hw) -> tuple:
        """A measured box in full-frame pixels, clamped to the frame."""
        h, w = int(hw[0]), int(hw[1])
        x0, y0, x1, y1 = (int(round(float(v) * float(scale))) for v in box)
        return (max(0, min(x0, w - 1)), max(0, min(y0, h - 1)),
                max(1, min(x1, w)), max(1, min(y1, h)))

    def _is_whole_frame(self, box) -> bool:
        """Is this rectangle "I could not find the chassis" rather than an answer?"""
        hw = None if self.overlay is None else self.overlay.hw
        if hw is None or box is None:
            return False
        x0, y0, x1, y1 = (int(v) for v in box)
        return x0 <= 0 and y0 <= 0 and x1 >= int(hw[1]) and y1 >= int(hw[0])

    @S.guard
    def on_roi_box(self, box: object) -> None:
        """The ROI tool finished a drag; the rectangle is checked, not trusted."""
        from tda.core.db_pose import clean_roi

        hw = None if self.overlay is None else self.overlay.hw
        # whatever the worker comes back with, a rectangle a human dragged wins
        self._roi_dragged = True
        try:
            self.roi_draft = tuple(clean_roi(list(box), hw))  # type: ignore[arg-type]
        except ValueError as refused:
            self.report(f"that rectangle is not usable: {refused}")
            self.canvas.set_rubber_band(self.roi_draft)
            return
        self.canvas.set_rubber_band(self.roi_draft)

    @S.guard
    def accept_roi(self) -> None:
        """Store the rectangle on the pose segment and zoom to it.

        A whole-frame rectangle is never stored: it is what the detector
        returns when it has *not* found the chassis, and storing it silently is
        what let the difference map treat the scan bed as part of the machine.
        The annotator drags one instead, and the tool stays armed until they do.
        """
        key = self.session.current()
        segment = self._pose_segment(key)
        if self.roi_draft is None and self.roi_proposer.pending():
            # Enter before the measurement landed: the annotator means "take the
            # box", and there will be one in a moment. Cancelling the ROI edit
            # here instead would look like Enter did nothing at all.
            self.report(ROI_STILL_MEASURING)
            return
        if self.roi_draft is None or segment is None:
            self.cancel_roi_edit()
            return
        if self._is_whole_frame(self.roi_draft):
            self.report(NO_CHASSIS_FOUND)
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
        """``R`` + a drag: the staging-area rectangle of a part on the bench.

        The instance an ``add_bench_box`` card item armed the tool for wins:
        that is the one the annotator activated, whatever the instance table
        happens to have selected.  With nothing armed the selected row is used
        only when the *frame* says that part is on the bench -- a rectangle
        drawn over a part that is still in the machine is a mistake, and filing
        it as that part's bench box is not a guess worth making.
        """
        instance = getattr(self, "bench_instance", None)
        if not instance:
            instance = self._selected_bench_instance()
        if not instance:
            self.report("先选中台面上的零件（或用任务卡的 ▭ 项）/ "
                        "select a part the frame says is on the bench first")
            return
        self.canvas.set_rubber_band(None)
        try:
            self.session.commit_box(instance, [float(v) for v in box])  # type: ignore[misc]
        except ValueError as refused:
            self.report_error(f"refused: {refused}")
            return
        self.disarm_bench()
        self.refresh_overlay()
        self.report(f"bench box stored for {instance}")

    def _selected_bench_instance(self) -> Optional[str]:
        """The selected instance, but only while this frame has it on the bench."""
        instance = (self.instances.selected_instance()
                    or self.task_card.current_instance())
        if not instance:
            return None
        on_bench = any(str(row.get("key")) == instance
                       and row.get("placement") == ON_BENCH
                       for row in self.session.instance_rows())
        return instance if on_bench else None

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
        cid = self.review.selected_conflict()
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
    def act_rework_selected(self) -> None:
        """``R`` in Review: rework whatever the open queue has selected."""
        step = self.review.selected_step()
        if step is None:
            self.report("select a queue entry first")
            return
        self.on_rework(int(step))

    @S.guard
    def act_resolve(self, resolution: str) -> None:
        """``K`` / ``N`` in Review: settle the selected conflict."""
        self.resolve_selected(resolution)

    @S.guard
    def on_rework(self, step: int) -> None:
        """``R`` in Review mode: open the frame in Annotate mode to work on it.

        Review mode's canvas is read-only, so "rework" is literally the move
        into the mode where the frame can be edited -- through the gate, like
        every other way of leaving the frame that is open now.
        """
        def reopen() -> None:
            self.session.goto(int(step), force=True)
            self.set_mode("annotate")
            self.report(f"step {step} reopened for rework")

        self.leave_frame(reopen)

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
        """Put the recovered layer back into the editing layer.

        Refused while the layer already holds uncommitted strokes: restoring
        would replace work that is newer than the file being offered.
        """
        if self._restore_offer is not None and not self.can_leave_edit():
            return
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
