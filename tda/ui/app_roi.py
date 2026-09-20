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

#: What the ROI bar says while the rectangle is on screen: what it is, what it
#: is for, and what to do about it.  "ROI 这个不是很明显，让我很迷惑" -- the
#: annotator's first trial; the rectangle appeared, waited for an answer and
#: said none of that (task U1, ruling U-ROI-1).
ROI_BAR_EDITING = (
    "机箱范围（ROI）/ chassis range — 差异图和 SAM 提示框只在这个框里算，"
    "画面也按它缩放 / the difference map and SAM's prompt boxes are computed "
    "inside it and the view zooms to it。拖动边或角可以调整，框里按住可以整体"
    "移动，空白处拖动重画 → Enter 保存；Esc 先跳过，之后按 Shift+R 再画。"
)
#: The compact reminder that replaces it once the annotator starts working.
ROI_BAR_PENDING = (
    "ROI 未确认 / chassis range not answered — 现在差异图和 SAM 提示框会在整张"
    "图上算 / the difference map and SAM prompt boxes run over the whole frame。"
)
#: Said once when the question is answered with "no rectangle at all".
ROI_NONE_STORED = ("好，这一段不用 ROI / no ROI for this pose segment: the "
                   "difference map will run over the whole frame")
#: Said when "确认建议框" is pressed before anything has been measured.
ROI_NO_PROPOSAL = ("还没有可确认的框：按 Shift+R 自己画一个 / nothing measured "
                   "yet -- press Shift+R and drag one")
#: Said when "确认建议框" has to go and measure the segment again first.
ROI_REMEASURING = ("正在重新测量机箱范围，量到就存 / re-measuring the chassis "
                   "range; it will be stored when it lands")
#: Appended to the bar when the offered rectangle is one that will not be
#: stored, so that a button that is going to refuse says so before it is
#: pressed rather than after (round 2, C1).
ROI_BAR_UNUSABLE = ("自动框没找到机箱：整幅图不作为 ROI，请按 Shift+R 自己画 / "
                    "the detector found no chassis; a whole-frame rectangle is "
                    "never stored -- draw one with Shift+R。")
#: Refused: too small to be a chassis.  The floor is the ruled 64 px / 5 % of
#: the frame's shorter side, capped at half of it so that a small frame (the
#: 64x64 test scene, a thumbnail) still has usable rectangles at all.
ROI_TOO_SMALL = ("这个框太小了，不像机箱（短边至少 {floor} 像素）/ that rectangle "
                 "is too small for a chassis: the shorter side must be at "
                 "least {floor} px")
#: Said when the ROI is stored while pixels are uncommitted: storing is safe,
#: moving the view under a half-drawn mask is not (round 2, M2).
ROI_FIT_DEFERRED = ("ROI 已存 {roi}；画面等这次编辑结束再缩放（或按 F）/ ROI "
                    "stored; the view will fit it after this edit, or press F")


def roi_min_side(hw: Optional[tuple]) -> int:
    """Smallest side a stored chassis rectangle may have, in image pixels."""
    if not hw:
        return 0
    short = float(min(int(hw[0]), int(hw[1])))
    return int(min(max(64.0, 0.05 * short), short / 2.0))


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
        first, its middle and its last that have an image -- and what they agree
        on is what is offered (:func:`tda.core.cache.suggest_roi_over`). Inside
        a pose segment the machine does not move, so all three are measurements
        of the same rectangle. On the scanner a per-frame box can only be too
        small -- segmentation clips where contrast fails, which by the late
        steps is most of an emptied chassis -- so the three are unioned. On an
        OAK view a box can also be too *big*, the tape square having let the
        bench in, so each edge is the median of the three instead. Measuring the
        frame that happened to be open was worse still -- that is the *last*
        step, a bright empty box where both scanner strategies give up.

        It is **asynchronous**. Three decodes and three detections are 165 ms on
        a 12 MP OAK segment and 465 ms on a scanner one, which is not something
        to spend before the first paint, so the tool is armed at once and the
        rectangle arrives when :meth:`_on_roi_proposed` is called -- unless the
        annotator has already dragged one, which always wins.
        """
        image = self.session.image()
        if image is None:
            return
        # The rectangle is about to take ``Enter`` and ``Esc``; a draft ghost
        # holding them at the same time would leave two things on screen
        # claiming the same two keys.
        self.forget_draft_ghost()
        # And the ROI is what the difference map was computed inside, so an
        # alternate box prompt walked to with ``Shift+C`` is an offer about the
        # rectangle that is being replaced.
        self.reset_prompt_rank()
        stored = self.roi()
        self._roi_dragged = False
        self._roi_awaiting = stored is None
        self._roi_wanted = self._roi_segment_key()
        if stored is not None:
            self.roi_draft = tuple(int(v) for v in stored)
        else:
            self.roi_draft = None
            if self._roi_wanted is not None:
                self._roi_pending.add(self._roi_wanted)
        if self.roi_draft is not None and self._roi_wanted is not None:
            self._roi_proposal[self._roi_wanted] = self.roi_draft
        self.roi_editing = True
        # Arm the tool first: detaching a SAM tool clears the rubber band, so
        # painting the draft before the swap would erase it again.
        self._attach_tool()
        self._show_roi_rect()
        # Also on this path: ``Shift+R`` re-arms the rectangle without going
        # through the frame hook, and a rectangle with no bar is the thing the
        # annotator could not read in the first place.
        self.refresh_roi_bar()
        self.logger.info("roi proposal opened segment=%s stored=%s",
                         self._roi_wanted, stored)
        self.update_status()
        if stored is not None:
            self.report("拖动边或角调整机箱范围，Enter 保存 / "
                        "drag an edge or a corner, Enter to save")
            return
        self.report("正在寻找机箱…可直接拖框 / looking for the chassis -- "
                    "drag a box any time")
        self.roi_proposer.request(self._roi_wanted, self.session.view,
                                  self._roi_sample_paths())

    def _roi_segment_key(self) -> Optional[tuple]:
        """What a pending proposal is *about*, so a stale one can be dropped.

        The same five-tuple a dismissal is remembered by
        (:meth:`~tda.ui.app_edit.EditMixin.roi_key`): desktop, view, segment
        number **and the segment's step range**. The range is what makes it
        self-invalidating -- a pose re-cut renumbers segments and moves their
        ranges, so a measurement taken for "segment 2 of D13 oak1" before the
        cut is not an answer about the segment 2 that exists after it, and an
        identity that stopped at the number would have drawn it anyway.
        """
        return self.roi_key()

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
        # the steps this view can actually be measured on: a step flagged
        # `missing` is in `steps()` (the state machine runs through it) but has
        # no image, and sampling one silently leaves the union a frame short --
        # on the real D61 the scanner's last step is exactly that
        steps = sorted(int(s) for s in self.session.available_steps())
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

        Four ways it can be stale and all four are ordinary: the annotator left
        the segment, accepted the rectangle, dismissed the proposal, or dragged
        their own rectangle while the worker was reading three 12 MP frames. A
        rectangle a human drew is never replaced by one a detector measured.

        ``_roi_awaiting`` is the one that catches the nastiest case, because
        every other guard passes it: request, drag, ``Enter``, ``Shift+R``, and
        *then* the old answer lands -- same segment, nothing dragged since the
        tool was re-armed, the edit open. The tool was re-armed over a segment
        that now **has** a rectangle, so it asked for nothing, so there is
        nothing for an answer to be an answer to.
        :meth:`tda.ui.app_roi_worker.RoiProposer.cancel` stops the worker
        delivering it at all; this is the same thing said at the point of use,
        where a payload that arrives by any other route is also turned away.
        """
        if not isinstance(payload, dict):
            return
        wanted = getattr(self, "_roi_accept_when_measured", None)
        if wanted is not None and payload.get("segment") == wanted:
            self._on_roi_measured_for_accept(wanted, payload)
            return
        if not self.roi_editing:
            return
        if not getattr(self, "_roi_awaiting", False):
            return
        if payload.get("segment") != getattr(self, "_roi_wanted", None):
            return
        if getattr(self, "_roi_dragged", False):
            return
        self._roi_awaiting = False
        box = payload.get("box")
        image = self.session.image() if compat.is_open(self.session) else None
        if box is None or image is None:
            self.report(NO_CHASSIS_FOUND)
            return
        self.roi_draft = self._scaled_box(box, 1.0, image.shape[:2])
        self._remember_proposal(self.roi_draft)
        self._show_roi_rect()
        self.refresh_roi_bar()
        if self._is_whole_frame(self.roi_draft):
            self.report(NO_CHASSIS_FOUND)
        else:
            self.report("拖动边或角调整机箱范围，Enter 保存 / "
                        "drag an edge or a corner, Enter to save")

    def _remember_proposal(self, box) -> None:
        """Keep one rectangle against the segment it was offered for."""
        wanted = getattr(self, "_roi_wanted", None) or self.roi_key()
        if wanted is not None and box is not None:
            self._roi_proposal[wanted] = tuple(int(v) for v in box)

    @S.guard
    def _on_roi_measured_for_accept(self, wanted, payload: dict) -> None:
        """A measurement asked for by ``确认建议框`` came back (round 2, I1)."""
        self._roi_accept_when_measured = None
        if wanted != self.roi_key():
            return                      # the annotator walked away meanwhile
        box = payload.get("box")
        image = self.session.image() if compat.is_open(self.session) else None
        if box is not None and image is not None:
            box = self._scaled_box(box, 1.0, image.shape[:2])
            self._roi_proposal[wanted] = box
        refusal = self.roi_refusal(box)
        if refusal:
            self.refresh_roi_bar()
            self.report(refusal)
            return
        segment = self._pose_segment(self.session.current())
        if segment is None:
            self.report(ROI_NO_PROPOSAL)
            return
        self.store_roi(box, segment, self.session.current())

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

    def roi_refusal(self, box) -> str:
        """Why this rectangle will not be stored, or ``""``.

        Asked **before** anything is armed or written, so that every caller --
        ``Enter`` on the rectangle, the ``确认建议框`` button, and the button's
        own visibility -- agrees about what is storable. A button that refuses
        after it has switched the window into rectangle mode is how ``Enter``
        came to be routed at a rectangle nobody could see (round 2, C1).
        """
        if box is None:
            return ROI_NO_PROPOSAL
        if self._is_whole_frame(box):
            return NO_CHASSIS_FOUND
        hw = None if self.overlay is None else self.overlay.hw
        floor = roi_min_side(hw)
        x0, y0, x1, y1 = (float(v) for v in box)
        if floor and min(x1 - x0, y1 - y0) < floor:
            return ROI_TOO_SMALL.format(floor=floor)
        return ""

    # ------------------------------------------------------- the ROI on screen
    def _show_roi_rect(self) -> None:
        """Put the rectangle (or the stored one) on the canvas and in the tool."""
        if self.roi_editing:
            self.canvas.set_roi(self.roi_draft, editing=True)
            self.roi_tool.set_rect(self.roi_draft)
            return
        self.roi_tool.set_rect(None)
        self.canvas.set_roi(self.roi(), editing=False)

    def roi_unanswered(self) -> bool:
        """Is this segment's ROI question still open (ruling U-ROI-3)?

        Per **segment key**, and it lives until the question is *answered* --
        a stored rectangle, ``确认建议框`` or ``不用 ROI`` -- or until
        :meth:`reset_roi_proposals`. Skipping with ``Esc`` only ends the
        rectangle mode, and leaving for another view and coming back must find
        the reminder where it was (round 2, I1).
        """
        key = self.roi_key()
        return key is not None and key in getattr(self, "_roi_pending", ())

    def roi_proposal_for_segment(self):
        """The rectangle on offer for the open segment, or ``None``."""
        return self._roi_proposal.get(self.roi_key())

    def refresh_roi_bar(self) -> None:
        """Show the right half of the ROI question, or nothing at all."""
        buttons = self._roi_buttons
        offered = (self.roi_draft if self.roi_editing
                   else self.roi_proposal_for_segment())
        # A whole-frame "not found" is not something ``确认建议框`` will store,
        # so the bar says so instead of letting the button refuse afterwards.
        unusable = offered is not None and bool(self.roi_refusal(offered))
        if self.roi_editing:
            for name in ("save", "skip"):
                buttons[name].setVisible(True)
            for name in ("accept", "redraw", "none"):
                buttons[name].setVisible(False)
            self.roi_bar.show_text(
                ROI_BAR_EDITING + (ROI_BAR_UNUSABLE if unusable else ""))
            return
        if self.roi_unanswered():
            buttons["save"].setVisible(False)
            buttons["skip"].setVisible(False)
            # Hidden only when there *is* an offer and it would be refused;
            # with none at all the button re-measures (round 2, I1).
            buttons["accept"].setVisible(not unusable)
            buttons["redraw"].setVisible(True)
            buttons["none"].setVisible(True)
            self.roi_bar.show_text(
                ROI_BAR_PENDING + (ROI_BAR_UNUSABLE if unusable else ""))
            return
        self.roi_bar.hide()

    def on_frame_changed_roi(self) -> None:
        """Keep the rectangle, the bar and the status label with the frame.

        Nothing is forgotten here: the unanswered set and the offers are per
        segment key and outlive every frame, view and desktop change.
        """
        self._show_roi_rect()
        self.refresh_roi_bar()
        self.fit_pending_roi()

    @S.guard
    def on_roi_preview(self, box: object) -> None:
        """Mid-drag: draw what the rectangle would be, store nothing.

        ``None`` is the tool taking a gesture back -- a click, or a drag too
        small to be a rectangle -- and the draft on screen goes back to the one
        the window holds.
        """
        if box is None:
            self._show_roi_rect()
            return
        self.canvas.set_roi(tuple(float(v) for v in box), editing=True)

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
            self._show_roi_rect()
            return
        self._remember_proposal(self.roi_draft)
        self._show_roi_rect()
        self.refresh_roi_bar()
        refusal = self.roi_refusal(self.roi_draft)
        if refusal:
            self.report(refusal)

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
        refusal = self.roi_refusal(self.roi_draft)
        if refusal:
            # The rectangle stays on screen and so does everything about it:
            # the annotator drags a better one or presses Esc.
            self.report(refusal)
            return
        # There is a rectangle now, and it is the annotator's. A measurement
        # still in flight is about the same segment and nothing has been dragged
        # since, so every guard in _on_roi_proposed would let it through the
        # next time the tool is armed.
        self.roi_proposer.cancel()
        self._roi_awaiting = False
        self.store_roi(self.roi_draft, segment, key)

    def store_roi(self, box, segment: int, key) -> bool:
        """Write one rectangle onto the pose segment; ``True`` when it landed.

        **Every** exit leaves :attr:`roi_editing`, the armed tool and the bar
        consistent with what actually happened. A refusal or a failed write
        used to return with ``roi_editing`` left standing from the caller, so
        ``Enter`` was routed at a rectangle that was not on screen and the open
        edit could not be committed at all until ``Esc`` (round 2, C1).
        """
        hw = None if self.overlay is None else self.overlay.hw
        try:
            accepted = tuple(self.db.set_pose_segment_roi(
                int(key.desktop), str(key.view), int(segment), list(box),
                annotator=self.annotator, hw=hw,
            ))
        except Exception as exc:  # noqa: BLE001 - a write that failed is news
            self.logger.exception("the ROI could not be stored: %s", exc)
            self.report_error(f"ROI 没能存下：{exc} / the ROI could not be stored")
            return False
        answered = self.roi_key()
        self.roi_draft = accepted
        self.roi_editing = False
        self._roi_accept_when_measured = None
        # The question is answered: the reminder never comes back for it.
        self._roi_pending.discard(answered)
        self._roi_proposal[answered] = accepted
        self._roi_dismissed.discard(answered)
        self.canvas.set_rubber_band(None)
        self._attach_tool()
        self._show_roi_rect()
        self.refresh_roi_bar()
        self.logger.info("roi accepted segment=%s roi=%s", answered, accepted)
        self.update_status()
        self.request_assist()
        if self.has_uncommitted_edit():
            # Storing loses nothing, but moving the view under a half-drawn
            # mask does: the fit waits for the next frame change or ``F``.
            self._roi_fit_pending = True
            self.report(ROI_FIT_DEFERRED.format(roi=accepted))
        else:
            self.canvas.zoom_to(accepted)
            self.report(f"ROI stored: {accepted}")
        return True

    def fit_pending_roi(self) -> None:
        """Do the zoom :meth:`store_roi` put off, once it is safe (round 2, M2)."""
        if not getattr(self, "_roi_fit_pending", False):
            return
        if self.has_uncommitted_edit():
            return
        self._roi_fit_pending = False
        roi = self.roi()
        if roi is not None:
            self.canvas.zoom_to(roi)
            self.update_status()

    @S.guard
    def act_accept_roi_proposal(self) -> None:
        """``确认建议框``: store the offered rectangle exactly as it stands.

        It never *arms* the rectangle to do it. Arming first and delegating to
        :meth:`accept_roi` meant that a refused proposal -- the detector's
        whole-frame "not found" is the common one -- left the window in
        rectangle mode with no rectangle drawn (round 2, C1).
        """
        if self.roi_editing:
            self.accept_roi()
            return
        wanted = self.roi_key()
        box = self._roi_proposal.get(wanted)
        if box is None:
            self._remeasure_for_accept(wanted)
            return
        refusal = self.roi_refusal(box)
        if refusal:
            self.report(refusal)
            return
        segment = self._pose_segment(self.session.current())
        if segment is None:
            self.report(ROI_NO_PROPOSAL)
            return
        self.roi_proposer.cancel()
        self._roi_awaiting = False
        self.store_roi(box, segment, self.session.current())

    def _remeasure_for_accept(self, wanted) -> None:
        """No proposal in hand (a reopened session): go and measure one.

        The alternative was to hide the button, which leaves the annotator with
        a reminder they cannot answer without drawing the rectangle themselves.
        """
        if wanted is None or not compat.is_open(self.session):
            self.report(ROI_NO_PROPOSAL)
            return
        paths = self._roi_sample_paths()
        if not paths:
            self.report(ROI_NO_PROPOSAL)
            return
        self._roi_accept_when_measured = wanted
        self._roi_wanted = wanted
        self.roi_proposer.request(wanted, self.session.view, paths)
        self.report(ROI_REMEASURING)

    @S.guard
    def act_no_roi(self) -> None:
        """``不用 ROI``: answer the question with "none", and stop asking.

        The dismissal is recorded exactly the way leaving the rectangle records
        it -- ``(desktop, view, seg, start_step, end_step)``, so a pose re-cut
        still asks -- and the reminder goes with it: this *is* the answer.
        """
        if self.roi_editing:
            self.cancel_roi_edit()
        answered = self.roi_key()
        if answered is not None:
            self._roi_dismissed.add(answered)
            self._roi_pending.discard(answered)
            self._roi_proposal.pop(answered, None)
        self._roi_accept_when_measured = None
        self.roi_proposer.cancel()
        self._roi_awaiting = False
        self._show_roi_rect()
        self.refresh_roi_bar()
        self.logger.info("roi dismissed segment=%s (no roi)", answered)
        self.update_status()
        self.report(ROI_NONE_STORED)

    def cancel_roi_edit(self) -> None:
        """Leave ROI editing without storing anything -- and remember that.

        Every way the rectangle leaves the screen without being stored ends
        here: ``Esc``, picking another tool, starting an instance edit, arming
        a bench box.  All of them are the annotator saying "not now", so the
        segment is recorded as dismissed and the next frame of it does not pop
        the rectangle up again.  The memory carries the segment's step range,
        so the next *re-cut* does ask.

        What it no longer does is make the question **disappear**: the segment
        stays in :attr:`_roi_pending` and the bar keeps a one-line reminder with
        the three answers (ruling U-ROI-3).  The first annotator started an edit
        over an unanswered proposal, and from then on the difference map and
        every SAM prompt box ran over the whole frame with nothing saying so.
        """
        if not self.roi_editing:
            return
        self.roi_editing = False
        self.roi_draft = self.roi()
        self._roi_awaiting = False
        self.roi_proposer.cancel()   # nobody is waiting for it any more
        dismissed = self.roi_key()
        if dismissed is not None:
            self._roi_dismissed.add(dismissed)
        self.canvas.set_rubber_band(None)
        self._attach_tool()
        self._show_roi_rect()
        self.refresh_roi_bar()
        self.logger.info("roi rectangle dismissed segment=%s; unanswered=%s",
                         dismissed, self.roi_unanswered())
        self.update_status()

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
                               "key": found["key"],
                               "adopted": list(found.get("adopted") or [])}
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
        # The layer is about to be replaced from outside the tool, which is
        # what ``_sync_editing_layer`` handles on every other such path -- and
        # ``set_sam_instance`` above does nothing at all when the restored
        # instance is the one already being edited, which is the common case.
        self.reset_sam_prompt()
        self.set_editing_mask(offer["mask"])
        # The layer is back; so is what it was built from, or the commit that
        # follows would file somebody's adopted draft as hand-drawn work.
        self.note_restored_adoptions(offer["key"], offer["instance"],
                                     offer.get("adopted"))
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
