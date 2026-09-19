"""SAM prompt tools for the canvas (spec 4.6), split out of ``tools.py``.

The pixel tools in :mod:`tda.ui.canvas.tools` are synchronous: a stroke lands in
the editing layer while the mouse is still down.  Everything here is not -- a
prompt travels to a worker thread and the mask comes back some tens of
milliseconds later, by which time the annotator may have clicked again, moved to
another frame, or picked another instance.  Most of the code below exists to
make sure a late answer to an old question is thrown away instead of written
into whatever is on screen now:

* every submission is stamped with a monotonically increasing **token**, and
  only the newest token may apply -- a superseded result is dropped silently;
* it also carries a **frame identity**: the token from
  :meth:`SamToolBase.set_frame_token`, which the session **must** set on every
  frame change, plus the instance the mask will be written to. A result whose
  identity no longer matches is dropped with :attr:`~SamToolBase.sigError`,
  because applying it would write the mask of frame k into frame k-1. Prompting
  without a frame token is refused outright rather than guessed at;
* the crop rectangle is bounds-checked against the overlay before any write, so
  a stale rectangle can never raise out of a Qt slot;
* :meth:`SamToolBase.detach` cancels whatever is in flight, so switching tool
  cannot be undone by an answer that arrives afterwards.

Threading: :class:`~tda.models.sam_service.SamQueue` invokes its callback on the
worker thread.  :class:`SamResultBridge` is the only thing that touches it
there; it re-emits the payload through a queued signal so the mask is applied
on the GUI thread.
"""
from __future__ import annotations

import logging
import time

from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal

from tda.models.sam_service import SamRequest, SamResult
from tda.ui.canvas.overlay import LabelOverlay
from tda.ui.canvas.sam_crop import (
    MAX_SAM_SIDE,
    SamResultBridge,
    norm_box,
    viewport_crop,
)
from tda.ui.canvas.sam_prompt import (  # re-exported: this was their home
    FALLBACK_INSTANCE,
    HINT_EDITED,
    CandidatesMixin,
)
from tda.ui.canvas.tools import Box, Point, Rect, Tool

log = logging.getLogger(__name__)

__all__ = [
    "MAX_SAM_SIDE",
    "HINT_EDITED",
    "ERR_FRAME_CHANGED",
    "ERR_NO_FRAME_TOKEN",
    "ERR_OUT_OF_BOUNDS",
    "FALLBACK_INSTANCE",
    "SamResultBridge",
    "SamToolBase",
    "SamPointTool",
    "SamBoxTool",
    "viewport_crop",
]


#: Emitted on :attr:`SamToolBase.sigError` for a result that arrived too late.
ERR_FRAME_CHANGED = "SAM result dropped: the frame or instance changed"
ERR_OUT_OF_BOUNDS = "SAM result dropped: the crop no longer fits the frame"
#: Emitted on :attr:`SamToolBase.sigError` when the tool has not been told which
#: frame it is on, which makes every later staleness check meaningless.
ERR_NO_FRAME_TOKEN = "frame token not set: call set_frame_token(...) on frame change"
#: Emitted when a prompt is attempted while the canvas is showing another frame.
ERR_FLASHING = "松开 Tab 再操作 / release Tab first: another frame is on screen"




class SamToolBase(CandidatesMixin, Tool):
    """Shared plumbing for the SAM prompt tools.

    Attributes:
        queue: a ``SamQueue`` (or any object with ``submit(req, cb)``); ``None``
            disables submission so the UI still works without a checkpoint.
        refine: when True the current editing mask is sent as ``mask_input`` and
            only the region near the new points changes (spec 4.6).
        instance: instance key the result belongs to; ``None`` keeps whatever
            the overlay is already editing.
        prompt_box: optional box sent along with the points, set via
            :meth:`set_prompt_box` -- normally the box of a
            :class:`~tda.core.diffmap.DiffBlob` from the frame difference map,
            or the last box dragged with :class:`SamBoxTool`.
        stroke_before: the editing layer as it was immediately before the last
            application, exactly like ``PaintTool.stroke_before``, so the
            session can build one ``edit_editing_mask`` op per applied mask --
            including per candidate switch -- without the tool knowing about the
            undo stack.

    Signals:
        sigStroke: dirty rect of an applied mask (inherited).
        sigHint: a human-readable note for the status bar, e.g. :data:`HINT_EDITED`.
        sigError: a result was dropped; the payload says why.
    """

    sigHint = Signal(str)

    def __init__(
        self,
        canvas: Any = None,
        overlay: Optional[LabelOverlay] = None,
        queue: Any = None,
        refine: bool = False,
        instance: Optional[str] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(canvas, overlay, parent)
        self.queue = queue
        self.refine = bool(refine)
        self.instance = instance
        self.last_result: Optional[SamResult] = None
        #: The frame to crop prompts from.  ``None`` means "whatever the canvas
        #: is displaying", which is only the same thing while nothing is being
        #: flashed over it; the window sets this on every frame change.
        self.image: Optional[np.ndarray] = None
        #: While true the tool accepts no prompt: the canvas is showing another
        #: frame (``Tab``), so a click on it is not about this one.
        self.paused = False
        self.prompt_box: Optional[Box] = None
        self.stroke_before: Optional[np.ndarray] = None
        self._frame_token: Any = None
        self._token = 0
        self._candidates: list[np.ndarray] = []
        self._candidate_index = 0
        self._candidate_rect: Optional[Rect] = None
        self._candidate_base: Optional[np.ndarray] = None
        self._candidate_identity: Any = None
        #: ``(frame token, instance)`` the prompt being built belongs to.
        self._prompt_identity: Any = None
        #: When the in-flight prompt was submitted, for the log line.
        self._submitted_at: float = 0.0
        self._renders: Optional[list[Optional[np.ndarray]]] = None
        self._bridge = SamResultBridge(self)
        self._bridge.sigResult.connect(
            self._on_result, Qt.ConnectionType.QueuedConnection
        )
        self._bridge.sigFailed.connect(
            self.sigError, Qt.ConnectionType.QueuedConnection
        )

    # -- lifecycle ----------------------------------------------------------
    def detach(self) -> None:
        """Stop listening to the canvas and cancel whatever was in flight.

        Switching tool must not let the previous one paint a second later, so
        the pending prompt is invalidated here rather than allowed to land on a
        canvas its tool no longer owns. Re-:meth:`attach` starts clean.
        """
        super().detach()
        self._cancel()

    def _cancel(self) -> None:
        """Invalidate the in-flight prompt, forget the state, clear the band."""
        self.reset_prompt()
        rubber_band = getattr(self.canvas, "set_rubber_band", None)
        if rubber_band is not None:
            rubber_band(None)

    def reset_prompt(self) -> None:
        """Forget everything about the prompt being built, keeping the tool armed.

        Including the request that is still **in flight**.  Forgetting the
        points but leaving the token alone meant that click, ``Esc``,
        re-activate the same instance painted 2,464 px into the layer the
        annotator had just emptied: the answer to a prompt they had discarded
        still matched the identity check and was applied.

        A prompt describes **one instance on one frame**: the points clicked so
        far, the box they are being sent with, and the candidates that came
        back.  The moment any of that stops being true -- the frame changes, the
        target instance changes, the layer is replaced from outside the tool
        (a commit, ``Esc``, an undo) -- none of it is a prompt any more.

        Subclasses extend it with their own half of the prompt (the points, the
        drag).  ``clear_points()`` used to be the only way to drop the points
        and **nothing called it**, so every mask after the first commit was a
        union over everything the annotator had clicked that session.
        """
        self._token += 1  # nothing already submitted can match again
        self._reset_candidates()
        self.prompt_box = None

    # -- frame identity -----------------------------------------------------
    @property
    def frame_token(self) -> Any:
        """The frame the tool was told it is on, or ``None`` while unset.

        There is deliberately **no fallback**. An earlier version derived one
        from ``(id(overlay), hw)``; CPython reuses the address of a dropped
        overlay for the next same-sized one often enough (measured: 200/200 for
        a drop-then-create cycle) that it silently matched for every consecutive
        scanner pair -- exactly the case the check exists for.
        """
        return self._frame_token

    def set_frame_token(self, token: Any) -> None:
        """Declare which frame the tool is on; **required** before prompting.

        ``token`` is any hashable identity -- a
        :class:`~tda.core.model.FrameKey` is the obvious choice. Call it on
        every frame change: it invalidates the pending prompt (a result stamped
        with the old identity is dropped instead of applied), drops the
        candidates and clears the prompt box, since the box came from the
        previous frame's difference map.

        ``None`` un-sets it, which disables prompting -- :meth:`_submit` then
        refuses with :data:`ERR_NO_FRAME_TOKEN` rather than issuing a request it
        could not later validate.
        """
        if token == self._frame_token:
            return
        self._frame_token = token
        # Everything collected so far is in the *previous* frame's coordinates.
        self.reset_prompt()

    def _target_instance(self) -> Optional[str]:
        """The instance key an applied mask will be written to.

        Resolved the same way in the identity stamp and in the write, fallback
        included. Reading ``overlay.editing_instance`` raw in one place and
        applying the ``FALLBACK_INSTANCE`` default in the other made the tool
        invalidate its own first result on a fresh overlay: the stamp said
        ``None``, the write said ``"editing"``, and the next check called that a
        frame change.
        """
        if self.instance is not None:
            return self.instance
        if self.overlay is None:
            return None
        return self.overlay.editing_instance or FALLBACK_INSTANCE

    def _identity(self) -> Any:
        """What a result must still match to be safe to apply."""
        if self.overlay is None:
            return None
        return (self.frame_token, self._target_instance())

    def _sync_identity(self) -> None:
        """Drop state belonging to another frame or instance, once we notice.

        :meth:`set_frame_token` covers the frame; the *instance* changes on the
        overlay or through :attr:`instance`, which the tool cannot observe, so
        the drift is detected the next time the tool is used instead.  Points,
        box and candidates all describe the previous target and none of them
        may survive it: a point on part A is not a prompt for part B.

        The identity is remembered even when no result has come back yet --
        otherwise the very sequence this exists for (click A, commit, activate
        B, click B) saw no candidates to compare against and kept A's point.
        """
        identity = self._identity()
        if self._prompt_identity is not None and self._prompt_identity != identity:
            self.reset_prompt()
        self._prompt_identity = identity

    def _fits(self, rect: Optional[Rect]) -> bool:
        """True when ``rect`` is a non-empty window inside the overlay."""
        if self.overlay is None or rect is None:
            return False
        h, w = self.overlay.hw
        x0, y0, x1, y1 = rect
        return 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h

    # -- prompt box ---------------------------------------------------------
    def set_prompt_box(self, box: Optional[Box]) -> None:
        """Attach (or clear with ``None``) a box prompt sent with every click.

        The model comparison (``experiments/sam_compare/REPORT.md``) measured a
        lone point at IoU 0.24 on parts above 20k px but point+box at 0.76, so
        the reverse-order flow feeds the changed region's box in here and lets
        the annotator's click say *which* part inside it.
        """
        self.prompt_box = (
            None
            if box is None
            else (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        )

    # -- submission ---------------------------------------------------------
    def _submit(self, points: Sequence[Point], box: Optional[Box] = None) -> None:
        if self.queue is None or self.canvas is None or self.overlay is None:
            return
        if self.paused:
            # The canvas is showing another frame; the crop would be of that one
            # and the mask would be written to this one.
            self.sigError.emit(ERR_FLASHING)
            return
        if self._frame_token is None:
            # Without an identity a late result could not be told apart from a
            # fresh one, so refuse to create one rather than accept it blindly.
            self.sigError.emit(ERR_NO_FRAME_TOKEN)
            return
        prepared = viewport_crop(self.canvas, image=self.image)
        if prepared is None:
            return
        crop, rect, scale = prepared
        x0, y0, x1, y1 = rect

        crop_points: list[Point] = [
            ((px - x0) * scale, (py - y0) * scale, int(label))
            for px, py, label in points
            if x0 <= px < x1 and y0 <= py < y1
        ]
        crop_box: Optional[Box] = None
        if box is not None:
            bx0 = (min(max(box[0], x0), x1) - x0) * scale
            by0 = (min(max(box[1], y0), y1) - y0) * scale
            bx1 = (min(max(box[2], x0), x1) - x0) * scale
            by1 = (min(max(box[3], y0), y1) - y0) * scale
            # A box the viewport clipped away to nothing is not a prompt: SAM
            # would return an empty mask, so fall back to the point alone.
            if bx1 - bx0 >= 1.0 and by1 - by0 >= 1.0:
                crop_box = (bx0, by0, bx1, by1)
        if not crop_points and crop_box is None:
            return

        mask_input = self._mask_input(rect, crop.shape[:2])
        req = SamRequest(
            image_crop=crop,
            points=crop_points,
            box=crop_box,
            mask_input=mask_input,
            # Every prompt without a prior mask is ambiguous -- "part or whole
            # assembly?" is the question SAM cannot answer from geometry -- so
            # ask for the three candidates and let ``C`` walk them.  Asking for
            # one whenever a box was given measured badly on real frames: a
            # correct but large box around the motherboard came back as the
            # whole chassis (627k px) with no second answer to fall back on.
            # A refinement is different: it already has the shape to improve,
            # and only the blended result is a valid edit.
            multimask=mask_input is None,
        )
        self._reset_candidates()
        self._token += 1
        self._submitted_at = time.perf_counter()
        log.info("sam prompt instance=%s points=%d box=%s refine=%s multimask=%s",
                 self._target_instance(), len(crop_points), crop_box is not None,
                 bool(self.refine), bool(req.multimask))
        stamp = (self._token, self._identity())
        bridge, refine = self._bridge, self.refine
        # on_error matters as much as the callback: without it a failed
        # inference (out of memory, a malformed prompt) leaves the annotator
        # waiting for a mask that is never coming, with nothing on screen.
        self._submit_to_queue(
            req,
            lambda res: bridge.deliver((res, rect, refine, stamp)),
            lambda exc: bridge.deliver_error(f"SAM failed: {exc}"),
        )

    def _submit_to_queue(self, req, callback, on_error) -> None:
        """``queue.submit`` with the error hook, for queues that accept one."""
        try:
            self.queue.submit(req, callback, on_error)
        except TypeError:  # an older queue (or a stub) without the hook
            self.queue.submit(req, callback)

    def _mask_input(
        self, rect: Rect, crop_hw: tuple[int, int]
    ) -> Optional[np.ndarray]:
        """The editing mask cropped to ``rect``, or ``None`` outside refine mode."""
        if not self.refine or self.overlay is None:
            return None
        x0, y0, x1, y1 = rect
        prior = self.overlay.editing[y0:y1, x0:x1]
        if not prior.any():
            return None
        if prior.shape != tuple(crop_hw):
            prior = cv2.resize(
                prior.astype(np.uint8),
                (crop_hw[1], crop_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return np.ascontiguousarray(prior, dtype=bool)

    # -- result -------------------------------------------------------------
    def _on_result(self, payload: object) -> None:
        """Accept or drop a SAM result, then apply its best mask (GUI thread).

        Every rejection path returns quietly instead of raising: this runs as a
        Qt slot, where an exception would escape into the event loop.
        """
        result, rect, refine, stamp = payload  # type: ignore[misc]
        token, identity = stamp
        if token != self._token:
            # Superseded.  By a newer prompt -- nothing to report, the annotator
            # is already looking at what they asked for -- or by the prompt
            # being cancelled, which for a frame or instance change is worth a
            # line: they clicked and the answer went nowhere.
            if identity != self._identity():
                self.sigError.emit(ERR_FRAME_CHANGED)
            return
        if self.overlay is None or identity != self._identity():
            self.sigError.emit(ERR_FRAME_CHANGED)
            return
        if not self._fits(rect):
            self.sigError.emit(ERR_OUT_OF_BOUNDS)
            return

        elapsed = ((time.perf_counter() - self._submitted_at) * 1000.0
                   if self._submitted_at else -1.0)
        log.info("sam result instance=%s candidates=%d ms=%.0f picked=1",
                 self._target_instance(),
                 len(result.candidates or [result.mask]), elapsed)
        self.last_result = result
        self._candidates = [
            np.asarray(mask).astype(bool)
            for mask in (result.candidates or [result.mask])
        ]
        self._candidate_index = 0
        self._candidate_rect = rect
        self._candidate_identity = identity
        self._renders = None
        # Outside the crop the prediction says nothing: in refine mode the prior
        # mask survives there, otherwise the layer is replaced outright. The
        # base is snapshotted once so that switching candidates re-renders from
        # the same starting point instead of compounding onto the previous one.
        self._candidate_base = self.overlay.editing.copy() if refine else None
        self._apply_candidate()
class SamPointTool(SamToolBase):
    """Point prompts: left click = positive, right click = negative.

    Points accumulate so every click refines the same proposal, and they are
    dropped the moment they stop describing one -- see
    :meth:`~SamToolBase.reset_prompt`, which the tool calls itself on a frame
    or instance change and which the window calls whenever it replaces the
    editing layer.

    A lone first click is sent with ``multimask=True`` and the three proposals
    are then reachable with :meth:`~SamToolBase.cycle_candidate`. When
    :meth:`~SamToolBase.set_prompt_box` holds a box -- the changed region from
    the difference map, or the last :class:`SamBoxTool` drag -- the click is
    sent as point+box instead, which needs no candidates.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.points: list[Point] = []

    def on_press(self, x: float, y: float, ev: Any) -> None:
        """Left click = foreground, right click **or ``Alt`` + click** = background.

        The ``Alt`` spelling exists because a right click is also how a tablet
        pen's barrel button and most trackpads raise a context menu, and because
        "hold a modifier" is one hand on a keyboard the annotator already has.
        """
        self._sync_identity()  # the box may belong to the previous target
        self.points.append((float(x), float(y), 0 if self._is_negative(ev) else 1))
        self._submit(self.points, box=self.prompt_box)

    @staticmethod
    def _is_negative(ev: Any) -> bool:
        button = getattr(ev, "button", None)
        try:
            if button is not None and button() == Qt.MouseButton.RightButton:
                return True
        except TypeError:  # pragma: no cover - a stub without a callable button
            return False
        modifiers = getattr(ev, "modifiers", None)
        try:
            return (modifiers is not None
                    and bool(modifiers() & Qt.KeyboardModifier.AltModifier))
        except TypeError:  # pragma: no cover
            return False

    def clear_points(self) -> None:
        """Forget the collected points, keeping the box and the candidates.

        The narrow half of :meth:`reset_prompt`, for a caller that wants to
        start the points again against the same box.
        """
        self.points = []

    def reset_prompt(self) -> None:
        """Also forget the points: they belong to the prompt, not to the session."""
        super().reset_prompt()
        self.clear_points()


class SamBoxTool(SamToolBase):
    """Box prompt dragged over the part; shows a rubber band while dragging."""

    #: Drags smaller than this (image px on either side) are treated as clicks.
    MIN_BOX = 2.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.box: Optional[Box] = None
        self._start: Optional[tuple[float, float]] = None
        self._dragging = False

    def reset_prompt(self) -> None:
        """Also forget the drag: a half-dragged box is in the old coordinates."""
        super().reset_prompt()
        self.box = None
        self._start = None
        self._dragging = False

    def on_press(self, x: float, y: float, ev: Any) -> None:
        self._start = (float(x), float(y))
        self._dragging = True
        self.box = None

    def on_move(self, x: float, y: float, ev: Any) -> None:
        if not self._dragging or self._start is None:
            return
        self.box = norm_box(self._start, (float(x), float(y)))
        if self.canvas is not None:
            self.canvas.set_rubber_band(self.box)

    def on_release(self, x: float, y: float, ev: Any) -> None:
        if not self._dragging or self._start is None:
            return
        self._dragging = False
        box = norm_box(self._start, (float(x), float(y)))
        self._start = None
        if self.canvas is not None:
            self.canvas.set_rubber_band(None)
        if box[2] - box[0] < self.MIN_BOX or box[3] - box[1] < self.MIN_BOX:
            self.box = None
            return
        self.box = box
        self._submit([], box=box)
