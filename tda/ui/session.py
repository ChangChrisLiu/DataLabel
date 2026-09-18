"""The annotation session: which frame is open, and what the panels see.

:class:`AnnotationSession` is the object every dock panel talks to (spec 10.1):
it owns the open desktop/view, the current logical step, the image cache, the
editing layer and the undo stack, and it implements
:class:`tda.ui.session_api.SessionLike` in full.  It holds **no widgets** -- the
panels connect to its three signals and call its methods, nothing more.

The editing rules themselves live in :mod:`tda.ui.session_edit`, which is
Qt-free: this module turns a gesture into a call there, invalidates its caches,
records the operation for undo and tells the panels what changed.

Annotation runs backwards (spec 4.2): :meth:`open` starts at the last frame
that actually has an image, "advance" means ``step - 1``, and confirming a
frame steps back rather than forward.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from tda.core import masks
from tda.core.compiler import CompiledFrame
from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import instances_of, state_of
from tda.ui import session_edit as edit
from tda.ui.commands import Op, UndoStack
from tda.ui.session_api import SessionRefusal
from tda.ui.session_commits import CommitMixin
from tda.ui.session_images import ImageCache
from tda.ui.session_layer import EditingLayer
from tda.ui import session_rows as rows
from tda.ui.session_queues import ReviewState
from tda.ui.session_review import ReviewMixin
from tda.ui.session_truth import TruthCacheMixin
from tda.ui.session_sweep import TruthSweeper

__all__ = ["AnnotationSession"]

log = logging.getLogger("tda.session")

class AnnotationSession(CommitMixin, ReviewMixin, TruthCacheMixin, QObject):
    """One annotator's working set: a desktop, a view, and the frame in hand."""

    #: The current frame changed; payload is a :class:`~tda.core.model.FrameKey`.
    sigFrameChanged = Signal(object)
    #: Unsaved changes flag.
    sigDirty = Signal(bool)
    #: Problems of the current frame, as ``list[str]``.
    sigProblems = Signal(list)
    #: The editing layer was replaced from the outside (an undo or redo of a
    #: stroke); payload is the new mask, or ``None`` when the edit was dropped.
    sigEditingChanged = Signal(object)
    #: The session let go of its desktop/view; the panels must detach.
    sigClosed = Signal()
    #: ``(done, total, failed)`` of the background re-check of frozen frames
    #: (spec 3.4); ``done`` counts successes only. A two-argument slot still
    #: works -- Qt drops the third.
    sigSweepProgress = Signal(int, int, int)
    #: ``(step, text)`` -- a background re-check failed and the frame is still
    #: pending; also emitted on :attr:`sigProblems` for panels that only watch
    #: that one.
    sigSweepError = Signal(int, str)
    #: Something the review queues show has changed.
    sigQueuesChanged = Signal()

    def __init__(self, db: Db, tax: Taxonomy, truth: TruthService, cache_dir: str,
                 annotator: str) -> None:
        super().__init__()
        self.db = db
        self.tax = tax
        self.truth = truth
        self.cache_dir = str(cache_dir)
        self.annotator = str(annotator)

        self.desktop: Optional[int] = None
        self.view: str = ""
        self._step: Optional[int] = None
        self._steps: list[int] = []
        self._available: list[int] = []

        self.images = ImageCache(db, self.cache_dir)
        self.review = ReviewState(db)
        #: Compiled frames kept in memory: step -> (edit epoch, frame).
        self._compiled: dict[int, tuple[int, CompiledFrame]] = {}
        #: Bumped by every change to the annotation inputs, which is what makes
        #: a compilation from before it (a prefetch in flight) unusable.
        self._epoch = 0
        self._hidden: set[str] = set()
        #: Set False by a test that wants to inspect the queue before it drains.
        self.sweeper_enabled = True
        self.sweeper = TruthSweeper(db.path, tax, self.cache_dir,
                                    truth.compiler_version, parent=self)
        self.sweeper.sigProgress.connect(self.sigSweepProgress)
        self.sweeper.sigError.connect(self._on_sweep_error)
        self.sweeper.sigQueuesChanged.connect(self._on_queues_changed)
        self.sweeper.sigPrefetched.connect(self._on_prefetched)
        self.sweeper.sigPrefetchedImage.connect(self._on_prefetched_image)
        self._dirty = False

        self.layer = EditingLayer()
        #: Which way the annotator is walking the teardown. Reverse is the
        #: order the work is done in (spec 4.2); forward is for repairs.
        self.browsing = edit.REVERSE

        self.undo_stack = UndoStack()
        self._saved_at = self._history_mark()
        self._register_handlers()

    # ------------------------------------------------------------------ setup
    def _register_handlers(self) -> None:
        """Bind every op kind the session logs to its apply/undo function.

        Both directions of an op are the same call with a different payload
        (see :mod:`tda.ui.session_edit`), so one function serves as ``do`` and
        as ``undo``; the canvas registers ``edit_editing_mask`` separately.
        """
        for kind, apply in (
            ("commit_keyframe", edit.apply_keyframes),
            ("set_zorder", edit.apply_zorder),
            ("set_pair_override", edit.apply_pair_override),
            ("set_frame_override", edit.apply_frame_override),
            ("set_occluder", edit.apply_occluder),
        ):
            handler = self._make_handler(apply)
            self.undo_stack.register(kind, handler, handler)
        # a brush stroke is undone in the same history, but it touches nothing
        # but the editing layer: no database write, no recompilation
        self.undo_stack.register(
            "edit_editing_mask", self._apply_stroke, self._apply_stroke
        )

    def _apply_stroke(self, payload: dict) -> None:
        """Set the editing layer to a stroke op's ``rle_after`` (spec 4.6).

        The window owns the overlay the strokes were painted into, so an undo
        has to hand the restored mask back rather than expect a shared buffer.
        """
        self.layer.apply_stroke(payload, masks.decode_rle(payload["rle_after"]))
        self.sigEditingChanged.emit(self.layer.mask())

    def _make_handler(self, apply: Callable[..., dict]) -> Callable[[dict], None]:
        """Wrap one ``apply_*`` function so the caches and the panels follow it."""
        def handler(payload: dict) -> None:
            stats = apply(self.db, self.truth, payload, self._step)
            self.review.problems.update(stats["problems"])
            self._settled(stats.get("frame"))
            self._hand_to_sweeper(stats.get("rechecks") or [])
            self._announce()

        return handler

    def _history_mark(self) -> tuple:
        """A cheap identity of the undo history's current position.

        The length alone is not enough: undoing one edit and making a different
        one leaves the stack the same size, and the document is then *not* the
        one that was saved.
        """
        ops = self.undo_stack.ops
        return (len(ops), id(ops[-1]) if ops else None)

    def open(self, desktop: int, view: str, *, force: bool = False) -> None:
        """Open one desktop/view and stand on the frame annotation starts from.

        Every logical step with a frame row stays in :meth:`steps` -- the
        timeline has to show a gap, and the state machine and the shape anchors
        run through it either way -- but a step flagged ``missing`` has no image
        to draw on, so the starting frame is the last step that has one
        (spec 4.2, 缺帧处理).

        Raises :class:`~tda.ui.session_api.SessionRefusal` on an uncommitted
        editing layer, unless ``force``.
        """
        self._refuse_if_editing("opening another desktop or view", force)
        self.desktop = int(desktop)
        self.view = str(view)
        self._steps = [int(row["step"]) for row in self.db.frames_for(self.desktop, self.view)]
        self._available = edit.annotatable_steps(self.db, self.desktop, self.view, self._steps)
        self.browsing = edit.REVERSE  # every view is annotated backwards first
        self.review.open(self.desktop, self.view, self._coverage)
        self._step = self._available[-1] if self._available else (
            self._steps[-1] if self._steps else None
        )
        self._reset_working_set()
        self._set_dirty(False)
        if self.sweeper_enabled:
            self.sweeper.open(self.desktop, self.view)
            # a re-check the last session did not get to is still owed
            self.sweeper.enqueue(self.db.rechecks(self.desktop, self.view))
        if self._step is not None:
            self._announce()

    def close(self, *, force: bool = False) -> None:
        """Drop the working set; the database itself belongs to the caller.

        The sweeper is joined here and nowhere else: it is the one place the GUI
        may wait for it, and leaving a thread writing to the database behind a
        closed session is how a half-written re-check would happen.

        Raises :class:`~tda.ui.session_api.SessionRefusal` on an uncommitted
        editing layer, unless ``force`` -- which is what the window passes once
        its close dialog has been answered.
        """
        self._refuse_if_editing("closing the session", force)
        self.save()
        if not self.sweeper.stop():
            log.error("closing the session left the truth sweeper running on %s/%s",
                      self.desktop, self.view)
        self._steps = []
        self._available = []
        self._step = None
        self._reset_working_set()
        self.desktop = None
        self.view = ""
        self.review.close()
        self.sigClosed.emit()

    def _reset_working_set(self) -> None:
        """Forget everything that belonged to the desktop/view being left."""
        self.images.clear()
        self.review.clear()
        self._compiled.clear()
        self._hidden.clear()
        self.clear_edit()
        self.undo_stack.clear()
        self._saved_at = self._history_mark()
        self._invalidate()

    def save(self) -> None:
        """Mark the session clean.

        Every edit is committed by :mod:`tda.core.db` as it happens, so there is
        nothing to flush; what ``save`` does is remember *where in the undo
        history* the annotator last called the work done, so that undoing back
        to that point clears the flag again.
        """
        self._saved_at = self._history_mark()
        self._set_dirty(False)

    # ----------------------------------------------------------------- frames
    @property
    def is_open(self) -> bool:
        """Is a desktop/view open? The window asks before touching anything else."""
        return self.desktop is not None and self._step is not None

    def steps(self) -> list[int]:
        """Logical steps of the open desktop/view, ascending."""
        return list(self._steps)

    def current(self) -> FrameKey:
        """The frame being annotated."""
        if self.desktop is None or self._step is None:
            raise RuntimeError("no frame is open; call open() first")
        return FrameKey(self.desktop, self._step, self.view)

    def _refuse_if_editing(self, what: str, force: bool) -> None:
        """Stop a move that would drop an editing layer nobody committed.

        The window guards every gesture it owns, but it is not the only caller
        and a new one is one line away: this is the backstop, so that losing an
        uncommitted layer takes an explicit ``force=True`` rather than a
        forgotten check.  ``force`` is what the window passes once the annotator
        has answered -- committed, discarded, or told the close dialog to.
        """
        if force or not self.layer.changed():
            return
        raise SessionRefusal(
            f"uncommitted edit on {self.layer.instance}: commit it or clear it "
            f"before {what}"
        )

    def goto(self, step: int, *, force: bool = False) -> None:
        """Open ``step`` of the current desktop/view.

        A step flagged ``missing`` can be opened deliberately -- the panels do
        it when the annotator clicks it in the timeline -- it is only skipped by
        :meth:`prev` and :meth:`next`.

        Raises :class:`~tda.ui.session_api.SessionRefusal` when the editing
        layer holds uncommitted pixels, unless ``force``.
        """
        step = int(step)
        if step not in self._steps:
            return
        self._refuse_if_editing(f"leaving step {self._step}", force)
        self._step = step
        self.clear_edit()
        self.review.invalidate()
        self._compile_on_visit()
        self._announce()
        self._prefetch_next()

    def prev(self, *, force: bool = False) -> None:
        """Go one step towards the start of the teardown (the reverse-order 前进)."""
        self._step_to([s for s in self._available if s < (self._step or 0)],
                      last=True, force=force)

    def next(self, *, force: bool = False) -> None:
        """Go one step towards the end of the teardown."""
        self._step_to([s for s in self._available if s > (self._step or 0)],
                      last=False, force=force)

    def _step_to(self, candidates: list[int], last: bool, force: bool = False) -> bool:
        if not candidates:
            return False
        self.goto(candidates[-1] if last else candidates[0], force=force)
        return True

    def frame_status(self, step: int) -> str:
        """One of :data:`tda.ui.session_api.FRAME_STATUSES` (spec 4.5 colours)."""
        return self.review.frame_status(step)

    # ----------------------------------------------------------------- images
    def image(self) -> Optional[np.ndarray]:
        """RGB pixels of the current frame, or ``None`` when it has no image."""
        return self.image_at(self._step) if self._step is not None else None

    def flash_compare(self, other: bool = False) -> Optional[np.ndarray]:
        """The image the ``Tab`` flash compares this frame against (spec 4.5).

        By default the frame the task card is about -- :meth:`task_neighbour` --
        so that flashing shows exactly the difference the card describes.
        ``other=True`` gives the neighbour on the far side, for a look at where
        the teardown is going rather than where it came from.
        """
        direction = self.browsing
        if other:
            direction = edit.FORWARD if direction == edit.REVERSE else edit.REVERSE
        step = self._neighbour(direction)
        return None if step is None else self.image_at(step)

    def image_at(self, step: int) -> Optional[np.ndarray]:
        """One step's image as RGB, decoded at most once (see :mod:`session_images`)."""
        key = self._key(step)
        return None if key is None else self.images.get(key)

    @property
    def image_budget_bytes(self) -> int:
        """Memory the decoded-image cache may use."""
        return self.images.budget_bytes

    @image_budget_bytes.setter
    def image_budget_bytes(self, value: int) -> None:
        self.images.budget_bytes = int(value)
        self.images._trim()

    @property
    def image_cache(self) -> dict:
        """The decoded images held right now (read-only; tests and reporting)."""
        return self.images.as_dict()

    def image_path(self, step: int) -> Optional[str]:
        """The full-resolution cached image of one step, or ``None``."""
        key = self._key(step)
        return None if key is None else self.images.image_path(key)

    def thumb_path(self, step: int) -> Optional[str]:
        """The timeline thumbnail of one step, falling back to the full image."""
        key = self._key(step)
        return None if key is None else self.images.thumb_path(key)

    def _key(self, step: Optional[int]) -> Optional[FrameKey]:
        """``FrameKey`` for a step of the open view, or ``None`` when none is open."""
        if self.desktop is None or step is None:
            return None
        return FrameKey(self.desktop, int(step), self.view)

    # ---------------------------------------------------------------- content
    def instance_rows(self) -> list[dict]:
        """One row per instance of the current frame, top-most layer first."""
        key = self.current()
        return rows.instance_rows(
            self.compiled(),
            state_of(self.db, self.tax, key.desktop, key.step),
            instances_of(self.db, key.desktop),
            self._hidden,
        )

    def overlay_layers(self) -> tuple[dict[str, np.ndarray], list[str]]:
        """Visible masks and bottom-up paint order for the canvas overlay."""
        return rows.overlay_layers(self.compiled(), self._hidden)

    def task_card(self) -> list[dict]:
        """What has to be annotated on the frame in front of the annotator.

        The card describes *this* frame, diffed against the neighbour the
        annotator came from (:meth:`task_neighbour`): in reverse order that is
        ``k+1``, which is already done, so the card asks for the parts that are
        back in the machine on the image being looked at.
        """
        key = self.current()
        return edit.task_card_for(self.db, self.tax, key.desktop, key.view, key.step,
                                  neighbour=self.task_neighbour(), span=self.task_span())

    def task_neighbour(self) -> Optional[int]:
        """The annotated frame :meth:`task_card` is diffed against, or ``None``.

        The frame the annotator came from: the next available step in the
        browsing direction, skipping one with no image.  ``None`` on the start
        frame, which has nobody behind it and whose card is "draw everything".
        The window uses the same frame for the difference map and for the
        ``Tab`` flash, so that all three agree on what is being compared.
        """
        return self._neighbour(self.browsing)

    def task_span(self) -> list[int]:
        """The action steps the card covers, ascending.

        Usually one -- the neighbour's -- but a missing frame is skipped rather
        than annotated, so the card can describe two steps of work at once and
        the window has to be able to say which.
        """
        if self._step is None:
            return []
        neighbour = self.task_neighbour()
        if neighbour is None:
            return []
        low, high = sorted((self._step, neighbour))
        return [s for s in self._steps if low < s <= high]

    def _neighbour(self, direction: str) -> Optional[int]:
        if self._step is None:
            return None
        if direction == edit.REVERSE:
            later = [s for s in self._available if s > self._step]
            return later[0] if later else None
        earlier = [s for s in self._available if s < self._step]
        return earlier[-1] if earlier else None

    def browse_reverse(self) -> None:
        """Walk the teardown backwards, which is the order it is annotated in."""
        self.browsing = edit.REVERSE

    def browse_forward(self) -> None:
        """Walk it forwards, to repair a frame that was already done."""
        self.browsing = edit.FORWARD

    def _after_edit(self, result: dict) -> dict:
        """Record the op, refresh the caches and tell the panels (spec 4.6)."""
        op = result.get("op")
        if isinstance(op, Op):
            self.undo_stack.push(op, apply=False)
        self.review.problems.update(result.get("problems") or {})
        self._settled(result.get("frame"))
        self._refresh_dirty()
        self._hand_to_sweeper(result.get("rechecks") or [])
        self._announce()
        return result

    def _settled(self, frame) -> None:
        """Move on to the next edit epoch, keeping the frame the write produced.

        The refresh already compiled the current step; invalidating and letting
        :meth:`compiled` build it again was a second full compilation of the
        same frame on the GUI thread -- a third of the cost of a commit.
        """
        self._invalidate()
        if isinstance(frame, CompiledFrame) and frame.key.step == self._step:
            self._keep_compiled(frame.key.step, self._epoch, frame)

    def _hand_to_sweeper(self, steps) -> None:
        """Let the background worker know which frozen frames are owed a check.

        The request is already in the database, so dropping it here delays the
        verdict but never loses it: :meth:`open` re-queues whatever is left.
        """
        if steps and self.sweeper_enabled:
            self.sweeper.enqueue(steps)

    # ------------------------------------------------------------------- undo
    def undo(self) -> bool:
        """Take the newest operation back; ``False`` when there is none."""
        return self._replay(self.undo_stack.undo())

    def redo(self) -> bool:
        """Re-apply the most recently undone operation."""
        return self._replay(self.undo_stack.redo())

    def _replay(self, op: Optional[Op]) -> bool:
        """Finish an undo/redo: the handler did the work, this reports it."""
        if op is None:
            return False
        self._refresh_dirty()
        if op.kind == "edit_editing_mask":
            self._announce()  # no database change, but the panels still repaint
        return True

    # --------------------------------------------------------------- internals

    def _announce(self) -> None:
        """Tell the panels which frame is open and what is wrong with it.

        The two always travel together: a panel that shows problems has no other
        way to learn that the frame it is showing them for has changed.
        """
        self.sigFrameChanged.emit(self.current())
        self.sigProblems.emit(self._current_problems())

    def _current_problems(self) -> list[str]:
        """The compiler's problems for the open frame, or none when it has no image."""
        key = self.current()
        if key.step not in self._available or self.image_path(key.step) is None:
            return []
        return list(self.compiled().problems)

    def _refresh_dirty(self) -> None:
        """Set the unsaved-changes flag from where the history now stands."""
        self._set_dirty(self._history_mark() != self._saved_at)

    def _set_dirty(self, value: bool) -> None:
        if bool(value) != self._dirty:
            self._dirty = bool(value)
            self.sigDirty.emit(self._dirty)

    @property
    def dirty(self) -> bool:
        """Whether anything has been edited since the last :meth:`save`."""
        return self._dirty
