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

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Signal

from tda.core import masks
from tda.core.compiler import CompiledFrame
from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy
from tda.core.truth import StaleConflictError, TruthService
from tda.core.truth_inputs import instances_of, state_of
from tda.ui import session_edit as edit
from tda.ui.commands import Op, UndoStack
from tda.ui.session_commits import CommitMixin
from tda.ui.session_coverage import coverage
from tda.ui.session_images import ImageCache
from tda.ui.session_layer import EditingLayer
from tda.ui import session_rows as rows
from tda.ui.session_queues import ReviewState
from tda.ui.session_sweep import TruthSweeper

__all__ = ["COMPILED_CACHE_SIZE", "AnnotationSession"]

#: Compiled frames kept in memory. Four covers the Tab compare between
#: k and k-1 plus the prefetched next frame, at a few MB of masks each.
COMPILED_CACHE_SIZE = 4


class AnnotationSession(CommitMixin, QObject):
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
    #: ``(done, total)`` of the background re-check of frozen frames (spec 3.4).
    sigSweepProgress = Signal(int, int)
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
        self.sweeper.sigProgress.connect(self._on_sweep_progress)
        self.sweeper.sigQueuesChanged.connect(self._on_queues_changed)
        self.sweeper.sigPrefetched.connect(self._on_prefetched)
        self.sweeper.sigPrefetchedImage.connect(self._on_prefetched_image)
        self._dirty = False

        self.layer = EditingLayer()

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
            self._invalidate()
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

    def open(self, desktop: int, view: str) -> None:
        """Open one desktop/view and stand on the frame annotation starts from.

        Every logical step with a frame row stays in :meth:`steps` -- the
        timeline has to show a gap, and the state machine and the shape anchors
        run through it either way -- but a step flagged ``missing`` has no image
        to draw on, so the starting frame is the last step that has one
        (spec 4.2, 缺帧处理).
        """
        self.desktop = int(desktop)
        self.view = str(view)
        self._steps = [int(row["step"]) for row in self.db.frames_for(self.desktop, self.view)]
        self._available = edit.annotatable_steps(self.db, self.desktop, self.view, self._steps)
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

    def close(self) -> None:
        """Drop the working set; the database itself belongs to the caller.

        The sweeper is joined here and nowhere else: it is the one place the GUI
        may wait for it, and leaving a thread writing to the database behind a
        closed session is how a half-written re-check would happen.
        """
        self.save()
        self.sweeper.stop()
        self._steps = []
        self._available = []
        self._step = None
        self._reset_working_set()
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

    def goto(self, step: int) -> None:
        """Open ``step`` of the current desktop/view.

        A step flagged ``missing`` can be opened deliberately -- the panels do
        it when the annotator clicks it in the timeline -- it is only skipped by
        :meth:`prev` and :meth:`next`.
        """
        step = int(step)
        if step not in self._steps:
            return
        self._step = step
        self.clear_edit()
        self.review.invalidate()
        self._compile_on_visit()
        self._announce()
        self._prefetch_next()

    def prev(self) -> None:
        """Go one step towards the start of the teardown (the reverse-order 前进)."""
        self._step_to([s for s in self._available if s < (self._step or 0)], last=True)

    def next(self) -> None:
        """Go one step towards the end of the teardown."""
        self._step_to([s for s in self._available if s > (self._step or 0)], last=False)

    def _step_to(self, candidates: list[int], last: bool) -> bool:
        if not candidates:
            return False
        self.goto(candidates[-1] if last else candidates[0])
        return True

    def frame_status(self, step: int) -> str:
        """One of :data:`tda.ui.session_api.FRAME_STATUSES` (spec 4.5 colours)."""
        return self.review.frame_status(step)

    # ----------------------------------------------------------------- images
    def image(self) -> Optional[np.ndarray]:
        """RGB pixels of the current frame, or ``None`` when it has no image."""
        return self.image_at(self._step) if self._step is not None else None

    def flash_compare(self) -> Optional[np.ndarray]:
        """The image of step ``k-1``, for the ``Tab`` flash compare (spec 4.5)."""
        earlier = [s for s in self._available if s < (self._step or 0)]
        return self.image_at(earlier[-1]) if earlier else None

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
    def compiled(self) -> CompiledFrame:
        """Compiler output for the current frame.

        Kept in memory per step and per edit epoch, so that stepping back and
        forth between two frames -- which is what the ``Tab`` compare does -- is
        free, and so that a frame the sweeper compiled ahead of time is used
        rather than compiled again.
        """
        key = self.current()
        hit = self._compiled.get(key.step)
        if hit is not None and hit[0] == self._epoch:
            return hit[1]
        compiled = self.truth.compile(key)
        self._keep_compiled(key.step, self._epoch, compiled)
        return compiled

    def _compile_on_visit(self) -> None:
        """Bring the frame just opened up to date in the truth table (spec 3.4).

        A commit only compiles the frame in front of the annotator, so the rows
        of every other frame it reached are stale until somebody looks at them.
        This is that moment: one frame, and only when its stored rows do not
        already carry the current inputs' hash.
        """
        key = self.current()
        if key.step not in self._available:
            return
        held = self._compiled.get(key.step)
        if held is not None and held[0] == self._epoch:
            stored = self.db.compiled(key)
            if stored and all(row["input_hash"] == held[1].input_hash
                              for row in stored.values()):
                return  # the sweeper already brought this frame up to date
        # one compilation for the whole visit: the refresh hands back the frame
        # it made its decisions from, which is the one the panels are about to
        # ask for -- compiling it twice is what made arriving cost 0.9 s
        stats = self.truth.refresh(key)
        self.review.problems[key.step] = list(stats["problems"])
        self.review.invalidate()
        self._keep_compiled(key.step, self._epoch, stats["compiled"])

    def _keep_compiled(self, step: int, epoch: int, compiled: CompiledFrame) -> None:
        """Remember one compilation, and what the truth table owes because of it."""
        self._compiled[int(step)] = (int(epoch), compiled)
        self.review.problems[int(step)] = list(compiled.problems)
        while len(self._compiled) > COMPILED_CACHE_SIZE:
            self._compiled.pop(next(iter(self._compiled)))

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
        """The instructions for stepping from the current frame back to ``k-1``."""
        key = self.current()
        return edit.task_card_for(
            self.db, self.tax, key.desktop, key.view, key.step,
            start_step=self._available[-1] if self._available else None,
        )

    def _after_edit(self, result: dict) -> dict:
        """Record the op, refresh the caches and tell the panels (spec 4.6)."""
        op = result.get("op")
        if isinstance(op, Op):
            self.undo_stack.push(op, apply=False)
        self.review.problems.update(result.get("problems") or {})
        self._invalidate()
        self._refresh_dirty()
        self._hand_to_sweeper(result.get("rechecks") or [])
        self._announce()
        return result

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

    # --------------------------------------------------------- confirm/review
    def confirm_frame(self) -> bool:
        """Freeze the frame and step back (spec 4.2 step 5).

        ``False`` means the compilation still has a blocking problem -- a
        chassis instance without a shape, a contradictory layer order; the
        problem list is emitted on :attr:`sigProblems` first, so a panel can
        show exactly what the annotator has to fix.
        """
        key = self.current()
        try:
            self.truth.verify_frame(key, self.annotator)
        except ValueError:
            self._invalidate()
            problems = list(self.compiled().problems)
            self.review.problems[key.step] = problems
            self.sigProblems.emit(problems)
            return False
        self._invalidate()
        # exactly one frame change: the step back is the change, and a panel
        # that reloads on every emit must not reload the frame being left
        if not self._step_to([s for s in self._available if s < key.step], last=True):
            self._announce()
        return True

    def _coverage(self) -> dict:
        """What has been drawn in every frame of this view, without compiling."""
        if self.desktop is None:
            return {}
        return coverage(self.db, self.tax, self.desktop, self.view, self._available)

    def refresh_all(self) -> dict:
        """Recompile every frame of the open view (spec 3.4).

        The queues of :meth:`queues` are fed by whatever has been compiled so
        far, so this is what a session calls to fill them in one sweep -- after
        a bulk import, or when the review panel is opened on a view nobody has
        visited in this session.
        """
        self.truth.run_pending_rechecks(self.desktop, self.view)
        stats = edit.refresh_steps(self.db, self.truth, self.desktop, self.view,
                                   self._available)
        self.review.problems.update(stats["problems"])
        self._invalidate()
        return stats

    def queues(self) -> dict[str, list[dict]]:
        """The four review queues of spec 4.4, keyed by :data:`QUEUE_NAMES`."""
        return self.review.queues()

    def set_unexplained(self, step: int, boxes) -> None:
        """Record the difference-map regions of one frame that nothing explains.

        The diff map itself belongs to the window (spec 4.2 step 4); the session
        only carries the result into the fourth review queue.  An empty list
        clears the step.
        """
        self.review.set_unexplained(step, boxes)

    def resolve_conflict(self, cid: int, resolution: str) -> str:
        """Settle one queued disagreement; never raises (spec 3.4, 4.4).

        Returns what happened, because all three outcomes are ordinary:

        ``"resolved"``
            the decision was applied and the frame recompiled;
        ``"superseded"``
            the inputs had moved on since the conflict was queued, so nothing
            was confirmed -- the current disagreement is in the queue instead
            and the panel has to show that one;
        ``"refused"``
            the conflict is gone, already settled, or the decision cannot be
            applied to it.

        The reason is emitted on :attr:`sigProblems` for the last two.
        """
        conflict = self.db.get_conflict(int(cid))
        outcome = "resolved"
        try:
            self.truth.resolve_conflict(int(cid), resolution, self.annotator)
        except StaleConflictError as stale:
            outcome = "superseded"
            self.sigProblems.emit([f"conflict {cid} superseded: {stale}"])
        except (KeyError, ValueError) as refused:
            outcome = "refused"
            self.sigProblems.emit([f"conflict {cid} not resolved: {refused}"])
        if conflict is not None:
            stats = edit.refresh_steps(self.db, self.truth, self.desktop, self.view,
                                       [conflict["step"]])
            self.review.problems.update(stats["problems"])
        self._invalidate()
        self.sigFrameChanged.emit(self.current())
        return outcome

    # --------------------------------------------------------------- internals
    def _invalidate(self) -> None:
        """The annotation inputs changed: everything derived from them is stale."""
        self._epoch += 1
        self._compiled.clear()
        self.review.invalidate()

    # -------------------------------------------------- the background worker
    def _on_sweep_progress(self, done: int, total: int) -> None:
        self.sigSweepProgress.emit(done, total)

    def _on_queues_changed(self) -> None:
        """A re-check finished: its verdict may have changed a status or a queue."""
        self.review.invalidate()
        self.sigQueuesChanged.emit()

    def _on_prefetched(self, step: int, epoch: int, compiled: object) -> None:
        """Adopt a frame the sweeper compiled ahead, unless it went stale."""
        if epoch == self._epoch and isinstance(compiled, CompiledFrame):
            self._keep_compiled(step, epoch, compiled)

    def _on_prefetched_image(self, step: int, rgb: object) -> None:
        """Adopt a frame the sweeper decoded ahead of the annotator reaching it."""
        key = self._key(step)
        if key is not None and isinstance(rgb, np.ndarray):
            self.images.put(key, rgb)

    def _prefetch_next(self) -> None:
        """Warm the frame the annotator is about to reach: ``k-1`` (spec 4.2).

        Reverse-order annotation always lands there next, and both halves of the
        cost -- bringing its truth rows up to date and decoding its image -- are
        done on the sweeper thread, so arriving is free.
        """
        if not self.sweeper_enabled or self._step is None:
            return
        earlier = [s for s in self._available if s < self._step]
        if not earlier:
            return
        step = earlier[-1]
        if self._compiled.get(step, (None,))[0] != self._epoch:
            self.sweeper.prefetch(step, self._epoch)

    def drain_sweeper(self, timeout: float = 30.0) -> bool:
        """Wait for the background re-checks to finish (tests, exports, quit)."""
        drained = self.sweeper.wait_idle(timeout)
        QCoreApplication.processEvents()
        self.review.invalidate()
        return drained

    #: The GUI never waits for a prefetch; a test that measures one does.
    drain_prefetch = drain_sweeper

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
