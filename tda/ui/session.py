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
from PySide6.QtCore import QObject, Signal

from tda.core import masks
from tda.core.compiler import CompiledFrame
from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy
from tda.core.truth import StaleConflictError, TruthService
from tda.core.truth_inputs import frame_hw, instances_of, state_of
from tda.ui import session_api as api
from tda.ui import session_edit as edit
from tda.ui.commands import Op, UndoStack
from tda.ui.session_images import ImageCache
from tda.ui.session_layer import EditingLayer
from tda.ui.session_ops import GEOM_BOX, ON_BENCH
from tda.ui import session_rows as rows
from tda.ui.session_queues import ReviewState

__all__ = ["AnnotationSession"]


class AnnotationSession(QObject):
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
        self._compiled: Optional[CompiledFrame] = None
        self._hidden: set[str] = set()
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
            stats = apply(self.db, self.truth, payload)
            self.review.problems.update(stats["problems"])
            self._invalidate()
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
        self.review.open(self.desktop, self.view)
        self._step = self._available[-1] if self._available else (
            self._steps[-1] if self._steps else None
        )
        self._reset_working_set()
        self._set_dirty(False)
        if self._step is not None:
            self._announce()

    def close(self) -> None:
        """Drop the working set; the database itself belongs to the caller."""
        self.save()
        self._steps = []
        self._available = []
        self._step = None
        self._reset_working_set()
        self.sigClosed.emit()

    def _reset_working_set(self) -> None:
        """Forget everything that belonged to the desktop/view being left."""
        self.images.clear()
        self.review.clear()
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
        self._invalidate()
        self._announce()

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
        """Compiler output for the current frame (cached until something changes)."""
        if self._compiled is None:
            self._compiled = self.truth.compile(self.current())
            self.review.problems[self.current().step] = list(self._compiled.problems)
        return self._compiled

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

    # ------------------------------------------------------------------ edits
    def _editable_frame(self) -> FrameKey:
        """The current frame, refusing one there is nothing to draw on.

        A step flagged ``missing``, or whose image never made it into the cache,
        has no canvas: its state and its shape anchors are real (spec 4.2
        缺帧处理) but a mask drawn "on" it would be in nobody's coordinates.
        """
        key = self.current()
        if key.step not in self._available or self.image_path(key.step) is None:
            raise ValueError(
                f"step {key.step} of {self.view} has no image: it cannot be annotated"
            )
        return key

    def begin_edit(self, instance: str) -> None:
        """Load the instance's amodal shape into the editing layer (spec 4.3)."""
        key = self._editable_frame()
        found = self.compiled().instances.get(instance)
        self.layer.begin(instance, None if found is None else found.amodal,
                         frame_hw(self.db, key))

    @property
    def editing_instance(self) -> Optional[str]:
        """The instance being drawn, or ``None``."""
        return self.layer.instance

    def editing_mask(self) -> Optional[np.ndarray]:
        """The editing layer as the session last saw it, or ``None``.

        The array belongs to the session: the window paints into its own overlay
        buffer and hands the result over with :meth:`set_editing_mask`.
        """
        return self.layer.mask()

    def set_editing_mask(self, mask: np.ndarray) -> None:
        """Take a copy of the layer the window has been painting into."""
        self.layer.set(mask)

    def push_stroke(self, before: np.ndarray, after: np.ndarray) -> None:
        """Record one brush/eraser stroke on the undo stack (spec 4.6).

        The pixels are already painted, so the op is logged rather than applied;
        undoing it hands the earlier mask back on :attr:`sigEditingChanged`.
        """
        self.undo_stack.push(self.layer.stroke_op(before, after), apply=False)
        self._refresh_dirty()

    def clear_edit(self) -> None:
        """Drop the editing layer without writing anything."""
        self.layer.clear()

    def commit_edit(self, scope: str, direction: str = edit.REVERSE) -> dict:
        """Write the editing layer back with the scope the annotator chose.

        ``scope`` is one of :data:`tda.ui.session_api.COMMIT_SCOPES`, or one of
        the two layering answers :func:`tda.ui.session_edit.suggest_scope` gives
        -- ``zorder:above:<B>`` / ``zorder:below:<B>`` -- which write a
        ``PairOverride`` instead of pixels (spec 4.3 改层级).

        A pixel scope with nothing changed is a no-op: loading a shape and
        pressing Enter must not mint a new version of it.  An explicit layering
        scope is always honoured, because there the pixels are not the point.
        """
        if not self.layer.active:
            raise RuntimeError("commit_edit() needs begin_edit() first")
        key, instance = self._editable_frame(), self.layer.instance
        pair = edit.split_zorder_scope(scope)
        if pair is None and not self.layer.changed():
            return {"changed": False, "affected": [], "conflicts": 0, "problems": {},
                    "scope": scope}
        if pair is not None:
            other, above = pair
            result = edit.commit_pair_override(
                self.db, self.truth, key,
                instance if above else other, other if above else instance,
                self.annotator, known=self._known_instances(),
            )
        else:
            self._refuse_mask_on_bench(key, instance)
            result = edit.commit_edit(self.db, self.truth, key, instance, self.layer.mask(),
                                      scope, direction, self.annotator)
        return self._after_edit(result)

    def _known_instances(self) -> set[str]:
        """The instances this frame has, which a layering gesture may name."""
        return set(self.compiled().instances)

    def _refuse_mask_on_bench(self, key: FrameKey, instance: str) -> None:
        """A part on the bench is tracked by a rectangle, not by a mask (spec 4.2)."""
        placement = edit.placement_of(self.db, self.tax, key, instance)
        if placement == ON_BENCH:
            raise ValueError(
                f"{instance} is on the bench at step {key.step}: use the bench box tool"
            )

    def commit_box(self, instance: str, box, direction: str = edit.REVERSE) -> dict:
        """Draw the staging-area rectangle of a part on the bench (spec 4.2 S4)."""
        result = edit.commit_box(self.db, self.truth, self._editable_frame(), instance, box,
                                 direction=direction, annotator=self.annotator)
        return self._after_edit(result)

    def commit_occluder(self, mask: np.ndarray, occluder_type: str = "hand") -> dict:
        """Store one occluder layer of the current frame (spec 4.2 step 4)."""
        result = edit.commit_occluder(self.db, self.truth, self._editable_frame(), mask,
                                      occluder_type, self.annotator)
        return self._after_edit(result)

    def preview(self, scope: str, direction: str = edit.REVERSE) -> dict:
        """How far the pending edit would reach, before it is written (spec 4.3).

        ``{"steps", "verified_steps"}`` -- the frames the scope would change and
        the already-confirmed ones among them, i.e. the "影响 N 帧 / 将产生 N 个
        冲突" strip.  Nothing is written and no pixels are touched.
        """
        if self.editing_instance is None:
            return {"steps": [], "verified_steps": []}
        geom = GEOM_BOX if edit.placement_of(
            self.db, self.tax, self.current(), self.editing_instance
        ) == ON_BENCH else None
        return edit.preview(self.db, self.truth, self.current(), self.editing_instance,
                            scope, direction, **({"geom_type": geom} if geom else {}))

    def set_visibility(self, instance: str, vis: str) -> None:
        """Override one instance's visibility label on this frame (spec 6.2)."""
        self._after_edit(
            edit.set_visibility(self.db, self.truth, self.current(), instance, vis,
                                self.annotator)
        )

    def set_hidden(self, instance: str, hidden: bool) -> None:
        """Show or hide an instance in the canvas.

        A view setting, not data: it is never written to the database and never
        logged, and it is forgotten when another desktop/view is opened.
        """
        if hidden:
            self._hidden.add(instance)
        else:
            self._hidden.discard(instance)

    def set_zorder_move(self, instance: str, above_of: str) -> None:
        """Move ``instance`` directly above ``above_of`` in the layer order."""
        self._after_edit(
            edit.set_zorder_move(self.db, self.truth, self.current(), instance, above_of,
                                 self.annotator, known=self._known_instances())
        )

    def suggest_scope(self, edited: Optional[np.ndarray] = None) -> str:
        """The scope the pending edit would default to (spec 4.3).

        Answered from the pixels that *changed* since :meth:`begin_edit`, so
        opening a shape and touching nothing suggests nothing.
        """
        if not self.layer.active:
            return api.SCOPE_KEYFRAME
        mask = self.layer.mask() if edited is None else edited
        return edit.suggest_scope(self.compiled(), self.layer.instance,
                                  self.layer.before(), mask)

    def _after_edit(self, result: dict) -> dict:
        """Record the op, refresh the caches and tell the panels (spec 4.6)."""
        op = result.get("op")
        if isinstance(op, Op):
            self.undo_stack.push(op, apply=False)
        self.review.problems.update(result.get("problems") or {})
        self._invalidate()
        self._refresh_dirty()
        self._announce()
        return result

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

    def refresh_all(self) -> dict:
        """Recompile every frame of the open view (spec 3.4).

        The queues of :meth:`queues` are fed by whatever has been compiled so
        far, so this is what a session calls to fill them in one sweep -- after
        a bulk import, or when the review panel is opened on a view nobody has
        visited in this session.
        """
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
        """Forget what depends on the annotation inputs of the current frame."""
        self._compiled = None
        self.review.invalidate()

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
