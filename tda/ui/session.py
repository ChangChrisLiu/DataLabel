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

from collections import OrderedDict
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from tda.core import masks
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.compiler import CompiledFrame
from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import frame_hw, instances_of, pose_segment_of, state_of
from tda.ui import session_api as api
from tda.ui import session_edit as edit
from tda.ui.commands import Op, UndoStack

__all__ = ["AnnotationSession", "IMAGE_CACHE_SIZE"]

#: How many decoded frames the session keeps in memory.  Eight covers the
#: ``Tab`` flash between k and k-1 plus a little browsing without holding a
#: whole desktop's worth of 12 MP images.
IMAGE_CACHE_SIZE = 8

_MISSING_SHAPE = "missing_shape:"
_VERIFIED = "verified"
_NEEDS_REVIEW = "needs_review"
_ZORDER_SCOPE = "zorder:"


class AnnotationSession(QObject):
    """One annotator's working set: a desktop, a view, and the frame in hand."""

    #: The current frame changed; payload is a :class:`~tda.core.model.FrameKey`.
    sigFrameChanged = Signal(object)
    #: Unsaved changes flag.
    sigDirty = Signal(bool)
    #: Problems of the current frame, as ``list[str]``.
    sigProblems = Signal(list)

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

        self._images: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()
        self._compiled: Optional[CompiledFrame] = None
        #: step -> the compiler problems of its last refresh (spec 4.4 queues).
        self._problems: dict[int, list[str]] = {}
        self._hidden: set[str] = set()
        self._dirty = False

        self.editing_instance: Optional[str] = None
        self._editing: Optional[np.ndarray] = None

        self.undo_stack = UndoStack()
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
        """Set the editing layer to a stroke op's ``rle_after`` (spec 4.6)."""
        self.editing_instance = payload["instance"]
        self._editing = masks.decode_rle(payload["rle_after"])

    def _make_handler(self, apply: Callable[..., dict]) -> Callable[[dict], None]:
        """Wrap one ``apply_*`` function so the caches and the panels follow it."""
        def handler(payload: dict) -> None:
            stats = apply(self.db, self.truth, payload)
            self._problems.update(stats["problems"])
            self._invalidate()
            self.sigFrameChanged.emit(self.current())

        return handler

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
        self._step = self._available[-1] if self._available else (
            self._steps[-1] if self._steps else None
        )
        self._images.clear()
        self._problems.clear()
        self._hidden.clear()
        self.undo_stack.clear()
        self._invalidate()
        self._set_dirty(False)
        if self._step is not None:
            self.sigFrameChanged.emit(self.current())

    def close(self) -> None:
        """Drop the working set; the database itself belongs to the caller."""
        self.save()
        self._steps = []
        self._available = []
        self._step = None
        self._images.clear()
        self._problems.clear()
        self._hidden.clear()
        self.undo_stack.clear()
        self._invalidate()

    def save(self) -> None:
        """Mark the session clean.

        Every edit is committed by :mod:`tda.core.db` as it happens, so there is
        nothing to flush; what ``save`` does is clear the "unsaved changes"
        flag the window title and the close prompt watch.
        """
        self._set_dirty(False)

    # ----------------------------------------------------------------- frames
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
        self._invalidate()
        self.sigFrameChanged.emit(self.current())

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
        if self.desktop is None:
            return api.STATUS_UNLABELED
        key = FrameKey(self.desktop, int(step), self.view)
        row = self.db.get_frame(key)
        if row is None or row.get("missing"):
            return api.STATUS_MISSING
        if any(c["step"] == int(step) for c in self._open_conflicts()):
            return api.STATUS_CONFLICT
        status = row.get("review_status")
        if status == _NEEDS_REVIEW:
            return api.STATUS_NEEDS_REVIEW
        if status == _VERIFIED:
            return api.STATUS_VERIFIED
        return api.STATUS_AUTO if self.db.compiled(key) else api.STATUS_UNLABELED

    # ----------------------------------------------------------------- images
    def image(self) -> Optional[np.ndarray]:
        """RGB pixels of the current frame, or ``None`` when it has no image."""
        return self.image_at(self._step) if self._step is not None else None

    def flash_compare(self) -> Optional[np.ndarray]:
        """The image of step ``k-1``, for the ``Tab`` flash compare (spec 4.5)."""
        earlier = [s for s in self._available if s < (self._step or 0)]
        return self.image_at(earlier[-1]) if earlier else None

    def image_at(self, step: int) -> Optional[np.ndarray]:
        """One step's cached image as RGB, LRU-cached for :data:`IMAGE_CACHE_SIZE`."""
        slot = (self.view, int(step))
        hit = self._images.get(slot)
        if hit is not None:
            self._images.move_to_end(slot)
            return hit
        path = self.thumb_path(step)
        if path is None:
            return None
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._images[slot] = rgb
        while len(self._images) > IMAGE_CACHE_SIZE:
            self._images.popitem(last=False)
        return rgb

    @property
    def image_cache(self) -> dict:
        """The LRU itself (read-only; for tests and memory reporting)."""
        return dict(self._images)

    def thumb_path(self, step: int) -> Optional[str]:
        """The cached image file of one step, or ``None`` when it is not there.

        The cache layout is :func:`tda.core.cache.cache_path`'s; the frame row's
        own path is the fallback for a database whose images were never copied
        into the local cache.
        """
        if self.desktop is None:
            return None
        key = FrameKey(self.desktop, int(step), self.view)
        candidates = [cache_path(self.cache_dir, key, VIEW_EXT.get(self.view, "png"))]
        row = self.db.get_frame(key) or {}
        for candidate in ((row.get("aux") or {}).get("cache_path"), row.get("path")):
            if candidate:
                candidates.append(str(candidate))
        for candidate in candidates:
            try:
                with open(candidate, "rb"):
                    return candidate
            except OSError:
                continue
        return None

    # ---------------------------------------------------------------- content
    def compiled(self) -> CompiledFrame:
        """Compiler output for the current frame (cached until something changes)."""
        if self._compiled is None:
            self._compiled = self.truth.compile(self.current())
            self._problems[self.current().step] = list(self._compiled.problems)
        return self._compiled

    def instance_rows(self) -> list[dict]:
        """One row per instance of the current frame, top-most layer first."""
        key = self.current()
        compiled = self.compiled()
        state = state_of(self.db, self.tax, key.desktop, key.step)
        instances = instances_of(self.db, key.desktop)
        bottom_up = self._layer_order(key, set(compiled.instances))
        rows = []
        for z, instance in enumerate(bottom_up):
            inst = compiled.instances[instance]
            rec = instances.get(instance)
            held = state.get(instance)
            rows.append({
                "key": instance,
                "cls": "" if rec is None else rec.cls,
                "state": "" if held is None else held.state,
                "placement": inst.placement,
                "visibility": inst.visibility,
                "z": z,
                "hidden": instance in self._hidden,
            })
        rows.reverse()
        return rows

    def _layer_order(self, key: FrameKey, present: set[str]) -> list[str]:
        """The frame's instances bottom-to-top; unordered ones sit on top."""
        seg = pose_segment_of(self.db, key)
        ordered: list[str] = []
        for instance, _part in self.db.zorder(key.desktop, key.view, seg).order:
            if instance in present and instance not in ordered:
                ordered.append(instance)
        return ordered + sorted(present - set(ordered))

    def task_card(self) -> list[dict]:
        """The instructions for stepping from the current frame back to ``k-1``."""
        key = self.current()
        return edit.task_card_for(
            self.db, self.tax, key.desktop, key.view, key.step,
            start_step=self._available[-1] if self._available else None,
        )

    # ------------------------------------------------------------------ edits
    def begin_edit(self, instance: str) -> None:
        """Load the instance's amodal shape into the editing layer (spec 4.3)."""
        key = self.current()
        hw = frame_hw(self.db, key)
        found = self.compiled().instances.get(instance)
        amodal = None if found is None else found.amodal
        self._editing = (np.zeros(hw, dtype=bool) if amodal is None
                         else np.array(amodal, dtype=bool, copy=True))
        self.editing_instance = instance

    def editing_mask(self) -> Optional[np.ndarray]:
        """The editing layer itself; the canvas paints into it in place."""
        return self._editing

    def set_editing_mask(self, mask: np.ndarray) -> None:
        """Replace the editing layer (what a SAM proposal or an undo does)."""
        self._editing = np.asarray(mask, dtype=bool)

    def clear_edit(self) -> None:
        """Drop the editing layer without writing anything."""
        self._editing = None
        self.editing_instance = None

    def commit_edit(self, scope: str, direction: str = edit.REVERSE) -> dict:
        """Write the editing layer back with the scope the annotator chose.

        ``scope`` is one of :data:`tda.ui.session_api.COMMIT_SCOPES`, or
        ``"zorder:<other instance>"`` -- what
        :func:`tda.ui.session_edit.suggest_scope` proposes when the edited
        pixels fall inside another instance's shape, which is a statement about
        layering rather than about the silhouette (spec 4.3).
        """
        if self.editing_instance is None or self._editing is None:
            raise RuntimeError("commit_edit() needs begin_edit() first")
        key, instance = self.current(), self.editing_instance
        if scope.startswith(_ZORDER_SCOPE):
            result = edit.commit_pair_override(
                self.db, self.truth, key, instance, scope[len(_ZORDER_SCOPE):],
                self.annotator,
            )
        else:
            result = edit.commit_edit(self.db, self.truth, key, instance, self._editing,
                                      scope, direction, self.annotator)
        return self._after_edit(result)

    def commit_box(self, instance: str, box, direction: str = edit.REVERSE) -> dict:
        """Draw the staging-area rectangle of a part on the bench (spec 4.2 S4)."""
        result = edit.commit_box(self.db, self.truth, self.current(), instance, box,
                                 direction=direction, annotator=self.annotator)
        return self._after_edit(result)

    def commit_occluder(self, mask: np.ndarray, occluder_type: str = "hand") -> dict:
        """Store one occluder layer of the current frame (spec 4.2 step 4)."""
        result = edit.commit_occluder(self.db, self.truth, self.current(), mask,
                                      occluder_type, self.annotator)
        return self._after_edit(result)

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
                                 self.annotator)
        )

    def suggest_scope(self, edited: Optional[np.ndarray] = None) -> str:
        """The scope the current edit would default to (spec 4.3)."""
        if self.editing_instance is None:
            return api.SCOPE_KEYFRAME
        mask = self._editing if edited is None else edited
        if mask is None:
            return api.SCOPE_KEYFRAME
        return edit.suggest_scope(self.compiled(), self.editing_instance, mask)

    def _after_edit(self, result: dict) -> dict:
        """Record the op, refresh the caches and tell the panels (spec 4.6)."""
        op = result.get("op")
        if isinstance(op, Op):
            self.undo_stack.push(op, apply=False)
        self._problems.update(result.get("problems") or {})
        self._invalidate()
        self._set_dirty(True)
        self.sigFrameChanged.emit(self.current())
        return result

    # ------------------------------------------------------------------- undo
    def undo(self) -> bool:
        """Take the newest operation back; ``False`` when there is none."""
        return self.undo_stack.undo() is not None

    def redo(self) -> bool:
        """Re-apply the most recently undone operation."""
        return self.undo_stack.redo() is not None

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
            self._problems[key.step] = problems
            self.sigProblems.emit(problems)
            return False
        self._invalidate()
        self.sigFrameChanged.emit(key)
        if not self._step_to([s for s in self._available if s < key.step], last=True):
            self.sigFrameChanged.emit(self.current())
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
        self._problems.update(stats["problems"])
        self._invalidate()
        return stats

    def _open_conflicts(self) -> list[dict]:
        return self.db.conflicts(self.desktop, self.view, open_only=True)

    def queues(self) -> dict[str, list[dict]]:
        """The four review queues of spec 4.4, keyed by :data:`QUEUE_NAMES`.

        ``missing_shape`` is read from the problems of each frame's *last*
        refresh, which the session keeps in memory: recompiling the whole view
        to populate a list would make opening the review panel cost as much as
        a full sweep.  ``unexplained`` stays empty until the difference map
        lands (spec 4.2 step 4).
        """
        conflicts = [
            {"id": row["id"], "step": row["step"], "instance": row["instance"],
             "sym_diff_px": row["sym_diff_px"]}
            for row in self._open_conflicts()
        ]
        needs_review = [
            {"step": row["step"]}
            for row in self.db.frames_for(self.desktop, self.view)
            if row.get("review_status") == _NEEDS_REVIEW
        ]
        missing = [
            {"step": step, "instance": problem[len(_MISSING_SHAPE):]}
            for step in sorted(self._problems)
            for problem in self._problems[step]
            if problem.startswith(_MISSING_SHAPE)
        ]
        return {
            api.QUEUE_CONFLICTS: conflicts,
            api.QUEUE_NEEDS_REVIEW: needs_review,
            api.QUEUE_MISSING_SHAPE: missing,
            api.QUEUE_UNEXPLAINED: [],
        }

    def resolve_conflict(self, cid: int, resolution: str) -> None:
        """Resolve one queued disagreement and recompile the frame it was on."""
        conflict = self.db.get_conflict(int(cid))
        self.truth.resolve_conflict(int(cid), resolution, self.annotator)
        if conflict is not None:
            stats = edit.refresh_steps(self.db, self.truth, self.desktop, self.view,
                                       [conflict["step"]])
            self._problems.update(stats["problems"])
        self._invalidate()
        self.sigFrameChanged.emit(self.current())

    # --------------------------------------------------------------- internals
    def _invalidate(self) -> None:
        """Forget what depends on the annotation inputs of the current frame."""
        self._compiled = None

    def _set_dirty(self, value: bool) -> None:
        if bool(value) != self._dirty:
            self._dirty = bool(value)
            self.sigDirty.emit(self._dirty)

    @property
    def dirty(self) -> bool:
        """Whether anything has been edited since the last :meth:`save`."""
        return self._dirty
