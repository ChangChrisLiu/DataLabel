"""Operation log with undo/redo (spec 4.6, 10.2).

Every user edit is recorded as an :class:`Op` -- a *source level* description of
what changed plus the inverse patch needed to take it back.  The stack itself
knows nothing about the document: callers :meth:`UndoStack.register` a
``(do, undo)`` pair per :data:`KINDS` entry and the stack only routes payloads.

Pixel edits travel as COCO RLE (``tda.core.masks.encode_rle``), never as raw
arrays: a 200-entry history of 12 MP boolean masks would cost gigabytes, while
the RLE of a part silhouette is a few kilobytes.  Both directions of an
``edit_editing_mask`` op carry the full before/after mask, so an undo restores
the mask exactly rather than replaying strokes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from tda.core import masks as _masks

__all__ = ["KINDS", "Op", "UndoStack", "edit_editing_mask_op"]

#: Every operation kind the annotator can log.  ``Op`` validates against this,
#: which turns a typo in a kind string into an error at construction time
#: instead of a silently unrecorded edit.
KINDS: frozenset[str] = frozenset(
    {
        "edit_editing_mask",  # pixels of one instance's editing layer
        "set_zorder",  # instance stacking order of a frame
        "set_pair_override",  # PairOverride (A above B) for a pose segment
        "set_frame_override",  # FrameOverride (single-frame mask/visibility)
        "set_occluder",  # OccluderMask of one occluder_type on a frame
        "commit_keyframe",  # write/split a ShapeKeyframe
    }
)

Handler = Callable[[dict], None]


@dataclass
class Op:
    """One undoable operation.

    Attributes:
        kind: an entry of :data:`KINDS`.
        payload: arguments for the *do* handler (JSON/RLE friendly values only).
        inverse: arguments for the *undo* handler, i.e. the inverse patch.
    """

    kind: str
    payload: dict = field(default_factory=dict)
    inverse: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown op kind {self.kind!r}; expected one of {sorted(KINDS)}"
            )


def edit_editing_mask_op(
    instance: str, before: np.ndarray, after: np.ndarray
) -> Op:
    """Build an ``edit_editing_mask`` op from the two mask states.

    The inverse is the same payload with ``rle_before``/``rle_after`` swapped,
    so a single "set the mask to ``rle_after``" handler serves both directions.
    """
    # two full-canvas encodes per recorded stroke, on the GUI thread
    rle_before = _masks.encode_rle_boxed(before)
    rle_after = _masks.encode_rle_boxed(after)
    return Op(
        kind="edit_editing_mask",
        payload={"instance": instance, "rle_before": rle_before, "rle_after": rle_after},
        inverse={"instance": instance, "rle_before": rle_after, "rle_after": rle_before},
    )


class UndoStack:
    """Bounded undo/redo history over registered ``(do, undo)`` handlers.

    The stack is capped at :data:`LIMIT` entries; pushing past the cap drops the
    oldest op, which then becomes permanent.  Pushing also discards the redo
    branch, the usual linear-history behaviour.
    """

    #: Maximum number of ops kept; older ones fall off the bottom.
    LIMIT = 200

    def __init__(self, limit: Optional[int] = None) -> None:
        self._limit = int(limit) if limit is not None else self.LIMIT
        self._handlers: dict[str, tuple[Handler, Handler]] = {}
        self._done: list[Op] = []
        self._undone: list[Op] = []

    # -- wiring -------------------------------------------------------------
    def register(self, kind: str, do: Handler, undo: Handler) -> None:
        """Bind the forward and backward handler for one op kind."""
        if kind not in KINDS:
            raise ValueError(f"unknown op kind {kind!r}")
        self._handlers[kind] = (do, undo)

    def _handler(self, kind: str) -> tuple[Handler, Handler]:
        try:
            return self._handlers[kind]
        except KeyError:
            raise LookupError(
                f"no handler registered for op kind {kind!r}"
            ) from None

    # -- history ------------------------------------------------------------
    def push(self, op: Op, apply: bool = True) -> None:
        """Record ``op``, optionally applying it through its *do* handler.

        ``apply=False`` is for edits the caller has already performed (a brush
        stroke is painted live, then logged), so the mask is not written twice.
        The handler must still exist -- undoing needs it.
        """
        do, _ = self._handler(op.kind)
        if apply:
            do(op.payload)
        self._undone.clear()
        self._done.append(op)
        if len(self._done) > self._limit:
            del self._done[: len(self._done) - self._limit]

    def undo(self) -> Optional[Op]:
        """Revert the newest op and return it (``None`` when there is none).

        The op moves between the stacks **only once its handler has returned**.
        Popping first and applying afterwards meant that a handler which raised
        -- a write that failed half way -- took the op off the undo stack
        without ever reaching the redo stack: it was neither redo-able nor
        undo-able again, and nothing said so.  On a failure the stacks are
        exactly as they were and the exception is re-raised, which is what the
        window's guard reports.
        """
        if not self._done:
            return None
        op = self._done[-1]
        _, undo = self._handler(op.kind)
        undo(op.inverse)
        self._done.pop()
        self._undone.append(op)
        return op

    def redo(self) -> Optional[Op]:
        """Re-apply the most recently undone op and return it.

        Same rule as :meth:`undo`: the op changes stacks only on success.
        """
        if not self._undone:
            return None
        op = self._undone[-1]
        do, _ = self._handler(op.kind)
        do(op.payload)
        self._undone.pop()
        self._done.append(op)
        return op

    def clear(self) -> None:
        """Forget the whole history (e.g. when switching frame)."""
        self._done.clear()
        self._undone.clear()

    # -- introspection ------------------------------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    @property
    def ops(self) -> list[Op]:
        """The applied ops, oldest first (a copy; do not mutate the stack)."""
        return list(self._done)

    def __len__(self) -> int:
        return len(self._done)
