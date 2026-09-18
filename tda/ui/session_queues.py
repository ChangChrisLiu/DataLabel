"""Frame status and the four review queues (spec 3.4, 4.4, 4.5).

Both are read constantly -- the timeline asks for every step's status on every
repaint, and the review panel rebuilds its four lists whenever anything changes
-- so neither may cost a compile or a query per step.  :class:`ReviewState` is
what makes that possible: it holds the per-frame problem lists of the last
refresh, the diff-map regions nobody has explained, and a per-change memo of
which steps carry an open conflict.
"""
from __future__ import annotations

from typing import Callable, Optional

from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.truth import NEEDS_REVIEW, VERIFIED
from tda.ui import session_api as api

__all__ = ["MISSING_SHAPE", "ReviewState"]

MISSING_SHAPE = "missing_shape:"


class ReviewState:
    """The session's memory of what is wrong with the open desktop/view."""

    def __init__(self, db: Db) -> None:
        self.db = db
        self.desktop: Optional[int] = None
        self.view: str = ""
        #: step -> the compiler problems of its last refresh.
        self.problems: dict[int, list[str]] = {}
        #: step -> difference-map regions the window could not explain.
        self.unexplained: dict[int, list] = {}
        #: ``() -> {step: FrameCoverage}``: what has been drawn, without pixels.
        self.coverage: Callable[[], dict] = dict
        self._conflict_steps: Optional[set[int]] = None
        self._pending: Optional[set[int]] = None
        self._coverage: Optional[dict] = None

    def open(self, desktop: int, view: str, coverage: Callable[[], dict]) -> None:
        self.desktop, self.view = desktop, view
        self.coverage = coverage
        self.clear()

    def clear(self) -> None:
        self.problems.clear()
        self.unexplained.clear()
        self.invalidate()

    def invalidate(self) -> None:
        """Forget every per-change memo; the next query builds it again."""
        self._conflict_steps = None
        self._pending = None
        self._coverage = None

    def drawn(self) -> dict:
        """``step -> FrameCoverage`` for the whole view, computed at most once.

        Cheap enough to redo on every edit (no pixels are touched), which is
        what lets the timeline and the review panel stop depending on stored
        compiled rows.
        """
        if self._coverage is None:
            self._coverage = self.coverage()
        return self._coverage

    def pending_rechecks(self) -> set[int]:
        """Frozen frames of this view still waiting for the sweeper."""
        if self._pending is None:
            self._pending = set(self.db.rechecks(self.desktop, self.view))
        return self._pending

    # -- status -------------------------------------------------------------
    def conflicted_steps(self) -> set[int]:
        """Steps of this view with an open conflict, read at most once per change.

        The timeline asks for every step's status on every repaint, so this must
        not be one query per row.
        """
        if self._conflict_steps is None:
            self._conflict_steps = {c["step"] for c in self.open_conflicts()}
        return self._conflict_steps

    def open_conflicts(self) -> list[dict]:
        return self.db.conflicts(self.desktop, self.view, open_only=True)

    def frame_status(self, step: int) -> str:
        """One of :data:`tda.ui.session_api.FRAME_STATUSES` (spec 4.5 colours).

        The order matters: a frozen frame with an open disagreement is a
        *conflict* first, because that is what somebody has to act on.
        """
        if self.desktop is None:
            return api.STATUS_UNLABELED
        key = FrameKey(self.desktop, int(step), self.view)
        row = self.db.get_frame(key)
        if row is None or row.get("missing"):
            return api.STATUS_MISSING
        if int(step) in self.conflicted_steps():
            return api.STATUS_CONFLICT
        status = row.get("review_status")
        if status == NEEDS_REVIEW:
            return api.STATUS_NEEDS_REVIEW
        if status == VERIFIED:
            # frozen, but its inputs moved and nobody has compared them yet: it
            # may still turn into a conflict, so it is not settled (spec 3.4)
            return (api.STATUS_RECHECK if int(step) in self.pending_rechecks()
                    else api.STATUS_VERIFIED)
        # an unverified frame's compiled rows are a cache that a commit leaves
        # stale on purpose, so "has anything been drawn here" is asked of the
        # keyframes instead of of the truth table
        found = self.drawn().get(int(step))
        return (api.STATUS_AUTO if found is not None and found.annotated
                else api.STATUS_UNLABELED)

    # -- queues -------------------------------------------------------------
    def set_unexplained(self, step: int, boxes) -> None:
        """Record (or clear) the unexplained difference regions of one frame."""
        boxes = list(boxes or [])
        if boxes:
            self.unexplained[int(step)] = boxes
        else:
            self.unexplained.pop(int(step), None)

    def queues(self) -> dict[str, list[dict]]:
        """The four review queues of spec 4.4, keyed by :data:`QUEUE_NAMES`.

        ``missing_shape`` is derived from the state machine and the keyframe
        chains (:mod:`tda.ui.session_coverage`), never from the compiled rows:
        an unverified frame's rows are a cache a commit leaves stale on purpose,
        and compiling the view to fill the list would cost exactly what the lazy
        truth table exists to avoid.
        """
        return {
            api.QUEUE_CONFLICTS: [
                {"id": row["id"], "step": row["step"], "instance": row["instance"],
                 "sym_diff_px": row["sym_diff_px"]}
                for row in self.open_conflicts()
            ],
            api.QUEUE_NEEDS_REVIEW: [
                {"step": row["step"]}
                for row in self.db.frames_for(self.desktop, self.view)
                if row.get("review_status") == NEEDS_REVIEW
            ],
            api.QUEUE_MISSING_SHAPE: [
                {"step": step, "instance": instance}
                for step, found in sorted(self.drawn().items())
                for instance in found.missing
            ],
            api.QUEUE_UNEXPLAINED: [
                {"step": step, "boxes": list(boxes)}
                for step, boxes in sorted(self.unexplained.items())
                if boxes
            ],
        }
