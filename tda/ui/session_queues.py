"""Frame status and the four review queues (spec 3.4, 4.4, 4.5).

Both are read constantly -- the timeline asks for every step's status on every
repaint, and the review panel rebuilds its four lists whenever anything changes
-- so neither may cost a compile or a query per step.  :class:`ReviewState` is
what makes that possible: it holds the per-frame problem lists of the last
refresh, the diff-map regions nobody has explained, and a per-change memo of
which steps carry an open conflict.
"""
from __future__ import annotations

from typing import Optional

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
        self._conflict_steps: Optional[set[int]] = None

    def open(self, desktop: int, view: str) -> None:
        self.desktop, self.view = desktop, view
        self.clear()

    def clear(self) -> None:
        self.problems.clear()
        self.unexplained.clear()
        self.invalidate()

    def invalidate(self) -> None:
        """Forget the conflict memo; the next status query reads it again."""
        self._conflict_steps = None

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
            return api.STATUS_VERIFIED
        return api.STATUS_AUTO if self.db.compiled(key) else api.STATUS_UNLABELED

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

        ``missing_shape`` is read from the problems of each frame's *last*
        refresh, which the session keeps in memory: recompiling the whole view
        to populate a list would make opening the review panel cost as much as a
        full sweep.
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
                {"step": step, "instance": problem[len(MISSING_SHAPE):]}
                for step in sorted(self.problems)
                for problem in self.problems[step]
                if problem.startswith(MISSING_SHAPE)
            ],
            api.QUEUE_UNEXPLAINED: [
                {"step": step, "boxes": list(boxes)}
                for step, boxes in sorted(self.unexplained.items())
                if boxes
            ],
        }
