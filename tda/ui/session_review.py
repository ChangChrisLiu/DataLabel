"""Confirming a frame, and the four review queues behind it (spec 4.2, 4.4).

Everything here is about a frame being *finished*, or about what is stopping it:
freezing one the annotator has confirmed, saying in the card's own words what a
refused confirmation is still missing, working the re-check backlog off, and
handing the review panel its queues.

Split off :mod:`tda.ui.session` for size; the methods are part of
:class:`~tda.ui.session.AnnotationSession`'s surface.
"""
from __future__ import annotations

from tda.core.truth import StaleConflictError
from tda.core.truth_inputs import instances_of
from tda.ui.session_coverage import coverage
from tda.ui import session_api as api
from tda.ui import session_edit as edit

__all__ = ["ReviewMixin"]

MISSING_SHAPE = "missing_shape:"


class ReviewMixin:
    """Confirming, the queues and the re-check backlog, for the session."""

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
            self.sigProblems.emit(problems + self._how_to_fix(problems))
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

    def _how_to_fix(self, problems: list[str]) -> list[str]:
        """Say a blocking problem in the words the task card uses for it.

        ``missing_shape:cpu_cooler.01`` is what the compiler calls it; what the
        annotator needs to read is which card item to act on. One formatter, so
        a panel can match a problem to its item rather than guess.
        """
        listed = {item["instance"]: item["text"] for item in self.task_card()}
        records = instances_of(self.db, self.current().desktop)
        out = []
        for problem in problems:
            if not problem.startswith(MISSING_SHAPE):
                continue
            instance = problem.split(":", 1)[1]
            # an instance the card does not mention -- the chassis never changes
            # state, so it is only ever on the start card -- still gets the same
            # sentence, from the same formatter
            out.append(listed.get(instance) or edit.item_text(
                api.KIND_ADD_SHAPE, instance, {}, records.get(instance)))
        return out

    def refresh_all(self) -> dict:
        """Recompile every frame of the open view (spec 3.4).

        The queues of :meth:`queues` are fed by whatever has been compiled so
        far, so this is what a session calls to fill them in one sweep -- after
        a bulk import, or when the review panel is opened on a view nobody has
        visited in this session.
        """
        running = self.sweeper.is_running
        if not self.sweeper.stop():  # one drain, not two racing over the same rows
            log.error("refresh_all aborted: the truth sweeper would not stop")
            raise RuntimeError("the truth sweeper is still running; cannot refresh now")
        try:
            self.truth.run_pending_rechecks(self.desktop, self.view)
            stats = edit.refresh_steps(self.db, self.truth, self.desktop, self.view,
                                       self._available)
        finally:
            if running and self.sweeper_enabled:
                self.sweeper.open(self.desktop, self.view)
                self.sweeper.enqueue(self.db.rechecks(self.desktop, self.view))
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
        if not self.is_open:
            self.sigProblems.emit([f"conflict {cid} not resolved: no view is open"])
            return "refused"
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
