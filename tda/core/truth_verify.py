"""Freezing a frame a human confirmed, and sending one back (spec 3.4, 4.2).

Split off :mod:`tda.core.truth` for size; the methods are part of
:class:`~tda.core.truth.TruthService`'s surface.

Everything here is about the one line the truth table draws: a ``verified`` row
is a human's signature, and the compiler may never write over it or delete it.
:meth:`VerifyMixin.verify_frame` is the only place that turns ``auto`` rows into
signed ones, so it is also the only place that can break the invariant -- and it
used to, three ways at once, so the three refusals live here together.

:meth:`VerifyMixin.demote_frame` is the way back: a frozen frame whose instance
set no longer matches the inputs is not wrong, it is unreviewed, and the
annotator is asked to look at it again rather than told it was mistaken.
"""
from __future__ import annotations

from typing import Optional

from tda.core.compiler import CompiledFrame
from tda.core.model import FrameKey, Placement
from tda.core.truth_conflicts import row_payload, row_values

__all__ = ["BLOCKING_PROBLEMS", "VerifyMixin"]

#: Problem prefixes that stop a frame from being verified (spec 3.3 step 3).
#: Everything else -- ``bench_missing``, ``zorder_missing``, ``empty_visible``,
#: ``pose_segment_ambiguous`` -- is a warning the annotator may accept.
BLOCKING_PROBLEMS = ("missing_shape:", "zorder_cycle:", "shape_size_mismatch:")

VERIFIED = "verified"
NEEDS_REVIEW = "needs_review"
ON_BENCH = Placement.ON_BENCH.value
BENCH_MISSING = "bench_missing:"
SYSTEM = "system"


class VerifyMixin:
    """Confirming and demoting one frame, for :class:`~tda.core.truth.TruthService`."""

    def verify_frame(self, key: FrameKey, annotator: str) -> None:
        """Freeze every row of one frame after a human confirmed it (spec 4.2).

        Raises :class:`ValueError` and writes no truth row in three cases:

        * an **open conflict** of this frame. A disagreement is the one thing
          in the truth table only a human can settle, and until it is settled
          nobody knows which value they are confirming. Confirming anyway used
          to delete the frozen row the conflict was about and leave the
          disagreement open for ever -- reached simply by pressing Space on the
          frame in the ``needs_review`` queue, which is where the demotion had
          just put it.
        * the compilation has a **blocking problem** (:data:`BLOCKING_PROBLEMS`)
          -- a missing chassis shape, a contradictory z-order. Warnings, such as
          a bench part nobody has boxed yet, do not stop the confirmation; they
          only leave ``bench_annotated`` false.
        * a **frozen row the inputs no longer contain**. A human confirmed that
          instance here, so its disappearance is a disagreement like any other:
          it goes into the conflict queue and the frame is left for that
          decision. Only an ``auto`` row -- a cache the compiler owns -- may be
          dropped by a confirmation.

        Every write goes into one transaction: a frame is either confirmed
        whole -- rows, flag and op log -- or not at all.
        """
        refused = f"frame {key.desktop}/{key.view}/step {key.step} cannot be verified: "
        open_ids = self._open_conflict_ids(key)
        if open_ids:
            raise ValueError(
                refused + "conflict(s) "
                + ", ".join(str(cid) for cid in open_ids)
                + " are still open; settle them in the review queue first"
            )
        _, compiled = self._compile(key)
        blocking = [p for p in compiled.problems if p.startswith(BLOCKING_PROBLEMS)]
        if blocking:
            raise ValueError(refused + ", ".join(blocking))
        stored = self.db.compiled(key)
        gone = sorted(set(stored) - set(compiled.instances))
        vanished = [i for i in gone if stored[i]["status"] == VERIFIED]
        if vanished:
            self._queue_vanished(key, stored, vanished)
            raise ValueError(
                refused + ", ".join(vanished) + " no longer "
                + ("belong" if len(vanished) > 1 else "belongs")
                + " to this frame, but a human confirmed "
                + ("them" if len(vanished) > 1 else "it")
                + " here; the disagreement is now in the review queue"
            )
        previous = self._review_status(key)
        with self.db.transaction():
            for instance in sorted(compiled.instances):
                self._put_row(
                    key, instance, row_values(compiled.instances[instance]),
                    VERIFIED, compiled.input_hash, verified_by=annotator,
                )
            for instance in gone:
                # `auto` only: the guard above turned every frozen one away
                self.db.delete_compiled(key, instance)
            self._mark_bench(key, compiled)
            self._stamp(key, self.inputs_digest(key))
            self.db.set_frame_flags(key, review_status=VERIFIED)
            self.db.log_op(
                key.desktop, key.view, "verify_frame",
                {"step": key.step, "instances": sorted(compiled.instances),
                 "problems": list(compiled.problems), "input_hash": compiled.input_hash},
                {"kind": "set_review_status", "step": key.step, "review_status": previous},
                annotator,
            )

    def demote_frame(self, key: FrameKey, reason: str) -> None:
        """Send a frame back to the review queue (spec 3.4, "需复核")."""
        previous = self._review_status(key)
        self.db.set_frame_flags(key, review_status=NEEDS_REVIEW)
        self.db.log_op(
            key.desktop, key.view, "demote_frame",
            {"step": key.step, "reason": reason},
            {"kind": "set_review_status", "step": key.step, "review_status": previous},
            SYSTEM,
        )

    # --------------------------------------------------------------- internals

    def _open_conflict_ids(self, key: FrameKey) -> list[int]:
        """Ids of the open disagreements about one frame, ascending."""
        return sorted(
            int(row["id"])
            for row in self.db.conflicts(key.desktop, key.view, open_only=True)
            if int(row["step"]) == int(key.step)
        )

    def _queue_vanished(self, key: FrameKey, stored: dict[str, dict],
                        vanished: list[str]) -> None:
        """Queue "this confirmed instance is no longer in the frame" (spec 3.4).

        The same entry :meth:`~tda.core.truth.TruthService.refresh` would have
        made, in its own transaction, so the refusal the caller is about to
        raise leaves the annotator with something to act on rather than a frame
        they cannot confirm and cannot see the reason for. The frame's digest
        goes with it: the rows keep their frozen values and therefore do not
        describe these inputs.
        """
        queued: Optional[list[dict]] = None
        with self.db.transaction():
            for instance in vanished:
                payload = row_payload(stored[instance])
                queued, _new = self._queue_conflict(
                    key, instance, payload, None, self._payload_area(payload), queued
                )
            self.db.clear_frame_digest(key)

    def _review_status(self, key: FrameKey) -> Optional[str]:
        frame = self.db.get_frame(key)
        return None if frame is None else frame.get("review_status")

    def _frame_is_verified(self, key: FrameKey, stored: dict[str, dict]) -> bool:
        """Has a human confirmed this frame? The flag, or any frozen row."""
        if self._review_status(key) == VERIFIED:
            return True
        return any(row["status"] == VERIFIED for row in stored.values())

    @staticmethod
    def _demotion_reason(fresh: set[str], known: set[str]) -> str:
        parts = []
        if fresh - known:
            parts.append("gained " + ", ".join(sorted(fresh - known)))
        if known - fresh:
            parts.append("lost " + ", ".join(sorted(known - fresh)))
        return "the verified frame " + " and ".join(parts)

    def _mark_bench(self, key: FrameKey, compiled: CompiledFrame) -> None:
        """Keep ``bench_annotated`` in step with the bench shapes (spec 3.3.3).

        Only frames that actually have a part on the bench carry the flag; a
        missing bench shape clears it instead of blocking the frame.
        """
        if not any(inst.placement == ON_BENCH for inst in compiled.instances.values()):
            # no bench instance in this frame at all -- either nothing is out of
            # the machine yet, or this view has no staging area and the gate in
            # `gather` dropped them (spec 3.3 step 2). Either way the flag is
            # about something that is not here, so it is left as it was.
            return
        annotated = not any(p.startswith(BENCH_MISSING) for p in compiled.problems)
        frame = self.db.get_frame(key)
        if frame is not None and frame.get("bench_annotated") is annotated:
            return  # unchanged: a refresh that changes nothing writes nothing
        self.db.set_frame_flags(key, bench_annotated=annotated)
