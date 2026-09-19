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
from tda.core.truth_conflicts import (
    disagreement,
    geom_payload,
    label_changes,
    label_text,
    row_payload,
    row_values,
)
from tda.core.truth_fresh import digest_of

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


def _prepared_for(key: FrameKey, prepared):
    """A caller's ``(inputs, compilation)`` pair, when it really is this frame's.

    Cheap paranoia around an optimisation: confirming the wrong frame's
    compilation would freeze somebody else's pixels under this annotator's
    name, so the pair is checked against the key rather than trusted.
    """
    if not prepared:
        return None
    inputs, compiled = prepared
    if inputs is None or compiled is None:
        return None
    if inputs.key != key or compiled.key != key:
        return None
    return inputs, compiled


class VerifyMixin:
    """Confirming and demoting one frame, for :class:`~tda.core.truth.TruthService`."""

    def verify_frame(self, key: FrameKey, annotator: str, prepared=None) -> None:
        """Freeze every row of one frame after a human confirmed it (spec 4.2).

        ``prepared`` is an ``(inputs, compilation)`` pair the caller already
        has for **this** frame, from
        :meth:`~tda.core.truth.TruthService.compile_with_inputs` or a
        :meth:`~tda.core.truth.TruthService.refresh`. The session keeps the one
        it made when the annotator arrived at the frame and hands it back here,
        so pressing Space does not compile at 1600x1600 what was compiled on
        arrival -- and confirms exactly the pixels that were on screen. It is
        used only when it is for this frame; anything else is compiled afresh.

        Raises :class:`ValueError` and writes no truth row in four cases, and
        the last two are the same rule twice: **a confirmation may never be how
        a frozen row changes**.

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
        * a **frozen row the inputs no longer contain**.
        * a **frozen row the inputs have moved under** -- still in the frame,
          but its geometry or its labels no longer agree. This is the case a
          queued re-check exists for, and until that re-check runs the row is
          the only record of what the human signed: overwriting it here and
          stamping the digest turned the re-check into a no-op, so the conflict
          was never raised at all.

        Either way the disagreement goes into the conflict queue and the frame
        is left for that decision. Only an ``auto`` row -- a cache the compiler
        owns -- is ever rewritten or dropped by a confirmation; a frozen row
        that **agrees** is left exactly as it is, signature included, so
        confirming an already-confirmed frame writes nothing but the digest.

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
        inputs, compiled = _prepared_for(key, prepared) or self._compile(key)
        blocking = [p for p in compiled.problems if p.startswith(BLOCKING_PROBLEMS)]
        if blocking:
            raise ValueError(refused + ", ".join(blocking))
        stored = self.db.compiled(key)
        gone = sorted(set(stored) - set(compiled.instances))
        disputed = self._queue_frozen_disagreements(key, stored, compiled, inputs, gone)
        if disputed:
            raise ValueError(refused + "; ".join(disputed)
                             + "; the disagreement is now in the review queue")
        previous = self._review_status(key)
        with self.db.transaction():
            for instance in sorted(compiled.instances):
                row = stored.get(instance)
                if row is not None and row["status"] == VERIFIED:
                    # It agrees -- the gate above proved it -- so it is the same
                    # annotation and there is nothing to write. Rewriting it
                    # moved a signature somebody else had already made, and a
                    # re-trace within tolerance drifted the stored geometry one
                    # pixel per confirmation away from what was confirmed.
                    # `refresh` skips exactly this row for exactly this reason.
                    continue
                self._put_row(
                    key, instance, row_values(compiled.instances[instance]),
                    VERIFIED, compiled.input_hash, verified_by=annotator,
                )
            for instance in gone:
                # `auto` only: the guard above turned every frozen one away
                self.db.delete_compiled(key, instance)
            self._mark_bench(key, compiled)
            self._stamp(key, digest_of(inputs, self.compiler_version))
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

    def _queue_frozen_disagreements(self, key: FrameKey, stored: dict[str, dict],
                                    compiled, inputs, gone: list[str]) -> list[str]:
        """Queue every frozen row this compilation disagrees with; say which.

        Exactly the entries :meth:`~tda.core.truth.TruthService.refresh` would
        have made, written in one transaction of their own, so the refusal the
        caller is about to raise leaves the annotator with something to act on
        rather than a frame they cannot confirm and cannot see the reason for.
        The frame's digest goes with them: the rows keep their frozen values and
        therefore do not describe these inputs -- stamping it here is what used
        to turn the queued re-check into a no-op.

        Returns one sentence per instance, in key order, or ``[]`` when nothing
        a human signed is in dispute.

        A row whose stored ``input_hash`` is this compilation's cannot disagree
        with it -- it *is* this compilation -- so it is skipped without decoding
        anything, which is the same short-circuit
        :meth:`~tda.core.truth.TruthService._write_refresh` makes. Comparing
        masks means decoding them, and Space on a forty-row frame at 1600x1600
        was spending half a second of the GUI thread proving rows agree with
        inputs they were derived from.
        """
        reasons: list[tuple[str, str, Optional[dict], Optional[dict], int]] = []
        for instance in gone:
            if stored[instance]["status"] != VERIFIED:
                continue
            payload = row_payload(stored[instance])
            reasons.append((instance, f"{instance} is no longer in this frame",
                            payload, None, self._payload_area(payload)))
        for instance in sorted(set(stored) & set(compiled.instances)):
            row = stored[instance]
            if row["status"] != VERIFIED or row["input_hash"] == compiled.input_hash:
                continue
            compiled_inst = compiled.instances[instance]
            diff = disagreement(row, compiled_inst)
            labels = label_changes(row, compiled_inst,
                                   inputs.frame_overrides.get(instance))
            if diff is None and not labels:
                continue
            what = label_text(labels) if labels else f"{diff} px differ"
            values = row_values(compiled_inst)
            reasons.append((
                instance, f"the confirmed {instance} no longer agrees ({what})",
                row_payload(row),
                geom_payload(values.visible_rle, values.box, labels), int(diff or 0),
            ))
        if not reasons:
            return []
        queued: Optional[list[dict]] = None
        with self.db.transaction():
            for instance, _text, old, new, pixels in reasons:
                queued, _new = self._queue_conflict(key, instance, old, new,
                                                    pixels, queued)
            self.db.clear_frame_digest(key)
        return [text for _inst, text, *_rest in reasons]

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
