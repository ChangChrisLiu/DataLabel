"""The persisted truth table: refresh, verification and conflicts (spec 3.4).

:mod:`tda.core.compiler` derives one frame's geometry from explicit inputs;
:class:`TruthService` is what turns that pure function into the stored
``compiled_mask`` table:

* :meth:`TruthService.compile`  -- gather this frame's inputs and compile it.
* :meth:`TruthService.refresh`  -- write the result into the truth table.
* :meth:`TruthService.verify_frame` / :meth:`TruthService.demote_frame`
  -- freeze a frame a human confirmed, or send it back for review.
* :meth:`TruthService.resolve_conflict` -- close one queued disagreement by
  accepting the new value or pinning the frozen one.
* :meth:`TruthService.affected_steps` -- which frames one shape reaches, for
  the canvas' "affects N frames" strip (spec 4.3).

Geometry, the frozen-vs-recompiled comparison and the conflict payloads live
next door in :mod:`tda.core.truth_conflicts`; reading the inputs out of the
database is :mod:`tda.core.truth_inputs`' job.

The rules the table lives by (spec 3.4)
---------------------------------------
* A row is ``auto`` while the compiler owns it: any change to the inputs
  rewrites it.
* A row is ``verified`` once a human confirmed the frame. From then on the
  compiler may **never** overwrite it. When a re-compilation disagrees with a
  frozen row beyond the re-tracing tolerance of :func:`tda.core.masks.is_conflict`
  (box rows: a corner moving more than :data:`BOX_TOL_PX`), the disagreement is
  queued in the ``conflict`` table -- old value, new value and the number of
  differing pixels -- and the row keeps its frozen value until a human resolves
  it with :meth:`TruthService.resolve_conflict`. Nothing is ever silently
  overwritten, and a disagreement already sitting in the open queue is not
  queued twice.
* A verified frame that gains or loses an instance drops back to
  ``needs_review``: the human confirmed a different set of objects than the one
  the inputs now describe.

Costs
-----
:meth:`refresh` compiles the frame and then compares hashes: when every stored
row already carries this frame's ``input_hash`` and the instance set is
unchanged, nothing is written at all. :meth:`refresh_range` shares one
:class:`~tda.core.truth_inputs.InputCache` across its steps, so the instance
table, the event log and the keyframes are read once per sweep.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from tda.core import masks
from tda.core.compiler import CompiledFrame, compile_frame, select_keyframe
from tda.core.db import RESOLUTIONS, Db
from tda.core.model import FrameKey, FrameOverride, Placement, ShapeKeyframe
from tda.core.states import needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_fresh import FreshMixin, digest_of
from tda.core.truth_resolve import ResolveMixin, StaleConflictError
from tda.core.truth_conflicts import (
    BOX_TOL_PX,
    GEOM_BOX,
    disagreement,
    geom_payload,
    label_changes,
    payload_geometry,
    payload_labels,
    row_payload,
    row_values,
)
from tda.core.truth_inputs import (
    FrameInputs,
    InputCache,
    events_of,
    gather,
    instances_of,
    pose_segment_of,
    state_of,
)

__all__ = ["BLOCKING_PROBLEMS", "BOX_TOL_PX", "BY_HAND", "StaleConflictError", "TruthService"]

#: Problem prefixes that stop a frame from being verified (spec 3.3 step 3).
#: Everything else -- ``bench_missing``, ``zorder_missing``, ``empty_visible``,
#: ``pose_segment_ambiguous`` -- is a warning the annotator may accept.
BLOCKING_PROBLEMS = ("missing_shape:", "zorder_cycle:", "shape_size_mismatch:")

ACCEPT_NEW = "accept_new"
KEEP_OLD = "keep_old"
SUPERSEDED = "superseded"
OPEN = "open"

#: The resolutions a human may ask for; ``superseded`` is only ever recorded by
#: :meth:`TruthService.resolve_conflict` itself.
BY_HAND = tuple(r for r in RESOLUTIONS if r != SUPERSEDED)

AUTO = "auto"
VERIFIED = "verified"
NEEDS_REVIEW = "needs_review"
ON_BENCH = Placement.ON_BENCH.value
BENCH_MISSING = "bench_missing:"
SYSTEM = "system"


class TruthService(FreshMixin, ResolveMixin):
    """Reads the annotations, compiles frames and owns the ``compiled_mask`` table."""

    def __init__(self, db: Db, tax: Taxonomy, compiler_version: str = "1"):
        self.db = db
        self.tax = tax
        self.compiler_version = compiler_version

    # ------------------------------------------------------------------ compile

    def compile(self, key: FrameKey) -> CompiledFrame:
        """Compile one frame from the database's current state (spec 3.3)."""
        return self._compile(key)[1]

    def _compile(
        self, key: FrameKey, cache: Optional[InputCache] = None
    ) -> tuple[FrameInputs, CompiledFrame]:
        """The frame's inputs and its compilation; ``hw`` is needed by callers."""
        inputs = gather(self.db, self.tax, key, cache)
        return inputs, self._compile_inputs(key, inputs)

    def _compile_inputs(self, key: FrameKey, inputs: FrameInputs) -> CompiledFrame:
        """Compile from inputs the caller already gathered."""
        return compile_frame(
            key,
            inputs.hw,
            inputs.needs,
            inputs.keyframes,
            inputs.zorder,
            inputs.overrides,
            inputs.occluders,
            inputs.frame_overrides,
            inputs.transform,
            self.compiler_version,
            placements=inputs.placements,
            pose_segment=inputs.pose_segment,
            bench_roi=inputs.bench_roi,
        )

    # ------------------------------------------------------------------ refresh

    def refresh(self, key: FrameKey, cache: Optional[InputCache] = None,
                guard: Optional[str] = None, want_compiled: bool = False,
                ignore_digest: bool = False) -> dict:
        """Bring one frame's truth rows up to date with the current inputs.

        Returns ``{"updated", "conflicts", "skipped", "problems", "compiled",
        "stale"}``: rows written (a deleted row counts as written), frozen rows
        found in disagreement, rows left alone, the compiler's problem list, the
        compilation the whole decision was made from -- handed back so that a
        caller which also needs the frame does not compile it a second time --
        and whether the write was abandoned.

        ``cache`` is :meth:`refresh_range`'s way of reading the step-independent
        inputs once; callers outside this module leave it out.

        ``ignore_digest`` compiles the frame whatever the stored digest says,
        which is what the session's "I do not trust the cache" refresh is for.

        A frame whose stored digest still describes its inputs is **not
        compiled at all**: the pixel work is what a batch pass over an untouched
        view used to spend all its time on. ``compiled`` then comes back
        ``None`` unless ``want_compiled`` asks for it, which is what the session
        does when it needs the arrays to draw.

        ``guard`` is an :meth:`inputs_digest` taken before the compilation, for
        a caller working off the GUI thread. The pixel work happens outside any
        transaction; every write then happens inside **one**, which begins by
        taking the digest again. If it moved, nothing is written and ``stale``
        comes back true: the frame has to be looked at again against the inputs
        it has now, and a conflict describing the old ones would be a conflict
        nobody caused.
        """
        inputs = gather(self.db, self.tax, key, cache)
        digest = digest_of(inputs, self.compiler_version)
        if not ignore_digest and self._digest_is_current(key, digest):
            # the rows already describe exactly these inputs: there is nothing
            # to derive and, since nothing moved, nothing a frozen row could
            # disagree with either
            result = {"updated": 0, "conflicts": 0, "skipped": len(self.db.compiled(key)),
                      "problems": [], "compiled": None, "stale": False}
            if want_compiled:
                result["compiled"] = self._compile_inputs(key, inputs)
            return result
        # the inputs are already in hand: gathering them a second time reads the
        # whole keyframe table again, which is most of a batch pass's time
        compiled = self._compile_inputs(key, inputs)  # no transaction: pixels only
        with self.db.transaction():
            return self._write_refresh(key, compiled, guard, digest,
                                       inputs.frame_overrides)

    def _write_refresh(self, key: FrameKey, compiled: CompiledFrame,
                       guard: Optional[str], digest: str,
                       overrides: dict[str, FrameOverride]) -> dict:
        """The write half of :meth:`refresh`; the caller holds the transaction."""
        result: dict = {
            "updated": 0,
            "conflicts": 0,
            "skipped": 0,
            "problems": list(compiled.problems),
            "compiled": compiled,
            "stale": False,
        }
        if guard is not None and self.inputs_digest(key) != guard:
            result["stale"] = True  # nothing written; the block commits nothing
            return result
        stored = self.db.compiled(key)
        new_hash = compiled.input_hash
        self._mark_bench(key, compiled)

        fresh = set(compiled.instances)
        known = set(stored)
        if known == fresh and all(row["input_hash"] == new_hash for row in stored.values()):
            # nothing to write -- but the rows *do* describe these inputs, and
            # saying so is the whole point: a database that predates the digest
            # table is entirely in this state, and without the stamp no pass
            # over it is ever cheaper than the first
            result["skipped"] = len(stored)
            self._stamp(key, digest)
            return result

        verified_frame = self._frame_is_verified(key, stored)
        queued: Optional[list[dict]] = None

        for instance in sorted(fresh):
            compiled_inst = compiled.instances[instance]
            row = stored.get(instance)
            if row is not None and row["input_hash"] == new_hash:
                result["skipped"] += 1
                continue
            if row is not None and row["status"] == VERIFIED:
                diff = disagreement(row, compiled_inst)
                labels = label_changes(row, compiled_inst, overrides.get(instance))
                if diff is None and not labels:
                    result["skipped"] += 1
                    continue
                values = row_values(compiled_inst)
                queued, _new = self._queue_conflict(
                    key, instance, row_payload(row),
                    geom_payload(values.visible_rle, values.box, labels),
                    int(diff or 0), queued,
                )
                result["conflicts"] += 1
                continue
            self._put_row(key, instance, row_values(compiled_inst), AUTO, new_hash)
            result["updated"] += 1

        for instance in sorted(known - fresh):
            row = stored[instance]
            if row["status"] == VERIFIED:
                old_payload = row_payload(row)
                queued, _new = self._queue_conflict(
                    key, instance, old_payload, None, self._payload_area(old_payload), queued
                )
                result["conflicts"] += 1
            else:
                self.db.delete_compiled(key, instance)
                result["updated"] += 1

        if verified_frame and (fresh - known or known - fresh):
            self.demote_frame(key, self._demotion_reason(fresh, known))
        if result["conflicts"]:
            # the frozen rows deliberately still hold their old value, so they
            # do NOT describe these inputs: stamping the digest here would make
            # the next pass skip the frame and the disagreement would never be
            # raised again once the queue entry was resolved (spec 3.4)
            self.db.clear_frame_digest(key)
        else:
            self._stamp(key, digest)
        return result

    def _stamp(self, key: FrameKey, digest: str) -> None:
        """Record what the rows just written were derived from (same transaction).

        Only ever called when the rows do describe those inputs: a frame with a
        queued conflict keeps its frozen value instead, so its digest is dropped
        and the next pass looks at it again.
        """
        self.db.set_frame_digest(key, digest, self.compiler_version,
                                 len(self.db.compiled(key)))

    def refresh_range(self, desktop: int, view: str, steps: Iterable[int],
                      per_step: bool = False, ignore_digest: bool = False) -> dict:
        """:meth:`refresh` every step of one view, with the totals summed up.

        ``problems`` is the concatenation of the per-step problem lists in the
        order the steps were given -- or, with ``per_step``, a dict keyed by
        step, which is what a caller needs to say *where* a missing shape is
        (spec 4.4 queues).
        """
        cache = InputCache()
        total: dict = {"updated": 0, "conflicts": 0, "skipped": 0,
                       "problems": {} if per_step else []}
        for step in steps:
            one = self.refresh(FrameKey(desktop, int(step), view), cache,
                               ignore_digest=ignore_digest)
            for counter in ("updated", "conflicts", "skipped"):
                total[counter] += one[counter]
            if per_step:
                total["problems"][int(step)] = list(one["problems"])
            else:
                total["problems"].extend(one["problems"])
        return total

    # ------------------------------------------------------- pending rechecks

    def queue_rechecks(self, desktop: int, view: str, steps: Iterable[int]) -> list[int]:
        """Remember that these frozen frames have to be compared again (spec 3.4).

        Editing a shape changes the inputs of every frame it reaches, and a
        ``verified`` frame among them may now disagree with what a human froze.
        Compiling them all before the annotator can carry on is what made an
        edit cost seconds, so the check is deferred -- and, because a conflict
        that is never raised is worse than a slow tool, the *request* is stored
        rather than kept in memory. Returns the steps queued.
        """
        wanted = [int(s) for s in steps]
        if wanted:
            self.db.add_rechecks(desktop, view, wanted)
        return sorted(set(wanted))

    def pending_rechecks(self, desktop: int, view: str) -> list[int]:
        """Frozen frames of one view still waiting to be compared, ascending.

        An export or a quality check must run these first: the truth table is
        only trustworthy once nothing is outstanding.
        """
        return self.db.rechecks(desktop, view)

    def run_pending_rechecks(self, desktop: int, view: str) -> dict:
        """Work the queue off synchronously; same totals as :meth:`refresh_range`.

        This is the batch path -- ``cli check`` and the exports -- next to the
        background sweeper the GUI uses. Each request is retired under the
        generation it was read with, so one that arrives during the drain stays
        queued rather than being cleared unchecked.
        """
        cache = InputCache()
        total: dict = {"updated": 0, "conflicts": 0, "skipped": 0, "problems": []}
        for step, gen in self.db.recheck_items(desktop, view):
            one = self.refresh(FrameKey(desktop, int(step), view), cache)
            for counter in ("updated", "conflicts", "skipped"):
                total[counter] += one[counter]
            total["problems"].extend(one["problems"])
            self.db.clear_recheck(desktop, view, step, gen)
        return total

    # ------------------------------------------------------- verify and demote

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

        The same entry :meth:`refresh` would have made, in its own transaction,
        so the refusal the caller is about to raise leaves the annotator with
        something to act on rather than a frame they cannot confirm and cannot
        see the reason for. The frame's digest goes with it: the rows keep their
        frozen values and therefore do not describe these inputs.
        """
        queued: Optional[list[dict]] = None
        with self.db.transaction():
            for instance in vanished:
                payload = row_payload(stored[instance])
                queued, _new = self._queue_conflict(
                    key, instance, payload, None, self._payload_area(payload), queued
                )
            self.db.clear_frame_digest(key)

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

    # ------------------------------------------------------ conflict resolution

    # -------------------------------------------------------------- keyframes

    def affected_steps(
        self, desktop: int, view: str, instance: str, keyframe: ShapeKeyframe
    ) -> list[int]:
        """The logical steps whose compilation would select ``keyframe``.

        A shape reaches a step when that step is in the keyframe's pose segment,
        the instance needs geometry there, it is in the placement the keyframe
        was drawn for, and :func:`tda.core.compiler.select_keyframe` picks this
        keyframe out of the chain -- i.e. exactly the frames an edit to this
        shape would change (spec 4.3).

        "Needs geometry there" is asked of :func:`tda.core.states.needs_geom`
        with this segment's bench ROI, exactly as :func:`gather` asks it: a
        bench box on a view with no staging area reaches no frame at all, and
        saying it reached four was a promise about pixels nobody compiles.
        """
        cache = InputCache()
        instances = instances_of(self.db, desktop, cache)
        events = events_of(self.db, self.tax, desktop, cache)
        bench_roi = self.db.bench_roi(desktop, view, keyframe.pose_segment)
        chain = [
            kf
            for kf in self.db.keyframes(desktop, view, instance)
            if kf.pose_segment == keyframe.pose_segment and kf.placement == keyframe.placement
        ]
        out: list[int] = []
        for rec in self.db.steps(desktop):
            step = rec.step
            key = FrameKey(desktop, step, view)
            if pose_segment_of(self.db, key, cache) != keyframe.pose_segment:
                continue
            state = state_of(self.db, self.tax, desktop, step, cache)
            if instance not in needs_geom(instances, state, self.tax,
                                          bench_roi=bench_roi):
                continue
            if state[instance].placement != keyframe.placement:
                continue
            if self._same_keyframe(select_keyframe(chain, step), keyframe):
                out.append(step)
        return sorted(out)

    # --------------------------------------------------------------- internals

    @staticmethod
    def _same_keyframe(chosen: Optional[ShapeKeyframe], wanted: ShapeKeyframe) -> bool:
        """Is the selected keyframe the one asked about? By id, or by identity."""
        if chosen is None:
            return False
        if chosen.id is not None and wanted.id is not None:
            return int(chosen.id) == int(wanted.id)
        return chosen is wanted

    def _put_row(
        self,
        key: FrameKey,
        instance: str,
        values,
        status: str,
        input_hash: str,
        verified_by: Optional[str] = None,
    ) -> None:
        """Write one truth row from the values :func:`row_values` derived."""
        self.db.put_compiled(
            key, instance, values.visible_rle, values.occlusion_ratio, values.visibility,
            values.placement, status, input_hash, verified_by=verified_by,
            geom_type=values.geom_type, box=values.box,
        )

    @staticmethod
    def _box_mask(box, hw: tuple[int, int]) -> np.ndarray:
        """A filled rectangle clipped to the canvas, ``x1``/``y1`` exclusive."""
        height, width = int(hw[0]), int(hw[1])
        mask = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask

    @staticmethod
    def _payload_area(payload: Optional[dict]) -> int:
        """How many pixels one conflict side covers -- its ``sym_diff_px`` alone."""
        geom_type, visible_rle, box = payload_geometry(payload)
        if geom_type == GEOM_BOX and box is not None:
            return int(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
        return 0 if visible_rle is None else masks.area(masks.decode_rle(visible_rle))

    def _queue_conflict(
        self,
        key: FrameKey,
        instance: str,
        old_payload: Optional[dict],
        new_payload: Optional[dict],
        pixels: int,
        queued: Optional[list[dict]],
    ) -> tuple[list[dict], bool]:
        """Queue one frozen-vs-recompiled disagreement, unless it is queued already.

        Deduplication is on the **open** queue only, keyed by
        ``(step, instance, new value)``: refreshing the same frame twice must
        not pile up copies of one disagreement, while a conflict a human already
        resolved may legitimately be raised again -- the resolution writes the
        inputs it settled on, so an identical conflict coming back means the
        inputs moved again.

        Returns ``(the queue it read, whether a row was inserted)``. The queue
        comes back so one refresh reads it at most once; the flag comes back
        because a caller that tells the annotator what happened must not say
        "queued again" about an insert the deduplication suppressed.
        """
        if queued is None:
            queued = self.db.conflicts(key.desktop, key.view, open_only=True)
        wanted = self._geom_id(new_payload)
        for conflict in queued:
            if (
                conflict["step"] == key.step
                and conflict["instance"] == instance
                and self._geom_id(conflict["new_rle"]) == wanted
            ):
                return queued, False
        self.db.add_conflict(key, instance, old_payload, new_payload, int(pixels))
        return queued, True

    @staticmethod
    def _geom_id(payload: Optional[dict]) -> Optional[str]:
        """The cheapest identity of a conflict side: its geometry and its labels.

        The labels are part of the identity because two disagreements about the
        same unchanged outline -- "it is occluded_partial", then "no, it is
        too_small" -- are two different things to decide, and deduplicating them
        onto one queue entry would lose the second.
        """
        if not payload:
            return None
        if "box" in payload:
            geom: Optional[str] = "box:" + ",".join(
                f"{float(v):.3f}" for v in payload["box"]
            )
        else:
            geom = masks.rle_counts(payload)
        labels = payload_labels(payload)
        if not labels:
            return geom
        tag = ";".join(f"{c.get('field')}={c.get('new')}" for c in labels)
        return f"{geom or ''}|{tag}"

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
