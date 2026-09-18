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

import hashlib
from typing import Iterable, Optional

import numpy as np

from tda.core import masks
from tda.core.compiler import CompiledFrame, compile_frame, select_keyframe
from tda.core.db import RESOLUTIONS, Db
from tda.core.model import FrameKey, FrameOverride, Placement, ShapeKeyframe, Visibility
from tda.core.states import needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_conflicts import (
    BOX_TOL_PX,
    GEOM_BOX,
    disagreement,
    geom_payload,
    payload_geometry,
    payload_row,
    row_payload,
    row_values,
)
from tda.core.truth_inputs import (
    FrameInputs,
    annotatable_steps,
    InputCache,
    events_of,
    frame_hw,
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


class StaleConflictError(ValueError):
    """A queued conflict the inputs have overtaken; it was re-queued, not accepted.

    Raised by :meth:`TruthService.resolve_conflict` on ``accept_new`` when the
    frame no longer compiles to the value the conflict was queued with. The old
    conflict is closed as ``superseded`` and the current disagreement is in the
    queue by the time this reaches the caller, whose job is to show that one.
    """


class TruthService:
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
        compiled = compile_frame(
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
        )
        return inputs, compiled

    # ------------------------------------------------------------------ refresh

    def inputs_digest(self, key: FrameKey, cache: Optional[InputCache] = None) -> str:
        """A fingerprint of one frame's compiler inputs, computed without pixels.

        :attr:`CompiledFrame.input_hash` says the same thing but only exists
        after a compilation, which is the expensive half. This is the cheap half
        -- identities and versions, no mask is decoded -- so a worker can take it
        *before* the pixel work and check it again inside the write transaction:
        if it moved, the annotator edited the frame meanwhile and the result
        about to be written describes inputs nobody has any more (spec 3.4).
        """
        inputs = gather(self.db, self.tax, key, cache)
        parts = [
            f"{key.desktop}/{key.view}/{key.step}", str(inputs.hw),
            str(inputs.pose_segment), str(sorted(inputs.needs.items())),
            str(sorted(inputs.placements.items())),
            str((inputs.transform.scale, inputs.transform.theta,
                 inputs.transform.tx, inputs.transform.ty)),
            str((inputs.zorder.version, inputs.zorder.order)),
            str([(p.above, p.below) for p in inputs.overrides]),
            str(sorted((kf.instance, kf.id, kf.version, kf.anchor_step, kf.placement,
                        kf.pose_segment, kf.geom_type)
                       for chain in inputs.keyframes.values() for kf in chain)),
            str(sorted((o.occluder_type, (o.rle or {}).get("counts"))
                       for o in inputs.occluders)),
            str(sorted((i, (o.visible_rle or {}).get("counts"), o.visibility)
                       for i, o in inputs.frame_overrides.items())),
            self.compiler_version,
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    def refresh(self, key: FrameKey, cache: Optional[InputCache] = None,
                guard: Optional[str] = None) -> dict:
        """Bring one frame's truth rows up to date with the current inputs.

        Returns ``{"updated", "conflicts", "skipped", "problems", "compiled",
        "stale"}``: rows written (a deleted row counts as written), frozen rows
        found in disagreement, rows left alone, the compiler's problem list, the
        compilation the whole decision was made from -- handed back so that a
        caller which also needs the frame does not compile it a second time --
        and whether the write was abandoned.

        ``cache`` is :meth:`refresh_range`'s way of reading the step-independent
        inputs once; callers outside this module leave it out.

        ``guard`` is an :meth:`inputs_digest` taken before the compilation, for
        a caller working off the GUI thread. The pixel work happens outside any
        transaction; every write then happens inside **one**, which begins by
        taking the digest again. If it moved, nothing is written and ``stale``
        comes back true: the frame has to be looked at again against the inputs
        it has now, and a conflict describing the old ones would be a conflict
        nobody caused.
        """
        inputs, compiled = self._compile(key, cache)  # no transaction: pixels only
        with self.db.transaction():
            return self._write_refresh(key, compiled, guard)

    def _write_refresh(self, key: FrameKey, compiled: CompiledFrame,
                       guard: Optional[str]) -> dict:
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
            result["skipped"] = len(stored)  # nothing changed: no work at all
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
                if diff is None:
                    result["skipped"] += 1
                    continue
                values = row_values(compiled_inst)
                queued = self._queue_conflict(
                    key, instance, row_payload(row),
                    geom_payload(values.visible_rle, values.box), diff, queued,
                )
                result["conflicts"] += 1
                continue
            self._put_row(key, instance, row_values(compiled_inst), AUTO, new_hash)
            result["updated"] += 1

        for instance in sorted(known - fresh):
            row = stored[instance]
            if row["status"] == VERIFIED:
                old_payload = row_payload(row)
                queued = self._queue_conflict(
                    key, instance, old_payload, None, self._payload_area(old_payload), queued
                )
                result["conflicts"] += 1
            else:
                self.db.delete_compiled(key, instance)
                result["updated"] += 1

        if verified_frame and (fresh - known or known - fresh):
            self.demote_frame(key, self._demotion_reason(fresh, known))
        return result

    def refresh_range(self, desktop: int, view: str, steps: Iterable[int],
                      per_step: bool = False) -> dict:
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
            one = self.refresh(FrameKey(desktop, int(step), view), cache)
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
        total: dict = {"updated": 0, "conflicts": 0, "skipped": 0, "problems": []}
        for step, gen in self.db.recheck_items(desktop, view):
            one = self.refresh(FrameKey(desktop, int(step), view))
            for counter in ("updated", "conflicts", "skipped"):
                total[counter] += one[counter]
            total["problems"].extend(one["problems"])
            self.db.clear_recheck(desktop, view, step, gen)
        return total

    def ensure_fresh(self, desktop: int, view: str, only_verified: bool = False) -> dict:
        """Make a whole view's truth table complete before it is read out.

        The compiled rows of an unverified frame are a cache the annotator's
        commits deliberately leave stale (spec 3.4), so anything that reads the
        table as a whole -- an export, a quality check -- has to fill it first,
        or it would silently publish a frame as it was several edits ago, or
        drop one that was never visited at all.

        Frozen frames are re-checked whatever ``only_verified`` says, because
        that is where conflicts come from; the rest of the view is recompiled
        unless only the frozen rows are wanted. Raises ``RuntimeError`` if any
        re-check is still outstanding afterwards: an export must not run on a
        frozen frame nobody has compared.
        """
        total = self.run_pending_rechecks(desktop, view)
        if not only_verified:
            steps = annotatable_steps(
                self.db, desktop, view,
                [row["step"] for row in self.db.frames_for(desktop, view)],
            )
            total = self.refresh_range(desktop, view, steps)
        left = self.pending_rechecks(desktop, view)
        if left:
            raise RuntimeError(
                f"desktop {desktop} view {view}: {len(left)} frozen frame(s) still "
                f"await a truth re-check ({left[:5]}...); run them before exporting"
            )
        return total

    # ------------------------------------------------------- verify and demote

    def verify_frame(self, key: FrameKey, annotator: str) -> None:
        """Freeze every row of one frame after a human confirmed it (spec 4.2).

        Raises :class:`ValueError` when the compilation has a blocking problem
        (:data:`BLOCKING_PROBLEMS`) -- a frame with a missing chassis shape or a
        contradictory z-order is not a truth anybody can confirm. Warnings, such
        as a bench part nobody has boxed yet, do not stop the confirmation; they
        only leave ``bench_annotated`` false.

        Every write goes into one transaction: a frame is either confirmed
        whole -- rows, flag and op log -- or not at all.
        """
        _, compiled = self._compile(key)
        blocking = [p for p in compiled.problems if p.startswith(BLOCKING_PROBLEMS)]
        if blocking:
            raise ValueError(
                f"frame {key.desktop}/{key.view}/step {key.step} cannot be verified: "
                + ", ".join(blocking)
            )
        stored = self.db.compiled(key)
        previous = self._review_status(key)
        with self.db.transaction():
            for instance in sorted(compiled.instances):
                self._put_row(
                    key, instance, row_values(compiled.instances[instance]),
                    VERIFIED, compiled.input_hash, verified_by=annotator,
                )
            for instance in sorted(set(stored) - set(compiled.instances)):
                self.db.delete_compiled(key, instance)  # not in this frame at all
            self._mark_bench(key, compiled)
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

    # ------------------------------------------------------ conflict resolution

    def resolve_conflict(self, cid: int, resolution: str, annotator: str) -> None:
        """Close one queued disagreement, and make the inputs agree with it.

        ``accept_new`` confirms what the inputs say **now**: the frame is
        recompiled and the queued value is compared against that compilation
        with the rule that queued it. Agreeing (a re-trace within tolerance
        counts as agreeing) writes the *current* geometry into the frozen row,
        which stays ``verified``, stamped with the frame's current
        ``input_hash`` so the next refresh has nothing left to do; when the
        current compilation has no geometry for the instance at all, the row is
        removed instead. Disagreeing means the inputs moved on after the
        conflict was queued: accepting it would confirm something nobody has
        seen, so this conflict is closed as ``superseded``, the disagreement
        against the *current* value is queued in its place and
        :class:`StaleConflictError` is raised for the caller to re-present it.

        ``keep_old`` pins the frame: the frozen value is written back as a
        :class:`~tda.core.model.FrameOverride`, which is what "only this frame"
        means everywhere else in the tool (spec 4.3). The compiler then produces
        the frozen value again and the disagreement cannot come back, while the
        shape itself keeps whatever the annotator changed it to for every other
        frame. An existing override of that instance is merged, not replaced. A
        box row is pinned by the rectangle's mask, so the override stays a plain
        visible mask, and a frozen *absence* is pinned by the row's own
        ``visibility``. It is refused when the instance is no longer in the
        frame's state at all: presence is the step table's and the event log's
        decision, not an override's, so that conflict is settled by fixing the
        step (or by ``accept_new``, which drops the frozen row).

        ``edited`` only closes the conflict: whoever edited the row wrote it.
        """
        if resolution not in BY_HAND:
            raise ValueError(f"resolution must be one of {BY_HAND}, got {resolution!r}")
        conflict = self.db.get_conflict(cid)
        if conflict is None:
            raise KeyError(f"no conflict with id={cid}")
        if conflict["status"] != OPEN:
            raise ValueError(
                f"conflict {cid} is already resolved "
                f"({conflict['resolution'] or conflict['status']})"
            )
        key = FrameKey(conflict["desktop"], conflict["step"], conflict["view"])
        instance = conflict["instance"]
        stale: Optional[str] = None
        with self.db.transaction():
            if resolution == ACCEPT_NEW:
                stale = self._accept_new(key, instance, conflict, annotator)
            elif resolution == KEEP_OLD:
                self._keep_old(key, instance, conflict)
            settled = SUPERSEDED if stale else resolution
            self.db.resolve_conflict(cid, settled)
            self.db.log_op(
                key.desktop, key.view, "resolve_conflict",
                {"step": key.step, "instance": instance, "conflict": int(cid),
                 "resolution": settled},
                {"kind": "reopen_conflict", "conflict": int(cid)},
                annotator,
            )
        if stale:  # raised only once the supersession itself is committed
            raise StaleConflictError(stale)

    def _accept_new(
        self, key: FrameKey, instance: str, conflict: dict, annotator: str
    ) -> Optional[str]:
        """Confirm the current compilation, or report that it overtook the conflict.

        Returns ``None`` when the row was written (or removed), and the message
        of the :class:`StaleConflictError` the caller must raise otherwise.
        """
        _, compiled = self._compile(key)
        compiled_inst = compiled.instances.get(instance)
        queued = payload_row(conflict["new_rle"])
        if compiled_inst is None:
            if conflict["new_rle"] is not None:
                return self._supersede(key, instance, conflict, None)
            self.db.delete_compiled(key, instance)  # the instance is gone: so is the row
            return None
        if disagreement(queued, compiled_inst) is not None:
            return self._supersede(key, instance, conflict, compiled_inst)
        self._put_row(
            key, instance, row_values(compiled_inst), VERIFIED, compiled.input_hash,
            verified_by=annotator,
        )
        return None

    def _supersede(
        self, key: FrameKey, instance: str, conflict: dict, compiled_inst
    ) -> str:
        """Queue the disagreement against the current value; returns the message."""
        row = self.db.compiled(key).get(instance)
        if row is None:
            return (
                f"conflict {conflict['id']} is stale: {instance} has no truth row on "
                f"step {key.step} any more"
            )
        diff = None if compiled_inst is None else disagreement(row, compiled_inst)
        if compiled_inst is None:
            diff = self._payload_area(row_payload(row))
        if diff is None:
            return (
                f"conflict {conflict['id']} is stale: the inputs changed again and now "
                f"agree with the frozen row of {instance} on step {key.step}"
            )
        values = None if compiled_inst is None else row_values(compiled_inst)
        new_payload = (
            None if values is None else geom_payload(values.visible_rle, values.box)
        )
        self._queue_conflict(key, instance, row_payload(row), new_payload, diff, None)
        return (
            f"conflict {conflict['id']} is stale: the inputs changed after it was queued, "
            f"so the disagreement about {instance} on step {key.step} was queued again"
        )

    def _keep_old(self, key: FrameKey, instance: str, conflict: dict) -> None:
        """Pin the frozen value onto this frame so the compiler reproduces it."""
        _, compiled = self._compile(key)
        if instance not in compiled.instances:
            raise ValueError(
                f"{instance} is not in the state of desktop {key.desktop} / {key.view} / "
                f"step {key.step}, so keeping the frozen row cannot be expressed as a "
                f"frame override: fix the step's actions or events, or accept_new to drop "
                f"the frozen row"
            )
        geom_type, visible_rle, box = payload_geometry(conflict["old_rle"])
        if geom_type == GEOM_BOX and box is not None:
            visible_rle = masks.encode_rle(self._box_mask(box, frame_hw(self.db, key)))
        existing = self.db.frame_overrides(key).get(instance)
        if visible_rle is None:
            # the frozen row had no geometry at all: pin the label it carried
            row = self.db.compiled(key).get(instance) or {}
            self.db.set_frame_override(FrameOverride(
                key, instance,
                visible_rle=None if existing is None else existing.visible_rle,
                visibility=row.get("visibility") or Visibility.OUT_OF_VIEW.value,
            ))
            return
        self.db.set_frame_override(FrameOverride(
            key, instance,
            visible_rle=visible_rle,
            visibility=None if existing is None else existing.visibility,
        ))

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
        """
        cache = InputCache()
        instances = instances_of(self.db, desktop, cache)
        events = events_of(self.db, self.tax, desktop, cache)
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
            if instance not in needs_geom(instances, state, self.tax):
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
    ) -> list[dict]:
        """Queue one frozen-vs-recompiled disagreement, unless it is queued already.

        Deduplication is on the **open** queue only, keyed by
        ``(step, instance, new value)``: refreshing the same frame twice must
        not pile up copies of one disagreement, while a conflict a human already
        resolved may legitimately be raised again -- the resolution writes the
        inputs it settled on, so an identical conflict coming back means the
        inputs moved again. Returns the queue it read, so one refresh reads it
        at most once.
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
                return queued
        self.db.add_conflict(key, instance, old_payload, new_payload, int(pixels))
        return queued

    @staticmethod
    def _geom_id(payload: Optional[dict]) -> Optional[str]:
        """The cheapest identity of a conflict side: its counts string or its box."""
        if not payload:
            return None
        if "box" in payload:
            return "box:" + ",".join(f"{float(v):.3f}" for v in payload["box"])
        counts = payload.get("counts")
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return None if counts is None else str(counts)

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
            return
        annotated = not any(p.startswith(BENCH_MISSING) for p in compiled.problems)
        frame = self.db.get_frame(key)
        if frame is not None and frame.get("bench_annotated") is annotated:
            return  # unchanged: a refresh that changes nothing writes nothing
        self.db.set_frame_flags(key, bench_annotated=annotated)
