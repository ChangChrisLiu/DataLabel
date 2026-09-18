"""The persisted truth table: refresh, verification and conflicts (spec 3.4).

:mod:`tda.core.compiler` derives one frame's geometry from explicit inputs;
:class:`TruthService` is what turns that pure function into the stored
``compiled_mask`` table:

* :meth:`TruthService.compile`  -- gather this frame's inputs and compile it.
* :meth:`TruthService.refresh`  -- write the result into the truth table.
* :meth:`TruthService.verify_frame` / :meth:`TruthService.demote_frame`
  -- freeze a frame a human confirmed, or send it back for review.
* :meth:`TruthService.affected_steps` -- which frames one shape reaches, for
  the canvas' "affects N frames" strip (spec 4.3).

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
  it. Nothing is ever silently overwritten, and a disagreement that is already
  queued is not queued a second time.
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
from tda.core.compiler import CompiledFrame, CompiledInstance, compile_frame, select_keyframe
from tda.core.db import Db
from tda.core.model import FrameKey, Placement, ShapeKeyframe
from tda.core.states import needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import (
    FrameInputs,
    InputCache,
    events_of,
    gather,
    instances_of,
    pose_segment_of,
    state_of,
)

__all__ = ["BLOCKING_PROBLEMS", "BOX_TOL_PX", "TruthService"]

#: Problem prefixes that stop a frame from being verified (spec 3.3 step 3).
#: Everything else -- ``bench_missing``, ``zorder_missing``, ``empty_visible``,
#: ``pose_segment_ambiguous`` -- is a warning the annotator may accept.
BLOCKING_PROBLEMS = ("missing_shape:", "zorder_cycle:", "shape_size_mismatch:")

#: How far a box corner may move before a frozen box row is in conflict.
BOX_TOL_PX = 2

AUTO = "auto"
VERIFIED = "verified"
NEEDS_REVIEW = "needs_review"
ON_BENCH = Placement.ON_BENCH.value
BENCH_MISSING = "bench_missing:"
SYSTEM = "system"


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

    def refresh(self, key: FrameKey, cache: Optional[InputCache] = None) -> dict:
        """Bring one frame's truth rows up to date with the current inputs.

        Returns ``{"updated", "conflicts", "skipped", "problems"}``: rows
        written (a deleted row counts as written), frozen rows found in
        disagreement, rows left alone, and the compiler's problem list.

        ``cache`` is :meth:`refresh_range`'s way of reading the step-independent
        inputs once; callers outside this module leave it out.
        """
        inputs, compiled = self._compile(key, cache)
        stored = self.db.compiled(key)
        new_hash = compiled.input_hash
        result: dict = {
            "updated": 0,
            "conflicts": 0,
            "skipped": 0,
            "problems": list(compiled.problems),
        }
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
            visible_rle, ratio, visibility, placement = self._row_values(compiled_inst, inputs.hw)
            row = stored.get(instance)
            if row is not None and row["input_hash"] == new_hash:
                result["skipped"] += 1
                continue
            if row is not None and row["status"] == VERIFIED:
                diff = self._disagreement(row["visible_rle"], compiled_inst, inputs.hw)
                if diff is None:
                    result["skipped"] += 1
                    continue
                queued = self._queue_conflict(
                    key, instance, row["visible_rle"], visible_rle, diff, queued
                )
                result["conflicts"] += 1
                continue
            self.db.put_compiled(
                key, instance, visible_rle, ratio, visibility, placement, AUTO, new_hash
            )
            result["updated"] += 1

        for instance in sorted(known - fresh):
            row = stored[instance]
            if row["status"] == VERIFIED:
                old_rle = row["visible_rle"]
                pixels = 0 if old_rle is None else masks.area(masks.decode_rle(old_rle))
                queued = self._queue_conflict(key, instance, old_rle, None, pixels, queued)
                result["conflicts"] += 1
            else:
                self._delete_row(key, instance)
                result["updated"] += 1

        if verified_frame and (fresh - known or known - fresh):
            self.demote_frame(key, self._demotion_reason(fresh, known))
        return result

    def refresh_range(self, desktop: int, view: str, steps: Iterable[int]) -> dict:
        """:meth:`refresh` every step of one view, with the totals summed up.

        ``problems`` is the concatenation of the per-step problem lists in the
        order the steps were given.
        """
        cache = InputCache()
        total: dict = {"updated": 0, "conflicts": 0, "skipped": 0, "problems": []}
        for step in steps:
            one = self.refresh(FrameKey(desktop, int(step), view), cache)
            for counter in ("updated", "conflicts", "skipped"):
                total[counter] += one[counter]
            total["problems"].extend(one["problems"])
        return total

    # ------------------------------------------------------- verify and demote

    def verify_frame(self, key: FrameKey, annotator: str) -> None:
        """Freeze every row of one frame after a human confirmed it (spec 4.2).

        Raises :class:`ValueError` when the compilation has a blocking problem
        (:data:`BLOCKING_PROBLEMS`) -- a frame with a missing chassis shape or a
        contradictory z-order is not a truth anybody can confirm. Warnings, such
        as a bench part nobody has boxed yet, do not stop the confirmation; they
        only leave ``bench_annotated`` false.
        """
        inputs, compiled = self._compile(key)
        blocking = [p for p in compiled.problems if p.startswith(BLOCKING_PROBLEMS)]
        if blocking:
            raise ValueError(
                f"frame {key.desktop}/{key.view}/step {key.step} cannot be verified: "
                + ", ".join(blocking)
            )
        stored = self.db.compiled(key)
        for instance in sorted(compiled.instances):
            visible_rle, ratio, visibility, placement = self._row_values(
                compiled.instances[instance], inputs.hw
            )
            self.db.put_compiled(
                key, instance, visible_rle, ratio, visibility, placement, VERIFIED,
                compiled.input_hash, verified_by=annotator,
            )
        for instance in sorted(set(stored) - set(compiled.instances)):
            self._delete_row(key, instance)  # confirmed away: it is not in this frame
        self._mark_bench(key, compiled)
        previous = self._review_status(key)
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

    def _row_values(
        self, compiled_inst: CompiledInstance, hw: tuple[int, int]
    ) -> tuple[Optional[dict], float, str, str]:
        """One truth row's column values: RLE, occlusion ratio, label, placement.

        The truth table holds one geometry column, so a box-only instance (a
        part lying on the bench) stores its box as a filled rectangle: the
        corners come back exactly through :func:`tda.core.masks.bbox`, and every
        consumer -- export, review, the conflict queue -- reads one kind of
        geometry instead of two.
        """
        visible = compiled_inst.visible
        if visible is not None:
            rle = masks.encode_rle(visible)
        elif compiled_inst.box is not None:
            rle = masks.encode_rle(self._box_mask(compiled_inst.box, hw))
        else:
            rle = None
        return (
            rle,
            float(compiled_inst.occlusion_ratio),
            compiled_inst.visibility,
            compiled_inst.placement,
        )

    @staticmethod
    def _clip_box(box, hw: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
        """``box`` rounded to whole pixels and clipped to the canvas, or ``None``."""
        height, width = int(hw[0]), int(hw[1])
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
        return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)

    @classmethod
    def _box_mask(cls, box, hw: tuple[int, int]) -> np.ndarray:
        """A filled rectangle, ``x1``/``y1`` exclusive like every other box."""
        mask = np.zeros((int(hw[0]), int(hw[1])), dtype=bool)
        clipped = cls._clip_box(box, hw)
        if clipped is not None:
            x0, y0, x1, y1 = clipped
            mask[y0:y1, x0:x1] = True
        return mask

    @staticmethod
    def _is_box(compiled_inst: CompiledInstance) -> bool:
        """Box-only geometry: a box, no mask and no amodal shape behind it."""
        return (
            compiled_inst.visible is None
            and compiled_inst.amodal is None
            and compiled_inst.box is not None
        )

    def _disagreement(
        self, old_rle: Optional[dict], compiled_inst: CompiledInstance, hw: tuple[int, int]
    ) -> Optional[int]:
        """Differing pixels between a frozen row and a re-compilation, or ``None``.

        ``None`` means "close enough to be the same annotation": within the
        re-tracing tolerance of :func:`tda.core.masks.is_conflict` for masks, or
        within :data:`BOX_TOL_PX` on every corner for boxes. Geometry appearing
        or disappearing always counts, and so does a change of canvas size,
        which no comparison could survive.
        """
        if self._is_box(compiled_inst):
            return self._box_disagreement(old_rle, compiled_inst.box, hw)
        old = None if old_rle is None else masks.decode_rle(old_rle)
        new = compiled_inst.visible
        if old is None and new is None:
            return None
        if old is None:
            return masks.area(new)
        if new is None:
            return masks.area(old)
        if old.shape != new.shape:
            return max(masks.area(old), masks.area(new))
        return masks.tolerant_sym_diff(old, new) if masks.is_conflict(old, new) else None

    def _box_disagreement(
        self, old_rle: Optional[dict], box, hw: tuple[int, int]
    ) -> Optional[int]:
        """The same question for box geometry: has a corner moved more than the tolerance?"""
        old_box = None if old_rle is None else masks.bbox(masks.decode_rle(old_rle))
        new_box = None if box is None else self._clip_box(box, hw)
        if old_box is None and new_box is None:
            return None
        if old_box is None or new_box is None:
            present = new_box or old_box
            return int((present[2] - present[0]) * (present[3] - present[1]))
        if max(abs(a - b) for a, b in zip(old_box, new_box)) <= BOX_TOL_PX:
            return None
        return self._box_sym_diff(old_box, new_box)

    @staticmethod
    def _box_sym_diff(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        """Pixels covered by exactly one of two axis-aligned boxes."""
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        return int(area_a + area_b - 2 * overlap)

    def _queue_conflict(
        self,
        key: FrameKey,
        instance: str,
        old_rle: Optional[dict],
        new_rle: Optional[dict],
        pixels: int,
        queued: Optional[list[dict]],
    ) -> list[dict]:
        """Queue one frozen-vs-recompiled disagreement, unless it already is.

        The queue is deduplicated on ``(step, instance, new value)``: refreshing
        the same frame twice must not pile up copies of one disagreement, and a
        conflict a human already dealt with does not come back while the inputs
        still say the same thing. Returns the conflict list it read, so one
        refresh reads it at most once.
        """
        if queued is None:
            queued = self.db.conflicts(key.desktop, key.view, open_only=False)
        wanted = self._counts(new_rle)
        for conflict in queued:
            if (
                conflict["step"] == key.step
                and conflict["instance"] == instance
                and self._counts(conflict["new_rle"]) == wanted
            ):
                return queued
        self.db.add_conflict(key, instance, old_rle, new_rle, int(pixels))
        return queued

    @staticmethod
    def _counts(rle: Optional[dict]) -> Optional[str]:
        """An RLE's ``counts`` string, the cheapest identity of a mask."""
        if not rle:
            return None
        counts = rle.get("counts")
        return counts.decode("ascii") if isinstance(counts, bytes) else counts

    def _delete_row(self, key: FrameKey, instance: str) -> None:
        """Drop one truth row; the repository has no delete for these yet."""
        with self.db.conn:
            self.db.conn.execute(
                "DELETE FROM compiled_mask WHERE desktop=? AND step=? AND view=? AND instance=?",
                (key.desktop, key.step, key.view, instance),
            )

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
