"""Closing one queued disagreement (spec 3.4, 保留旧值 / 接受新值 / 编辑).

A conflict is the one thing in the truth table a human has to settle: the
compiler may not overwrite a frozen row, so when a re-compilation disagrees with
one the disagreement is queued and the row keeps its value until somebody says
which of the two is right.  This module is that decision, split off
:mod:`tda.core.truth` only for size; the geometry the three answers are made of
is :mod:`tda.core.truth_conflicts`'.
"""
from __future__ import annotations

from typing import Optional

from tda.core import masks
from tda.core.db import RESOLUTIONS
from tda.core.model import FrameKey, FrameOverride, Placement, Visibility
from tda.core.truth_conflicts import (
    GEOM_BOX,
    disagreement,
    geom_payload,
    payload_geometry,
    payload_row,
    row_payload,
    row_values,
)
from tda.core.truth_inputs import frame_hw

__all__ = ["ACCEPT_NEW", "KEEP_OLD", "OPEN", "SUPERSEDED", "BY_HAND",
           "ResolveMixin", "StaleConflictError"]

ACCEPT_NEW = "accept_new"
KEEP_OLD = "keep_old"
EDITED = "edited"
SUPERSEDED = "superseded"
OPEN = "open"
VERIFIED = "verified"
ON_BENCH = Placement.ON_BENCH.value
#: The three a human may choose; ``superseded`` is the service's own answer.
BY_HAND = tuple(r for r in RESOLUTIONS if r != SUPERSEDED)


class StaleConflictError(ValueError):
    """A queued conflict the inputs have overtaken; it was re-queued, not accepted.

    Raised by :meth:`ResolveMixin.resolve_conflict` on ``accept_new`` when the
    frame no longer compiles to the value the conflict was queued with. The old
    conflict is closed as ``superseded`` and the current disagreement is in the
    queue by the time this reaches the caller, whose job is to show that one.
    """


class ResolveMixin:
    """Conflict resolution, for :class:`~tda.core.truth.TruthService`."""

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
            # whatever was decided, the rows may or may not describe the inputs
            # now (`edited` writes nothing at all), so the frame is left for the
            # next refresh to judge rather than stamped as up to date
            self.db.clear_frame_digest(key)
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
