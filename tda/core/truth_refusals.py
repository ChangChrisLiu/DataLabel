"""Why a frame cannot be confirmed right now, as data (task U2e).

:meth:`tda.core.truth_verify.VerifyMixin.verify_frame` refuses a frame for four
reasons, checked in this order:

1. an **open conflict** about the frame -- a disagreement only a human settles;
2. a **blocking compiler problem** (:data:`BLOCKING_PROBLEMS`);
3. a **frozen disagreement** -- a row somebody confirmed that the current inputs
   no longer produce (another frame changed the part's shape since);
4. an **input race** -- an edit landed while the confirmation was being made.

The window used to find out about the first and the third only by pressing
Space: the pane under the task card listed compiler codes, so a frame with an
open conflict showed nothing and the guide said "按 Space" -- the walk that
makes that likely is fixing a part with Enter on frame j, which reaches frames
already confirmed, whose re-check opens a conflict.

So the reasons are computed in one place, :func:`blocking_reasons`, and both
``verify_frame`` and the session's arrival read it.  Each reason is a
:class:`Blocker` with a *code* shaped like a compiler problem
(``open_conflict:12/chassis``, ``frozen_disagreement:chassis``,
``inputs_changed:``), so a panel lists it next to ``missing_shape:…`` and
:func:`is_blocking` counts it with them.  The race cannot be seen coming; it is
the one reason only a refusal reports (:func:`inputs_changed`).

**Two modes, one function** (U2e round 2).  ``verify_frame`` asks
``thorough=True``: its own order and early stops -- an open conflict refuses
before anything is compiled or decoded -- and every confirmed row behind the
inputs is compared.  The display asks ``thorough=False``: every category, and
the frozen rows are compared only when the frame's stored digest does not
already rule a disagreement out.  Comparing means decoding masks, and in the
normal reverse walk every confirmed frame is *behind but agreeing* (the first
commit of any new part moves the segment's layer order, and an agreeing
re-check leaves the old ``input_hash`` on purpose): decoding them on every
announce put 2 s on a 12 MP arrival.  :func:`frozen_disagreements` is the one
comparison; ``refresh`` queues from it too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from tda.core.compiler import CompiledFrame
from tda.core.model import FrameKey
from tda.core.truth_conflicts import (
    disagreement,
    geom_payload,
    label_changes,
    label_text,
    row_payload,
    row_values,
)

__all__ = [
    "BLOCKING_PROBLEMS", "Blocker", "CONFLICT", "FROZEN", "FROZEN_DISAGREEMENT",
    "INPUTS_CHANGED", "OPEN_CONFLICT", "PROBLEM", "RACE", "REFUSAL_CODES",
    "VerifyRefused", "blocking_reasons", "conflict_of", "frozen_disagreements",
    "inputs_changed", "is_blocking",
]

#: Problem prefixes that stop a frame from being verified (spec 3.3 step 3).
#: Everything else -- ``bench_missing``, ``zorder_missing``, ``empty_visible``,
#: ``pose_segment_ambiguous`` -- is a warning the annotator may accept.
BLOCKING_PROBLEMS = ("missing_shape:", "zorder_cycle:", "shape_size_mismatch:")

#: The codes of the three refusals that are not compiler problems.
OPEN_CONFLICT = "open_conflict:"
FROZEN_DISAGREEMENT = "frozen_disagreement:"
INPUTS_CHANGED = "inputs_changed:"
REFUSAL_CODES = (OPEN_CONFLICT, FROZEN_DISAGREEMENT, INPUTS_CHANGED)

#: :attr:`Blocker.kind`, one per reason, in the order ``verify_frame`` checks them.
CONFLICT, PROBLEM, FROZEN, RACE = "conflict", "problem", "frozen", "race"

VERIFIED = "verified"


def is_blocking(problem: str) -> bool:
    """Does this line stop the frame from being verified?

    A blocking compiler problem or one of the refusals of :func:`blocking_reasons`.
    The one test ``verify_frame`` applies, and the one the task card and the
    guide ask when they say what stands between the annotator and ``Space``
    (tasks U2d, U2e) -- so the two can never name different sets.
    """
    return str(problem).startswith(BLOCKING_PROBLEMS + REFUSAL_CODES)


@dataclass(frozen=True)
class Blocker:
    """One reason the frame cannot be confirmed now."""

    #: :data:`CONFLICT`, :data:`PROBLEM`, :data:`FROZEN` or :data:`RACE`.
    kind: str
    #: What a panel lists it under; :func:`is_blocking` is true for it.
    code: str
    #: The refusal's own words for it (English, the op-log's language).
    text: str
    instance: str = ""
    conflict_id: Optional[int] = None
    #: A frozen disagreement's ``(old payload, new payload, pixels or None)``:
    #: what ``verify_frame`` queues its conflict with.
    queue: Optional[tuple] = field(default=None, compare=False, repr=False)


class VerifyRefused(ValueError):
    """``verify_frame``'s refusal, with the reasons it was refused for."""

    def __init__(self, message: str, blockers: list[Blocker]) -> None:
        super().__init__(message)
        self.blockers = list(blockers)


def inputs_changed() -> Blocker:
    """The input race: an edit landed while the frame was being confirmed.

    Transient, and only ever known after the fact: pressing Space again
    confirms the frame against the inputs it has now.
    """
    return Blocker(RACE, INPUTS_CHANGED, "the inputs changed while confirming; "
                                         "press Space again")


def conflict_of(code: str) -> Optional[tuple[int, str]]:
    """``(conflict id, instance)`` of an ``open_conflict:`` code, else ``None``."""
    if not str(code).startswith(OPEN_CONFLICT):
        return None
    cid, _sep, instance = str(code)[len(OPEN_CONFLICT):].partition("/")
    try:
        return int(cid), instance
    except ValueError:
        return None


#: What :func:`blocking_reasons` gets the frame from: a zero-argument callable
#: returning ``(compiled, frame overrides, stored truth rows)`` -- either of the
#: last two ``None`` to have it read here.  A callable, so that an open conflict
#: refuses a confirmation before anything is compiled (main's order).
FrameSource = Callable[[], tuple]


def blocking_reasons(db, key: FrameKey, frame: FrameSource, *, thorough: bool,
                     ruled_out: Optional[Callable[[CompiledFrame], bool]] = None
                     ) -> list[Blocker]:
    """Why ``verify_frame`` refuses this frame now, as :class:`Blocker` items.

    ``thorough=True`` is ``verify_frame``'s check, in its order, and the
    answer is the first category that has anything: open conflicts (``frame``
    is not even called), else blocking compiler problems, else every confirmed
    row behind the inputs compared (:func:`frozen_disagreements`).

    ``thorough=False`` is what the window shows: every category at once.  A
    frozen disagreement about a part that already has an open conflict is left
    out -- settling the conflict settles it -- so each part is named once.  And
    the confirmed rows are compared only when they can disagree at all: some
    row is behind these inputs (one indexed query), and ``ruled_out(compiled)``
    -- the frame's stored digest is current for them -- is false.  That rule
    holds because a digest is only ever stamped when no disagreement stands,
    and cleared whenever one is queued (see the U2e report's round 2).

    Nothing is written: a frozen disagreement is only *queued*, by the
    confirmation or by a refresh.  The race is not here: :func:`inputs_changed`.
    """
    conflicts = [Blocker(CONFLICT, f"{OPEN_CONFLICT}{cid}/{instance}",
                         f"conflict {cid} ({instance}) is still open", instance, cid)
                 for cid, instance in db.open_conflicts_at(key)]
    if thorough and conflicts:
        return conflicts
    compiled, overrides, stored = frame()
    problems = [Blocker(PROBLEM, p, p) for p in compiled.problems
                if str(p).startswith(BLOCKING_PROBLEMS)]
    if thorough:
        return problems or frozen_disagreements(db, key, compiled, overrides, stored)
    if stored is None and not db.frozen_rows_behind(key, compiled.input_hash):
        frozen: list[Blocker] = []          # the common case, one indexed query
    elif ruled_out is not None and ruled_out(compiled):
        frozen = []                         # behind, but provably agreeing
    else:
        frozen = frozen_disagreements(db, key, compiled, overrides, stored)
    disputed = {b.instance for b in conflicts}
    return conflicts + problems + [b for b in frozen if b.instance not in disputed]


def frozen_disagreements(db, key: FrameKey, compiled: CompiledFrame,
                         overrides: Optional[dict] = None,
                         stored: Optional[dict] = None) -> list[Blocker]:
    """The confirmed rows these inputs no longer produce, one per instance.

    The one comparison: ``verify_frame`` refuses on it and
    :meth:`~tda.core.truth.TruthService.refresh` queues from it, so the two
    cannot drift.  A confirmed row whose instance left the frame comes first,
    then one whose geometry or labels moved beyond the re-tracing tolerance --
    the order ``verify_frame`` has always queued them in.  A row carrying this
    compilation's own ``input_hash`` *is* this compilation and is skipped
    without decoding anything.
    """
    if stored is None:
        stored = db.compiled(key)
    if overrides is None:
        overrides = db.frame_overrides(key)
    out: list[Blocker] = []
    for instance in sorted(set(stored) - set(compiled.instances)):
        if stored[instance]["status"] != VERIFIED:
            continue
        out.append(Blocker(FROZEN, f"{FROZEN_DISAGREEMENT}{instance}",
                           f"{instance} is no longer in this frame", instance,
                           queue=(row_payload(stored[instance]), None, None)))
    for instance in sorted(set(stored) & set(compiled.instances)):
        row = stored[instance]
        if row["status"] != VERIFIED or row["input_hash"] == compiled.input_hash:
            continue
        compiled_inst = compiled.instances[instance]
        diff = disagreement(row, compiled_inst)
        labels = label_changes(row, compiled_inst, overrides.get(instance))
        if diff is None and not labels:
            continue
        what = label_text(labels) if labels else f"{diff} px differ"
        values = row_values(compiled_inst)
        out.append(Blocker(
            FROZEN, f"{FROZEN_DISAGREEMENT}{instance}",
            f"the confirmed {instance} no longer agrees ({what})", instance,
            queue=(row_payload(row), geom_payload(values.visible_rle, values.box, labels),
                   int(diff or 0)),
        ))
    return out
