"""Which instances are still missing a shape, per frame, without compiling.

Two things used to need a compilation each: the ``missing_shape`` queue
(spec 4.4) and telling an ``unlabeled`` frame from an ``auto`` one in the
timeline (spec 4.5).  Both were paid for out of the stored ``compiled_mask``
rows, which is why every frame in an edit's interval had to be recompiled before
the annotator could carry on -- 13 s for a chassis shape at 1600x1600.

But neither question is about pixels.  "Does this instance have a shape here?"
is answered by the state machine and the keyframe chains alone: which instances
need geometry at this step, and does a keyframe of the right chain cover it.
That is a few hundred dictionary lookups for a whole view, so the session can
keep the answer for every step and throw it away whenever a shape changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from tda.core.compiler import select_keyframe
from tda.core.db import Db
from tda.core.model import FrameKey, ShapeKeyframe
from tda.core.states import needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import (
    InputCache,
    frame_hw as hw_of,
    instances_of,
    pose_segment_of,
    state_of,
)

__all__ = ["FrameCoverage", "coverage"]


@dataclass
class FrameCoverage:
    """What one frame has and has not been drawn yet."""

    step: int
    #: Instances needing a mask inside the chassis that have no keyframe here.
    missing: list[str] = field(default_factory=list)
    #: The same for parts on the bench, which never block a confirmation.
    bench_missing: list[str] = field(default_factory=list)
    #: How many instances of this frame do have geometry.
    drawn: int = 0
    #: How many need it at all.
    needed: int = 0

    @property
    def annotated(self) -> bool:
        """Has anything been drawn for this frame yet (``auto`` vs ``unlabeled``)?"""
        return self.drawn > 0


def coverage(db: Db, tax: Taxonomy, desktop: int, view: str, steps: Iterable[int],
             cache: Optional[InputCache] = None) -> dict[int, FrameCoverage]:
    """``step -> FrameCoverage`` for a whole view, reading no compiled row.

    The keyframe chains are read once and indexed by
    ``(instance, pose segment, placement)``, so the per-step work is one
    :func:`~tda.core.compiler.select_keyframe` per instance that needs geometry.
    """
    cache = cache if cache is not None else InputCache()
    instances = instances_of(db, desktop, cache)
    chains = _chains(db, desktop, view)
    out: dict[int, FrameCoverage] = {}
    for step in steps:
        step = int(step)
        seg = pose_segment_of(db, FrameKey(desktop, step, view), cache)
        state = state_of(db, tax, desktop, step, cache)
        found = FrameCoverage(step=step)
        for instance, kind in sorted(needs_geom(instances, state, tax).items()):
            placement = state[instance].placement
            found.needed += 1
            chosen = select_keyframe(chains.get((instance, seg, placement), []), step)
            if chosen is not None and _usable(chosen, hw_of(db, FrameKey(desktop, step, view))):
                found.drawn += 1
            elif kind == "box":
                found.bench_missing.append(instance)
            else:
                found.missing.append(instance)
        out[step] = found
    return out


def _usable(kf: ShapeKeyframe, hw: tuple[int, int]) -> bool:
    """Would the compiler get geometry out of this keyframe on this canvas?

    A part whose RLE was traced on a different canvas is reported as
    ``shape_size_mismatch`` and dropped (spec 3.3 step 4), so counting it as
    "drawn" would hide a frame that in fact has nothing on it.
    """
    for part in kf.parts:
        if part.box is not None:
            return True
        size = (part.rle or {}).get("size")
        if size is not None and (int(size[0]), int(size[1])) == hw:
            return True
    return False


def _chains(db: Db, desktop: int, view: str
            ) -> dict[tuple[str, int, str], list[ShapeKeyframe]]:
    """Every keyframe of one view, indexed the way the compiler narrows them."""
    out: dict[tuple[str, int, str], list[ShapeKeyframe]] = {}
    for kf in db.keyframes(desktop, view):
        out.setdefault((kf.instance, kf.pose_segment, kf.placement), []).append(kf)
    return out
