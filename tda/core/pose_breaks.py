"""Where a view's pose segments are cut, and what a re-cut does to the numbers.

Spec 2.5 (v1.5) cuts a view's pose segments at the ``reorient`` steps -- which
the four views share, because the chassis is flipped for all of them -- **union
that view's own pose breaks**: the camera was knocked, or the chassis slid, in
front of one lens and not the others.  A break is a step number: the step the
new segment *starts* at.

This module is the arithmetic of that, with no database in sight:

* :func:`boundaries` folds the two sources into one sorted list of starts;
* :func:`recut_plan` turns "these are the segments now, these are the boundaries
  the view should have" into a :class:`RecutPlan` -- the new ranges, which old
  segment number becomes which new one, which old segment was split into
  several, and which steps are affected at all;
* :func:`straddling` answers the one question the split dialog asks the human:
  *which shapes does this boundary cut through?*

Numbers, not identities.  A pose segment is ``(desktop, view, seg)`` with ``seg``
counting from 1 in step order, so inserting a boundary renumbers every segment
after it -- and every row keyed by one (:mod:`tda.core.db_pose` moves them).  The
plan is computed whole before anything is written precisely because those two
things must not be able to disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = ["RecutPlan", "boundaries", "describe_discard", "describe_uncarried",
           "merge_orders", "recut_plan", "straddling"]

#: A break row's ``status``.  Only ``accepted`` ever cuts a segment.
PROPOSED = "proposed"
ACCEPTED = "accepted"
REJECTED = "rejected"
STATUSES: tuple[str, ...] = (PROPOSED, ACCEPTED, REJECTED)

#: A break row's ``kind``: what moved.
KIND_CAMERA = "camera"
KIND_CHASSIS = "chassis"
KIND_MANUAL = "manual"
KINDS: tuple[str, ...] = (KIND_CAMERA, KIND_CHASSIS, KIND_MANUAL)

#: ``shape_keyframe.source`` of a keyframe duplicated across a new boundary.
CARRIED = "carried"


def boundaries(n_steps: int, reorients: Iterable[int], accepted: Iterable[int]) -> list[int]:
    """Sorted unique segment starts of one view: ``reorients`` ∪ ``accepted``.

    A boundary is the step the *next* segment begins at, so step 1 is not one
    (segment 1 begins there anyway) and neither is anything past ``n_steps``.
    A ``reorient`` step that also carries a manual break yields **one**
    boundary, not an empty segment between the two.
    """
    everything = list(reorients) + list(accepted)
    return sorted({int(b) for b in everything if 1 < int(b) <= int(n_steps)})


@dataclass(frozen=True)
class RecutPlan:
    """What one view's segment table should look like, and how to get there.

    Attributes:
        ranges: ``[(seg, start, end)]`` of the view after the re-cut, ``seg``
            counting from 1.
        renumber: ``{old_seg: new_seg}``, the new segment that holds the old
            one's **first** step.  Several old segments map to one new one when
            a boundary was removed.
        split: ``{old_seg: [new_seg, ...]}`` for the old segments a new boundary
            cut in two or more; an old segment that came through whole is not
            listed.
        merged: ``{new_seg: [old_seg, ...]}``, the inverse of ``renumber``.
        old: the ranges this plan started from, so the caller can ask what moved.
    """

    ranges: list[tuple[int, int, int]]
    renumber: dict[int, int]
    split: dict[int, list[int]]
    merged: dict[int, list[int]] = field(default_factory=dict)
    old: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Is this a re-cut at all, or is the view already exactly like this?"""
        return (self.ranges != self.old
                or any(old != new for old, new in self.renumber.items()))

    def segment_of(self, step: int) -> Optional[int]:
        """The **new** segment a step falls in, or ``None`` when it falls outside."""
        return next((seg for seg, start, end in self.ranges if start <= step <= end), None)

    def old_segment_of(self, step: int) -> Optional[int]:
        """The segment the step was in before the re-cut, or ``None``."""
        return next((seg for seg, start, end in self.old if start <= step <= end), None)

    def affected_steps(self) -> set[int]:
        """Every step whose compiler inputs the re-cut can move (spec 3.3, 3.4).

        An old segment that keeps both its number and its range is untouched:
        its keyframes, its layer order and its ROI are the same rows and the
        frames in it hash the same.  Any other old segment affects **all** of
        its steps, on both sides of a new boundary -- the earlier side loses the
        keyframes whose anchor is now in the later one, which is exactly the
        "redraw it in the new pose" signal -- and so does a segment that merely
        got a different number, because the number is part of the frame digest.
        """
        new_by_seg = {seg: (start, end) for seg, start, end in self.ranges}
        out: set[int] = set()
        for seg, start, end in self.old:
            new_seg = self.renumber.get(seg)
            if new_seg is None or new_by_seg.get(new_seg) != (start, end):
                out.update(range(start, end + 1))
        return out


def recut_plan(old: list[tuple[int, int, int]], new_bounds: list[int],
               n_steps: int) -> RecutPlan:
    """Plan the re-cut of one view: ``old`` ranges + boundaries -> new ranges.

    ``old`` is ``[(seg, start, end)]`` as stored, ``new_bounds`` the **whole**
    boundary list the view should have (:func:`boundaries`), not a delta -- a
    re-cut is always a re-derivation, which is what makes running it twice a
    no-op and what lets ``load-index`` re-run it without knowing what a human
    added in between.

    A view with no segments yet gets no plan: there is nothing to re-key, and
    ``_seed_pose_segments`` creates segment 1 before this is ever asked.
    """
    old = [(int(s), int(a), int(b)) for s, a, b in sorted(old, key=lambda r: r[1])]
    if not old:
        return RecutPlan(ranges=[], renumber={}, split={}, merged={}, old=[])
    last = max(end for _seg, _start, end in old)
    n_steps = max(int(n_steps), last)
    starts = [1] + [b for b in sorted(set(new_bounds)) if 1 < b <= n_steps]
    ranges = [
        (i + 1, start, (starts[i + 1] - 1) if i + 1 < len(starts) else n_steps)
        for i, start in enumerate(starts)
    ]

    def holder(step: int) -> int:
        """The new segment a step falls in; the last one catches an overrun."""
        for seg, start, end in ranges:
            if start <= step <= end:
                return seg
        return ranges[-1][0]

    renumber = {seg: holder(start) for seg, start, _end in old}
    split: dict[int, list[int]] = {}
    for seg, start, end in old:
        pieces = [s for s, a, b in ranges if a <= end and b >= start]
        if len(pieces) > 1:
            split[seg] = pieces
    merged: dict[int, list[int]] = {}
    for seg, _start, _end in old:
        merged.setdefault(renumber[seg], []).append(seg)
    return RecutPlan(ranges=ranges, renumber=renumber, split=split, merged=merged, old=old)


def merge_orders(keeper: Iterable, other: Iterable) -> tuple[list, list]:
    """Fold two layer orders into one; returns ``(merged, changed pairs)``.

    Undoing a split merges two segments, and each may carry a
    :class:`~tda.core.model.ZOrderRec` -- a *total* order, which has no union.
    The rule, so that nothing silently disappears:

    * the keeper's order comes first, exactly as it is;
    * every layer key that exists only in ``other`` is **appended**, keeping
      their relative order among themselves.  Nothing drops out, so no instance
      falls into ``zorder_missing`` because two segments became one;
    * what genuinely could not be kept is the *relative* order of a pair that
      ``other`` ordered the other way round.  Those pairs -- and only those --
      come back as ``[a, b]`` meaning "``other`` had ``a`` above ``b``, the
      merged order does not".

    Identical orders therefore produce no changed pairs at all, which is what
    makes "the split you just undid" silent.
    """
    kept = [tuple(entry) for entry in keeper]
    rest = [tuple(entry) for entry in other]
    merged = kept + [entry for entry in rest if entry not in kept]
    rank = {entry: i for i, entry in enumerate(merged)}
    changed = [
        [list(a), list(b)]
        for i, a in enumerate(rest) for b in rest[i + 1:]
        if a in rank and b in rank and rank[a] > rank[b]
    ]
    return [list(entry) for entry in merged], changed


def describe_discard(view: str, item: dict) -> str:
    """The one sentence every caller says about something a re-cut could not keep.

    The pipeline writes it into ``pose_issues``, the command line prints it and
    the window puts it in the status bar, so an annotator cannot be told three
    different things about one event -- and a renderer that assumes the wrong
    shape cannot abort an import (``d['row']`` is a dict for a segment row and a
    list of layer keys for an order).
    """
    where = f"{view} pose segment {item['pose_segment']}"
    if item.get("orphan"):
        table = item["table"]
        if item.get("replaced"):
            return (f"{where}: a stray {table} row of a segment that no longer exists "
                    f"was replaced by segment {item['into']}'s own; it is in the op log")
        return (f"{where}: a stray {table} row of a segment that no longer exists was "
                f"left in place, and nothing reads it")
    if item["table"] == "zorder":
        pairs = item.get("changed_pairs") or []
        shown = "; ".join(f"{a[0]} over {b[0]}" for a, b in pairs[:3])
        more = f" (+{len(pairs) - 3} more)" if len(pairs) > 3 else ""
        return (f"{where}: merged into segment {item['into']}, which kept its own layer "
                f"order; {len(pairs)} pair(s) changed places: {shown}{more}")
    if item["table"] == "pose_segment":
        fields = ", ".join(sorted(item.get("row") or {}))
        return (f"{where}: merged into segment {item['into']}, which kept its own "
                f"{fields}; the discarded values are in the op log")
    return (f"{where}: merged into segment {item['into']}; the discarded values are in "
            f"the op log")


def describe_uncarried(kept: list) -> str:
    """The one sentence about carried shapes a merge had to keep, with the instances.

    "2 carried keyframes were kept" is not something anybody can act on;
    "psu.01 at step 18" is -- that is the frame to open and the anchor to look
    at.
    """
    named = ", ".join(f"{k['instance']} at step {k['anchor_step']}" for k in kept[:3])
    more = f" (+{len(kept) - 3} more)" if len(kept) > 3 else ""
    return (f"{len(kept)} carried shape(s) had been edited and were kept when the "
            f"break was removed: {named}{more}; check their anchors")


def straddling(anchors: Iterable[int], start: int, boundary: int) -> list[int]:
    """Anchors of one chain whose coverage crosses ``boundary``.

    A keyframe with anchor ``a`` applies to ``(previous anchor, a]`` (spec 3.3
    step 3), and the first one of a chain reaches back to the segment's own
    ``start``.  A new segment beginning at ``boundary`` therefore cuts the
    keyframe whose coverage has steps on both sides -- the frames before the
    boundary lose their shape and become ``missing_shape`` unless the annotator
    asks for it to be carried across.

    ``anchors`` is one chain: one ``(instance, placement, pose_segment)``.
    """
    ordered = sorted({int(a) for a in anchors})
    out = []
    previous = int(start) - 1
    for anchor in ordered:
        if anchor >= int(boundary) and previous <= int(boundary) - 2:
            out.append(anchor)
        previous = anchor
    return out
