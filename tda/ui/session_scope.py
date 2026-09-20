"""Which scope an edit means, read off the pixels that changed (spec 4.3).

The annotator paints; what they *meant* is a separate question, and spec 4.3
answers it from where the changed pixels landed.  Adding pixels under an
instance that is painted over this one is a statement about layering, not about
the silhouette; erasing exactly where another shape lies under this one is the
same statement the other way round; anything else is the shape itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tda.core import masks as _masks
from tda.core.compiler import CompiledFrame
from tda.ui import session_api as api

Box = tuple[int, int, int, int]

__all__ = ["SCOPE_SPLIT_PREFIX", "SCOPE_ZORDER_ABOVE", "SCOPE_ZORDER_BELOW",
           "ZORDER_HINT_FRAC", "split_zorder_scope", "splits_the_shape",
           "suggest_scope"]

#: How much of the *changed* pixels has to land inside another instance before
#: the default scope becomes "change the layering" (spec 4.3).
ZORDER_HINT_FRAC = 0.6

#: ``suggest_scope`` answers, and what ``commit_edit`` accepts besides the three
#: :data:`~tda.ui.session_api.COMMIT_SCOPES`.
SCOPE_ZORDER_ABOVE = "zorder:above:"
SCOPE_ZORDER_BELOW = "zorder:below:"

#: ``split+zorder:above:<B>`` -- the same layering answer, but cutting a new
#: version of the shape instead of re-tracing the keyframe in force.  ``Ctrl+K``
#: on an open suggestion means this: without the pair it wrote the pixels and
#: left them hidden under ``B``, so the screen did not change at all.
SCOPE_SPLIT_PREFIX = "split+"


def splits_the_shape(scope: str) -> bool:
    """Does this layering scope ask for a new shape version rather than a re-trace?"""
    return str(scope).startswith(SCOPE_SPLIT_PREFIX)


def split_zorder_scope(scope: str) -> Optional[tuple[str, bool]]:
    """``(other instance, this one goes above)`` for a layering scope, else ``None``."""
    scope = str(scope)
    if splits_the_shape(scope):
        scope = scope[len(SCOPE_SPLIT_PREFIX):]
    for prefix, above in ((SCOPE_ZORDER_ABOVE, True), (SCOPE_ZORDER_BELOW, False)):
        if scope.startswith(prefix):
            return scope[len(prefix):], above
    return None


def suggest_scope(compiled: CompiledFrame, instance: str, before: np.ndarray,
                  edited: np.ndarray) -> str:
    """The scope an edit defaults to (spec 4.3, 默认触发).

    The question is about the pixels that actually *changed*, never about the
    whole shape -- loading an instance into the editing layer and touching
    nothing is not a statement about anything:

    * pixels **added** where another instance ``B`` currently paints *over* this
      one say "I want to see this one there instead" -- i.e. put it above ``B``;
    * pixels **erased** exactly where ``B``'s shape lies under this one say the
      opposite: ``B`` should have been on top all along;
    * anything else is a change to the silhouette, so it edits the keyframe.

    Both answers name the pair and the direction, ``zorder:above:<B>`` and
    ``zorder:below:<B>``, because "change the layering" alone does not say which
    way round.  A hint is only given when at least
    :data:`ZORDER_HINT_FRAC` of the changed pixels fall inside ``B``.

    **Where it looks.**  The answer is about the pixels that changed, so the
    box those pixels live in is the only part of the canvas that can carry it,
    and each instance's own window is the only part *of that box* the instance
    can be in.  The comparison runs inside the two, which is the same answer
    for a great deal less reading: at 4032x3040 with forty instances the
    whole-canvas version was 1.09 s of a 1.9 s ``Enter`` -- eighty-four 12 MP
    ``&`` passes to explain a brush stroke.
    """
    before = np.asarray(before, dtype=bool)
    edited = np.asarray(edited, dtype=bool)
    window = _masks.bbox(before ^ edited)
    if window is None:      # nothing changed: not a statement about anything
        return api.SCOPE_KEYFRAME
    here, there = _crop(before, window), _crop(edited, window)
    added, erased = there & ~here, here & ~there

    covering = _partner(compiled, instance, added, window, above=True)
    if covering is not None:
        return f"zorder:above:{covering}"
    covered = _partner(compiled, instance, erased, window, above=False)
    if covered is not None:
        return f"zorder:below:{covered}"
    return api.SCOPE_KEYFRAME


def _crop(mask: np.ndarray, box: Box) -> np.ndarray:
    """``mask`` inside ``box`` -- a view, so nothing is copied."""
    x0, y0, x1, y1 = box
    return mask[y0:y1, x0:x1]


def _intersect(a: Optional[Box], b: Optional[Box]) -> Optional[Box]:
    """The box both cover, or ``None`` when they do not meet.

    ``None`` on the way *in* means "no bound known", which is not the same as
    "empty": a caller that cannot say where its pixels are must be compared
    everywhere the other one allows.
    """
    if a is None:
        return b
    if b is None:
        return a
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return None if box[2] <= box[0] or box[3] <= box[1] else box


def _partner(compiled: CompiledFrame, instance: str, changed: np.ndarray,
             window: Box, above: bool) -> Optional[str]:
    """The instance the changed pixels are a layering statement about, if any.

    ``above=True`` looks for an instance painting *over* ``instance`` whose
    visible pixels the edit reached into; ``above=False`` for one painting
    *under* it, compared on the amodal shapes, since what lies under is by
    definition not visible.

    ``changed`` is the changed pixels **cropped to** ``window``; each instance
    is then read only where its own window and ``window`` overlap.  An instance
    with no window is read across the whole of ``window``, so a compiled frame
    that cannot say where an instance is still gets the same answer.
    """
    total = int(changed.sum())
    if total == 0:
        return None
    best, best_overlap = None, 0
    for other, inst in compiled.instances.items():
        if other == instance or _paints_above(compiled, other, instance) is not above:
            continue
        region = inst.visible if above else inst.amodal
        if region is None:
            continue
        box = _intersect(window, inst.window if above else inst.amodal_window)
        if box is None:
            continue
        local = (box[0] - window[0], box[1] - window[1],
                 box[2] - window[0], box[3] - window[1])
        overlap = int(np.count_nonzero(_crop(changed, local) & _crop(region, box)))
        if overlap > best_overlap:
            best, best_overlap = other, overlap
    return best if best_overlap >= ZORDER_HINT_FRAC * total else None


def _paints_above(compiled: CompiledFrame, other: str, instance: str) -> Optional[bool]:
    """Is ``other`` painted over ``instance``? ``None`` when they never meet."""
    for order in compiled.painted.values():
        if other in order and instance in order:
            return order.index(other) > order.index(instance)
    return None
