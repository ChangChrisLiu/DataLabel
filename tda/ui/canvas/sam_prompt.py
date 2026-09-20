"""The candidates half of a SAM prompt: keeping them, walking them, applying one.

Split out of :mod:`tda.ui.canvas.sam_tools`, which is otherwise about the
prompt itself and about not applying a stale answer.  What is here is the small
state machine behind ``C``: SAM returns up to three proposals for one click
(part, sub-assembly, assembly), the annotator flips through them, and the tool
has to survive an undo or a hand edit landing in the middle of that.

The index is **derived from the layer** rather than trusted: each candidate is
rendered once, the current editing mask is compared against those renderings,
and the walk continues from whichever one matches.  When none does, the
annotator has painted on the proposal -- the candidates are dropped and their
edit is never overwritten.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

#: Instance key used when neither the tool nor the overlay names one yet.
FALLBACK_INSTANCE = "editing"

log = logging.getLogger(__name__)

__all__ = ["FALLBACK_INSTANCE", "HINT_EDITED", "NEGATIVE_NOTE", "CandidatesMixin",
           "applied_text"]

#: Emitted on ``sigHint`` when cycling is abandoned because the annotator
#: painted on the proposal (their edit is never discarded).
HINT_EDITED = "candidates discarded: the mask was edited"

#: Appended to the status clause when the prompt held a negative point, which
#: is the only way a *new* SAM result is allowed to take pixels off (R2b).
NEGATIVE_NOTE = "（含负点 / with a negative point）"
#: ... and when ``C`` swapped this prompt's contribution for a smaller one,
#: which is the other -- and only other -- way the layer can shrink under SAM.
CYCLE_NOTE = "（换候选 / candidate swap）"


def applied_text(added: int, removed: int, negative: bool) -> str:
    """What a SAM application did, in one clause for the status bar.

    ``SAM +12,345 px`` when nothing came off, and ``+a / −r`` with the reason
    when something did -- a right click, or ``C`` swapping this prompt's
    contribution for a smaller candidate.  A bare ``SAM +0 px`` over a layer
    that had just lost 79,333 px (measured, cycling on D13/scan/42) is the
    silence this whole task is about.
    """
    if int(removed) <= 0:
        return f"SAM +{int(added):,} px"
    note = NEGATIVE_NOTE if negative else CYCLE_NOTE
    return f"SAM +{int(added):,} / −{int(removed):,} px{note}"


class CandidatesMixin:
    """``C``, the rendering cache, and applying one proposal to the layer."""

    # -- candidates ---------------------------------------------------------
    @property
    def candidate_count(self) -> int:
        """Number of masks the last result offered (0 before the first one)."""
        return len(self._candidates)

    @property
    def candidate_index(self) -> int:
        """Index of the candidate currently in the editing layer."""
        return self._candidate_index

    def cycle_candidate(self, step: int = 1) -> int:
        """Replace the editing layer with the next candidate; return its index.

        Meant to be bound to ``C``. A single positive point is ambiguous on a
        large part, so SAM's three proposals are kept and the annotator flips
        through them instead of re-clicking.

        The index is *derived from the layer*, not trusted: the current editing
        mask is compared against what each candidate would produce and the walk
        continues from whichever one matches. That keeps cycling correct after
        an undo or redo has moved the layer behind the tool's back. When the
        layer matches no candidate the annotator has painted on the proposal, so
        the candidates are dropped, :attr:`sigHint` explains why, and nothing is
        overwritten -- a manual edit is never discarded. With fewer than two
        candidates this is a no-op, which also keeps a pointless entry out of
        the undo stack.
        """
        self._sync_identity()
        if self.overlay is None or not self._candidates:
            return self._candidate_index
        if len(self._candidates) < 2:
            return self._candidate_index
        renders = self._ensure_renders()
        if not renders:
            self._reset_candidates()
            return 0
        current = self.overlay.editing
        match = next(
            (i for i, layer in enumerate(renders) if np.array_equal(current, layer)),
            None,
        )
        if match is None:
            self._reset_candidates()
            self.sigHint.emit(HINT_EDITED)
            return 0
        self._candidate_index = (match + int(step)) % len(renders)
        self._apply_candidate(why="cycle")
        return self._candidate_index

    def _reset_candidates(self) -> None:
        """Forget the offered masks and their renderings (frees the cache)."""
        self._candidates = []
        self._candidate_index = 0
        self._candidate_rect = None
        self._candidate_base = None
        self._candidate_union = True
        self._candidate_identity = None
        self._renders = None

    def owned_mask(self) -> Optional[np.ndarray]:
        """The pixels the current prompt found in the layer and may not touch.

        ``None`` before a result has landed.  What "owned" means is spelled out
        in :meth:`~tda.ui.canvas.sam_tools.SamToolBase._on_result`: everything
        that was in the editing layer when the answer arrived, because none of
        it came from this prompt.
        """
        return self._candidate_base

    def _render(self, index: int) -> Optional[np.ndarray]:
        """The full-frame editing layer candidate ``index`` would produce.

        ``owned ∪ candidate``, except inside the crop of a prompt that carried
        the prior mask as ``mask_input``: there SAM was shown the pixels and its
        answer replaces them, which is what keeps a negative point able to
        remove something.
        """
        if self.overlay is None or not self._fits(self._candidate_rect):
            return None
        if not 0 <= index < len(self._candidates):
            return None
        base = self._candidate_base
        if base is not None and base.shape != self.overlay.hw:
            return None
        assert self._candidate_rect is not None
        x0, y0, x1, y1 = self._candidate_rect
        mask = self._candidates[index]
        if mask.shape != (y1 - y0, x1 - x0):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (x1 - x0, y1 - y0),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        full = base.copy() if base is not None else np.zeros(self.overlay.hw, dtype=bool)
        if self._candidate_union:
            full[y0:y1, x0:x1] |= mask
        else:
            full[y0:y1, x0:x1] = mask
        return full

    def _ensure_renders(self) -> list[np.ndarray]:
        """Render every candidate once; ``[]`` when they cannot be applied.

        At most three full-frame boolean layers, held only for the latest
        result and freed by :meth:`_reset_candidates`.
        """
        if self._renders is None:
            rendered = [self._render(i) for i in range(len(self._candidates))]
            self._renders = [] if any(r is None for r in rendered) else rendered
        return [r for r in self._renders if r is not None]

    def _apply_candidate(self, why: str = "apply") -> None:
        """Write the selected candidate into the editing layer (GUI thread)."""
        renders = self._ensure_renders()
        if not renders or self.overlay is None or self._candidate_rect is None:
            return
        layer = renders[self._candidate_index]
        # Same contract as PaintTool: the pre-edit layer is available when
        # sigStroke fires, so one applied mask is one undoable op.
        self.stroke_before = self.overlay.editing.copy()
        instance = self._target_instance() or FALLBACK_INSTANCE
        owned = self._candidate_base
        # ``count_nonzero`` throughout and one temporary: ``.sum()`` on a 12 MP
        # boolean accumulates in int64 and costs 4.9 ms a call.
        before, after = self.stroke_before, layer
        both = int(np.count_nonzero(before & after))
        was, now = int(np.count_nonzero(before)), int(np.count_nonzero(after))
        added, removed = now - both, was - both
        log.info(
            "sam %s instance=%s candidate=%d/%d compose=%s layer %d -> %d px "
            "(+%d/-%d) owned %d kept %d",
            why, instance, self._candidate_index + 1, len(renders),
            "add" if self._candidate_union else "add+remove",
            was, now, added, removed,
            0 if owned is None else int(np.count_nonzero(owned)),
            0 if owned is None else int(np.count_nonzero(owned & layer)),
        )
        self.overlay.set_editing(instance, layer)
        if self.canvas is not None:
            self.canvas.refresh(self._candidate_rect)
        self.sigHint.emit(applied_text(added, removed,
                                       negative=not self._candidate_union))
        self.sigStroke.emit(self._candidate_rect)


