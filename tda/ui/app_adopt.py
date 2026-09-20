"""Adopting one of the team's old Label Studio polygons (``Shift+A``).

Mixed into :class:`tda.ui.app.MainWindow` next to
:class:`tda.ui.app_edit.EditMixin`.  :mod:`tda.core.ls_adopt` decides *which*
drafts could be the shape being drawn; what is here is the conversation about
them, and it is deliberately the same conversation the ROI rectangle has:

* ``Shift+A`` puts the best candidate on screen as a **ghost** -- the overlay's
  preview layer, display-only.  Pressing it again walks the rest.
* while a ghost is up it **owns** ``Enter`` and ``Esc``, through the same
  layered chain in :mod:`tda.ui.app_commit` that already gives them to the ROI
  rectangle and to the area warning.  ``Enter`` copies the ghost into the
  editing layer as one undoable stroke (the sidecar follows, as after any
  stroke); ``Esc`` takes the ghost away and leaves the layer exactly as it was.
* anything that changes what the annotator is looking at -- another frame,
  another instance, another view, a commit, an undo, the ROI rectangle -- takes
  the ghost and the candidates with it, which is the hygiene a SAM prompt gets
  for the same reason: a proposal is about one instance on one frame.

The draft itself is never touched.  What the commit carries is one line in its
op-log ``extra`` -- ``adopted_from`` / ``adopted_step`` -- so the paper can say
how much of the old annotation was reused.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tda.core import ls_adopt
from tda.ui import app_compat as compat
from tda.ui import app_support as S

__all__ = ["ADOPT_BUSY", "ADOPT_NEEDS_INSTANCE", "ADOPT_NO_IMAGE", "ADOPT_WRONG_MODE",
           "AdoptMixin", "no_drafts_text"]

#: Shown when ``Shift+A`` is pressed with nothing being edited.
ADOPT_NEEDS_INSTANCE = ("先双击任务卡或实例表里的一条再按 Shift+A  "
                        "(adopt needs an instance being edited)")
#: Shown when another non-modal bar already owns ``Enter`` / ``Esc``.
ADOPT_BUSY = ("先处理屏幕下方那条提示（Enter / Esc）再采纳草稿  "
              "(answer the bar that owns Enter/Esc first)")
#: Shown in Steps or Review mode, where there is no editing layer to adopt into.
ADOPT_WRONG_MODE = "切到标注模式再采纳草稿 / switch to Annotate mode to adopt a draft"
#: Shown on a frame this view never photographed.
ADOPT_NO_IMAGE = "本帧没有图像，无法采纳草稿 / no image on this frame"


def no_drafts_text(cls: str, stats: dict) -> str:
    """"Nothing to offer", and what was dropped on the way to that answer.

    The skipped count is the part worth saying: "no drafts" and "three drafts
    that were traced at another resolution" are different answers, and only the
    second one means the old annotation of this part exists but cannot be used.
    """
    skipped = int(stats.get("wrong_size", 0))
    tail = (f"（{skipped} 个草稿尺寸不符，已跳过 / {skipped} skipped: "
            f"wrong size）" if skipped else "")
    return f"{cls}: 这一帧附近没有可用的旧草稿 / no Label Studio draft here{tail}"


class AdoptMixin:
    """``Shift+A``, the ghost, and what the commit records about it."""

    # ------------------------------------------------------------------ setup
    def _init_adopt(self) -> None:
        #: Candidates of the ghost currently on screen; empty means no ghost.
        self._draft_candidates: list[ls_adopt.DraftCandidate] = []
        self._draft_index = 0
        #: What an adopted draft adds to the next commit's op-log ``extra``.
        self._adopted_facts: Optional[dict] = None

    # ------------------------------------------------------------------ state
    def showing_draft_ghost(self) -> bool:
        """Is a draft on screen waiting for ``Enter``?"""
        return bool(self._draft_candidates)

    def draft_candidates(self) -> list:
        """The candidates being walked (a copy; for the tests and the report)."""
        return list(self._draft_candidates)

    def adopted_facts(self) -> Optional[dict]:
        """The ``adopted_from`` note the next commit of this layer carries."""
        return None if self._adopted_facts is None else dict(self._adopted_facts)

    # ----------------------------------------------------------------- action
    @S.guard
    def act_adopt_draft(self) -> None:
        """``Shift+A``: offer the old polygons of this class, one at a time.

        Inert while ``Tab`` is held: the canvas is showing another frame, so
        every tool is detached there and a proposal about *this* frame would be
        read against the wrong picture.
        """
        if self.is_flashing():
            from tda.ui.app_edit import FLASH_HINT, FLASH_HINT_HOLD_MS

            self.report(FLASH_HINT, hold_ms=FLASH_HINT_HOLD_MS)
            return
        if self.showing_draft_ghost():
            self._draft_index = (self._draft_index + 1) % len(self._draft_candidates)
            self._show_draft()
            return
        refusal = self._adopt_refusal()
        if refusal:
            self.report(refusal)
            return
        self._offer_drafts()

    def _adopt_refusal(self) -> str:
        """Why ``Shift+A`` cannot open a ghost right now, or ``""``.

        The last three are the ones that matter: a rectangle, a scope
        suggestion and an area warning each already own ``Enter`` and ``Esc``,
        and two owners of one key is how a rectangle gets stored by a press
        that was meant for something else.
        """
        if self.mode != "annotate":
            return ADOPT_WRONG_MODE
        if not compat.is_open(self.session) or self.session.image() is None:
            return ADOPT_NO_IMAGE
        if getattr(self.session, "editing_instance", None) is None:
            return ADOPT_NEEDS_INSTANCE
        if self.overlay is None:
            return ADOPT_NO_IMAGE
        if self.roi_editing or self._pending_scope is not None \
                or self._pending_warning is not None:
            return ADOPT_BUSY
        return ""

    def _offer_drafts(self) -> None:
        """Read the candidates for the instance being edited and show the first."""
        key = self.session.current()
        instance = str(self.session.editing_instance)
        cls = self._class_of(instance)
        stats: dict = {}
        found = ls_adopt.drafts_for(
            self.db, self.session.tax, int(key.desktop), str(key.view), int(key.step),
            cls, hw=self.overlay.hw, editing=self._adopt_reference(), stats=stats,
        )
        self.logger.info("adopt D%s/%s step %s %s (%s): %s", key.desktop, key.view,
                         key.step, instance, cls, stats)
        if not found:
            self.report(no_drafts_text(cls, stats))
            return
        self._draft_candidates = found
        self._draft_index = 0
        self._show_draft()

    def _adopt_reference(self) -> Optional[np.ndarray]:
        """What the candidates are ranked against: the layer, or the proposal.

        The editing layer once it holds pixels -- the annotator has already
        said roughly where the part is -- and otherwise the difference map's
        armed prompt box, which is the same answer one step earlier.  With
        neither, :func:`tda.core.ls_adopt.drafts_for` falls back to step
        distance.
        """
        mask = self.session.editing_mask()
        if mask is not None and np.asarray(mask, dtype=bool).any():
            return np.asarray(mask, dtype=bool)
        return self._box_mask(self._prompt_box)

    def _box_mask(self, box) -> Optional[np.ndarray]:
        """A rectangle as a full-frame mask, clipped; ``None`` for no box."""
        if box is None or self.overlay is None:
            return None
        h, w = self.overlay.hw
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x0, y0 = max(0, min(x0, w)), max(0, min(y0, h))
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    def _show_draft(self) -> None:
        """Paint the selected candidate as the overlay's ghost and say what it is."""
        candidate = self._draft_candidates[self._draft_index]
        if self.overlay is None or candidate.mask.shape != self.overlay.hw:
            self.clear_draft_ghost()
            self.report(ADOPT_NO_IMAGE)
            return
        self.overlay.set_ghost(candidate.mask)
        self.canvas.refresh()
        self.report(
            f"草稿 {self._draft_index + 1}/{len(self._draft_candidates)}："
            f"{candidate.key} @ step {candidate.step} "
            f"(IoU {candidate.iou_with_editing:.2f}) —— Enter 采纳 / Esc 取消  "
            f"(Shift+A for the next one)"
        )

    # ------------------------------------------------------- Enter and Escape
    def accept_draft_ghost(self) -> bool:
        """``Enter`` while a ghost is up: copy it into the layer, once.

        ``True`` means the press was the ghost's and the commit chain must stop
        -- the same answer the ROI rectangle gives.  The layer is replaced when
        it is still empty and unioned when it is not: a draft is a starting
        point for an empty layer, and a correction for one that already holds
        work.  Either way it is one
        :meth:`~tda.ui.app_edit.EditMixin.set_editing_mask` call, so it is one
        entry on the same history as a brush stroke and one debounced sidecar
        write.
        """
        if not self.showing_draft_ghost():
            return False
        candidate = self._draft_candidates[self._draft_index]
        instance = getattr(self.session, "editing_instance", None)
        if instance is None or self.overlay is None \
                or candidate.mask.shape != self.overlay.hw:
            self.clear_draft_ghost()
            self.report(ADOPT_NEEDS_INSTANCE)
            return True
        current = self.session.editing_mask()
        empty = current is None or not np.asarray(current, dtype=bool).any()
        merged = (candidate.mask.copy() if empty
                  else np.asarray(current, dtype=bool) | candidate.mask)
        self.clear_draft_ghost()
        self.set_editing_mask(merged, undoable=True)
        # After the stroke, so that the hygiene hook the stroke does *not* run
        # cannot take it back out again.
        self._adopted_facts = {"adopted_from": str(candidate.key),
                               "adopted_step": int(candidate.step)}
        self.logger.info("adopted %s (step %s) into %s: %s", candidate.key,
                         candidate.step, instance, "replace" if empty else "union")
        self.report(f"采纳 {candidate.key}（{'替换' if empty else '并入'}编辑层）/ "
                    f"{'replaced' if empty else 'unioned'} from {candidate.key} "
                    f"— Enter 提交 / Esc 放弃")
        return True

    def clear_draft_ghost(self) -> bool:
        """Take the ghost off the screen and forget the candidates.

        ``True`` when there was one, which is what makes ``Esc`` stop at the
        ghost rather than going on to discard the edit underneath it.
        """
        if not self._draft_candidates:
            return False
        self._draft_candidates = []
        self._draft_index = 0
        if self.overlay is not None and self.overlay.has_ghost:
            self.overlay.clear_ghost()
            self.canvas.refresh()
        return True

    def forget_draft_ghost(self) -> None:
        """Drop the ghost *and* the adoption note: this layer is not that one.

        Called wherever the editing layer is replaced from outside a tool (a
        commit, ``Esc``, an undo, a frame change) -- the same place
        :meth:`~tda.ui.app_assist.AssistMixin.reset_sam_prompt` is called, and
        for the same reason.  The note goes too: a commit that claimed
        ``adopted_from`` for pixels an undo had taken back would be a false
        entry in the one audit trail that says how much of the old annotation
        was reused.
        """
        self.clear_draft_ghost()
        self._adopted_facts = None
