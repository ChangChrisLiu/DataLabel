"""Adopting one of the team's old Label Studio polygons (``Shift+A``).

Mixed into :class:`tda.ui.app.MainWindow` next to
:class:`tda.ui.app_edit.EditMixin`.  :mod:`tda.core.ls_adopt` decides *which*
drafts could be the shape being drawn; what is here is the conversation about
them, and it is deliberately the same conversation the ROI rectangle has:

* ``Shift+A`` puts the draft **under the mouse cursor** on screen as a ghost --
  the overlay's preview layer, display-only.  Pressing it again walks the rest,
  nearest first.  The cursor is what tells fifteen "PSU to Motherboard
  Connector" drafts of one frame apart; their ordinals cannot, and an order
  that ignored the pointer put the right one somewhere in the middle of a list
  nobody walks to the end of.
* while a ghost is up it **owns every commit key** -- ``Enter``, ``Alt+Enter``,
  ``Ctrl+K``, ``Space`` and the task card's buttons -- through the layered
  chain in :mod:`tda.ui.app_commit` that already gives ``Enter``/``Esc`` to the
  ROI rectangle and the area warning.  ``Enter`` copies the ghost into the
  editing layer as one undoable stroke (the sidecar follows, as after any
  stroke); the others are refused with a line saying to finish the preview
  first; ``Esc`` takes the ghost away and leaves the layer exactly as it was.
* anything that changes what the annotator is looking at -- another frame,
  another instance, another view, another mode, a commit, an undo, the ROI
  rectangle, a ``Tab`` flash, hiding the overlays -- takes the ghost with it.

**Provenance is read back out of the history, never carried on the side.**  The
stroke an adoption makes is stamped with the draft it came from, so at commit
time the window asks the undo stack which adoptions are applied-and-not-undone
and checks each of them against the pixels actually being written
(``overlap_px``).  An undone adoption claims nothing; an adoption erased before
the commit claims nothing; two adoptions are two entries.  The crash sidecar
carries the same list, because a crash takes the undo stack with it.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from tda.core import ls_adopt
from tda.core import masks as _masks
from tda.ui import app_compat as compat
from tda.ui import app_support as S

__all__ = ["ADOPT_BENCH", "ADOPT_BUSY", "ADOPT_FIRST", "ADOPT_NEEDS_INSTANCE",
           "ADOPT_NO_IMAGE", "ADOPT_RESTORE", "ADOPT_SHAPE_MISMATCH",
           "ADOPT_WRONG_MODE", "AdoptMixin", "no_drafts_text"]

#: Shown when ``Shift+A`` is pressed with nothing being edited.
ADOPT_NEEDS_INSTANCE = ("先双击任务卡或实例表里的一条再按 Shift+A  "
                        "(adopt needs an instance being edited)")
#: Shown when another non-modal bar already owns ``Enter`` / ``Esc``.
ADOPT_BUSY = ("先处理屏幕下方那条提示（Enter / Esc）再采纳草稿  "
              "(answer the bar that owns Enter/Esc first)")
#: Shown when a commit key that is not ``Enter`` is pressed under a ghost.
ADOPT_FIRST = ("先处理草稿预览：Enter 采纳 / Esc 取消  "
               "(finish the draft preview first: Enter adopt / Esc cancel)")
#: Shown in Steps or Review mode, where there is no editing layer to adopt into.
ADOPT_WRONG_MODE = "切到标注模式再采纳草稿 / switch to Annotate mode to adopt a draft"
#: Shown on a frame this view never photographed.
ADOPT_NO_IMAGE = "本帧没有图像，无法采纳草稿 / no image on this frame"
#: Shown while the box tool is armed for a part on the bench.
ADOPT_BENCH = ("台面框工具已武装：草稿是掩码，不是矩形  "
               "(a bench box is armed: draft adoption draws a mask, not a box)")
#: Shown while an unanswered crash-restore offer is on screen.
ADOPT_RESTORE = ("先回答恢复提示（恢复 / 丢弃）再采纳草稿  "
                 "(answer Restore/Discard first)")
#: Shown when the draft's mask does not fit this frame's canvas after all.
ADOPT_SHAPE_MISMATCH = ("草稿尺寸和当前帧不符，已跳过  "
                        "(this draft does not fit the frame and was dropped)")


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
    """``Shift+A``, the ghost, and the provenance a commit reads off the history."""

    # ------------------------------------------------------------------ setup
    def _init_adopt(self) -> None:
        #: Candidates of the ghost currently on screen; empty means no ghost.
        self._draft_candidates: list[ls_adopt.DraftCandidate] = []
        self._draft_index = 0
        #: Draft keys this view's op log already records as adopted, read when
        #: the candidates are; only the status line uses them.
        self._draft_used: set[str] = set()
        #: Adoptions a crash-restored layer brought back, which the undo stack
        #: cannot know about: ``(edit id, [adopted note, ...])``.  Keyed by the
        #: **edit** Restore started, so discarding that edit takes them with it
        #: -- there is nothing to clear and therefore nothing to forget.
        self._restored_adoptions: tuple[Optional[int], list[dict]] = (None, [])

    # ------------------------------------------------------------------ state
    def showing_draft_ghost(self) -> bool:
        """Is a draft on screen waiting for ``Enter``?"""
        return bool(self._draft_candidates)

    def draft_candidates(self) -> list:
        """The candidates being walked (a copy; for the tests and the report)."""
        return list(self._draft_candidates)

    # ----------------------------------------------------------------- action
    @S.guard
    def act_adopt_draft(self, at: Optional[tuple] = None) -> None:
        """``Shift+A``: offer the drafts under the pointer, one at a time.

        ``at`` overrides the mouse position with an image coordinate (the tests
        and any future panel gesture); normally the cursor is read where the
        key was pressed, which is the whole point -- *that* is the annotator
        saying which of the frame's fifteen connectors they mean.

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
        self._offer_drafts(self._cursor_xy(at))

    def _adopt_refusal(self) -> str:
        """Why ``Shift+A`` cannot open a ghost right now, or ``""``.

        The last four are the ones that matter: a rectangle, a scope
        suggestion, an area warning and a crash-restore offer each already own
        ``Enter`` and ``Esc``, and two owners of one key is how a rectangle gets
        stored by a press that was meant for something else.
        """
        if self.mode != "annotate":
            return ADOPT_WRONG_MODE
        if not compat.is_open(self.session) or self.session.image() is None:
            return ADOPT_NO_IMAGE
        if getattr(self, "bench_instance", None) is not None:
            return ADOPT_BENCH
        if getattr(self.session, "editing_instance", None) is None:
            return ADOPT_NEEDS_INSTANCE
        if self.overlay is None:
            return ADOPT_NO_IMAGE
        if self.roi_editing or self._pending_scope is not None \
                or self._pending_warning is not None:
            return ADOPT_BUSY
        if self.pending_restore() is not None:
            return ADOPT_RESTORE
        return ""

    # ----------------------------------------------------------- the pointer
    def _cursor_xy(self, at: Optional[tuple] = None) -> Optional[tuple]:
        """Where the annotator is pointing, in image pixels, or ``None``.

        In order: the caller's own coordinate, the mouse cursor over the canvas,
        and -- when the pointer is somewhere else entirely -- the centre of the
        strongest change this frame has not explained yet, which is the same
        question asked by the difference map instead of by the hand.
        """
        point = self._as_image_point(at) if at is not None else self._mouse_image_xy()
        return point if point is not None else self._unexplained_centre()

    def _mouse_image_xy(self) -> Optional[tuple]:
        """The mouse position in image coordinates, if it is over the canvas."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QCursor

        viewport = self.canvas.viewport()
        try:
            local = viewport.mapFromGlobal(QCursor.pos())
        except Exception:  # noqa: BLE001 - no pointer at all (offscreen, remote)
            return None
        if not viewport.rect().contains(local):
            return None
        return self._as_image_point(self.canvas.image_pos(QPointF(local)))

    def _as_image_point(self, xy) -> Optional[tuple]:
        """``(x, y)`` clipped to the frame, or ``None`` when it is outside it."""
        if xy is None or self.overlay is None:
            return None
        h, w = self.overlay.hw
        x, y = float(xy[0]), float(xy[1])
        if not (0 <= x < w and 0 <= y < h):
            return None
        return (x, y)

    def _unexplained_centre(self) -> Optional[tuple]:
        """Centre of the strongest unexplained change on this frame, or ``None``."""
        from tda.ui.app_diff import best_unexplained

        blob = best_unexplained(getattr(self, "assist_result", None))
        if blob is None:
            return None
        x0, y0, x1, y1 = (float(v) for v in blob.box)
        return self._as_image_point(((x0 + x1) / 2.0, (y0 + y1) / 2.0))

    # --------------------------------------------------------------- the list
    def _offer_drafts(self, cursor: Optional[tuple]) -> None:
        """Read the candidates for the instance being edited and show the first."""
        key = self.session.current()
        instance = str(self.session.editing_instance)
        cls = self._class_of(instance)
        stats: dict = {}
        found = ls_adopt.drafts_for(
            self.db, self.session.tax, int(key.desktop), str(key.view), int(key.step),
            cls, hw=self.overlay.hw, editing=self._adopt_reference(), cursor=cursor,
            stats=stats,
        )
        self.logger.info("adopt D%s/%s step %s %s (%s) cursor=%s: %s", key.desktop,
                         key.view, key.step, instance, cls, cursor, stats)
        if not found:
            self.report(no_drafts_text(cls, stats))
            return
        self._draft_candidates = found
        self._draft_index = 0
        self._draft_used = self.adopted_draft_keys()
        self._show_draft()

    def _adopt_reference(self) -> Optional[np.ndarray]:
        """What ranks the candidates when there is no pointer at all.

        The editing layer once it holds pixels -- the annotator has already
        said roughly where the part is -- and otherwise the difference map's
        armed prompt box.  With neither,
        :func:`tda.core.ls_adopt.drafts_for` falls back to step distance.
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
        """Paint the selected candidate as the ghost and say what it is.

        Exactly one candidate holds a decoded mask: the one on screen.  The
        others are RLEs until they are shown, which is what makes offering
        thirty-five drafts of a 12 MP frame cost nothing.
        """
        candidate = self._draft_candidates[self._draft_index]
        for other in self._draft_candidates:
            if other is not candidate:
                other.release()
        if self.overlay is None or candidate.mask.shape != self.overlay.hw:
            self.clear_draft_ghost()
            self.report(ADOPT_SHAPE_MISMATCH)
            return
        self.overlay.set_ghost(candidate.mask, candidate.box)
        self.canvas.refresh(candidate.box)
        seen = "（本视图已采纳过 / already adopted here）" \
            if candidate.key in self._draft_used else ""
        other_class = "" if candidate.same_class else f"【{candidate.cls}】"
        self.report(
            f"草稿 {self._draft_index + 1}/{len(self._draft_candidates)}："
            f"{other_class}{candidate.key} @ step {candidate.step} "
            f"({candidate.area} px){seen} —— Enter 采纳 / Esc 取消  "
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
        entry on the same history as a brush stroke, one debounced sidecar
        write, and one stamp saying where the pixels came from.
        """
        if not self.showing_draft_ghost():
            return False
        if self.is_flashing():
            # The canvas is showing the neighbour; pasting "what is under the
            # ghost" would be pasting from a picture this frame is not about.
            from tda.ui.app_edit import FLASH_HINT, FLASH_HINT_HOLD_MS

            self.report(FLASH_HINT, hold_ms=FLASH_HINT_HOLD_MS)
            return True
        candidate = self._draft_candidates[self._draft_index]
        instance = getattr(self.session, "editing_instance", None)
        if instance is None:
            self.clear_draft_ghost()
            self.report(ADOPT_NEEDS_INSTANCE)
            return True
        if self.overlay is None or candidate.mask.shape != self.overlay.hw:
            self.clear_draft_ghost()
            self.report(ADOPT_SHAPE_MISMATCH)
            return True
        key = self.session.current()
        current = self.session.editing_mask()
        empty = current is None or not np.asarray(current, dtype=bool).any()
        merged = (candidate.mask.copy() if empty
                  else np.asarray(current, dtype=bool) | candidate.mask)
        note = {"adopted_from": str(candidate.key),
                "adopted_step": int(candidate.step),
                "keyframe_id": None if candidate.keyframe_id is None
                else int(candidate.keyframe_id),
                "instance": str(instance),
                "frame": [int(key.desktop), int(key.step), str(key.view)]}
        self.clear_draft_ghost()
        self.set_editing_mask(merged, undoable=True, adopted=note)
        self.logger.info("adopted %s (step %s) into %s: %s", candidate.key,
                         candidate.step, instance, "replace" if empty else "union")
        self.report(f"采纳 {candidate.key}（{'替换' if empty else '并入'}编辑层）/ "
                    f"{'replaced' if empty else 'unioned'} from {candidate.key} "
                    f"— Enter 提交 / Esc 放弃")
        return True

    def refuse_under_ghost(self) -> bool:
        """``True`` (and a line) when a commit key is not the ghost's ``Enter``.

        Every other way of saying "write it" -- ``Alt+Enter``, ``Ctrl+K``,
        ``Space``, the task card's four buttons -- would commit the layer
        *under* the preview, which is not what the annotator is looking at.

        One deliberate exception, and it does not come through here: the close
        dialog's **Save** (:meth:`~tda.ui.app_commit.CommitMixin.commit_with_suggested_scope`).
        There is no non-modal conversation left to have at that point, and the
        thing worth saving is the layer the annotator built, not an offer they
        never accepted -- so it writes the layer and the ghost simply goes.
        """
        if not self.showing_draft_ghost():
            return False
        self.report(ADOPT_FIRST)
        return True

    def clear_draft_ghost(self) -> bool:
        """Take the ghost off the screen and forget the candidates.

        ``True`` when there was one, which is what makes ``Esc`` stop at the
        ghost rather than going on to discard the edit underneath it.
        """
        if not self._draft_candidates:
            return False
        candidates, self._draft_candidates = self._draft_candidates, []
        self._draft_index = 0
        for candidate in candidates:
            candidate.release()
        if self.overlay is not None and self.overlay.has_ghost:
            rect = self.overlay.ghost_rect
            self.overlay.clear_ghost()
            self.canvas.refresh(rect)
        return True

    def forget_draft_ghost(self) -> None:
        """Drop the ghost: the layer it was offered against is not this one.

        Called wherever the editing layer is replaced from outside a tool (a
        commit, ``Esc``, an undo, a frame change) -- the same place
        :meth:`~tda.ui.app_assist.AssistMixin.reset_sam_prompt` is called, and
        for the same reason.  The *provenance* is not dropped here: it lives on
        the undo history and is checked against the committed pixels, so an
        undo takes it back by itself and nothing has to remember to.
        """
        self.clear_draft_ghost()

    # ----------------------------------------------------------- provenance
    def pending_adoptions(self, key, instance: Optional[str]) -> list[dict]:
        """Adoptions applied to **the edit that is open right now**.

        The undo stack's applied strokes of this edit, plus whatever a
        crash-restored layer brought into it.  No pixel check here: this is
        what the sidecar carries, and the sidecar's job is to lose nothing.

        The filter is the ``edit_id``, not the frame and the instance.  The
        history is per view and is not cleared between edits, so "same frame,
        same instance" also matched *the last time this shape was drawn* --
        and a hand-drawn re-trace of an adopted shape was filed as draft reuse.
        """
        edit_id = getattr(self.session, "editing_id", None)
        if instance is None or edit_id is None:
            return []
        frame = [int(key.desktop), int(key.step), str(key.view)]
        restored_id, restored = self._restored_adoptions
        out = [dict(note) for note in restored] if restored_id == edit_id else []
        for note in compat.adoptions_in_history(self.session):
            if note.get("edit_id") != edit_id:
                continue
            if str(note.get("instance")) != str(instance):
                continue
            if list(note.get("frame") or []) != frame:
                continue
            out.append(dict(note))
        return out

    def adoptions_for_commit(self, key, instance: Optional[str],
                             mask: Optional[np.ndarray]) -> list[dict]:
        """What the commit may claim: the adoptions still visible in the mask.

        Each entry carries ``overlap_px`` -- how many of the pixels about to be
        written lie inside that draft -- measured off the run lengths.  An entry
        with no overlap left is dropped: the annotator adopted a draft, erased
        it and drew the part by hand, and a commit that still said
        ``adopted_from`` would put a false provenance into the only record of
        how much old annotation this dataset reused.
        """
        pending = self.pending_adoptions(key, instance)
        if not pending or mask is None:
            return []
        written = _masks.encode_rle(np.asarray(mask, dtype=bool))
        out: dict[tuple, dict] = {}
        for note in pending:
            rle = ls_adopt.draft_rle(self.db, int(key.desktop), str(key.view),
                                     str(note.get("adopted_from")),
                                     note.get("keyframe_id"))
            overlap = _masks.rle_overlap(written, rle)
            if overlap <= 0:
                continue
            entry = {"adopted_from": str(note.get("adopted_from")),
                     "adopted_step": int(note.get("adopted_step") or 0),
                     "keyframe_id": note.get("keyframe_id"),
                     "overlap_px": int(overlap)}
            identity = (entry["adopted_from"], entry["keyframe_id"])
            if overlap > int(out.get(identity, {}).get("overlap_px", -1)):
                out[identity] = entry
        return [out[k] for k in sorted(out, key=lambda k: (-out[k]["overlap_px"], k[0]))]

    def note_restored_adoptions(self, key, instance: str, entries) -> None:
        """Take the adoption list of a crash-restored layer back (spec 4.6).

        A crash loses the undo history, so without this the recovered layer
        would be committed as work nobody can trace to a draft.  The list is
        filed under the **edit** ``Restore`` has just begun: discard that edit
        with ``Esc`` and it stops applying, because the next edit has another
        id -- no cleanup to remember, and none to forget.
        """
        edit_id = getattr(self.session, "editing_id", None)
        kept = [dict(entry) for entry in (entries or []) if entry.get("adopted_from")]
        for entry in kept:
            entry.setdefault("instance", str(instance))
            entry.setdefault("frame", [int(key.desktop), int(key.step), str(key.view)])
            entry["edit_id"] = edit_id
        self._restored_adoptions = (edit_id, kept) if kept else (None, [])

    def adopted_draft_keys(self, instance: Optional[str] = None) -> set[str]:
        """Draft keys this view's op log already records as adopted.

        Said, not refused.  The same draft is legitimately taken twice -- an
        undo and a second try, a split that re-traces the same shape from this
        step on -- so stopping the annotator would be wrong.  What this rules
        out is the *quiet* duplicate: two instances of one view taking their
        pixels from one old polygon with nothing on screen saying so.

        Asked as a query over the whole view, because "the last two hundred
        operations" stops being the answer on the second day of a machine.  The
        SQL narrows to the rows that mention an adoption at all -- a JSON *key*,
        so the pattern cannot be broken by a separator's whitespace -- and the
        payloads are then parsed rather than pattern-matched, which is also how
        ``instance`` is compared.
        """
        used: set[str] = set()
        if not compat.is_open(self.session):
            return used
        try:
            rows = self.db.conn.execute(
                "SELECT payload_json FROM op_log WHERE desktop=? AND view=? "
                "AND kind='commit_keyframe' AND payload_json LIKE '%\"adopted_from\"%'",
                (int(self.session.desktop), str(self.session.view)),
            ).fetchall()
        except Exception:  # noqa: BLE001 - a note, never a reason to fail Shift+A
            return used
        for row in rows:
            try:
                payload = json.loads(row[0] or "{}")
            except ValueError:
                continue
            if instance is not None and str(payload.get("instance")) != str(instance):
                continue
            for entry in payload.get("adopted") or ():
                if entry.get("adopted_from"):
                    used.add(str(entry["adopted_from"]))
        return used
