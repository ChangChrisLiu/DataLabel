"""Committing an edit, the scope bar, undo and the visibility keys.

Mixed into :class:`tda.ui.app.MainWindow` next to
:class:`tda.ui.app_edit.EditMixin`, which owns the layer these write back: what
is here is the moment the annotator says "yes" -- which scope the edit means,
what that scope is going to write, what ``Esc`` takes back, and the one history
every one of them lands in.
"""
from __future__ import annotations

import time

from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_priors
from tda.ui import app_support as S
from tda.ui import session_api as api

__all__ = ["CommitMixin"]

#: The two halves of the layering vocabulary the bar has to reason about; the
#: session owns their meaning (:mod:`tda.ui.session_scope`), the window only
#: needs to know which suggestion is on the bar.
A_ZORDER_ABOVE = "zorder:above:"
SPLIT_PREFIX = "split+"


class CommitMixin:
    """``Enter`` / ``Esc`` / ``Space``, the scope bar, undo-redo, visibility."""

    # --------------------------------------------------------------- commits
    @S.guard
    def act_commit(self) -> None:
        """``Enter``: store the ROI, accept the pending scope, or commit the edit.

        The ROI comes first because while its bar is up it is what the bar is
        talking about: "Enter 确认" next to a rectangle that ``Enter`` did not
        store -- it committed the instance loaded behind it instead -- is the
        kind of thing nobody notices until the rectangle is missing.
        """
        if self.roi_editing:
            self.accept_roi()
            return
        if self._pending_scope is not None:
            self._commit(self._pending_scope)
            return
        if getattr(self.session, "editing_instance", None) is not None \
                and self.session.editing_mask() is not None:
            if not self.has_uncommitted_edit():
                # Enter on a layer nobody has touched used to report
                # "committed (keyframe): False", write nothing, and clear the
                # layer -- so the annotator lost the instance they had just
                # loaded and had to double-click it again.
                self.report("没有可提交的修改 / nothing to commit on this layer")
                return
            scope = self.session.suggest_scope()
            if scope == api.SCOPE_KEYFRAME:
                self._commit(scope)
            else:
                self._offer_scope(scope)

    @S.guard
    def commit_with_suggested_scope(self) -> None:
        """Commit straight away with whatever the session suggests, no bar.

        The close dialog's "Save": there is no non-modal conversation left to
        have, so a layering or split suggestion is written as suggested rather
        than parked behind a bar nobody will read.
        """
        if getattr(self.session, "editing_instance", None) is None:
            return
        self._pending_scope = None
        self._commit(self.session.suggest_scope())

    @S.guard
    def act_commit_override(self) -> None:
        """``Alt+Enter``: this frame only."""
        self._commit(api.SCOPE_FRAME_OVERRIDE)

    @S.guard
    def act_commit_split(self) -> None:
        """``Ctrl+K``: a new shape version from this step on.

        On an open layering suggestion it is still an *answer to that
        suggestion*, so the pair goes with it (``split+zorder:above:<B>``): a
        split that dropped the pair saved the pixels and left them hidden under
        ``B``, and the screen did not change -- which is the one outcome the
        annotator can neither see nor explain.  ``Alt+Enter`` is different on
        purpose: a frame override makes this instance visible here by itself.
        """
        pending = self._pending_scope or ""
        if pending.startswith(A_ZORDER_ABOVE):
            self._commit(SPLIT_PREFIX + pending)
            return
        self._commit(api.SCOPE_SPLIT)

    def scope_bar_text(self) -> str:
        """What the non-modal scope bar is currently saying."""
        return self.scope_bar.label.text()

    def _offer_scope(self, scope: str) -> None:
        """Show the suggestion and its alternatives without blocking the canvas."""
        self._pending_scope = scope
        # A layering statement reaches frames too -- the session answers for a
        # pair override the same way it answers for pixels -- so the strip says
        # so for every scope now.
        counts = compat.preview(self.session, scope)
        detail = ""
        if counts:
            detail = (f"，影响 {len(counts.get('steps', []))} 帧 / 将产生 "
                      f"{len(counts.get('verified_steps', []))} 个冲突")
        self.scope_bar.show_text(f"建议范围 {scope}{self._writes(scope)}{detail}")
        self.report(f"suggested scope: {scope}")

    @staticmethod
    def _writes(scope: str) -> str:
        """What each of the bar's three keys will write, for this suggestion.

        "Above B" with pixels painted into B re-traces the shape as well -- they
        are not in it yet -- while the eraser direction only reverses the pair;
        and the two alternatives beside ``Enter`` write different things again.
        Three buttons that all read "commit" and do three different things is
        how ``Ctrl+K`` came to save pixels nobody could see.
        """
        if not str(scope).startswith("zorder:"):
            return ""
        if scope.startswith(A_ZORDER_ABOVE):
            return ("（Enter：层级 + 形状 / Alt+Enter：仅本帧覆盖 / "
                    "Ctrl+K：层级 + 拆分新形状）")
        return ("（Enter：仅层级 / Alt+Enter：仅本帧覆盖 / "
                "Ctrl+K：仅拆分新形状）")

    def _commit(self, scope: str) -> None:
        """Write the edit, or show why the session will not take it.

        A refusal -- "this part is on the bench, use the bench box" -- is an
        ordinary answer, not a failure: it keeps the editing layer so the
        annotator can press ``R`` and draw the box instead, and it does not go
        through the exception path, which would log a traceback for something
        the annotator simply has to read.
        """
        instance = getattr(self.session, "editing_instance", None)
        if instance is None:
            self.report("nothing is being edited")
            return
        if not self._area_is_plausible(instance, scope):
            return
        key = self.session.current()
        mask = self.session.editing_mask()
        pixels = int(mask.sum()) if mask is not None else 0
        started = time.perf_counter()
        try:
            result = self.session.commit_edit(scope) or {}
        except ValueError as refused:
            self._pending_scope = None
            self.scope_bar.hide()
            self.logger.info("refused commit %s %s step %s: %s",
                             instance, scope, key.step, refused)
            self.report_error(f"refused: {refused}")
            return
        self.logger.info("commit %s scope=%s step=%s px=%d ms=%.0f affected=%d",
                         instance, scope, key.step, pixels,
                         (time.perf_counter() - started) * 1000.0,
                         len(result.get("affected") or []))
        self._pending_scope = None
        self.scope_bar.hide()
        self.session.clear_edit()
        # One of the three: what any copy of this instance held is in the
        # database now, so even one from an earlier run is stale.
        self.drop_sidecar(key, instance, foreign_ok=True)
        self.set_sam_instance(None)
        self._sync_editing_layer()
        self.refresh_overlay()
        # An edit reaches other frames: those rows change colour now, not when
        # the annotator next happens to stand on one of them.
        self.timeline.refresh_statuses()
        # No re_explain() here: committing re-renders the frame, which clears
        # assist_result, so a re-split would run against nothing.  The real one
        # happens at confirm time, where the answer is actually used.
        self.report(f"committed ({scope}): {result.get('changed', '')}".strip())

    @S.guard
    def act_clear_edit(self) -> None:
        """``Esc``: abandon the ROI rectangle, the bench arm or the editing layer.

        Whichever gesture owns the canvas right now is the one ``Esc`` answers:
        while the ROI bar is up that is the rectangle, not the instance that
        happens to be loaded behind it.
        """
        if self._pending_warning is not None:
            # The warning is the thing on screen: Esc answers it by taking the
            # annotator back to the layer, not by throwing the layer away.
            self._pending_warning = None
            self.warn_bar.hide()
            self.report("回到编辑 / back to the mask")
            return
        if self.roi_editing:
            self.cancel_roi_edit()
            self.report("ROI unchanged")
            return
        if self.bench_instance is not None:
            self.disarm_bench()
            self.report("bench box cancelled")
            return
        instance = getattr(self.session, "editing_instance", None)
        if instance is not None:
            key = self.session.current()
            self.session.clear_edit()
            # Esc discards *the layer on screen*.  A crash copy from an earlier
            # run is a question nobody has answered, so it -- and its offer --
            # stay; only this window's own copy goes.
            self.drop_sidecar(key, instance)
            self._offer_restore(key, instance)
            self._pending_scope = None
            self.scope_bar.hide()
            self.set_sam_instance(None)
            self._sync_editing_layer()
            self.report("edit discarded")

    @S.guard
    def act_confirm(self) -> bool:
        """``Space``: verify the frame; on refusal the problems are shown, not a dialog.

        The unexplained differences are re-derived *here*, against what the frame
        holds now, and only what is still unexplained goes to the review queue.
        If the comparison has not come back yet it is finished synchronously
        under a short cap; if even that fails the step is recorded as "not
        analysed" rather than as "nothing unexplained", which would quietly
        claim the frame had been checked.
        """
        if not self.can_leave_edit():
            return False      # confirming steps the frame back: same gate
        if not self._open_the_selected_entry():
            return False
        step = self.session.current().step
        blobs = self.unexplained_at_confirm()
        ok = self.task_card.confirm()
        self.logger.info("confirm step %s: %s", step, "ok" if ok else "refused")
        if ok:
            self.hand_over_unexplained(step, blobs)
            # One more frame is done: both places that count say so.
            self.timeline.refresh_statuses()
            self.refresh_desktop_counts()
            self.report(f"step {step} confirmed")
        else:
            # One line, whatever the frame is missing.  A real start frame has
            # 60+ missing shapes, which put 2,550 characters into the one-line
            # status bar and asked the window to be 30,612 px wide.
            count = len(self.task_card.problem_rows())
            self.report(f"step {step} is not complete: {count} problem(s) — "
                        f"见任务卡 / see the task card")
        return bool(ok)

    # ------------------------------------------------------- the size warning
    def warn_bar_text(self) -> str:
        """What the area warning bar is currently saying."""
        return self.warn_bar.label.text()

    def _area_is_plausible(self, instance: str, scope: str) -> bool:
        """``False`` when a warning was raised and is waiting for a second Enter.

        Never blocks: the second press writes the mask exactly as it is and the
        override is logged, because the annotator is the authority on what a
        part looks like.  The rehearsal committed 1,502,386 px as a ``screw``
        and 40 masks under 50 px with nothing said either way, which is the
        only outcome this rules out.
        """
        pending = self._pending_warning
        if pending is not None and pending[0] == scope:
            self._pending_warning = None
            self.warn_bar.hide()
            # Not in the op-log payload: writing it there needs a session
            # change (``commit_edit`` would have to carry it), and that file
            # belongs to another worker this round.  The log line is searchable
            # and carries the same facts.
            self.logger.info("area_warning_overridden instance=%s scope=%s: %s",
                             instance, scope, pending[1])
            return True
        mask = self.session.editing_mask()
        if mask is None:
            return True
        warning = app_priors.area_warning(
            mask, self._class_of(instance), self._roi_area(), self.priors)
        if warning is None:
            self._pending_warning = None
            self.warn_bar.hide()
            return True
        self._pending_warning = (scope, warning)
        self.warn_bar.show_text(f"{warning}  —— Enter 仍然提交 / Esc 回去改")
        self.report(warning)
        self.logger.info("area_warning instance=%s scope=%s: %s",
                         instance, scope, warning)
        return False

    def _class_of(self, instance: str) -> str:
        """The taxonomy class of an instance, for the per-class prior."""
        for row in self.session.instance_rows():
            if str(row.get("key")) == str(instance):
                return str(row.get("cls") or "")
        return str(instance).split(".", 1)[0] if instance else ""

    def _roi_area(self) -> float:
        """Area of the stored ROI, or of the whole frame when there is none."""
        roi = self.roi()
        if roi is not None:
            return max(1.0, float(roi[2] - roi[0]) * float(roi[3] - roi[1]))
        hw = None if self.overlay is None else self.overlay.hw
        return max(1.0, float(hw[0]) * float(hw[1])) if hw else 1.0

    def _open_the_selected_entry(self) -> bool:
        """In Review mode, make ``Enter`` mean what its label says.

        The binding reads "accept the frame the queue points at" and it
        confirmed whatever was on the canvas: the annotator clicked an entry,
        pressed ``Enter``, and a different frame was marked verified.  The
        selected entry's frame is opened first -- through the same gate as any
        other move -- and only then confirmed.  ``False`` means the move was
        refused, so nothing is confirmed either.
        """
        if self.mode != A.MODE_REVIEW:
            return True
        step = self.review.selected_step()
        if step is None or int(step) == int(self.session.current().step):
            return True
        return self.leave_frame(
            lambda: self.session.goto(int(step), force=True)
        )

    # ------------------------------------------------------------------ undo
    @S.guard
    def act_undo(self) -> None:
        self._history(self.session.undo(), "undo")

    @S.guard
    def act_redo(self) -> None:
        self._history(self.session.redo(), "redo")

    def _history(self, moved: bool, what: str) -> None:
        if not moved:
            self.report(f"nothing to {what}")
            return
        self.refresh_overlay()
        self._sync_editing_layer()
        # An undo can take the layer back to what begin_edit loaded, and then
        # the sidecar it wrote (or is about to) describes work that no longer
        # exists: re-queue it, which drops both when there is nothing left.
        instance = getattr(self.session, "editing_instance", None)
        mask = self.session.editing_mask()
        if instance is not None and mask is not None:
            self.queue_sidecar(self.session.current(), instance, mask)
        self.timeline.refresh_statuses()   # it takes back other frames too
        self.update_status()
        self.report(what)

    @S.guard
    def _on_editing_changed(self, mask: object) -> None:
        """The session moved the editing layer behind our back (undo/redo)."""
        self._sync_editing_layer()

    # ------------------------------------------------------------ visibility
    @S.guard
    def act_set_visibility(self, value: str) -> None:
        if not self._editable_row():
            return
        self.instances.set_visibility(value)
        self.refresh_overlay()

    @S.guard
    def act_cycle_visibility(self) -> None:
        if not self._editable_row():
            return
        self.instances.cycle_visibility()
        self.refresh_overlay()

    @S.guard
    def act_toggle_hidden(self) -> None:
        if not self._editable_row():
            return
        self.instances.toggle_hidden()
        self.refresh_overlay()

    def _editable_row(self) -> bool:
        """Is the instance table pointing at something these keys can change?

        The ``Removed`` toggle lists parts the frame no longer has; they are
        read-only, and ``H`` / ``V`` / ``1``-``7`` on one of them did nothing at
        all -- which reads exactly like the key not working.
        """
        if self.instances.selected_is_removed():
            self.report("已移除的零件不可编辑 / this part has been removed")
            return False
        return True

