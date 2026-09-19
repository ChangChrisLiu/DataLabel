"""Committing an edit, the scope bar, undo and the visibility keys.

Mixed into :class:`tda.ui.app.MainWindow` next to
:class:`tda.ui.app_edit.EditMixin`, which owns the layer these write back: what
is here is the moment the annotator says "yes" -- which scope the edit means,
what that scope is going to write, what ``Esc`` takes back, and the one history
every one of them lands in.
"""
from __future__ import annotations

from tda.ui import app_compat as compat
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
        key = self.session.current()
        try:
            result = self.session.commit_edit(scope) or {}
        except ValueError as refused:
            self._pending_scope = None
            self.scope_bar.hide()
            self.report_error(f"refused: {refused}")
            return
        self._pending_scope = None
        self.scope_bar.hide()
        self.session.clear_edit()
        self.drop_sidecar(key, instance)
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
            self.drop_sidecar(key, instance)
            self._restore_offer = None
            self.restore_bar.hide()
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
        step = self.session.current().step
        blobs = self.unexplained_at_confirm()
        ok = self.task_card.confirm()
        if ok:
            self.hand_over_unexplained(step, blobs)
            self.report(f"step {step} confirmed")
        else:
            problems = self.task_card.problems()
            self.report(f"step {step} is not complete: {'; '.join(problems) or 'see the task card'}")
        return bool(ok)

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
        self.instances.set_visibility(value)
        self.refresh_overlay()

    @S.guard
    def act_cycle_visibility(self) -> None:
        self.instances.cycle_visibility()
        self.refresh_overlay()

    @S.guard
    def act_toggle_hidden(self) -> None:
        self.instances.toggle_hidden()
        self.refresh_overlay()

