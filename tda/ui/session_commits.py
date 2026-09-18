"""Every gesture that writes, as the session exposes it to the window.

:mod:`tda.ui.session_edit` holds the rules; this is the half that knows *which
frame is open* and *what is being drawn* -- the editing layer, the refusals that
depend on the frame (no image, a part on the bench), the scope suggestion and
the preview of an edit's reach.  It is a mixin rather than a module of functions
because every one of these is about the session's own state.

Split off :mod:`tda.ui.session` only to keep both files readable; the methods
are part of :class:`~tda.ui.session.AnnotationSession`'s public surface and of
the :class:`~tda.ui.session_api.SessionLike` protocol.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tda.core.model import FrameKey
from tda.core.truth_inputs import frame_hw
from tda.ui import session_api as api
from tda.ui.session_api import SessionRefusal
from tda.ui import session_edit as edit
from tda.ui.session_ops import GEOM_BOX, ON_BENCH

__all__ = ["CommitMixin"]


class CommitMixin:
    """The editing layer and the four edit scopes, for :class:`AnnotationSession`."""

    def _editable_frame(self) -> FrameKey:
        """The current frame, refusing one there is nothing to draw on.

        A step flagged ``missing``, or whose image never made it into the cache,
        has no canvas: its state and its shape anchors are real (spec 4.2
        缺帧处理) but a mask drawn "on" it would be in nobody's coordinates.
        """
        key = self.current()
        if key.step not in self._available or self.image_path(key.step) is None:
            raise SessionRefusal(
                f"step {key.step} of {self.view} has no image: it cannot be annotated"
            )
        return key

    def begin_edit(self, instance: str) -> None:
        """Load the instance's amodal shape into the editing layer (spec 4.3)."""
        key = self._editable_frame()
        found = self.compiled().instances.get(instance)
        self.layer.begin(instance, None if found is None else found.amodal,
                         frame_hw(self.db, key))

    @property
    def editing_instance(self) -> Optional[str]:
        """The instance being drawn, or ``None``."""
        return self.layer.instance

    def editing_mask(self) -> Optional[np.ndarray]:
        """The editing layer as the session last saw it, or ``None``.

        The array belongs to the session: the window paints into its own overlay
        buffer and hands the result over with :meth:`set_editing_mask`.
        """
        return self.layer.mask()

    def set_editing_mask(self, mask: np.ndarray) -> None:
        """Take a copy of the layer the window has been painting into."""
        self.layer.set(mask)

    def push_stroke(self, before: np.ndarray, after: np.ndarray) -> None:
        """Record one brush/eraser stroke on the undo stack (spec 4.6).

        The pixels are already painted, so the op is logged rather than applied;
        undoing it hands the earlier mask back on :attr:`sigEditingChanged`.
        """
        self.undo_stack.push(self.layer.stroke_op(before, after), apply=False)
        self._refresh_dirty()

    def clear_edit(self) -> None:
        """Drop the editing layer without writing anything."""
        self.layer.clear()

    def commit_edit(self, scope: str, direction: str = edit.REVERSE) -> dict:
        """Write the editing layer back with the scope the annotator chose.

        ``scope`` is one of :data:`tda.ui.session_api.COMMIT_SCOPES`, or one of
        the two layering answers :func:`tda.ui.session_edit.suggest_scope` gives
        -- ``zorder:above:<B>`` / ``zorder:below:<B>`` -- which write a
        ``PairOverride`` instead of pixels (spec 4.3 改层级).

        A pixel scope with nothing changed is a no-op: loading a shape and
        pressing Enter must not mint a new version of it.  An explicit layering
        scope is always honoured, because there the pixels are not the point.
        """
        if not self.layer.active:
            raise RuntimeError("commit_edit() needs begin_edit() first")
        key, instance = self._editable_frame(), self.layer.instance
        pair = edit.split_zorder_scope(scope)
        if pair is None and not self.layer.changed():
            return {"changed": False, "affected": [], "conflicts": 0, "problems": {},
                    "scope": scope}
        if pair is not None:
            other, above = pair
            result = edit.commit_pair_override(
                self.db, self.truth, key,
                instance if above else other, other if above else instance,
                self.annotator, known=self._known_instances(),
            )
        else:
            self._refuse_mask_on_bench(key, instance)
            result = edit.commit_edit(self.db, self.truth, key, instance, self.layer.mask(),
                                      scope, direction, self.annotator)
        return self._after_edit(result)

    def _known_instances(self) -> set[str]:
        """The instances this frame has, which a layering gesture may name."""
        return set(self.compiled().instances)

    def _refuse_mask_on_bench(self, key: FrameKey, instance: str) -> None:
        """A part on the bench is tracked by a rectangle, not by a mask (spec 4.2)."""
        placement = edit.placement_of(self.db, self.tax, key, instance)
        if placement == ON_BENCH:
            raise SessionRefusal(
                f"{instance} is on the bench at step {key.step}: use the bench box tool"
            )

    def commit_box(self, instance: str, box, direction: str = edit.REVERSE) -> dict:
        """Draw the staging-area rectangle of a part on the bench (spec 4.2 S4)."""
        result = edit.commit_box(self.db, self.truth, self._editable_frame(), instance, box,
                                 direction=direction, annotator=self.annotator)
        return self._after_edit(result)

    def commit_occluder(self, mask: np.ndarray, occluder_type: str = "hand") -> dict:
        """Store one occluder layer of the current frame (spec 4.2 step 4)."""
        result = edit.commit_occluder(self.db, self.truth, self._editable_frame(), mask,
                                      occluder_type, self.annotator)
        return self._after_edit(result)

    def preview(self, scope: str, direction: str = edit.REVERSE) -> dict:
        """How far the pending edit would reach, before it is written (spec 4.3).

        ``{"steps", "verified_steps"}`` -- the frames the scope would change and
        the already-confirmed ones among them, i.e. the "影响 N 帧 / 将产生 N 个
        冲突" strip.  Nothing is written and no pixels are touched.
        """
        if self.editing_instance is None:
            return {"steps": [], "verified_steps": []}
        pair = edit.split_zorder_scope(scope)
        if pair is not None:
            other, above = pair
            edit.require_instance(self._known_instances(), other)
            return edit.preview_pair(self.db, self.truth, self.current(),
                                     self.editing_instance, other)
        geom = GEOM_BOX if edit.placement_of(
            self.db, self.tax, self.current(), self.editing_instance
        ) == ON_BENCH else None
        return edit.preview(self.db, self.truth, self.current(), self.editing_instance,
                            scope, direction, **({"geom_type": geom} if geom else {}))

    def set_visibility(self, instance: str, vis: str) -> None:
        """Override one instance's visibility label on this frame (spec 6.2)."""
        self._after_edit(
            edit.set_visibility(self.db, self.truth, self.current(), instance, vis,
                                self.annotator)
        )

    def set_hidden(self, instance: str, hidden: bool) -> None:
        """Show or hide an instance in the canvas.

        A view setting, not data: it is never written to the database and never
        logged, and it is forgotten when another desktop/view is opened.
        """
        if hidden:
            self._hidden.add(instance)
        else:
            self._hidden.discard(instance)

    def set_zorder_move(self, instance: str, above_of: str) -> None:
        """Move ``instance`` directly above ``above_of`` in the layer order."""
        self._after_edit(
            edit.set_zorder_move(self.db, self.truth, self.current(), instance, above_of,
                                 self.annotator, known=self._known_instances())
        )

    def suggest_scope(self, edited: Optional[np.ndarray] = None) -> str:
        """The scope the pending edit would default to (spec 4.3).

        Answered from the pixels that *changed* since :meth:`begin_edit`, so
        opening a shape and touching nothing suggests nothing.
        """
        if not self.layer.active:
            return api.SCOPE_KEYFRAME
        mask = self.layer.mask() if edited is None else edited
        return edit.suggest_scope(self.compiled(), self.layer.instance,
                                  self.layer.before(), mask)
