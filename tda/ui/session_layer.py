"""The editing layer: the one instance the annotator is drawing right now.

Spec 4.3 turns on a distinction this little object exists to keep: an edit is
not "the mask the annotator ended up with", it is **what changed** between the
shape they loaded and the one they are about to commit.  Adding pixels means
one thing, erasing them another, and touching nothing at all must mean nothing.

Ownership is the second reason.  The canvas paints into the overlay buffer that
:mod:`tda.ui.canvas.overlay` owns; the session may not depend on that buffer
still holding what it held when the commit was requested, so every mask that
crosses the boundary is copied.
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from tda.core.masks import BoxedMask
from tda.ui.commands import Op, edit_editing_mask_op

__all__ = ["EditingLayer"]


class EditingLayer:
    """One instance's mask under the cursor, plus the shape it started from."""

    #: Hands out :attr:`edit_id`; a plain counter, unique within a run.
    _next_id = itertools.count(1)

    def __init__(self) -> None:
        self.instance: Optional[str] = None
        #: Identity of the *edit* -- one ``begin`` to the ``clear`` or commit
        #: that ends it.  Two edits of the same instance on the same frame are
        #: two different pieces of work, and things that describe one of them
        #: (which Label Studio draft it was built from, spec 3.1 初稿引用) must
        #: not survive into the next: the undo history outlives both, so
        #: "same frame, same instance" was never enough to tell them apart.
        self.edit_id: Optional[int] = None
        self._mask: Optional[np.ndarray] = None
        self._before: Optional[np.ndarray] = None
        #: Pixels the annotator has deliberately taken off **during this edit**
        #: -- eraser strokes, ``Shift+D``'s specks, and what a negative-point
        #: SAM prompt removed -- as a :class:`~tda.core.masks.BoxedMask`.
        #: ``None`` while nothing has been erased, which is the common case and
        #: costs nothing.  No SAM result ever puts these back: an erase is a
        #: manual edit and a manual edit is never lost (ruling E1).
        self.erased: Optional[BoxedMask] = None

    # -- state --------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.instance is not None and self._mask is not None

    def begin(self, instance: str, amodal: Optional[np.ndarray],
              hw: tuple[int, int]) -> None:
        """Load an instance's amodal shape (or an empty canvas) for editing."""
        self._mask = (np.zeros(hw, dtype=bool) if amodal is None
                      else np.array(amodal, dtype=bool, copy=True))
        self._before = self._mask.copy()
        self.instance = instance
        self.edit_id = next(self._next_id)
        self.erased = None

    def clear(self) -> None:
        self.instance = None
        self.edit_id = None
        self._mask = None
        self._before = None
        # The protection belongs to the edit, not to the instance: it goes out
        # with the layer it was about.
        self.erased = None

    def mask(self) -> Optional[np.ndarray]:
        """The layer as the session last saw it; the session owns this array."""
        return self._mask

    def before(self) -> Optional[np.ndarray]:
        """The shape :meth:`begin` loaded, against which a change is measured."""
        return self._before

    def set(self, mask: np.ndarray) -> None:
        """Take a copy of what the window has been painting into.

        A copy, not a reference: the next stroke on the window's own buffer must
        not be able to change what a pending commit is about to write.
        """
        self._mask = np.array(mask, dtype=bool, copy=True)

    def changed(self) -> bool:
        """Has the shape actually been altered since :meth:`begin`?"""
        if self._mask is None or self._before is None:
            return False
        return not np.array_equal(self._mask, self._before)

    def settle(self) -> None:
        """Take the current mask as the new baseline: it has been written.

        Called after a successful commit.  Without it a committed layer still
        reads as "changed" and the navigation guard would refuse to leave a
        frame whose work is already in the database.
        """
        if self._mask is not None:
            self._before = self._mask.copy()

    # -- undo ---------------------------------------------------------------
    def set_erased(self, erased) -> None:
        """Replace the protected set (``None`` = nothing is protected).

        A :class:`~tda.core.masks.BoxedMask` is taken **as it is**: it is an
        immutable value the caller has just built, so copying it here was
        3 ms of every stroke for nothing (round 3). A raw canvas is boxed.
        """
        self.erased = BoxedMask.of(erased)

    def stroke_op(self, before: np.ndarray, after: np.ndarray,
                  adopted: Optional[dict] = None, erased=None) -> Op:
        """The undoable record of one brush or eraser stroke (spec 4.6).

        ``adopted`` travels with a stroke that came from a Label Studio draft,
        so the commit can read its provenance off the history itself.  It is
        stamped with :attr:`edit_id` **here** rather than by the caller: the
        note has to name the edit that is open at the moment the stroke lands,
        and this is the only place that knows both.
        """
        if self.instance is None:
            raise RuntimeError("a stroke needs an instance; call begin_edit() first")
        was_erased = self.erased
        self.set(after)
        self.set_erased(erased)
        if adopted:
            adopted = dict(adopted) | {"edit_id": self.edit_id}
        return edit_editing_mask_op(self.instance, before, after, adopted,
                                    erased_before=was_erased,
                                    erased_after=self.erased)

    def apply_stroke(self, payload: dict, mask: np.ndarray, erased=None) -> None:
        """Adopt the mask an undo or redo of a stroke restored.

        ``erased`` is the protected set that belongs with it; ``None`` means
        the payload carried none, i.e. nothing was protected at that point in
        the history.
        """
        self.instance = payload["instance"]
        self._mask = mask
        self.set_erased(erased)
