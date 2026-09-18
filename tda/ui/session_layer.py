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

from typing import Optional

import numpy as np

from tda.ui.commands import Op, edit_editing_mask_op

__all__ = ["EditingLayer"]


class EditingLayer:
    """One instance's mask under the cursor, plus the shape it started from."""

    def __init__(self) -> None:
        self.instance: Optional[str] = None
        self._mask: Optional[np.ndarray] = None
        self._before: Optional[np.ndarray] = None

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

    def clear(self) -> None:
        self.instance = None
        self._mask = None
        self._before = None

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
    def stroke_op(self, before: np.ndarray, after: np.ndarray) -> Op:
        """The undoable record of one brush or eraser stroke (spec 4.6)."""
        if self.instance is None:
            raise RuntimeError("a stroke needs an instance; call begin_edit() first")
        self.set(after)
        return edit_editing_mask_op(self.instance, before, after)

    def apply_stroke(self, payload: dict, mask: np.ndarray) -> None:
        """Adopt the mask an undo or redo of a stroke restored."""
        self.instance = payload["instance"]
        self._mask = mask
