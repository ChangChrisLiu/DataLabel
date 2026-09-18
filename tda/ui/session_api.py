"""The annotation session as seen by the dock panels.

The panels hold no business logic (spec 10.1): every gesture ends in a call on
the session, which owns the database, the compiler and the op log.  This module
is the contract between the two halves -- a :class:`typing.Protocol` plus the
string constants of the domains that cross it -- so that
``tda.ui.session.AnnotationSession`` and the panels can be written and tested
independently and still agree on every name.

Nothing here imports the real session, and nothing here does any work: it is
types and strings only.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from PySide6.QtCore import Signal

from tda.core.compiler import CompiledFrame
from tda.core.model import FrameKey, Visibility

__all__ = [
    "SessionLike",
    "COMMIT_SCOPES",
    "FRAME_STATUSES",
    "QUEUE_NAMES",
    "RESOLUTIONS",
    "TASK_KINDS",
    "VISIBILITY_VALUES",
]

# --------------------------------------------------------------------------- #
# frame status -- what the timeline colours (spec 4.5)
# --------------------------------------------------------------------------- #
#: No shape or state has been annotated for the frame yet.
STATUS_UNLABELED = "unlabeled"
#: Propagated or model-drafted content that nobody has confirmed.
STATUS_AUTO = "auto"
#: Confirmed by an annotator (``verify_frame`` has run).
STATUS_VERIFIED = "verified"
#: Queued for a second pair of eyes.
STATUS_NEEDS_REVIEW = "needs_review"
#: An edit contradicts an already frozen frame.
STATUS_CONFLICT = "conflict"
#: The view has no image for this logical step (spec 4.2, "缺帧处理").
STATUS_MISSING = "missing"

FRAME_STATUSES: tuple[str, ...] = (
    STATUS_UNLABELED,
    STATUS_AUTO,
    STATUS_VERIFIED,
    STATUS_NEEDS_REVIEW,
    STATUS_CONFLICT,
    STATUS_MISSING,
)

# --------------------------------------------------------------------------- #
# task card -- the work implied by stepping from k to k-1 (spec 4.2)
# --------------------------------------------------------------------------- #
#: ``removed -> installed``: draw the part back into the chassis.
KIND_ADD_SHAPE = "add_shape"
#: ``open -> closed`` / ``unplugged -> plugged`` / ``displaced -> installed``.
KIND_SPLIT_KEYFRAME = "split_keyframe"
#: ``loosened -> fastened``: the shape carries over, only the state changes.
KIND_STATE_ONLY = "state_only"
#: The part is back in the chassis, so its bench box ends here.
KIND_REMOVE_BENCH_BOX = "remove_bench_box"
#: ``dupli`` / ``failed`` frames: nothing to draw, one keystroke.
KIND_CONFIRM = "confirm"

TASK_KINDS: tuple[str, ...] = (
    KIND_ADD_SHAPE,
    KIND_SPLIT_KEYFRAME,
    KIND_STATE_ONLY,
    KIND_REMOVE_BENCH_BOX,
    KIND_CONFIRM,
)

# --------------------------------------------------------------------------- #
# edit scope (spec 4.3)
# --------------------------------------------------------------------------- #
#: Edit the keyframe in force: affects its whole interval.
SCOPE_KEYFRAME = "keyframe"
#: Write a ``FrameOverride``: this frame only (``Alt``).
SCOPE_FRAME_OVERRIDE = "frame_override"
#: New shape version from this step on, in the browsing direction (``Ctrl+K``).
SCOPE_SPLIT = "split"

COMMIT_SCOPES: tuple[str, ...] = (SCOPE_KEYFRAME, SCOPE_FRAME_OVERRIDE, SCOPE_SPLIT)

# --------------------------------------------------------------------------- #
# review mode (spec 4.4)
# --------------------------------------------------------------------------- #
QUEUE_CONFLICTS = "conflicts"
QUEUE_NEEDS_REVIEW = "needs_review"
QUEUE_MISSING_SHAPE = "missing_shape"
QUEUE_UNEXPLAINED = "unexplained"

QUEUE_NAMES: tuple[str, ...] = (
    QUEUE_CONFLICTS,
    QUEUE_NEEDS_REVIEW,
    QUEUE_MISSING_SHAPE,
    QUEUE_UNEXPLAINED,
)

#: Keep the frozen shape and drop the conflicting edit.
RESOLVE_KEEP_OLD = "keep_old"
#: Take the edit and re-freeze the affected frames.
RESOLVE_ACCEPT_NEW = "accept_new"

RESOLUTIONS: tuple[str, ...] = (RESOLVE_KEEP_OLD, RESOLVE_ACCEPT_NEW)

#: The seven frame-level visibility values of spec 6.2, in the order the
#: number keys ``1``-``7`` set them in the instance list.
VISIBILITY_VALUES: tuple[str, ...] = tuple(v.value for v in Visibility)


@runtime_checkable
class SessionLike(Protocol):
    """What the dock panels require of an annotation session.

    ``AnnotationSession`` (``tda.ui.session``) is the production implementation
    and the tests use a stub; both are ``QObject``\\ s, because the panels
    connect to the three signals below.

    Contract notes the panels rely on:

    * :meth:`steps` returns the logical steps of the open desktop/view in
      *ascending* order; the timeline reverses them itself, since annotation
      runs backwards (spec 4.2).
    * :meth:`goto`, :meth:`prev` and :meth:`next` emit ``sigFrameChanged``;
      "advance" means ``step - 1``.
    * :meth:`instance_rows` is ordered top-most layer first.
    * :meth:`confirm_frame` emits ``sigProblems`` before returning ``False``,
      so a panel can show the list it just received.
    """

    #: The current frame changed; payload is a :class:`FrameKey`.
    sigFrameChanged: Signal
    #: Unsaved changes flag.
    sigDirty: Signal
    #: Problems of the current frame, as ``list[str]``.
    sigProblems: Signal

    # -- frames -------------------------------------------------------------
    def steps(self) -> list[int]:
        """Logical steps of the open desktop/view, ascending."""

    def frame_status(self, step: int) -> str:
        """One of :data:`FRAME_STATUSES`."""

    def thumb_path(self, step: int) -> str | None:
        """Cached thumbnail file for the step, or ``None`` if there is none."""

    def current(self) -> FrameKey:
        """The frame being annotated."""

    def goto(self, step: int) -> None:
        """Open ``step`` of the current desktop/view."""

    def prev(self) -> None:
        """Go one step towards the start of the teardown."""

    def next(self) -> None:
        """Go one step towards the end of the teardown."""

    def compiled(self) -> CompiledFrame:
        """Compiler output for the current frame."""

    # -- content ------------------------------------------------------------
    def instance_rows(self) -> list[dict]:
        """``{"key","cls","state","placement","visibility","z","hidden"}``, top first."""

    def task_card(self) -> list[dict]:
        """``{"instance","kind","text","done"}`` with ``kind`` in :data:`TASK_KINDS`."""

    # -- edits --------------------------------------------------------------
    def begin_edit(self, instance: str) -> None:
        """Load the instance's shape into the editing layer."""

    def commit_edit(self, scope: str) -> None:
        """Write the editing layer back; ``scope`` is one of :data:`COMMIT_SCOPES`."""

    def set_visibility(self, instance: str, vis: str) -> None:
        """Override the frame-level visibility; ``vis`` in :data:`VISIBILITY_VALUES`."""

    def set_hidden(self, instance: str, hidden: bool) -> None:
        """Show or hide the instance in the canvas (a view setting, not data)."""

    def set_zorder_move(self, instance: str, above_of: str) -> None:
        """Move ``instance`` directly above ``above_of`` in the z-order."""

    def confirm_frame(self) -> bool:
        """Verify the frame; ``False`` (with ``sigProblems``) when it has problems."""

    # -- review -------------------------------------------------------------
    def queues(self) -> dict[str, list[dict]]:
        """The four review queues, keyed by :data:`QUEUE_NAMES`."""

    def resolve_conflict(self, cid: int, resolution: str) -> None:
        """Resolve conflict ``cid`` with one of :data:`RESOLUTIONS`."""
