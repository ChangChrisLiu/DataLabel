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
    "SessionRefusal",
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
#: A frozen frame whose inputs changed and that the truth sweeper has not
#: compared against them yet (spec 3.4): it may yet turn into a conflict, so it
#: is not a frame anybody should treat as settled.
STATUS_RECHECK = "recheck"
#: The view has no image for this logical step; the state and the shape anchor
#: still exist, the frame simply has nothing to draw on (spec 4.2, missing frames).
STATUS_MISSING = "missing"

FRAME_STATUSES: tuple[str, ...] = (
    STATUS_UNLABELED,
    STATUS_AUTO,
    STATUS_VERIFIED,
    STATUS_NEEDS_REVIEW,
    STATUS_CONFLICT,
    STATUS_RECHECK,
    STATUS_MISSING,
)

# --------------------------------------------------------------------------- #
# task card -- the work that belongs to the frame on screen (spec 4.2)
# --------------------------------------------------------------------------- #
#: The part is in the chassis on this frame and has no shape here yet.
KIND_ADD_SHAPE = "add_shape"
#: ``open -> closed`` / ``unplugged -> plugged`` / ``displaced -> installed``.
KIND_SPLIT_KEYFRAME = "split_keyframe"
#: ``loosened -> fastened``: the shape carries over, only the state changes.
KIND_STATE_ONLY = "state_only"
#: The part is out of the chassis and the view can see the staging area, so it
#: needs a rectangle there -- box work, not brush work (spec 4.2 item 1).
KIND_ADD_BENCH_BOX = "add_bench_box"
#: The part is back in the chassis, so its bench box ends here.
KIND_REMOVE_BENCH_BOX = "remove_bench_box"
#: ``dupli`` / ``failed`` frames: nothing to draw, one keystroke.
KIND_CONFIRM = "confirm"

TASK_KINDS: tuple[str, ...] = (
    KIND_ADD_SHAPE,
    KIND_SPLIT_KEYFRAME,
    KIND_STATE_ONLY,
    KIND_ADD_BENCH_BOX,
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


class SessionRefusal(ValueError):
    """A gesture the session declines, with a sentence the annotator can act on.

    Refusals are ordinary: drawing on a frame that has no image, painting a mask
    on a part lying on the bench, naming an instance this frame does not have.
    They are told apart from real errors so the window can show the text next to
    the canvas instead of logging a traceback at somebody who did nothing wrong.
    """


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
    #: The editing layer was replaced from the outside (an undone stroke);
    #: payload is the new mask, or ``None``.
    sigEditingChanged: Signal
    #: The session let go of its desktop/view; the panels must detach.
    sigClosed: Signal
    #: ``(done, total, failed)`` of the background re-check of frozen frames.
    sigSweepProgress: Signal
    #: ``(step, text)`` -- a background re-check failed; the frame stays pending.
    sigSweepError: Signal
    #: Something the review queues show has changed (coalesced).
    sigQueuesChanged: Signal

    # -- frames -------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        """Is a desktop/view open? Everything below needs one."""

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

    def overlay_layers(self):
        """``({instance: visible mask}, bottom-up order)`` for the canvas overlay."""

    def task_neighbour(self) -> int | None:
        """The annotated frame :meth:`task_card` is diffed against, or ``None``."""

    # -- content ------------------------------------------------------------
    def instance_rows(self) -> list[dict]:
        """``{"key","cls","state","placement","visibility","z","hidden"}``, top first."""

    def task_card(self) -> list[dict]:
        """``{"instance","kind","text","done"}`` with ``kind`` in :data:`TASK_KINDS`.

        The work of the frame on screen, diffed against :meth:`task_neighbour`.
        """

    # -- edits --------------------------------------------------------------
    def begin_edit(self, instance: str) -> None:
        """Load the instance's shape into the editing layer."""

    def set_editing_mask(self, mask) -> None:
        """Hand over the layer the window has been painting into."""

    def push_stroke(self, before, after) -> None:
        """Record one brush/eraser stroke on the undo stack."""

    def commit_edit(self, scope: str) -> dict:
        """Write the editing layer back; ``scope`` is one of :data:`COMMIT_SCOPES`.

        Also accepts ``zorder:above:<B>`` / ``zorder:below:<B>``, which write a
        layering exception rather than pixels (spec 4.3).
        """

    def preview(self, scope: str) -> dict:
        """``{"steps", "verified_steps"}`` an edit would reach, writing nothing."""

    def set_visibility(self, instance: str, vis: str) -> None:
        """Override the frame-level visibility; ``vis`` in :data:`VISIBILITY_VALUES`."""

    def set_hidden(self, instance: str, hidden: bool) -> None:
        """Show or hide the instance in the canvas (a view setting, not data)."""

    def set_zorder_move(self, instance: str, above_of: str) -> None:
        """Move ``instance`` directly above ``above_of`` in the z-order."""

    def confirm_frame(self) -> bool:
        """Verify the frame; ``False`` (with ``sigProblems``) when it has problems.

        Raises :class:`SessionRefusal` when the editing layer still holds
        uncommitted pixels: confirming steps the frame back, so it is a way out
        of the edit like any other and it is not the caller's to skip.
        """

    # -- review -------------------------------------------------------------
    def queues(self) -> dict[str, list[dict]]:
        """The four review queues, keyed by :data:`QUEUE_NAMES`."""

    def resolve_conflict(self, cid: int, resolution: str) -> str:
        """Resolve conflict ``cid``; ``"resolved"`` / ``"superseded"`` / ``"refused"``."""

    def set_unexplained(self, step: int, boxes) -> None:
        """Record the difference-map regions of one frame that nothing explains."""

    def retry_rechecks(self) -> int:
        """Re-arm every outstanding truth re-check; returns how many."""
