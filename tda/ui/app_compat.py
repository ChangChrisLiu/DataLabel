"""Adapters for the session methods the main window was promised but may not find.

Task 13b was written while task 12a's fix round was still running, so a handful
of session methods (``overlay_layers``, ``push_stroke``, ``sigEditingChanged``,
``preview``, ``set_unexplained``, a non-raising ``resolve_conflict``) may or may
not exist on the ``AnnotationSession`` this window is handed.  Every one of them
is reached through a function here: if the real method is there it is called, and
if it is not, the *same observable behaviour* is reconstructed from the public
surface that does exist.  Nothing is faked -- a gap that cannot be reconstructed
returns ``None`` and the window degrades visibly (no preview counts, no
unexplained queue) rather than pretending.

:data:`ADAPTED` records which fallback was actually used, so the window logs it
once at start-up and the task report can list it.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from tda.ui.commands import edit_editing_mask_op

__all__ = [
    "ADAPTED",
    "close_session",
    "retry_rechecks",
    "editing_changed_signal",
    "flash_image",
    "is_open",
    "layer_changed",
    "overlay_layers",
    "preview",
    "push_stroke",
    "resolve_conflict",
    "session_refusal",
    "set_unexplained",
    "task_neighbour",
]

#: ``method name -> why the fallback ran``; populated the first time each gap is
#: hit.  The window logs it; the task report copies it verbatim.
ADAPTED: dict[str, str] = {}


def _note(name: str, why: str) -> None:
    ADAPTED.setdefault(name, why)


def _has(session: Any, name: str) -> bool:
    return callable(getattr(session, name, None))


# --------------------------------------------------------------------------- #
# the overlay
# --------------------------------------------------------------------------- #
def overlay_layers(session: Any) -> tuple[dict[str, np.ndarray], list[str]]:
    """``(masks, bottom-up order)`` of the visible instances of the open frame.

    Fallback: ``instance_rows()`` is already the layer order (top first) and
    carries the ``hidden`` flag, and ``compiled()`` holds the masks, so the
    promised method is exactly a reversal and a filter of those two.
    """
    if _has(session, "overlay_layers"):
        found = session.overlay_layers()
        return dict(found[0]), list(found[1])
    _note("overlay_layers", "built from instance_rows() + compiled()")
    compiled = session.compiled()
    masks: dict[str, np.ndarray] = {}
    order: list[str] = []
    for row in reversed(session.instance_rows()):  # bottom-up painting order
        key = str(row.get("key", ""))
        if row.get("hidden") or not key:
            continue
        inst = compiled.instances.get(key)
        if inst is None or inst.visible is None:
            continue
        masks[key] = np.asarray(inst.visible, dtype=bool)
        order.append(key)
    return masks, order


# --------------------------------------------------------------------------- #
# the editing layer
# --------------------------------------------------------------------------- #
def push_stroke(session: Any, instance: str, before: Optional[np.ndarray],
                after: np.ndarray) -> None:
    """Record one finished stroke as a single undoable op on the session.

    Fallback: the session registers an ``edit_editing_mask`` handler on its own
    undo stack (it is how it replays a stroke), so the op is built and pushed
    there without applying it -- the layer already holds ``after``.
    """
    after = np.asarray(after, dtype=bool)
    if _has(session, "push_stroke"):
        session.push_stroke(before, after)
        return
    _note("push_stroke", "op pushed onto session.undo_stack directly")
    if before is None:
        before = np.zeros_like(after)
    session.set_editing_mask(after)
    session.undo_stack.push(
        edit_editing_mask_op(instance, np.asarray(before, dtype=bool), after),
        apply=False,
    )


def editing_changed_signal(session: Any) -> Optional[Any]:
    """``session.sigEditingChanged`` when it exists, else ``None``.

    Without it the window re-reads ``editing_mask()`` after every undo/redo,
    which is the same information one event later.
    """
    signal = getattr(session, "sigEditingChanged", None)
    if signal is None:
        _note("sigEditingChanged", "editing layer re-read after undo/redo instead")
    return signal


def preview(session: Any, scope: str) -> Optional[dict]:
    """``{"steps", "verified_steps"}`` for the scope bar, or ``None``.

    There is no way to compute this from the public surface without doing the
    commit, so the bar simply leaves the counts out when the session cannot
    answer.
    """
    if _has(session, "preview"):
        try:
            return dict(session.preview(scope))
        except Exception:  # noqa: BLE001 - a preview must never block a commit
            return None
    _note("preview", "scope bar shows no affected-frame counts")
    return None


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #
#: What :func:`resolve_conflict` reports back.
RESOLVED = "resolved"
SUPERSEDED = "superseded"
REFUSED = "refused"


def resolve_conflict(session: Any, cid: int, resolution: str) -> tuple[str, str]:
    """``(verdict, message)``; never raises, whichever session version is in use.

    The fix round turns ``resolve_conflict`` into a function returning one of
    ``resolved`` / ``superseded`` / ``refused``.  Until then it raises
    :class:`~tda.core.truth.StaleConflictError` for "superseded and re-queued"
    and a plain ``ValueError`` for a refusal -- and the stale case must be
    caught *before* the plain one, because it is a subclass of it.
    """
    from tda.core.truth import StaleConflictError

    try:
        outcome = session.resolve_conflict(int(cid), str(resolution))
    except StaleConflictError as exc:
        return SUPERSEDED, str(exc) or "conflict superseded and re-queued"
    except ValueError as exc:
        return REFUSED, str(exc)
    if isinstance(outcome, str):
        return outcome, ""
    _note("resolve_conflict", "verdict derived from the raised exception type")
    return RESOLVED, ""


def set_unexplained(session: Any, step: int, boxes) -> bool:
    """Hand the frame's unexplained difference blobs to the review queue.

    ``False`` means the session has no such queue yet, in which case the window
    keeps the boxes in memory for the current run only.
    """
    if _has(session, "set_unexplained"):
        session.set_unexplained(int(step), list(boxes))
        return True
    _note("set_unexplained", "unexplained blobs kept in the window for this run")
    return False


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def session_refusal() -> type:
    """``SessionRefusal`` once the session exports it, ``ValueError`` before.

    Every refusal the session makes is a ``ValueError`` subclass either way, so
    catching the result of this call is correct in both worlds.
    """
    from tda.ui import session_api

    found = getattr(session_api, "SessionRefusal", None)
    if isinstance(found, type) and issubclass(found, Exception):
        return found
    _note("SessionRefusal", "refusals caught as plain ValueError")
    return ValueError


def task_neighbour(session: Any) -> Optional[int]:
    """The step the task card is diffed against -- ``j + 1`` in reverse order.

    The card lists the work of the frame on screen, i.e. the transitions from
    the state at the *neighbour* step to the state here.  Annotation runs
    backwards, so the neighbour is the next **available** step above the current
    one: the frame the annotator came from.
    """
    if _has(session, "task_neighbour"):
        found = session.task_neighbour()
        return None if found is None else int(found)
    _note("task_neighbour", "next available step above the current one")
    return _neighbour_step(session, above=True)


def _neighbour_step(session: Any, above: bool) -> Optional[int]:
    """The nearest step on one side that actually has an image."""
    try:
        here = int(session.current().step)
    except Exception:  # noqa: BLE001 - no frame is open
        return None
    steps = sorted(int(s) for s in session.steps())
    side = [s for s in steps if (s > here if above else s < here)]
    if not above:
        side.reverse()
    for step in side:
        if session.image_at(step) is not None:
            return step
    return None


def flash_image(session: Any, other: bool = False):
    """The frame ``Tab`` (or ``Shift+Tab``) flashes in place of the current one.

    ``Tab`` shows the neighbour the task card is written against -- the frame
    the annotator came from -- and ``Shift+Tab`` the opposite side.
    """
    if _has(session, "flash_compare"):
        try:
            return session.flash_compare(other=other)
        except TypeError:
            pass  # the old single-sided signature; fall through to the adapter
    _note("flash_compare(other=...)", "image_at() on the neighbour step")
    step = _neighbour_step(session, above=not other)
    return None if step is None else session.image_at(step)


def layer_changed(session: Any) -> bool:
    """Whether the editing layer holds pixels that have not been committed."""
    layer = getattr(session, "layer", None)
    if layer is not None and callable(getattr(layer, "changed", None)):
        return bool(layer.changed())
    _note("layer.changed()", "assumed unchanged; navigation is never blocked")
    return False


def is_open(session: Any) -> bool:
    """Whether the session currently has a frame open."""
    flag = getattr(session, "is_open", None)
    if isinstance(flag, bool):
        return flag
    if callable(flag):
        return bool(flag())
    try:
        session.current()
    except Exception:  # noqa: BLE001 - "no frame is open" is the answer
        return False
    return True


def close_session(session: Any, force: bool = True) -> None:
    """``session.close()``; called before the exit backup so the sweeper joins.

    ``force`` by default: the window has already settled any uncommitted layer
    through its close dialog by the time it gets here.
    """
    closer: Optional[Callable] = getattr(session, "close", None)
    if not callable(closer):
        return
    try:
        closer(force=force)
    except TypeError:
        _note("close(force=...)", "the session's close takes no force flag")
        closer()


def retry_rechecks(session: Any) -> int:
    """Re-queue the frames whose background re-check failed; returns how many.

    It only *enqueues*: the sweeper drains the queue on its own thread, so a
    deep backlog must not freeze the window that asked for the retry.
    """
    retry = getattr(session, "retry_rechecks", None)
    if callable(retry):
        return int(retry() or 0)
    _note("retry_rechecks", "queued through the sweeper from db.rechecks()")
    sweeper = getattr(session, "sweeper", None)
    db = getattr(session, "db", None)
    if sweeper is None or db is None or not callable(getattr(db, "rechecks", None)):
        return 0
    steps = list(db.rechecks(session.desktop, session.view))
    if steps:
        sweeper.enqueue(steps)
    return len(steps)


