"""Removing an instance identity row in stage S1.

This is the one step-table command that cannot wait for ``Apply``:
:meth:`~tda.ui.steps_model.StepTableData.save` only upserts, so a deletion has
to be written through. That makes it the one place where "written through" has
to mean *completely* written through, which is what this module is for.

Two halves:

* :func:`check_deletable` -- refuse while anything a *human* made still depends
  on the key: a step targeting it, a shape keyframe in any of the four views, a
  constraint edge (spec 7.1), a frame override, a layering exception, an entry
  in a z-order, an open conflict, a hand-written state event, or a **verified**
  compiled row. An ``auto`` compiled row is not one of those: the compiler
  writes one per instance the frame needs, geometry or not, so counting it made
  every instance the app had compiled in the background undeletable. It is a
  cache, and it goes with the delete, as the derived (``auto=True``) events do.
* :func:`delete_instance` -- do it in one transaction: drop the derived events
  and the cached rows, drop the identity row, and rewrite every neighbour that
  pointed at the key. A neighbour's pointer is *cleared*, except
  :data:`REVERTIBLE_FIELD`, which goes back to the class name when the deleted
  instance was an implied one -- that is the state the log importer left it in.
  The neighbours are read back from the database rather than taken from memory,
  so an unsaved edit elsewhere in one of those rows is not flushed along with
  the repair. Deleting an implied instance also **records the refusal** for the
  desktop (:meth:`~tda.core.db.Db.decline_implied`), in the same transaction, so
  the next import does not create it again. Nothing in memory changes until the
  transaction has committed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tda.core.db import Db
from tda.core.implied import is_implied
from tda.core.logs import CHASSIS_KEY
from tda.core.model import VIEWS, InstanceRec
from tda.ui.steps_values import RELATION_FIELDS, EditError

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, fine for typing
    from tda.ui.steps_model import StepTableData

__all__ = ["check_deletable", "delete_instance", "stored_neighbours"]


def check_deletable(data: "StepTableData", db: Db, key: str) -> None:
    """Raise :class:`EditError` unless nothing depends on ``key`` any more."""
    if key not in data.instances:
        raise EditError(f"D{data.desktop:02d} has no instance {key!r}")
    if key == CHASSIS_KEY:
        raise EditError("the chassis is implicit and cannot be deleted")
    steps = sorted({a.step for a in data.actions if a.target == key})
    if steps:
        raise EditError(f"{key!r} is still the target of step(s) {', '.join(map(str, steps))}")
    for view in VIEWS:
        if db.keyframes(data.desktop, view, key):
            raise EditError(f"{key!r} still has shape keyframes in view {view!r}")
    for rel in db.relations(data.desktop):
        if key in (rel.get("target"), rel.get("blocker")):
            raise EditError(f"{key!r} is still used by a {rel.get('type')!r} constraint edge")
    counts = db.instance_reference_counts(data.desktop, key)
    if counts:
        named = ", ".join(f"{n} row(s) in {table}" for table, n in sorted(counts.items()))
        raise EditError(f"{key!r} is still referenced by {named}")


#: The one relational field with a documented *placeholder* state: the log
#: importer writes the bare class name into it when the step says which part
#: carries the socket but not which one, and stage S1 narrows it down. Reverting
#: any other field to a class name would make it permanently coarse -- the
#: heuristic fills ``fastens``/``parent`` only when they are blank -- and would
#: silence the "captive screw without parent" question with an answer that is
#: not one.
REVERTIBLE_FIELD = "socket_host"


def stored_neighbours(
    db: Db, desktop: int, key: str, revert_to: Optional[str] = None
) -> list[InstanceRec]:
    """The **stored** instances pointing at ``key``, with that pointer cleared.

    Read back from the database on purpose: writing the in-memory copies would
    flush whatever else the annotator has changed on those rows but not applied
    yet.

    ``revert_to`` is the taxonomy *class* to leave in :data:`REVERTIBLE_FIELD`
    instead of nothing, and it is what deleting an **implied** instance does:
    the importer had written ``socket_host = "motherboard"`` there, and the
    implied board was only what resolved it to a key. Clearing it would throw
    that reference away -- the desktop would stop looking like one that is
    missing a motherboard, and re-creating the instance later
    (``--reset-declined``) would find nothing to hang on. Every other field is
    cleared, exactly as an ordinary delete clears it.
    """
    cleaned: list[InstanceRec] = []
    for other_key, stored in db.instances(desktop).items():
        if other_key == key or not any(getattr(stored, n) == key for n in RELATION_FIELDS):
            continue
        for name in RELATION_FIELDS:
            if getattr(stored, name) == key:
                setattr(stored, name, revert_to if name == REVERTIBLE_FIELD else None)
        cleaned.append(stored)
    return cleaned


def delete_instance(data: "StepTableData", db: Db, key: str) -> None:
    """Delete one instance and every pointer to it, atomically.

    Deleting an **implied** instance (:mod:`tda.core.implied`) also records its
    class as declined for this desktop, in the same transaction: the importer
    creates implied instances from scratch on every run, so without a durable
    "no" the deleted motherboard is back after the next ``import-logs``, with a
    mask on every frame. A delete that rolls back therefore declines nothing
    either.

    On any failure the transaction rolls back, the in-memory session is left
    exactly as it was, and the reason comes back as an :class:`EditError`.
    """
    check_deletable(data, db, key)
    implied_cls = data.instances[key].cls if is_implied(data.instances[key]) else None
    neighbours = stored_neighbours(db, data.desktop, key, revert_to=implied_cls)
    try:
        with db.transaction():
            db.delete_auto_events(data.desktop, key)
            db.delete_instance(data.desktop, key)
            for rec in neighbours:
                db.upsert_instance(rec)
            if implied_cls:
                db.decline_implied(data.desktop, implied_cls)
    except Exception as error:  # the database rolled back; so must memory
        raise EditError(f"could not delete {key!r}: {error}") from error

    del data.instances[key]
    for other in data.instances.values():
        for name in RELATION_FIELDS:
            if getattr(other, name) == key:
                setattr(other, name,
                        implied_cls if name == REVERTIBLE_FIELD else None)
    data.refresh_issues()
