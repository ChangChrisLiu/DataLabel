"""Removing an instance identity row in stage S1.

This is the one step-table command that cannot wait for ``Apply``:
:meth:`~tda.ui.steps_model.StepTableData.save` only upserts, so a deletion has
to be written through. That makes it the one place where "written through" has
to mean *completely* written through, which is what this module is for.

Two halves:

* :func:`check_deletable` -- refuse while anything a *human* made still depends
  on the key: a step targeting it, a shape keyframe in any of the four views, a
  constraint edge somebody decided on (manual / override / imported -- a
  **rule** edge is derived from the instance table itself and goes with the
  delete, see :func:`derived_relations`), a frame override, a layering exception, an entry
  in a z-order, an open conflict, a hand-written state event, or a **verified**
  compiled row. An ``auto`` compiled row is not one of those: the compiler
  writes one per instance the frame needs, geometry or not, so counting it made
  every instance the app had compiled in the background undeletable. It is a
  cache, and it goes with the delete, as the derived (``auto=True``) events do.
  A Label Studio **draft** key (:func:`tda.core.model.is_provisional`) has one
  more exception: the ``source="labelstudio"`` keyframes the importer gave it
  are part of the draft, not work done on it, so they do not block the delete --
  they go with it. Anything a human drew onto that key still does.
* :func:`delete_instance` -- do it in one transaction: drop the draft keyframes
  if it is a draft, drop the derived events and the cached rows, drop the
  identity row, and rewrite every neighbour that pointed at the key. A neighbour's pointer is *cleared*, except
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
from tda.core.graph_derive import RULE
from tda.core.graph_rules import HARD_TYPES, active_edges
from tda.core.implied import is_implied
from tda.core.logs import CHASSIS_KEY
from tda.core.ls_import import SOURCE as LS_SOURCE
from tda.core.model import VIEWS, InstanceRec, is_provisional
from tda.ui.steps_values import RELATION_FIELDS, EditError

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, fine for typing
    from tda.ui.steps_model import StepTableData

__all__ = ["ANNOTATOR", "OP_KIND", "OP_VIEW", "PAIRED_WITH_PARENT",
           "check_deletable", "delete_instance", "derived_relations",
           "stored_neighbours"]

#: ``op_log.annotator`` when the window did not say who is editing.
ANNOTATOR = "ui:steps"


def derived_relations(db: Db, desktop: int, key: str) -> list[tuple[str, str, str]]:
    """The **rule** edges naming ``key``: derived, so they go with the delete.

    A rule edge is not somebody's work, it is a reading of the instance table
    (spec 7.3), and the instance is about to leave that table -- the next
    derivation would drop the edge anyway. Dropping them here, in the same
    transaction, keeps the database free of a row pointing at a part that no
    longer exists. Every other source is a human's decision and blocks the
    delete instead (:func:`check_deletable`).
    """
    return [(rel["type"], rel["target"], rel["blocker"])
            for rel in db.relations(desktop)
            if rel.get("source") == RULE
            and key in (rel.get("target"), rel.get("blocker"))]


def check_deletable(data: "StepTableData", db: Db, key: str) -> None:
    """Raise :class:`EditError` unless nothing depends on ``key`` any more."""
    if key not in data.instances:
        raise EditError(f"D{data.desktop:02d} has no instance {key!r}")
    if key == CHASSIS_KEY:
        raise EditError("the chassis is implicit and cannot be deleted")
    steps = sorted({a.step for a in data.actions if a.target == key})
    if steps:
        raise EditError(f"{key!r} is still the target of step(s) {', '.join(map(str, steps))}")
    draft = is_provisional(key)
    for view in VIEWS:
        held = db.keyframes(data.desktop, view, key)
        if draft:
            # a draft's own draft keyframes go with it (see `delete_instance`);
            # anything a human drew onto it does not, and blocks the delete the
            # way it would on any other instance
            held = [kf for kf in held if kf.source != LS_SOURCE]
        if held:
            raise EditError(f"{key!r} still has shape keyframes in view {view!r}")
    for rel in db.relations(data.desktop):
        if key in (rel.get("target"), rel.get("blocker")) and rel.get("source") != RULE:
            raise EditError(f"{key!r} is still used by a {rel.get('type')!r} constraint "
                            f"edge ({rel.get('source')})")
    # A staged edge is not in the table yet, so the loop above cannot see it --
    # and this delete writes straight through, which would leave the Relations
    # tab holding an edge whose endpoint no longer exists and `Apply` writing a
    # dangling row. The staging is refused rather than silently cleaned: the
    # annotator asserted that edge one minute ago.
    for edge in data.relations.names(key) if data.relations is not None else ():
        raise EditError(
            f"{key!r} 还挂着未保存的 {edge.type} 约束边 / {key!r} is named by a staged "
            f"{edge.type} constraint edge; apply or revert the Relations tab first"
        )
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

    Clearing ``parent`` also unticks :data:`PAIRED_WITH_PARENT`, because the two
    are one answer: nothing reads ``attached`` without ``parent``
    (:func:`tda.core.states._attached_children` needs both), and the heuristics
    only ever *tick* the flag in the pass that fills the parent
    (:func:`tda.core.graph_infer._infer_screw`). A ``True`` left behind is
    therefore a decision about a part that no longer exists, waiting to be
    inherited by whatever parent is written in next -- for a captive screw and
    for a board-mounted latch alike. The next inference run refills both
    together, which is the only place either is meant to be set.
    """
    cleaned: list[InstanceRec] = []
    for other_key, stored in db.instances(desktop).items():
        if other_key == key or not any(getattr(stored, n) == key for n in RELATION_FIELDS):
            continue
        _clear_pointers(stored, key, revert_to)
        cleaned.append(stored)
    return cleaned


#: Unticked whenever ``parent`` is cleared; see :func:`stored_neighbours`.
PAIRED_WITH_PARENT = "attached"

#: ``op_log`` scope and kind of the row a delete leaves behind.
OP_KIND = "instance_delete"
OP_VIEW = "-"


def _restamp(db: Db, data: "StepTableData", key: str, dropped: int) -> None:
    """Re-stamp the desktop's graph meta after a delete. **In the transaction.**

    The delete takes the instance's rule edges with it, so the stored
    ``graph_version`` is about a graph that no longer exists -- and an export
    quoting it would be quoting something else. The version is read back through
    the accessor, exactly as the derivation and the ``constraints`` command do.
    """
    from tda.core.graph import edges_from_db, graph_version   # local: import cycle
    from tda.core.graph_derive import settled_instances
    from tda.core.graph_plan import find_deadlocks
    from tda.pipeline import merge_desktop_meta                # late: heavy module

    left = {k: rec for k, rec in data.instances.items() if k != key}
    edges = [e for e in edges_from_db(db, data.desktop) if e.type in HARD_TYPES]
    merge_desktop_meta(db, data.desktop, {
        "graph_version": graph_version(db, data.desktop),
        "graph_edges": len(active_edges(edges)),
        "graph_cycles": len(find_deadlocks(edges, settled_instances(left), data.tax)),
    })
    db.log_op(data.desktop, OP_VIEW, OP_KIND,
              {"instance": key, "relations": dropped},
              {"instance": key}, ANNOTATOR)


def _clear_pointers(rec: InstanceRec, key: str, revert_to: Optional[str]) -> None:
    """Drop every pointer ``rec`` holds to ``key``, in place."""
    for name in RELATION_FIELDS:
        if getattr(rec, name) != key:
            continue
        setattr(rec, name, revert_to if name == REVERTIBLE_FIELD else None)
        if name == "parent":
            setattr(rec, PAIRED_WITH_PARENT, False)


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
    derived = derived_relations(db, data.desktop, key)
    try:
        with db.transaction():
            for triple in derived:
                db.delete_relation(data.desktop, *triple)
            if is_provisional(key):
                # the draft and the shapes it was made of are one thing: a key
                # nobody adopted leaves nothing behind but orphaned pixels
                db.delete_keyframes_by_source(LS_SOURCE, [data.desktop], key)
            db.delete_auto_events(data.desktop, key)
            db.delete_instance(data.desktop, key)
            for rec in neighbours:
                db.upsert_instance(rec)
            if implied_cls:
                db.decline_implied(data.desktop, implied_cls)
            _restamp(db, data, key, len(derived))
    except Exception as error:  # the database rolled back; so must memory
        raise EditError(f"could not delete {key!r}: {error}") from error

    del data.instances[key]
    for other in data.instances.values():
        _clear_pointers(other, key, implied_cls)
    if implied_cls:
        # the same "no" the transaction just wrote, so the questions
        # `refresh_issues` is about to re-derive do not argue with it
        data.declined.add(implied_cls)
    data.refresh_issues()
