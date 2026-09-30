"""Parts the picture shows but the log never names (task U5b, spec 3.2).

The log importer creates an instance for every part a *step* names, and nothing
else. That is the right rule for identity, and it leaves a hole wherever a part
is visible but was never operated: D13 has four RAM slots and eight retention
clips, the sheet opened the four clips of the two filled slots (steps 14-17 ->
``ram_latch.01``-``04``), and the four clips of the two empty slots exist in
every frame and in no row of the database. A visible part with no instance is
trained as *background* by the detector, and a VLM question about the state of
"every clip" cannot be answered for it.

So stage S1 lets an annotator add them. An **extra** instance is an ordinary
:class:`~tda.core.model.InstanceRec`:

* its key continues the class's ordinal run (``ram_latch.05`` after ``.04``,
  :func:`planned_keys`), so nothing downstream can tell it by its name;
* ``parent`` + ``attached`` follow the class's host rule: the unique instance of
  the taxonomy's ``host_class`` when there is exactly one
  (:func:`default_parent`), otherwise whatever the annotator picks. That is the
  whole of its lifetime -- it has no action, so it stands in its initial state
  from step 1 until its host leaves and the spec-3.3 cascade takes it out,
  which makes it a ✚ row "跟 motherboard.01 一起装回来的" exactly like the clips
  the log did name;
* its initial state is one of the class's states (default ``default_state``)
  and, when it is not the default, a hand-written ``state`` event at
  :data:`~tda.core.states.INITIAL_STEP` -- the existing state mechanism, not a
  new column;
* ``attrs`` carry the provenance (``added_by``, ``added_at``, ``reason``, an
  optional ``note``) and ``raw_names`` stays empty: no sheet ever named it.

``socket_host`` and ``cable`` are deliberately left empty. They are relations
the constraint rules turn into ``connected_to`` edges, and an edge nobody
decided on would make the recorded teardown "violate" a constraint on a part
the operator never touched. The annotator can fill them in the Instances table.

Everything here is Qt-free. :class:`ExtraBatch` is what one "＋ 添加零件" makes;
:func:`write_batch` stores it with one ``op_log`` row whose inverse undoes it
(:func:`undo_op`), :func:`delete_patch` is the record a deletion logs and
:func:`adopt_patch` the one ``import-logs --adopt-extras`` logs when a sheet
starts naming an added key (:func:`tda.pipeline_logs.extra_clash` refuses that
import otherwise). Every undo recompiles the stored automatic events and
re-stamps ``graph_version``, so the database is as it was before the op.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from tda.core.db import Db
from tda.core.dbrows import now_iso
from tda.core.log_identity import CHASSIS_KEY, instance_key
from tda.core.model import (
    EXTRA_ATTR,
    VIEWS,
    InstanceRec,
    StateEvent,
    is_extra,
    is_provisional,
)
from tda.core.states import ATTR_STATE, INITIAL_STEP, REMOVED
from tda.core.taxonomy import Taxonomy

__all__ = [
    "ADDED_AT", "ADDED_BY", "MAX_COUNT", "NOTE_ATTR", "OP_ADD", "OP_ADOPT", "OP_DELETE",
    "OP_UNDO", "OP_VIEW", "REASON", "REASON_ATTR", "ExtraBatch",
    "ExtraError", "adopt_patch", "default_parent", "delete_patch", "extra_keys",
    "host_instances", "initial_events", "is_extra", "next_ordinal", "parent_choices",
    "plan_extras", "planned_keys", "recompile_events", "relation_rows",
    "restamp_graph", "state_choices", "undo_op", "write_batch",
]

#: ``attrs`` provenance keys of an extra instance.
ADDED_BY = EXTRA_ATTR
ADDED_AT = "added_at"
REASON_ATTR = "reason"
NOTE_ATTR = "note"
#: ``attrs["reason"]`` of every extra: why it exists at all.
REASON = "not in the log"

#: ``op_log.kind`` of one "add parts" batch, of one deletion, and of an undo
#: through the op log's inverse.
OP_ADD = "add_extra_instances"
OP_DELETE = "delete_extra_instance"
OP_UNDO = "undo_extra_instances"
#: ``op_log.kind`` of ``import-logs --adopt-extras`` handing added instances to
#: a sheet that now names their keys (:mod:`tda.pipeline_logs`).
OP_ADOPT = "adopt_extra_instances"
#: Instance fields holding another instance's key (``steps_values.RELATION_FIELDS``).
_POINTERS = ("parent", "mounted_on", "fastens", "socket_host")
#: ``op_log`` is scoped per ``(desktop, view)``; an identity row belongs to none.
OP_VIEW = "-"

#: How many instances one batch may add (the dialog's spin box range).
MIN_COUNT, MAX_COUNT = 1, 16
#: ``relation.source`` of an edge derived from the instance table (mirrors
#: :data:`tda.core.graph_derive.RULE`, which this module may not import: the
#: graph stack sits above the database layer).
_RULE = "rule"

_ORDINAL = re.compile(r"\.(\d+)$")


class ExtraError(ValueError):
    """An extra instance that cannot be made or removed; the text is UI-ready."""


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #
def next_ordinal(keys: Iterable[str], cls: str, attrs: Optional[dict] = None) -> int:
    """One past the highest ordinal among the keys of the same ordinal run.

    The run is the class plus its discriminator (``role`` / ``kind``), exactly
    as :func:`tda.core.log_identity.instance_key` builds keys, so an added
    ``screw.motherboard`` continues after ``screw.motherboard.07`` and not after
    ``screw.cpu_cooler.04``.
    """
    if cls == CHASSIS_KEY:
        return 1
    disc = str((attrs or {}).get("role") or (attrs or {}).get("kind") or "")
    prefix = f"{cls}.{disc}." if disc else f"{cls}."
    top = 0
    for key in keys:
        if not key.startswith(prefix):
            continue
        match = _ORDINAL.match(key[len(prefix) - 1:])
        if match:
            top = max(top, int(match.group(1)))
    return top + 1


def planned_keys(keys: Iterable[str], cls: str, attrs: Optional[dict],
                 count: int) -> list[str]:
    """The ``count`` keys a batch of ``cls`` would get, in order."""
    first = next_ordinal(list(keys), cls, attrs)
    return [instance_key(cls, dict(attrs or {}), first + i) for i in range(int(count))]


def extra_keys(instances: dict[str, InstanceRec]) -> list[str]:
    """The keys of the extra instances of one desktop, sorted."""
    return sorted(key for key, rec in instances.items() if is_extra(rec))


# --------------------------------------------------------------------------- #
# host and state
# --------------------------------------------------------------------------- #
def host_instances(instances: dict[str, InstanceRec], tax: Taxonomy, cls: str) -> list[str]:
    """The real instances of ``cls``'s ``host_class``; empty when it has none."""
    host = tax.host_class(cls)
    if not host:
        return []
    return sorted(key for key, rec in instances.items()
                  if rec.cls == host and not is_provisional(key))


def default_parent(instances: dict[str, InstanceRec], tax: Taxonomy,
                   cls: str) -> Optional[str]:
    """The parent a new ``cls`` instance gets unless the annotator says else.

    The unique instance of the class's ``host_class`` (a ``ram_latch`` rides on
    the one ``motherboard``), and nothing when the class declares no host or the
    desktop has none or several: then the annotator picks.
    """
    found = host_instances(instances, tax, cls)
    return found[0] if len(found) == 1 else None


def parent_choices(instances: dict[str, InstanceRec], tax: Taxonomy, cls: str) -> list[str]:
    """What the dialog offers as a parent: the host-class instances first.

    Then every other real instance, because a clip can ride on a cage or a
    cooler just as well -- only the taxonomy's host rule is a default.
    """
    first = host_instances(instances, tax, cls)
    rest = sorted(key for key, rec in instances.items()
                  if not is_provisional(key) and key not in first)
    return [*first, *rest]


def state_choices(tax: Taxonomy, cls: str) -> list[str]:
    """The states a part the picture shows can start in: every one but ``removed``.

    ``removed`` is where a part is *not* in the picture, which is the one thing
    an extra instance never is when it is added.
    """
    return [s for s in tax.states_of(cls) if s != REMOVED]


# --------------------------------------------------------------------------- #
# one batch
# --------------------------------------------------------------------------- #
@dataclass
class ExtraBatch:
    """What one "＋ 添加零件" makes: the records and their initial-state events."""

    desktop: int
    records: list[InstanceRec] = field(default_factory=list)
    #: Hand-written ``state`` events at :data:`INITIAL_STEP`, one per record
    #: whose initial state is not the class default; empty otherwise.
    events: list[StateEvent] = field(default_factory=list)

    @property
    def keys(self) -> list[str]:
        return [rec.key for rec in self.records]

    def payload(self) -> dict:
        """The ``op_log`` payload: every record and event, in full."""
        return {"instances": [asdict(rec) for rec in self.records],
                "events": [asdict(event) for event in self.events]}

    def inverse(self) -> dict:
        """The patch that takes the batch back: :func:`undo_op` applies it."""
        return {"delete_instances": self.keys,
                "delete_events": [asdict(event) for event in self.events]}

    def drop(self, key: str) -> None:
        """Forget one staged record (and its event) before it is written."""
        self.records = [rec for rec in self.records if rec.key != key]
        self.events = [event for event in self.events if event.target != key]


def plan_extras(
    instances: dict[str, InstanceRec],
    tax: Taxonomy,
    desktop: int,
    cls: str,
    count: int,
    *,
    parent: Optional[str] = None,
    state: Optional[str] = None,
    note: str = "",
    annotator: str = "",
    attrs: Optional[dict] = None,
    now: Optional[str] = None,
) -> ExtraBatch:
    """Build ``count`` extra instances of ``cls`` without registering them.

    ``parent`` is taken as given -- the caller pre-fills it with
    :func:`default_parent` -- and ticks ``attached`` when set, so the part
    leaves the chassis with it. ``state`` defaults to the class's
    ``default_state``. Raises :class:`ExtraError` for anything the taxonomy or
    the instance table refuses.
    """
    if cls not in tax.classes:
        raise ExtraError(f"{cls!r} is not a taxonomy class")
    if cls == CHASSIS_KEY:
        raise ExtraError("the chassis is unique per desktop and cannot be added")
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise ExtraError(f"count must be a whole number, got {count!r}") from None
    if not MIN_COUNT <= count <= MAX_COUNT:
        raise ExtraError(f"count must be {MIN_COUNT}-{MAX_COUNT}, got {count}")
    default = tax.default_state(cls)
    state = str(state or default)
    allowed = state_choices(tax, cls)
    if state not in allowed:
        raise ExtraError(f"{state!r} is not a starting state of {cls!r} "
                         f"({', '.join(allowed)})")
    parent = str(parent).strip() if parent else None
    if parent is not None:
        if parent not in instances:
            raise ExtraError(f"D{desktop:02d} has no instance {parent!r}")
        if is_provisional(parent):
            raise ExtraError(f"{parent!r} is a Label Studio draft, not a part")
    stamp = now or now_iso()
    base = {k: v for k, v in dict(attrs or {}).items() if v not in (None, "")}
    batch = ExtraBatch(desktop=desktop)
    for key in planned_keys(instances, cls, base, count):
        if key in instances:  # the ordinal run is ahead of the table: never
            raise ExtraError(f"D{desktop:02d} already has an instance {key!r}")
        rec_attrs = dict(base)
        rec_attrs.update({ADDED_BY: str(annotator or "unknown"), ADDED_AT: stamp,
                          REASON_ATTR: REASON})
        if note and note.strip():
            rec_attrs[NOTE_ATTR] = note.strip()
        batch.records.append(InstanceRec(
            key=key, desktop=desktop, cls=cls, attrs=rec_attrs,
            parent=parent, attached=parent is not None, raw_names=[],
        ))
        if state != default:
            batch.events.append(StateEvent(
                desktop=desktop, step=INITIAL_STEP, target=key, attr=ATTR_STATE,
                old=default, new=state, evidence_view=None, auto=False,
            ))
    return batch


def write_batch(db: Db, batch: ExtraBatch, annotator: str) -> int:
    """Store one batch -- rows, events, one ``op_log`` row -- in one transaction.

    Re-entrant: inside S1's ``Apply`` it joins that transaction. Returns the
    ``op_log`` id.
    """
    with db.transaction():
        for rec in batch.records:
            db.upsert_instance(rec)
        db.add_events(batch.events)
        return db.log_op(batch.desktop, OP_VIEW, OP_ADD, batch.payload(),
                         batch.inverse(), annotator)


# --------------------------------------------------------------------------- #
# deletion and undo
# --------------------------------------------------------------------------- #
def initial_events(db: Db, desktop: int, key: str) -> list[StateEvent]:
    """The hand-written events of ``key`` at or before :data:`INITIAL_STEP`."""
    return [e for e in db.events(desktop)
            if e.target == key and not e.auto and e.step <= INITIAL_STEP]


#: The ``relation`` columns an inverse carries, so an edge comes back as it was.
_RELATION_COLUMNS = ("type", "target", "blocker", "necessity", "mode", "reason",
                     "source", "evidence_step", "status")


def relation_rows(db: Db, desktop: int, key: str, source: str = _RULE) -> list[dict]:
    """The stored ``source`` edges naming ``key``, as restorable dicts."""
    return [{c: row[c] for c in _RELATION_COLUMNS} for row in db.relations(desktop)
            if key in (row["target"], row["blocker"]) and row["source"] == source]


def delete_patch(rec: InstanceRec, events: list[StateEvent],
                 relations: Iterable[dict] = ()) -> tuple[dict, dict]:
    """``(payload, inverse)`` of the ``op_log`` row deleting one extra.

    The inverse is the whole record, its initial events and the rule edges the
    delete took with it (:func:`relation_rows`), so :func:`undo_op` can put the
    instance back exactly as it was.
    """
    record = {"instances": [asdict(rec)], "events": [asdict(e) for e in events],
              "relations": [dict(r) for r in relations]}
    payload = {"instance": rec.key, **record}
    return payload, dict(record)


def adopt_patch(records: Iterable[InstanceRec], events: Iterable[StateEvent],
                taken_by: dict[str, str]) -> tuple[dict, dict]:
    """``(payload, inverse)`` of the ``op_log`` row an adoption writes.

    ``taken_by`` says per key what now owns it (``"log"`` or ``"implied"``).
    The inverse is the added records and their initial events as they were, so
    :func:`undo_op` can hand the keys back; the shapes never moved.
    """
    records = list(records)
    record = {"instances": [asdict(r) for r in records],
              "events": [asdict(e) for e in events]}
    payload = {"adopted": [r.key for r in records], "taken_by": dict(taken_by), **record}
    return payload, dict(record)


def _refusal(db: Db, desktop: int, key: str) -> str:
    """Why ``key`` may not be removed by an undo, or ``""``."""
    views = [view for view in VIEWS if db.keyframes(desktop, view, key)]
    if views:
        return f"{key} already has shapes in {', '.join(views)}"
    steps = sorted({a.step for a in db.actions(desktop) if a.target == key})
    if steps:
        return f"{key} is the target of step(s) {', '.join(map(str, steps))}"
    named = sorted(f"{other}.{name}" for other, rec in db.instances(desktop).items()
                   for name in _POINTERS if other != key and getattr(rec, name) == key)
    if named:
        return f"{key} is named by {', '.join(named)}"
    counts = {table: n for table, n in db.instance_reference_counts(desktop, key).items()
              if table != "state_event"}
    later = [e for e in db.events(desktop)
             if e.target == key and not e.auto and e.step > INITIAL_STEP]
    if later:
        counts["state_event"] = len(later)
    decided = [r for r in db.relations(desktop)
               if key in (r["target"], r["blocker"]) and r["source"] != _RULE]
    if decided:
        counts["relation"] = len(decided)
    if counts:
        named = ", ".join(f"{n} {table}" for table, n in sorted(counts.items()))
        return f"{key} is still referenced ({named})"
    return ""


def _event(data: dict) -> StateEvent:
    return StateEvent(**{k: data[k] for k in (
        "desktop", "step", "target", "attr", "old", "new", "evidence_view", "auto")})


def recompile_events(db: Db, desktop: int, tax: Taxonomy) -> None:
    """Rewrite the stored automatic event log of ``desktop`` from its actions.

    What ``Apply`` stores, from the same inputs: the instance table, the actions
    and the hand-written initial states. **In the caller's transaction.**
    """
    from tda.core.states import events_from_actions, initial_overrides

    manual = [e for e in db.events(desktop) if not e.auto]
    db.replace_events(desktop, events_from_actions(
        db.instances(desktop), db.actions(desktop), tax,
        initial=initial_overrides(manual)), auto_only=True)


def restamp_graph(db: Db, desktop: int, tax: Taxonomy) -> None:
    """Re-stamp the desktop's graph meta from what is stored. **In the transaction.**

    The same three fields :func:`tda.ui.steps_delete._restamp` writes after a
    delete: an undo that takes a rule edge away (or brings one back) must not
    leave ``graph_version`` describing a graph that no longer exists.
    """
    from tda.core.graph import edges_from_db, graph_version
    from tda.core.graph_derive import settled_instances
    from tda.core.graph_plan import find_deadlocks
    from tda.core.graph_rules import HARD_TYPES, active_edges
    from tda.core.states import initial_overrides
    from tda.pipeline import merge_desktop_meta   # late: tda.pipeline is heavy

    edges = [e for e in edges_from_db(db, desktop) if e.type in HARD_TYPES]
    initial = initial_overrides(e for e in db.events(desktop) if not e.auto)
    merge_desktop_meta(db, desktop, {
        "graph_version": graph_version(db, desktop),
        "graph_edges": len(active_edges(edges)),
        "graph_cycles": len(find_deadlocks(edges, settled_instances(db.instances(desktop)),
                                           tax, initial=initial)),
    })


def _restore(db: Db, desktop: int, inverse: dict) -> None:
    """Put records, their initial events and their edges back. **In the transaction.**"""
    for data in inverse.get("instances") or []:
        db.upsert_instance(InstanceRec(**data))
    keys = [str(data["key"]) for data in inverse.get("instances") or []]
    for key in keys:
        # an adopted key's initial state is the log's now: whatever hand-written
        # one is stored for it is replaced by the one the inverse carries
        db.delete_manual_events(desktop, key, up_to_step=INITIAL_STEP)
    db.add_events(_event(data) for data in inverse.get("events") or [])
    for row in inverse.get("relations") or []:
        db.add_relation(desktop, row["type"], row["target"], row["blocker"],
                        necessity=row.get("necessity") or "required", mode=row.get("mode"),
                        reason=row.get("reason"), source=row.get("source") or _RULE,
                        evidence_step=row.get("evidence_step"),
                        status=row.get("status") or "active")


def undo_op(db: Db, op: dict, annotator: str = "", tax: Optional[Taxonomy] = None) -> int:
    """Apply the inverse of one ``op_log`` row this module (or the importer) wrote.

    ``op`` is a row as :meth:`tda.core.db.Db.ops` returns it.

    * :data:`OP_ADD` is undone by deleting the instances it added -- refused,
      with nothing written, while any of them carries a shape, an action, a
      pointer from another instance or anything else a human made;
    * :data:`OP_DELETE` by putting the record, its initial events and the rule
      edges the delete took back;
    * :data:`OP_ADOPT` (``import-logs --adopt-extras``) by putting the added
      records back over the rows the log took them over with.

    Every branch then recompiles the stored automatic events -- the cascade
    that takes an added clip out with its board is part of that log -- and
    re-stamps ``graph_version``, so the database is as it was before the op.
    The undo is logged as :data:`OP_UNDO`, with the op it undid as its own
    inverse. Returns that row's id.
    """
    if tax is None:
        from tda.core.taxonomy import load_taxonomy

        tax = load_taxonomy()
    kind = str(op.get("kind") or "")
    inverse = op.get("inverse") or {}
    desktop = int(op["desktop"])
    who = annotator or str(op.get("annotator") or "")
    if kind == OP_ADD:
        keys = [str(k) for k in inverse.get("delete_instances") or []]
        stored = db.instances(desktop)
        for key in keys:
            if key in stored and not is_extra(stored[key]):
                raise ExtraError(f"{key} is not an added instance any more")
            why = _refusal(db, desktop, key) if key in stored else ""
            if why:
                raise ExtraError(f"cannot undo the addition: {why}")
        with db.transaction():
            for key in keys:
                for row in db.relations(desktop):
                    # a rule edge is a reading of the instance table: it goes
                    # with the instance it was read off (`_refusal` has already
                    # refused any edge a human decided on)
                    if key in (row["target"], row["blocker"]) and row["source"] == _RULE:
                        db.delete_relation(desktop, row["type"], row["target"],
                                           row["blocker"])
                db.delete_manual_events(desktop, key, up_to_step=INITIAL_STEP)
                db.delete_auto_events(desktop, key)
                db.delete_instance(desktop, key)
            recompile_events(db, desktop, tax)
            restamp_graph(db, desktop, tax)
            return db.log_op(desktop, OP_VIEW, OP_UNDO, {"undid": int(op["id"]), **inverse},
                             op.get("payload") or {}, who)
    if kind in (OP_DELETE, OP_ADOPT):
        records = [InstanceRec(**data) for data in inverse.get("instances") or []]
        stored = db.instances(desktop)
        if kind == OP_DELETE:
            clash = [rec.key for rec in records if rec.key in stored]
            if clash:
                raise ExtraError(f"cannot undo the deletion: {', '.join(clash)} exists again")
        with db.transaction():
            _restore(db, desktop, inverse)
            recompile_events(db, desktop, tax)
            restamp_graph(db, desktop, tax)
            return db.log_op(desktop, OP_VIEW, OP_UNDO, {"undid": int(op["id"]), **inverse},
                             op.get("payload") or {}, who)
    raise ExtraError(f"op {op.get('id')} ({kind!r}) is not one this module can undo")
