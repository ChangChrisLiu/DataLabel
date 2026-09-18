"""``infer-relations``: apply the spec-7.3 heuristic to a database already on disk.

::

    python -m tda.cli infer-relations [--desktops 1-66] [--dry-run]

``import-logs`` runs :func:`~tda.core.graph_rules.infer_relational_fields` as
part of each desktop's transaction, so every freshly imported machine already
has its ``screw.fastens``, its captive ``parent``/``attached`` pair, its latch
``of`` and a ``socket_host`` that names an *instance* rather than a class. This
command is for the database that was imported before that was wired in: it does
the same thing in place, without re-reading a single sheet, so no manual edit to
a step table is touched.

What makes it safe to run on live work:

* it takes the single-user lock and backs the database up first, exactly like
  ``import-logs --force`` (``--dry-run`` does neither, because it writes
  nothing);
* it never overwrites a field that already holds a value, so every human
  correction survives -- which also makes a second run report zero fills;
* one desktop is one transaction, and a desktop that raises is reported and
  skipped rather than ending the run;
* every changed instance gets an ``op_log`` row of kind :data:`OP_KIND` holding
  the old and the new value of each field, so the run is auditable and can be
  undone record by record.

Instance *keys* never change here -- only the relational columns -- so nothing
that points at an instance (keyframes above all) can be orphaned by it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Optional

from tda.core.db import Db
from tda.core.graph_infer import (
    AMBIGUOUS,
    NO_CANDIDATE,
    infer_relational_fields,
    unresolved_kind,
    unresolved_relations,
)
from tda.core.model import InstanceRec
from tda.core.states import events_from_actions
from tda.core.taxonomy import Taxonomy, load_taxonomy

__all__ = [
    "ANNOTATOR", "OP_KIND", "OP_VIEW", "DesktopRelations", "RelationsRun",
    "infer_relations_into_db",
]

#: ``op_log.kind`` of every row this command writes.
OP_KIND = "infer_relations"
#: ``op_log`` is scoped per ``(desktop, view)``; a relational field belongs to
#: no view, so it gets this placeholder rather than an arbitrary real one.
OP_VIEW = "-"
#: ``op_log.annotator`` -- this is machinery, not a person.
ANNOTATOR = "cli:infer-relations"

#: The instance columns the heuristic may fill (``attrs`` carries a latch's
#: ``of``), in the order they are reported.
TRACKED = ("parent", "attached", "mounted_on", "fastens", "socket_host", "attrs")


# --------------------------------------------------------------------------- #
# run records
# --------------------------------------------------------------------------- #
@dataclass
class DesktopRelations:
    """What ``infer-relations`` did for one desktop."""

    desktop: int
    status: str  # applied | failed
    fills: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    changed: int = 0  # instances whose stored row was rewritten
    error: str = ""

    def unresolved_of(self, kind: str) -> int:
        """How many unresolved lines are of one kind (ambiguous / no candidate)."""
        return sum(1 for line in self.unresolved if unresolved_kind(line) == kind)


@dataclass
class RelationsRun:
    """The whole run."""

    dry_run: bool = False
    runs: list[DesktopRelations] = field(default_factory=list)

    @property
    def applied(self) -> list[DesktopRelations]:
        return [r for r in self.runs if r.status == "applied"]

    @property
    def failed(self) -> list[DesktopRelations]:
        return [r for r in self.runs if r.status == "failed"]

    @property
    def fills(self) -> int:
        return sum(len(r.fills) for r in self.runs)

    @property
    def unresolved(self) -> int:
        return sum(len(r.unresolved) for r in self.runs)

    def unresolved_of(self, kind: str) -> int:
        return sum(r.unresolved_of(kind) for r in self.runs)

    @property
    def changed(self) -> int:
        return sum(r.changed for r in self.runs)

    def tally(self) -> str:
        """``N unresolved (A ambiguous, C no candidate)`` for a summary line.

        The split is what decides who does the next piece of work: an ambiguous
        reference needs one of several instances picked, a missing one needs an
        instance created first -- and whether a never-removed part gets an
        instance at all is the user's call (decision C7), not this command's.
        """
        return (
            f"{self.unresolved} unresolved ({self.unresolved_of(AMBIGUOUS)} "
            f"{AMBIGUOUS}, {self.unresolved_of(NO_CANDIDATE)} {NO_CANDIDATE})"
        )


# --------------------------------------------------------------------------- #
# what changed
# --------------------------------------------------------------------------- #
def _snapshot(rec: InstanceRec) -> dict[str, Any]:
    """The tracked fields of one instance, deep enough to compare afterwards."""
    return {
        name: dict(rec.attrs) if name == "attrs" else getattr(rec, name)
        for name in TRACKED
    }


def _diff(before: dict[str, Any], after: dict[str, Any]) -> tuple[dict, dict]:
    """``(new, old)`` for the fields that differ; both empty when nothing did."""
    names = [name for name in TRACKED if before[name] != after[name]]
    return ({n: after[n] for n in names}, {n: before[n] for n in names})


# --------------------------------------------------------------------------- #
# one desktop
# --------------------------------------------------------------------------- #
def _apply_one(db: Db, tax: Taxonomy, desktop: int, dry_run: bool) -> DesktopRelations:
    """Infer, and (unless ``dry_run``) write, one desktop's relational fields.

    The whole desktop lands in one transaction: the changed instance rows, an
    ``op_log`` row per changed instance, and -- only when something actually
    changed -- the stored automatic state events, which are recompiled because
    the cascade of spec 3.3 reads ``attached``/``parent`` off the instance
    table. (:func:`tda.core.truth_inputs.events_of` re-derives that log on every
    read and ignores the stored ``auto=True`` rows, so this is about the copy
    :mod:`tda.ui.steps_model` and the status report read directly.)
    """
    instances = db.instances(desktop)
    actions = db.actions(desktop)
    before = {key: _snapshot(rec) for key, rec in instances.items()}
    fills = infer_relational_fields(instances, tax, actions)
    changes = []
    for key, rec in sorted(instances.items()):
        new, old = _diff(before[key], _snapshot(rec))
        if new:
            changes.append((key, new, old))
    out = DesktopRelations(
        desktop=desktop, status="applied", fills=fills,
        unresolved=unresolved_relations(instances, tax, actions),
        changed=len(changes),
    )
    if dry_run or not changes:
        return out
    with db.transaction():
        for key, new, old in changes:
            db.upsert_instance(instances[key])
            db.log_op(
                desktop, OP_VIEW, OP_KIND,
                {"instance": key, "fields": new}, {"instance": key, "fields": old},
                ANNOTATOR,
            )
        db.replace_events(
            desktop, events_from_actions(instances, actions, tax), auto_only=True
        )
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def infer_relations_into_db(
    db: Db,
    tax: Taxonomy,
    desktops: Optional[set[int]] = None,
    dry_run: bool = False,
    log=None,
) -> RelationsRun:
    """Run the heuristic over every desktop in the database, one transaction each."""
    run = RelationsRun(dry_run=dry_run)
    prefix = "[infer-relations]" + (" (dry run)" if dry_run else "")
    for desktop in db.desktop_ids():
        if desktops is not None and desktop not in desktops:
            continue
        try:
            one = _apply_one(db, tax, desktop, dry_run)
        except Exception as exc:  # one bad desktop must not end the run
            one = DesktopRelations(
                desktop=desktop, status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        run.runs.append(one)
        if log:
            _log_desktop(log, prefix, one)
    if log:
        log(f"{prefix} {len(run.applied)} desktops, {run.fills} fills on {run.changed} "
            f"instances, {run.tally()}, {len(run.failed)} failed")
    return run


def _log_desktop(log, prefix: str, one: DesktopRelations) -> None:
    """One desktop's headline plus every line it produced."""
    if one.status == "failed":
        log(f"{prefix} D{one.desktop:02d}: FAILED, {one.error}")
        return
    log(f"{prefix} D{one.desktop:02d}: {len(one.fills)} fills on {one.changed} "
        f"instances, {len(one.unresolved)} unresolved "
        f"({one.unresolved_of(AMBIGUOUS)} {AMBIGUOUS}, "
        f"{one.unresolved_of(NO_CANDIDATE)} {NO_CANDIDATE})")
    for text in one.fills:
        log(f"{prefix}   {text}")
    for text in one.unresolved:
        log(f"{prefix}   {text}")


# --------------------------------------------------------------------------- #
# command
# --------------------------------------------------------------------------- #
def cmd_infer_relations(args: argparse.Namespace) -> int:
    """Fill the relational fields of a database that was imported without them."""
    # late import: tda.cli imports this module to register the subcommand
    from tda.cli import EXIT_ERROR, EXIT_OK, _desktops, _safety_backup, _session

    with _session(args, lock=True) as (paths, db):
        if not args.dry_run:
            _safety_backup(paths, db, "infer-relations",
                           "it rewrites the relational fields of every desktop")
        run = infer_relations_into_db(
            db, load_taxonomy(), _desktops(args), args.dry_run, log=print
        )
        if run.failed:
            listed = ", ".join(f"D{r.desktop:02d}" for r in run.failed)
            print(f"[infer-relations] {listed} failed; every other desktop was applied")
            return EXIT_ERROR
        return EXIT_OK


def _add_infer_relations(sub) -> None:
    p = sub.add_parser(
        "infer-relations",
        help="fill empty screw.fastens / parent / latch.of / socket_host in place",
    )
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be filled without writing anything "
                        "(and without taking a backup)")
    p.set_defaults(func=cmd_infer_relations)
