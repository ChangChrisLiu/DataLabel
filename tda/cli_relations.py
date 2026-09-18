"""``infer-relations``: apply the spec-7.3 heuristic to a database already on disk.

::

    python -m tda.cli infer-relations [--desktops 1-66] [--dry-run] [--force]

``import-logs`` runs :func:`~tda.core.graph_rules.infer_relational_fields` as
part of each desktop's transaction, so every freshly imported machine already
has its ``screw.fastens``, its captive ``parent``/``attached`` pair, its latch
``of`` and a ``socket_host`` that names an *instance* rather than a class. This
command is for the database that was imported before that was wired in: it does
the same thing in place, without re-reading a single sheet, so no manual edit to
a step table is touched.

What makes it safe to run on live work:

* it takes the single-user lock and backs the database up first, exactly like
  ``import-logs --force`` (``--dry-run`` does neither, because it writes no
  data -- the database is still opened and schema-checked under the lock);
* a desktop that already carries **verified** frames is reported and refused
  unless ``--force``, because filling a captive screw's ``parent`` changes what
  later frames need geometry for;
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
from tda.core.implied import OP_KIND as IMPLIED_OP_KIND
from tda.core.implied import implied_instances
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
    status: str  # applied | refused | failed
    fills: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    changed: int = 0  # instances whose stored row was rewritten
    #: One line per instance ``--add-implied`` created for this desktop.
    implied: list[str] = field(default_factory=list)
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
    def refused(self) -> list[DesktopRelations]:
        """Desktops skipped because they carry verified work and ``--force`` was not given."""
        return [r for r in self.runs if r.status == "refused"]

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

    @property
    def implied(self) -> int:
        return sum(len(r.implied) for r in self.runs)

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
def _apply_one(db: Db, tax: Taxonomy, desktop: int, dry_run: bool,
               add_implied: bool = False) -> DesktopRelations:
    """Infer, and (unless ``dry_run``) write, one desktop's relational fields.

    The whole desktop lands in one transaction: the implied instances, the
    changed instance rows, an ``op_log`` row per new or changed instance, and --
    only when something actually changed -- the stored automatic state events,
    which are recompiled because the cascade of spec 3.3 reads
    ``attached``/``parent`` off the instance table.
    (:func:`tda.core.truth_inputs.events_of` re-derives that log on every read
    and ignores the stored ``auto=True`` rows, so this is about the copy
    :mod:`tda.ui.steps_model` and the status report read directly.)

    ``add_implied`` runs :func:`tda.core.implied.implied_instances` **before**
    the heuristic, exactly as ``import-logs`` does, so the references the four
    never-lifted motherboards leave behind resolve onto the new instance in the
    same pass.
    """
    instances = db.instances(desktop)
    actions = db.actions(desktop)
    new_instances = implied_instances(instances, actions, tax) if add_implied else []
    for rec in new_instances:
        instances[rec.key] = rec
    before = {key: _snapshot(rec) for key, rec in instances.items()}
    fills = infer_relational_fields(instances, tax, actions)
    made = {rec.key for rec in new_instances}
    changes = []
    for key, rec in sorted(instances.items()):
        new, old = _diff(before[key], _snapshot(rec))
        if new and key not in made:
            changes.append((key, new, old))
    out = DesktopRelations(
        desktop=desktop, status="applied", fills=fills,
        unresolved=unresolved_relations(instances, tax, actions),
        changed=len(changes),
        implied=[f"implied instance {rec.key}: {rec.attrs.get('note', '')}"
                 for rec in new_instances],
    )
    if dry_run or not (changes or new_instances):
        return out
    with db.transaction():
        for rec in new_instances:
            db.upsert_instance(instances[rec.key])
            db.log_op(
                desktop, OP_VIEW, IMPLIED_OP_KIND,
                {"instance": rec.key, "cls": rec.cls, "attrs": dict(instances[rec.key].attrs)},
                {"instance": rec.key}, ANNOTATOR,
            )
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
def verified_frames(db: Db, desktop: int) -> int:
    """How many *frames* of this desktop a human has frozen (spec 3.4).

    One frame is one ``(view, step)``. Counting ``compiled_mask`` rows instead
    -- one per instance of the frame -- is what made this report "43 verified
    frames" for a desktop with two, which reads like a reason to stop rather
    than like the truth. :meth:`tda.core.db.Db.verified_frame_count` owns the
    query now, so the guard and the re-check queue cannot drift apart.
    """
    return db.verified_frame_count(desktop)


def _queue_rechecks(db: Db, desktop: int, log=None) -> int:
    """Queue this desktop's verified frames for a re-check by the truth service.

    The relational fills can change which instances a later frame needs, so
    every frozen frame of the desktop is handed to the persisted re-check queue
    (one request per view). Nothing frozen is lost either way: the truth
    service raises a conflict when a verified row disappears from a recompiled
    frame; queueing only makes the annotator meet it now rather than later.

    The queueing happens whether or not there is anywhere to report it: a
    ``log`` of ``None`` is a caller that prints nothing, not a caller that wants
    the frozen frames left unchecked. Returns how many were queued.
    """
    by_view: dict[str, list[int]] = {}
    for view, step in db.verified_frames(desktop):
        by_view.setdefault(view, []).append(step)
    queued = 0
    for view, steps in by_view.items():
        queued += len(db.add_rechecks(desktop, view, steps))
    if log:
        log(f"[infer-relations] D{desktop:02d}: queued {queued} verified frames for "
            f"re-check (open the desktop in the app, or run `python -m tda.cli check "
            f"--desktop {desktop}`)")
    return queued


def _selected(db: Db, desktops: Optional[set[int]], log) -> list[int]:
    """The desktops to work on, warning about any ``--desktops`` id the DB lacks."""
    known = db.desktop_ids()
    if desktops is None:
        return known
    missing = sorted(desktops - set(known))
    if missing and log:
        for desktop in missing:
            log(f"[infer-relations] D{desktop:02d} is not in the database; skipped")
    return [desktop for desktop in known if desktop in desktops]


def infer_relations_into_db(
    db: Db,
    tax: Taxonomy,
    desktops: Optional[set[int]] = None,
    dry_run: bool = False,
    force: bool = False,
    log=None,
    add_implied: bool = False,
) -> RelationsRun:
    """Run the heuristic over every desktop in the database, one transaction each.

    A desktop that already carries **verified** compiled rows is reported and
    then refused unless ``force``: filling a captive screw's ``parent`` changes
    what :func:`tda.core.states.needs_geom` answers on every later frame, so a
    row a human froze can stop being produced. That is not silent -- the truth
    service turns a vanished verified row into a conflict on the next refresh --
    but it is not something to spring on an annotator either. ``dry_run``
    reports the count and carries on, because it writes nothing.
    """
    run = RelationsRun(dry_run=dry_run)
    prefix = "[infer-relations]" + (" (dry run)" if dry_run else "")
    selected = _selected(db, desktops, log)

    # Pre-scan: a run that would refuse desktop 40 must not first have rewritten
    # thirty-nine. The whole selection is checked before anything is written, so
    # the annotator can decide once -- --force, or a narrower --desktops.
    frozen_by_desktop = {d: verified_frames(db, d) for d in selected}
    for desktop, frozen in frozen_by_desktop.items():
        if frozen and log:
            log(f"{prefix} D{desktop:02d} has {frozen} verified frames; their frozen "
                f"rows will be re-checked and may raise conflicts")
    if not (force or dry_run) and any(frozen_by_desktop.values()):
        for desktop, frozen in frozen_by_desktop.items():
            if frozen:
                run.runs.append(DesktopRelations(
                    desktop=desktop, status="refused",
                    error=f"{frozen} verified frames; re-run with --force to proceed",
                ))
        if log:
            listed = ", ".join(f"D{r.desktop:02d}" for r in run.refused)
            log(f"{prefix} refused before touching anything: {listed} carry verified "
                f"frames (use --force to proceed anyway)")
        return run

    for desktop in selected:
        frozen = frozen_by_desktop[desktop]
        try:
            one = _apply_one(db, tax, desktop, dry_run, add_implied)
        except Exception as exc:  # one bad desktop must not end the run
            one = DesktopRelations(
                desktop=desktop, status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        run.runs.append(one)
        if log:
            _log_desktop(log, prefix, one)
        if frozen and one.changed and not dry_run:
            _queue_rechecks(db, desktop, log)
    if log:
        log(f"{prefix} {len(run.applied)} desktops, {run.implied} implied instances, "
            f"{run.fills} fills on {run.changed} instances, {run.tally()}, "
            f"{len(run.refused)} refused, {len(run.failed)} failed")
        if run.failed:
            # the one thing a reader wants after a stack of per-desktop lines:
            # is the rest of the database in the state this command promises?
            log(f"{prefix} {len(run.failed)} desktops failed; every other desktop was "
                f"applied. Re-run for them once the cause is fixed: "
                f"--desktops "
                + ",".join(str(r.desktop) for r in run.failed))
    return run


def _log_desktop(log, prefix: str, one: DesktopRelations) -> None:
    """One desktop's headline plus every line it produced."""
    if one.status == "failed":
        log(f"{prefix} D{one.desktop:02d}: FAILED, {one.error}")
        return
    log(f"{prefix} D{one.desktop:02d}: {len(one.implied)} implied, {len(one.fills)} "
        f"fills on {one.changed} instances, {len(one.unresolved)} unresolved "
        f"({one.unresolved_of(AMBIGUOUS)} {AMBIGUOUS}, "
        f"{one.unresolved_of(NO_CANDIDATE)} {NO_CANDIDATE})")
    for text in one.implied:
        log(f"{prefix}   {text}")
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
        if not args.dry_run and not _safety_backup(
                paths, db, "infer-relations",
                "it rewrites the relational fields of every desktop"):
            return EXIT_ERROR  # the line is printed and nothing was written
        run = infer_relations_into_db(
            db, load_taxonomy(), _desktops(args), args.dry_run, args.force, log=print,
            add_implied=args.add_implied,
        )
        if run.refused:
            listed = ", ".join(f"D{r.desktop:02d}" for r in run.refused)
            print(f"[infer-relations] refused: {listed} carry verified frames. Re-run "
                  f"with --force to fill them anyway (their frozen rows are then "
                  f"re-checked), or select the other desktops with --desktops.")
        if run.failed or run.refused:
            return EXIT_ERROR
        return EXIT_OK


def _add_infer_relations(sub) -> None:
    p = sub.add_parser(
        "infer-relations",
        help="fill empty screw.fastens / parent / latch.of / socket_host in place",
    )
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be filled but write no data (the database "
                        "is still opened and schema-checked under the lock); no "
                        "backup is taken and a desktop with verified frames is "
                        "reported rather than refused")
    p.add_argument("--force", action="store_true",
                   help="also fill desktops that already carry verified frames, whose "
                        "frozen rows are then re-checked and may raise conflicts")
    p.add_argument("--add-implied", action="store_true",
                   help="first create the instances configs/taxonomy.yaml's "
                        "implied_when_referenced allows: a part the desktop clearly "
                        "has (the motherboard of D49/D62/D63/D64) that its log never "
                        "operates on. OFF here because this command exists to repair "
                        "a database in place; import-logs always does it")
    p.set_defaults(func=cmd_infer_relations)
