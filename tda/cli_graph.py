"""``constraints``: derive, store and check each desktop's constraint graph.

::

    python -m tda.cli constraints [--desktops 1-66] [--dry-run] [--validate]
                                  [--report PATH]

Spec 7 was implemented and never called. :func:`~tda.core.graph.propose_edges`,
:func:`~tda.core.graph.find_cycles` and
:func:`~tda.core.graph.validate_sequence` had no caller outside the tests, so
the only ``relation`` rows the database held came from the Label Studio import
and every VLM export wrote ``"graph_version": None``. This command is the
missing caller.

Per desktop, in one transaction:

1. its settled instances (Label Studio drafts excluded -- a draft carries no
   relations until S1 makes it real) and its actions are read;
2. :func:`~tda.core.graph.propose_edges` derives the spec 7.3 edges;
3. the derived set **replaces** the desktop's ``source="rule"`` rows, and
   touches nothing else: an edge a human wrote or the Label Studio import
   brought keeps its reason, its status and its provenance, even when the rules
   would have derived the same triple;
4. :func:`~tda.core.graph.find_cycles` checks the spec 7.4 acyclicity
   invariant, and ``--validate`` additionally replays the log against the graph;
5. a content hash of the resulting edge set is stamped in the desktop meta as
   ``graph_version``, so an export can say which graph it shipped.

Safety is the same as the other writing commands: the single-user lock, a
backup before anything is written, one desktop per transaction so a desktop that
raises is reported and skipped rather than half-stored. ``--dry-run`` writes no
database row and no report, unless ``--report`` names one explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from tda import pipeline as P
from tda.cli_common import (
    EXIT_ERROR,
    EXIT_OK,
    desktops as _desktops,
    safety_backup as _safety_backup,
    session as _session,
)
from tda.core.db import Db
from tda.core.graph import (
    HARD_TYPES,
    Edge,
    edge_digest,
    edges_from_db,
    edges_to_db,
    find_cycles,
    is_provisional,
    propose_edges,
    unresolved_fan_owners,
    validate_sequence,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy

__all__ = [
    "ANNOTATOR", "CONSTRAINTS_REPORT_NAME", "DesktopGraph", "GraphRun", "OP_KIND",
    "OP_VIEW", "RULE", "constraints_into_db", "constraints_report",
]

#: The provenance this command owns. Everything else is somebody's decision.
RULE = "rule"
#: ``op_log.kind`` of every row this command writes.
OP_KIND = "constraints"
#: ``op_log`` is scoped per ``(desktop, view)``; an edge belongs to no view.
OP_VIEW = "-"
#: ``op_log.annotator`` -- this is machinery, not a person.
ANNOTATOR = "cli:constraints"
#: Where the Markdown report goes when ``--report`` does not say.
CONSTRAINTS_REPORT_NAME = "constraints_report.md"

#: The hint :func:`~tda.core.graph.validate_sequence` ends a "nothing was in its
#: way" line with; the report pulls those out separately for the annotator.
MISSING_EDGE = "missing edge?"

Triple = tuple[str, str, str]


# --------------------------------------------------------------------------- #
# run records
# --------------------------------------------------------------------------- #
@dataclass
class DesktopGraph:
    """What ``constraints`` derived, stored and found for one desktop."""

    desktop: int
    status: str = "applied"  # applied | failed
    by_type: dict[str, int] = field(default_factory=dict)
    stored: int = 0  # rule edges written
    removed: int = 0  # rule edges the rules no longer propose
    protected: int = 0  # manual / Label Studio edges left alone
    protected_other: int = 0  # rows of a type that is not a hard constraint
    cycles: list[list[str]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    version: Optional[str] = None
    validated: bool = False
    error: str = ""

    @property
    def edges(self) -> int:
        return sum(self.by_type.values())

    @property
    def hints(self) -> list[str]:
        """Failed attempts nothing in the graph explains -- "missing edge?"."""
        return [text for text in self.violations if text.endswith(MISSING_EDGE)]

    @property
    def breaches(self) -> list[str]:
        """Successful actions that broke a hard constraint (spec 7.4)."""
        return [text for text in self.violations if not text.endswith(MISSING_EDGE)]


@dataclass
class GraphRun:
    """The whole run."""

    dry_run: bool = False
    validate: bool = False
    runs: list[DesktopGraph] = field(default_factory=list)

    @property
    def applied(self) -> list[DesktopGraph]:
        return [r for r in self.runs if r.status == "applied"]

    @property
    def failed(self) -> list[DesktopGraph]:
        return [r for r in self.runs if r.status == "failed"]

    @property
    def edges(self) -> int:
        return sum(r.edges for r in self.applied)

    @property
    def by_type(self) -> dict[str, int]:
        total: Counter = Counter()
        for r in self.applied:
            total.update(r.by_type)
        return dict(total)

    @property
    def cycles(self) -> int:
        return sum(len(r.cycles) for r in self.applied)

    @property
    def violations(self) -> int:
        return sum(len(r.violations) for r in self.applied)

    def tally(self) -> str:
        """The one summary line the command ends with."""
        types = ", ".join(f"{t} {self.by_type.get(t, 0)}" for t in HARD_TYPES
                          if self.by_type.get(t))
        return (f"{len(self.applied)} desktops, {self.edges} edges ({types or 'none'}), "
                f"{self.cycles} cycles, {self.violations} violations, "
                f"{len(self.failed)} failed")


# --------------------------------------------------------------------------- #
# one desktop
# --------------------------------------------------------------------------- #
def _triple(edge: Edge) -> Triple:
    return (edge.type, edge.target, edge.blocker)


def _settled(db: Db, desktop: int) -> dict:
    """The desktop's instances without the Label Studio drafts.

    A ``ls:*`` row is a *shape* somebody drew, not a part anybody has decided
    exists (spec 3.2). Deriving constraints from one would invent edges that
    disappear the moment S1 merges or deletes it.
    """
    return {k: rec for k, rec in db.instances(desktop).items() if not is_provisional(k)}


def _apply_one(db: Db, tax: Taxonomy, desktop: int, dry_run: bool,
               validate: bool) -> DesktopGraph:
    """Derive, store and check one desktop's graph inside a single transaction."""
    instances = _settled(db, desktop)
    proposed = [
        Edge(type=e.type, target=e.target, blocker=e.blocker, necessity=e.necessity,
             mode=e.mode, reason=e.reason, source=RULE, evidence_step=e.evidence_step,
             status=e.status)
        for e in propose_edges(instances, tax)
    ]
    existing = edges_from_db(db, desktop)
    keep = [e for e in existing if e.source != RULE]
    # Only the five hard-constraint types of spec 7.2 are the *constraint graph*.
    # The Label Studio import also files `partner_of` / `is_pre-request_of` rows
    # in the same table; they are annotations of another kind, they gate nothing,
    # and feeding them to `find_cycles` or `unmet` invents eight cycles and a
    # stack of violations out of rows nobody claimed were constraints. They are
    # left exactly where they are and counted apart.
    protected = {_triple(e): e for e in keep if e.type in HARD_TYPES}
    proposed_triples = {_triple(p) for p in proposed}
    stale = [_triple(e) for e in existing
             if e.source == RULE and _triple(e) not in proposed_triples]
    to_store = [e for e in proposed if _triple(e) not in protected]

    # What the desktop's constraint graph will be afterwards: the rules' edges
    # plus every hard edge a human or the import owns. Cycles, the replay and the
    # version are all asked of *that*, so a dry run and a real run answer the
    # same thing -- and a manual `blocked_by` that closes a loop with a rule edge
    # is caught.
    effective = to_store + list(protected.values())
    out = DesktopGraph(
        desktop=desktop,
        by_type=dict(Counter(e.type for e in effective)),
        stored=len(to_store),
        removed=len(stale),
        protected=len(protected),
        protected_other=len(keep) - len(protected),
        cycles=find_cycles(effective),
        unresolved=unresolved_fan_owners(instances),
        version=edge_digest(effective),
        validated=validate,
    )
    if validate:
        out.violations = validate_sequence(instances, effective, db.actions(desktop), tax)
    if dry_run:
        return out
    with db.transaction():
        for rel_type, target, blocker in stale:
            db.delete_relation(desktop, rel_type, target, blocker)
        edges_to_db(db, desktop, to_store)
        P.merge_desktop_meta(db, desktop, {
            "graph_version": out.version,
            "graph_edges": out.edges,
            "graph_cycles": len(out.cycles),
        })
        db.log_op(
            desktop, OP_VIEW, OP_KIND,
            {"stored": out.stored, "removed": out.removed, "version": out.version},
            {"removed": [list(t) for t in stale]}, ANNOTATOR,
        )
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def _selected(db: Db, desktops: Optional[set[int]], log) -> list[int]:
    """The desktops to work on, warning about any ``--desktops`` id the DB lacks."""
    known = db.desktop_ids()
    if desktops is None:
        return known
    for desktop in sorted(desktops - set(known)):
        if log:
            log(f"[constraints] D{desktop:02d} is not in the database; skipped")
    return [desktop for desktop in known if desktop in desktops]


def constraints_into_db(
    db: Db,
    tax: Taxonomy,
    desktops: Optional[set[int]] = None,
    dry_run: bool = False,
    validate: bool = False,
    log=None,
) -> GraphRun:
    """Derive and store every selected desktop's constraint graph.

    One desktop is one transaction, and a desktop that raises is reported and
    skipped rather than ending the run -- the other sixty-five are still worth
    having. Re-running is a no-op: the rules are pure, so the same instances
    derive the same edges, the same rows are upserted and the same
    ``graph_version`` comes out.
    """
    run = GraphRun(dry_run=dry_run, validate=validate)
    prefix = "[constraints]" + (" (dry run)" if dry_run else "")
    for desktop in _selected(db, desktops, log):
        try:
            one = _apply_one(db, tax, desktop, dry_run, validate)
        except Exception as exc:  # one bad desktop must not end the run
            one = DesktopGraph(desktop=desktop, status="failed",
                               error=f"{type(exc).__name__}: {exc}")
        run.runs.append(one)
        if log:
            _log_desktop(log, prefix, one)
    if log:
        log(f"{prefix} {run.tally()}")
        if run.failed:
            log(f"{prefix} {len(run.failed)} desktops failed; every other desktop was "
                f"stored. Re-run for them once the cause is fixed: --desktops "
                + ",".join(str(r.desktop) for r in run.failed))
    return run


def _log_desktop(log, prefix: str, one: DesktopGraph) -> None:
    """One desktop's headline, plus anything that needs a human."""
    if one.status == "failed":
        log(f"{prefix} D{one.desktop:02d}: FAILED, {one.error}")
        return
    other = f", {one.protected_other} non-constraint rows untouched" \
        if one.protected_other else ""
    log(f"{prefix} D{one.desktop:02d}: {one.edges} edges "
        f"({one.stored} rule, {one.protected} kept, {one.removed} dropped{other}), "
        f"{len(one.cycles)} cycles, {len(one.violations)} violations, "
        f"graph_version {one.version or '-'}")
    for cycle in one.cycles:
        log(f"{prefix}   CYCLE {' -> '.join(cycle)}")
    for text in one.unresolved:
        log(f"{prefix}   {text}")


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def constraints_report(run: GraphRun) -> str:
    """The Markdown report: one summary table, then one section per desktop."""
    applied = run.applied
    lines = [
        "# Constraint graph",
        "",
        "Generated by `python -m tda.cli constraints"
        + (" --validate" if run.validate else "")
        + (" --dry-run" if run.dry_run else "") + "`.",
        "",
        f"- desktops: {len(applied)} stored, {len(run.failed)} failed",
        f"- edges: {run.edges} ("
        + ", ".join(f"{t} {run.by_type.get(t, 0)}" for t in HARD_TYPES) + ")",
        f"- cycles: {run.cycles}",
        f"- violations: {run.violations}"
        + ("" if run.validate else " (not checked; pass --validate)"),
        "",
        "| desktop | edges | " + " | ".join(HARD_TYPES)
        + " | cycles | violations | graph_version |",
        "|---|---|" + "---|" * len(HARD_TYPES) + "---|---|---|",
    ]
    for r in run.runs:
        if r.status != "applied":
            lines.append(f"| D{r.desktop:02d} | FAILED | "
                         + "| " * len(HARD_TYPES) + "| | |")
            continue
        counts = " | ".join(str(r.by_type.get(t, 0)) for t in HARD_TYPES)
        lines.append(f"| D{r.desktop:02d} | {r.edges} | {counts} | {len(r.cycles)} "
                     f"| {len(r.violations)} | {r.version or '-'} |")
    lines.append("")
    for r in run.runs:
        lines.extend(_desktop_section(r))
    return "\n".join(lines)


def _desktop_section(r: DesktopGraph) -> list[str]:
    """One desktop's own heading and everything a human has to look at."""
    lines = [f"## D{r.desktop:02d}", ""]
    if r.status == "failed":
        return lines + [f"- FAILED: {r.error}", ""]
    lines.append("- edges: "
                 + ", ".join(f"{t} {r.by_type.get(t, 0)}" for t in HARD_TYPES))
    lines.append(f"- graph_version: `{r.version or '-'}`")
    lines.append(f"- rule edges written: {r.stored}, dropped: {r.removed}; "
                 f"manual / Label Studio hard edges kept: {r.protected}; "
                 f"rows of a non-constraint type left untouched: {r.protected_other}")
    lines.append("")
    lines.append("### cycles")
    lines.append("")
    lines.extend([f"- {' -> '.join(cycle)}" for cycle in r.cycles]
                 or ["- none (the graph is acyclic, as spec 7.4 requires)"])
    lines.append("")
    lines.append("### violations")
    lines.append("")
    if not r.validated:
        lines.append("- not checked; re-run with --validate")
    else:
        lines.extend([f"- {text}" for text in r.breaches]
                     or ["- none: every recorded action was legal when it happened"])
    lines.append("")
    lines.append("### failed attempts with no unmet constraint")
    lines.append("")
    if not r.validated:
        lines.append("- not checked; re-run with --validate")
    else:
        lines.extend([f"- {text}" for text in r.hints]
                     or ["- none"])
    lines.append("")
    if r.unresolved:
        lines.append("### unresolved references")
        lines.append("")
        lines.extend(f"- {text}" for text in r.unresolved)
        lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# command
# --------------------------------------------------------------------------- #
def cmd_constraints(args: argparse.Namespace) -> int:
    """Derive and store the constraint graph of every selected desktop."""
    with _session(args, lock=True) as (paths, db):
        if not args.dry_run and not _safety_backup(
                paths, db, "constraints",
                "it replaces every rule edge of the selected desktops"):
            return EXIT_ERROR  # the line is printed and nothing was written
        run = constraints_into_db(
            db, load_taxonomy(), _desktops(args), args.dry_run, args.validate, log=print,
        )
        # A dry run writes nothing by default, report included; --report is the
        # annotator asking for one anyway, which is the whole point of a rehearsal.
        report = args.report or (None if args.dry_run
                                 else P.cache_file(paths, CONSTRAINTS_REPORT_NAME))
        if report:
            with open(report, "w", encoding="utf-8") as fh:
                fh.write(constraints_report(run))
            print(f"[constraints] wrote {report}")
        if run.failed or run.cycles or (run.validate and run.violations):
            return EXIT_ERROR
        return EXIT_OK


def _add_constraints(sub) -> None:
    p = sub.add_parser(
        "constraints",
        help="derive and store each desktop's spec-7 constraint graph",
    )
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66")
    p.add_argument("--dry-run", action="store_true",
                   help="derive and check but write nothing: no edge, no "
                        "graph_version, no backup and no report (the database is "
                        "still opened and schema-checked under the lock). Pass "
                        "--report to get the report anyway")
    p.add_argument("--validate", action="store_true",
                   help="also replay each desktop's log against its graph (spec "
                        "7.4): every action that broke a hard constraint, and "
                        "every failed attempt nothing in the graph explains, which "
                        "is the cue to add a blocked_by edge by hand. Off by "
                        "default because it reports on the *logs*, not on the "
                        "graph this command writes")
    p.add_argument("--report", default=None,
                   help="report Markdown (default <cache_dir>/"
                        + CONSTRAINTS_REPORT_NAME + ")")
    p.set_defaults(func=cmd_constraints)
