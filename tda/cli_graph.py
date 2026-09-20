"""``constraints``: derive, store and check each desktop's constraint graph.

::

    python -m tda.cli constraints [--desktops 1-66] [--dry-run] [--validate]
                                  [--report PATH]

Spec 7 was implemented and never called. :func:`~tda.core.graph.propose_edges`,
:func:`~tda.core.graph.find_deadlocks` and
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
4. :func:`~tda.core.graph.find_deadlocks` checks the spec 7.4 acyclicity
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
    find_deadlocks,
    graph_version,
    is_provisional,
    unresolved_fan_owners,
    validate_sequence,
)
from tda.core.graph_derive import Derivation, derive_edges, write_derivation
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
    """What ``constraints`` derived, stored and found for one desktop.

    ``by_type`` and :attr:`edges` count the **active** edges -- what the graph
    actually gates with. A rejected rule edge and an orphaned decision are rows
    the table still holds and are reported on their own lines, because a number
    that quietly includes them is not "the size of the graph".
    """

    desktop: int
    status: str = "applied"  # applied | failed
    by_type: dict[str, int] = field(default_factory=dict)
    stored: int = 0  # rule edges the rules propose and own
    removed: int = 0  # rule edges the rules no longer propose
    accepted: int = 0  # rule edges a human accepted (spec 7.3)
    rejected: int = 0  # ... and rejected
    orphans: list[str] = field(default_factory=list)  # decisions without a rule edge
    manual: int = 0  # edges a human wrote
    imported: int = 0  # hard edges from the Label Studio import
    protected_other: int = 0  # rows of a type that is not a hard constraint
    cycles: list[str] = field(default_factory=list)  # deadlock labels
    violations: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    version: Optional[str] = None
    validated: bool = False
    error: str = ""

    @property
    def edges(self) -> int:
        return sum(self.by_type.values())

    @property
    def decided(self) -> int:
        """Rule edges a human has decided about, orphans included."""
        return self.accepted + self.rejected + len(self.orphans)

    @property
    def hints(self) -> list[str]:
        """Failed attempts nothing in the graph explains -- "missing edge?".

        The annotator's move is the opposite of :attr:`breaches`: something
        really was in the way and the graph does not know it, so the answer is a
        manual ``blocked_by`` edge rather than a correction to the log.
        """
        return [text for text in self.violations if text.endswith(MISSING_EDGE)]

    @property
    def breaches(self) -> list[str]:
        """Successful actions that broke a hard constraint (spec 7.4).

        Almost always a gap in the log rather than a physical impossibility: a
        step that was done but never written down, so the replay reaches the
        action with a blocker still in its original state.
        """
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
def _settled(db: Db, desktop: int) -> dict:
    """The desktop's instances without the Label Studio drafts.

    A ``ls:*`` row is a *shape* somebody drew, not a part anybody has decided
    exists (spec 3.2). Deriving constraints from one would invent edges that
    disappear the moment S1 merges or deletes it.
    """
    return {k: rec for k, rec in db.instances(desktop).items() if not is_provisional(k)}


def _summary(desktop: int, derivation: Derivation, instances: dict, tax: Taxonomy,
             validate: bool) -> DesktopGraph:
    """The run record of one derivation, before anything is written."""
    counts = derivation.counts()
    return DesktopGraph(
        desktop=desktop,
        by_type=derivation.by_type,
        stored=counts["rule"],
        removed=counts["stale"],
        accepted=counts["decided"] - counts["rejected"] - counts["orphaned"],
        rejected=counts["rejected"],
        orphans=[f"{e.status} {e.label()}" for e in derivation.orphans],
        manual=counts["manual"],
        imported=counts["imported"],
        protected_other=counts["other"],
        cycles=[d.label() for d in find_deadlocks(derivation.edges, instances, tax)],
        version=edge_digest(derivation.edges),
        validated=validate,
    )


def _apply_one(db: Db, tax: Taxonomy, desktop: int, dry_run: bool,
               validate: bool) -> DesktopGraph:
    """Derive, store and check one desktop's graph inside a single transaction.

    The derivation itself is :func:`tda.core.graph_derive.derive_edges`, which
    the S1 ``Apply`` runs too -- there is one set of rules and one answer, so
    the Relations tab cannot describe a graph the command line would not write.
    Only the five hard types of spec 7.2 are the constraint graph; the Label
    Studio ``partner_of`` rows share the table, gate nothing, and are left
    exactly where they are and counted apart.
    """
    instances = _settled(db, desktop)
    stored = edges_from_db(db, desktop)
    derivation = derive_edges(instances, stored, tax)
    out = _summary(desktop, derivation, instances, tax, validate)
    out.unresolved = unresolved_fan_owners(instances)
    if validate:
        out.violations = validate_sequence(
            instances, derivation.edges, db.actions(desktop), tax)
    if dry_run:
        return out
    with db.transaction():
        write_derivation(db, desktop, derivation, stored)
        # read the accessor back rather than stamping the digest computed above:
        # the meta and `graph.graph_version` are then the same answer by
        # construction, which is what the exports quote
        out.version = graph_version(db, desktop)
        P.merge_desktop_meta(db, desktop, {
            "graph_version": out.version,
            "graph_edges": out.edges,
            "graph_cycles": len(out.cycles),
        })
        db.log_op(
            desktop, OP_VIEW, OP_KIND,
            {"stored": out.stored, "removed": out.removed,
             "orphaned": len(out.orphans), "version": out.version},
            {"removed": [list(t) for t in derivation.stale]}, ANNOTATOR,
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
    # the breakdown is of the ROWS the table holds and does not add up to the
    # active total (a rejected or orphaned row gates nothing), so it is fenced
    # off rather than left looking like a sum
    log(f"{prefix} D{one.desktop:02d}: {one.edges} active edges; rows: "
        f"{one.stored} rule + {one.decided} decided "
        f"({one.rejected} rejected, {len(one.orphans)} orphaned) + "
        f"{one.manual} manual + {one.imported} imported{other}; "
        f"{one.removed} rule rows dropped; "
        f"{len(one.cycles)} deadlocks, {len(one.violations)} violations, "
        f"graph_version {one.version or '-'}")
    for cycle in one.cycles:
        log(f"{prefix}   DEADLOCK {cycle}")
    for text in one.orphans:
        log(f"{prefix}   ORPHANED DECISION {text} (its rule edge is no longer derived)")
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
        f"- edges: {run.edges} active ("
        + ", ".join(f"{t} {run.by_type.get(t, 0)}" for t in HARD_TYPES) + ")",
        f"- decided by hand: {sum(r.accepted for r in applied)} accepted, "
        f"{sum(r.rejected for r in applied)} rejected, "
        f"{sum(len(r.orphans) for r in applied)} orphaned; "
        f"{sum(r.manual for r in applied)} manual edges",
        f"- deadlocks (spec 7.4): {run.cycles}",
        f"- violations: {run.violations}"
        + (f" ({sum(len(r.breaches) for r in applied)} likely log gaps, "
           f"{sum(len(r.hints) for r in applied)} failed attempts wanting a "
           f"manual blocked_by)" if run.validate
           else " (not checked; pass --validate)"),
        "",
        "| desktop | active edges | " + " | ".join(HARD_TYPES)
        + " | deadlocks | violations | graph_version |",
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
    lines.append("- active edges: "
                 + ", ".join(f"{t} {r.by_type.get(t, 0)}" for t in HARD_TYPES))
    lines.append(f"- graph_version: `{r.version or '-'}`")
    lines.append(f"- rule edges: {r.stored} derived, {r.removed} dropped")
    lines.append(f"- decided by hand: {r.accepted} accepted, {r.rejected} rejected, "
                 f"{len(r.orphans)} orphaned (the rule edge is no longer derived)")
    lines.append(f"- manual edges: {r.manual}; Label Studio hard edges: {r.imported}; "
                 f"rows of a non-constraint type left untouched: {r.protected_other}")
    lines.append("")
    if r.orphans:
        lines.append("### decisions whose rule edge is no longer derived")
        lines.append("")
        lines.append("A rule edge these decisions were about is not derived any more, "
                     "so they gate nothing and are counted apart. In the Relations tab: "
                     "keep one as a manual edge, or clear it.")
        lines.append("")
        lines.extend(f"- {text}" for text in r.orphans)
        lines.append("")
    lines.append("### deadlocks (spec 7.4)")
    lines.append("")
    lines.extend([f"- {cycle}" for cycle in r.cycles]
                 or ["- none (no loop of actions waits on itself, as spec 7.4 "
                     "requires; the graph is acyclic in the sense that matters)"])
    lines.append("")
    lines.append("### likely log gaps")
    lines.append("")
    if not r.validated:
        lines.append("- not checked; re-run with --validate")
    else:
        lines.append("A recorded action that broke a hard constraint. The graph "
                     "says the blocker was still in its way, so a step was most "
                     "likely done and never written down -- S1 work.")
        lines.append("")
        lines.extend([f"- {text}" for text in r.breaches]
                     or ["- none: every recorded action was legal when it happened"])
    lines.append("")
    lines.append("### failed attempt without an unmet constraint (add a manual "
                 "blocked_by?)")
    lines.append("")
    if not r.validated:
        lines.append("- not checked; re-run with --validate")
    else:
        lines.append("The annotator tried and could not. Something was in the way "
                     "and the graph does not know what, so this wants a manual "
                     "`blocked_by` edge -- S6 work.")
        lines.append("")
        lines.extend([f"- {text}" for text in r.hints] or ["- none"])
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
