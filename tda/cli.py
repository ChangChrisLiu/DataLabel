"""Command line of the Teardown Annotator: the data pipeline that seeds the DB.

::

    python -m tda.cli build-index [--desktops 1-66]   # F: -> cache/index.json
    python -m tda.cli load-index  [--desktops ...]    # index.json -> frames, pose segments
    python -m tda.cli import-logs [--desktops ...]    # Drive sheets -> steps/actions/...
    python -m tda.cli import-ls   [--export PATH]     # Label Studio export -> drafts
    python -m tda.cli infer-relations [--dry-run]     # fill empty relational fields
    python -m tda.cli backup                          # SQLite backup API -> backup_dir
    python -m tda.cli status      [--desktop N]       # what the database holds

Those four import commands must run in that order: ``load-index`` needs the
index file, ``import-logs`` needs the index for its step durations, and
``import-ls`` needs the step table its notes are merged into (it refuses with
exit code 2 otherwise).

Every command takes ``--paths`` (default ``configs/paths.yaml``) and ``--db``,
which overrides ``paths.yaml``'s ``db_path``; both go **before** the
subcommand. The work itself lives in :mod:`tda.pipeline` and
:mod:`tda.pipeline_logs`.

Safety: the three writing commands take the single-user lock of spec 3.5 and
refuse with exit code 3 while somebody else holds it, and the two destructive
options (``import-logs --force``, ``import-ls --purge-all``) back the database
up into ``backup_dir`` before they touch anything.

Adding a command (the GUI ``app``, ``check``, ``build-cache``, ``export-*``):
write a ``_add_<name>`` registrar that calls ``sub.add_parser(...)`` and sets
``func=<handler>``, then list it in :data:`SUBCOMMANDS`. A handler takes the
parsed arguments and returns the process exit code; ``session(args, lock=True)``
of :mod:`tda.cli_common` hands it ``(paths, db)`` with the lock held. Everything
shared lives there and **nothing imports this module**: run as ``python -m
tda.cli`` this file is ``__main__``, so an import of ``tda.cli`` would load it a
second time and give the importer a different ``Locked`` class than the one
:func:`main` catches.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Callable, Optional

from tda import pipeline as P
from tda import pipeline_logs as L
from tda.cli_app import SUBCOMMANDS as _APP_SUBCOMMANDS
from tda.cli_common import (
    EXIT_ERROR,
    EXIT_LOCKED,
    EXIT_OK,
    EXIT_ORDER,
    ConfigError,
    Locked,
    desktops as _desktops,
    load_paths,
    safety_backup as _safety_backup,
    session as _session,
)
from tda.cli_relations import _add_infer_relations
from tda.core.db import Db
from tda.core.index import build_index, load_index, save_index
from tda.core.index_report import write_report
from tda.core.model import VIEWS
from tda.core.taxonomy import load_taxonomy

__all__ = ["main"]


# --------------------------------------------------------------------------- #
# build-index
# --------------------------------------------------------------------------- #
def cmd_build_index(args: argparse.Namespace) -> int:
    """Scan the source trees on F: and write ``cache/index.json`` + its report."""
    paths = load_paths(args.paths)
    wanted = _desktops(args) or set(P.ALL_DESKTOPS)
    roots = {k: P.require(paths, k) for k in ("oak_root", "scanner_root", "rs_root")}
    out = args.out or P.index_path(paths)
    report = args.report or P.cache_file(paths, P.INDEX_REPORT_NAME)

    index = build_index(sorted(wanted), roots, args.fixes, progress=True)
    save_index(index, out)
    write_report(index, report)
    print(f"[build-index] {len(index)} desktops, "
          f"{sum(d.n_steps for d in index.values())} steps, "
          f"{sum(len(d.frames) for d in index.values())} frames, "
          f"{sum(len(d.missing) for d in index.values())} missing")
    print(f"[build-index] wrote {out}\n[build-index] wrote {report}")
    return EXIT_OK


def _add_build_index(sub) -> None:
    p = sub.add_parser("build-index", help="scan F: and write cache/index.json")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66 or 1-3,13")
    p.add_argument("--out", default=None, help="index JSON (default <cache_dir>/index.json)")
    p.add_argument("--report", default=None, help="index report Markdown")
    p.add_argument("--fixes", default="configs/index_fixes.yaml")
    p.set_defaults(func=cmd_build_index)


# --------------------------------------------------------------------------- #
# load-index
# --------------------------------------------------------------------------- #
def cmd_load_index(args: argparse.Namespace) -> int:
    """Write the frames and pose segments of ``index.json`` into the database.

    Destructive, quietly: a re-built index can move a ``reorient`` step, which
    re-cuts the pose segments, and a segment whose reference frame leaves it
    loses the corners, homography and ROI a human clicked (spec 2.4). So it
    takes the same safety copy as the other three.
    """
    with _session(args, lock=True) as (paths, db):
        target = args.index or P.index_path(paths)
        try:
            index = load_index(target)
        except OSError as exc:
            print(f"[load-index] cannot read {target}: {exc}; run build-index first")
            return EXIT_ERROR
        if not _safety_backup(paths, db, "load-index",
                              "it re-cuts the pose segments"):
            return EXIT_ERROR
        counts = P.load_index_into_db(db, index, _desktops(args), log=print)
        failed = counts.get("failed") or []
        print(f"[load-index] {counts['desktops']} desktops, {counts['frames']} frames, "
              f"{counts['missing']} missing, {counts['segments']} pose segments, "
              f"{len(failed)} failed")
        if failed:
            for text in failed:
                print(f"[load-index] FAILED {text}")
            print(f"[load-index] {len(failed)} desktops failed; every other desktop "
                  f"was loaded. Re-run build-index for them, then load-index again.")
            return EXIT_ERROR
        return EXIT_OK


def _add_load_index(sub) -> None:
    p = sub.add_parser("load-index", help="index.json -> frame rows and pose segments")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66")
    p.add_argument("--index", default=None, help="index JSON (default <cache_dir>/index.json)")
    p.set_defaults(func=cmd_load_index)


# --------------------------------------------------------------------------- #
# import-logs
# --------------------------------------------------------------------------- #
def logs_report(run: L.LogsRun, expected: Optional[dict[int, int]] = None) -> str:
    """Render the ``import-logs`` issue report (Markdown).

    One summary table over every desktop the run touched -- with the sheet's own
    ``n_logged_steps`` beside the imported step count -- then one section per
    desktop listing what the importer could not decide on its own (stage S1).
    """
    expected = expected or {}
    imported = run.imported
    lines = [
        "# Drive log import into the database",
        "",
        f"Generated by `python -m tda.cli import-logs --dir {run.directory}`.",
        "",
        f"- desktops imported: {len(imported)}",
        f"- desktops skipped (already had steps): {len(run.skipped)}",
        f"- desktops refused (verified frames, no --force-verified): {len(run.refused)}",
        f"- desktops that failed to import: {len(run.failed)}",
        f"- steps: {sum(r.steps for r in imported)}",
        f"- actions: {sum(r.actions for r in imported)}",
        f"- instances: {sum(r.instances for r in imported)}",
        f"- state events: {sum(r.events for r in imported)}",
        f"- step durations from the index: {sum(r.durations for r in imported)}",
        f"- {L.INFERRED_HEADING}: {sum(len(r.fills) for r in imported)}",
        f"- issues: {sum(len(r.issues) for r in imported)}",
        "",
        "| desktop | brand | steps | n_logged_steps | match | actions | instances "
        "| events | durations | issues |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in run.runs:
        if r.status != "imported":
            lines.append(f"| D{r.desktop:02d} | | | | {r.status} | | | | | |")
            continue
        want = expected.get(r.desktop)
        mark = "-" if want is None else ("ok" if want == r.steps else "MISMATCH")
        lines.append(
            f"| D{r.desktop:02d} | {r.brand} | {r.steps} | {'' if want is None else want} "
            f"| {mark} | {r.actions} | {r.instances} | {r.events} | {r.durations} "
            f"| {len(r.issues)} |"
        )
    lines.append("")
    for r in run.failed + run.refused:
        lines.append(f"## D{r.desktop:02d} - {r.status.upper()}")
        lines.append("")
        lines.extend(f"- {text}" for text in r.issues)
        lines.append("")
    for r in imported:
        lines.append(f"## D{r.desktop:02d} - {r.brand}")
        lines.append("")
        if r.issues:
            lines.extend(f"- {text}" for text in r.issues)
        else:
            lines.append("- no issues")
        lines.append("")
        lines.extend(L.inferred_section(r))
    return "\n".join(lines)


def cmd_import_logs(args: argparse.Namespace) -> int:
    """Import the Drive sheets into steps, actions, instances and state events."""
    if args.force_verified and not args.force:
        print("[import-logs] --force-verified only means something with --force: "
              "without --force a desktop that already has steps is skipped before "
              "its verified frames are ever looked at.")
        return EXIT_ERROR
    with _session(args, lock=True) as (paths, db):
        directory = args.dir or P.drive_dir(paths)
        index = P.read_index(paths, args.index)
        if not index:
            print("[import-logs] no index.json; step durations will stay unset")
        if args.force and not _safety_backup(
                paths, db, "import-logs", "--force rewrites the step tables"):
            return EXIT_ERROR
        run = L.import_logs_into_db(
            db, directory, load_taxonomy(), index, _desktops(args), args.force,
            log=print, force_verified=args.force_verified,
        )
        imported = run.imported
        print(f"[import-logs] {len(imported)} desktops imported, {len(run.skipped)} skipped, "
              f"{len(run.refused)} refused, {len(run.failed)} failed; "
              f"{sum(r.steps for r in imported)} steps, "
              f"{sum(r.actions for r in imported)} actions, "
              f"{sum(r.instances for r in imported)} instances, "
              f"{sum(r.events for r in imported)} events, "
              f"{sum(len(r.issues) for r in imported)} issues")
        for r in run.dropped_ls_notes:
            print(f"[import-logs] {L.dropped_notes_line(r.desktop, r.ls_notes_dropped)}")
        carried = run.with_ls_notes
        if carried:
            listed = ", ".join(f"D{d:02d}" for d in carried)
            print(f"[import-logs] {listed} had Label Studio notes; "
                  f"run 'python -m tda.cli import-ls' to rebuild them from the export "
                  f"if anything looks wrong")
        if run.refused:
            listed = ", ".join(f"D{r.desktop:02d}" for r in run.refused)
            print(f"[import-logs] refused: {listed} carry verified frames, which were "
                  f"compiled from the step table this would replace. Re-run with "
                  f"--force-verified to re-import them anyway, or select the other "
                  f"desktops with --desktops.")
        # A run that imported nothing has nothing to report, and overwriting the
        # file would throw away the issue list of the run that did the work.
        if not imported and not run.failed:
            print("[import-logs] nothing imported, kept the previous report")
            return EXIT_ERROR if run.refused else EXIT_OK
        report = args.report or P.cache_file(paths, P.LOGS_REPORT_NAME)
        with open(report, "w", encoding="utf-8") as fh:
            fh.write(logs_report(run, L.expected_steps(directory)))
        run.report_path = report
        print(f"[import-logs] wrote {report}")
        return EXIT_ERROR if (run.failed or run.refused) else EXIT_OK


def _add_import_logs(sub) -> None:
    p = sub.add_parser("import-logs", help="Drive sheets -> steps, actions, instances, events")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66")
    p.add_argument("--dir", default=None, help="log folder (default <raw_logs_dir>/drive)")
    p.add_argument("--index", default=None, help="index JSON used for the step durations")
    p.add_argument("--report", default=None, help="issue report Markdown")
    p.add_argument("--force", action="store_true",
                   help="re-import desktops that already have steps: the database is "
                        "backed up first and the Label Studio notes are kept where the "
                        "step number and name still match, but every other manual edit "
                        "to the step table is lost")
    p.add_argument("--force-verified", action="store_true",
                   help="with --force, also re-import desktops that carry verified "
                        "frames. Those frames were compiled from the step table this "
                        "replaces, so they are refused without it")
    p.set_defaults(func=cmd_import_logs)


# --------------------------------------------------------------------------- #
# import-ls
# --------------------------------------------------------------------------- #
def cmd_import_ls(args: argparse.Namespace) -> int:
    """Import the Label Studio export as draft shapes (after the logs are in)."""
    from tda.core.ls_export import iter_tasks, load_export
    from tda.core.ls_import import import_ls_export

    with _session(args, lock=True) as (paths, db):
        export = args.export or P.ls_export_path(paths)
        try:
            covered = {frame.desktop for _p, _t, frame in iter_tasks(load_export(export))}
        except OSError as exc:
            print(f"[import-ls] cannot read {export}: {exc}")
            return EXIT_ERROR
        blocked = P.desktops_without_steps(db, covered)
        if blocked and not args.allow_missing_steps:
            listed = ", ".join(f"D{d:02d}" for d in blocked)
            print(f"[import-ls] refused: no steps yet for {listed}. The pipeline order is "
                  f"build-index -> load-index -> import-logs -> import-ls; run import-logs "
                  f"first, or pass --allow-missing-steps.")
            return EXIT_ORDER
        if args.purge_all and not _safety_backup(
                paths, db, "import-ls",
                "--purge-all drops every desktop's Label Studio rows"):
            return EXIT_ERROR
        summary = import_ls_export(
            export, db, load_taxonomy(), P.read_index(paths, args.index),
            progress=args.progress, purge_all=args.purge_all,
        )
        out = args.summary or P.cache_file(paths, P.LS_SUMMARY_NAME)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, ensure_ascii=False, sort_keys=True)
        print(f"[import-ls] {summary['frames']} frames, {summary['keyframes']} keyframes, "
              f"{summary['instance_rows']} provisional instances, "
              f"{summary['relation_edges']} relations, "
              f"{summary['steps_with_notes']} steps with notes")
        print(f"[import-ls] wrote {out}")
        return EXIT_OK


def _add_import_ls(sub) -> None:
    p = sub.add_parser("import-ls", help="Label Studio export -> draft shapes")
    p.add_argument("--export", default=None,
                   help="export JSON (default <raw_logs_dir>/labelstudio/...)")
    p.add_argument("--index", default=None, help="index JSON (default <cache_dir>/index.json)")
    p.add_argument("--summary", default=None, help="summary JSON written by the run")
    p.add_argument("--purge-all", action="store_true",
                   help="clear every desktop's Label Studio rows first, not just the "
                        "desktops this export covers")
    p.add_argument("--allow-missing-steps", action="store_true",
                   help="import even though some desktops have no step table yet")
    p.add_argument("--progress", action="store_true", help="print each project as it is read")
    p.set_defaults(func=cmd_import_ls)


# --------------------------------------------------------------------------- #
# backup
# --------------------------------------------------------------------------- #
def cmd_backup(args: argparse.Namespace) -> int:
    """Copy the live database into ``backup_dir`` with the SQLite backup API.

    A bad ``--dest`` is a usage error and goes through :func:`main`; a copy that
    could not be made or could not be verified is reported like every other
    backup failure, without a traceback.
    """
    with _session(args) as (paths, db):
        dest = P.backup_dest(paths, args.dest)
        try:
            out = db.backup(dest, P.backup_keep(paths, prune=not args.no_prune))
        except (OSError, sqlite3.Error) as exc:
            print(f"[backup] backup failed: {exc}")
            return EXIT_ERROR
        print(f"[backup] wrote {out}")
        return EXIT_OK


def _add_backup(sub) -> None:
    p = sub.add_parser("backup", help="back the database up into backup_dir")
    p.add_argument("--dest", default=None,
                   help="a folder inside paths.yaml's backup_dir (the default). It "
                        "narrows that setting and never replaces it: without a "
                        "backup_dir in paths.yaml there is nowhere a backup may go, "
                        "and any --dest outside it is refused. A differently cased "
                        "or subst-mapped spelling of the same folder is accepted")
    p.add_argument("--no-prune", action="store_true",
                   help="keep every older backup. Normally the oldest copies "
                        "beyond paths.yaml's backup_keep are removed once the new "
                        "one is verified, except the newest of each of the last 14 "
                        "days; no file this tool did not write is ever touched")
    p.set_defaults(func=cmd_backup)


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
VIEW_CELL_WIDTH = 19  # "frames/miss  kf/ver"


def _view_cell(counts: dict) -> str:
    """One view's ``frames/missing  keyframes/verified`` cell."""
    return (f"{counts['frames']:>4}/{counts['missing']:<3}"
            f"{counts['keyframes']:>6}/{counts['verified']:<4}")


def format_status(rows: list[dict]) -> str:
    """The whole-database table: one line per desktop, four columns per view."""
    head = f"{'desktop':<8}{'steps':>6}{'inst':>6} | " + " | ".join(
        f"{view:^{VIEW_CELL_WIDTH}}" for view in VIEWS
    )
    lines = [head, f"{'':<20} | " + " | ".join(
        f"{'frames/miss  kf/ver':^{VIEW_CELL_WIDTH}}" for _ in VIEWS
    ), "-" * len(head)]
    for row in rows:
        lines.append(
            f"D{row['desktop']:02d}{'':<5}{row['steps']:>6}{row['instances']:>6} | "
            + " | ".join(_view_cell(row["views"][view]) for view in VIEWS)
        )
    pending = sum(sum(r["views"][view].get("rechecks", 0) for view in VIEWS) for r in rows)
    total = {view: {k: sum(r["views"][view][k] for r in rows)
                    for k in ("frames", "missing", "keyframes", "verified")} for view in VIEWS}
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<8}{sum(r['steps'] for r in rows):>6}"
        f"{sum(r['instances'] for r in rows):>6} | "
        + " | ".join(_view_cell(total[view]) for view in VIEWS)
    )
    if pending:
        # status is read-only, so it reports the backlog rather than working it
        # off: until it is zero the truth table of those views is not one to
        # export (spec 3.4). `cli check` holds the lock and drains them.
        lines.append(f"re-checks pending: {pending}")
    return "\n".join(lines)


def format_desktop(db: Db, row: dict) -> str:
    """The detailed single-desktop report of ``status --desktop N``."""
    types = ", ".join(f"{k} {n}" for k, n in db.step_type_counts(row["desktop"]).items())
    lines = [
        f"Desktop {row['desktop']}  {row['brand']}",
        f"  steps {row['steps']} ({types or 'none'}), actions {row['actions']}, "
        f"instances {row['instances']}, state events {row['events']}",
        f"  {'view':<6}{'frames':>8}{'missing':>9}{'keyframes':>11}{'verified':>10}",
    ]
    for view in VIEWS:
        counts = row["views"][view]
        lines.append(
            f"  {view:<6}{counts['frames']:>8}{counts['missing']:>9}"
            f"{counts['keyframes']:>11}{counts['verified']:>10}"
        )
    for view in VIEWS:
        waiting = row["views"][view].get("rechecks", 0)
        if waiting:
            lines.append(f"  re-checks pending: {view} {waiting}")
    for view in VIEWS:
        rendered = ", ".join(
            f"#{s['seg']} [{s['start_step']}-{s['end_step']}] ref {s['ref_step']}"
            f"{' corners' if s['corners'] else ''}"
            for s in db.pose_segments(row["desktop"], view)
        )
        lines.append(f"  pose {view:<5} {rendered or 'none'}")
    issues = (db.get_desktop(row["desktop"]) or {}).get("pose_issues") or []
    lines.extend(f"  ! {text}" for text in issues)
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    """Print what the database holds, for one desktop or for all of them."""
    with _session(args) as (_paths, db):
        wanted = P.parse_desktops(str(args.desktop)) if args.desktop else _desktops(args)
        rows = P.status_rows(db, wanted)
        if not rows:
            subject = f"desktop {args.desktop} is not in the database" if args.desktop \
                else "the database is empty"
            print(f"[status] {subject}; run load-index first")
            return EXIT_OK
        if args.desktop:
            print(format_desktop(db, rows[0]))
        else:
            print(format_status(rows))
        return EXIT_OK


def _add_status(sub) -> None:
    p = sub.add_parser("status", help="print per-desktop counters")
    p.add_argument("--desktop", type=int, default=None, help="detail one desktop")
    p.add_argument("--desktops", default=None, help="limit the table, e.g. 1-20")
    p.set_defaults(func=cmd_status)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
#: Registrars, in the order the commands appear in ``--help``: first the four
#: pipeline imports and the two database utilities, then the commands of
#: :mod:`tda.cli_app` that use what they produced (``app``, ``check``,
#: ``build-cache``, ``export-coco``, ``export-vlm``).
SUBCOMMANDS: tuple[Callable[[argparse._SubParsersAction], None], ...] = (
    _add_build_index,
    _add_load_index,
    _add_import_logs,
    _add_import_ls,
    _add_infer_relations,
    _add_backup,
    _add_status,
    *_APP_SUBCOMMANDS,
)


def build_parser() -> argparse.ArgumentParser:
    """The whole command line; ``--paths``/``--db`` are global and come first."""
    ap = argparse.ArgumentParser(
        prog="python -m tda.cli",
        description="Teardown Annotator data pipeline.",
        epilog="pipeline order: build-index -> load-index -> import-logs -> import-ls",
    )
    ap.add_argument("--paths", default=P.DEFAULT_PATHS_PATH, help="paths.yaml to use")
    ap.add_argument("--db", default=None, help="database file (overrides paths.yaml)")
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")
    for register in SUBCOMMANDS:
        register(sub)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    """Run one subcommand; returns the process exit code.

    ``0`` ok, ``1`` a configuration or input error, ``2`` the pipeline order was
    not respected, ``3`` somebody else holds the single-user lock.

    The three exceptions caught here are the ones an annotator causes by typing:
    a database somebody else has open, an unusable ``paths.yaml``, a ``--desktops``
    range that reads backwards. Each is one line and no traceback -- there is
    nothing in a traceback they could act on. Everything else is a bug and keeps
    its traceback, which is the only place it can be read.
    """
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Locked as exc:
        print(f"[{args.command}] refused: {exc}. Close the annotator (or wait for the "
              f"lock to expire) and try again.")
        return EXIT_LOCKED
    except ConfigError as exc:  # missing, unreadable or invalid paths.yaml
        print(f"[{args.command}] {exc}")
        return EXIT_ERROR
    except ValueError as exc:  # bad --desktops spec, bad paths.yaml, bad --dest
        print(f"[{args.command}] {exc}")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
