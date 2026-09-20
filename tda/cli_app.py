"""The commands that come after the data pipeline: the app, checks and exports.

:mod:`tda.cli` holds the four import commands that seed the database; this
module holds the five that use it -- open the annotator, recompile and count the
problems of one view, build the local image cache, and write the two rehearsal
exports.  They are registered by appending :data:`SUBCOMMANDS` to the parser
there, so ``python -m tda.cli --help`` lists all of them together.

Everything that writes takes the single-user lock through
:func:`tda.cli_common.session` (exit code 3 while somebody else holds it),
including ``check``: recompiling a view writes the ``compiled_mask`` rows it
finds stale.  That import is at module level and points at
:mod:`tda.cli_common`, never at :mod:`tda.cli`: importing the latter from here
loads it a second time under ``python -m tda.cli`` and hands this module a
``Locked`` class that ``main()`` does not catch.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

from tda import pipeline as P
from tda.cli_common import EXIT_ERROR, EXIT_OK, load_paths, session
from tda.core.model import VIEWS
from tda.core.taxonomy import load_taxonomy

__all__ = ["SUBCOMMANDS", "cmd_app", "cmd_build_cache", "cmd_check",
           "cmd_export_coco", "cmd_export_vlm"]


def _desktop_list(spec: str | None) -> list[int]:
    """``--desktops 1-3,13`` -> ``[1, 2, 3, 13]``; nothing given -> ``[]``."""
    wanted = P.parse_desktops(spec) if spec else None
    return sorted(wanted) if wanted else []


def _view_list(spec: str) -> list[str]:
    """``"scan,oak1"`` -> ``["scan", "oak1"]``, in order and without repeats.

    ``ValueError`` for a view that does not exist, which :func:`tda.cli.main`
    turns into one line and exit 1. That replaces argparse's ``choices``, which
    cannot describe a comma-separated list and answered "invalid choice:
    'scan,oak1'" for a spelling this now accepts.
    """
    known = ", ".join(VIEWS)
    wanted = [v.strip() for v in str(spec).split(",") if v.strip()]
    if not wanted:
        raise ValueError(f"no view given; the views are {known}")
    unknown = [v for v in wanted if v not in VIEWS]
    if unknown:
        raise ValueError(f"not a view: {', '.join(unknown)}; the views are {known}")
    out: list[str] = []
    for view in wanted:
        if view not in out:
            out.append(view)
    return out


def _add_view_flags(p, default: str = "scan", plural_first: bool = False) -> None:
    """``--view`` and ``--views``: one option, both spellings, one or many views.

    The exports spelled it ``--view`` and ``build-cache`` spelled it ``--views``,
    and each rejected the other -- a coin toss the annotator loses half the time
    in the middle of a release. They are the same option now, and it takes a
    comma-separated list wherever more than one view makes sense.
    """
    names = ("--views", "--view") if plural_first else ("--view", "--views")
    p.add_argument(*names, dest="views", default=default, metavar="VIEW[,VIEW...]",
                   help=f"one view or a comma-separated list ({', '.join(VIEWS)}); "
                        f"--view and --views are the same option")


def _count(stats: dict, key: str) -> int:
    """How many of something an exporter reported, whether it gave a list or a number."""
    value = stats.get(key, 0)
    if isinstance(value, int):
        return value
    try:
        return len(value)
    except TypeError:
        return 0


def _runs(desktops: Sequence[int]) -> list[tuple[int, int]]:
    """Contiguous ``(first, last)`` runs of a desktop list.

    ``tda.core.cache.main`` only speaks first/last, so ``--desktops 1-3,13``
    used to collapse into ``--first 1 --last 13`` and cache ten machines nobody
    asked for -- hours of copying from ``F:``.  One call per run instead.
    """
    runs: list[tuple[int, int]] = []
    for desktop in sorted(set(int(d) for d in desktops)):
        if runs and desktop == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], desktop)
        else:
            runs.append((desktop, desktop))
    return runs


def _pending_rechecks(truth, desktop: int, view: str) -> list[int]:
    """Steps the truth service still owes a re-check, when it tracks them."""
    found = getattr(truth, "pending_rechecks", None)
    if not callable(found):
        return []
    try:
        return [int(s) for s in found(desktop, view)]
    except Exception:  # noqa: BLE001 - a missing table is "nothing pending"
        return []


def _open_conflicts(truth, db, desktops: Sequence[int], view: str) -> int:
    """How many conflicts of these ``(desktop, view)`` are still open.

    An open conflict is a frozen row the recompile disagrees with: a human has
    to choose between the two, and until they do the view's truth table is not
    one to ship (spec 3.4).  ``check`` reported only the conflicts *this run*
    raised, so a queue somebody left behind yesterday read as a clean desktop.

    Worker F1 owns ``TruthService.open_conflicts``; it is preferred as soon as
    it exists, and until then the same question is answered straight from the
    table it will read anyway.
    """
    found = getattr(truth, "open_conflicts", None)
    if not callable(found):
        def found(desktop: int, view: str) -> list:
            return db.conflicts(desktop, view)
    return sum(_how_many(found(int(desktop), str(view))) for desktop in desktops)


def _how_many(answer) -> int:
    """A count, whether the service answered with the rows or with their number.

    ``TruthService.open_conflicts`` returns a list; a future one may well return
    an ``int``, and this adapter is the wrong place to care which.
    """
    if isinstance(answer, int):
        return answer
    try:
        return len(answer)
    except TypeError:
        return 0


def _frame_counts(db, desktops: Sequence[int], view: str) -> tuple[int, int]:
    """``(verified, not verified)`` frames of one view over the selection.

    An export that filters on ``verified`` and finds none writes an empty
    artefact, so it has to say what it looked at: "0 verified, 214 not verified"
    is a diagnosis, "0 images" is a mystery.
    """
    verified = total = 0
    for desktop in desktops:
        frames = [row for row in db.frames_for(int(desktop), str(view))
                  if not row.get("missing")]
        total += len(frames)
        verified += sum(1 for v, _step in db.verified_frames(int(desktop)) if v == view)
    return verified, max(total - verified, 0)


def _nothing_exported(command: str, view: str, out: str, only_verified: bool,
                      also: Sequence[Path] = ()) -> int:
    """Warn, remove the empty artefact **and its siblings**, and fail.

    An export that found nothing used to write the file anyway and exit 0, so a
    day of annotation came back as a COCO with zero images or a zero-byte JSONL
    and nothing at all to say why. The file goes too: an empty artefact left on
    disk is the one that gets shipped, or diffed against, or trained on -- and
    so do the files written beside it, or the next run finds an image manifest
    and a summary describing a JSONL nobody wrote.
    """
    subject = ("0 verified frames" if only_verified else "no exportable frames")
    hint = (" (use --no-only-verified to include unverified frames)"
            if only_verified else "")
    print(f"[{command}] WARNING: {subject} for the requested desktops/{view} "
          f"- nothing exported{hint}")
    for path in [Path(out), *also]:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # a locked or read-only file: say so, do not crash
            print(f"[{command}] could not remove the empty {path}: {exc}")
    return EXIT_ERROR


def _out_paths(out: str | None, views: Sequence[str], paths: dict, pattern: str,
               command: str) -> Optional[dict[str, str]]:
    """Where each view's file goes, or ``None`` when ``--out`` cannot cover them."""
    if out and len(views) > 1:
        print(f"[{command}] refused: --out names one file but {len(views)} views "
              f"were asked for ({', '.join(views)}). Drop --out to write one file "
              f"per view into cache_dir, or export one view at a time.")
        return None
    if out:
        return {views[0]: str(out)}
    return {view: P.cache_file(paths, pattern.format(view=view)) for view in views}


def _call_export(export, *args, allow_conflicts: bool, **kw):
    """Call an exporter, passing ``allow_conflicts`` only when it takes it.

    The keyword is F1's; an exporter that predates it must still run rather
    than die of an unexpected argument.  A ``TypeError`` raised *inside* the
    exporter says nothing about ``allow_conflicts`` and is left alone -- it is
    a bug, and swallowing it would hide it behind a second, argument-less call.
    """
    try:
        return export(*args, allow_conflicts=allow_conflicts, **kw)
    except TypeError as exc:
        if "allow_conflicts" not in str(exc):
            raise
        return export(*args, **kw)


def _prepare_truth(db, tax, desktops: Sequence[int], view: str,
                   refresh: bool = True) -> dict:
    """Bring the compiled truth of ``desktops`` up to date before it is read.

    Three commands read the truth table and all three were reading it raw:
    ``check`` recompiled but ignored the re-check queue the session fills in the
    background, and the two exports read whatever happened to be stored.  A
    frame whose inputs changed after it was frozen therefore exported its *old*
    geometry.  This is the one place that fixes it:

    * the pending re-checks are drained first (a frozen frame the session
      queued but never got to);
    * unless ``refresh`` is off, every step of the view is recompiled, so the
      automatic rows are current too.  ``refresh`` is **not**
      ``only_verified``: that one is the exporter's filter over what to write,
      and reading them as the same flag made ``--only-verified`` silently skip
      the recompile.

    One :class:`~tda.core.truth.TruthService` is built and returned in
    ``truth``: the exports take it as a keyword argument and refresh lazily
    through the same object, so a frame is never compiled twice per run.

    Returns ``{"truth", "desktops", "view", "steps", "updated", "conflicts",
    "problems", "pending"}``; ``pending`` is what is *still* queued afterwards,
    which is what the exports refuse on.
    """
    from tda.core.truth import TruthService

    truth = TruthService(db, tax)
    total = {"truth": truth, "desktops": list(desktops), "view": str(view),
             "steps": 0, "updated": 0, "conflicts": 0, "problems": [], "pending": []}
    for desktop in desktops:
        runner = getattr(truth, "run_pending_rechecks", None)
        if callable(runner):
            runner(int(desktop), str(view))
        if refresh:
            steps = [int(row["step"]) for row in db.frames_for(int(desktop), str(view))
                     if not row.get("missing")]
            stats = truth.refresh_range(int(desktop), str(view), steps)
            total["steps"] += len(steps)
            total["updated"] += int(stats.get("updated", 0))
            total["conflicts"] += int(stats.get("conflicts", 0))
            total["problems"].extend(stats.get("problems") or [])
        total["pending"].extend(_pending_rechecks(truth, int(desktop), str(view)))
    return total


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
def cmd_app(args: argparse.Namespace) -> int:
    """Open the annotator window; the exit code is the window's.

    The lock is taken by the window rather than by ``session`` here: it has to
    stay held for the whole session and be released on close, next to the exit
    backup, and a refusal has to reach the annotator as a message box.
    """
    from tda.ui import app as app_module

    return int(app_module.main(
        paths=args.paths,
        desktop=None if args.desktop is None else int(args.desktop),
        view=None if args.view is None else str(args.view),
        annotator=str(args.annotator),
        step=None if args.step is None else int(args.step),
        db=args.db,
    ))


def _add_app(sub) -> None:
    p = sub.add_parser("app", help="open the annotator (resumes the last frame)")
    p.add_argument("--desktop", type=int, default=None,
                   help="machine to open (default: where this annotator left off)")
    # one view, not a list: the window opens on a single view. The plural
    # spelling is accepted so that a hand that has just typed `--views` at an
    # export does not get "unrecognized arguments" here.
    p.add_argument("--view", "--views", dest="view", default=None, choices=list(VIEWS),
                   help="the ONE view to open (default: the last one, else scan); "
                        "unlike check and the exports this takes no list")
    p.add_argument("--annotator", required=True, help="who is annotating (the lock holder)")
    p.add_argument("--step", type=int, default=None, help="open this logical step")
    p.set_defaults(func=cmd_app)


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    """Recompile one view and print what the compiler is unhappy about.

    Exit code 1 when anything is reported, so a shell loop over the desktops
    can stop on the first machine that needs attention. "Anything" is three
    things, not one: the compiler's problems, the conflicts still standing
    open, and the re-checks still queued. A desktop with none of the first and
    six of the second is not a desktop to export, and answering 0 there was how
    an open conflict survived all the way into a release artefact.
    """
    with session(args, lock=True) as (_paths, db):
        desktop = int(args.desktop)
        tax = load_taxonomy()
        worst = EXIT_OK
        for view in _view_list(args.views):
            worst = max(worst, _check_one(db, tax, args, desktop, view))
        return worst


def _check_one(db, tax, args: argparse.Namespace, desktop: int, view: str) -> int:
    """Recompile one ``(desktop, view)`` and report it; the exit code is its own."""
    stats = _prepare_truth(db, tax, [desktop], view, refresh=True)
    problems = list(stats["problems"])
    standing = _open_conflicts(stats["truth"], db, [desktop], view)
    print(f"[check] D{desktop:02d} {view}: {stats['steps']} steps, "
          f"{stats['updated']} rows written, {stats['conflicts']} conflicts")
    if stats["pending"]:
        print(f"[check] {len(stats['pending'])} frames are still queued for "
              f"a re-check: {sorted(stats['pending'])[:10]}")
    for text in problems[: int(args.limit)]:
        print(f"  - {text}")
    if len(problems) > int(args.limit):
        print(f"  ... {len(problems) - int(args.limit)} more")
    print(f"problems: {len(problems)}")
    print(f"open conflicts: {standing}")
    return EXIT_ERROR if (problems or standing or stats["pending"]) else EXIT_OK


def _add_check(sub) -> None:
    p = sub.add_parser("check", help="recompile one or more views and count the problems")
    p.add_argument("--desktop", type=int, required=True)
    _add_view_flags(p)
    p.add_argument("--limit", type=int, default=20, help="how many problems to list")
    p.set_defaults(func=cmd_check)


# --------------------------------------------------------------------------- #
# build-cache
# --------------------------------------------------------------------------- #
def cmd_build_cache(args: argparse.Namespace) -> int:
    """Copy the chosen frames from ``F:`` into the local cache.

    A thin wrapper: :func:`tda.core.cache.main` owns the burst selection, the
    manifest and the progress log; this only resolves ``--cache`` from
    ``paths.yaml`` so the two command lines agree on where the cache is.
    """
    from tda.core import cache

    paths = load_paths(args.paths)
    wanted = _desktop_list(args.desktops)
    runs = _runs(wanted) if wanted else [(int(args.first), int(args.last))]
    extra: list[str] = []
    if args.index:
        extra += ["--index", str(args.index)]
    if args.log:
        extra += ["--log", str(args.log)]
    if args.recompute:
        extra.append("--recompute")
    if args.force:
        extra.append("--force")

    worst = 0
    views = ",".join(_view_list(args.views))
    for first, last in runs:
        argv = ["--cache", P.require(paths, "cache_dir"), "--views", views,
                "--first", str(first), "--last", str(last), *extra]
        worst = max(worst, int(cache.main(argv)))
    return worst


def _add_build_cache(sub) -> None:
    p = sub.add_parser("build-cache", help="copy the chosen frames into the local cache")
    _add_view_flags(p, plural_first=True)
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66 or 1-3,13")
    # Kept because tda.core.cache.main speaks this and scripts already use it;
    # --desktops wins when both are given.
    p.add_argument("--first", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--last", type=int, default=66, help=argparse.SUPPRESS)
    p.add_argument("--index", default=None)
    p.add_argument("--log", default=None)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build_cache)


# --------------------------------------------------------------------------- #
# exports
# --------------------------------------------------------------------------- #
def _export_prologue(args, db, desktops: Sequence[int], view: str, command: str):
    """``(exit code, truth)``: refuse while anything is unresolved, else go.

    An export is a release artefact: shipping a frame whose stored geometry the
    session had already marked stale -- or one a human is still being asked to
    adjudicate -- is worse than not shipping at all, so the command stops and
    says what to run ``check`` on.  The service it built is handed back so the
    exporter refreshes through the same one.

    Two reasons to stop, and the annotator can only overrule the second:
    a queued re-check is a job this command could not finish, while an open
    conflict is a question only they can answer.  ``--allow-conflicts`` says
    they have decided to ship over it.
    """
    stats = _prepare_truth(db, load_taxonomy(), desktops, view,
                           refresh=bool(getattr(args, "refresh", True)))
    verified, unverified = _frame_counts(db, desktops, view)
    print(f"[{command}] {view}: {verified} verified frames, {unverified} not "
          f"verified, {verified + unverified} considered")
    if stats["pending"]:
        print(f"[{command}] refused: {len(stats['pending'])} frames are still "
              f"pending a re-check ({sorted(stats['pending'])[:10]}). Run "
              f"'python -m tda.cli check --desktop N --view {view}' first.")
        return EXIT_ERROR, stats["truth"]
    if not bool(getattr(args, "allow_conflicts", False)):
        standing = _open_conflicts(stats["truth"], db, desktops, view)
        if standing:
            print(f"[{command}] refused: {standing} open conflict"
                  f"{'' if standing == 1 else 's'} on the requested desktops. "
                  f"Resolve them in the annotator, or pass --allow-conflicts to "
                  f"export the frozen rows as they stand.")
            return EXIT_ERROR, stats["truth"]
    return None, stats["truth"]


def cmd_export_coco(args: argparse.Namespace) -> int:
    """Write the compiled truth of one view as a COCO file."""
    from tda.core.export.coco import export_coco

    with session(args, lock=True) as (paths, db):
        desktops = _desktop_list(args.desktops)
        if not desktops:
            print("[export-coco] nothing to export: pass --desktops")
            return EXIT_ERROR
        views = _view_list(args.views)
        targets = _out_paths(args.out, views, paths, "coco_{view}.json", "export-coco")
        if targets is None:
            return EXIT_ERROR
        worst = EXIT_OK
        for view in views:
            out = targets[view]
            refused, truth = _export_prologue(args, db, desktops, view, "export-coco")
            if refused is not None:
                worst = max(worst, refused)
                continue
            stats = _call_export(
                export_coco, db, load_taxonomy(), desktops, view, str(out),
                allow_conflicts=bool(args.allow_conflicts),
                only_verified=bool(args.only_verified),
                roi_crop=bool(args.roi_crop), truth=truth,
            )
            if not _count(stats, "annotations"):
                worst = max(worst, _nothing_exported("export-coco", view, out,
                                                     bool(args.only_verified)))
                continue
            # ``images``/``annotations`` are the *lists*: printing them put the
            # whole COCO document on the terminal.
            print(f"[export-coco] {_count(stats, 'images')} images, "
                  f"{_count(stats, 'annotations')} annotations -> {out}")
        return worst


def _add_export_coco(sub) -> None:
    p = sub.add_parser("export-coco", help="compiled truth -> one COCO json per view")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-20")
    _add_view_flags(p)
    p.add_argument("--out", default=None,
                   help="one output file; only with a single view (without it each "
                        "view is written to <cache_dir>/coco_<view>.json)")
    p.add_argument("--roi-crop", action="store_true", help="crop to the pose-segment ROI")
    p.add_argument("--only-verified", dest="only_verified", action="store_true",
                   default=True, help="export confirmed rows only (the default)")
    p.add_argument("--no-only-verified", dest="only_verified", action="store_false",
                   help="export the automatic rows as well")
    _add_refresh_flags(p)
    p.set_defaults(func=cmd_export_coco)


def _add_refresh_flags(p) -> None:
    """What an export does before it reads the truth table, and what stops it."""
    p.add_argument("--refresh", dest="refresh", action="store_true", default=True,
                   help="recompile the requested frames first (the default)")
    p.add_argument("--no-refresh", dest="refresh", action="store_false",
                   help="export what is stored, only draining the re-check queue")
    p.add_argument("--allow-conflicts", dest="allow_conflicts", action="store_true",
                   help="export even though frames of these desktops still carry "
                        "an open conflict. Without it the export refuses: a "
                        "conflict is a frozen row the recompile disagrees with, "
                        "and only a human can say which of the two ships")


def cmd_export_vlm(args: argparse.Namespace) -> int:
    """Write the P0 question set of spec 8.2 for one view as JSONL."""
    from tda.core.export.vlm import TASKS, export_vlm

    with session(args, lock=True) as (paths, db):
        desktops = _desktop_list(args.desktops)
        if not desktops:
            print("[export-vlm] nothing to export: pass --desktops")
            return EXIT_ERROR
        views = _view_list(args.views)
        targets = _out_paths(args.out, views, paths, "vlm_{view}.jsonl", "export-vlm")
        if targets is None:
            return EXIT_ERROR
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()] or list(TASKS)
        unknown = [t for t in tasks if t not in TASKS]
        if unknown:
            print(f"[export-vlm] not a task: {', '.join(unknown)}; "
                  f"the P0 set is {','.join(TASKS)}")
            return EXIT_ERROR
        worst = EXIT_OK
        for view in views:
            out = targets[view]
            refused, truth = _export_prologue(args, db, desktops, view, "export-vlm")
            if refused is not None:
                worst = max(worst, refused)
                continue
            stats = _call_export(
                export_vlm, db, load_taxonomy(), desktops, view, str(out),
                allow_conflicts=bool(args.allow_conflicts), tasks=tasks,
                only_verified=bool(args.only_verified), truth=truth,
            )
            _report_vlm(stats, view)
            if not _count(stats, "records"):
                worst = max(worst, _nothing_exported("export-vlm", view, out,
                                                     bool(args.only_verified),
                                                     siblings_of_vlm(str(out))))
                continue
            # the total goes last and stays short: it is the line a script reads
            print(f"[export-vlm] {_count(stats, 'records')} records -> {out}")
        return worst


def siblings_of_vlm(out: str) -> Sequence[Path]:
    """The manifest and the summary written beside one VLM JSONL."""
    from tda.core.export.vlm import siblings

    return siblings(out)


def _report_vlm(stats: dict, view: str) -> None:
    """The per-task counts and everything the export declined to answer.

    Three silences, and a release run has to see all three. ``illegal_steps``:
    the logged action there breaks a hard constraint, so the graph and the log
    contradict each other and no affordance answer may be published as ground
    truth (the 18 lines of ``reports/constraints_report.md``).
    ``excluded_desktops``: no graph at all, so every answer would be "nothing is
    blocked". ``v10_excluded``: the log names a target nobody resolved, and V10
    is scored exhaustively, so an incomplete history is a wrong answer.
    """
    by_task = stats.get("by_task") or {}
    if by_task:
        print(f"[export-vlm] {view}: "
              + " ".join(f"{task}={n}" for task, n in sorted(by_task.items())))
        source = stats.get("by_source") or {}
        if source:
            print(f"[export-vlm] {view}: "
                  + " ".join(f"{k}={v}" for k, v in sorted(source.items())))
    illegal = stats.get("illegal_steps") or {}
    if illegal:
        listed = ", ".join(f"D{d}:{sorted(steps)}" for d, steps in sorted(illegal.items()))
        print(f"[export-vlm] {len(illegal)} desktop(s) have steps whose logged "
              f"action the graph forbids; V4/V5/V6/V16 say nothing there: {listed}")
    excluded = stats.get("excluded_desktops") or {}
    if excluded:
        listed = ", ".join(f"D{d}:{why}" for d, why in sorted(excluded.items()))
        print(f"[export-vlm] {len(excluded)} desktop(s) answer no affordance or "
              f"planning question at all: {listed}")
    no_v10 = stats.get("v10_excluded") or {}
    if no_v10:
        listed = ", ".join(f"D{d}:{len(t)}" for d, t in sorted(no_v10.items()))
        print(f"[export-vlm] {len(no_v10)} desktop(s) have logged actions whose "
              f"target is unresolved, so V10 is not asked of them: {listed}")
    _report_shortcuts(stats, view)


def _report_shortcuts(stats: dict, view: str) -> None:
    """How much of this file a model could answer without looking at an image.

    Printed, not buried: a generated benchmark is always at risk of being
    solvable from its own wording, and the only defence is to measure it every
    time and put the number where a release run sees it. Each classifier is
    fitted to the file it scores, so these are upper bounds no text-only model
    can beat.
    """
    shortcuts = stats.get("shortcuts") or {}
    for task in sorted(shortcuts):
        row = shortcuts[task]
        print(f"[export-vlm] {view}: {task} text-only upper bound — "
              f"majority {row['majority']:.1%}, verb {row['verb']:.1%}, "
              f"class {row['class']:.1%}, verb+class {row['verb_class']:.1%}, "
              f"template {row['template']:.1%} ({row['records']} records)")
    option = stats.get("v6_option_shortcut") or {}
    if option.get("records"):
        print(f"[export-vlm] {view}: V6 option-verb shortcut "
              f"{option['shortcut']:.1%} against {option['random']:.1%} random "
              f"({option['records']} records with options)")


def _add_export_vlm(sub) -> None:
    from tda.core.export.vlm_tasks import TASKS

    p = sub.add_parser("export-vlm", help="compiled truth -> one VLM JSONL per view")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-20")
    _add_view_flags(p)
    p.add_argument("--out", default=None,
                   help="one output file; only with a single view (without it each "
                        "view is written to <cache_dir>/vlm_<view>.jsonl)")
    p.add_argument("--tasks", default=",".join(TASKS),
                   help=f"comma-separated task ids; the P0 set is {','.join(TASKS)}")
    p.add_argument("--only-verified", dest="only_verified", action="store_true",
                   default=False, help="ask only about confirmed rows")
    p.add_argument("--no-only-verified", dest="only_verified", action="store_false")
    _add_refresh_flags(p)
    p.set_defaults(func=cmd_export_vlm)


#: Appended to ``tda.cli.SUBCOMMANDS`` in the order they appear in ``--help``.
SUBCOMMANDS: tuple[Callable[[argparse._SubParsersAction], None], ...] = (
    _add_app,
    _add_check,
    _add_build_cache,
    _add_export_coco,
    _add_export_vlm,
)
