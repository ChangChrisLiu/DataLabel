"""The commands that come after the data pipeline: the app, checks and exports.

:mod:`tda.cli` holds the four import commands that seed the database; this
module holds the five that use it -- open the annotator, recompile and count the
problems of one view, build the local image cache, and write the two rehearsal
exports.  They are registered by appending :data:`SUBCOMMANDS` to the parser
there, so ``python -m tda.cli --help`` lists all of them together.

Everything that writes takes the single-user lock through ``tda.cli._session``
(exit code 3 while somebody else holds it), including ``check``: recompiling a
view writes the ``compiled_mask`` rows it finds stale.
"""
from __future__ import annotations

import argparse
from typing import Callable, Sequence

from tda import pipeline as P
from tda.core.model import VIEWS
from tda.core.taxonomy import load_taxonomy

__all__ = ["SUBCOMMANDS", "cmd_app", "cmd_build_cache", "cmd_check",
           "cmd_export_coco", "cmd_export_vlm"]


def _desktop_list(spec: str | None) -> list[int]:
    """``--desktops 1-3,13`` -> ``[1, 2, 3, 13]``; nothing given -> ``[]``."""
    wanted = P.parse_desktops(spec) if spec else None
    return sorted(wanted) if wanted else []


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

    The lock is taken by the window rather than by ``_session`` here: it has to
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
    p.add_argument("--view", default=None, choices=list(VIEWS),
                   help="view to open (default: the last one, else scan)")
    p.add_argument("--annotator", required=True, help="who is annotating (the lock holder)")
    p.add_argument("--step", type=int, default=None, help="open this logical step")
    p.set_defaults(func=cmd_app)


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    """Recompile one view and print what the compiler is unhappy about.

    Exit code 1 when anything is reported, so a shell loop over the desktops
    can stop on the first machine that needs attention.
    """
    from tda.cli import EXIT_ERROR, EXIT_OK, _session

    with _session(args, lock=True) as (_paths, db):
        stats = _prepare_truth(db, load_taxonomy(), [int(args.desktop)],
                               str(args.view), refresh=True)
        problems = list(stats["problems"])
        print(f"[check] D{args.desktop:02d} {args.view}: {stats['steps']} steps, "
              f"{stats['updated']} rows written, {stats['conflicts']} conflicts")
        if stats["pending"]:
            print(f"[check] {len(stats['pending'])} frames are still queued for "
                  f"a re-check: {sorted(stats['pending'])[:10]}")
        for text in problems[: int(args.limit)]:
            print(f"  - {text}")
        if len(problems) > int(args.limit):
            print(f"  ... {len(problems) - int(args.limit)} more")
        print(f"problems: {len(problems)}")
        return EXIT_ERROR if problems else EXIT_OK


def _add_check(sub) -> None:
    p = sub.add_parser("check", help="recompile one view and count its problems")
    p.add_argument("--desktop", type=int, required=True)
    p.add_argument("--view", default="scan", choices=list(VIEWS))
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

    paths = P.load_paths(args.paths)
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
    for first, last in runs:
        argv = ["--cache", P.require(paths, "cache_dir"), "--views", str(args.views),
                "--first", str(first), "--last", str(last), *extra]
        worst = max(worst, int(cache.main(argv)))
    return worst


def _add_build_cache(sub) -> None:
    p = sub.add_parser("build-cache", help="copy the chosen frames into the local cache")
    p.add_argument("--views", default="scan", help="comma separated views")
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
def _export_prologue(args, db, desktops: Sequence[int], command: str):
    """``(exit code, truth)``: refuse while re-checks are still queued, else go.

    An export is a release artefact: shipping a frame whose stored geometry the
    session had already marked stale is worse than not shipping at all, so the
    command stops and says which frames to run ``check`` on.  The service it
    built is handed back so the exporter refreshes through the same one.
    """
    from tda.cli import EXIT_ERROR

    stats = _prepare_truth(db, load_taxonomy(), desktops, str(args.view),
                           refresh=bool(getattr(args, "refresh", True)))
    if stats["pending"]:
        print(f"[{command}] refused: {len(stats['pending'])} frames are still "
              f"pending a re-check ({sorted(stats['pending'])[:10]}). Run "
              f"'python -m tda.cli check --desktop N --view {args.view}' first.")
        return EXIT_ERROR, stats["truth"]
    return None, stats["truth"]


def cmd_export_coco(args: argparse.Namespace) -> int:
    """Write the compiled truth of one view as a COCO file."""
    from tda.cli import EXIT_ERROR, EXIT_OK, _session
    from tda.core.export.coco import export_coco

    with _session(args, lock=True) as (paths, db):
        out = args.out or P.cache_file(paths, f"coco_{args.view}.json")
        desktops = _desktop_list(args.desktops)
        if not desktops:
            print("[export-coco] nothing to export: pass --desktops")
            return EXIT_ERROR
        refused, truth = _export_prologue(args, db, desktops, "export-coco")
        if refused is not None:
            return refused
        stats = export_coco(db, load_taxonomy(), desktops, str(args.view), str(out),
                            only_verified=bool(args.only_verified),
                            roi_crop=bool(args.roi_crop), truth=truth)
        # ``images``/``annotations`` are the *lists*: printing them put the
        # whole COCO document on the terminal.
        print(f"[export-coco] {_count(stats, 'images')} images, "
              f"{_count(stats, 'annotations')} annotations -> {out}")
        return EXIT_OK


def _add_export_coco(sub) -> None:
    p = sub.add_parser("export-coco", help="compiled truth -> one COCO json")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-20")
    p.add_argument("--view", default="scan", choices=list(VIEWS))
    p.add_argument("--out", default=None)
    p.add_argument("--roi-crop", action="store_true", help="crop to the pose-segment ROI")
    p.add_argument("--only-verified", dest="only_verified", action="store_true",
                   default=True, help="export confirmed rows only (the default)")
    p.add_argument("--no-only-verified", dest="only_verified", action="store_false",
                   help="export the automatic rows as well")
    _add_refresh_flags(p)
    p.set_defaults(func=cmd_export_coco)


def _add_refresh_flags(p) -> None:
    """Whether the export recompiles first (it does; ``--no-refresh`` skips it)."""
    p.add_argument("--refresh", dest="refresh", action="store_true", default=True,
                   help="recompile the requested frames first (the default)")
    p.add_argument("--no-refresh", dest="refresh", action="store_false",
                   help="export what is stored, only draining the re-check queue")


def cmd_export_vlm(args: argparse.Namespace) -> int:
    """Write the V1/V2/V3 question set of one view as JSONL."""
    from tda.cli import EXIT_ERROR, EXIT_OK, _session
    from tda.core.export.vlm import TASKS, export_vlm

    with _session(args, lock=True) as (paths, db):
        out = args.out or P.cache_file(paths, f"vlm_{args.view}.jsonl")
        desktops = _desktop_list(args.desktops)
        if not desktops:
            print("[export-vlm] nothing to export: pass --desktops")
            return EXIT_ERROR
        refused, truth = _export_prologue(args, db, desktops, "export-vlm")
        if refused is not None:
            return refused
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()] or list(TASKS)
        stats = export_vlm(db, load_taxonomy(), desktops, str(args.view), str(out),
                           tasks=tasks, only_verified=bool(args.only_verified),
                           truth=truth)
        print(f"[export-vlm] {stats.get('records', 0)} records "
              f"({stats.get('by_task', {})}) -> {out}")
        return EXIT_OK


def _add_export_vlm(sub) -> None:
    p = sub.add_parser("export-vlm", help="compiled truth -> one VLM JSONL")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-20")
    p.add_argument("--view", default="scan", choices=list(VIEWS))
    p.add_argument("--out", default=None)
    p.add_argument("--tasks", default="V1,V2,V3")
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
