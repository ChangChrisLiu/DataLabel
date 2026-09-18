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
from typing import Callable

from tda import pipeline as P
from tda.core.model import VIEWS
from tda.core.taxonomy import load_taxonomy

__all__ = ["SUBCOMMANDS", "cmd_app", "cmd_build_cache", "cmd_check",
           "cmd_export_coco", "cmd_export_vlm"]


def _desktop_list(spec: str | None, fallback: int | None = None) -> list[int]:
    wanted = P.parse_desktops(spec) if spec else None
    if wanted:
        return sorted(wanted)
    return [] if fallback is None else [int(fallback)]


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
        paths=args.paths, desktop=int(args.desktop), view=str(args.view),
        annotator=str(args.annotator), step=None if args.step is None else int(args.step),
        db=args.db,
    ))


def _add_app(sub) -> None:
    p = sub.add_parser("app", help="open the annotator on one desktop/view")
    p.add_argument("--desktop", type=int, required=True)
    p.add_argument("--view", default="scan", choices=list(VIEWS))
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
    from tda.core.truth import TruthService

    with _session(args, lock=True) as (_paths, db):
        steps = [int(row["step"]) for row in db.frames_for(args.desktop, args.view)
                 if not row.get("missing")]
        truth = TruthService(db, load_taxonomy())
        stats = truth.refresh_range(int(args.desktop), str(args.view), steps)
        problems = list(stats.get("problems") or [])
        print(f"[check] D{args.desktop:02d} {args.view}: {len(steps)} steps, "
              f"{stats.get('updated', 0)} rows written, "
              f"{stats.get('conflicts', 0)} conflicts")
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
    argv = ["--cache", P.require(paths, "cache_dir"),
            "--views", str(args.views),
            "--first", str(args.first), "--last", str(args.last)]
    if args.index:
        argv += ["--index", str(args.index)]
    if args.log:
        argv += ["--log", str(args.log)]
    if args.recompute:
        argv.append("--recompute")
    if args.force:
        argv.append("--force")
    return int(cache.main(argv))


def _add_build_cache(sub) -> None:
    p = sub.add_parser("build-cache", help="copy the chosen frames into the local cache")
    p.add_argument("--views", default="scan", help="comma separated views")
    p.add_argument("--first", type=int, default=1)
    p.add_argument("--last", type=int, default=66)
    p.add_argument("--index", default=None)
    p.add_argument("--log", default=None)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build_cache)


# --------------------------------------------------------------------------- #
# exports
# --------------------------------------------------------------------------- #
def cmd_export_coco(args: argparse.Namespace) -> int:
    """Write the compiled truth of one view as a COCO file."""
    from tda.cli import EXIT_ERROR, EXIT_OK, _session
    from tda.core.export.coco import export_coco

    with _session(args) as (paths, db):
        out = args.out or P.cache_file(paths, f"coco_{args.view}.json")
        desktops = _desktop_list(args.desktops)
        if not desktops:
            print("[export-coco] nothing to export: pass --desktops")
            return EXIT_ERROR
        stats = export_coco(db, load_taxonomy(), desktops, str(args.view), str(out),
                            only_verified=bool(args.only_verified),
                            roi_crop=bool(args.roi_crop))
        print(f"[export-coco] {stats.get('images', 0)} images, "
              f"{stats.get('annotations', 0)} annotations -> {out}")
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
    p.set_defaults(func=cmd_export_coco)


def cmd_export_vlm(args: argparse.Namespace) -> int:
    """Write the V1/V2/V3 question set of one view as JSONL."""
    from tda.cli import EXIT_ERROR, EXIT_OK, _session
    from tda.core.export.vlm import TASKS, export_vlm

    with _session(args) as (paths, db):
        out = args.out or P.cache_file(paths, f"vlm_{args.view}.jsonl")
        desktops = _desktop_list(args.desktops)
        if not desktops:
            print("[export-vlm] nothing to export: pass --desktops")
            return EXIT_ERROR
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()] or list(TASKS)
        stats = export_vlm(db, load_taxonomy(), desktops, str(args.view), str(out),
                           tasks=tasks, only_verified=bool(args.only_verified))
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
    p.set_defaults(func=cmd_export_vlm)


#: Appended to ``tda.cli.SUBCOMMANDS`` in the order they appear in ``--help``.
SUBCOMMANDS: tuple[Callable[[argparse._SubParsersAction], None], ...] = (
    _add_app,
    _add_check,
    _add_build_cache,
    _add_export_coco,
    _add_export_vlm,
)
