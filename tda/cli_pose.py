"""``pose-breaks``: list, import, accept and reject a view's own pose breaks.

::

    python -m tda.cli pose-breaks list [--desktop N] [--view V] [--status S]
    python -m tda.cli pose-breaks import <events.csv> [--min-px 2] [--dry-run]
    python -m tda.cli pose-breaks accept --desktop N --view V --step S [--carry]
    python -m tda.cli pose-breaks reject --desktop N --view V --step S

Spec 2.5 (v1.5): a view's pose segments are cut at the ``reorient`` steps --
which the four views share, because the chassis is flipped for all of them --
**union that view's own breaks**.  The camera audit
(``experiments_out/plan_b_probe/camera_moves``) measured those: six camera moves
inside a sequence (oak1 D2 s5->6, D4 s6->7, D29 s46->47, D32 s38->39, D36
s18->19 at 190 px; oak2 D1 s33->34) and one chassis rotation the scanner saw
(D63 s31->32, ~90 degrees, typed ``compound`` in the log and therefore invisible
to the ``reorient`` rule).

``import`` turns that CSV into **proposals**, never into cuts: a re-cut moves
every keyframe, layer order and ROI of two segments, and a 2 px estimate from a
feature matcher is not a reason to do that to somebody's afternoon.  The
annotator flashes the two frames against each other (``Tab`` in the window) and
answers; ``accept`` and ``reject`` are the two answers, and a rejected row stays
in the table so the next import does not ask again.

Safety is the same as ``infer-relations``: the single-user lock, a backup before
anything is written, and ``--dry-run`` which writes nothing at all (the database
is still opened and schema-checked under the lock).
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

from tda.cli_common import EXIT_ERROR, EXIT_OK, safety_backup as _safety_backup, session as _session
from tda.core.db import Db
from tda.core.model import VIEWS
from tda.core.pose_breaks import ACCEPTED, KIND_CHASSIS, PROPOSED, REJECTED

__all__ = [
    "ANNOTATOR", "AUDIT_KINDS", "AUDIT_VERDICTS", "CARRY_BELOW_PX", "Proposal",
    "ImportRun", "accept_break", "import_events", "read_events", "reject_break",
]

#: ``op_log.annotator`` -- this is machinery, not a person.
ANNOTATOR = "cli:pose-breaks"
#: Audit rows that describe a real move of something.
AUDIT_KINDS = ("camera", "chassis")
#: ...and were decided by a human looking at the frames, rather than clustered
#: away as undetermined.  ``chassis_manual`` is the verdict the D63 rotation
#: carries: the tape test could not decide it and somebody looked.
AUDIT_VERDICTS = ("confirmed", "chassis_manual")
#: Below this much movement the shapes are still roughly in place, so carrying
#: them across the new boundary is the default (the plan's split dialog uses the
#: same number).  Above it the annotator is redrawing anyway.
CARRY_BELOW_PX = 25.0
#: Default ``--min-px``: the audit's own noise floor.
MIN_PX = 2.0


@dataclass(frozen=True)
class Proposal:
    """One break the audit proposes for one view."""

    desktop: int
    view: str
    step: int
    kind: str
    magnitude_px: Optional[float]
    note: str = ""

    def line(self) -> str:
        size = "" if self.magnitude_px is None else f" ~{self.magnitude_px:.1f} px"
        extra = f" ({self.note})" if self.note else ""
        return (f"D{self.desktop:02d} {self.view} step {self.step}: {self.kind}"
                f"{size}{extra}")


@dataclass
class ImportRun:
    """What one ``pose-breaks import`` did (or would have done)."""

    source: str = ""
    dry_run: bool = False
    proposals: list[Proposal] = field(default_factory=list)
    written: list[Proposal] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: Rows the CSV offered that a stored decision already answers.
    known: list[Proposal] = field(default_factory=list)

    def per_view(self) -> dict[str, int]:
        out = {view: 0 for view in VIEWS}
        for p in self.proposals:
            out[p.view] = out.get(p.view, 0) + 1
        return out


# --------------------------------------------------------------------------- #
# the audit CSV
# --------------------------------------------------------------------------- #
def read_events(path: str, min_px: float = MIN_PX) -> tuple[list[Proposal], list[str]]:
    """Proposals of one ``events.csv``, plus a line per row that was left out.

    A ``chassis`` event is proposed for **all four views** -- the machine moved,
    so every lens saw it -- and a ``camera`` event only for the view whose lens
    was knocked.  The break step is ``step_to``: the new segment starts at the
    first frame that shows the new pose.
    """
    name = os.path.basename(str(path))
    out: list[Proposal] = []
    skipped: list[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            kind = (row.get("kind") or "").strip()
            verdict = (row.get("verdict") or "").strip()
            if kind not in AUDIT_KINDS or verdict not in AUDIT_VERDICTS:
                continue
            magnitude = _float(row.get("magnitude_px"))
            if magnitude is not None and magnitude < float(min_px):
                skipped.append(f"{name}: {row.get('view')} D{row.get('desktop')} "
                               f"step {row.get('step_to')} moved {magnitude:.2f} px, "
                               f"below --min-px {float(min_px):g}")
                continue
            desktop, step = _int(row.get("desktop")), _int(row.get("step_to"))
            if desktop is None or step is None:
                skipped.append(f"{name}: a {kind} row has no desktop/step_to; skipped")
                continue
            views = VIEWS if kind == KIND_CHASSIS else ((row.get("view") or "").strip(),)
            for view in views:
                if view not in VIEWS:
                    skipped.append(f"{name}: unknown view {view!r}; skipped")
                    continue
                out.append(Proposal(desktop, view, step, kind, magnitude,
                                    (row.get("note") or "").strip()))
    return out, skipped


def _float(value) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def import_events(db: Db, path: str, *, min_px: float = MIN_PX, dry_run: bool = False,
                  log=None) -> ImportRun:
    """Store the audit's findings as ``proposed`` breaks; never overwrite a decision.

    A row the table already has is reported and left alone, whatever its status:
    the same six camera moves come out of every re-run of the audit, and an
    accepted or rejected break is a human's answer, not something an import may
    take back.
    """
    proposals, skipped = read_events(path, min_px)
    run = ImportRun(source=os.path.basename(str(path)), dry_run=dry_run,
                    proposals=proposals, skipped=skipped)
    source = f"audit:{run.source}"
    for p in proposals:
        stored = db.pose_break(p.desktop, p.view, p.step)
        if stored is not None:
            run.known.append(p)
            continue
        run.written.append(p)
        if not dry_run:
            db.add_pose_break(p.desktop, p.view, p.step, status=PROPOSED, kind=p.kind,
                              magnitude_px=p.magnitude_px, source=source, note=p.note)
    if log:
        prefix = "[pose-breaks]" + (" (dry run)" if dry_run else "")
        for text in run.skipped:
            log(f"{prefix} {text}")
        for p in run.written:
            log(f"{prefix} {'would propose' if dry_run else 'proposed'} {p.line()}")
        for p in run.known:
            log(f"{prefix} already decided, left alone: {p.line()}")
        counts = ", ".join(f"{view} {n}" for view, n in run.per_view().items())
        log(f"{prefix} {len(run.proposals)} proposals from {run.source} ({counts}); "
            f"{len(run.written)} new, {len(run.known)} already stored, "
            f"{len(run.skipped)} skipped")
    return run


# --------------------------------------------------------------------------- #
# accept / reject
# --------------------------------------------------------------------------- #
def accept_break(db: Db, desktop: int, view: str, step: int, *,
                 carry: Optional[bool] = None, dry_run: bool = False, log=None) -> dict:
    """Accept one break and re-cut its view; returns the re-cut summary.

    ``carry`` duplicates every shape the new boundary cuts through into the
    earlier segment, so the frames before it keep a shape instead of becoming
    ``missing_shape``.  Left unset it follows the measured movement: on by
    default below :data:`CARRY_BELOW_PX`, off above it, which is the same rule
    the window's split dialog starts from.
    """
    row = db.pose_break(desktop, view, step)
    if row is None:
        raise ValueError(f"D{desktop:02d} {view} has no break at step {step}; "
                         f"add one with `pose-breaks import`, or in the window")
    if carry is None:
        magnitude = row["magnitude_px"]
        carry = magnitude is not None and float(magnitude) < CARRY_BELOW_PX
    if dry_run:
        bounds, n_steps = db.view_boundaries(desktop, view)
        if log:
            log(f"[pose-breaks] (dry run) would accept D{desktop:02d} {view} step "
                f"{step} (carry={'on' if carry else 'off'}); the view would be cut "
                f"at {sorted(set(bounds) | {int(step)})} of {n_steps} steps")
        return {"changed": False, "ranges": [], "rechecked": [], "carried": [],
                "uncarried": [], "discarded": [], "ref_moves": []}
    with db.transaction():
        db.set_pose_break_status(desktop, view, step, ACCEPTED)
        out = db.recut_view(desktop, view, carry_at=[int(step)] if carry else (),
                            annotator=ANNOTATOR)
    if log:
        _log_recut(log, f"accepted D{desktop:02d} {view} step {step}", out, carry)
    return out


def reject_break(db: Db, desktop: int, view: str, step: int, *, dry_run: bool = False,
                 log=None) -> dict:
    """Reject one break and re-cut its view, which merges the two segments back."""
    if db.pose_break(desktop, view, step) is None:
        raise ValueError(f"D{desktop:02d} {view} has no break at step {step}")
    if dry_run:
        if log:
            log(f"[pose-breaks] (dry run) would reject D{desktop:02d} {view} step {step}")
        return {"changed": False, "ranges": [], "rechecked": [], "carried": [],
                "uncarried": [], "discarded": [], "ref_moves": []}
    with db.transaction():
        db.set_pose_break_status(desktop, view, step, REJECTED)
        out = db.recut_view(desktop, view, annotator=ANNOTATOR)
    if log:
        _log_recut(log, f"rejected D{desktop:02d} {view} step {step}", out, None)
    return out


def _log_recut(log, what: str, out: dict, carry: Optional[bool]) -> None:
    """One line for the verdict, then one per thing the re-cut could not keep."""
    ranges = ", ".join(f"{seg}:[{start}-{end}]" for seg, start, end in out["ranges"])
    tail = "" if carry is None else f", carry {'on' if carry else 'off'}"
    log(f"[pose-breaks] {what}{tail}: {len(out['ranges'])} segments ({ranges})")
    if out["carried"]:
        log(f"[pose-breaks]   {len(out['carried'])} shapes carried across the boundary")
    if out["uncarried"]:
        log(f"[pose-breaks]   {len(out['uncarried'])} carried shapes had been edited "
            f"and were kept; check their anchors")
    for dropped in out["discarded"]:
        log(f"[pose-breaks]   {dropped['table']} of segment {dropped['pose_segment']} "
            f"could not survive the merge into segment {dropped['into']}; it is in "
            f"the op log")
    if out["rechecked"]:
        log(f"[pose-breaks]   {len(out['rechecked'])} verified frames queued for a "
            f"re-check: {out['rechecked']}")


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def list_breaks(db: Db, desktop: Optional[int] = None, view: Optional[str] = None,
                status: Optional[str] = None, log=print) -> list[dict]:
    """Print (and return) the stored breaks, newest desktop last."""
    desktops = [int(desktop)] if desktop is not None else db.desktop_ids()
    rows: list[dict] = []
    for one in desktops:
        rows += db.pose_breaks(one, view, status)
    for row in rows:
        size = "" if row["magnitude_px"] is None else f" ~{row['magnitude_px']:.1f} px"
        note = f"  {row['note']}" if row["note"] else ""
        log(f"[pose-breaks] D{row['desktop']:02d} {row['view']:5s} step "
            f"{row['step']:3d}  {row['status']:8s} {row['kind'] or '-':8s}{size}"
            f"  {row['source']}{note}")
    log(f"[pose-breaks] {len(rows)} breaks"
        + (f" for D{int(desktop):02d}" if desktop is not None else "")
        + (f", view {view}" if view else "")
        + (f", status {status}" if status else ""))
    return rows


# --------------------------------------------------------------------------- #
# command
# --------------------------------------------------------------------------- #
def cmd_pose_breaks(args: argparse.Namespace) -> int:
    """Run one of ``list`` / ``import`` / ``accept`` / ``reject``."""
    writes = args.action in ("import", "accept", "reject") and not args.dry_run
    if args.action == "import" and not args.events:
        print("[pose-breaks] import needs the audit's events.csv: "
              "`pose-breaks import experiments_out/plan_b_probe/camera_moves/events.csv`")
        return EXIT_ERROR
    if args.action in ("accept", "reject") and (
            args.desktop is None or not args.view or args.step is None):
        print(f"[pose-breaks] {args.action} needs --desktop, --view and --step")
        return EXIT_ERROR

    with _session(args, lock=True) as (paths, db):
        if writes and not _safety_backup(
                paths, db, "pose-breaks",
                "it re-cuts a view's pose segments and moves the rows keyed by them"):
            return EXIT_ERROR  # the line is printed and nothing was written
        try:
            return _run(args, db)
        except (ValueError, OSError) as exc:
            print(f"[pose-breaks] {exc}")
            return EXIT_ERROR


def _run(args: argparse.Namespace, db: Db) -> int:
    if args.action == "list":
        list_breaks(db, args.desktop, args.view, args.status, log=print)
        return EXIT_OK
    if args.action == "import":
        import_events(db, args.events, min_px=args.min_px, dry_run=args.dry_run,
                      log=print)
        return EXIT_OK
    carry = None if args.carry is None else bool(args.carry)
    if args.action == "accept":
        accept_break(db, int(args.desktop), str(args.view), int(args.step),
                     carry=carry, dry_run=args.dry_run, log=print)
    else:
        reject_break(db, int(args.desktop), str(args.view), int(args.step),
                     dry_run=args.dry_run, log=print)
    return EXIT_OK


def _add_pose_breaks(sub) -> None:
    p = sub.add_parser(
        "pose-breaks",
        help="list / import / accept / reject the per-view pose breaks of spec 2.5",
    )
    p.add_argument("action", choices=("list", "import", "accept", "reject"))
    p.add_argument("events", nargs="?", default=None,
                   help="for `import`: the camera audit's events.csv")
    p.add_argument("--desktop", type=int, default=None)
    p.add_argument("--view", default=None, choices=VIEWS)
    p.add_argument("--step", type=int, default=None,
                   help="the step the NEW segment starts at")
    p.add_argument("--status", default=None, choices=(PROPOSED, ACCEPTED, REJECTED),
                   help="for `list`: show only these")
    p.add_argument("--min-px", type=float, default=MIN_PX, dest="min_px",
                   help=f"for `import`: ignore movements smaller than this "
                        f"(default {MIN_PX:g} px, the audit's noise floor)")
    p.add_argument("--carry", dest="carry", action="store_true", default=None,
                   help="for `accept`: duplicate every shape the new boundary cuts "
                        f"through into the earlier segment. Default: on below "
                        f"{CARRY_BELOW_PX:g} px of measured movement, off above it")
    p.add_argument("--no-carry", dest="carry", action="store_false",
                   help="for `accept`: let the frames before the boundary become "
                        "missing_shape, which is the cue to redraw them in the new pose")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would happen and write nothing (the database is "
                        "still opened and schema-checked under the lock); no backup "
                        "is taken")
    p.set_defaults(func=cmd_pose_breaks)
