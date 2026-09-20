"""The index half of the data pipeline behind ``python -m tda.cli`` (spec 2.2-2.5).

:mod:`tda.cli` holds the argument parsing and the printing, this module and
:mod:`tda.pipeline_logs` hold the work, so every step is callable from a test or
from the GUI without going through ``argparse``.

The pipeline runs in one order, because each step needs what the previous one
wrote::

    build-index -> load-index -> import-logs -> import-ls

Here: the shared configuration helpers, ``index.json`` -> ``frame`` rows and
pose segments, the pose-segment split at the ``reorient`` steps, and the
counters behind ``status``. The Drive-sheet import lives next door in
:mod:`tda.pipeline_logs`.

Nothing here writes outside ``D:`` -- :meth:`tda.core.db.Db.backup` is the one
command that may reach ``F:``, and only into ``backup_dir``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional

from tda.core import db_pose
from tda.core.db import Db
from tda.core.db_backup import DEFAULT_KEEP
from tda.core.db_status import VIEW_COUNTERS
from tda.core.index import DesktopIndex, load_index
from tda.core.model import VIEWS
from tda.core.truth_inputs import TRUTH_AUX_KEYS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS_PATH = "configs/paths.yaml"
INDEX_NAME = "index.json"
INDEX_REPORT_NAME = "index_report.md"
LOGS_REPORT_NAME = "import_logs_issues.md"
LS_SUMMARY_NAME = "ls_import_summary.json"
LS_EXPORT_NAME = "labelstudio/humansignal_annotated_projects_export.json"
DRIVE_SUBDIR = "drive"
ALL_DESKTOPS = range(1, 67)
#: How many pose issues one desktop's meta keeps (newest last).
POSE_ISSUE_LIMIT = 50
#: ``op_log.annotator`` of a re-cut the pipeline ran, rather than a human.
SPLIT_ANNOTATOR = "cli:split-pose-segments"

#: ``print`` by default; tests and the GUI pass their own sink.
Log = Callable[[str], None]

__all__ = [
    "backup_dest", "backup_keep", "cache_file", "desktops_without_steps", "drive_dir", "index_path",
    "load_index_into_db", "load_paths", "ls_export_path", "merge_desktop_meta", "open_db",
    "parse_desktops", "read_index", "require", "split_pose_segments", "status_rows",
]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_paths(path: str = DEFAULT_PATHS_PATH) -> dict:
    """Load ``paths.yaml``; a relative path may be given from the repo root."""
    import yaml

    if not (os.path.isabs(path) or os.path.exists(path)):
        path = str(REPO_ROOT / path)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def require(paths: dict, key: str) -> str:
    """One path from ``paths.yaml``, with a readable error when it is missing."""
    value = paths.get(key)
    if not value:
        raise ValueError(f"paths.yaml defines no {key}")
    return str(value)


def parse_desktops(spec: Optional[str]) -> Optional[set[int]]:
    """Parse ``"13"`` / ``"1-66"`` / ``"1-3,13,60-61"``; ``None`` means "all"."""
    if spec is None:
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        try:
            lo = int(first)
            hi = int(last) if last else lo
        except ValueError:
            raise ValueError(f"not a desktop range: {part!r}") from None
        if hi < lo:
            raise ValueError(f"empty desktop range: {part!r}")
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError(f"no desktops in {spec!r}")
    return out


def wanted(desktop: int, desktops: Optional[set[int]]) -> bool:
    """Is this desktop in the ``--desktops`` selection (``None`` means all)?"""
    return desktops is None or desktop in desktops


def index_path(paths: dict) -> str:
    """``<cache_dir>/index.json``."""
    return os.path.join(paths.get("cache_dir", "cache"), INDEX_NAME)


def cache_file(paths: dict, name: str) -> str:
    """A file inside ``cache_dir`` (the directory is created on demand)."""
    cache = paths.get("cache_dir", "cache")
    os.makedirs(cache, exist_ok=True)
    return os.path.join(cache, name)


def drive_dir(paths: dict) -> str:
    """``<raw_logs_dir>/drive`` -- the exported Drive sheets."""
    return os.path.join(paths.get("raw_logs_dir", "raw_logs"), DRIVE_SUBDIR)


def ls_export_path(paths: dict) -> str:
    """``<raw_logs_dir>/labelstudio/humansignal_annotated_projects_export.json``."""
    return os.path.join(paths.get("raw_logs_dir", "raw_logs"), *LS_EXPORT_NAME.split("/"))


def _within(target: Path, root: Path) -> bool:
    """Is ``target`` ``root`` itself or something under it?

    ``resolve()`` has already followed symlinks and ``subst`` drive mappings, so
    two spellings of one directory arrive here as one path -- except for its
    *case*, which ``resolve()`` can only repair for components that exist on
    disk, and ``backup_dir`` usually does not exist before the first backup.
    ``os.path.normcase`` finishes the job (and is a no-op on a case-sensitive
    filesystem). The separator is appended before the prefix test so that
    ``backups_old`` is not read as being inside ``backups``.
    """
    root_text = os.path.normcase(str(root)).rstrip("\\/")
    target_text = os.path.normcase(str(target)).rstrip("\\/")
    return target_text == root_text or target_text.startswith(root_text + os.sep)


def backup_dest(paths: dict, dest: Optional[str] = None) -> str:
    """Where a backup may go: ``backup_dir`` itself or a folder inside it.

    ``backup_dir`` is the one place on the read-only ``F:`` drive this tool
    writes to (spec 3.5), so an explicit ``--dest`` is confined to it rather
    than trusted. A configuration without a ``backup_dir`` raises ``ValueError``
    from :func:`require` -- ``--dest`` narrows that setting, it never replaces
    it.
    """
    root = Path(require(paths, "backup_dir")).resolve()
    if dest is None:
        return str(root)
    target = Path(dest).resolve()
    if not _within(target, root):
        raise ValueError(f"--dest must be inside the configured backup_dir ({root})")
    return str(target)


def backup_keep(paths: dict, prune: bool = True) -> Optional[int]:
    """How many backups to keep: ``paths.yaml``'s ``backup_keep``, or the default.

    ``prune=False`` (``backup --no-prune``) answers ``None``, which
    :func:`tda.core.db_backup.prune_backups` reads as "no limit".
    """
    if not prune:
        return None
    try:
        return int(paths.get("backup_keep", DEFAULT_KEEP))
    except (TypeError, ValueError):
        raise ValueError(
            f"paths.yaml's backup_keep must be a whole number of files, not "
            f"{paths.get('backup_keep')!r}"
        ) from None


def open_db(paths: dict, override: Optional[str] = None) -> Db:
    """Open the database named by ``--db`` or by ``paths.yaml``."""
    return Db(override or require(paths, "db_path"))


def read_index(paths: dict, path: Optional[str] = None) -> dict[int, DesktopIndex]:
    """Read ``index.json``; an absent file yields an empty index."""
    target = path or index_path(paths)
    if not os.path.exists(target):
        return {}
    return load_index(target)


# --------------------------------------------------------------------------- #
# desktop meta
# --------------------------------------------------------------------------- #
def merge_desktop_meta(db: Db, desktop: int, updates: dict) -> None:
    """Update part of a desktop's meta, keeping what other steps wrote.

    :meth:`tda.core.db.Db.upsert_desktop` replaces the whole row, so the stored
    meta is read back first: ``load-index`` must not drop the brand
    ``import-logs`` wrote, and vice versa. A ``None`` in ``updates`` clears its
    field.
    """
    meta = db.get_desktop(desktop) or {}
    meta.pop("id", None)
    meta.update(updates)
    db.upsert_desktop(desktop, {k: v for k, v in meta.items() if v is not None})


def add_desktop_issues(db: Db, desktop: int, key: str, lines: Iterable[str]) -> None:
    """Append issue lines to one meta list, without repeating what is there."""
    meta = db.get_desktop(desktop) or {}
    kept = list(meta.get(key) or [])
    for line in lines:
        if line not in kept:
            kept.append(line)
    merge_desktop_meta(db, desktop, {key: kept[-POSE_ISSUE_LIMIT:]})


# --------------------------------------------------------------------------- #
# load-index
# --------------------------------------------------------------------------- #
def load_index_into_db(
    db: Db,
    index: dict[int, DesktopIndex],
    desktops: Optional[set[int]] = None,
    log: Optional[Log] = None,
) -> dict:
    """Write ``index.json`` into the database: frames + pose segment 1.

    Every indexed frame becomes a ``frame`` row carrying its path, aux files and
    capture time; every key in :attr:`~tda.core.index.DesktopIndex.missing` gets
    a row with ``missing=1`` so the logical step still exists in every view
    (spec 2.2). Each view then gets one pose segment covering steps ``1..n`` --
    immediately re-cut at the ``reorient`` steps when the logs are already in
    (see :func:`split_pose_segments`).

    The frames' own ``pose_segment`` column is left unset on purpose: a frame
    resolves its segment through the step range, so re-cutting the segments
    never has to rewrite thousands of frame rows.

    One desktop is one transaction **and** one try/except: a malformed entry --
    a frame key the index cannot decode, a step count that is not a number --
    rolls its own desktop back and is listed in ``failed``, rather than ending a
    66-desktop run at number 40 and leaving the caller to guess which ones
    landed. The caller exits non-zero when ``failed`` is non-empty.
    """
    counts: dict = {"desktops": 0, "frames": 0, "missing": 0, "segments": 0,
                    "skipped": 0, "failed": []}
    for desktop in sorted(index):
        if not wanted(desktop, desktops):
            continue
        try:
            _load_one(db, desktop, index[desktop], counts, log)
            counts["desktops"] += 1  # only a desktop that really landed counts
        except Exception as exc:  # one bad entry must not end the run
            counts["failed"].append(
                f"D{desktop:02d}: {type(exc).__name__}: {exc}"
            )
            if log:
                log(f"[load-index] D{desktop:02d}: FAILED, {type(exc).__name__}: {exc}; "
                    f"nothing was written for it")
    return counts


def _merged_aux(db: Db, ff) -> dict:
    """The index's ``aux`` for one frame, with the truth table's keys kept.

    ``aux`` has two writers. The index puts there what it found on the source
    drive; :func:`tda.core.truth_inputs.frame_hw` puts back the image size it
    measured, where it measured it, and the cached copy it read
    (:data:`~tda.core.truth_inputs.TRUTH_AUX_KEYS`). Re-building the index
    replaced the whole object, so every measurement was thrown away and the next
    compile went to the source drive again -- and inferred the view's nominal
    size instead whenever that drive was detached, which silently changes the
    canvas every compiled mask is in.
    """
    held = (db.get_frame(ff.key) or {}).get("aux") or {}
    kept = {k: held[k] for k in TRUTH_AUX_KEYS if k in held}
    return dict(ff.aux or {}) | kept


def _load_one(
    db: Db, desktop: int, di: DesktopIndex, counts: dict, log: Optional[Log]
) -> None:
    """One desktop's frames and pose segments, in one transaction."""
    with db.transaction():
        merge_desktop_meta(
            db, desktop, {"index_issues": list(di.issues), "index_n_steps": di.n_steps}
        )
        frames = di.frames
        for ff in frames.values():
            db.upsert_frame(ff.key, ff.path, _merged_aux(db, ff), ff.ts,
                            {"missing": False})
        for key in di.missing:
            db.upsert_frame(key, None, {}, None, {"missing": True})
        if di.n_steps < 1:
            counts["frames"] += len(frames)
            counts["missing"] += len(di.missing)
            counts["skipped"] += 1
            if log:
                log(f"[load-index] D{desktop:02d}: no steps in the index, no pose segment")
            return
        _seed_pose_segments(db, desktop, di.n_steps)
        segments = split_pose_segments(db, desktop)
    counts["frames"] += len(frames)
    counts["missing"] += len(di.missing)
    counts["segments"] += sum(segments.values())
    if log:
        extra = f", {len(di.missing)} missing" if di.missing else ""
        cut = f", {max(segments.values())} pose segments" if segments else ""
        log(
            f"[load-index] D{desktop:02d}: {di.n_steps} steps, "
            f"{len(frames)} frames{extra}{cut}"
        )


# --------------------------------------------------------------------------- #
# pose segments
# --------------------------------------------------------------------------- #
def _seed_pose_segments(db: Db, desktop: int, n_steps: int) -> None:
    """Give every view a segment covering ``1..n_steps`` without erasing anything.

    A view with no segment yet gets segment 1. A view that already has segments
    keeps them -- only the last one's ``end_step`` follows a re-built index --
    because rewriting segment 1 wholesale would drop the chassis corners a human
    clicked (spec 2.4). :func:`split_pose_segments` then re-derives the cuts.
    """
    for view in VIEWS:
        segments = db.pose_segments(desktop, view)
        if not segments:
            db.set_pose_segment(desktop, view, 1, 1, n_steps, n_steps, None, None)
        elif segments[-1]["end_step"] != n_steps:
            db.update_pose_segment(desktop, view, segments[-1]["seg"], end_step=n_steps)


def split_pose_segments(db: Db, desktop: int) -> dict[str, int]:
    """Re-cut every view's pose segments: ``reorient`` steps ∪ that view's breaks.

    Flipping the chassis breaks the pose in all four views at once (spec 2.5),
    so a step typed ``reorient`` opens a new segment everywhere: segment ``k+1``
    starts *at* the reorient step itself.  Since v1.5 a view also has breaks of
    its **own** -- the camera was knocked, or the chassis slid, in front of one
    lens -- which is why the boundary list is computed per view from the
    ``accepted`` rows of ``pose_break``.  A reorient step that also carries a
    manual break is one boundary, not an empty segment between two.

    The function is idempotent and is called by both ``load-index`` and
    ``import-logs``: it re-derives the whole boundary list from the recorded
    steps and the stored breaks every time, so re-running either command over a
    view a human has since cut by hand leaves that cut exactly where it is.

    The rows are moved by :meth:`tda.core.db.Db.apply_recut`, one transaction
    per view -- keyframes to the segment of their own anchor, the layer order
    and both ROIs copied into every piece, the chassis corners kept only where
    the reference step still is, frozen frames queued for a re-check.  What a
    re-cut could not keep is recorded in the desktop's ``pose_issues`` meta.

    Returns ``{view: number of segments}`` for the views that have any.
    """
    out: dict[str, int] = {}
    issues: list[str] = []
    for view in VIEWS:
        result = db.recut_view(desktop, view, annotator=SPLIT_ANNOTATOR)
        if not result["ranges"]:
            continue            # a view nothing photographed has nothing to cut
        issues += _recut_issues(view, result)
        out[view] = len(result["ranges"])
    if issues:
        add_desktop_issues(db, desktop, "pose_issues", issues)
    return out


def _recut_issues(view: str, result: dict) -> list[str]:
    """The audit lines of one view's re-cut: moved references, discarded rows."""
    lines = [
        _ref_moved(view, move["seg"], move["old"], move["start"], move["end"],
                   move["ref_step"])
        for move in result["ref_moves"]
        # A segment that never had a reference step and carried nothing drawn
        # against one lost nothing, so it has nothing to report: saying "the
        # reference step moved from None to 42" only teaches the annotator to
        # skim this list.
        if move["dropped"] or move["old"].get("ref_step") is not None
    ]
    lines += [
        f"{view} pose segment {d['pose_segment']}: merged back into segment "
        f"{d['into']}, which kept its own {', '.join(sorted(d['row']))}; the "
        f"discarded values are in the op log"
        for d in result["discarded"]
    ]
    if result["uncarried"]:
        lines.append(
            f"{view}: {len(result['uncarried'])} carried keyframes had been edited "
            f"and were kept when the break was removed; check their anchors"
        )
    return lines


def _ref_moved(view: str, seg: int, old: dict, start: int, end: int, ref: int) -> str:
    """The audit line for a pose segment whose reference frame moved.

    Only ever called when something was actually lost: a reference step that
    really moved, or geometry drawn against one that is gone either way. A
    segment with no previous reference is worded as what it is, rather than as
    a move "from None".

    The two ROIs are **not** in that loss: they are rectangles in the view's own
    image rather than geometry drawn against the reference frame, so a re-cut
    copies them into every piece (:mod:`tda.core.db_pose`).
    """
    lost = " the stored corners/homography were dropped;" \
        if db_pose._has_ref_geometry(old) else ""
    moved = (
        f"gave it reference step {ref}" if old["ref_step"] is None
        else f"moved the reference step from {old['ref_step']} to {ref}"
    )
    return (
        f"{view} pose segment {seg}: a new cut moved the range to [{start}-{end}] and "
        f"{moved};{lost} shapes anchored to it need re-checking"
    )


# --------------------------------------------------------------------------- #
# import-ls order guard
# --------------------------------------------------------------------------- #
def desktops_without_steps(db: Db, desktops: Iterable[int]) -> list[int]:
    """Which of ``desktops`` have no step table yet (the pipeline-order guard)."""
    return [d for d in sorted(set(desktops)) if not db.steps(d)]


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
#: The per-desktop counters ``status`` shows (a subset of the repository's).
STATUS_TOTALS = ("steps", "actions", "instances", "events")


def status_rows(db: Db, desktops: Optional[set[int]] = None) -> list[dict]:
    """Per-desktop counters: steps, instances and the four views' frame counters.

    Each view carries ``frames`` (rows in the index, missing ones included),
    ``missing``, ``keyframes`` (drawn or imported shapes) and ``verified``
    (frames a human signed off).
    """
    totals = {name: db.count_per_desktop(name) for name in STATUS_TOTALS}
    per_view = {name: db.count_per_view(name) for name in VIEW_COUNTERS}
    ids = set(db.desktop_ids()) | set(totals["steps"]) | {d for d, _v in per_view["frames"]}
    out = []
    for desktop in sorted(d for d in ids if wanted(d, desktops)):
        meta = db.get_desktop(desktop) or {}
        out.append({
            "desktop": desktop,
            "brand": meta.get("brand_model_raw") or meta.get("brand") or "",
            **{name: totals[name].get(desktop, 0) for name in STATUS_TOTALS},
            "views": {
                view: {name: per_view[name].get((desktop, view), 0) for name in VIEW_COUNTERS}
                for view in VIEWS
            },
        })
    return out
