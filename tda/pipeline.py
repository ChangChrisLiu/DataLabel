"""The data-pipeline steps behind ``python -m tda.cli`` (spec 2.2-2.4, 3.5).

This module holds the work; :mod:`tda.cli` holds the argument parsing and the
printing, so every step here is callable from a test or from the GUI without
going through ``argparse``.

The pipeline runs in one order, because each step needs what the previous one
wrote::

    build-index -> load-index -> import-logs -> import-ls

* :func:`load_index_into_db`  -- ``index.json`` -> ``frame`` rows + pose segment 1.
* :func:`import_logs_into_db` -- the Drive sheets -> steps, actions, instances, events.
* :func:`split_pose_segments` -- cut a view's pose segments at the ``reorient`` steps.
* :func:`status_rows`         -- the per-desktop counters the ``status`` command prints.

Nothing here writes outside ``D:`` except :meth:`tda.core.db.Db.backup`, which
the ``backup`` command calls directly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from tda.core.db import Db
from tda.core.index import DesktopIndex, load_index
from tda.core.log_report import _expected_steps
from tda.core.logs import LogImport, import_log, iter_desktop_csvs, read_desktop_csv
from tda.core.model import VIEWS, StepRec, StepType
from tda.core.states import events_from_actions
from tda.core.taxonomy import Taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS_PATH = "configs/paths.yaml"
INDEX_NAME = "index.json"
INDEX_REPORT_NAME = "index_report.md"
LOGS_REPORT_NAME = "import_logs_issues.md"
LS_SUMMARY_NAME = "ls_import_summary.json"
LS_EXPORT_NAME = "labelstudio/humansignal_annotated_projects_export.json"
DRIVE_SUBDIR = "drive"
ALL_DESKTOPS = range(1, 67)

#: ``print`` by default; tests and the GUI pass their own sink.
Log = Callable[[str], None]

__all__ = [
    "DesktopRun", "LogsRun", "brand_model", "chassis_type", "desktops_without_steps",
    "drive_dir", "expected_steps", "import_logs_into_db", "index_path",
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


def _wanted(desktop: int, desktops: Optional[set[int]]) -> bool:
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


def open_db(paths: dict, override: Optional[str] = None) -> Db:
    """Open the database named by ``--db`` or by ``paths.yaml``."""
    return Db(override or require(paths, "db_path"))


def require(paths: dict, key: str) -> str:
    """One path from ``paths.yaml``, with a readable error when it is missing."""
    value = paths.get(key)
    if not value:
        raise ValueError(f"paths.yaml defines no {key}")
    return str(value)


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


#: Lower-cased first token of ``Desktop Brand`` -> the canonical brand.
BRANDS = {
    "hp": "HP", "hewlett-packard": "HP", "compaq": "HP", "compoq": "HP", "compac": "HP",
    "dell": "Dell", "delle": "Dell", "lenovo": "Lenovo", "apple": "Apple",
    "acer": "Acer", "asus": "ASUS", "ibm": "IBM", "gateway": "Gateway",
}
#: Model families that name their brand implicitly (the sheets often drop "Dell").
IMPLIED_BRANDS = {
    "optiplex": "Dell", "precision": "Dell", "inspiron": "Dell", "vostro": "Dell",
    "dimension": "Dell", "elitedesk": "HP", "prodesk": "HP", "pavilion": "HP",
    "thinkcentre": "Lenovo",
}
#: Chassis-type keywords, most specific first (the sheets spell SFF many ways).
CHASSIS_TYPES = (
    ("small form", "sff"), ("sff", "sff"), ("ssf", "sff"), ("sf", "sff"),
    ("cmt", "cmt"), ("twr", "twr"), ("tower", "twr"), ("mt", "mt"), ("dt", "dt"),
)


def brand_model(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Split ``Desktop Brand`` into ``(brand, model_family)``.

    ``"HP EliteDesk 800 G2 TWR"`` -> ``("HP", "EliteDesk 800 G2 TWR")`` and
    ``"Optiplex 7020"`` -> ``("Dell", "Optiplex 7020")``; an unrecognised name
    keeps its whole text as the model and no brand.
    """
    text = " ".join(str(raw or "").split())
    if not text:
        return None, None
    tokens = text.split(" ")
    head = tokens[0].lower()
    if head in BRANDS:
        rest = " ".join(tokens[1:]).strip()
        return BRANDS[head], rest or None
    return IMPLIED_BRANDS.get(head), text


def chassis_type(raw: str) -> Optional[str]:
    """Guess the form factor (``sff``/``mt``/``twr``/``cmt``/``dt``) from the name."""
    words = str(raw or "").lower().replace("-", " ").split()
    text = " ".join(words)
    for needle, value in CHASSIS_TYPES:
        if " " in needle:
            if needle in text:
                return value
        elif needle in words:
            return value
    return None


def desktop_fields(meta: dict) -> dict:
    """The ``desktop`` columns (plus meta_json extras) one sheet's metadata fills."""
    raw = str(meta.get("brand_model_raw") or "").strip()
    brand, model = brand_model(raw)
    return {
        "brand": brand,
        "model_family": model,
        "chassis_type": chassis_type(raw),
        "size": (meta.get("size_raw") or None),
        "date": (meta.get("collection_date") or None),
        "notes": (meta.get("notes") or None),
        "brand_model_raw": raw or None,
        "sheet_desktop_id": meta.get("desktop_id"),
    }


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
    """
    counts = {"desktops": 0, "frames": 0, "missing": 0, "segments": 0, "skipped": 0}
    for desktop in sorted(index):
        if not _wanted(desktop, desktops):
            continue
        di = index[desktop]
        counts["desktops"] += 1
        merge_desktop_meta(
            db, desktop, {"index_issues": list(di.issues), "index_n_steps": di.n_steps}
        )
        for ff in di.frames.values():
            db.upsert_frame(ff.key, ff.path, ff.aux, ff.ts, {"missing": False})
        for key in di.missing:
            db.upsert_frame(key, None, {}, None, {"missing": True})
        counts["frames"] += len(di.frames)
        counts["missing"] += len(di.missing)
        if di.n_steps < 1:
            counts["skipped"] += 1
            if log:
                log(f"[load-index] D{desktop:02d}: no steps in the index, no pose segment")
            continue
        _seed_pose_segments(db, desktop, di.n_steps)
        segments = split_pose_segments(db, desktop)
        counts["segments"] += sum(segments.values())
        if log:
            extra = f", {len(di.missing)} missing" if di.missing else ""
            cut = f", {max(segments.values())} pose segments" if segments else ""
            log(
                f"[load-index] D{desktop:02d}: {di.n_steps} steps, "
                f"{len(di.frames)} frames{extra}{cut}"
            )
    return counts


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
        rows = db.conn.execute(
            "SELECT seg, end_step FROM pose_segment WHERE desktop=? AND view=? ORDER BY seg",
            (desktop, view),
        ).fetchall()
        if not rows:
            db.set_pose_segment(desktop, view, 1, 1, n_steps, n_steps, None, None)
        elif rows[-1]["end_step"] != n_steps:
            with db.conn:
                db.conn.execute(
                    "UPDATE pose_segment SET end_step=? WHERE desktop=? AND view=? AND seg=?",
                    (n_steps, desktop, view, rows[-1]["seg"]),
                )


def _segment_ranges(n_steps: int, boundaries: Iterable[int]) -> list[tuple[int, int]]:
    """Cut ``1..n_steps`` at every boundary; segment k+1 starts at its boundary."""
    starts = [1] + [b for b in sorted(set(boundaries)) if 1 < b <= n_steps]
    return [
        (start, (starts[i + 1] - 1) if i + 1 < len(starts) else n_steps)
        for i, start in enumerate(starts)
    ]


def split_pose_segments(db: Db, desktop: int) -> dict[str, int]:
    """Re-cut every view's pose segments at the desktop's ``reorient`` steps.

    Flipping the chassis breaks the pose (spec 2.5), so the steps typed
    ``reorient`` open a new segment: segment ``k+1`` starts *at* the reorient
    step itself. The function is idempotent and is called by both ``load-index``
    and ``import-logs``, because either order leaves the same segments behind:
    it re-derives the whole segment list from the recorded steps every time, and
    keeps a segment's ``corners``/``ref_step`` when its start step is unchanged
    (a human may already have clicked the chassis corners).

    Returns ``{view: number of segments}`` for the views that have any.
    """
    reorients = [
        s.step for s in db.steps(desktop) if s.step_type == StepType.REORIENT.value
    ]
    out: dict[str, int] = {}
    for view in VIEWS:
        rows = db.conn.execute(
            "SELECT * FROM pose_segment WHERE desktop=? AND view=? ORDER BY seg",
            (desktop, view),
        ).fetchall()
        ends = [r["end_step"] for r in rows if r["end_step"] is not None]
        if not ends:
            continue
        n_steps = max(ends)
        existing = {r["seg"]: r for r in rows}
        ranges = _segment_ranges(n_steps, reorients)
        for seg, (start, end) in enumerate(ranges, start=1):
            old = existing.get(seg)
            same = old is not None and old["start_step"] == start
            ref = old["ref_step"] if same and old["ref_step"] is not None else end
            db.set_pose_segment(
                desktop, view, seg, start, end,
                ref_step=ref if start <= ref <= end else end,
                corners=_json(old, "corners_json") if same else None,
                homography=_json(old, "homography_json") if same else None,
            )
        if len(existing) > len(ranges):
            with db.conn:
                db.conn.execute(
                    "DELETE FROM pose_segment WHERE desktop=? AND view=? AND seg>?",
                    (desktop, view, len(ranges)),
                )
        out[view] = len(ranges)
    return out


def _json(row, column: str):
    """Decode one JSON column of a ``sqlite3.Row`` that may be ``None``."""
    if row is None or row[column] is None:
        return None
    return json.loads(row[column])


# --------------------------------------------------------------------------- #
# import-logs
# --------------------------------------------------------------------------- #
@dataclass
class DesktopRun:
    """What ``import-logs`` did for one desktop."""

    desktop: int
    source: str
    status: str  # imported | skipped | failed
    steps: int = 0
    actions: int = 0
    instances: int = 0
    events: int = 0
    durations: int = 0
    brand: str = ""
    issues: list[str] = field(default_factory=list)


@dataclass
class LogsRun:
    """The whole ``import-logs`` run."""

    directory: str
    runs: list[DesktopRun] = field(default_factory=list)
    report_path: str = ""

    @property
    def imported(self) -> list[DesktopRun]:
        return [r for r in self.runs if r.status == "imported"]

    @property
    def skipped(self) -> list[DesktopRun]:
        return [r for r in self.runs if r.status == "skipped"]


def _apply_index(
    steps: list[StepRec], di: Optional[DesktopIndex]
) -> tuple[int, list[str]]:
    """Fill ``duration_s`` from the index and cross-check it against the sheet.

    The duration of step ``k`` is ``ts_k - ts_(k-1)`` of the OAK cam1 captures
    (spec 2.2): the time between the two photographs that bracket the
    operation. Step 1 has no predecessor, and a step whose cam1 frame is missing
    (or whose timestamps run backwards) keeps ``None`` rather than a guess.

    Two disagreements between the sheet and the index are reported for the
    manual list of spec 2.2: a differing step count, and a ``reorient`` row past
    the last indexed step, which no view photographed and which therefore opens
    no pose segment.
    """
    if di is None:
        return 0, []
    times: dict[int, datetime] = {}
    for key, ff in di.frames.items():
        if key.view == "oak1" and ff.ts:
            try:
                times[key.step] = datetime.fromisoformat(ff.ts)
            except ValueError:
                pass
    filled, issues = 0, []
    for step in steps:
        before, after = times.get(step.step - 1), times.get(step.step)
        if before is None or after is None:
            continue
        seconds = (after - before).total_seconds()
        if seconds < 0:
            issues.append(
                f"D{step.desktop:02d} step {step.step}: oak1 capture times run backwards "
                f"({seconds:.0f}s); duration left unset"
            )
            continue
        step.duration_s = seconds
        filled += 1
    if len(steps) != di.n_steps:
        issues.append(
            f"D{di.desktop:02d}: the sheet has {len(steps)} steps but the index has "
            f"{di.n_steps} - needs manual confirmation (spec 2.2)"
        )
    beyond = [
        s.step for s in steps
        if s.step_type == StepType.REORIENT.value and s.step > di.n_steps
    ]
    if beyond:
        issues.append(
            f"D{di.desktop:02d}: reorient steps {beyond} are past the index's last step "
            f"({di.n_steps}), so they open no pose segment - no view photographed them"
        )
    return filled, issues


def _write_import(db: Db, li: LogImport, tax: Taxonomy) -> int:
    """Write one desktop's import into the database; returns the event count."""
    merge_desktop_meta(db, li.desktop, desktop_fields(li.meta))
    db.replace_steps(li.desktop, li.steps, li.actions)
    for inst in li.instances.values():
        db.upsert_instance(inst)
    events = events_from_actions(li.instances, li.actions, tax)
    db.replace_events(li.desktop, events, auto_only=True)
    split_pose_segments(db, li.desktop)
    return len(events)


def import_logs_into_db(
    db: Db,
    directory: str,
    tax: Taxonomy,
    index: Optional[dict[int, DesktopIndex]] = None,
    desktops: Optional[set[int]] = None,
    force: bool = False,
    log: Optional[Log] = None,
) -> LogsRun:
    """Import the exported Drive sheets into steps, actions, instances and events.

    A desktop that already has steps is **skipped** unless ``force`` is given:
    the step table is where the annotator resolves the ``?`` targets and the
    compound rows, and :meth:`tda.core.db.Db.replace_steps` would throw that
    work away. Auto state events are replaced (``auto_only=True``), so
    hand-written ones survive either way.
    """
    index = index or {}
    run = LogsRun(directory=str(directory))
    for desktop, path in iter_desktop_csvs(directory):
        if not _wanted(desktop, desktops):
            continue
        if db.steps(desktop) and not force:
            run.runs.append(DesktopRun(desktop, str(path), "skipped"))
            if log:
                log(f"[import-logs] D{desktop:02d}: skipped, the database already has "
                    f"steps (use --force to overwrite)")
            continue
        rows, meta = read_desktop_csv(path)
        li = import_log(desktop, rows, meta, tax)
        filled, issues = _apply_index(li.steps, index.get(desktop))
        sheet_id = meta.get("desktop_id")
        if sheet_id is not None and int(sheet_id) != desktop:
            issues.append(
                f"D{desktop:02d}: the sheet's own Desktop ID is {sheet_id}; "
                f"kept the file's number"
            )
        events = _write_import(db, li, tax)
        run.runs.append(DesktopRun(
            desktop=desktop, source=str(path), status="imported", steps=len(li.steps),
            actions=len(li.actions), instances=len(li.instances), events=events,
            durations=filled, brand=str(li.meta.get("brand_model_raw") or ""),
            issues=list(li.issues) + issues,
        ))
        if log:
            log(f"[import-logs] D{desktop:02d}: {len(li.steps)} steps, {len(li.actions)} "
                f"actions, {len(li.instances)} instances, {events} events, "
                f"{filled} durations, {len(li.issues) + len(issues)} issues")
    return run


def expected_steps(directory: str) -> dict[int, int]:
    """``desktop -> n_logged_steps`` from the export's summary table, if present."""
    return _expected_steps(Path(directory) / "desktop_meta.csv")


# --------------------------------------------------------------------------- #
# import-ls order guard
# --------------------------------------------------------------------------- #
def desktops_without_steps(db: Db, desktops: Iterable[int]) -> list[int]:
    """Which of ``desktops`` have no step table yet (the pipeline-order guard)."""
    return [d for d in sorted(set(desktops)) if not db.steps(d)]


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def _counts(db: Db, sql: str, *, per_view: bool) -> dict:
    """Run a ``GROUP BY`` counting query into ``{desktop: n}``/``{(desktop, view): n}``."""
    rows = db.conn.execute(sql).fetchall()
    if per_view:
        return {(r["desktop"], r["view"]): r["n"] for r in rows}
    return {r["desktop"]: r["n"] for r in rows}


#: ``status`` counter -> the table whose rows are counted per desktop.
DESKTOP_COUNTS = {
    "steps": "step", "actions": "action", "instances": "instance", "events": "state_event",
}
#: ``status`` counter -> the ``FROM``/``WHERE`` clause counted per (desktop, view).
VIEW_COUNTS = {
    "frames": "FROM frame",
    "missing": "FROM frame WHERE missing=1",
    "keyframes": "FROM shape_keyframe",
    "verified": "FROM frame WHERE review_status='verified'",
}


def status_rows(db: Db, desktops: Optional[set[int]] = None) -> list[dict]:
    """Per-desktop counters: steps, instances and the four views' frame counters.

    Each view carries ``frames`` (rows in the index, missing ones included),
    ``missing``, ``keyframes`` (drawn or imported shapes) and ``verified``
    (frames a human signed off).
    """
    totals = {
        name: _counts(db, f"SELECT desktop, COUNT(*) AS n FROM {table} GROUP BY desktop",
                      per_view=False)
        for name, table in DESKTOP_COUNTS.items()
    }
    per_view = {
        name: _counts(db, f"SELECT desktop, view, COUNT(*) AS n {source} "
                          f"GROUP BY desktop, view", per_view=True)
        for name, source in VIEW_COUNTS.items()
    }
    ids = {r["id"] for r in db.conn.execute("SELECT id FROM desktop").fetchall()}
    ids |= set(totals["steps"]) | {d for d, _v in per_view["frames"]}
    out = []
    for desktop in sorted(d for d in ids if _wanted(d, desktops)):
        meta = db.get_desktop(desktop) or {}
        out.append({
            "desktop": desktop,
            "brand": meta.get("brand_model_raw") or meta.get("brand") or "",
            **{name: totals[name].get(desktop, 0) for name in DESKTOP_COUNTS},
            "views": {
                view: {name: per_view[name].get((desktop, view), 0) for name in VIEW_COUNTS}
                for view in VIEWS
            },
        })
    return out


def pose_segments(db: Db, desktop: int) -> dict[str, list[dict]]:
    """The stored pose segments of one desktop, per view (used by ``status``)."""
    rows = db.conn.execute(
        "SELECT * FROM pose_segment WHERE desktop=? ORDER BY view, seg", (desktop,)
    ).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["view"], []).append(
            {"seg": r["seg"], "start": r["start_step"], "end": r["end_step"],
             "ref": r["ref_step"], "corners": r["corners_json"] is not None}
        )
    return out


def step_type_counts(db: Db, desktop: int) -> dict[str, int]:
    """``step_type -> count`` for one desktop."""
    rows = db.conn.execute(
        "SELECT step_type, COUNT(*) AS n FROM step WHERE desktop=? GROUP BY step_type "
        "ORDER BY step_type", (desktop,)
    ).fetchall()
    return {r["step_type"]: r["n"] for r in rows}
