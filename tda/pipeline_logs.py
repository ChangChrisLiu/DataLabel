"""The Drive-sheet half of the pipeline: sheets -> steps, actions, instances, events.

Second step of the order :mod:`tda.pipeline` documents. One exported sheet per
desktop becomes the step table the annotator works from (spec 2.3), the
instance identities behind it (3.2) and the automatic state-event log (6.3);
the frame index supplies the step durations and catches the two places where
sheet and cameras disagree.

The import is deliberately hard to lose work to. A desktop that already has
steps is skipped unless ``force`` is given, and even a forced run
* says per desktop what it is about to replace,
* carries the imported ``LS:`` note lines over to the new steps, and
* leaves the hand-written (``auto=False``) state events alone.

One desktop is one transaction, and a sheet that cannot be read is recorded as
``failed`` and skipped rather than ending the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from tda.core.db import Db
from tda.core.graph_rules import infer_relational_fields, unresolved_relations
from tda.core.index import DesktopIndex
from tda.core.log_report import _expected_steps
from tda.core.logs import LogImport, import_log, iter_desktop_csvs, read_desktop_csv
from tda.core.model import StepRec, StepType
from tda.core.states import events_from_actions
from tda.core.taxonomy import Taxonomy
from tda.pipeline import Log, merge_desktop_meta, split_pose_segments, wanted

#: Mirrors :data:`tda.core.ls_import.NOTES_PREFIX`; repeated so importing the
#: pipeline does not pull in numpy/pycocotools. ``tests/test_cli.py`` pins them
#: together.
LS_NOTES_PREFIX = "LS: "
#: A step whose two bracketing photographs are further apart than this is
#: reported: it is more likely a break than a 30-minute operation.
MAX_STEP_SECONDS = 1800.0
#: Heading of the import report's per-desktop list of inferred relational
#: fields. They are guesses, and every one of them is an S1 question.
INFERRED_HEADING = "inferred relational fields (heuristic - confirm in S1)"

__all__ = [
    "DesktopRun", "INFERRED_HEADING", "LogsRun", "brand_model", "chassis_type",
    "desktop_fields", "expected_steps", "import_logs_into_db", "inferred_section",
]


# --------------------------------------------------------------------------- #
# desktop metadata
# --------------------------------------------------------------------------- #
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
# run records
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
    ls_notes: int = 0  # steps whose Label Studio notes were carried over
    brand: str = ""
    issues: list[str] = field(default_factory=list)
    #: ``"<key>.<field> = <value>"`` per relational field the heuristic filled.
    fills: list[str] = field(default_factory=list)
    #: ``"unresolved: ..."`` per reference it deliberately did not guess at.
    unresolved: list[str] = field(default_factory=list)


def inferred_section(run: DesktopRun) -> list[str]:
    """The report block listing what the heuristic filled for one desktop.

    Empty when it filled nothing, so a desktop whose sheet already said
    everything adds no noise. Kept here rather than in :mod:`tda.cli` so the
    heading and the bullet format live next to the run record that carries them.
    """
    if not (run.fills or run.unresolved):
        return []
    lines = [f"**{INFERRED_HEADING}**", ""]
    lines.extend(f"- {text}" for text in run.fills)
    lines.extend(f"- {text}" for text in run.unresolved)
    lines.append("")
    return lines


@dataclass
class LogsRun:
    """The whole ``import-logs`` run."""

    directory: str
    runs: list[DesktopRun] = field(default_factory=list)
    report_path: str = ""

    def _with(self, status: str) -> list[DesktopRun]:
        return [r for r in self.runs if r.status == status]

    @property
    def imported(self) -> list[DesktopRun]:
        return self._with("imported")

    @property
    def skipped(self) -> list[DesktopRun]:
        return self._with("skipped")

    @property
    def failed(self) -> list[DesktopRun]:
        return self._with("failed")

    @property
    def with_ls_notes(self) -> list[int]:
        """Desktops whose Label Studio notes a forced re-import carried over."""
        return [r.desktop for r in self.imported if r.ls_notes]


# --------------------------------------------------------------------------- #
# the index's contribution
# --------------------------------------------------------------------------- #
def _apply_index(
    steps: list[StepRec], di: Optional[DesktopIndex]
) -> tuple[int, list[str]]:
    """Fill ``duration_s`` from the index and cross-check it against the sheet.

    The duration of step ``k`` is ``ts_k - ts_(k-1)`` of the OAK cam1 captures
    (spec 2.2): the time between the two photographs that bracket the
    operation. Step 1 has no predecessor, and a step whose cam1 frame is missing
    (or whose timestamps run backwards) keeps ``None`` rather than a guess.

    Three disagreements between the sheet and the cameras are reported for the
    manual list of spec 2.2: a differing step count, a ``reorient`` row past the
    last indexed step (no view photographed it, so it opens no pose segment),
    and a gap longer than :data:`MAX_STEP_SECONDS`, which is recorded but is
    more likely a break than an operation.
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
        if seconds > MAX_STEP_SECONDS:
            issues.append(
                f"D{step.desktop:02d} step {step.step}: {seconds / 60:.0f} min between the "
                f"oak1 captures of steps {step.step - 1} and {step.step} - a break rather "
                f"than the operation time? (recorded as measured)"
            )
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


# --------------------------------------------------------------------------- #
# writing one desktop
# --------------------------------------------------------------------------- #
def carry_ls_notes(steps: list[StepRec], previous: list[StepRec]) -> int:
    """Copy the ``LS:`` note lines of the stored steps onto the freshly parsed ones.

    The Label Studio import folds the annotators' form fields into
    ``Step.notes`` (:mod:`tda.core.ls_import`), and
    :meth:`~tda.core.db.Db.replace_steps` would drop them on a forced
    re-import. They are matched by step number and re-appended, so re-importing
    the sheets costs nothing that a second ``import-ls`` would have to rebuild.
    Returns how many steps kept a line.
    """
    by_step = {s.step: s for s in steps}
    kept = 0
    for old in previous:
        lines = [
            line for line in (old.notes or "").splitlines()
            if line.startswith(LS_NOTES_PREFIX)
        ]
        step = by_step.get(old.step)
        if not lines or step is None:
            continue
        base = (step.notes or "").strip()
        step.notes = "\n".join(([base] if base else []) + lines)
        kept += 1
    return kept


def _write_import(
    db: Db, li: LogImport, tax: Taxonomy, previous: list[StepRec]
) -> tuple[int, int, list[str], list[str]]:
    """Write one desktop's import atomically.

    Returns ``(events, steps with LS notes, fills, unresolved)``.

    The relational heuristic of spec 7.3 runs here, *before* the instances are
    written and inside the same transaction: ``logs.py`` leaves ``fastens``,
    ``parent``/``attached`` and a latch's ``of`` empty and may put a bare class
    name in ``socket_host``, and the state machine reads those the moment the
    annotator opens the desktop -- a captive cooler screw with no ``parent``
    stays in the chassis after its cooler is gone (spec 3.3). The freshly
    parsed actions go in with them, because the step order is what breaks the
    tie on a desktop that lists two coolers. Filling them
    first also means ``events_from_actions`` below already emits the cascade,
    so the stored automatic log and the one
    :func:`tda.core.truth_inputs.events_of` re-derives on every read agree.
    """
    with db.transaction():
        kept = carry_ls_notes(li.steps, previous)
        merge_desktop_meta(db, li.desktop, desktop_fields(li.meta))
        db.replace_steps(li.desktop, li.steps, li.actions)
        fills = infer_relational_fields(li.instances, tax, li.actions)
        for inst in li.instances.values():
            db.upsert_instance(inst)
        events = events_from_actions(li.instances, li.actions, tax)
        db.replace_events(li.desktop, events, auto_only=True)
        split_pose_segments(db, li.desktop)
    return len(events), kept, fills, unresolved_relations(li.instances, tax, li.actions)


def _force_warning(db: Db, desktop: int, previous: list[StepRec]) -> str:
    """What a ``--force`` run is about to overwrite, in one line."""
    ls_steps = db.steps_with_note_prefix(desktop, LS_NOTES_PREFIX)
    notes = (
        f"{len(ls_steps)} steps carry Label Studio notes (carried over); "
        if ls_steps else ""
    )
    return (
        f"[import-logs] D{desktop:02d}: --force, replacing {len(previous)} steps / "
        f"{len(db.actions(desktop))} actions; {notes}every other manual edit to the step "
        f"table is lost"
    )


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
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
    work away. A sheet that cannot be read is recorded as ``failed`` and the run
    carries on to the next desktop.
    """
    index = index or {}
    run = LogsRun(directory=str(directory))
    for desktop, path in iter_desktop_csvs(directory):
        if not wanted(desktop, desktops):
            continue
        previous = db.steps(desktop)
        if previous and not force:
            run.runs.append(DesktopRun(desktop, str(path), "skipped"))
            if log:
                log(f"[import-logs] D{desktop:02d}: skipped, the database already has "
                    f"steps (use --force to overwrite)")
            continue
        if previous and log:
            log(_force_warning(db, desktop, previous))
        try:
            run.runs.append(_import_one(db, desktop, path, tax, index, previous, log))
        except Exception as exc:  # one unreadable sheet must not end the run
            run.runs.append(DesktopRun(
                desktop, str(path), "failed",
                issues=[f"D{desktop:02d}: {type(exc).__name__}: {exc}"],
            ))
            if log:
                log(f"[import-logs] D{desktop:02d}: FAILED, {type(exc).__name__}: {exc}")
    return run


def _import_one(
    db: Db, desktop: int, path, tax: Taxonomy,
    index: dict[int, DesktopIndex], previous: list[StepRec], log: Optional[Log],
) -> DesktopRun:
    """Read and write one desktop; raises if the sheet cannot be parsed."""
    rows, meta = read_desktop_csv(path)
    li = import_log(desktop, rows, meta, tax)
    filled, issues = _apply_index(li.steps, index.get(desktop))
    sheet_id = meta.get("desktop_id")
    if sheet_id is not None and int(sheet_id) != desktop:
        issues.append(
            f"D{desktop:02d}: the sheet's own Desktop ID is {sheet_id}; "
            f"kept the file's number"
        )
    events, kept, fills, unresolved = _write_import(db, li, tax, previous)
    if log:
        log(f"[import-logs] D{desktop:02d}: {len(li.steps)} steps, {len(li.actions)} "
            f"actions, {len(li.instances)} instances, {events} events, "
            f"{filled} durations, {len(fills)} inferred relational fields, "
            f"{len(li.issues) + len(issues)} issues")
    return DesktopRun(
        desktop=desktop, source=str(path), status="imported", steps=len(li.steps),
        actions=len(li.actions), instances=len(li.instances), events=events,
        durations=filled, ls_notes=kept,
        brand=str(li.meta.get("brand_model_raw") or ""),
        issues=list(li.issues) + issues, fills=fills, unresolved=unresolved,
    )


def expected_steps(directory: str) -> dict[int, int]:
    """``desktop -> n_logged_steps`` from the export's summary table, if present."""
    return _expected_steps(Path(directory) / "desktop_meta.csv")
