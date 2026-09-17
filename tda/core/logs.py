"""Drive log import: one exported sheet per desktop -> steps, actions, instances.

Spec sections 2.3 (step-table import) and 3.2 (instance identity).

The annotators kept one Google Sheet per desktop; the export used here is the
first sheet (``Sequence Detail Information``), optionally followed by the small
``Desktop Metadata`` table either in the same file or in a ``*_meta.csv``
sidecar.  This module turns those rows into the drafts the annotation UI
pre-fills:

* one :class:`~tda.core.model.StepRec` per non-empty row, numbered by **row
  order** -- the sheet's own ``Sequence Number`` is advisory only and its gaps
  and restarts land in :attr:`LogImport.issues`;
* zero or one :class:`~tda.core.model.ActionRec` per step (markers draft none,
  compound rows draft a flagged placeholder);
* one :class:`~tda.core.model.InstanceRec` per operated target plus the
  implicit ``chassis``, with ordinals in order of first operation. Every row
  that names a target is a new instance unless the strict reuse test in
  ``_Importer._reuse`` says otherwise, so the annotators' restarted numbering
  never merges two physical parts.

Nothing here guesses what a human must decide: unresolved targets keep a ``?``
in the key, and every one of them is reported in ``issues`` for stage S1.

The batch run and its report live in :mod:`tda.core.log_report`::

    python -m tda.core.logs --all D:/DataSet/raw_logs/drive
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .model import ActionRec, InstanceRec, StepRec
from .taxonomy import (
    FALLBACK_RULE,
    ParsedTarget,
    Taxonomy,
    map_tool,
    parse_raw_name,
)

HEADER_CELL = "Sequence Number"
NAME_COLUMN = "Sequence Name"
NEST_COLUMN = "Target Nest Group"
TOOL_COLUMN = "Tool Utility"
NOTES_COLUMN = "Notes"

#: ``Desktop Metadata`` row labels -> ``LogImport.meta`` keys.
META_KEYS = {
    "Desktop ID": "desktop_id",
    "Desktop Brand": "brand_model_raw",
    "Desktop Size": "size_raw",
    "Collection Date": "collection_date",
}

UNRESOLVED = "?"
UNKNOWN_TOOL = "unknown"
CHASSIS_KEY = "chassis"

#: Step types that draft no action at all (spec 2.3).
NO_ACTION_TYPES = frozenset({"initial", "dupli", "ignore"})
HINT_TYPES = frozenset({"reorient", "ignore", "auxiliary"})

_DEFAULT_VERBS = {
    "screw": "unscrew",
    "connector": "disconnect",
    "cable_clip": "release",
    "chassis": "reorient",
}
_OPENABLE = frozenset(
    {
        "ram_latch",
        "cpu_socket_lever",
        "psu_latch",
        "drive_latch",
        "card_latch",
        "cooler_latch",
        "cover",
    }
)

#: Connector ``cable_owner`` values whose socket sits on the device itself;
#: every other owner (psu, case_fan, cpu_fan, front_panel) plugs into the
#: motherboard.
_DEVICE_SOCKET_HOSTS = {
    "storage_drive": "storage_drive",
    "optical_drive": "optical_drive",
}
DEFAULT_SOCKET_HOST = "motherboard"

#: Attributes the importer derives itself; they never take part in the
#: instance-identity comparison.
_DERIVED_ATTRS = frozenset({"instance_nos", "sheet_no", "captive", "captive_source"})

#: Brands whose CPU-cooler screws are captive (they stay in the bracket).
_CAPTIVE_BRANDS = ("dell", "optiplex")
_CAPTIVE_ROLE = "cpu_cooler"

_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_AND = re.compile(r"\band\b", re.IGNORECASE)
_MOTHERBOARD = re.compile(r"motherboard|main\s*board|mobo", re.IGNORECASE)
_DESKTOP_FILE = re.compile(r"^desktop_(\d+)(?:_(\w+))?\.csv$", re.IGNORECASE)
#: Desktop 28 was exported three times; the other two are copies of D27.
_DESKTOP_28_KEEP = "1E9qmg"


@dataclass
class LogImport:
    """Everything one desktop's sheet yields."""

    desktop: int
    meta: dict
    steps: list[StepRec] = field(default_factory=list)
    actions: list[ActionRec] = field(default_factory=list)
    instances: dict[str, InstanceRec] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Reading the exported sheet
# --------------------------------------------------------------------------- #
def _find_header(rows: list[list[str]]) -> int:
    for i, row in enumerate(rows):
        if HEADER_CELL in row:
            return i
    raise ValueError(f"no {HEADER_CELL!r} header row in the export")


def _split_tables(body: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """Split the rows below the header into (step rows, metadata rows)."""
    for i, row in enumerate(body):
        if row and row[0] in META_KEYS:
            return body[:i], body[i:]
    return body, []


def _first_value(cells: list[str]) -> str:
    return next((c for c in cells if c), "")


def _read_meta_rows(rows: list[list[str]]) -> dict:
    """Read the two-column ``Desktop Metadata`` table."""
    out: dict = {"brand_model_raw": "", "size_raw": "", "collection_date": "", "notes": ""}
    notes: list[str] = []
    for row in rows:
        if not row:
            continue
        label, value = row[0], _first_value(row[1:])
        if label in META_KEYS:
            out[META_KEYS[label]] = value
        elif not label and value:
            notes.append(value)
    out["notes"] = " | ".join(notes)
    date = _DATE.search(out.get("collection_date") or "")
    out["collection_date"] = date.group(1) if date else (out.get("collection_date") or "")
    raw_id = str(out.get("desktop_id") or "").strip()
    out["desktop_id"] = int(float(raw_id)) if raw_id.replace(".", "", 1).isdigit() else None
    return out


def read_desktop_csv(path: str | Path) -> tuple[list[dict], dict]:
    """Read one exported log.

    Returns ``(rows, meta)`` where ``rows`` are the step rows that carry a
    ``Sequence Name`` (the number-only filler rows at the bottom of every sheet
    are dropped), each keyed by its original column name, and ``meta`` holds
    ``brand_model_raw``, ``size_raw``, ``collection_date``, ``notes`` and
    ``desktop_id``.  The metadata table is read from the same file when the
    export contains it, otherwise from the ``*_meta.csv`` sidecar.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        raw = [[(cell or "").strip() for cell in row] for row in csv.reader(f)]
    header_idx = _find_header(raw)
    columns = [c for c in raw[header_idx]]
    step_rows, meta_rows = _split_tables(raw[header_idx + 1 :])

    rows: list[dict] = []
    for cells in step_rows:
        row = {name: (cells[i] if i < len(cells) else "") for i, name in enumerate(columns) if name}
        if row.get(NAME_COLUMN):
            rows.append(row)

    if not meta_rows:
        sidecar = path.with_name(f"{path.stem}_meta{path.suffix}")
        if sidecar.exists():
            with open(sidecar, newline="", encoding="utf-8-sig") as f:
                meta_rows = [[(c or "").strip() for c in r] for r in csv.reader(f)]
    return rows, _read_meta_rows(meta_rows)


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
def instance_key(cls: str, attrs: dict, ordinal: int) -> str:
    """Build an instance key: class, discriminator, 2-digit ordinal.

    The discriminator is ``attrs["role"]`` (screws) or ``attrs["kind"]``
    (connectors, drives, coolers, cards); ``chassis`` is unique per desktop and
    therefore carries no ordinal.
    """
    if cls == CHASSIS_KEY:
        return CHASSIS_KEY
    parts = [cls]
    disc = attrs.get("role") or attrs.get("kind") or ""
    if disc:
        parts.append(str(disc))
    parts.append(f"{ordinal:02d}")
    return ".".join(parts)


def _placeholder_key(cls: str, attrs: dict) -> str:
    """The key of a target a human still has to resolve."""
    if not cls:
        return UNRESOLVED
    disc = attrs.get("role") or attrs.get("kind") or ""
    return ".".join([cls] + ([str(disc)] if disc else []) + [UNRESOLVED])


def _default_verb(cls: str) -> str:
    if cls in _DEFAULT_VERBS:
        return _DEFAULT_VERBS[cls]
    return "open" if cls in _OPENABLE else "remove"


def _is_chassis(parsed: ParsedTarget, step_type: str) -> bool:
    return step_type == "reorient" or parsed.cls == CHASSIS_KEY


def _identity(cls: str, disc: str, number: int, attrs: dict) -> tuple:
    """What a later row must match exactly before it may reuse an instance.

    Spec 3.2's class + role + number, plus every attribute that names *which*
    part the row means: the cable owner, the annotator's bracketed qualifier
    and ``of`` (what a cover, cage or latch belongs to).  Two "Connector 1"
    rows whose cables run to different devices are different connectors, and a
    RAM cover is never the heatsink cover.
    """
    return (
        cls,
        disc,
        number,
        str(attrs.get("cable_owner") or ""),
        str(attrs.get("qualifier") or ""),
        str(attrs.get("of") or ""),
    )


def _attr_conflicts(stored: dict, new: dict) -> list[tuple[str, object, object]]:
    """Attributes the new row states differently from the stored instance."""
    return [
        (name, stored[name], value)
        for name, value in new.items()
        if name not in _DERIVED_ATTRS and name in stored and stored[name] != value
    ]


def _merge_attrs(stored: dict, new: dict) -> None:
    """Add the new row's extra attributes; conflicts are handled by the caller."""
    for name, value in new.items():
        if name not in _DERIVED_ATTRS:
            stored.setdefault(name, value)


def _is_captive(cls: str, attrs: dict, meta: dict) -> Optional[bool]:
    if cls != "screw":
        return None
    brand = str(meta.get("brand_model_raw") or "").lower()
    dell = any(token in brand for token in _CAPTIVE_BRANDS)
    return bool(dell and attrs.get("role") == _CAPTIVE_ROLE)


class _Importer:
    """One desktop's import; kept private so the module API stays functional."""

    def __init__(self, desktop: int, meta: dict, taxonomy: Taxonomy) -> None:
        self.out = LogImport(desktop=desktop, meta=dict(meta))
        self.tax = taxonomy
        self.desktop = desktop
        self._ordinals: dict[tuple[str, str], int] = {}
        # (cls, discriminator, the number the annotator wrote) -> keys, in
        # creation order; the only place a later row may find an earlier one.
        self._numbered: dict[tuple[str, str, int], list[str]] = {}
        # instance key -> the (step, verb) pairs already applied to it.
        self._ops: dict[str, list[tuple[int, str]]] = {}
        self._seen_seq: dict[int, int] = {}
        self._prev_seq: Optional[int] = None

    # -- issues ----------------------------------------------------------- #
    def issue(self, step: Optional[int], text: str) -> None:
        where = f" step {step}:" if step is not None else ""
        self.out.issues.append(f"D{self.desktop:02d}{where} {text}")

    # -- run -------------------------------------------------------------- #
    def run(self, rows: list[dict]) -> LogImport:
        self._chassis()
        for step, row in enumerate(rows, start=1):
            self._row(step, row)
        self._resolve_socket_hosts()
        self._check_operations()
        return self.out

    def _chassis(self) -> None:
        self.out.instances[CHASSIS_KEY] = InstanceRec(
            key=CHASSIS_KEY, desktop=self.desktop, cls=CHASSIS_KEY
        )

    def _row(self, step: int, row: dict) -> None:
        raw_name = (row.get(NAME_COLUMN) or "").strip()
        nest = (row.get(NEST_COLUMN) or "").strip()
        parsed = parse_raw_name(raw_name, nest)
        step_type = self._step_type(step, raw_name, parsed)
        self._check_sequence_number(step, row)
        self.out.steps.append(
            StepRec(
                desktop=self.desktop,
                step=step,
                step_type=step_type,
                raw_name=raw_name,
                dupli=step_type == "dupli",
                notes=(row.get(NOTES_COLUMN) or "").strip(),
                duration_s=None,  # timestamps come from the frame index (task 3)
            )
        )
        if step_type in NO_ACTION_TYPES:
            return
        self.out.actions.append(self._action(step, raw_name, parsed, step_type, row))

    # -- step type -------------------------------------------------------- #
    def _step_type(self, step: int, raw_name: str, parsed: ParsedTarget) -> str:
        hint = parsed.step_type_hint
        if step == 1 or hint == "initial":
            return "initial"
        if hint == "dupli":
            return "dupli"
        if hint in HINT_TYPES:
            return hint
        if parsed.attempt:
            return "failed"
        if parsed.multi or _AND.search(raw_name):
            return "compound"
        return "normal"

    # -- actions ---------------------------------------------------------- #
    def _action(
        self, step: int, raw_name: str, parsed: ParsedTarget, step_type: str, row: dict
    ) -> ActionRec:
        cls = CHASSIS_KEY if _is_chassis(parsed, step_type) else parsed.cls
        verb = parsed.verb or _default_verb(cls)
        self._check_verb(step, cls, verb)
        target = self._target(step, raw_name, parsed, step_type, verb)
        return ActionRec(
            desktop=self.desktop,
            step=step,
            idx=0,
            target=target,
            verb=verb,
            tool=self._tool(step, raw_name, cls, row),
            direction="none",
            result="failed" if step_type == "failed" else "success",
            failure_reason=None,  # filled in during S1 review
        )

    def _target(
        self, step: int, raw_name: str, parsed: ParsedTarget, step_type: str, verb: str
    ) -> str:
        """Resolve the target key; every ``?`` that survives is reported."""
        if _is_chassis(parsed, step_type):
            return CHASSIS_KEY
        fallback = parsed.matched_rule == FALLBACK_RULE
        if fallback:
            self.issue(step, f"unrecognized name {raw_name!r} (fallback rule) - needs manual target")
        if step_type == "compound":
            numbers = parsed.attrs.get("instance_nos")
            seen = f" (names {numbers})" if numbers else ""
            self.issue(
                step,
                f"compound row {raw_name!r}{seen} names several targets "
                f"- needs manual target resolution",
            )
            return _placeholder_key(parsed.cls, parsed.attrs)
        if not parsed.cls:
            if not fallback:
                self.issue(step, f"{raw_name!r} names no target class - needs manual target")
            return UNRESOLVED
        return self._instance(step, raw_name, parsed, verb)

    def _check_verb(self, step: int, cls: str, verb: str) -> None:
        spec = self.tax.verbs.get(verb)
        if spec is None:
            self.issue(step, f"verb {verb!r} is not in the taxonomy")
        elif cls and cls not in spec["applies_to"]:
            self.issue(step, f"verb {verb!r} does not apply to class {cls!r}")

    def _tool(self, step: int, raw_name: str, cls: str, row: dict) -> str:
        raw_tool = (row.get(TOOL_COLUMN) or "").strip()
        tool = map_tool(raw_tool)
        if cls == "screw" and tool == "none":
            detail = f"unrecognized tool {raw_tool!r}" if raw_tool else "no tool recorded"
            self.issue(step, f"screw step {raw_name!r} has {detail} - tool set to {UNKNOWN_TOOL!r}")
            return UNKNOWN_TOOL
        return tool

    # -- instances -------------------------------------------------------- #
    def _instance(self, step: int, raw_name: str, parsed: ParsedTarget, verb: str) -> str:
        """The target key for one row.

        Spec 3.2: every row that names a target is a **new** instance unless
        all three reuse conditions hold (see :meth:`_reuse`) -- the sheet's
        numbering restarts ("CPU fan screw 1..5" then "Heatsink screw 1..4")
        and its unnumbered repeats ("Case - motherboard connector" seven times)
        are genuinely different parts.
        """
        cls, attrs = parsed.cls, parsed.attrs
        disc = str(attrs.get("role") or attrs.get("kind") or "")
        number = parsed.instance_no
        key = self._reuse(step, raw_name, cls, disc, number, attrs, verb)
        if key is None:
            key = self._create(cls, disc, number, parsed)
        self.out.instances[key].raw_names.append(raw_name)
        self._ops.setdefault(key, []).append((step, verb))
        return key

    def _reuse(
        self,
        step: int,
        raw_name: str,
        cls: str,
        disc: str,
        number: Optional[int],
        attrs: dict,
        verb: str,
    ) -> Optional[str]:
        """An existing instance this row operates again, or ``None``.

        All of these must hold: the name carries an explicit instance number,
        an earlier instance of the same ``(cls, discriminator, number)`` has
        no conflicting attribute, and the verb was not already applied to it
        (e.g. loosen then remove).  Reuse and attribute conflicts are both
        reported, so nothing is merged or dropped silently.
        """
        if number is None:
            return None
        for key in self._numbered.get(_identity(cls, disc, number, attrs), []):
            prior = [v for _, v in self._ops.get(key, [])]
            if verb in prior:
                continue  # the same operation again means another part
            self.issue(
                step,
                f"reused instance {key} at step {step} for {raw_name!r} "
                f"(prior verbs {', '.join(prior) or 'none'}; new verb {verb})",
            )
            stored = self.out.instances[key].attrs
            for name, old, new in _attr_conflicts(stored, attrs):
                self.issue(
                    step,
                    f"instance {key} attribute {name!r} conflicts with the stored value "
                    f"({old!r} vs {new!r}) - kept {old!r}",
                )
            _merge_attrs(stored, attrs)
            return key
        return None

    def _create(self, cls: str, disc: str, number: Optional[int], parsed: ParsedTarget) -> str:
        group = (cls, disc)
        ordinal = self._ordinals.get(group, 0) + 1
        self._ordinals[group] = ordinal
        key = instance_key(cls, parsed.attrs, ordinal)
        self.out.instances[key] = self._new_instance(key, cls, parsed.attrs, parsed)
        if number is not None:
            identity = _identity(cls, disc, number, parsed.attrs)
            self._numbered.setdefault(identity, []).append(key)
        return key

    def _check_operations(self) -> None:
        """Guard the one-operation-per-instance invariant.

        Multi-step instances are the reused ones and are already reported at
        the point of reuse; the same verb twice can only mean the importer
        merged two different parts, so it is reported loudly.  ``_ops`` only
        holds the instances ``_instance`` created, so the implicit chassis --
        whose ``reorient`` legitimately repeats -- never reaches this check.
        """
        for key, ops in self._ops.items():
            seen: dict[str, int] = {}
            for step, verb in ops:
                if verb in seen:
                    self.issue(
                        None,
                        f"INVARIANT: instance {key} received verb {verb!r} twice "
                        f"(steps {seen[verb]} and {step})",
                    )
                seen[verb] = step

    def _new_instance(
        self, key: str, cls: str, attrs: dict, parsed: ParsedTarget
    ) -> InstanceRec:
        own = {k: v for k, v in attrs.items() if k != "instance_nos"}
        if parsed.instance_no is not None:
            own["sheet_no"] = parsed.instance_no
        captive = _is_captive(cls, attrs, self.out.meta)
        if captive is not None:
            own["captive"] = captive
            own["captive_source"] = "heuristic"
        owner = attrs.get("cable_owner")
        return InstanceRec(
            key=key,
            desktop=self.desktop,
            cls=cls,
            attrs=own,
            cable=f"cable:{owner}" if owner else None,
        )

    def _resolve_socket_hosts(self) -> None:
        """Give every connector a socket host once all instances are known."""
        by_class: dict[str, str] = {}
        for key, inst in self.out.instances.items():
            by_class.setdefault(inst.cls, key)
        for inst in self.out.instances.values():
            if inst.cls != "connector":
                continue
            names = " ".join(inst.raw_names)
            host = DEFAULT_SOCKET_HOST
            if not _MOTHERBOARD.search(names):
                host_cls = _DEVICE_SOCKET_HOSTS.get(str(inst.attrs.get("cable_owner") or ""))
                host = by_class.get(host_cls or "", DEFAULT_SOCKET_HOST)
            inst.socket_host = host

    # -- sequence numbers ------------------------------------------------- #
    def _check_sequence_number(self, step: int, row: dict) -> None:
        raw = (row.get(HEADER_CELL) or "").strip()
        if not raw:
            self.issue(step, "blank Sequence Number in the sheet")
            return
        try:
            seq = int(float(raw))
        except ValueError:
            self.issue(step, f"Sequence Number {raw!r} is not a number")
            return
        if seq in self._seen_seq:
            self.issue(
                step,
                f"Sequence Number {seq} duplicated (first seen at step {self._seen_seq[seq]})",
            )
        elif self._prev_seq is not None and seq > self._prev_seq + 1:
            first, last = self._prev_seq + 1, seq - 1
            span = f"{first}" if first == last else f"{first}-{last}"
            plural = "" if first == last else "s"
            self.issue(step, f"Sequence Number{plural} {span} skipped before this row")
        self._seen_seq.setdefault(seq, step)
        self._prev_seq = seq


def import_log(desktop: int, rows: list[dict], meta: dict, taxonomy: Taxonomy) -> LogImport:
    """Turn one desktop's step rows into step/action/instance drafts."""
    return _Importer(desktop, meta, taxonomy).run(rows)


# --------------------------------------------------------------------------- #
# CLI: import every log in a directory and write the issue report
# --------------------------------------------------------------------------- #
def iter_desktop_csvs(directory: str | Path) -> Iterator[tuple[int, Path]]:
    """Yield ``(desktop, path)`` for the canonical export of every desktop.

    Desktop 28 was exported three times; only the ``1E9qmg`` copy is the real
    D28 sheet (the other two repeat D27), so the other two are skipped.
    """
    found: dict[int, Path] = {}
    for path in sorted(Path(directory).glob("desktop_*.csv")):
        match = _DESKTOP_FILE.match(path.name)
        if not match:
            continue
        number, suffix = int(match.group(1)), match.group(2)
        if suffix == "meta":  # the sidecar metadata table, not a step table
            continue
        if number == 28 and suffix != _DESKTOP_28_KEEP:
            continue
        found[number] = path
    for number in sorted(found):
        yield number, found[number]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    from .log_report import main

    sys.exit(main())


