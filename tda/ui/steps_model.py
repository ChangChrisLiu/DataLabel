"""Qt-free view-model behind stage S1, "步骤与实例核对" (spec 4.1).

The annotator reviews one desktop's imported disassembly log before drawing any
mask: every logical step is one row carrying the parsed action (target, verb,
tool, direction, difficulty, result and failure reason) next to the scanner
thumbnails of the frames before and after it, and a second table lists the
desktop's instances with the relational attributes of spec 7.1.

:class:`StepTableData` owns that whole edit session. It loads the drafts stage
S0 wrote, validates every edit against the taxonomy (spec 6), and on
:meth:`StepTableData.save` writes the step/action/instance tables back and
recompiles the automatic state-event log, returning whatever
:func:`~tda.core.states.validate_events` still objects to.

No Qt here: :mod:`tda.ui.panels.steptable` is the only widget layer, so this
module stays unit-testable head-less.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from tda.core.db import Db
from tda.core.logs import CHASSIS_KEY, NO_ACTION_TYPES, UNRESOLVED, instance_key
from tda.core.model import ActionRec, InstanceRec, StepRec, StepType
from tda.core.states import (
    CABLE_CLASS,
    CABLE_PREFIX,
    events_from_actions,
    validate_events,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy, parse_raw_name

__all__ = [
    "DIFFICULTY_MAX",
    "DIFFICULTY_MIN",
    "EditError",
    "FAILURE_REASONS",
    "GROUP_ORDERS",
    "RESULTS",
    "SCREW_HEADS",
    "STEP_TYPES",
    "StepRow",
    "StepTableData",
    "thumb_path",
]

#: ``Action.failure_reason`` enum of spec 3.1.
FAILURE_REASONS = ("blocked_by_cable", "blocked_by_part", "fastener_stuck", "wrong_tool", "other")
RESULTS = ("success", "failed")
#: ``Instance.group_order`` enum of spec 3.1.
GROUP_ORDERS = ("unordered", "sequential", "opposite_pairs")
#: ``screw.head`` vocabulary of spec 6.1.
SCREW_HEADS = ("PH1", "PH2", "PH3", "T15", "T20", "unknown")
STEP_TYPES = tuple(t.value for t in StepType)
DIFFICULTY_MIN, DIFFICULTY_MAX = 1, 5

#: The verb a brand-new action starts from until the annotator picks one.
DEFAULT_VERB = "remove"

_ACTION_FIELDS = ("verb", "tool", "direction", "result", "failure_reason", "difficulty")
_STEP_FIELDS = ("step_type", "notes")
#: Instance fields holding another instance's key.
_RELATION_FIELDS = ("parent", "mounted_on", "fastens", "socket_host")
#: Relations only one class may carry (spec 7.1).
_RELATION_CLASS = {"fastens": "screw", "socket_host": "connector"}
_ATTR_FIELDS = ("head", "captive")

_ORDINAL = re.compile(r"\.(\d+)$")
_TRUE_WORDS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_WORDS = frozenset({"", "0", "false", "no", "n", "off", "none"})


class EditError(ValueError):
    """An edit the taxonomy or the instance table refuses; the text is UI-ready."""


# --------------------------------------------------------------------------- #
# thumbnails
# --------------------------------------------------------------------------- #
def thumb_path(
    cache_dir: str | Path, desktop: int, step: int, view: str = "scan"
) -> Optional[Path]:
    """Path of one cached frame, or ``None`` when it is not on disk.

    The local cache (spec 2.4) is laid out as
    ``<cache_dir>/<view>/D<nn>/s<kkk>.png``. Step ``0`` -- the "before" cell of
    the very first row -- has no frame, and a cache root that does not exist
    yet is not an error: the panel simply shows an empty cell.
    """
    if step < 1:
        return None
    path = Path(cache_dir) / view / f"D{desktop:02d}" / f"s{step:03d}.png"
    try:
        return path if path.is_file() else None
    except OSError:  # unreachable drive, bad path -- treat as "no thumbnail"
        return None


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #
@dataclass
class StepRow:
    """One logical step: its record, its actions and the issues it still has."""

    step: StepRec
    actions: list[ActionRec] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def number(self) -> int:
        return self.step.step

    @property
    def step_type(self) -> str:
        return self.step.step_type

    @property
    def raw_name(self) -> str:
        return self.step.raw_name

    @property
    def notes(self) -> str:
        return self.step.notes

    def action(self, idx: int = 0) -> Optional[ActionRec]:
        """The action at position ``idx``, or ``None`` when the row has none."""
        return self.actions[idx] if 0 <= idx < len(self.actions) else None


# --------------------------------------------------------------------------- #
# the edit session
# --------------------------------------------------------------------------- #
@dataclass
class StepTableData:
    """One desktop's step table and instance table, loaded for review."""

    desktop: int
    tax: Taxonomy
    rows: list[StepRow] = field(default_factory=list)
    instances: dict[str, InstanceRec] = field(default_factory=dict)
    #: Validation messages of the last :meth:`save`.
    messages: list[str] = field(default_factory=list)

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(cls, db: Db, desktop: int, tax: Taxonomy | None = None) -> "StepTableData":
        """Read one desktop's steps, actions and instances out of the database."""
        tax = tax or load_taxonomy()
        by_step: dict[int, list[ActionRec]] = {}
        for action in db.actions(desktop):
            by_step.setdefault(action.step, []).append(action)
        rows = [StepRow(step=rec, actions=by_step.get(rec.step, [])) for rec in db.steps(desktop)]
        data = cls(desktop=desktop, tax=tax, rows=rows, instances=db.instances(desktop))
        data.refresh_issues()
        return data

    # -- lookups ---------------------------------------------------------- #
    def row(self, step: int) -> StepRow:
        """The row of one logical step; raises :class:`EditError` when unknown."""
        for row in self.rows:
            if row.number == step:
                return row
        raise EditError(f"D{self.desktop:02d} has no step {step}")

    @property
    def steps(self) -> list[StepRec]:
        return [row.step for row in self.rows]

    @property
    def actions(self) -> list[ActionRec]:
        return [action for row in self.rows for action in row.actions]

    @property
    def issues(self) -> list[str]:
        """Every row issue, prefixed with its step, in step order."""
        return [f"step {row.number}: {text}" for row in self.rows for text in row.issues]

    def instance_keys(self) -> list[str]:
        """Instance keys in a stable, human-readable order."""
        return sorted(self.instances)

    # -- step / action edits ---------------------------------------------- #
    def apply_edit(self, step: int, field: str, value: Any, action_idx: int = 0) -> None:
        """Apply one cell edit, raising :class:`EditError` when it is refused."""
        row = self.row(step)
        if field == "step_type":
            if value not in STEP_TYPES:
                raise EditError(f"{value!r} is not a step type ({', '.join(STEP_TYPES)})")
            row.step.step_type = str(value)
            row.step.dupli = value == StepType.DUPLI.value
        elif field == "notes":
            row.step.notes = "" if value is None else str(value)
        elif field == "target":
            self.change_target(step, target=value, action_idx=action_idx)
            return  # change_target refreshed the issues already
        elif field in _ACTION_FIELDS:
            action = self._action(row, action_idx)
            setattr(action, field, self._checked_action_value(field, value, action))
        else:
            raise EditError(f"{field!r} is not an editable step field")
        self.refresh_issues()

    def add_action(
        self,
        step: int,
        target: str | None = None,
        verb: str | None = None,
        tool: str | None = None,
        direction: str | None = None,
    ) -> ActionRec:
        """Append one action to a step, cloning the last one as the template."""
        row = self.row(step)
        last = row.actions[-1] if row.actions else None
        action = ActionRec(
            desktop=self.desktop,
            step=step,
            idx=len(row.actions),
            target=target if target is not None else (last.target if last else UNRESOLVED),
            verb=verb or (last.verb if last else DEFAULT_VERB),
            tool=tool or (last.tool if last else "none"),
            direction=direction or (last.direction if last else "none"),
            result=last.result if last else "success",
            failure_reason=last.failure_reason if last else None,
            difficulty=last.difficulty if last else None,
        )
        row.actions.append(action)
        self.refresh_issues()
        return action

    def remove_action(self, step: int, idx: int) -> None:
        """Drop one action of a step and renumber the ones that follow."""
        row = self.row(step)
        if not 0 <= idx < len(row.actions):
            raise EditError(f"step {step} has no action {idx}")
        row.actions.pop(idx)
        for position, action in enumerate(row.actions):
            action.idx = position
        self.refresh_issues()

    def change_target(
        self,
        step: int,
        target: str | None = None,
        action_idx: int = 0,
        cls: str | None = None,
        attrs: dict | None = None,
        verb: str | None = None,
    ) -> str:
        """Point an action at another instance, or at a newly created one.

        Pass ``target`` to pick an existing instance (or a ``cable:*`` virtual
        node), or ``cls``/``attrs`` to create one; the new key follows the
        ordinal rules of :func:`tda.core.logs.instance_key`, continuing the
        numbering of the instances already present.

        The verb must apply to the new target's class. ``verb`` sets it in the
        same call, which is how a target of a different class is picked without
        ever passing through an illegal verb/target pair.
        """
        row = self.row(step)
        action = self._action(row, action_idx)
        wanted = verb or action.verb
        if target:
            target = str(target).strip()
            if not target.startswith(CABLE_PREFIX) and target not in self.instances:
                raise EditError(f"D{self.desktop:02d} has no instance {target!r}")
            self._check_verb(wanted, self._class_of(target), target)
        elif cls:
            target = self._create_instance(cls, dict(attrs or {}), row.raw_name, wanted).key
        else:
            raise EditError("change_target needs either an existing target or a class")
        action.target = target
        action.verb = wanted
        self.refresh_issues()
        return target

    def split_compound(self, step: int, n: int) -> list[str]:
        """Turn a flagged compound placeholder row into ``n`` actions/instances.

        The importer drafts one action with an unresolved key (``screw.?``) for
        a row naming several parts; S1 resolves it by saying how many parts it
        really was. The verb, tool, direction and result of the placeholder
        carry over to all ``n`` actions, and the row becomes a normal step.
        """
        row = self.row(step)
        if n < 2:
            raise EditError("splitting a compound row needs at least two targets")
        if len(row.actions) != 1:
            raise EditError(f"step {step} must hold exactly one action before it is split")
        action = row.actions[0]
        if row.step_type != StepType.COMPOUND.value and UNRESOLVED not in action.target:
            raise EditError(f"step {step} is not flagged as a compound row")

        parsed = parse_raw_name(row.raw_name)
        cls = parsed.cls or self._class_of(action.target) or ""
        if cls not in self.tax.classes:
            raise EditError(
                f"step {step}: {row.raw_name!r} names no taxonomy class - pick the targets by hand"
            )
        attrs = {k: v for k, v in parsed.attrs.items() if k != "instance_nos"}
        template = ActionRec(**vars(action))
        row.actions.clear()
        keys: list[str] = []
        for idx in range(n):
            new = self._create_instance(cls, dict(attrs), row.raw_name, template.verb)
            keys.append(new.key)
            clone = ActionRec(**vars(template))
            clone.idx = idx
            clone.target = new.key
            row.actions.append(clone)
        row.step.step_type = StepType.NORMAL.value
        row.step.dupli = False
        self.refresh_issues()
        return keys

    # -- instance edits ---------------------------------------------------- #
    def apply_instance_edit(self, key: str, field: str, value: Any) -> None:
        """Edit one relational attribute of an instance (spec 7.1)."""
        inst = self.instances.get(key)
        if inst is None:
            raise EditError(f"D{self.desktop:02d} has no instance {key!r}")
        if field in _RELATION_FIELDS:
            self._set_relation(inst, field, value)
        elif field == "cable":
            text = _text(value)
            if text and not text.startswith(CABLE_PREFIX):
                raise EditError(f"a cable node id must start with {CABLE_PREFIX!r}, got {text!r}")
            inst.cable = text or None
        elif field == "attached":
            inst.attached = _as_bool(value)
        elif field in _ATTR_FIELDS:
            self._set_attr(inst, field, value)
        elif field == "group_order":
            if value not in GROUP_ORDERS:
                raise EditError(f"{value!r} is not a group order ({', '.join(GROUP_ORDERS)})")
            inst.group_order = str(value)
        elif field == "group_id":
            inst.group_id = _text(value) or None
        elif field == "slot_id":
            inst.slot_id = _text(value) or None
        elif field == "removal_direction":
            text = _text(value)
            if text and text not in self.tax.directions:
                raise EditError(
                    f"{text!r} is not a direction ({', '.join(self.tax.directions)})"
                )
            inst.removal_direction = text or None
        else:
            raise EditError(f"{field!r} is not an editable instance field")
        self.refresh_issues()

    def _set_relation(self, inst: InstanceRec, field: str, value: Any) -> None:
        text = _text(value)
        required = _RELATION_CLASS.get(field)
        if text and required is not None and inst.cls != required:
            raise EditError(f"only a {required} carries {field!r}, {inst.key} is a {inst.cls}")
        if text and text not in self.instances:
            raise EditError(f"D{self.desktop:02d} has no instance {text!r}")
        if field == "parent" and text and self._would_cycle(inst.key, text):
            raise EditError(f"{text!r} is already below {inst.key} - parent would form a cycle")
        setattr(inst, field, text or None)

    def _set_attr(self, inst: InstanceRec, field: str, value: Any) -> None:
        if inst.cls != "screw":
            raise EditError(f"only a screw carries {field!r}, {inst.key} is a {inst.cls}")
        if field == "head":
            text = _text(value)
            if text and text not in SCREW_HEADS:
                raise EditError(f"{text!r} is not a screw head ({', '.join(SCREW_HEADS)})")
            if text:
                inst.attrs["head"] = text
                inst.attrs.setdefault("head_source", "manual")
            else:
                inst.attrs.pop("head", None)
        else:
            inst.attrs["captive"] = _as_bool(value)
            inst.attrs["captive_source"] = "manual"

    def _would_cycle(self, key: str, parent: str) -> bool:
        """Would setting ``parent`` on ``key`` close a parent loop?"""
        seen = {key}
        current: Optional[str] = parent
        while current:
            if current in seen:
                return True
            seen.add(current)
            rec = self.instances.get(current)
            current = rec.parent if rec else None
        return False

    # -- saving ------------------------------------------------------------ #
    def save(self, db: Db) -> list[str]:
        """Write the session back and recompile the automatic state events.

        The hand-written (``auto=False``) events survive; the returned list is
        what :func:`~tda.core.states.validate_events` says about the whole log
        afterwards, and is also kept in :attr:`messages`.
        """
        actions = self.actions
        db.replace_steps(self.desktop, self.steps, actions)
        for inst in self.instances.values():
            db.upsert_instance(inst)
        events = events_from_actions(self.instances, actions, self.tax)
        db.replace_events(self.desktop, events, auto_only=True)
        self.messages = validate_events(self.instances, db.events(self.desktop), self.tax)
        self.refresh_issues()
        return list(self.messages)

    # -- issues ------------------------------------------------------------ #
    def refresh_issues(self) -> None:
        """Re-derive every row's open questions from the loaded records.

        The importer's own ``LogImport.issues`` are not stored, so S1 rebuilds
        the same questions from the drafts it is looking at: unresolved targets,
        compound rows still to split, values outside the taxonomy, and failed
        attempts without a reason.
        """
        for row in self.rows:
            row.issues = list(self._row_issues(row))

    def _row_issues(self, row: StepRow) -> Iterator[str]:
        if row.step_type == StepType.COMPOUND.value:
            yield f"compound row {row.raw_name!r} - split it into one action per target"
        if not row.actions and row.step_type not in NO_ACTION_TYPES:
            yield f"{row.step_type} step {row.raw_name!r} has no action"
        for action in row.actions:
            yield from self._action_issues(action)

    def _action_issues(self, action: ActionRec) -> Iterator[str]:
        if UNRESOLVED in action.target:
            yield f"unresolved target {action.target!r} - pick or create the instance"
        cls = self._class_of(action.target)
        spec = self.tax.verbs.get(action.verb)
        if spec is None:
            yield f"verb {action.verb!r} is not in the taxonomy"
        elif cls is not None and cls not in spec["applies_to"]:
            yield f"verb {action.verb!r} does not apply to class {cls!r}"
        if action.tool not in self.tax.tools:
            yield f"tool {action.tool!r} is not in the taxonomy - pick one"
        if action.direction not in self.tax.directions:
            yield f"direction {action.direction!r} is not in the taxonomy"
        if action.difficulty is not None and not (
            DIFFICULTY_MIN <= action.difficulty <= DIFFICULTY_MAX
        ):
            yield f"difficulty {action.difficulty!r} is outside {DIFFICULTY_MIN}-{DIFFICULTY_MAX}"
        if action.result == "failed" and not action.failure_reason:
            yield "a failed attempt needs a failure_reason"
        if action.result == "success" and action.failure_reason:
            yield f"failure_reason {action.failure_reason!r} on a successful action"

    # -- helpers ----------------------------------------------------------- #
    def _action(self, row: StepRow, idx: int) -> ActionRec:
        action = row.action(idx)
        if action is None:
            raise EditError(f"step {row.number} ({row.step_type}) has no action {idx}")
        return action

    def _class_of(self, target: str) -> Optional[str]:
        """The taxonomy class an action target belongs to, when it is knowable."""
        rec = self.instances.get(target)
        if rec is not None:
            return rec.cls
        if target.startswith(CABLE_PREFIX):
            return CABLE_CLASS
        head = target.split(".")[0]
        return head if head in self.tax.classes else None

    def _check_verb(self, verb: str, cls: Optional[str], target: str) -> None:
        spec = self.tax.verbs.get(verb)
        if spec is None:
            raise EditError(f"verb {verb!r} is not in the taxonomy")
        if cls is not None and cls not in spec["applies_to"]:
            raise EditError(f"verb {verb!r} does not apply to class {cls!r} (target {target})")

    def _checked_action_value(self, field: str, value: Any, action: ActionRec) -> Any:
        if field == "verb":
            self._check_verb(str(value), self._class_of(action.target), action.target)
            return str(value)
        if field == "tool":
            if value not in self.tax.tools:
                raise EditError(f"{value!r} is not a tool ({', '.join(self.tax.tools)})")
            return str(value)
        if field == "direction":
            if value not in self.tax.directions:
                raise EditError(
                    f"{value!r} is not a direction ({', '.join(self.tax.directions)})"
                )
            return str(value)
        if field == "result":
            if value not in RESULTS:
                raise EditError(f"{value!r} is not a result ({', '.join(RESULTS)})")
            return str(value)
        if field == "failure_reason":
            text = _text(value)
            if text and text not in FAILURE_REASONS:
                raise EditError(
                    f"{text!r} is not a failure reason ({', '.join(FAILURE_REASONS)})"
                )
            return text or None
        return _checked_difficulty(value)

    def _create_instance(
        self, cls: str, attrs: dict, raw_name: str, verb: str
    ) -> InstanceRec:
        """Create one instance of ``cls``, continuing the existing numbering."""
        if cls not in self.tax.classes:
            raise EditError(f"{cls!r} is not a taxonomy class")
        self._check_verb(verb, cls, f"new {cls}")
        attrs = {k: v for k, v in attrs.items() if k != "instance_nos"}
        key = instance_key(cls, attrs, self._next_ordinal(cls, attrs))
        if key in self.instances:  # only `chassis`, which is unique per desktop
            raise EditError(f"D{self.desktop:02d} already has an instance {key!r}")
        owner = attrs.get("cable_owner")
        rec = InstanceRec(
            key=key,
            desktop=self.desktop,
            cls=cls,
            attrs=attrs,
            cable=f"{CABLE_PREFIX}{owner}" if owner else None,
            raw_names=[raw_name] if raw_name else [],
        )
        self.instances[key] = rec
        return rec

    def _next_ordinal(self, cls: str, attrs: dict) -> int:
        """One past the highest ordinal among the instances of the same group."""
        if cls == CHASSIS_KEY:
            return 1  # `chassis` carries no ordinal at all
        disc = str(attrs.get("role") or attrs.get("kind") or "")
        prefix = f"{cls}.{disc}." if disc else f"{cls}."
        top = 0
        for key in self.instances:
            if not key.startswith(prefix):
                continue
            match = _ORDINAL.match(key[len(prefix) - 1 :])
            if match:
                top = max(top, int(match.group(1)))
        return top + 1


# --------------------------------------------------------------------------- #
# value coercion
# --------------------------------------------------------------------------- #
def _text(value: Any) -> str:
    """A cell value as trimmed text; ``None`` becomes the empty string."""
    return "" if value is None else str(value).strip()


def _as_bool(value: Any) -> bool:
    """Read a check-box / combo / text value as a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _text(value).lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    raise EditError(f"{value!r} is not a yes/no value")


def _checked_difficulty(value: Any) -> Optional[int]:
    """Validate ``Action.difficulty``: 1-5, or empty for "not recorded"."""
    if value is None or _text(value) == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise EditError(f"difficulty {value!r} is not a whole number") from None
    if not DIFFICULTY_MIN <= number <= DIFFICULTY_MAX:
        raise EditError(f"difficulty {number} is outside {DIFFICULTY_MIN}-{DIFFICULTY_MAX}")
    return number
