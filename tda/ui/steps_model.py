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
module stays unit-testable head-less. The vocabularies it validates against,
the value coercion and the ``LS:`` note handling live in
:mod:`tda.ui.steps_values` and are re-exported here, so callers need only this
one import.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from tda.core.db import Db
from tda.core.logs import CHASSIS_KEY, UNRESOLVED, instance_key
from tda.core.model import ActionRec, InstanceRec, StepRec, StepType
from tda.core.states import (
    CABLE_CLASS,
    CABLE_PREFIX,
    events_from_actions,
    validate_events,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy, parse_raw_name
from tda.ui.steps_delete import delete_instance
from tda.ui.steps_relations import RelationsData
from tda.ui.steps_issues import (
    dangling_issues,
    draft_issues,
    orphan_issues,
    row_issues,
    unresolved_issues,
)
from tda.ui.steps_values import (
    DISCRIMINATORS,
    FAILURE_REASONS,
    GROUP_ORDERS,
    LS_NOTE_PREFIX,
    RELATION_FIELDS,
    RESULTS,
    SCREW_HEADS,
    STEP_TYPES,
    EditError,
    as_bool,
    checked_difficulty,
    human_notes,
    ls_notes,
    merge_notes,
    text_of,
    thumb_path,
)

__all__ = [
    "EditError",
    "FAILURE_REASONS",
    "GROUP_ORDERS",
    "LS_NOTE_PREFIX",
    "RESULTS",
    "SCREW_HEADS",
    "STEP_TYPES",
    "StepRow",
    "StepTableData",
    "human_notes",
    "ls_notes",
    "merge_notes",
    "thumb_path",
]

#: The verb a brand-new action starts from until the annotator picks one.
DEFAULT_VERB = "remove"

_ACTION_FIELDS = ("verb", "tool", "direction", "result", "failure_reason", "difficulty")
#: Relations only one class may carry (spec 7.1).
_RELATION_CLASS = {"fastens": "screw", "socket_host": "connector"}
_ATTR_FIELDS = ("head", "captive")

_ORDINAL = re.compile(r"\.(\d+)$")


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
        """The operator's notes; the imported ``LS:`` lines stay out of the editor."""
        return human_notes(self.step.notes)

    @property
    def ls_notes(self) -> list[str]:
        """The ``LS:`` lines this step carries, shown read-only."""
        return ls_notes(self.step.notes)

    def action(self, idx: int = 0) -> Optional[ActionRec]:
        """The action at position ``idx``, or ``None`` when the row has none."""
        return self.actions[idx] if 0 <= idx < len(self.actions) else None


# --------------------------------------------------------------------------- #
# the edit session
# --------------------------------------------------------------------------- #
def _reorients_moved(before: list[StepRec], after: list[StepRec]) -> bool:
    """Did any step become, or stop being, a ``reorient``?

    Only that: every other edit in S1 -- a verb, a target, a split compound row
    -- leaves the pose boundaries exactly where they were, and re-cutting on
    every Apply would put a ``pose_recut`` row in the op log for each of them.
    """
    kind = StepType.REORIENT.value
    was = {s.step for s in before if s.step_type == kind}
    now = {s.step for s in after if s.step_type == kind}
    return was != now


@dataclass
class StepTableData:
    """One desktop's step table and instance table, loaded for review."""

    desktop: int
    tax: Taxonomy
    rows: list[StepRow] = field(default_factory=list)
    instances: dict[str, InstanceRec] = field(default_factory=dict)
    #: Validation messages of the last :meth:`save`.
    messages: list[str] = field(default_factory=list)
    #: Instances no action targets any more, re-derived with the row issues.
    orphans: list[str] = field(default_factory=list)
    #: Classes whose implied instance this desktop's annotator has deleted
    #: (:meth:`tda.core.db.Db.declined_implied`). Read once at load and updated
    #: by :func:`tda.ui.steps_delete.delete_instance`, so the questions
    #: :mod:`tda.ui.steps_issues` asks do not argue with a decision already made.
    declined: set[str] = field(default_factory=set)
    #: ``{view: segments}`` when the last :meth:`save` re-cut the pose segments
    #: because a step became -- or stopped being -- a ``reorient``; empty
    #: otherwise. The panel says so, because the annotator has just moved a
    #: boundary that every shape of that view is anchored to.
    recut: dict[str, int] = field(default_factory=dict)
    #: The desktop's constraint edges, staged like everything else here and
    #: written by :meth:`save` in the same transaction
    #: (:mod:`tda.ui.steps_relations`, stage S6).
    relations: "RelationsData" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.relations is None:
            self.relations = RelationsData(data=self)

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(cls, db: Db, desktop: int, tax: Taxonomy | None = None) -> "StepTableData":
        """Read one desktop's steps, actions, instances and constraint edges."""
        tax = tax or load_taxonomy()
        by_step: dict[int, list[ActionRec]] = {}
        for action in db.actions(desktop):
            by_step.setdefault(action.step, []).append(action)
        rows = [StepRow(step=rec, actions=by_step.get(rec.step, [])) for rec in db.steps(desktop)]
        data = cls(desktop=desktop, tax=tax, rows=rows, instances=db.instances(desktop),
                   declined=set(db.declined_implied(desktop)))
        data.relations.reload(db)
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
        """Every open question: the step rows first, then the instance table."""
        rows = [f"step {row.number}: {text}" for row in self.rows for text in row.issues]
        return rows + list(self.orphans)

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
            row.step.notes = merge_notes(row.step.notes, value)
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
            self._check_verb(wanted, self.class_of(target), target)
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
        cls = parsed.cls or self.class_of(action.target) or ""
        if cls not in self.tax.classes:
            raise EditError(
                f"step {step}: {row.raw_name!r} names no taxonomy class - pick the targets by hand"
            )
        attrs = {k: v for k, v in parsed.attrs.items() if k != "instance_nos"}
        template = ActionRec(**vars(action))

        # Build everything first: a split that is refused half-way through must
        # leave the row exactly as it was, not headless.
        staged: list[InstanceRec] = []
        actions: list[ActionRec] = []
        for idx in range(n):
            new = self._build_instance(
                cls, dict(attrs), row.raw_name, template.verb, {rec.key for rec in staged}
            )
            staged.append(new)
            clone = ActionRec(**vars(template))
            clone.idx = idx
            clone.target = new.key
            actions.append(clone)

        for rec in staged:
            self.instances[rec.key] = rec
        row.actions[:] = actions
        row.step.step_type = StepType.NORMAL.value
        row.step.dupli = False
        self.refresh_issues()
        return [rec.key for rec in staged]

    # -- instance edits ---------------------------------------------------- #
    def apply_instance_edit(self, key: str, field: str, value: Any) -> None:
        """Edit one relational attribute of an instance (spec 7.1)."""
        inst = self.instances.get(key)
        if inst is None:
            raise EditError(f"D{self.desktop:02d} has no instance {key!r}")
        if field in RELATION_FIELDS:
            self._set_relation(inst, field, value)
        elif field == "cable":
            text = text_of(value)
            if text and not text.startswith(CABLE_PREFIX):
                raise EditError(f"a cable node id must start with {CABLE_PREFIX!r}, got {text!r}")
            inst.cable = text or None
        elif field == "attached":
            inst.attached = as_bool(value)
        elif field in _ATTR_FIELDS:
            self._set_attr(inst, field, value)
        elif field == "group_order":
            if value not in GROUP_ORDERS:
                raise EditError(f"{value!r} is not a group order ({', '.join(GROUP_ORDERS)})")
            inst.group_order = str(value)
        elif field == "group_id":
            inst.group_id = text_of(value) or None
        elif field == "slot_id":
            inst.slot_id = text_of(value) or None
        elif field == "removal_direction":
            text = text_of(value)
            if text and text not in self.tax.directions:
                raise EditError(
                    f"{text!r} is not a direction ({', '.join(self.tax.directions)})"
                )
            inst.removal_direction = text or None
        else:
            raise EditError(f"{field!r} is not an editable instance field")
        self.refresh_issues()

    def _set_relation(self, inst: InstanceRec, field: str, value: Any) -> None:
        text = text_of(value)
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
            text = text_of(value)
            if text and text not in SCREW_HEADS:
                raise EditError(f"{text!r} is not a screw head ({', '.join(SCREW_HEADS)})")
            if text:
                inst.attrs["head"] = text
                inst.attrs.setdefault("head_source", "manual")
            else:
                inst.attrs.pop("head", None)
        else:
            inst.attrs["captive"] = as_bool(value)
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
    def delete_instance(self, db: Db, key: str) -> None:
        """Drop an instance nothing points at any more, writing straight through.

        See :mod:`tda.ui.steps_delete` for what is refused, what goes with the
        identity row and why the neighbours are repaired from their stored
        records rather than from memory.
        """
        delete_instance(self, db, key)

    def save(self, db: Db) -> list[str]:
        """Write the session back and recompile the automatic state events.

        Steps, actions, instances and the automatic event log land in one
        transaction, so a failure part-way through leaves the database exactly
        as it was. The hand-written (``auto=False``) events survive; the
        returned list is what :func:`~tda.core.states.validate_events` says
        about the whole log afterwards, and is also kept in :attr:`messages`.

        A step whose type becomes -- or stops being -- ``reorient`` moves a pose
        boundary in **all four views** (spec 2.5), so the segments are re-cut in
        the same transaction. Leaving that to the next ``load-index`` was the
        gap: the annotator corrected a compound row in S1, carried on drawing,
        and the shapes went into a segment the pipeline later renumbered under
        them. The re-cut moves the keyframes, the layer order and the ROIs and
        queues the frozen frames for a re-check exactly as
        :meth:`tda.core.db.Db.apply_recut` does everywhere else -- and a re-cut
        that raises takes the whole Apply with it.

        The staged constraint edges (stage S6) go in the same transaction, last,
        so an edge can name an instance this very Apply created and a re-cut
        that fails rolls the edges back with everything else.
        """
        from tda.pipeline import split_pose_segments    # late: tda.pipeline is heavy

        actions = self.actions
        self.recut = {}
        with db.transaction():
            moved = _reorients_moved(db.steps(self.desktop), self.steps)
            db.replace_steps(self.desktop, self.steps, actions)
            for inst in self.instances.values():
                db.upsert_instance(inst)
            events = events_from_actions(self.instances, actions, self.tax)
            db.replace_events(self.desktop, events, auto_only=True)
            if moved:
                self.recut = split_pose_segments(db, self.desktop)
            self.relations.write(db)
        self.relations.committed()
        self.messages = validate_events(self.instances, db.events(self.desktop), self.tax)
        self.refresh_issues()
        return list(self.messages)

    # -- issues ------------------------------------------------------------ #
    def refresh_issues(self) -> None:
        """Re-derive every open question from the drafts currently loaded.

        See :mod:`tda.ui.steps_issues` for what is asked about and why the
        importer's own ``LogImport.issues`` are not replayed.
        """
        for row in self.rows:
            row.issues = list(row_issues(row, self.tax, self.class_of))
        self.orphans = (
            list(orphan_issues(self.instances, self.actions))
            + list(dangling_issues(self.instances, self.tax))
            + list(unresolved_issues(self.instances, self.tax, self.declined))
            + list(draft_issues(self.instances))  # last: it asks for nothing
        )

    # -- helpers ----------------------------------------------------------- #
    def _action(self, row: StepRow, idx: int) -> ActionRec:
        action = row.action(idx)
        if action is None:
            raise EditError(f"step {row.number} ({row.step_type}) has no action {idx}")
        return action

    def class_of(self, target: str) -> Optional[str]:
        """The taxonomy class an action target belongs to, when it is knowable.

        ``None`` for a placeholder whose class the annotator has not settled --
        the caller then knows not to judge the verb against anything.
        """
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
            self._check_verb(str(value), self.class_of(action.target), action.target)
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
            text = text_of(value)
            if text and text not in FAILURE_REASONS:
                raise EditError(
                    f"{text!r} is not a failure reason ({', '.join(FAILURE_REASONS)})"
                )
            return text or None
        return checked_difficulty(value)

    def _create_instance(self, cls: str, attrs: dict, raw_name: str, verb: str) -> InstanceRec:
        """Build one instance of ``cls`` and register it in the instance table."""
        rec = self._build_instance(cls, attrs, raw_name, verb, frozenset())
        self.instances[rec.key] = rec
        return rec

    def _build_instance(
        self, cls: str, attrs: dict, raw_name: str, verb: str, staged: frozenset | set
    ) -> InstanceRec:
        """One instance of ``cls``, *not* registered yet.

        ``staged`` holds the keys of instances built in the same batch but not
        stored either, so a batch numbers itself consecutively while staying
        free to be thrown away whole.
        """
        if cls not in self.tax.classes:
            raise EditError(f"{cls!r} is not a taxonomy class")
        self._check_verb(verb, cls, f"new {cls}")
        attrs = {k: v for k, v in attrs.items() if k != "instance_nos"}
        self._check_discriminator(cls, attrs)
        key = instance_key(cls, attrs, self._next_ordinal(cls, attrs, staged))
        if key in self.instances or key in staged:  # only `chassis`, unique per desktop
            raise EditError(f"D{self.desktop:02d} already has an instance {key!r}")
        owner = attrs.get("cable_owner")
        return InstanceRec(
            key=key,
            desktop=self.desktop,
            cls=cls,
            attrs=attrs,
            cable=f"{CABLE_PREFIX}{owner}" if owner else None,
            raw_names=[raw_name] if raw_name else [],
        )

    def _check_discriminator(self, cls: str, attrs: dict) -> None:
        """The ``role`` / ``kind`` of a new instance must be one the class has.

        It becomes part of the instance key, so a typo would silently start a
        parallel ordinal run (``screw.mainboard.01`` next to
        ``screw.motherboard.07``). The rule lives here rather than in the
        dialog, so it holds however the instance is created. A class whose
        taxonomy entry lists the attribute with an empty value list (e.g.
        ``ram_latch.of``) accepts anything -- that list is deliberately open.
        """
        defined = self.tax.classes[cls].get("attrs") or {}
        for name in DISCRIMINATORS:
            value = attrs.get(name)
            if value is None:
                continue
            if name not in defined:
                raise EditError(f"class {cls!r} has no {name!r} attribute")
            allowed = defined[name]
            if allowed and value not in allowed:
                raise EditError(
                    f"{value!r} is not a {name} of {cls!r} ({', '.join(map(str, allowed))})"
                )

    def discriminator_of(self, cls: str) -> tuple[str, list[str]]:
        """The class's key discriminator and its vocabulary, or ``("", [])``.

        What the retarget dialog offers: ``("role", [...])`` for a screw,
        ``("kind", [...])`` for a drive or a connector, nothing for a class
        that carries no discriminator at all.
        """
        defined = self.tax.classes.get(cls, {}).get("attrs") or {}
        for name in DISCRIMINATORS:
            if name in defined:
                return name, [str(v) for v in defined[name]]
        return "", []

    def _next_ordinal(self, cls: str, attrs: dict, staged: frozenset | set = frozenset()) -> int:
        """One past the highest ordinal among the instances of the same group."""
        if cls == CHASSIS_KEY:
            return 1  # `chassis` carries no ordinal at all
        disc = str(attrs.get("role") or attrs.get("kind") or "")
        prefix = f"{cls}.{disc}." if disc else f"{cls}."
        top = 0
        for key in (*self.instances, *staged):
            if not key.startswith(prefix):
                continue
            match = _ORDINAL.match(key[len(prefix) - 1 :])
            if match:
                top = max(top, int(match.group(1)))
        return top + 1
