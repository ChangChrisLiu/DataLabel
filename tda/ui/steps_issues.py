"""Re-derive stage S1's open questions from the drafts on screen.

``LogImport.issues`` is written to the import report, not to the database, so
the step-table panel cannot replay it -- and should not, because the annotator
is looking at edited drafts, not at the CSV. These pure functions rebuild the
same questions from whatever :class:`~tda.ui.steps_model.StepTableData` holds
right now, which also means an issue disappears the moment it is answered.

What is asked about:

* a compound row still to split, and any target left unresolved (``?``);
* a step type that ought to carry an action but has none;
* a verb, tool or direction outside the taxonomy, or a verb that does not apply
  to its target's class (spec 6.3-6.5);
* a difficulty outside 1-5, a failed attempt without a ``failure_reason`` and a
  ``failure_reason`` on a successful one (spec 3.1);
* an instance no action targets any more -- retargeting a row leaves its old
  instance behind, and nothing else would ever mention it again;
* an instance whose ``parent`` / ``mounted_on`` / ``fastens`` / ``socket_host``
  names something that is not an instance, so a stale pointer can never sit in
  the table unnoticed;
* last, and lowest priority, what the relational heuristic of spec 7.3
  (:func:`tda.core.graph_rules.infer_relational_fields`) could not settle by
  itself -- a connector still pointing at a bare class name and a captive screw
  with no parent to leave the chassis with.
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, Optional

from tda.core.graph_infer import is_provisional, real_instances
from tda.core.implied import is_implied
from tda.core.logs import CHASSIS_KEY, NO_ACTION_TYPES, UNRESOLVED
from tda.core.model import ActionRec, InstanceRec, StepType
from tda.core.taxonomy import Taxonomy
from tda.ui.steps_values import DIFFICULTY_MAX, DIFFICULTY_MIN, RELATION_FIELDS

__all__ = [
    "action_issues", "dangling_issues", "draft_issues", "orphan_issues", "row_issues",
    "unresolved_issues",
]

#: ``StepTableData.class_of``: the taxonomy class of a target, when knowable.
ClassOf = Callable[[str], Optional[str]]


def row_issues(row, tax: Taxonomy, class_of: ClassOf) -> Iterator[str]:
    """Everything still open about one step row."""
    if row.step_type == StepType.COMPOUND.value:
        yield f"compound row {row.raw_name!r} - split it into one action per target"
    if not row.actions and row.step_type not in NO_ACTION_TYPES:
        yield f"{row.step_type} step {row.raw_name!r} has no action"
    for action in row.actions:
        yield from action_issues(action, tax, class_of)


def action_issues(action: ActionRec, tax: Taxonomy, class_of: ClassOf) -> Iterator[str]:
    """Everything still open about one action."""
    if UNRESOLVED in action.target:
        yield f"unresolved target {action.target!r} - pick or create the instance"
    cls = class_of(action.target)
    spec = tax.verbs.get(action.verb)
    if spec is None:
        yield f"verb {action.verb!r} is not in the taxonomy"
    elif cls is not None and cls not in spec["applies_to"]:
        yield f"verb {action.verb!r} does not apply to class {cls!r}"
    if action.tool not in tax.tools:
        yield f"tool {action.tool!r} is not in the taxonomy - pick one"
    if action.direction not in tax.directions:
        yield f"direction {action.direction!r} is not in the taxonomy"
    if action.difficulty is not None and not DIFFICULTY_MIN <= action.difficulty <= DIFFICULTY_MAX:
        yield f"difficulty {action.difficulty!r} is outside {DIFFICULTY_MIN}-{DIFFICULTY_MAX}"
    if action.result == "failed" and not action.failure_reason:
        yield "a failed attempt needs a failure_reason"
    if action.result == "success" and action.failure_reason:
        yield f"failure_reason {action.failure_reason!r} on a successful action"


def orphan_issues(
    instances: dict[str, InstanceRec], actions: Iterable[ActionRec]
) -> Iterator[str]:
    """Instances nothing operates on; ``chassis`` is implicit and exempt.

    So is an *implied* instance (:mod:`tda.core.implied`): "no action names it"
    is its definition, not a defect, and :func:`unresolved_issues` already asks
    the one question that matters about it. So is a Label Studio draft, for the
    same reason and more strongly -- see :func:`draft_issues`.
    """
    targeted = {action.target for action in actions}
    for key in sorted(instances):
        if (key == CHASSIS_KEY or key in targeted or is_provisional(key)
                or is_implied(instances[key])):
            continue
        yield f"no action references {key} - delete it or retarget a step at it"


def draft_issues(instances: dict[str, InstanceRec]) -> Iterator[str]:
    """One line saying how many Label Studio drafts this desktop carries.

    They are not orphaned parts of the machine and never will be: nothing
    references them because they are the team's old tracings, waiting for an
    annotator to adopt them onto real instances (spec 3.2). Listed one by one
    they were 63 lines on D13 telling the annotator to delete or retarget each
    -- work that does not exist, burying the work that does.

    Yielded last, after every real question, and nothing follows from it: the
    drafts stay deletable in the ordinary way.
    """
    drafts = [key for key in instances if is_provisional(key)]
    if drafts:
        yield (
            f"{len(drafts)} Label Studio drafts (ls:*) on this desktop - they are "
            f"reference material and need no action"
        )


def dangling_issues(instances: dict[str, InstanceRec], tax: Taxonomy) -> Iterator[str]:
    """Relations pointing at something that is not an instance of this desktop.

    A bare taxonomy class name is *not* dangling: the log importer writes
    ``socket_host = "motherboard"`` when it knows which class carries the
    socket but not which instance, and every motherboard-side connector of
    every desktop comes out that way. Narrowing those down is ordinary S1 work,
    not a broken pointer -- whereas a key like ``screw.cpu_cooler.02`` that no
    longer exists always is one, and is what this catches.
    """
    for key in sorted(instances):
        inst = instances[key]
        for name in RELATION_FIELDS:
            value = getattr(inst, name)
            if value and value not in instances and value not in tax.classes:
                yield (
                    f"{key}.{name} points at {value}, which is not an instance "
                    f"- clear it or recreate the instance"
                )


def unresolved_issues(
    instances: dict[str, InstanceRec], tax: Taxonomy
) -> Iterator[str]:
    """What the relational heuristic left open -- the lowest-priority questions.

    :func:`tda.core.graph_rules.infer_relational_fields` fills what it can and
    guesses at nothing, so two gaps survive it and would otherwise be invisible
    until an export or the constraint panel went wrong:

    * **unresolved socket host** -- ``connector.socket_host`` is still the bare
      taxonomy class the importer wrote, because the desktop has no unique
      instance of it. Rule 7.2 (the connector gates the part it plugs into)
      cannot fire on a class name. Two different jobs hide behind that, so they
      are two different lines: when the desktop has *several* instances one has
      to be picked, and when it has *none* one has to be created first -- which
      is a decision about what this desktop tracks at all (user decision C7),
      never something to invent automatically.
    * **captive screw without parent** -- a captive screw stays in its part when
      the part comes out (spec 7.1), which only happens if ``parent`` names it;
      without one the screw is left behind ``loosened`` and in the chassis, and
      the annotator is asked for its shape on every later frame.
    * **host-mounted instance without parent** -- the same defect on the classes
      that declare a ``host_class`` (``ram_latch`` and ``cpu_socket_lever`` ride
      on the motherboard). The desktop has no unique host to hang it on, so the
      latch stands in the chassis once the board is gone and is a missing shape
      on every frame after it.
    * **implied instance** -- a part :mod:`tda.core.implied` created because the
      desktop clearly has one and the log simply stops before touching it (the
      motherboard of D49/D62/D63/D64). It is a judgement, so it is put to the
      annotator once: keeping it costs a mask on every frame, deleting it leaves
      the references unresolved again.

    Deliberately yielded *after* :func:`orphan_issues` and
    :func:`dangling_issues`: none is a broken record, only work still to do.
    Provisional ``ls:*`` drafts carry no relations yet and are skipped.
    """
    for key in sorted(instances):
        inst = instances[key]
        if is_provisional(key):
            continue
        host = inst.socket_host
        if inst.cls == "connector" and host and host not in instances and host in tax.classes:
            yield _socket_host_issue(instances, key, host)
        if inst.cls == "screw" and inst.attrs.get("captive") and not inst.parent:
            yield (
                f"captive screw without parent: {key} is captive but leaves the "
                f"chassis with nothing - name the part it stays in"
            )
        mount = tax.host_class(inst.cls)
        if mount and not inst.parent:
            yield (
                f"host-mounted instance without parent: {key} rides on the "
                f"{mount} and has none - it will be asked for a shape on every "
                f"frame after the {mount} is out"
            )
        if is_implied(inst):
            yield (
                f"implied instance {key}: never operated in the log - keep it (it "
                f"gets a mask on every frame) or delete it in the Instances tab; "
                f"deleting it is remembered, so no re-import brings it back"
            )


def _socket_host_issue(instances: dict[str, InstanceRec], key: str, host: str) -> str:
    """The socket-host question, worded for the work it actually needs."""
    if real_instances(instances, host):
        return (
            f"unresolved socket host: {key}.socket_host is still the class "
            f"{host!r} - name the instance it plugs into"
        )
    return (
        f"unresolved socket host: class {host!r} has no instance on this desktop "
        f"- add the instance in the Instances tab or leave it unresolved"
    )
