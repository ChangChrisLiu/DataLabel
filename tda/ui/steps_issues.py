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
  instance behind, and nothing else would ever mention it again.
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, Optional

from tda.core.logs import CHASSIS_KEY, NO_ACTION_TYPES, UNRESOLVED
from tda.core.model import ActionRec, InstanceRec, StepType
from tda.core.taxonomy import Taxonomy
from tda.ui.steps_values import DIFFICULTY_MAX, DIFFICULTY_MIN

__all__ = ["action_issues", "orphan_issues", "row_issues"]

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
    """Instances nothing operates on; ``chassis`` is implicit and exempt."""
    targeted = {action.target for action in actions}
    for key in sorted(instances):
        if key != CHASSIS_KEY and key not in targeted:
            yield f"no action references {key} - delete it or retarget a step at it"
