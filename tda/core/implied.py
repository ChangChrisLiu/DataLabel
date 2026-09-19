"""The part every frame shows but the log never touches (user decision C7).

:mod:`tda.core.logs` only ever creates an instance a *step* names. That is the
right rule almost everywhere -- a desktop the sheet never mentions a PSU latch
on probably has none -- but it breaks on the four teardowns that stop before the
motherboard is lifted out: D49 ends at "CPU", D62 at "Remove half screws on the
motherboard", D63 and D64 unscrew the board and then only "change a direction".
Those sheets unscrew it, unplug it and end, so:

* 33 connectors and screws are left pointing at the *class* ``motherboard``,
  which has no instance on those desktops, and
* the board itself -- plainly visible in every single frame of them -- carries
  no identity, so it can never be labelled in the detection or segmentation
  export.

:func:`implied_instances` fills exactly that gap and nothing else. For a class
``configs/taxonomy.yaml`` lists under ``implied_when_referenced``, and only when
the desktop has no real instance of it yet *and* something references it, it
returns one ``<class>.01`` record carrying ``attrs["implied"] = True``. Nothing
operates on it, so the state machine leaves it ``installed`` / ``in_chassis``
for the whole sequence, which is precisely the truth about a board that was
never taken out. Afterwards the ordinary heuristic of
:mod:`tda.core.graph_infer` resolves the 33 references onto it.

The function is pure and idempotent: one instance per class at most, never one
where a real instance exists, and running it twice creates nothing the second
time. Stage S1 asks about every one of them
(:func:`tda.ui.steps_issues.unresolved_issues`), because "the dataset should
contain this part" is a judgement a human may overrule by deleting the row.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core.graph_infer import SCREW_ROLE_CLASSES, is_provisional, real_instances
from tda.core.model import ActionRec, InstanceRec
from tda.core.taxonomy import Taxonomy

__all__ = [
    "IMPLIED_ATTR",
    "NOTE_ATTR",
    "OP_KIND",
    "implied_instances",
    "is_implied",
    "referencing_instances",
]

#: ``attrs`` flag marking a record this module created rather than the log.
IMPLIED_ATTR = "implied"
#: ``attrs`` key holding the one-line explanation shown in stage S1.
NOTE_ATTR = "note"
#: ``op_log.kind`` written for every implied instance, so it can be audited and
#: undone record by record like any other machine-made change.
OP_KIND = "implied_instance"


def is_implied(rec: InstanceRec) -> bool:
    """Was this identity row implied rather than named by a step?"""
    return bool(rec.attrs.get(IMPLIED_ATTR))


def referencing_instances(instances: dict[str, InstanceRec], cls: str) -> list[str]:
    """Keys of the instances that point at ``cls`` without naming an instance.

    Two references exist in practice and both come straight out of the sheet:
    a connector whose ``socket_host`` is the bare class name (``logs.py`` writes
    ``"motherboard"`` when the step name says which part carries the socket but
    not which one), and a screw whose ``role`` names the class
    (:data:`~tda.core.graph_infer.SCREW_ROLE_CLASSES`). Those are exactly the
    rows :func:`~tda.core.graph_infer.unresolved_relations` reports as
    ``no candidate``.

    Provisional ``ls:*`` drafts are **not** references. They carry no relations
    yet by construction, so nothing about them is unresolved and nothing about
    them is evidence that the machine has this part: D66 holds eight
    ``ls:Motherboard Screw#k`` drafts and no imported sheet at all, and implying
    a board from those would invent an instance out of a drawing.
    """
    out: list[str] = []
    for key, rec in sorted(instances.items()):
        if is_provisional(key):
            continue
        if rec.cls == "connector" and str(rec.socket_host or "").strip() == cls:
            out.append(key)
        elif rec.cls == "screw":
            role = str(rec.attrs.get("role") or "")
            if cls in SCREW_ROLE_CLASSES.get(role, ()):
                out.append(key)
    return out


def _note(referees: list[str]) -> str:
    """The one line stage S1 shows about an implied instance."""
    count = len(referees)
    return (
        f"implied: referenced by {count} instance{'' if count == 1 else 's'}, "
        f"never operated in the log"
    )


def implied_instances(
    instances: dict[str, InstanceRec],
    actions: Optional[Iterable[ActionRec]],
    tax: Taxonomy,
    declined: Optional[Iterable[str]] = None,
) -> list[InstanceRec]:
    """The instances this desktop obviously has but its log never names.

    One record per class of ``tax.implied_when_referenced`` that

    * is not in ``declined`` -- the classes whose implied instance this
      desktop's annotator has already deleted in S1
      (:meth:`tda.core.db.Db.declined_implied`). Implying is a judgement, and
      "no" is an answer that has to outlive the next ``import-logs``: without
      this, the deleted board came back on every re-import with a mask on every
      frame;
    * has **no** real (non-``ls:``) instance on this desktop -- a Label Studio
      draft is a draft, not a settled identity, and cannot stand in for one;
    * is referenced by at least one other instance
      (:func:`referencing_instances`); and
    * has no action aimed at its ``<class>.01`` key, which would mean the log
      *did* operate on it and something else is wrong.

    The caller adds them to ``instances`` **before** running
    :func:`~tda.core.graph_infer.infer_relational_fields`, which then resolves
    the references onto them. Returns an empty list -- never ``None`` -- when
    there is nothing to imply, so a caller can write ``for rec in ...``
    unconditionally.
    """
    if not instances:
        return []
    desktop = next(iter(instances.values())).desktop
    targeted = {a.target for a in (actions or ())}
    refused = {str(c) for c in (declined or ())}
    out: list[InstanceRec] = []
    for cls in tax.implied_when_referenced:
        key = f"{cls}.01"
        if cls in refused:
            continue
        if key in instances or real_instances(instances, cls) or key in targeted:
            continue
        referees = referencing_instances(instances, cls)
        if not referees:
            continue
        out.append(InstanceRec(
            key=key,
            desktop=desktop,
            cls=cls,
            attrs={IMPLIED_ATTR: True, NOTE_ATTR: _note(referees)},
        ))
    return out
