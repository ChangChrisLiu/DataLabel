"""Family templates for the constraint graph (spec 7.3 item 2).

Machines of one model family share a "part slot map": the first desktop of a
family is completed by hand, saved as a template, and every sibling gets its
edges instantiated from it. The template is slot-level, never instance-level --
each endpoint is stored as the instance's ``slot_id``, falling back to its
``instance_key`` when the instance carries no slot -- so the YAML stays readable
and reviewable, which is the point of spec 7.3.

A sibling that lacks a slot (fewer mainboard screws, no optical drive) simply
does not get that edge; the skipped slots are reported rather than guessed at.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from tda.core.graph_rules import Edge
from tda.core.model import InstanceRec

__all__ = ["apply_template", "load_template", "save_template"]

TEMPLATE_VERSION = 1

#: The edge fields a template carries. ``source`` and ``status`` are set on
#: instantiation (``template`` / ``proposed``), and ``evidence_step`` points at a
#: step of the *donor* desktop, so neither travels.
_FIELDS = ("necessity", "mode", "reason")


def _slot_of(instances: dict[str, InstanceRec], key: str) -> Optional[str]:
    """The template slot an instance key stands for."""
    rec = instances.get(key)
    if rec is None:
        return None
    return rec.slot_id or rec.key


def save_template(
    path: str | Path,
    instances: dict[str, InstanceRec],
    edges: list[Edge],
) -> int:
    """Write ``edges`` to ``path`` as a slot-level YAML; returns how many.

    An edge whose endpoints are not instances of this desktop is dropped -- it
    has no slot and could not be instantiated anywhere.
    """
    rows = []
    for edge in edges:
        target = _slot_of(instances, edge.target)
        blocker = _slot_of(instances, edge.blocker)
        if target is None or blocker is None:
            continue
        row = {"type": edge.type, "target": target, "blocker": blocker}
        row.update({name: getattr(edge, name) for name in _FIELDS})
        rows.append(row)
    doc = {"version": TEMPLATE_VERSION, "edges": rows}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    return len(rows)


def load_template(path: str | Path) -> list[dict]:
    """The raw slot-level rows of a template file."""
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    return list(doc.get("edges") or [])


def _slot_index(instances: dict[str, InstanceRec]) -> dict[str, list[str]]:
    """slot (or instance key) -> the keys that fill it, sorted."""
    index: dict[str, list[str]] = {}
    for key, rec in sorted(instances.items()):
        index.setdefault(rec.slot_id or rec.key, []).append(key)
    return index


def apply_template(
    path: str | Path,
    instances: dict[str, InstanceRec],
    report: Optional[list[str]] = None,
) -> list[Edge]:
    """Instantiate a family template onto another desktop's instances.

    Every endpoint slot is looked up among ``instances``; an edge is kept only
    when both ends resolve to exactly one instance. A slot this machine does not
    have -- or one filled ambiguously by several instances -- is skipped, and a
    line saying so is appended to ``report`` when the caller passes a list. The
    report is per *slot*, not per edge: one missing screw slot that appears in
    four edges is one line, because that is the one thing a human has to fix.

    The edges come back with ``source="template"`` and ``status="proposed"``:
    a template is a strong suggestion, not an observation, and the S1 panel is
    where a human accepts it.
    """
    index = _slot_index(instances)
    notes = report if report is not None else []
    out: list[Edge] = []

    for row in load_template(path):
        target = _one(index, row.get("target"), notes)
        blocker = _one(index, row.get("blocker"), notes)
        if target is None or blocker is None:
            continue
        out.append(
            Edge(
                type=str(row.get("type") or ""),
                target=target,
                blocker=blocker,
                necessity=str(row.get("necessity") or "required"),
                mode=row.get("mode"),
                reason=str(row.get("reason") or ""),
                source="template",
                status="proposed",
            )
        )
    return out


def _one(index: dict[str, list[str]], slot: Optional[str], notes: list[str]) -> Optional[str]:
    """The single instance filling ``slot``, or ``None`` plus a note."""
    if not slot:
        _note(notes, "template edge with an empty endpoint slot skipped")
        return None
    keys = index.get(str(slot))
    if not keys:
        _note(notes, f"slot {slot!r} has no instance on this desktop; edge skipped")
        return None
    if len(keys) > 1:
        _note(notes, f"slot {slot!r} is filled by {len(keys)} instances; edge skipped")
        return None
    return keys[0]


def _note(notes: list[str], text: str) -> None:
    """Append one report line, once -- a slot missing from four edges is one gap."""
    if text not in notes:
        notes.append(text)
