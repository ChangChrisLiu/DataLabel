"""The closed operation set of a VLM rationale chain (spec 8.2).

Spec 8.2 fixes seven operations and nothing else may appear in a chain::

    observe | recall_relation | check_precondition | propagate_state
    compare_frames | conclude | abstain

Two rules come with them and are enforced here rather than remembered at each
call site:

* **every ``observe`` carries a box and a ``visibility``** -- it is the step the
  grounded-accuracy metric is measured on, so a value nobody can point at is
  :func:`propagate_state`, not a boxless observation;
* a chain ends in exactly one terminal, :func:`conclude` or :func:`abstain`, and
  its ``depth`` is the number of steps including that terminal, which is what
  spec 8.2 stratifies the evaluation by.

Everything here is pure and JSON-shaped; :mod:`tda.core.export.vlm_tasks` builds
the chains and :mod:`tda.core.export.vlm` writes them out.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "OPS",
    "TERMINALS",
    "abstain",
    "check_precondition",
    "compare_frames",
    "conclude",
    "observe",
    "propagate_state",
    "recall_relation",
    "rationale",
    "validate_chain",
]

#: Spec 8.2's closed operation set, in the order the spec lists them.
OPS = ("observe", "recall_relation", "check_precondition", "propagate_state",
       "compare_frames", "conclude", "abstain")

#: The two operations a chain may end on.
TERMINALS = ("conclude", "abstain")


def observe(instance: str, value: Any, view: str, bbox: Sequence[float],
            visibility: Optional[str], step: Optional[int] = None) -> dict:
    """Read a value off this image, naming the box and the visibility it came from.

    ``bbox`` is required and must be non-empty: an ``observe`` without one is
    the one shape of chain the grounding metric cannot score, and the whole
    point of spec 8.2's rationale format is that the intermediate boxes are
    checkable. Use :func:`propagate_state` for a value this frame cannot show.
    """
    if not bbox:
        raise ValueError(f"observe({instance!r}) needs the box it read the value off")
    evidence: dict[str, Any] = {"view": view, "bbox": list(bbox),
                                "visibility": visibility}
    if step is not None:
        evidence["step"] = int(step)
    return {"op": "observe", "target": instance, "value": value, "evidence": evidence}


def propagate_state(instance: str, value: Any) -> dict:
    """Carry a value over from the event log: not seen in this frame."""
    return {"op": "propagate_state", "target": instance, "value": value}


def recall_relation(edge_type: str, target: str, blocker: str,
                    necessity: str = "required", mode: Optional[str] = None) -> dict:
    """Name one hard-constraint edge of spec 7.2 the conclusion rests on."""
    step: dict[str, Any] = {"op": "recall_relation", "edge": [target, edge_type, blocker],
                            "necessity": necessity}
    if mode:
        step["mode"] = mode
    return step


def check_precondition(blocker: str, required: Iterable[str], actual: Optional[str],
                       satisfied: bool) -> dict:
    """Compare one blocker's state against what spec 7.2 requires of it."""
    return {"op": "check_precondition", "target": blocker,
            "required": sorted(required), "actual": actual,
            "satisfied": bool(satisfied)}


def compare_frames(steps: Sequence[int], view: str) -> dict:
    """Put two frames of one view side by side (V3, V12, V15)."""
    return {"op": "compare_frames", "frames": [int(s) for s in steps], "view": view}


def conclude(value: Any = None) -> dict:
    step: dict[str, Any] = {"op": "conclude"}
    if value is not None:
        step["value"] = value
    return step


def abstain(reason: str) -> dict:
    """End a chain by refusing: this view cannot decide (spec 8.2 principle 5)."""
    return {"op": "abstain", "reason": reason}


def rationale(steps: Sequence[dict], terminal: Optional[dict] = None) -> dict:
    """Close a chain and record its depth.

    ``terminal`` defaults to a bare :func:`conclude`; pass :func:`abstain` for a
    chain whose answer is "this view cannot say". A chain that already ends on a
    terminal is left alone, so a caller may build the whole thing itself.
    """
    chain = list(steps)
    if not chain or chain[-1].get("op") not in TERMINALS:
        chain.append(terminal if terminal is not None else conclude())
    validate_chain(chain)
    return {"depth": len(chain), "steps": chain}


def validate_chain(steps: Sequence[dict]) -> None:
    """Raise when a chain leaves the closed set or ends on a non-terminal."""
    for step in steps:
        if step.get("op") not in OPS:
            raise ValueError(f"{step.get('op')!r} is not one of spec 8.2's operations")
        if step["op"] == "observe" and not (step.get("evidence") or {}).get("bbox"):
            raise ValueError(f"observe({step.get('target')!r}) carries no box")
    if not steps or steps[-1].get("op") not in TERMINALS:
        raise ValueError("a rationale chain must end on conclude or abstain")
