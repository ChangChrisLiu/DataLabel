"""Side information a method may use: area bands and already-drawn siblings.

Both are things the app knows when the annotator lands on frame ``j`` of the
reverse walk:

* the **class** of the part to add comes from the step log (``action.target``,
  e.g. ``screw.motherboard.03`` -> class ``screw``, role ``motherboard``);
* the **siblings** are shapes of that class drawn on frames ``>= j+1``, which the
  walk has already visited -- simulated with the LS drafts at those steps;
* the **area band** of the class, either the shipped
  ``configs/area_priors.yaml`` (fitted on these very drafts, so optimistic) or
  a leave-one-desktop-out band fitted here from the other two desktops.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l1_localise import base  # noqa: E402
from experiments.plan_b_probe import diff_eval as E  # noqa: E402

_LABEL_ATTR: Optional[dict[str, str]] = None


def label_attr(label: str) -> Optional[str]:
    """The role/kind an LS label carries (``configs/ls_label_map.yaml``)."""
    global _LABEL_ATTR
    if _LABEL_ATTR is None:
        raw = yaml.safe_load((base.REPO / "configs" / "ls_label_map.yaml")
                             .read_text("utf-8"))
        _LABEL_ATTR = {}
        for name, entry in (raw.get("labels") or {}).items():
            attrs = (entry or {}).get("attrs") or {}
            val = attrs.get("role") or attrs.get("kind")
            if val:
                _LABEL_ATTR[str(name)] = str(val)
    return _LABEL_ATTR.get(label)


def target_subtype(target: str) -> Optional[str]:
    """``screw.motherboard.03`` -> ``motherboard``; ``connector.02`` -> None."""
    parts = str(target).split(".")
    if len(parts) >= 3 and not parts[1].isdigit():
        return parts[1]
    return None


def sibling_pool(ev: base.Ev, by_frame) -> tuple[Optional[int], list]:
    """Nearest step ``s >= k`` with drafts of the target's class (and role).

    Returns ``(s, drafts)`` or ``(None, [])``.  At ``s >= k`` the removed part
    is gone by construction, so every such draft is a *different* part.
    """
    sub = target_subtype(ev.target)
    steps = sorted(s for (d, v, s) in by_frame
                   if d == ev.desktop and v == ev.view and s >= ev.step)
    for s in steps:
        pool = []
        for sh in by_frame[(ev.desktop, ev.view, s)].values():
            if sh.cls != ev.cls:
                continue
            if sub is not None:
                attr = label_attr(sh.label)
                if attr is not None and attr != sub:
                    continue
            pool.append(sh)
        if pool:
            return s, pool
    return None, []


class PrevParts:
    """Where the part removed at step ``k+1`` sits on frame ``k``.

    The reverse walk draws it on frame ``k`` just before it reaches ``k-1``, so
    its location is known when the guess for step ``k`` is made.  Resolved by
    location, like GT-B: the one draft of step ``k+1``'s target class on frame
    ``k`` with no same-class partner on frame ``k+1``.
    """

    SAME_IOU = 0.3

    def __init__(self, con, by_frame) -> None:
        from experiments.plan_b_probe.transfer.m4_m5_diff import (
            REMOVAL_VERBS, action_class,
        )

        classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}
        self.by_frame = by_frame
        self.actions: dict = defaultdict(list)
        for r in con.execute("SELECT desktop, step, verb, target FROM action"):
            if r["verb"] in REMOVAL_VERBS:
                cls = action_class(r["target"], classes)
                if cls:
                    self.actions[(int(r["desktop"]), int(r["step"]))].append(cls)

    def get(self, ev: base.Ev):
        """``(box, cls)`` of the part removed at ``k+1``, or ``None``."""
        nxt = ev.step + 1
        classes = self.actions.get((ev.desktop, nxt), [])
        if len(classes) != 1:
            return None
        cls = classes[0]
        before = self.by_frame.get((ev.desktop, ev.view, ev.step), {})
        after = self.by_frame.get((ev.desktop, ev.view, nxt), {})
        same_a = [s for s in after.values() if s.cls == cls]
        orphans = [s for s in before.values() if s.cls == cls
                   and all(base.box_iou(s.box, o.box) < self.SAME_IOU
                           for o in same_a)]
        if len(orphans) != 1:
            return None
        return tuple(int(v) for v in orphans[0].box), cls


def known_boxes_at(ev: base.Ev, by_frame) -> list[tuple[int, int, int, int]]:
    """Boxes of the same-class shapes already drawn on frame ``k = j+1``."""
    return [tuple(int(v) for v in sh.box)
            for sh in by_frame.get((ev.desktop, ev.view, ev.step), {}).values()
            if sh.cls == ev.cls]


class Priors:
    """Area bands in pixels for an event's ROI."""

    MIN_SAMPLES = 8

    def __init__(self, shapes) -> None:
        from tda.ui.app_priors import load_priors

        self.ship = load_priors()
        self.rois = {}
        for d in {s.desktop for s in shapes}:
            for v in base.VIEWS:
                if any(s.desktop == d and s.view == v for s in shapes):
                    self.rois[(d, v)] = E.roi_of(shapes, d, v)
        self.fracs: dict = defaultdict(list)       # (desktop, view, cls) -> [frac]
        for s in shapes:
            roi = self.rois.get((s.desktop, s.view))
            if roi is None or not s.cls:
                continue
            area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
            self.fracs[(s.desktop, s.view, s.cls)].append(s.area / area)

    def _pool(self, ev: base.Ev) -> list[float]:
        same_view = [f for (d, v, c), fs in self.fracs.items()
                     if d != ev.desktop and v == ev.view and c == ev.cls
                     for f in fs]
        pool = same_view
        if len(pool) < self.MIN_SAMPLES:
            pool = [f for (d, v, c), fs in self.fracs.items()
                    if d != ev.desktop and c == ev.cls for f in fs]
        return pool if len(pool) >= self.MIN_SAMPLES else []

    def _lodo(self, ev: base.Ev) -> Optional[tuple[float, float]]:
        pool = self._pool(ev)
        if not pool:
            return None
        return (float(np.percentile(pool, 5)), float(np.percentile(pool, 95)))

    def median_side(self, ev: base.Ev) -> Optional[float]:
        """``sqrt`` of the class's median draft area here (other desktops)."""
        pool = self._pool(ev)
        if not pool:
            return None
        roi_area = float((ev.roi[2] - ev.roi[0]) * (ev.roi[3] - ev.roi[1]))
        return float(np.sqrt(np.median(pool) * roi_area))

    def bands(self, ev: base.Ev) -> dict:
        roi_area = float((ev.roi[2] - ev.roi[0]) * (ev.roi[3] - ev.roi[1]))
        ship_frac = E.priors_for(ev.cls, self.ship)
        lodo_frac = self._lodo(ev)
        return {
            "ship_frac": ship_frac,
            "ship": ([round(ship_frac[0] * roi_area, 1),
                      round(ship_frac[1] * roi_area, 1)] if ship_frac else None),
            "lodo_frac": lodo_frac,
            "lodo": ([round(lodo_frac[0] * roi_area, 1),
                      round(lodo_frac[1] * roi_area, 1)] if lodo_frac else None),
        }
