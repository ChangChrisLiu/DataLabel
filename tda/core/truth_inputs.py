"""Gathering one frame's compiler inputs out of the database (spec 3.3).

:mod:`tda.core.compiler` is a pure function over explicit inputs;
:mod:`tda.core.truth` is the persisted truth table. This module is the thin
layer in between: it reads the state machine's inputs (instances, events),
folds them into the state at one logical step, and collects the geometry of
that frame -- keyframes, z-order, pairwise overrides, occluders, frame
overrides and the registration transform -- into one :class:`FrameInputs`.

Nothing here writes to the database except :func:`frame_hw`, which stores the
image size it measured or inferred back onto the frame row, so the next
compilation of that frame (and its ``input_hash``) uses the same canvas.

An :class:`InputCache` makes a sweep over many steps of the same view read the
instance table, the event log and the keyframes once instead of once per step.
It is deliberately **per call**: a long-lived cache would go stale the moment an
annotator edits a shape, so :func:`gather` builds a fresh one when it is not
handed one, and :meth:`tda.core.truth.TruthService.refresh_range` keeps one for
the duration of its own loop only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from tda.core.db import Db
from tda.core.model import (
    FrameKey,
    FrameOverride,
    InstanceRec,
    OccluderMask,
    PairOverride,
    ShapeKeyframe,
    Similarity,
    StateEvent,
    ZOrderRec,
)
from tda.core.model import Placement
from tda.core.states import (
    FrameState,
    events_from_actions,
    needs_geom,
    state_at,
)
from tda.core.taxonomy import Taxonomy

__all__ = [
    "DEFAULT_POSE_SEGMENT",
    "annotatable_steps",
    "HW_INFERRED",
    "HW_MEASURED",
    "VIEW_HW",
    "FrameInputs",
    "InputCache",
    "events_of",
    "frame_hw",
    "gather",
    "infer_hw",
    "instances_of",
    "pose_segment_of",
    "state_of",
]

#: The pose segment a frame belongs to when nothing is recorded. Segment 0 is
#: what :meth:`tda.core.db.Db.pose_segment_for` reports for "no segment
#: recorded", so the shapes of a database without pose segments all live in 1.
DEFAULT_POSE_SEGMENT = 1

#: ``(H, W)`` of each view's original images (spec 2.1/2.4).
VIEW_HW: dict[str, tuple[int, int]] = {
    "scan": (1600, 1600),
    "oak1": (3040, 4032),
    "oak2": (3040, 4032),
    "rs": (720, 1280),
}

#: Values of ``aux["hw_source"]``: the view's nominal size vs. the real image's.
HW_INFERRED = "inferred"
HW_MEASURED = "measured"

ON_BENCH = Placement.ON_BENCH.value


# --------------------------------------------------------------------------- #
# image size
# --------------------------------------------------------------------------- #
def annotatable_steps(db: Db, desktop: int, view: str, steps) -> list[int]:
    """The subset of ``steps`` that has an image to compile against.

    A logical step whose frame row is absent, or flagged ``missing``, carries no
    canvas: compiling it would fall back to the view's nominal size and produce
    masks in the wrong coordinates. The state machine and the shape anchors
    still run through it (spec 4.2, 缺帧处理), only the truth table skips it.
    """
    out = []
    for step in sorted({int(s) for s in steps}):
        row = db.get_frame(FrameKey(desktop, step, view))
        if row is not None and not row.get("missing"):
            out.append(step)
    return out


def infer_hw(view: str) -> tuple[int, int]:
    """The nominal ``(H, W)`` of a view's images; raises for an unknown view."""
    known = VIEW_HW.get(view)
    if known is not None:
        return known
    if view.startswith("oak"):
        return VIEW_HW["oak1"]
    raise ValueError(f"cannot infer the image size of view {view!r}")


def _read_hw(row: Optional[dict]) -> Optional[tuple[int, int]]:
    """``(H, W)`` recorded on a frame row, or ``None``.

    Read from the row itself first (a future ``hw`` column) and then from
    ``aux``, which is where :func:`frame_hw` puts it today. A malformed value
    is ignored rather than raised, so one bad row does not block a refresh.
    """
    if not row:
        return None
    for candidate in (row.get("hw"), (row.get("aux") or {}).get("hw")):
        if candidate is None:
            continue
        try:
            height, width = (int(v) for v in candidate)
        except (TypeError, ValueError):
            continue
        if height > 0 and width > 0:
            return (height, width)
    return None


def _image_path(row: Optional[dict]) -> Optional[str]:
    """Where this frame's pixels are: the cached copy, else the frame's own path."""
    if not row:
        return None
    aux = row.get("aux") or {}
    for candidate in (aux.get("cache_path"), row.get("path")):
        if candidate:
            return str(candidate)
    return None


def _measure_hw(row: Optional[dict]) -> Optional[tuple[int, int]]:
    """``(H, W)`` read off the image file, or ``None`` when there is none to read."""
    path = _image_path(row)
    if not path:
        return None
    try:
        if not Path(path).exists():
            return None
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError):
        return None
    if image is None or getattr(image, "ndim", 0) < 2:
        return None
    height, width = (int(v) for v in image.shape[:2])
    return (height, width) if height > 0 and width > 0 else None


def _store_hw(
    db: Db, key: FrameKey, row: Optional[dict], hw: tuple[int, int], source: str
) -> None:
    """Write a size and where it came from onto the frame row, keeping path/ts."""
    aux = dict((row or {}).get("aux") or {})
    aux["hw"] = [int(hw[0]), int(hw[1])]
    aux["hw_source"] = source
    db.upsert_frame(key, (row or {}).get("path"), aux, (row or {}).get("ts"))


def frame_hw(db: Db, key: FrameKey) -> tuple[int, int]:
    """The frame's image size: measured from the image, or inferred from the view.

    The canvas enters every compiled mask and the frame's ``input_hash``, so it
    must not change silently between two runs: whatever this returns is written
    back onto the frame row together with ``aux["hw_source"]``
    (:data:`HW_MEASURED` or :data:`HW_INFERRED`). The image is read at most
    once -- a measured size is never questioned again -- and a measurement
    supersedes an earlier guess as soon as the cached image is there.
    """
    row = db.get_frame(key)
    stored = _read_hw(row)
    source = ((row or {}).get("aux") or {}).get("hw_source")
    if stored is not None and source == HW_MEASURED:
        return stored
    measured = _measure_hw(row)
    if measured is not None:
        _store_hw(db, key, row, measured, HW_MEASURED)
        return measured
    if stored is not None:
        return stored
    inferred = infer_hw(key.view)
    _store_hw(db, key, row, inferred, HW_INFERRED)
    return inferred


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
@dataclass
class InputCache:
    """Per-sweep memo of the reads that do not depend on the logical step."""

    instances: dict[int, dict[str, InstanceRec]] = field(default_factory=dict)
    events: dict[int, list[StateEvent]] = field(default_factory=dict)
    keyframes: dict[tuple[int, str], dict[str, list[ShapeKeyframe]]] = field(
        default_factory=dict
    )
    zorder: dict[tuple[int, str, int], ZOrderRec] = field(default_factory=dict)
    overrides: dict[tuple[int, str, int], list[PairOverride]] = field(default_factory=dict)


def instances_of(
    db: Db, desktop: int, cache: Optional[InputCache] = None
) -> dict[str, InstanceRec]:
    """The instance table of one desktop."""
    if cache is None:
        return db.instances(desktop)
    if desktop not in cache.instances:
        cache.instances[desktop] = db.instances(desktop)
    return cache.instances[desktop]


def _merge_events(derived: list[StateEvent], manual: list[StateEvent]) -> list[StateEvent]:
    """Both logs in one chronological list, manual events last within a step.

    Sorting is stable and keyed only on ``(step, source)``, so each log keeps
    its own order and a hand-written event always folds in *after* the derived
    events of the same step -- which is what lets an annotator correct what the
    step table says without having to delete the action.
    """
    tagged = [(event.step, 0, event) for event in derived]
    tagged += [(event.step, 1, event) for event in manual]
    return [event for _, _, event in sorted(tagged, key=lambda item: (item[0], item[1]))]


def events_of(
    db: Db, tax: Taxonomy, desktop: int, cache: Optional[InputCache] = None
) -> list[StateEvent]:
    """The state-event log: always derived from the actions, manual events on top.

    The recorded actions are the authority on what happened (spec 6.3), so they
    are compiled on every read and the stored ``auto=True`` rows -- a cached
    copy of exactly that -- are ignored. Stored events an annotator entered by
    hand (``auto=False``) are merged in afterwards: they correct or complete the
    derived log instead of replacing it, so one manual note can no longer erase
    every action's effect.

    The derived log is **not** persisted: it is a view of the step table, and
    writing it here would compete with :meth:`tda.core.db.Db.replace_events`.
    """
    if cache is not None and desktop in cache.events:
        return cache.events[desktop]
    instances = instances_of(db, desktop, cache)
    derived = events_from_actions(instances, db.actions(desktop), tax)
    manual = [event for event in db.events(desktop) if not event.auto]
    events = _merge_events(derived, manual)
    if cache is not None:
        cache.events[desktop] = events
    return events


def _keyframes_of(
    db: Db, desktop: int, view: str, cache: Optional[InputCache] = None
) -> dict[str, list[ShapeKeyframe]]:
    """``instance -> keyframes`` of one view (every segment and placement)."""
    if cache is not None and (desktop, view) in cache.keyframes:
        return cache.keyframes[(desktop, view)]
    chains: dict[str, list[ShapeKeyframe]] = {}
    for kf in db.keyframes(desktop, view):
        chains.setdefault(kf.instance, []).append(kf)
    if cache is not None:
        cache.keyframes[(desktop, view)] = chains
    return chains


def _zorder_of(
    db: Db, desktop: int, view: str, seg: int, cache: Optional[InputCache] = None
) -> ZOrderRec:
    if cache is None:
        return db.zorder(desktop, view, seg)
    if (desktop, view, seg) not in cache.zorder:
        cache.zorder[(desktop, view, seg)] = db.zorder(desktop, view, seg)
    return cache.zorder[(desktop, view, seg)]


def _overrides_of(
    db: Db, desktop: int, view: str, seg: int, cache: Optional[InputCache] = None
) -> list[PairOverride]:
    if cache is None:
        return db.pair_overrides(desktop, view, seg)
    if (desktop, view, seg) not in cache.overrides:
        cache.overrides[(desktop, view, seg)] = db.pair_overrides(desktop, view, seg)
    return cache.overrides[(desktop, view, seg)]


# --------------------------------------------------------------------------- #
# state and segment
# --------------------------------------------------------------------------- #
def state_of(
    db: Db, tax: Taxonomy, desktop: int, step: int, cache: Optional[InputCache] = None
) -> FrameState:
    """Every instance's state and placement at one logical step (spec 3.3.1)."""
    return state_at(
        instances_of(db, desktop, cache), events_of(db, tax, desktop, cache), step, tax
    )


def pose_segment_of(db: Db, key: FrameKey, cache: Optional[InputCache] = None) -> int:
    """The pose segment of one frame, :data:`DEFAULT_POSE_SEGMENT` when unknown."""
    pose = db.pose_segment_for(key)
    seg = None if pose is None else pose.get("seg")
    return DEFAULT_POSE_SEGMENT if not seg else int(seg)


# --------------------------------------------------------------------------- #
# the whole bundle
# --------------------------------------------------------------------------- #
def _seen_here(needs: dict[str, str], state: FrameState,
               bench_roi) -> dict[str, str]:
    """Drop what this view cannot see: the staging area, when it has none.

    A part lying on the bench is not annotated on a view without a bench ROI, so
    it is not an instance of that frame at all -- not missing, not compiled, and
    not something ``bench_annotated`` is about (spec 4.2 item 1).
    """
    if bench_roi is not None:
        return needs
    return {inst: kind for inst, kind in needs.items()
            if state[inst].placement != ON_BENCH}


@dataclass
class FrameInputs:
    """Everything :func:`tda.core.compiler.compile_frame` needs for one frame."""

    key: FrameKey
    hw: tuple[int, int]
    needs: dict[str, str]
    keyframes: dict[str, list[ShapeKeyframe]]
    zorder: ZOrderRec
    overrides: list[PairOverride]
    occluders: list[OccluderMask]
    frame_overrides: dict[str, FrameOverride]
    transform: Similarity
    placements: dict[str, str]
    pose_segment: int
    #: The staging area this view can see at this frame, or ``None``. Spec 4.2
    #: only asks for a part on the bench to be boxed 若该视角有堆放区 ROI, so
    #: without one a bench part is not annotated and not a problem either.
    bench_roi: Optional[list] = None


def gather(
    db: Db, tax: Taxonomy, key: FrameKey, cache: Optional[InputCache] = None
) -> FrameInputs:
    """Read one frame's compiler inputs from the database.

    Spec 3.3 step 2 makes the staging area part of the geometry policy: an
    instance on the bench needs geometry *and only exists as a row* where the
    view has a bench ROI to see it in. That gate lives here rather than in
    :func:`tda.core.states.needs_geom`, which is pure and knows nothing about
    views: the same state machine serves four of them, and only some can see
    the bench.
    """
    cache = cache if cache is not None else InputCache()
    instances = instances_of(db, key.desktop, cache)
    state = state_of(db, tax, key.desktop, key.step, cache)
    seg = pose_segment_of(db, key, cache)
    bench_roi = db.bench_roi(key.desktop, key.view, seg)
    needs = _seen_here(needs_geom(instances, state, tax), state, bench_roi)
    return FrameInputs(
        key=key,
        hw=frame_hw(db, key),
        needs=needs,
        keyframes=_keyframes_of(db, key.desktop, key.view, cache),
        zorder=_zorder_of(db, key.desktop, key.view, seg, cache),
        overrides=_overrides_of(db, key.desktop, key.view, seg, cache),
        occluders=db.occluders(key),
        frame_overrides=db.frame_overrides(key),
        transform=db.transform(key),
        placements={inst: st.placement for inst, st in state.items()
                    if inst in needs or st.placement != ON_BENCH},
        pose_segment=seg,
        bench_roi=bench_roi,
    )
