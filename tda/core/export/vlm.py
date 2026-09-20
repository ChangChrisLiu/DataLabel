"""The VLM question/answer export -- the P0 task set of spec 8.2.

One JSON object per line::

    {"id", "task", "layer", "template_id", "desktop", "model_family",
     "chassis_type", "step", "view", ("views",) "images", "question", "answer",
     "answer_check", "evidence", "rationale", "tier", "verified",
     "graph_version", ("negative",)}

Twelve tasks -- ``V1 V2 V3 V4 V5 V6 V8 V10 V12 V14 V15 V16`` -- each generated
by one function in :mod:`tda.core.export.vlm_tasks`, each carrying an
``answer_check`` block that says how an independent checker re-derives the
answer. **A question whose answer cannot be checked by program is not emitted**,
which is the whole reason this dataset renders its text from structured labels
instead of asking a model to write it.

``tier`` is the **view's** annotation standard (spec 8.1: scanner and OAK1 gold,
OAK2 silver, RealSense bronze) and is the same for every record of one file;
``verified`` says whether a human confirmed the frame and every row this
particular answer was read off. They are two fields because they are two facts.

Perception (V1, V2, V8, V12, V14, V15) is emitted **only from verified frames**
and only about instances with a compiled row in this view; planning and history
(V3, V4, V5, V6, V10, V16) come from the step log, the state machine and the
constraint graph, so they exist for every desktop whose steps have been
imported -- all 66 of them, with no pixel annotated yet.

Where the log and the graph contradict each other -- the 18 violations of
``reports/constraints_report.md`` -- V5, V6 and V16 emit nothing about that
moment and the summary reports the steps under ``illegal_steps``. A ground
truth that argues with itself is worse than a gap.

Answers come from the truth table and the state machine, never from the image,
so every record is reproducible from the database: templates are chosen by a
checksum of the record id rather than at random, hard negatives are drawn in a
permutation seeded by desktop and step, and the file is written in a fixed
frame/task order. Two exports of one database are byte-identical.

Nothing in a record may depend on a train/val/test split -- there is none yet --
so each record carries the desktop's ``model_family`` instead, which is what a
platform-disjoint split will be cut along (spec 8.1).

The export **writes**: like the COCO one it calls
:meth:`~tda.core.truth.TruthService.ensure_fresh` per desktop, so the compiled
rows it reads are complete. A caller therefore needs the single-user lock of
spec 3.5.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from tda.core.db import Db
from tda.core.export.coco import (
    VIEWS,
    DesktopCtx,
    conflicted_steps,
    frame_file_name,
    frame_is_verified,
    load_ctx,
    view_index,
    view_tier,
)
from tda.core.export.vlm_tasks import (
    GENERATORS,
    PERCEPTION_TASKS,
    PLANNING_TASKS,
    TASKS,
    FrameData,
    TaskCtx,
    class_label,
    illegal_steps,
    instance_label,
    pointable_rows,
    slim_row,
)
from tda.core.graph import constraint_edges, edges_from_db
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy
from tda.core.truth import TruthService

__all__ = [
    "PERCEPTION_TASKS",
    "PLANNING_TASKS",
    "TASKS",
    "class_label",
    "export_vlm",
    "graph_version_of",
    "instance_label",
]


def graph_version_of(db: Db, desktop: int) -> Optional[str]:
    """Which constraint graph this export shipped, when the tool can say.

    Read through :func:`getattr` on purpose: this export has to keep working on
    a build where :func:`tda.core.graph.graph_version` is not there yet, and a
    stamp nobody can compute is exactly what ``None`` has always meant here.
    """
    try:
        from tda.core import graph as graph_mod
    except ImportError:  # pragma: no cover - the module is part of the package
        return None
    stamp = getattr(graph_mod, "graph_version", None)
    if stamp is None:
        return None
    try:
        value = stamp(db, int(desktop))
    except (TypeError, ValueError, sqlite3.Error):
        return None
    return None if value is None else str(value)


# --------------------------------------------------------------------------- #
# loading one view
# --------------------------------------------------------------------------- #
def _read_view(db: Db, ctx: DesktopCtx, desktop: int, view: str,
               disputed: set[int], only_verified: bool
               ) -> tuple[dict[int, FrameData], list[int]]:
    """Every exportable frame of one view, with its rows slimmed of their masks."""
    frames: dict[int, FrameData] = {}
    order: list[int] = []
    for row in db.frames_for(desktop, view):
        step = int(row["step"])
        if not ctx.exportable(step):
            continue  # an `ignore` step is no moment of the teardown
        key = FrameKey(desktop, step, view)
        compiled = db.compiled(key)
        # a frame somebody is still arguing about answers nothing confirmed
        verified = step not in disputed and frame_is_verified(db, key, compiled)
        slim = {inst: slim_row(r) for inst, r in compiled.items()}
        for inst, r in compiled.items():
            slim[inst]["status"] = r.get("status")
        frames[step] = FrameData(
            step=step, image=frame_file_name(row, key), verified=verified,
            rows=slim, pointable=pointable_rows(ctx, slim, only_verified),
        )
        order.append(step)
    return frames, order


def _other_views(db: Db, ctx: DesktopCtx, desktop: int, view: str,
                 only_verified: bool, service: TruthService
                 ) -> dict[str, dict[int, FrameData]]:
    """The *later* views' frames, for V15 only.

    A cross-view record belongs to exactly one file, so it is written by the
    view that comes first in :data:`~tda.core.model.VIEWS`: exporting all four
    views therefore yields each unordered pair once, not twice.

    An open conflict in one of those views does **not** refuse this export. The
    refusal belongs to the view being published -- a disagreement about a
    RealSense frame is not a reason to withhold the scanner's questions -- so
    the conflicted frames are simply marked unverified here, and V15, which only
    speaks about confirmed frames, passes over them. It reads the other views,
    so it refreshes them too: a cross-view answer off a stale cache would be a
    cross-view answer about an edit nobody made.
    """
    out: dict[str, dict[int, FrameData]] = {}
    for other in VIEWS:
        if view_index(other) <= view_index(view):
            continue
        if not db.frames_for(desktop, other):
            continue
        service.ensure_fresh(desktop, other, only_verified)
        disputed = conflicted_steps(service, desktop, other, allow_conflicts=True)
        frames, _ = _read_view(db, ctx, desktop, other, disputed, only_verified)
        if frames:
            out[other] = frames
    return out


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_vlm(
    db: Db,
    tax: Taxonomy,
    desktops: list[int],
    view: str,
    out_jsonl: str,
    tasks: Iterable[str] = TASKS,
    only_verified: bool = False,
    *,
    truth: Optional[TruthService] = None,
    allow_conflicts: bool = False,
) -> dict:
    """Write the P0 question set of ``desktops`` in ``view`` as JSONL.

    Records are grouped by desktop, then by frame in step order and, inside a
    frame, by task in :data:`~tda.core.export.vlm_tasks.TASKS` order, so two
    exports of the same database are byte-identical. ``only_verified``
    restricts every question to the compiled rows a human confirmed and keeps
    only the records that come out ``verified``.

    ``allow_conflicts`` is the COCO export's flag and means the same here: a
    view with an open disagreement is **refused** without it
    (:func:`tda.core.export.coco.conflicted_steps`), and with it the frames
    involved answer ``verified: false``, so ``only_verified`` drops them.

    Returns ``{"path", "records", "by_task", "by_source", "desktops", "view",
    "open_conflicts", "illegal_steps"}``, where ``illegal_steps`` maps
    ``desktop -> {step: [edge, ...]}`` for every step whose logged action the
    constraint graph says was impossible -- the moments V5/V6/V16 refused to
    describe.
    """
    wanted = [t for t in TASKS if t in set(tasks)]
    records: list[dict] = []
    tier = view_tier(view, tax)
    service = truth or TruthService(db, tax)
    open_conflicts = 0
    illegal: dict[int, dict[int, list[str]]] = {}
    needs_others = "V15" in wanted

    for desktop in desktops:
        ctx = load_ctx(db, tax, desktop, view)
        # the compiled rows of an unverified frame are a cache the annotator's
        # commits leave stale (spec 3.4): fill it before reading the view out,
        # or a frame nobody visited is exported as it was several edits ago
        service.ensure_fresh(desktop, view, only_verified)
        disputed = conflicted_steps(service, desktop, view, allow_conflicts)
        open_conflicts += len(disputed)

        frames, order = _read_view(db, ctx, desktop, view, disputed, only_verified)
        edges = constraint_edges(edges_from_db(db, desktop))
        bad_steps = illegal_steps(ctx.instances, edges, ctx.actions, tax)
        if bad_steps:
            illegal[int(desktop)] = {int(k): sorted(set(v))
                                     for k, v in sorted(bad_steps.items())}
        tc = TaskCtx(
            ctx=ctx, view=view, tier=tier, graph_version=graph_version_of(db, desktop),
            meta=db.get_desktop(desktop) or {}, edges=edges, frames=frames,
            steps=order, illegal=bad_steps,
            others=(_other_views(db, ctx, desktop, view, only_verified, service)
                    if needs_others else {}),
        )
        for step in order:
            for task in wanted:
                for rec in GENERATORS[task](tc, step):
                    if only_verified and not rec["verified"]:
                        continue
                    records.append(rec)

    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_task: dict[str, int] = {}
    for rec in records:
        by_task[rec["task"]] = by_task.get(rec["task"], 0) + 1
    by_source = {
        "perception": sum(n for t, n in by_task.items() if t in PERCEPTION_TASKS),
        "planning": sum(n for t, n in by_task.items() if t in PLANNING_TASKS),
    }
    return {
        "path": str(out), "records": len(records), "by_task": by_task,
        "by_source": by_source, "desktops": [int(d) for d in desktops],
        "view": view, "open_conflicts": open_conflicts, "illegal_steps": illegal,
    }
