"""Import the team's Label Studio annotations as draft shapes.

The HumanSignal export (``raw_logs/labelstudio/humansignal_annotated_projects_export.json``)
holds 671 annotated frames with ~11,500 polygons, ~560 ellipses and ~80
rectangles drawn before this tool existed. This module writes them into the
database as

* :class:`~tda.core.model.ShapeKeyframe` rows with ``source="labelstudio"`` on
  *provisional* instance keys ``ls:<label>#<n>``, which the annotator resolves
  onto real instance keys in S1;
* :class:`~tda.core.model.InstanceRec` rows carrying the class and attributes
  that ``configs/ls_label_map.yaml`` maps each old label onto;
* the form fields (step name, task description, tool, complexity, ...) appended
  to ``Step.notes`` as one ``LS: <from_name>=<value>; ...`` line;
* the annotators' ``is_pre-request_of`` / ``partner_of`` links as ``relation``
  rows with ``source="labelstudio"``, ``status="proposed"``.

Reading the export format -- image names, percent geometry, the label map --
lives in :mod:`tda.core.ls_export`; the public names from there are re-exported
here so callers only need this module.

Geometry is stored in *percent* of the uploaded image in Label Studio, so it is
rasterised straight into the view's **native** resolution
(:data:`~tda.core.ls_export.NATIVE_HW`) without knowing what was uploaded --
except when the upload was not the native frame, which is the case for the 26
``_Align_`` OAK uploads (the depth-aligned 1280x800 stream instead of the
4032x3040 still). Those are a different crop of the scene, so their percent
coordinates do not transfer and the task is skipped;
``skipped_framing_labels`` in the summary says how much annotation that costs.

:func:`ls_reference_masks` re-reads the same export without a database, for the
SAM/YOLO comparison experiments.
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from tda.core.db import Db
from tda.core.index import DesktopIndex
from tda.core.ls_export import (
    ASPECT_TOL,
    GEOM_TYPES,
    HW,
    LS_LABEL_MAP_PATH,
    NATIVE_HW,
    TARGET_HINT,
    TEXT_TYPES,
    LabelMapping,
    geometry_of,
    iter_tasks,
    labels_of,
    load_export,
    load_label_map,
    ls_ellipse_to_mask,
    ls_polygon_to_mask,
    ls_rect_to_mask,
    ls_result_to_mask,
    parse_task_image,
    upload_is_native,
)
from tda.core.masks import bbox, encode_rle
from tda.core.model import (
    FrameKey,
    InstanceRec,
    Placement,
    ShapeKeyframe,
    ShapePart,
    StepRec,
    StepType,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy

__all__ = [
    # re-exported from tda.core.ls_export, so this module is the only import
    "ASPECT_TOL", "GEOM_TYPES", "HW", "LS_LABEL_MAP_PATH", "NATIVE_HW",
    "TARGET_HINT", "TEXT_TYPES", "LabelMapping", "load_label_map",
    "ls_ellipse_to_mask", "ls_polygon_to_mask", "ls_rect_to_mask",
    "ls_result_to_mask", "parse_task_image",
    # this module
    "NOTES_PREFIX", "SOURCE", "import_ls_export", "ls_reference_masks", "main",
]

NOTES_PREFIX = "LS: "
SOURCE = "labelstudio"
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# reference masks (no database)
# --------------------------------------------------------------------------- #
def ls_reference_masks(
    path: str, view: str, include_unlabeled: bool = False
) -> Iterable[tuple[FrameKey, str, np.ndarray]]:
    """Yield ``(frame, label, mask)`` for every geometry result of one view.

    No database is involved: this feeds the SAM/YOLO comparison experiments
    (Plan B) with the old annotations as reference masks. Masks come out at the
    view's native resolution and the label is the raw Label Studio label, so
    the caller can filter (``"Target"`` marks an action region, not a part).
    Geometry the annotator left unlabelled is skipped unless
    ``include_unlabeled`` asks for it, in which case its label is ``""``.
    """
    if view not in NATIVE_HW:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(NATIVE_HW)}")
    hw = NATIVE_HW[view]
    for _project, task, frame in iter_tasks(load_export(path)):
        if frame.view != view:
            continue
        for annotation in task["annotations"]:
            for _rid, label, result in geometry_of(annotation):
                if label is None and not include_unlabeled:
                    continue
                mask = ls_result_to_mask(result, hw)
                if mask is None or not mask.any():
                    continue
                yield frame, label or "", mask


# --------------------------------------------------------------------------- #
# step notes
# --------------------------------------------------------------------------- #
def _clean_note(text: str) -> str:
    """One-line, ``;``-free form of a text value (``;`` separates note fields)."""
    return _WS.sub(" ", str(text).replace(";", ",")).strip()


def _text_value(result: dict) -> str:
    value = result.get("value") or {}
    items = value.get("choices") or value.get("text") or []
    if isinstance(items, str):
        items = [items]
    return ", ".join(_clean_note(v) for v in items if _clean_note(v))


def _notes_line(fields: dict[str, list[str]], hints: list[dict]) -> str:
    """Render the ``LS: k=v; ...`` line, with the target hints last."""
    parts = [f"{k}={' | '.join(fields[k])}" for k in sorted(fields) if fields[k]]
    if hints:
        parts.append(f"{TARGET_HINT}={json.dumps(hints, ensure_ascii=False)}")
    return NOTES_PREFIX + "; ".join(parts) if parts else ""


def _without_ls(notes: str) -> str:
    """Drop any previously imported ``LS:`` line, so a re-import replaces it."""
    kept = [ln for ln in (notes or "").splitlines() if not ln.startswith(NOTES_PREFIX)]
    return "\n".join(kept).strip()


def _write_notes(db: Db, desktop: int, lines: dict[int, str]) -> int:
    """Merge the rendered note lines into the step table of one desktop.

    ``Db`` replaces a desktop's steps wholesale, so the existing rows (and the
    actions hanging off them) are read back and handed over untouched apart
    from the notes column.
    """
    existing = {s.step: s for s in db.steps(desktop)}
    actions = db.actions(desktop)
    for step, line in lines.items():
        row = existing.get(step)
        if row is None:
            row = StepRec(desktop, step, StepType.NORMAL.value, "", notes="")
            existing[step] = row
        base = _without_ls(row.notes)
        row.notes = f"{base}\n{line}" if base else line
    db.replace_steps(desktop, [existing[k] for k in sorted(existing)], actions)
    return len(lines)


# --------------------------------------------------------------------------- #
# database helpers
# --------------------------------------------------------------------------- #
def _draft_id(result_id: Optional[str]) -> Optional[int]:
    """Stable non-negative integer for a Label Studio result id (CRC-32).

    The export identifies a shape by a short string (``"lf8oo2XZ25"``) but
    ``shape_keyframe.draft_id`` is an integer, so it is hashed -- CRC-32 rather
    than ``hash()`` because it has to be the same on the next run.
    """
    if not result_id:
        return None
    return int(zlib.crc32(str(result_id).encode("utf-8")) & 0x7FFFFFFF)


def _purge(db: Db) -> tuple[int, int]:
    """Drop what a previous run of this importer wrote, making it re-runnable.

    ``shape_part`` rows go with their keyframe (``ON DELETE CASCADE``).
    Instances, step notes and relations are written through upserts, so only
    the append-only keyframe table needs clearing.
    """
    with db.conn:
        kfs = db.conn.execute(
            "DELETE FROM shape_keyframe WHERE source=?", (SOURCE,)
        ).rowcount
        rels = db.conn.execute(
            "SELECT COUNT(*) AS n FROM relation WHERE source=?", (SOURCE,)
        ).fetchone()["n"]
    return int(kfs or 0), int(rels or 0)


def _count_rows(db: Db) -> tuple[int, int]:
    """Distinct provisional instances and relation edges this import left behind.

    Both tables are keyed per *desktop*, not per frame, so the same provisional
    key (or the same edge) recurs across the steps of one desktop and collapses
    into a single row -- which is what the annotator wants to resolve in S1.
    """
    instances = db.conn.execute(
        'SELECT COUNT(*) AS n FROM instance WHERE "key" LIKE ?', ("ls:%",)
    ).fetchone()["n"]
    relations = db.conn.execute(
        "SELECT COUNT(*) AS n FROM relation WHERE source=?", (SOURCE,)
    ).fetchone()["n"]
    return int(instances), int(relations)


def _touch_frame(db: Db, frame: FrameKey, index: dict[int, DesktopIndex]) -> bool:
    """Mark the frame as carrying LS drafts; returns whether the index knew it.

    A frame the index knows but the database does not gets its real path here,
    so the row is complete rather than a bare flag holder. An existing row that
    a human already reviewed keeps its ``review_status``.
    """
    known = index.get(frame.desktop)
    ff = known.frames.get(frame) if known is not None else None
    row = db.get_frame(frame)
    if row is None and ff is not None:
        db.upsert_frame(frame, ff.path, ff.aux, ff.ts, {"review_status": "ls_draft"})
    elif row is None or row.get("review_status") in (None, "unlabeled"):
        db.set_frame_flags(frame, review_status="ls_draft")
    return ff is not None


def _blank_counts() -> dict[str, int]:
    return {"frames": 0, "keyframes": 0, "instances": 0, "relations": 0, "target_hints": 0}


def _bucket(s: dict, desktop: int) -> dict[str, int]:
    return s["by_desktop"].setdefault(desktop, _blank_counts())


def _bump(s: dict, frame: FrameKey, field: str, n: int = 1) -> None:
    """Add to the total, the per-view bucket and the per-desktop bucket at once."""
    s[field] += n
    s["by_view"][frame.view][field] += n
    _bucket(s, frame.desktop)[field] += n


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #
def import_ls_export(
    path: str,
    db: Db,
    tax: Taxonomy,
    index: dict[int, DesktopIndex],
    label_map_path: str | Path = LS_LABEL_MAP_PATH,
    progress: bool = False,
) -> dict:
    """Import a Label Studio export as draft shapes; returns a count summary.

    Shapes land on provisional instance keys ``ls:<label>#<n>``, numbered per
    frame and per label from left to right, and are marked
    ``amodal_complete=False`` because the annotators traced visible pixels
    only. Re-running the importer replaces what an earlier run wrote.

    In the summary, ``keyframes`` / ``instances`` / ``relations`` count the
    export items consumed, while ``instance_rows`` / ``relation_edges`` count
    the desktop-scoped rows they produced -- fewer, because one provisional key
    spans every step it appears on. ``by_view`` and ``by_desktop`` break the
    same counters down, and the ``skipped_*`` / ``*_geometry`` keys account for
    everything the importer did not take.
    """
    mapping = load_label_map(label_map_path, tax)
    export = load_export(path)
    purged_kf, purged_rel = _purge(db)

    s: dict[str, Any] = {
        "export": str(path),
        "exported_at": export.get("exported_at"),
        "label_map": str(label_map_path),
        "projects": len(export.get("projects") or []),
        "tasks_total": 0, "frames": 0, "keyframes": 0, "instances": 0,
        "instance_rows": 0, "relations": 0, "relation_edges": 0,
        "target_hints": 0, "steps_with_notes": 0,
        "by_view": {v: _blank_counts() for v in NATIVE_HW},
        "by_desktop": {},
        "skipped_labels": {}, "unknown_labels": {},
        "skipped_metadata_projects": 0, "skipped_unparseable": 0,
        "skipped_framing": 0, "skipped_framing_labels": 0,
        "degenerate_geometry": 0, "unlabeled_geometry": 0,
        "relations_unresolved": 0, "frames_not_in_index": 0,
        "steps_out_of_range": 0, "frames_seen_twice": 0,
        "purged_keyframes": purged_kf, "purged_relations": purged_rel,
    }
    state = _ImportState()

    for project in export.get("projects") or []:
        if str(project.get("title") or "").strip() == "Metadata":
            s["skipped_metadata_projects"] += 1
            continue
        if progress:
            print(f"[ls] {project.get('id')} {project.get('title')} "
                  f"{project.get('workspace')}", flush=True)
        for task in project.get("tasks") or []:
            annotations = task.get("annotations") or []
            if not annotations:
                continue
            s["tasks_total"] += 1
            image = (task.get("data") or {}).get("image") or ""
            parsed = parse_task_image(image)
            if parsed is None:
                s["skipped_unparseable"] += 1
                continue
            frame = FrameKey(*parsed)
            if not upload_is_native(image, frame.view, annotations):
                s["skipped_framing"] += 1
                s["skipped_framing_labels"] += sum(
                    len(labels_of(r)) for a in annotations for r in a.get("result") or []
                )
                continue
            _import_task(db, index, mapping, frame, annotations, state, s)

    for desktop in sorted(set(state.notes) | set(state.hints)):
        lines = {}
        for step in sorted(set(state.notes.get(desktop, {})) | set(state.hints.get(desktop, {}))):
            line = _notes_line(state.notes.get(desktop, {}).get(step, {}),
                               state.hints.get(desktop, {}).get(step, []))
            if line:
                lines[step] = line
        if lines:
            s["steps_with_notes"] += _write_notes(db, desktop, lines)
    s["frames"] = len(state.seen)
    s["instance_rows"], s["relation_edges"] = _count_rows(db)
    return s


class _ImportState:
    """Cross-task accumulators of one :func:`import_ls_export` call."""

    def __init__(self) -> None:
        # desktop -> step -> from_name -> distinct values, in first-seen order
        self.notes: dict[int, dict[int, dict[str, list[str]]]] = {}
        # desktop -> step -> action-target bboxes
        self.hints: dict[int, dict[int, list[dict]]] = {}
        # per-frame label counters, so a second task on one frame keeps numbering
        self.counters: dict[FrameKey, dict[str, int]] = {}
        self.seen: set[FrameKey] = set()


def _import_task(
    db: Db, index: dict[int, DesktopIndex],
    mapping: dict[str, Optional[LabelMapping]], frame: FrameKey,
    annotations: list[dict], state: _ImportState, s: dict,
) -> None:
    """Import one task's annotations onto one frame (see :func:`import_ls_export`)."""
    if frame in state.seen:
        s["frames_seen_twice"] += 1
    else:
        state.seen.add(frame)
        _bump(s, frame, "frames")
        if not _touch_frame(db, frame, index):
            s["frames_not_in_index"] += 1
        known = index.get(frame.desktop)
        if known is not None and frame.step > known.n_steps:
            s["steps_out_of_range"] += 1

    counter = state.counters.setdefault(frame, {})
    for annotation in annotations:
        keys: dict[str, str] = {}  # result id -> provisional instance key
        for rid, label, result in geometry_of(annotation):
            key = _import_shape(db, mapping, frame, rid, label, result, counter, state, s)
            if key is not None:
                keys[rid] = key
        for result in annotation.get("result") or []:
            kind = result.get("type")
            if kind in TEXT_TYPES:
                _collect_note(state, frame, result)
            elif kind == "relation":
                _add_relation(db, frame, result, keys, s)


def _import_shape(
    db: Db, mapping: dict[str, Optional[LabelMapping]], frame: FrameKey,
    rid: str, label: Optional[str], result: dict,
    counter: dict[str, int], state: _ImportState, s: dict,
) -> Optional[str]:
    """Write one geometry result; returns its provisional key, if it made one."""
    if label is None:
        s["unlabeled_geometry"] += 1
        return None
    if label not in mapping:
        s["unknown_labels"][label] = s["unknown_labels"].get(label, 0) + 1
        return None
    entry = mapping[label]
    if entry is None:
        s["skipped_labels"][label] = s["skipped_labels"].get(label, 0) + 1
        return None
    mask = ls_result_to_mask(result, NATIVE_HW[frame.view])
    if mask is None or not mask.any():
        s["degenerate_geometry"] += 1
        return None

    if entry.special == TARGET_HINT:
        state.hints.setdefault(frame.desktop, {}).setdefault(frame.step, []).append(
            {"view": frame.view, "label": label,
             "bbox": [int(v) for v in bbox(mask)]}
        )
        _bump(s, frame, "target_hints")
        return None

    counter[label] = counter.get(label, 0) + 1
    key = f"ls:{label}#{counter[label]}"
    attrs = dict(entry.attrs)
    if entry.state is not None:
        attrs["state"] = entry.state
    db.upsert_instance(InstanceRec(key=key, desktop=frame.desktop, cls=entry.cls,
                                   attrs=attrs, raw_names=[label]))
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=key, desktop=frame.desktop, view=frame.view, pose_segment=0,
        anchor_step=frame.step, placement=Placement.IN_CHASSIS.value, geom_type="mask",
        parts=[ShapePart("main", encode_rle(mask))], amodal_complete=False,
        source=SOURCE, draft_id=_draft_id(rid),
    ))
    _bump(s, frame, "keyframes")
    _bump(s, frame, "instances")
    return key


def _collect_note(state: _ImportState, frame: FrameKey, result: dict) -> None:
    """Remember one ``choices``/``textarea`` value for the frame's step."""
    value = _text_value(result)
    name = result.get("from_name")
    if not value or not name:
        return
    values = state.notes.setdefault(frame.desktop, {}).setdefault(
        frame.step, {}).setdefault(str(name), [])
    if value not in values:
        values.append(value)


def _add_relation(db: Db, frame: FrameKey, result: dict, keys: dict[str, str],
                  s: dict) -> None:
    """Store one Label Studio link as a proposed constraint edge.

    ``from_id`` is the node the annotators drew the arrow from, i.e. the one
    that has to change state first (the blocker); ``to_id`` is the target.
    """
    blocker = keys.get(result.get("from_id"))
    target = keys.get(result.get("to_id"))
    if not blocker or not target:
        s["relations_unresolved"] += 1
        return
    labels = result.get("labels") or []
    db.add_relation(
        frame.desktop, rel_type=str(labels[0]) if labels else "related_to",
        target=target, blocker=blocker, source=SOURCE, status="proposed",
        evidence_step=frame.step, reason=f"labelstudio:{frame.view}",
    )
    _bump(s, frame, "relations")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """CLI: import an export into a database and print the count summary."""
    import argparse

    ap = argparse.ArgumentParser(description="Import Label Studio annotations as drafts.")
    ap.add_argument("export", help="humansignal_annotated_projects_export.json")
    ap.add_argument("--db", required=True, help="target tda.sqlite")
    ap.add_argument("--index", default=None, help="cache/index.json (optional)")
    ap.add_argument("--summary", default=None, help="write the summary JSON here")
    args = ap.parse_args(argv)

    index: dict[int, DesktopIndex] = {}
    if args.index:
        from tda.core.index import load_index

        index = load_index(args.index)
    db = Db(args.db)
    try:
        summary = import_ls_export(args.export, db, load_taxonomy(), index, progress=True)
    finally:
        db.close()
    text = json.dumps(summary, indent=1, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.summary:
        Path(args.summary).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
