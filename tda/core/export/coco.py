"""COCO instance-segmentation export of the compiled truth table (spec 8.1).

:func:`export_coco` turns the ``compiled_mask`` rows of one view into a COCO
file: one image per frame that has a compiled row, one annotation per row that
carries geometry, and the 23 taxonomy classes as categories 1..23 in
``Taxonomy.classes`` order, so category ids are stable across exports as long as
``configs/taxonomy.yaml`` keeps its order.

Beyond the COCO fields every annotation carries an ``attributes`` block (state,
placement, visibility, occlusion ratio, amodal completeness, provenance, the
view's ``tier`` and whether it is ``verified``), because the spec's downstream
users -- and the VLM export next door -- need the truth table's semantics, not
only its pixels.

``tier`` and ``verified`` are two fields on purpose. The tier is the **view's**
(spec 8.1: scanner and OAK1 gold, OAK2 silver, RealSense bronze) and is read
from ``configs/taxonomy.yaml``; ``verified`` says whether a human confirmed this
row. One field spelling both -- "gold" for confirmed, "auto" for not -- answered
the second question in the first question's vocabulary.

Coordinates
-----------
``roi_crop=False`` (the default) writes full-frame coordinates. ``roi_crop=True``
writes them relative to the frame's ROI (``pose_segment.roi_json``): masks are
cropped, boxes shifted and clipped, the image entry records ``roi`` and its
width/height are the ROI's. A row falling entirely outside the ROI is dropped.

This module also owns the identity helpers shared with
:mod:`tda.core.export.vlm` -- image ids, cache-relative file names, frame size,
the per-desktop state context and the keyframe lookup -- so that both exports
name the same frame the same way.

The export **writes**: it brings the truth table up to date first
(:meth:`~tda.core.truth.TruthService.ensure_fresh`), so a caller needs the
single-user lock of spec 3.5. What it never writes is annotation *content* --
it only reads out what the database already implies -- and nothing here imports
Qt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from tda.core import masks
from tda.core.compiler import select_keyframe
from tda.core.db import Db
from tda.core.implied import is_implied
from tda.core.model import (
    NO_CHANGE_STEP_TYPES,
    SKIP_STEP_TYPES,
    VIEWS,
    ActionRec,
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    StateEvent,
    StepRec,
    StepType,
    Visibility,
    is_provisional,
    step_is_annotatable,
)
from tda.core.states import FrameState, state_at
from tda.core.taxonomy import Taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of, infer_hw, pose_segment_of

__all__ = [
    "ANSWERABLE",
    "NO_CHANGE_STEP_TYPES",
    "SKIP_STEP_TYPES",
    "DesktopCtx",
    "bbox_xywh",
    "cache_rel_path",
    "categories",
    "category_ids",
    "export_coco",
    "frame_file_name",
    "frame_hw",
    "frame_is_verified",
    "image_id",
    "load_ctx",
    "mask_bbox_xywh",
    "roi_of",
    "row_bbox_xywh",
    "view_index",
    "view_tier",
]

#: ``desktop * 100000 + step * 10 + view_index`` keeps ids unique and readable.
DESKTOP_STRIDE = 100_000
STEP_STRIDE = 10

#: The visibility labels whose row can be pointed at in this view.
ANSWERABLE = (
    Visibility.VISIBLE.value,
    Visibility.OCCLUDED_PARTIAL.value,
    Visibility.VISIBLE_TINY.value,
)

VERIFIED = "verified"
GEOM_BOX = "box"
DEFAULT_EXT = ".png"
EXPORT_VERSION = "1"

# What a step type means is :mod:`tda.core.model`'s table, re-exported here
# because :mod:`tda.core.export.vlm` reads it off this module. The truth table
# gates on the same table, so a step nobody may export is a step nobody was
# asked to annotate in the first place.


# --------------------------------------------------------------------------- #
# frame identity (shared with the VLM export)
# --------------------------------------------------------------------------- #
def view_index(view: str) -> int:
    """Position of ``view`` in :data:`tda.core.model.VIEWS` (scan=0 .. rs=3)."""
    try:
        return VIEWS.index(view)
    except ValueError:
        raise ValueError(f"unknown view: {view!r}") from None


def image_id(desktop: int, step: int, view: str) -> int:
    """Stable COCO image id of one frame."""
    return int(desktop) * DESKTOP_STRIDE + int(step) * STEP_STRIDE + view_index(view)


def cache_rel_path(desktop: int, step: int, view: str, ext: str = DEFAULT_EXT) -> str:
    """The frame's cached image path relative to ``cache_dir``.

    ``<view>/D<nn>/s<kkk>.<ext>`` -- the layout the image cache writes and the
    one every export references, so a COCO/JSONL file stays valid wherever the
    cache is mounted.
    """
    if not ext.startswith("."):
        ext = "." + ext
    return f"{view}/D{int(desktop):02d}/s{int(step):03d}{ext}"


def frame_file_name(frame: Optional[dict], key: FrameKey) -> str:
    """Cache-relative name of a frame, keeping the source file's extension."""
    source = (frame or {}).get("path") or ""
    ext = Path(source).suffix or DEFAULT_EXT
    return cache_rel_path(key.desktop, key.step, key.view, ext)


def frame_hw(frame: Optional[dict], compiled: dict[str, dict], view: str) -> tuple[int, int]:
    """``(H, W)`` of a frame: recorded size, else the masks', else the view's native one.

    The compiled RLEs are produced at the frame's own size, so they are the
    authority whenever the frame row does not carry an explicit ``aux["hw"]``.
    """
    recorded = ((frame or {}).get("aux") or {}).get("hw")
    if isinstance(recorded, (list, tuple)) and len(recorded) == 2:
        return (int(recorded[0]), int(recorded[1]))
    for row in compiled.values():
        size = (row.get("visible_rle") or {}).get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
    return infer_hw(view)


def bbox_xywh(box: Optional[Sequence[float]]) -> Optional[list]:
    """``(x0, y0, x1, y1)`` (exclusive) -> COCO ``[x, y, w, h]``."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return [x0, y0, x1 - x0, y1 - y0]


def mask_bbox_xywh(rle: Optional[dict]) -> Optional[list]:
    """COCO ``[x, y, w, h]`` of an RLE, or ``None`` when it is empty/absent.

    Measured off the run lengths rather than by decoding: an export writes one
    box and one area per annotation, and building two ``H x W`` arrays to read
    four numbers out of them is the single most expensive thing it used to do.
    """
    if not rle:
        return None
    box = masks.rle_bbox_xywh(rle)
    return None if box[2] <= 0 or box[3] <= 0 else [int(round(v)) for v in box]


def row_bbox_xywh(row: dict) -> Optional[list]:
    """COCO ``[x, y, w, h]`` of a compiled row, whichever geometry it carries.

    A ``mask`` row is measured off its visible RLE, a ``box`` row (a part lying
    on the bench, spec 3.4) reads its stored rectangle. ``None`` means the row
    has no geometry at all -- a missing shape, or one put out of view.
    """
    box = bbox_xywh(row.get("box")) if row.get("geom_type") == GEOM_BOX else None
    return box if box is not None else mask_bbox_xywh(row.get("visible_rle"))


# --------------------------------------------------------------------------- #
# per-desktop context (shared with the VLM export)
# --------------------------------------------------------------------------- #
@dataclass
class DesktopCtx:
    """Everything both exports need about one desktop, loaded once.

    ``events`` come from :func:`tda.core.truth_inputs.events_of`, so an export
    reads exactly the log the truth service compiled the frames from: always
    derived from the recorded actions, with the hand-written (``auto=False``)
    events merged on top.
    """

    desktop: int
    tax: Taxonomy
    instances: dict[str, InstanceRec]
    events: list[StateEvent]
    actions: list[ActionRec]
    steps: dict[int, StepRec]
    keyframes: dict[tuple[str, str, int], list[ShapeKeyframe]] = field(default_factory=dict)
    _states: dict[int, FrameState] = field(default_factory=dict, repr=False)

    def state_at(self, step: int) -> FrameState:
        """State of every instance at ``step`` (memoised)."""
        if step not in self._states:
            self._states[step] = state_at(self.instances, self.events, step, self.tax)
        return self._states[step]

    def cls_of(self, instance: str) -> Optional[str]:
        """Taxonomy class of an exportable instance key, else ``None``.

        **The exports' choke point for provisional keys.** ``None`` covers three
        things, and all three are skipped rather than allowed to crash or leak
        into a release:

        * an instance that is not in the table at all;
        * one carrying a class the taxonomy does not know;
        * a Label Studio draft (:func:`tda.core.model.is_provisional`). The
          importer gives those a *real* taxonomy class, so nothing else here
          would have stopped them -- and a draft nobody has resolved onto a real
          instance was published under ``tier: gold`` as if a human had signed
          it off.
        """
        if is_provisional(instance):
            return None
        rec = self.instances.get(instance)
        if rec is None or rec.cls not in self.tax.classes:
            return None
        return rec.cls

    def keyframe_at(self, instance: str, step: int, placement: str,
                    pose_segment: int) -> Optional[ShapeKeyframe]:
        """The keyframe that applies to one instance at one step, if any.

        Narrowed to the frame's placement chain *and* pose segment first, the
        way :func:`tda.core.compiler.compile_frame` does it: shapes of two pose
        segments are drawn in different reference frames and are not comparable.
        """
        chain = self.keyframes.get((instance, placement, pose_segment), [])
        return select_keyframe(chain, step)

    def step_type(self, step: int) -> str:
        """Recorded step type, or ``"normal"`` when the step table has no row."""
        rec = self.steps.get(step)
        return StepType.NORMAL.value if rec is None else rec.step_type

    def exportable(self, step: int) -> bool:
        """Is this step a moment worth exporting at all? (``ignore`` is not.)"""
        return step_is_annotatable(self.step_type(step))

    def actions_at(self, step: int, successful_only: bool = True) -> list[ActionRec]:
        """Actions recorded at ``step``, in ``idx`` order."""
        rows = [a for a in self.actions if a.step == step]
        if successful_only:
            rows = [a for a in rows if a.result == "success"]
        return sorted(rows, key=lambda a: a.idx)


def load_ctx(db: Db, tax: Taxonomy, desktop: int, view: str) -> DesktopCtx:
    """Read instances, steps, actions, events and keyframes of one desktop/view."""
    chains: dict[tuple[str, str, int], list[ShapeKeyframe]] = {}
    for kf in db.keyframes(desktop, view):
        chains.setdefault((kf.instance, kf.placement, kf.pose_segment), []).append(kf)
    return DesktopCtx(
        desktop=desktop, tax=tax, instances=db.instances(desktop),
        events=events_of(db, tax, desktop), actions=db.actions(desktop),
        steps={s.step: s for s in db.steps(desktop)}, keyframes=chains,
    )


# --------------------------------------------------------------------------- #
# categories
# --------------------------------------------------------------------------- #
def categories(tax: Taxonomy) -> list[dict]:
    """The 23 taxonomy classes as COCO categories, ids 1..23 in config order."""
    return [
        {"id": i, "name": name, "supercategory": defn.get("group") or "other"}
        for i, (name, defn) in enumerate(tax.classes.items(), start=1)
    ]


def category_ids(tax: Taxonomy) -> dict[str, int]:
    """``class name -> category id``."""
    return {c["name"]: c["id"] for c in categories(tax)}


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #
def roi_of(db: Db, key: FrameKey) -> Optional[tuple[int, int, int, int]]:
    """The frame's ROI as ``(x0, y0, x1, y1)``, or ``None`` for the full frame.

    Accepts the two shapes the ROI is stored in: ``[x0, y0, x1, y1]`` and a
    mapping with either ``x0/y0/x1/y1`` or ``x/y/w/h``.
    """
    raw = db.pose_segment_for(key).get("roi")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
    elif isinstance(raw, dict):
        if {"x0", "y0", "x1", "y1"} <= set(raw):
            x0, y0 = int(raw["x0"]), int(raw["y0"])
            x1, y1 = int(raw["x1"]), int(raw["y1"])
        elif {"x", "y", "w", "h"} <= set(raw):
            x0, y0 = int(raw["x"]), int(raw["y"])
            x1, y1 = x0 + int(raw["w"]), y0 + int(raw["h"])
        else:
            return None
    else:
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _crop_rle(rle: dict, roi: tuple[int, int, int, int]) -> Optional[dict]:
    """Re-encode an RLE in ROI coordinates; ``None`` when nothing is left."""
    cropped = masks.crop(masks.decode_rle(rle), roi)
    if not cropped.any():
        return None
    return masks.encode_rle(cropped)


def _clip_box(box: list, roi: tuple[int, int, int, int]) -> Optional[list]:
    """Shift a ``[x, y, w, h]`` box into ROI coordinates and clip it; ``None`` if empty."""
    x0, y0, x1, y1 = roi
    bx0, by0 = max(float(box[0]), float(x0)), max(float(box[1]), float(y0))
    bx1 = min(float(box[0]) + float(box[2]), float(x1))
    by1 = min(float(box[1]) + float(box[3]), float(y1))
    if bx1 <= bx0 or by1 <= by0:
        return None
    return [bx0 - x0, by0 - y0, bx1 - bx0, by1 - by0]


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #
def frame_is_verified(db: Db, key: FrameKey, rows: dict[str, dict]) -> bool:
    """Has a human signed this whole *frame* off? (spec 3.4)

    The frame-level notion :meth:`tda.core.truth.TruthService._frame_is_verified`
    works with, narrowed the safe way: the frame's own ``review_status``, or
    **every** compiled row of it frozen. The truth service can settle for *any*
    frozen row because it only needs to know whether a demotion is due; a
    tier is not a grade of the review, and one confirmed screw must not make the
    whole image count as confirmed.
    """
    frame = db.get_frame(key)
    if frame is not None and frame.get("review_status") == VERIFIED:
        return True
    return bool(rows) and all(row.get("status") == VERIFIED for row in rows.values())


def view_tier(view: str, tax: Taxonomy) -> str:
    """The annotation tier of one view (spec 8.1); raises for an unknown one.

    Read from ``configs/taxonomy.yaml``'s ``view_tiers`` and never written down
    in code: which camera is annotated to which standard is a decision about the
    dataset, and it has already been changed once.

    A view the configuration does not describe is a *configuration* problem, and
    a loud one: returning ``None`` instead put ``"tier": null`` on every record
    of that whole export and said nothing, which is the shape of a mistake
    nobody notices until the data is somewhere else.
    """
    try:
        return tax.view_tiers[str(view)]
    except KeyError:
        raise KeyError(
            f"no annotation tier for view {view!r}: add it to the view_tiers "
            f"table in configs/taxonomy.yaml (known views: "
            f"{', '.join(sorted(tax.view_tiers)) or 'none'})"
        ) from None


def _attributes(ctx: DesktopCtx, instance: str, row: dict, step: int,
                keyframe: Optional[ShapeKeyframe], tier: Optional[str]) -> dict:
    """The truth-table semantics carried alongside every annotation.

    ``tier`` and ``verified`` are deliberately two fields. The tier is the
    *view's* (spec 8.1: scanner and OAK1 gold, OAK2 silver, RealSense bronze)
    and says how carefully these frames are annotated at all; ``verified`` says
    whether a human confirmed this particular row. The single ``quality`` field
    that used to hold ``gold``/``auto`` answered the second question with the
    first question's vocabulary.

    ``implied`` is provenance: the instance was created by
    :mod:`tda.core.implied` because the desktop plainly has one and its log
    never touched it, not because a step named it.
    """
    inst_state = ctx.state_at(step).get(instance)
    rec = ctx.instances.get(instance)
    return {
        "instance_key": instance,
        "state": None if inst_state is None else inst_state.state,
        "placement": row.get("placement"),
        "visibility": row.get("visibility"),
        "occlusion_ratio": row.get("occlusion_ratio"),
        "amodal_complete": None if keyframe is None else bool(keyframe.amodal_complete),
        "implied": bool(rec is not None and is_implied(rec)),
        "tier": tier,
        "verified": row.get("status") == VERIFIED,
    }


def _annotation(ann_id: int, img_id: int, category: int, row: dict, attributes: dict,
                roi: Optional[tuple[int, int, int, int]], include_boxes: bool) -> Optional[dict]:
    """One COCO annotation, or ``None`` when the row carries nothing to export.

    A ``mask`` row becomes a full segmentation; a ``box`` row -- a bench part
    whose truth is a rectangle -- only becomes an annotation under
    ``include_boxes``, with ``segmentation: []``. Both geometries come from the
    compiled row itself, so nothing here re-derives what the truth service has
    already resolved.
    """
    rle = row.get("visible_rle")
    if rle:
        if roi is not None:
            rle = _crop_rle(rle, roi)
            if rle is None:
                return None
        box = mask_bbox_xywh(rle)
        if box is None:
            return None
        area = float(masks.rle_area(rle))
        segmentation: Any = rle
    else:
        if not include_boxes:
            return None
        box = bbox_xywh(row.get("box"))
        if box is None:
            return None
        if roi is not None:
            box = _clip_box(box, roi)
            if box is None:
                return None
        area = float(box[2]) * float(box[3])
        segmentation = []
    return {
        "id": ann_id, "image_id": img_id, "category_id": category,
        "segmentation": segmentation, "bbox": box, "area": _round_int(area),
        "iscrowd": 0, "attributes": attributes,
    }


def _round_int(value: float):
    """Keep whole numbers integral so mask areas read as pixel counts."""
    return int(value) if float(value).is_integer() else float(value)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_coco(
    db: Db,
    tax: Taxonomy,
    desktops: list[int],
    view: str,
    out_json: str,
    only_verified: bool = True,
    roi_crop: bool = False,
    *,
    include_boxes: bool = False,
    truth: Optional[TruthService] = None,
) -> dict:
    """Write the compiled truth of ``desktops`` in ``view`` as one COCO file.

    Parameters
    ----------
    desktops, view:
        What to export; frames without a compiled row are skipped entirely.
    out_json:
        Destination path; parent directories are created.
    only_verified:
        Keep only rows a human confirmed (``status == "verified"``). A frame
        whose rows are all dropped gets no image entry either.
    roi_crop:
        Write ROI-relative coordinates (see the module docstring).
    include_boxes:
        Also emit the compiled ``box`` rows -- a part lying on the bench, whose
        truth is a rectangle rather than a mask -- as bbox-only annotations with
        ``segmentation: []``.

    Steps typed ``ignore`` are skipped: they describe no moment of the teardown.

    Returns the COCO document that was written (``images``, ``annotations``,
    ``categories``, plus a deterministic ``info`` block: nothing in the file
    depends on the wall clock, so two exports of the same truth are identical).
    """
    doc: dict[str, Any] = {
        "info": {
            "description": "TDA rehearsal export (spec 8.1)",
            "version": EXPORT_VERSION,
            "view": view,
            "desktops": [int(d) for d in desktops],
            "only_verified": bool(only_verified),
            "roi_crop": bool(roi_crop),
            "include_boxes": bool(include_boxes),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories(tax),
    }
    cat_of = category_ids(tax)
    ann_id = 1
    service = truth or TruthService(db, tax)

    tier = view_tier(view, tax)  # the view's, so it is read once for the file
    for desktop in desktops:
        ctx = load_ctx(db, tax, desktop, view)
        # the compiled rows of an unverified frame are a cache the
        # annotator's commits leave stale (spec 3.4): fill it before
        # reading the view out, or a frame nobody visited is exported
        # as it was several edits ago -- or silently not at all
        service.ensure_fresh(desktop, view, only_verified)
        for frame in db.frames_for(desktop, view):
            key = FrameKey(desktop, frame["step"], view)
            if not ctx.exportable(key.step):
                continue
            rows = db.compiled(key)
            if only_verified:
                rows = {k: r for k, r in rows.items() if r.get("status") == VERIFIED}
            if not rows:
                continue
            roi = roi_of(db, key) if roi_crop else None
            hw = frame_hw(frame, rows, view)
            img_id = image_id(desktop, key.step, view)
            image = {
                "id": img_id,
                "file_name": frame_file_name(frame, key),
                "width": (roi[2] - roi[0]) if roi else hw[1],
                "height": (roi[3] - roi[1]) if roi else hw[0],
                # the frame-level answer to the same question the annotations
                # answer per row, so a consumer picking whole frames does not
                # have to read all of them; `review_status` is the raw column
                # underneath and stays for whoever already reads it
                "tier": tier,
                "verified": frame_is_verified(db, key, db.compiled(key)),
                "extra": {"desktop": desktop, "step": key.step, "view": view,
                          "review_status": frame.get("review_status")},
            }
            if roi is not None:
                image["roi"] = list(roi)
            doc["images"].append(image)

            segment = pose_segment_of(db, key)
            for instance in sorted(rows):
                row = rows[instance]
                cls = ctx.cls_of(instance)
                if cls is None:
                    continue  # an unresolved draft key: no category to export it under
                keyframe = ctx.keyframe_at(instance, key.step, row.get("placement") or "",
                                           segment)
                ann = _annotation(
                    ann_id, img_id, cat_of[cls], row,
                    _attributes(ctx, instance, row, key.step, keyframe, tier),
                    roi, include_boxes,
                )
                if ann is not None:
                    doc["annotations"].append(ann)
                    ann_id += 1

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc
