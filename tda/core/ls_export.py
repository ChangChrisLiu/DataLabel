"""Reading the Label Studio export format: task names, geometry, labels.

This is the database-free half of the Label Studio bridge; the writing half is
:mod:`tda.core.ls_import`. It knows three things:

* how a task's uploaded image name encodes ``(desktop, step, view)``
  -- :func:`parse_task_image`;
* how to turn Label Studio's *percent* geometry -- polygons, ellipses and
  rotated rectangles -- into masks at a view's native resolution
  -- :func:`ls_result_to_mask` and the per-shape functions it dispatches to;
* how ``configs/ls_label_map.yaml`` maps the annotators' 40 old labels onto the
  23-class taxonomy -- :func:`load_label_map`.

plus the traversal helpers that pair every geometry result with its label and
drop the tasks whose upload was not the native frame.
"""
from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np
import yaml

from tda.core.masks import polygons_to_mask
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]
LS_LABEL_MAP_PATH = REPO_ROOT / "configs" / "ls_label_map.yaml"

#: Native ``(height, width)`` of each view -- what the drafts are rasterised at.
NATIVE_HW: dict[str, tuple[int, int]] = {
    "scan": (1600, 1600),
    "oak1": (3040, 4032),
    "oak2": (3040, 4032),
    "rs": (720, 1280),
}

#: Relative tolerance on aspect ratio before an upload counts as a different crop.
ASPECT_TOL = 0.02

GEOM_TYPES = ("polygon", "polygonlabels", "ellipse", "rectangle")
TEXT_TYPES = ("choices", "textarea")
TARGET_HINT = "action_target_hint"

HW = tuple[int, int]

# ``<8 hex chars>-`` is the prefix Label Studio puts on every uploaded file.
_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")
# ``Front_OAK_Align_*`` / ``Side_OAK_Align_*`` are the depth-aligned 1280x800
# OAK stream, i.e. a different crop of the scene than the 4032x3040 stills.
_ALIGN = re.compile(r"_Align_", re.IGNORECASE)
_EXT = r"(?:png|jpg|jpeg)"
_OAK = re.compile(rf"^(Front|Side)_OAK(?:_Align)?_(\d+)_(\d+)\.{_EXT}$", re.IGNORECASE)
_RS = re.compile(rf"^Rs_(\d+)_(\d+)\.{_EXT}$", re.IGNORECASE)
_CTRL = re.compile(rf"^ctrl_desktop_data_(\d+)__(\d+)\.{_EXT}$", re.IGNORECASE)
_SCAN = re.compile(rf"^(\d+)_(\d+)\.{_EXT}$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# task image names
# --------------------------------------------------------------------------- #
def parse_task_image(name: str) -> Optional[tuple[int, int, str]]:
    """Map a task's image name onto ``(desktop, step, view)``.

    Accepts a bare basename, an ``<8 hex>-`` prefixed upload name or a full
    Label Studio storage URL. ``None`` means the name carries no usable
    ``(desktop, step)`` -- the four ``P_<n>.png`` tasks are the only real case.

    >>> parse_task_image("3e99d760-Rs_13_42.png")
    (13, 42, 'rs')
    """
    base = str(name or "").rsplit("/", 1)[-1]
    base = _UPLOAD_PREFIX.sub("", base, count=1)
    m = _OAK.match(base)
    if m:
        view = "oak1" if m.group(1).lower() == "front" else "oak2"
        return int(m.group(2)), int(m.group(3)), view
    for pattern, view in ((_RS, "rs"), (_CTRL, "scan"), (_SCAN, "scan")):
        m = pattern.match(base)
        if m:
            return int(m.group(1)), int(m.group(2)), view
    return None


# --------------------------------------------------------------------------- #
# percent -> pixel rasterisation
# --------------------------------------------------------------------------- #
def ls_polygon_to_mask(points_pct: list[list[float]], hw: HW) -> np.ndarray:
    """Rasterise a Label Studio polygon (``[[x%, y%], ...]``) into a bool mask.

    Fewer than three points cannot enclose an area and give an empty mask.
    """
    h, w = int(hw[0]), int(hw[1])
    pts = list(points_pct or [])
    if len(pts) < 3:
        return np.zeros((h, w), bool)
    flat: list[float] = []
    for point in pts:
        flat.append(float(point[0]) / 100.0 * w)
        flat.append(float(point[1]) / 100.0 * h)
    return polygons_to_mask([flat], (h, w))


def ls_ellipse_to_mask(value: dict, hw: HW) -> np.ndarray:
    """Rasterise a Label Studio ellipse.

    Centre ``x``/``y`` and ``radiusX``/``radiusY`` are percents of width and
    height respectively, ``rotation`` is in degrees.
    """
    h, w = int(hw[0]), int(hw[1])
    cx = int(round(float(value["x"]) / 100.0 * w))
    cy = int(round(float(value["y"]) / 100.0 * h))
    rx = max(1, int(round(float(value["radiusX"]) / 100.0 * w)))
    ry = max(1, int(round(float(value["radiusY"]) / 100.0 * h)))
    canvas = np.zeros((h, w), np.uint8)
    angle = float(value.get("rotation") or 0.0)
    cv2.ellipse(canvas, (cx, cy), (rx, ry), angle, 0.0, 360.0, 1, -1)
    return canvas.astype(bool)


def _rect_corners(value: dict, hw: HW) -> list[float]:
    """Pixel corners of a Label Studio rectangle, rotated about its anchor.

    ``x``/``y`` are the top-left corner and ``rotation`` turns the box about
    that corner (degrees, clockwise on screen). Rotating in *pixel* space is
    what Label Studio draws; rotating the percents would shear a non-square
    frame.
    """
    h, w = int(hw[0]), int(hw[1])
    x0 = float(value["x"]) / 100.0 * w
    y0 = float(value["y"]) / 100.0 * h
    bw = float(value["width"]) / 100.0 * w
    bh = float(value["height"]) / 100.0 * h
    rad = math.radians(float(value.get("rotation") or 0.0))
    cos, sin = math.cos(rad), math.sin(rad)
    flat: list[float] = []
    for dx, dy in ((0.0, 0.0), (bw, 0.0), (bw, bh), (0.0, bh)):
        flat.append(x0 + dx * cos - dy * sin)
        flat.append(y0 + dx * sin + dy * cos)
    return flat


def ls_rect_to_mask(value: dict, hw: HW) -> np.ndarray:
    """Rasterise a Label Studio rectangle (percent coordinates, degrees)."""
    return polygons_to_mask([_rect_corners(value, hw)], (int(hw[0]), int(hw[1])))


def ls_result_to_mask(result: dict, hw: HW) -> Optional[np.ndarray]:
    """Rasterise any geometry result; ``None`` for a non-geometry result."""
    kind = result.get("type")
    value = result.get("value") or {}
    if kind in ("polygon", "polygonlabels"):
        return ls_polygon_to_mask(value.get("points") or [], hw)
    if kind == "ellipse":
        return ls_ellipse_to_mask(value, hw)
    if kind == "rectangle":
        return ls_rect_to_mask(value, hw)
    return None


def result_pixel_bbox(result: dict, hw: HW) -> Optional[tuple[int, int, int, int]]:
    """Exclusive pixel bbox ``(x0, y0, x1, y1)`` of a geometry result.

    Computed from the percent coordinates, so nothing is allocated -- which
    matters on the 12 MP OAK frames. It equals
    ``masks.bbox(ls_result_to_mask(result, hw))`` for any shape that lies
    inside the frame: both round every coordinate to its own pixel.

    Two documented differences for shapes that do not:

    * a shape crossing a frame edge at an angle keeps the full extent of its
      *in-frame* columns and rows here, while the rasterised mask loses the
      corner that fell outside -- so this is a superset, never smaller;
    * a rotated ellipse gets its analytic extent, within a pixel of the
      rasterised one (every ellipse in the export has ``rotation == 0``).

    ``None`` means the result encloses no area (or is not geometry at all).
    """
    h, w = int(hw[0]), int(hw[1])
    kind = result.get("type")
    value = result.get("value") or {}
    if kind in ("polygon", "polygonlabels"):
        pts = value.get("points") or []
        if len(pts) < 3:
            return None
        xs = [int(round(float(p[0]) / 100.0 * w)) for p in pts]
        ys = [int(round(float(p[1]) / 100.0 * h)) for p in pts]
    elif kind == "ellipse":
        cx = int(round(float(value["x"]) / 100.0 * w))
        cy = int(round(float(value["y"]) / 100.0 * h))
        rx = max(1, int(round(float(value["radiusX"]) / 100.0 * w)))
        ry = max(1, int(round(float(value["radiusY"]) / 100.0 * h)))
        rad = math.radians(float(value.get("rotation") or 0.0))
        ex = int(round(math.hypot(rx * math.cos(rad), ry * math.sin(rad))))
        ey = int(round(math.hypot(rx * math.sin(rad), ry * math.cos(rad))))
        xs, ys = [cx - ex, cx + ex], [cy - ey, cy + ey]
    elif kind == "rectangle":
        flat = _rect_corners(value, (h, w))
        xs = [int(round(v)) for v in flat[0::2]]
        ys = [int(round(v)) for v in flat[1::2]]
    else:
        return None
    x0, x1 = max(0, min(xs)), min(w, max(xs) + 1)
    y0, y1 = max(0, min(ys)), min(h, max(ys) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _pct_anchor(result: dict) -> tuple[float, float]:
    """Top-left of the result's percent-space bbox, for left-to-right ordering.

    Percent space keeps this cheap: ordering by ``min x%`` is the same order as
    ordering by the rasterised mask's ``bbox`` x, without building the masks.
    """
    value = result.get("value") or {}
    kind = result.get("type")
    if kind in ("polygon", "polygonlabels"):
        pts = value.get("points") or []
        if not pts:
            return (0.0, 0.0)
        return (min(float(p[0]) for p in pts), min(float(p[1]) for p in pts))
    if kind == "ellipse":
        return (float(value["x"]) - float(value["radiusX"]),
                float(value["y"]) - float(value["radiusY"]))
    if kind == "rectangle":
        flat = _rect_corners(value, (100, 100))
        return (min(flat[0::2]), min(flat[1::2]))
    return (0.0, 0.0)


# --------------------------------------------------------------------------- #
# configs/ls_label_map.yaml
# --------------------------------------------------------------------------- #
@dataclass
class LabelMapping:
    """One row of ``configs/ls_label_map.yaml``."""

    label: str
    cls: Optional[str] = None
    attrs: dict = field(default_factory=dict)
    state: Optional[str] = None  # the state the label text itself asserts
    special: Optional[str] = None  # only ``action_target_hint``
    note: str = ""


def load_label_map(
    path: str | Path = LS_LABEL_MAP_PATH, tax: Optional[Taxonomy] = None
) -> dict[str, Optional[LabelMapping]]:
    """Load and validate the label map; a ``None`` value means "drop this label".

    Every entry is checked against the taxonomy, so a typo in the config fails
    here rather than silently writing an unknown class into the database.
    """
    tax = tax or load_taxonomy()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    out: dict[str, Optional[LabelMapping]] = {}
    for label, raw in (cfg.get("labels") or {}).items():
        if raw is None:
            out[label] = None
            continue
        entry = LabelMapping(
            label=label,
            cls=raw.get("cls"),
            attrs=dict(raw.get("attrs") or {}),
            state=raw.get("state"),
            special=raw.get("special"),
            note=raw.get("note") or "",
        )
        _validate_entry(entry, tax)
        out[label] = entry
    return out


def _validate_entry(entry: LabelMapping, tax: Taxonomy) -> None:
    """Raise ``ValueError`` when an entry does not fit the taxonomy."""
    where = f"ls_label_map[{entry.label!r}]"
    if entry.special is not None:
        if entry.special != TARGET_HINT:
            raise ValueError(f"{where}: unknown special {entry.special!r}")
        if entry.cls is not None:
            raise ValueError(f"{where}: `special` and `cls` are exclusive")
        return
    if entry.cls not in tax.classes:
        raise ValueError(f"{where}: unknown taxonomy class {entry.cls!r}")
    declared = tax.classes[entry.cls].get("attrs") or {}
    for key, value in entry.attrs.items():
        if key not in declared:
            raise ValueError(f"{where}: class {entry.cls} has no attribute {key!r}")
        allowed = declared[key]
        if allowed and value not in allowed:
            raise ValueError(f"{where}: {key}={value!r} is not one of {list(allowed)}")
    if entry.state is not None and entry.state not in tax.states_of(entry.cls):
        raise ValueError(f"{where}: state {entry.state!r} is not a state of {entry.cls}")


# --------------------------------------------------------------------------- #
# export traversal
# --------------------------------------------------------------------------- #
def load_export(path: str | Path) -> dict:
    """Read a Label Studio project export (16 MB for the real one)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _framing_matches(view: str, result: dict) -> Optional[bool]:
    """Does the uploaded image have the native aspect ratio? ``None``: unknown."""
    ow, oh = result.get("original_width"), result.get("original_height")
    if not ow or not oh:
        return None
    h, w = NATIVE_HW[view]
    native = w / h
    return abs(ow / oh - native) <= ASPECT_TOL * native


def upload_is_native(name: str, view: str, annotations: Iterable[dict]) -> bool:
    """Was this task's upload the native frame, so percent coordinates transfer?

    Two independent checks, because neither covers everything: the ``_Align_``
    name marks the depth-aligned OAK stream even when the task carries no
    geometry to measure, and the aspect-ratio test catches any other re-crop.
    """
    if _ALIGN.search(str(name or "").rsplit("/", 1)[-1]):
        return False
    for ann in annotations:
        for result in ann.get("result") or []:
            if result.get("type") in GEOM_TYPES:
                verdict = _framing_matches(view, result)
                if verdict is not None:
                    return verdict
    return True  # nothing to measure: assume the plain upload


def labels_of(result: dict) -> list[str]:
    """The label list of a ``labels`` / ``polygonlabels`` result (may be empty)."""
    value = result.get("value") or {}
    for key in ("labels", "polygonlabels"):
        got = value.get(key)
        if got:
            return [str(v) for v in got]
    return []


def geometry_of(annotation: dict) -> list[tuple[str, Optional[str], dict]]:
    """``(result_id, label, geometry result)`` of one annotation, left to right.

    Label Studio stores a geometry result and its ``labels`` result under the
    same result ``id``; ``polygonlabels`` carries its label inline. A label of
    ``None`` means the annotator drew a shape and never named it.
    """
    by_id: dict[Optional[str], list[dict]] = OrderedDict()
    for result in annotation.get("result") or []:
        by_id.setdefault(result.get("id"), []).append(result)
    items: list[tuple[str, Optional[str], dict]] = []
    for rid, group in by_id.items():
        if rid is None:
            continue
        named = [lab for r in group if r.get("type") == "labels" for lab in labels_of(r)]
        for result in group:
            if result.get("type") not in GEOM_TYPES:
                continue
            labels = labels_of(result) or named
            items.append((rid, labels[0] if labels else None, result))
    items.sort(key=lambda it: (_pct_anchor(it[2]), it[0]))
    return items


def iter_tasks(export: dict) -> Iterator[tuple[dict, dict, FrameKey]]:
    """Yield ``(project, task, frame)`` for every usable annotated task.

    Skips the ``Metadata`` projects (no images), the tasks whose image name
    carries no ``(desktop, step)`` and the non-native uploads.
    """
    for project in export.get("projects") or []:
        if str(project.get("title") or "").strip() == "Metadata":
            continue
        for task in project.get("tasks") or []:
            if not (task.get("annotations") or []):
                continue
            image = (task.get("data") or {}).get("image") or ""
            parsed = parse_task_image(image)
            if parsed is None:
                continue
            desktop, step, view = parsed
            if not upload_is_native(image, view, task["annotations"]):
                continue
            yield project, task, FrameKey(desktop, step, view)
