"""Shared protocol and metrics for the SAM 2.1 / SAM 3 / DINOv3 comparison.

Everything the annotation tool would do at click time lives here, so the two
models are measured through exactly the same pipeline:

* :func:`crop_window` -- the production-like viewport crop (bbox expanded to
  >=512 and <=1024 px, clipped to the frame, native resolution),
* :func:`box_prompt` / :func:`point_prompt` -- the three prompt flavours,
* :func:`iou` / :func:`boundary_f1` -- the two quality numbers,
* :func:`size_bucket` -- the area strata used in every table.

Reference masks are human Label Studio polygons, not ground truth; see
:func:`iou` for what that does to the numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from experiments.sam_compare import _env

#: Prompt names used as CSV/table keys throughout.
PROMPTS = ("box", "point", "point_box")
#: Model names used as CSV/table keys throughout.
MODELS = ("sam2.1", "sam3")

#: Padding (native px) added around the reference bbox to make a box prompt.
BOX_PAD_PX = 2
#: Crop window bounds (native px). See :func:`crop_window`.
MIN_WINDOW = 512
MAX_WINDOW = 1024
#: Fraction of context kept around the object inside the window.
WINDOW_CONTEXT = 1.4
#: Tolerance (px) of the boundary F-score.
BOUNDARY_TOL = 2

#: Upper edges of the area strata, in native px^2.
SIZE_EDGES = (400, 2_000, 20_000)
SIZE_NAMES = ("<400", "400-2k", "2k-20k", ">20k")


def size_bucket(area: int) -> str:
    """Name of the area stratum an object of ``area`` px^2 falls into."""
    for edge, name in zip(SIZE_EDGES, SIZE_NAMES):
        if area < edge:
            return name
    return SIZE_NAMES[-1]


# --------------------------------------------------------------------------- #
# frame selection
# --------------------------------------------------------------------------- #
def p0_steps(desktop: int) -> Optional[set[int]]:
    """Steps of one desktop whose cached scan image is the original ``P_0``.

    A handful of frames were too dark and got replaced by a different exposure;
    those are *not* the image the annotators saw, so their reference masks do
    not line up with the pixels and they are dropped. ``None`` means the desktop
    has no manifest at all (then the caller skips the desktop).
    """
    path = _env.SCAN_CACHE / f"D{desktop:02d}" / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return {int(k) for k, v in manifest.items() if (v or {}).get("reason") == "p0"}


def frame_image_path(desktop: int, step: int) -> Path:
    return _env.SCAN_CACHE / f"D{desktop:02d}" / f"s{step:03d}.png"


def load_frame_rgb(desktop: int, step: int) -> Optional[np.ndarray]:
    """Read one cached scanner frame as ``HxWx3`` RGB uint8 (None when absent)."""
    path = frame_image_path(desktop, step)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def mask_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Tight ``(x0, y0, x1, y1)`` half-open bbox of a boolean mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """The mask pixel farthest from the boundary (distance-transform argmax).

    This is the "obvious" click an annotator would make: deepest inside the
    object, so it is robust for thin screws and cable strands as well. The mask
    is padded by one pixel first, so an object touching the frame edge is not
    credited with infinite depth there.
    """
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    idx = int(np.argmax(dist))
    y, x = divmod(idx, dist.shape[1])
    return x - 1, y - 1


@dataclass(frozen=True)
class Window:
    """A viewport crop: source rect in frame coords plus the model-input scale.

    ``scale`` is 1.0 for the normal case (native resolution, the protocol under
    test). It drops below 1 only for objects whose bbox exceeds ``MAX_WINDOW``,
    where the window has to grow past 1024 px to still contain the object and
    is then downscaled for the model -- the honest production fallback, flagged
    per row so oversize objects can be reported separately.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    scale: float

    @property
    def size(self) -> tuple[int, int]:
        return self.x1 - self.x0, self.y1 - self.y0

    @property
    def oversize(self) -> bool:
        return self.scale < 1.0


def crop_window(bbox: tuple[int, int, int, int], frame_hw: tuple[int, int]) -> Window:
    """Production-like viewport around ``bbox``.

    The side is ``max(bbox side) * WINDOW_CONTEXT`` clamped to
    ``[MIN_WINDOW, MAX_WINDOW]``, centred on the bbox and shifted (not shrunk)
    to stay inside the frame. An object larger than ``MAX_WINDOW`` gets a window
    that still contains it, downscaled to ``MAX_WINDOW`` for the model.
    """
    h, w = frame_hw
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    longest = max(x1 - x0, y1 - y0)

    side = int(round(longest * WINDOW_CONTEXT))
    scale = 1.0
    if side > MAX_WINDOW:
        # keep the whole object visible, then downscale the crop for the model
        side = min(int(round(longest * 1.1)), min(h, w))
        scale = MAX_WINDOW / float(side)
    side = max(MIN_WINDOW, side)
    side = min(side, min(h, w))

    wx0 = int(round(cx - side / 2.0))
    wy0 = int(round(cy - side / 2.0))
    wx0 = max(0, min(wx0, w - side))
    wy0 = max(0, min(wy0, h - side))
    return Window(wx0, wy0, wx0 + side, wy0 + side, scale)


def crop_image(frame: np.ndarray, win: Window) -> np.ndarray:
    """Cut ``win`` out of ``frame``, downscaling only when ``win.oversize``."""
    patch = frame[win.y0 : win.y1, win.x0 : win.x1]
    if not win.oversize:
        return np.ascontiguousarray(patch)
    target = MAX_WINDOW
    return np.ascontiguousarray(
        cv2.resize(patch, (target, target), interpolation=cv2.INTER_AREA)
    )


def crop_mask(mask: np.ndarray, win: Window) -> np.ndarray:
    """Cut ``win`` out of a boolean mask, at *native* window resolution."""
    return np.ascontiguousarray(mask[win.y0 : win.y1, win.x0 : win.x1])


def paste_mask(pred: np.ndarray, win: Window) -> np.ndarray:
    """Bring a model mask back to native window resolution (identity if scale=1)."""
    wh = win.size
    if pred.shape == (wh[1], wh[0]):
        return pred
    return cv2.resize(pred.astype(np.uint8), wh, interpolation=cv2.INTER_NEAREST).astype(bool)


def to_model_coords(xy: tuple[float, float], win: Window) -> tuple[float, float]:
    """Frame coords -> coords of the (possibly downscaled) model input crop."""
    return (xy[0] - win.x0) * win.scale, (xy[1] - win.y0) * win.scale


def box_prompt(bbox: tuple[int, int, int, int], win: Window) -> np.ndarray:
    """The padded reference bbox, in model-input coordinates."""
    x0, y0, x1, y1 = bbox
    p = BOX_PAD_PX
    a = to_model_coords((x0 - p, y0 - p), win)
    b = to_model_coords((x1 + p, y1 + p), win)
    return np.array([a[0], a[1], b[0], b[1]], dtype=np.float32)


def point_prompt(point: tuple[int, int], win: Window) -> np.ndarray:
    """The single positive click, in model-input coordinates."""
    x, y = to_model_coords((point[0] + 0.5, point[1] + 0.5), win)
    return np.array([[x, y]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _roi(a: np.ndarray, b: np.ndarray, pad: int) -> tuple[slice, slice]:
    """Slices covering both masks plus ``pad``; keeps boundary work local.

    Boundary F is a dilation-heavy operation and the masks here are tiny next to
    their 512-1024 px window, so restricting it to the union bbox is what makes
    ~9,000 evaluations finish in minutes instead of hours. It is exact: outside
    the union bbox both masks are empty, so no boundary pixel can be lost.
    """
    h, w = a.shape
    union = a | b
    bb = mask_bbox(union)
    if bb is None:
        return slice(0, 0), slice(0, 0)
    x0, y0, x1, y1 = bb
    return (
        slice(max(0, y0 - pad), min(h, y1 + pad)),
        slice(max(0, x0 - pad), min(w, x1 + pad)),
    )


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over union.

    ``gt`` is a human Label Studio polygon, so a perfect segmenter does not
    score 1.0 here: polygon vertices cut corners, ellipse tools round off screw
    heads, and some shapes are visibly sloppy. Treat the numbers as a *relative*
    ranking between models and prompts, not as absolute accuracy.
    """
    inter = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    return inter / union if union else 1.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Inner boundary pixels (mask minus its 3x3 erosion)."""
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    return (m - eroded).astype(bool)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = BOUNDARY_TOL) -> float:
    """F-score of boundary pixels matched within ``tol`` px (Perazzi et al.).

    Precision = fraction of predicted boundary within ``tol`` of a reference
    boundary pixel; recall is the symmetric quantity. Much stricter than IoU for
    small parts: a 20 px screw is almost all boundary.
    """
    rows, cols = _roi(pred, gt, tol + 3)
    p, g = pred[rows, cols], gt[rows, cols]
    if p.size == 0:
        return 1.0
    pb, gb = _boundary(p), _boundary(g)
    n_p, n_g = int(pb.sum()), int(gb.sum())
    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0
    k = 2 * tol + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gb_d = cv2.dilate(gb.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                      borderValue=0).astype(bool)
    pb_d = cv2.dilate(pb.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                      borderValue=0).astype(bool)
    precision = float((pb & gb_d).sum()) / n_p
    recall = float((gb & pb_d).sum()) / n_g
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# mask (de)serialisation for the sample cache
# --------------------------------------------------------------------------- #
def pack_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Bit-pack the bbox crop of a mask (a 1600^2 bool costs 2.5 MB otherwise)."""
    x0, y0, x1, y1 = bbox
    return np.packbits(mask[y0:y1, x0:x1])


def unpack_mask(
    packed: np.ndarray, bbox: tuple[int, int, int, int], frame_hw: tuple[int, int]
) -> np.ndarray:
    """Inverse of :func:`pack_mask`, back into a full-frame boolean mask."""
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    flat = np.unpackbits(packed, count=h * w).astype(bool)
    out = np.zeros(frame_hw, dtype=bool)
    out[y0:y1, x0:x1] = flat.reshape(h, w)
    return out


def iter_reference_masks(limit_desktops: Optional[set[int]] = None) -> Iterator[dict]:
    """Yield one record per usable human scanner-view mask.

    Drops frames whose cached image is not the annotators' ``P_0`` and frames
    with no cached image at all, then records the geometry the experiment needs
    (bbox, area, click point) plus the bit-packed mask.
    """
    from tda.core.ls_import import ls_reference_masks

    ok_steps: dict[int, Optional[set[int]]] = {}
    has_image: dict[tuple[int, int], bool] = {}
    for frame, label, mask in ls_reference_masks(str(_env.LS_EXPORT), "scan"):
        if limit_desktops is not None and frame.desktop not in limit_desktops:
            continue
        if frame.desktop not in ok_steps:
            ok_steps[frame.desktop] = p0_steps(frame.desktop)
        steps = ok_steps[frame.desktop]
        if steps is None or frame.step not in steps:
            continue
        key = (frame.desktop, frame.step)
        if key not in has_image:
            has_image[key] = frame_image_path(*key).is_file()
        if not has_image[key]:
            continue
        bbox = mask_bbox(mask)
        if bbox is None:
            continue
        area = int(mask.sum())
        if area < 16:  # degenerate polygons: a click cannot be meaningful
            continue
        px, py = interior_point(mask)
        yield {
            "desktop": frame.desktop,
            "step": frame.step,
            "label": label,
            "bbox": list(bbox),
            "area": area,
            "point": [px, py],
            "frame_hw": list(mask.shape),
            "packed": pack_mask(mask, bbox),
        }
