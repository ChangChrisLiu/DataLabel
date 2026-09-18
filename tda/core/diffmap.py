"""Frame-difference map: where did the scene change between step k-1 and k?

Spec 4.2 overlays a difference heat map on the scanner view during reverse-order
annotation. It answers two questions:

* *where do I click* -- the hottest blob is the part that moved, and its box is
  a ready-made box prompt for SAM (the model comparison in
  ``experiments/sam_compare/REPORT.md`` found point+box reliable at IoU 0.76
  while a lone point collapses to 0.24 on large parts), and
* *what did I miss* -- :func:`explain_blobs` splits the blobs into the ones
  covered by the instances the step is supposed to change and the rest, which
  feeds the "unexplained change" queue of spec 4.2 step 4 and 4.4.

Conventions match ``tda.core.masks``: images are ``HxWx3`` RGB ``uint8``, a box
is ``(x0, y0, x1, y1)`` with exclusive ``x1``/``y1``, and every function here is
deterministic, Qt-free and side-effect free.

Robustness to exposure drift is the whole difficulty: the scanner re-meters
between passes, so two frames of an untouched scene differ by a few grey levels
everywhere. Both frames are therefore pulled onto a common per-channel median
inside the ROI *before* the CIE76 colour difference is taken, which cancels an
additive drift exactly and a small gain change approximately. The remaining
difference is rescaled by a robust maximum with an absolute floor
(:data:`MIN_DELTA_E`) so that a pair with no real change stays cold instead of
having its sensor noise stretched to full scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

__all__ = [
    "DiffBlob",
    "MIN_DELTA_E",
    "diff_heat",
    "diff_blobs",
    "explain_blobs",
    "heat_to_rgba",
]

Box = tuple[int, int, int, int]

#: Floor for the robust rescale, in CIE76 dE units (~3 is "just visible").
#: Without it an unchanged pair would divide its own noise by itself.
MIN_DELTA_E = 3.0
#: Percentile used as the robust maximum, so a few specular pixels cannot
#: flatten the rest of the map.
ROBUST_PCT = 99.5
#: Side of the structuring element that closes 1-2 px gaps inside a blob.
CLOSE_KSIZE = 5
#: Heat at which :func:`heat_to_rgba` starts mixing red towards yellow.
YELLOW_FROM = 0.5
#: Median estimation subsamples the ROI down to about this many pixels.
MEDIAN_SAMPLE = 262_144


@dataclass
class DiffBlob:
    """One connected region of change.

    Attributes:
        box: ``(x0, y0, x1, y1)`` in image coordinates, ``x1``/``y1`` exclusive.
        area: pixel count of the component (after the morphological close), which
            is smaller than the box area for anything non-rectangular.
        score: ``mean heat * sqrt(area)`` -- the square root keeps a screw-sized
            but unambiguous blob competitive with a large, lukewarm one.
    """

    box: Box
    area: int
    score: float


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_rgb_uint8(image: np.ndarray, name: str) -> np.ndarray:
    """Validate one frame and return it as an ``HxWx3`` uint8 array."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be HxWx3 RGB, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got dtype {arr.dtype}")
    return arr


def _clip_roi(roi: Optional[Box], h: int, w: int) -> Box:
    """The ROI clipped to the image, or the whole image when ``roi`` is None."""
    if roi is None:
        return (0, 0, w, h)
    x0, y0, x1, y1 = (int(round(float(v))) for v in roi)
    return (
        max(0, min(x0, w)),
        max(0, min(y0, h)),
        max(0, min(x1, w)),
        max(0, min(y1, h)),
    )


def _median_rgb(patch: np.ndarray) -> np.ndarray:
    """Per-channel median of a float ``HxWx3`` patch, on a subsampled grid."""
    pixels = patch.shape[0] * patch.shape[1]
    step = max(1, int(math.sqrt(pixels / MEDIAN_SAMPLE)))
    return np.median(patch[::step, ::step].reshape(-1, 3), axis=0)


def _iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter) / float(union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# heat map
# ---------------------------------------------------------------------------
def diff_heat(
    img_a: np.ndarray,
    img_b: np.ndarray,
    roi: Optional[Box] = None,
    blur: int = 5,
) -> np.ndarray:
    """Per-pixel change between two RGB frames of the same view.

    Args:
        img_a: frame of step k-1, ``HxWx3`` RGB uint8.
        img_b: frame of step k, same shape.
        roi: ``(x0, y0, x1, y1)`` to restrict the comparison to (the chassis, in
            practice). Outside it the returned map is exactly 0, and the
            photometric normalisation and the robust maximum are both estimated
            from inside it only -- the white table around the machine would
            otherwise dominate the median.
        blur: Gaussian kernel side in pixels; values below 3 disable the blur.
            Rounded up to the next odd number.

    Returns:
        ``float32`` ``HxW`` map in ``[0, 1]``; 1 marks the strongest change
        found. The scale is *relative to this pair*, so a heat of 0.3 means
        "clearly less changed than the biggest change here", never an absolute
        colour distance.

    Raises:
        ValueError: the two frames differ in shape, or either is not RGB uint8.
    """
    a = _as_rgb_uint8(img_a, "img_a")
    b = _as_rgb_uint8(img_b, "img_b")
    if a.shape != b.shape:
        raise ValueError(f"frames differ in shape: {a.shape} vs {b.shape}")

    h, w = a.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = _clip_roi(roi, h, w)
    if x1 <= x0 or y1 <= y0:
        return heat

    # cv2 wants float RGB in [0, 1] for Lab; staying in float avoids a uint8
    # round trip that would quantise the photometric correction away.
    af = a[y0:y1, x0:x1].astype(np.float32) / 255.0
    bf = b[y0:y1, x0:x1].astype(np.float32) / 255.0

    med_a, med_b = _median_rgb(af), _median_rgb(bf)
    mid = 0.5 * (med_a + med_b)
    af = np.clip(af + (mid - med_a).astype(np.float32), 0.0, 1.0)
    bf = np.clip(bf + (mid - med_b).astype(np.float32), 0.0, 1.0)

    delta = cv2.cvtColor(af, cv2.COLOR_RGB2Lab) - cv2.cvtColor(bf, cv2.COLOR_RGB2Lab)
    dist = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta), dtype=np.float32)

    k = int(blur)
    if k >= 3:
        k += 1 - (k % 2)  # cv2 needs an odd kernel
        dist = cv2.GaussianBlur(dist, (k, k), 0)

    scale = max(float(np.percentile(dist, ROBUST_PCT)), MIN_DELTA_E)
    heat[y0:y1, x0:x1] = np.clip(dist / scale, 0.0, 1.0)
    return heat


def diff_blobs(
    heat: np.ndarray,
    thresh: float = 0.25,
    min_area: int = 80,
    max_blobs: int = 8,
) -> list[DiffBlob]:
    """Connected regions of change, hottest and biggest first.

    Args:
        heat: map from :func:`diff_heat`.
        thresh: heat at or above which a pixel counts as changed.
        min_area: components smaller than this are noise and are dropped. The
            default is well below a scanner-resolution screw (~170 px).
        max_blobs: keep at most this many, by descending score.

    Returns:
        Up to ``max_blobs`` :class:`DiffBlob`, sorted by ``score`` descending.

    Raises:
        ValueError: ``heat`` is not a 2-D array.
    """
    arr = np.asarray(heat, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"heat must be HxW, got shape {arr.shape}")

    binary = (arr >= float(thresh)).astype(np.uint8)
    if not binary.any():
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE, CLOSE_KSIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs: list[DiffBlob] = []
    for i in range(1, count):  # 0 is the background component
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        inside = labels[y : y + bh, x : x + bw] == i
        mean_heat = float(arr[y : y + bh, x : x + bw][inside].mean())
        blobs.append(
            DiffBlob(
                box=(x, y, x + bw, y + bh),
                area=area,
                score=mean_heat * math.sqrt(area),
            )
        )

    blobs.sort(key=lambda blob: blob.score, reverse=True)
    return blobs[: max(0, int(max_blobs))]


def explain_blobs(
    blobs: list[DiffBlob],
    expected_boxes: list[Box],
    iou_min: float = 0.1,
) -> tuple[list[DiffBlob], list[DiffBlob]]:
    """Split blobs into the ones this step accounts for and the ones it does not.

    Args:
        blobs: output of :func:`diff_blobs`.
        expected_boxes: bounding boxes of the instances whose shape or state the
            step changes (spec 4.2 step 2), in the same image coordinates.
        iou_min: a blob is explained when its IoU with *any* expected box
            reaches this. IoU, not containment: a small blob far inside a large
            expected box therefore stays unexplained, which is deliberate --
            "something moved inside the PSU footprint" is exactly the kind of
            change the annotator should be asked about.

    Returns:
        ``(explained, unexplained)``, each keeping the input order.
    """
    explained: list[DiffBlob] = []
    unexplained: list[DiffBlob] = []
    boxes: Sequence[Box] = list(expected_boxes)
    for blob in blobs:
        if any(_iou(blob.box, box) >= float(iou_min) for box in boxes):
            explained.append(blob)
        else:
            unexplained.append(blob)
    return explained, unexplained


def heat_to_rgba(heat: np.ndarray, alpha_max: int = 140) -> np.ndarray:
    """Render a heat map as a transparent-to-red-to-yellow overlay.

    Args:
        heat: map from :func:`diff_heat`; values outside ``[0, 1]`` are clamped.
        alpha_max: opacity of the hottest pixel, 0-255. The default keeps the
            frame readable underneath.

    Returns:
        ``uint8`` ``HxWx4`` **RGBA** (not premultiplied): hue runs red at low
        heat to yellow at 1.0 while alpha rises linearly from 0, so both the
        colour and the transparency carry the signal.

    Raises:
        ValueError: ``heat`` is not a 2-D array.
    """
    arr = np.asarray(heat, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"heat must be HxW, got shape {arr.shape}")
    arr = np.clip(arr, 0.0, 1.0)

    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    green = np.clip((arr - YELLOW_FROM) / (1.0 - YELLOW_FROM), 0.0, 1.0)
    rgba[..., 0] = 255
    rgba[..., 1] = np.rint(green * 255.0).astype(np.uint8)
    rgba[..., 3] = np.rint(arr * float(max(0, min(255, int(alpha_max))))).astype(
        np.uint8
    )
    return rgba
