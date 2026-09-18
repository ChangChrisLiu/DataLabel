"""Frame-difference map: where did the scene change between step k-1 and k?

Spec 4.2 overlays a difference map on the scanner view during reverse-order
annotation. It answers two questions:

* *where do I click* -- the strongest blob is the part that moved, and its box is
  a ready-made box prompt for SAM (the model comparison in
  ``experiments/sam_compare/REPORT.md`` found point+box reliable at IoU 0.76
  while a lone point collapses to 0.24 on large parts), and
* *what did I miss* -- :func:`explain_blobs` splits the blobs into the ones
  covered by the instances the step is supposed to change and the rest, which
  feeds the "unexplained change" queue of spec 4.2 step 4 and 4.4.

Conventions match ``tda.core.masks``: images are ``HxWx3`` RGB ``uint8``, a box
is ``(x0, y0, x1, y1)`` with exclusive ``x1``/``y1``, and every function here is
deterministic, Qt-free and side-effect free.

Two maps, two jobs
------------------
:func:`diff_delta_e` returns the **absolute** CIE76 colour difference after
photometric equalisation, in dE units. It is what :func:`diff_blobs` thresholds,
so a blob means the same thing whatever ROI was passed. :func:`diff_heat`
rescales that map into ``[0, 1]`` for **display** only: its 1.0 is "the biggest
change in this ROI", which is the right thing for an overlay and the wrong thing
for a decision.

Not every difference is a change
--------------------------------
Two frames of an untouched scene are never identical, and both failure modes
manufacture blobs that waste the annotator's time:

* the scanner re-meters between passes, so the whole frame shifts by a few grey
  levels -- handled by pulling both frames onto a common per-channel median
  inside the ROI before the difference, which cancels an additive drift exactly;
* it also re-registers by a pixel or two, which turns every edge in the frame
  into a bright ridge -- handled by taking, per pixel, the *minimum* dE over the
  integer shifts within ``shift_px``. Blurring cannot do this: it widens the
  ridge instead of cancelling it. On a real D13 pair a pure 1 px shift drops from
  a 99.5th percentile of 27.9 dE to 0.0 while the real cooler removal only goes
  from 44.9 to 39.9.

The cost of the shift tolerance is that a genuine change thinner than
``shift_px`` is erased along with the misregistration. At scanner resolution the
smallest thing that matters is a screw of ~15 px, so 1 px is free.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

#: Both guards below silently change what the caller gets back, so both say so.
#: A run that keeps hitting them is a registration problem, not a diff problem,
#: and without a line in the log there is nothing to notice it by.
log = logging.getLogger(__name__)

__all__ = [
    "DiffBlob",
    "BLOB_DELTA_E",
    "MIN_SCALE_DELTA_E",
    "diff_delta_e",
    "diff_heat",
    "diff_blobs",
    "explain_blobs",
    "heat_to_rgba",
]

Box = tuple[int, int, int, int]

#: Default absolute threshold for :func:`diff_blobs`, in CIE76 dE units.
#: Calibrated on D13 with ``experiments/diffmap_calibrate.py``, inside the
#: suggested chassis ROI. It sits in a wide, flat valley: a removed cooler peaks
#: at 72 dE (99.5th percentile 41), a removed screw at 30, while a frame against
#: itself reaches 0.0 and a 1 px misregistration 1.1. The value is the *top* of
#: the usable range rather than its middle, because the failure below is worse:
#: at 8 dE the cooler blob bleeds across the whole board (24k px instead of 18k)
#: and stops being a usable box prompt, while at 16 dE a screw step vanishes
#: entirely. 12 is the largest threshold that still sees a screw and the
#: smallest that keeps a part tight.
BLOB_DELTA_E = 12.0
#: Floor for :func:`diff_heat`'s robust rescale (~3 dE is "just visible").
#: Without it an unchanged pair would divide its own noise by itself.
MIN_SCALE_DELTA_E = 3.0
#: Percentile used as :func:`diff_heat`'s robust maximum, so a few specular
#: pixels cannot flatten the rest of the map.
ROBUST_PCT = 99.5
#: Side of the structuring element that closes 1-2 px gaps inside a blob.
CLOSE_KSIZE = 5
#: Default cap on how many raw components :func:`diff_blobs` merges. Merging is
#: quadratic in the component count and a misregistered pair can produce
#: thousands of specks, none of which is the answer anyway.
MAX_COMPONENTS = 400
#: Safety cap on :func:`_merge_parts`' fixed-point iteration. Each round merges
#: every currently adjacent pair, so real inputs converge in a few.
MAX_MERGE_ROUNDS = 64
#: Heat at which :func:`heat_to_rgba` starts mixing red towards yellow.
YELLOW_FROM = 0.5
#: Median estimation subsamples the ROI down to about this many pixels.
MEDIAN_SAMPLE = 262_144


@dataclass
class DiffBlob:
    """One connected region of change.

    Attributes:
        box: ``(x0, y0, x1, y1)`` in image coordinates, ``x1``/``y1`` exclusive.
        area: pixel count of the region, which is smaller than the box area for
            anything non-rectangular.
        score: ``mean dE over the region * sqrt(area)`` -- the square root keeps
            a screw-sized but unambiguous blob competitive with a large,
            lukewarm one. Absolute, so it is comparable between frame pairs.
        mask: box-local boolean mask (``shape == (y1 - y0, x1 - x0)``, origin at
            ``box[:2]``) so the UI can outline the actual region instead of the
            box. Excluded from equality, which therefore compares geometry only.
    """

    box: Box
    area: int
    score: float
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)


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


def _norm(delta: np.ndarray) -> np.ndarray:
    """Euclidean length along the last axis (CIE76 dE)."""
    return np.sqrt(np.einsum("ijk,ijk->ij", delta, delta), dtype=np.float32)


def _iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a: Box, b: Box) -> int:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def _union_box(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _gap(a: Box, b: Box) -> int:
    """Largest axis gap between two boxes; 0 when they overlap or touch."""
    dx = max(0, a[0] - b[2], b[0] - a[2])
    dy = max(0, a[1] - b[3], b[1] - a[3])
    return max(dx, dy)


# ---------------------------------------------------------------------------
# difference maps
# ---------------------------------------------------------------------------
def diff_delta_e(
    img_a: np.ndarray,
    img_b: np.ndarray,
    roi: Optional[Box] = None,
    blur: int = 5,
    shift_px: int = 1,
    max_side: Optional[int] = None,
) -> np.ndarray:
    """Absolute per-pixel colour change between two RGB frames of the same view.

    Args:
        img_a: frame of step k-1, ``HxWx3`` RGB uint8.
        img_b: frame of step k, same shape.
        roi: ``(x0, y0, x1, y1)`` to restrict the comparison to (the chassis, in
            practice -- ``tda.core.cache.suggest_roi`` produces it). Outside it
            the map is exactly 0, and the photometric equalisation is estimated
            from inside it only: the white table around the machine would
            otherwise dominate the median. **Pass it.** Without it the scan bed's
            own edges show up as changes.
        blur: Gaussian kernel side in pixels applied to the dE field; values
            below 3 disable it. Rounded up to the next odd number.
        shift_px: integer misregistration tolerated, in pixels. ``0`` disables
            the tolerance; ``1`` is the default and enough for the scanner; ``2``
            costs 25 shifted comparisons instead of 9.
        max_side: when set and exceeded, the pair is compared at a reduced size
            and the map is resized back, which is how a 4032x3040 frame stays
            interactive. Boxes therefore stay in full-resolution coordinates, at
            the price of blob outlines quantised to the reduced grid.

    Returns:
        ``float32`` ``HxW`` map of CIE76 dE. Roughly: under 3 is invisible, 10-15
        is a clear material change, 40+ is a part that left the frame.

    Raises:
        ValueError: the two frames differ in shape, or either is not RGB uint8.
    """
    a = _as_rgb_uint8(img_a, "img_a")
    b = _as_rgb_uint8(img_b, "img_b")
    if a.shape != b.shape:
        raise ValueError(f"frames differ in shape: {a.shape} vs {b.shape}")

    h, w = a.shape[:2]
    scale = 1.0
    if max_side is not None and max(h, w) > int(max_side):
        scale = int(max_side) / float(max(h, w))
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)

    small_roi = (
        None if roi is None else tuple(int(round(float(v) * scale)) for v in roi)
    )
    delta = _delta_e_map(a, b, small_roi, blur, shift_px)  # type: ignore[arg-type]

    if delta.shape != (h, w):
        delta = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)
        if roi is not None:
            # The resize smears the ROI border; restore the hard edge.
            keep = np.zeros((h, w), dtype=bool)
            x0, y0, x1, y1 = _clip_roi(roi, h, w)
            keep[y0:y1, x0:x1] = True
            delta[~keep] = 0.0
    return np.ascontiguousarray(delta, dtype=np.float32)


def _delta_e_map(
    a: np.ndarray, b: np.ndarray, roi: Optional[Box], blur: int, shift_px: int
) -> np.ndarray:
    """dE inside the ROI after equalisation and shift matching; 0 outside."""
    h, w = a.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = _clip_roi(roi, h, w)
    if x1 <= x0 or y1 <= y0:
        return out

    # cv2 wants float RGB in [0, 1] for Lab; staying in float avoids a uint8
    # round trip that would quantise the photometric correction away.
    af = a[y0:y1, x0:x1].astype(np.float32) / 255.0
    bf = b[y0:y1, x0:x1].astype(np.float32) / 255.0
    med_a, med_b = _median_rgb(af), _median_rgb(bf)
    mid = 0.5 * (med_a + med_b)
    af = np.clip(af + (mid - med_a).astype(np.float32), 0.0, 1.0)
    bf = np.clip(bf + (mid - med_b).astype(np.float32), 0.0, 1.0)

    lab_a = cv2.cvtColor(af, cv2.COLOR_RGB2Lab)
    lab_b = cv2.cvtColor(bf, cv2.COLOR_RGB2Lab)

    radius = max(0, int(shift_px))
    if radius == 0:
        dist = _norm(lab_a - lab_b)
    else:
        ch, cw = lab_a.shape[:2]
        padded = cv2.copyMakeBorder(
            lab_b, radius, radius, radius, radius, cv2.BORDER_REPLICATE
        )
        dist = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                window = padded[
                    radius + dy : radius + dy + ch, radius + dx : radius + dx + cw
                ]
                one = _norm(lab_a - window)
                dist = one if dist is None else np.minimum(dist, one)
        assert dist is not None

    kernel = int(blur)
    if kernel >= 3:
        kernel += 1 - (kernel % 2)  # cv2 needs an odd kernel
        dist = cv2.GaussianBlur(dist, (kernel, kernel), 0)

    out[y0:y1, x0:x1] = dist
    return out


def diff_heat(
    img_a: np.ndarray,
    img_b: np.ndarray,
    roi: Optional[Box] = None,
    blur: int = 5,
    shift_px: int = 1,
    max_side: Optional[int] = None,
    min_delta_e: float = MIN_SCALE_DELTA_E,
    robust_pct: float = ROBUST_PCT,
) -> np.ndarray:
    """:func:`diff_delta_e` rescaled to ``[0, 1]`` for display.

    Args:
        img_a, img_b, roi, blur, shift_px, max_side: see :func:`diff_delta_e`.
        min_delta_e: floor for the rescale, so an unchanged pair stays cold
            instead of having its sensor noise stretched to full scale.
        robust_pct: percentile taken as the maximum.

    Returns:
        ``float32`` ``HxW`` in ``[0, 1]``, 1 marking the strongest change *in
        this ROI*. The scale is relative to the pair, so 0.3 means "clearly less
        changed than the biggest change here", never an absolute distance --
        threshold :func:`diff_delta_e` instead of this.
    """
    delta = diff_delta_e(
        img_a, img_b, roi=roi, blur=blur, shift_px=shift_px, max_side=max_side
    )
    h, w = delta.shape
    x0, y0, x1, y1 = _clip_roi(roi, h, w)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=np.float32)
    inside = delta[y0:y1, x0:x1]
    scale = max(float(np.percentile(inside, float(robust_pct))), float(min_delta_e))
    heat = np.zeros((h, w), dtype=np.float32)
    heat[y0:y1, x0:x1] = np.clip(inside / scale, 0.0, 1.0)
    return heat


# ---------------------------------------------------------------------------
# blobs
# ---------------------------------------------------------------------------
def diff_blobs(
    delta_e: np.ndarray,
    min_delta_e: float = BLOB_DELTA_E,
    min_area: int = 80,
    max_blobs: int = 8,
    merge_gap_px: int = 12,
    max_components: int = MAX_COMPONENTS,
) -> list[DiffBlob]:
    """Regions of change from an **absolute** dE map, strongest first.

    Args:
        delta_e: map from :func:`diff_delta_e` -- not :func:`diff_heat`, whose
            scale depends on the ROI.
        min_delta_e: dE at or above which a pixel counts as changed.
        min_area: components smaller than this are noise and are dropped. The
            default is below a scanner-resolution screw (~170 px); raise it for
            a higher-resolution view.
        max_blobs: keep at most this many, by descending score.
        merge_gap_px: components whose boxes overlap or come within this many
            pixels on both axes are merged into one blob (union box, union mask,
            score recomputed over the merged region). One part rarely thresholds
            into one component -- a cooler breaks into a body plus satellites --
            and without merging those satellites eat every slot and hand SAM a
            box smaller than the part. ``0`` disables merging.
        max_components: only the this many largest raw components take part in
            the merge; the rest are discarded first. A badly registered pair can
            threshold into thousands of specks, and merging is quadratic in the
            component count, so this is the guard that keeps a pathological pair
            from stalling the worker thread.

    Returns:
        Up to ``max_blobs`` :class:`DiffBlob`, sorted by ``score`` descending.

    Raises:
        ValueError: ``delta_e`` is not a 2-D array.
    """
    arr = np.asarray(delta_e, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"delta_e must be HxW, got shape {arr.shape}")

    binary = (arr >= float(min_delta_e)).astype(np.uint8)
    if not binary.any():
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE, CLOSE_KSIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = range(1, count)  # 0 is the background component
    if count - 1 > int(max_components):
        # Largest first, then back into label order so the output stays stable.
        largest = sorted(
            keep, key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i)
        )[: int(max_components)]
        keep = sorted(largest)  # type: ignore[assignment]
        log.warning(
            "diff_blobs: %d components thresholded, max_components=%d - the %d "
            "smallest were discarded before merging. A pair that does this is "
            "usually badly registered rather than genuinely busy.",
            count - 1, int(max_components), count - 1 - int(max_components),
        )
    parts: list[tuple[Box, np.ndarray]] = []
    for i in keep:
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        parts.append(((x, y, x + bw, y + bh), labels[y : y + bh, x : x + bw] == i))

    blobs: list[DiffBlob] = []
    for box, mask in _merge_parts(parts, int(merge_gap_px)):
        area = int(mask.sum())
        if area < int(min_area):
            continue
        x0, y0, x1, y1 = box
        mean = float(arr[y0:y1, x0:x1][mask].mean())
        blobs.append(
            DiffBlob(
                box=box,
                area=area,
                score=mean * math.sqrt(area),
                mask=np.ascontiguousarray(mask),
            )
        )

    blobs.sort(key=lambda blob: blob.score, reverse=True)
    return blobs[: max(0, int(max_blobs))]


def _merge_parts(
    parts: list[tuple[Box, np.ndarray]], gap: int
) -> list[tuple[Box, np.ndarray]]:
    """Fuse components whose boxes are within ``gap`` px; deterministic order.

    Union-find over a vectorised pairwise gap test, iterated on the *group*
    boxes until nothing more merges. The two halves of that matter separately:

    * union-find replaces the first implementation, which re-scanned from the
      start after every fusion -- O(n^3), 9 ms at 65 components but 897 ms at
      537, enough to stall the worker thread on a badly registered pair;
    * iterating on the group boxes rather than settling for one pass over the
      original boxes keeps the *result* identical. A merged group's box reaches
      further than either part's, and on the real D13 cooler two satellites sit
      inside the assembled box while being near no single original component:
      one pass leaves them as separate blobs and the part fragments again.

    Each round is one O(g^2) numpy comparison and the group count falls fast, so
    convergence takes a handful of rounds; :data:`MAX_MERGE_ROUNDS` caps it.

    Output order follows the smallest original index in each group, so it does
    not depend on the input order beyond that.
    """
    if gap <= 0 or len(parts) < 2:
        return list(parts)

    count = len(parts)
    boxes = np.asarray([part[0] for part in parts], dtype=np.int64)
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]  # path halving
            node = parent[node]
        return node

    for _ in range(MAX_MERGE_ROUNDS):
        roots = np.fromiter((find(i) for i in range(count)), dtype=np.int64, count=count)
        labels, inverse = np.unique(roots, return_inverse=True)
        if labels.size < 2:
            break
        group = _group_boxes(boxes, inverse, labels.size)
        pairs = _adjacent_pairs(group, int(gap))
        fused_any = False
        for left, right in pairs:
            root_l, root_r = find(int(labels[left])), find(int(labels[right]))
            if root_l != root_r:
                parent[max(root_l, root_r)] = min(root_l, root_r)
                fused_any = True
        if not fused_any:
            break
    else:
        # The loop ran out of rounds instead of settling, so the groups below
        # are one iteration short of the fixed point: a part may still come out
        # fragmented. Convergence normally takes a handful of rounds.
        log.warning(
            "_merge_parts: hit MAX_MERGE_ROUNDS=%d on %d components with gap=%d; "
            "the merge did not converge and a part may stay fragmented.",
            MAX_MERGE_ROUNDS, count, int(gap),
        )

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    merged: list[tuple[Box, np.ndarray]] = []
    for root in sorted(groups):
        members = groups[root]
        item = parts[members[0]]
        for index in members[1:]:
            item = _fuse(item, parts[index])
        merged.append(item)
    return merged


def _group_boxes(boxes: np.ndarray, inverse: np.ndarray, size: int) -> np.ndarray:
    """Bounding box of each group, as a ``(size, 4)`` int array."""
    big = np.iinfo(np.int64).max
    out = np.empty((size, 4), dtype=np.int64)
    out[:, 0] = big
    out[:, 1] = big
    out[:, 2] = -big
    out[:, 3] = -big
    np.minimum.at(out[:, 0], inverse, boxes[:, 0])
    np.minimum.at(out[:, 1], inverse, boxes[:, 1])
    np.maximum.at(out[:, 2], inverse, boxes[:, 2])
    np.maximum.at(out[:, 3], inverse, boxes[:, 3])
    return out


def _adjacent_pairs(boxes: np.ndarray, gap: int) -> np.ndarray:
    """Index pairs ``(i, j)``, ``i < j``, of boxes within ``gap`` px on both axes."""
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # Gap on each axis: 0 when the projections overlap, the distance otherwise.
    dx = np.maximum(0, np.maximum(x0[:, None] - x1[None, :], x0[None, :] - x1[:, None]))
    dy = np.maximum(0, np.maximum(y0[:, None] - y1[None, :], y0[None, :] - y1[:, None]))
    adjacent = np.triu(np.maximum(dx, dy) <= gap, k=1)
    return np.argwhere(adjacent)


def _fuse(
    left: tuple[Box, np.ndarray], right: tuple[Box, np.ndarray]
) -> tuple[Box, np.ndarray]:
    """One box + mask covering both inputs."""
    box = _union_box(left[0], right[0])
    mask = np.zeros((box[3] - box[1], box[2] - box[0]), dtype=bool)
    for part_box, part_mask in (left, right):
        ox, oy = part_box[0] - box[0], part_box[1] - box[1]
        mask[oy : oy + part_mask.shape[0], ox : ox + part_mask.shape[1]] |= part_mask
    return box, mask


def explain_blobs(
    blobs: list[DiffBlob],
    expected_boxes: list[Box],
    iou_min: float = 0.1,
    contain_min: Optional[float] = 0.7,
) -> tuple[list[DiffBlob], list[DiffBlob]]:
    """Split blobs into the ones this step accounts for and the ones it does not.

    Args:
        blobs: output of :func:`diff_blobs`.
        expected_boxes: bounding boxes of the instances whose shape or state the
            step changes (spec 4.2 step 2), in the same image coordinates.
        iou_min: a blob is explained when its IoU with any expected box reaches
            this -- the "same thing, roughly" case.
        contain_min: a blob is *also* explained when this fraction of its **box
            area** lies inside a single expected box. Removing a cooler reveals
            screw holes, a socket and a patch of board, all far too small for a
            useful IoU with the cooler's box but all obviously part of the same
            event. ``None`` restores the strict IoU rule.

    Returns:
        ``(explained, unexplained)``, each keeping the input order.
    """
    explained: list[DiffBlob] = []
    unexplained: list[DiffBlob] = []
    boxes: Sequence[Box] = expected_boxes
    for blob in blobs:
        area = _area(blob.box)
        covered = False
        for box in boxes:
            if _iou(blob.box, box) >= float(iou_min):
                covered = True
                break
            if contain_min is not None and area > 0:
                if _intersection(blob.box, box) / area >= float(contain_min):
                    covered = True
                    break
        (explained if covered else unexplained).append(blob)
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
