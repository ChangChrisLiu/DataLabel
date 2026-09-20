"""Turn a change region into a *part* proposal that can prompt SAM (spec 4.2).

:func:`tda.core.diffmap.diff_blobs` answers "where did something change?"; this
answers the narrower question the reverse-order flow actually asks: **which of
those changes is the part the task card wants, and where inside it can SAM be
pointed?**

The measurement that motivates every line here is in
``experiments_out/plan_b_probe/transfer/report.md`` §7 and reproduced by
``experiments/plan_b_probe/diff_eval.py``: on 201 real removal events the top
diff blob *overlaps* the removed part 47-64 % of the time but its **centroid**
is inside it only 21-41 % of the time. Removing a part reveals a hole, a socket
and a patch of board, ``diff_blobs`` merges all of that into one region, and the
centroid of the merged region is on none of it. So:

* **split** the merged region back apart -- components at a second, higher dE
  threshold, plus a watershed cut at the distance-transform necks of the
  low-threshold components;
* **rank** the pieces instead of taking the biggest (the class's area prior from
  ``configs/area_priors.yaml``, how solid the piece is, and the *direction* of
  the change: the part is still there in the frame being annotated and gone in
  the next one, so its interior should lose texture rather than gain it);
* take the point from the **distance transform**, not from the centroid, so the
  point is inside the region by construction -- a C-shaped bracket's centroid is
  in the air, its distance-transform peak is in the metal.

Direction, because it is easy to get backwards
----------------------------------------------
Annotation runs in reverse. ``prev_rgb`` is the frame **being annotated**, where
the part is still present; ``cur_rgb`` is the next step, where it is gone. That
is the same order :func:`tda.core.diffmap.diff_delta_e` is called in by the
probe harness, and the opposite of the argument order the window happens to use
(:meth:`tda.ui.app_diff.AssistController.compute` passes the neighbour first);
the dE map is symmetric so only the ranking cue below cares, and it is the one
thing that must not be fed the wrong way round.

Conventions match :mod:`tda.core.diffmap`: ``HxWx3`` RGB ``uint8`` frames, boxes
``(x0, y0, x1, y1)`` with exclusive ends **in full-frame coordinates whatever
ROI was passed**, and everything here is deterministic, Qt-free and side-effect
free.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from tda.core.diffmap import BLOB_DELTA_E, DiffBlob, diff_blobs, diff_delta_e

log = logging.getLogger(__name__)

__all__ = [
    "HIGH_DELTA_E",
    "MAX_ROI_FRAC",
    "PartProposal",
    "propose_parts",
]

Box = tuple[int, int, int, int]

#: Second, stricter threshold. A merged region's *cores* survive it while the
#: lukewarm bridge between them -- the shadow, the cable, the board that merely
#: changed shade -- does not, which is what pulls the part away from the hole.
HIGH_DELTA_E = 2.0 * BLOB_DELTA_E
#: A proposal covering more of the ROI than this is not a prompt; it says
#: "everything changed". Mirrors ``tda.ui.app_assist.MAX_PROMPT_BOX_FRAC``, and
#: it is why a lighting jump comes back as ``[]`` instead of as one huge box.
MAX_ROI_FRAC = 0.6
#: Two candidates this similar are the same answer; the weaker one is dropped.
DEDUP_IOU = 0.75
#: A candidate covered this much by an already-drawn shape is already explained.
KNOWN_COVER = 0.7
#: Distance-transform fraction that separates "surely one lobe" from the necks
#: between lobes, for the watershed markers.
MARKER_FRAC = 0.5
#: How many ``diff_blobs`` regions are split at all.
MAX_PARENTS = 8
#: Exponent on the area-prior fit. 1.0 = the prior may move a candidate by the
#: full factor the fit reports.
PRIOR_WEIGHT = 1.0
#: Weight of the texture-direction cue; the multiplier lands in [1-w, 1+w].
APPEAR_WEIGHT = 0.5
#: Exponent on solidity (region area / box area). A merged region of two lobes
#: fills about half of its own box; a part fills most of it.
SOLIDITY_WEIGHT = 0.5


@dataclass(frozen=True)
class PartProposal:
    """One candidate part, ready to be handed to SAM as ``box + point``.

    Attributes:
        box: ``(x0, y0, x1, y1)`` in full-frame coordinates, ends exclusive.
        point: a point **inside** the proposed region -- the distance-transform
            peak, never the centroid (see the module docstring).
        score: ranking score; absolute, so it is comparable between candidates
            of one frame pair but not between frame pairs.
        blob_index: index of the :func:`~tda.core.diffmap.diff_blobs` blob this
            piece came out of, so the UI can still talk about "the change".
        area: pixel count of the region, smaller than the box for anything that
            is not a rectangle.
        mask: box-local boolean mask, origin at ``box[:2]``, so the canvas can
            outline the piece. Excluded from equality.
    """

    box: Box
    point: tuple[int, int]
    score: float
    blob_index: int
    area: int = 0
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_rgb_uint8(image: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be HxWx3 RGB, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got dtype {arr.dtype}")
    return arr


def _clip_roi(roi: Optional[Box], h: int, w: int) -> Box:
    if roi is None:
        return (0, 0, w, h)
    x0, y0, x1, y1 = (int(round(float(v))) for v in roi)
    return (max(0, min(x0, w)), max(0, min(y0, h)),
            max(0, min(x1, w)), max(0, min(y1, h)))


def _box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _components(binary: np.ndarray, min_area: int) -> list[tuple[Box, np.ndarray]]:
    """Connected components of a boolean array, **unmerged**, as box + mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    out: list[tuple[Box, np.ndarray]] = []
    for i in range(1, count):
        if int(stats[i, cv2.CC_STAT_AREA]) < int(min_area):
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append(((x, y, x + bw, y + bh),
                    np.ascontiguousarray(labels[y:y + bh, x:x + bw] == i)))
    return out


def _watershed_split(region: np.ndarray, weight: np.ndarray,
                     min_area: int) -> list[np.ndarray]:
    """Cut one boolean region at the necks of its distance transform.

    Returns ``[]`` when the region has a single distance-transform lobe, which
    is the guard that keeps a plain rectangle -- a lighting jump over the whole
    ROI, a part that really is one blob -- from being chopped into arbitrary
    pieces. ``weight`` is the dE map over the same window; it only decides where
    inside the neck the cut falls.
    """
    binary = np.ascontiguousarray(region.astype(np.uint8))
    if int(binary.sum()) < 2 * int(min_area):
        return []
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak <= 1.0:
        return []
    sure = (dist >= MARKER_FRAC * peak).astype(np.uint8)
    count, markers = cv2.connectedComponents(sure)
    if count - 1 < 2:
        return []
    markers = markers.astype(np.int32) + 1      # background label becomes 1
    markers[binary == 0] = 1
    markers[(binary != 0) & (sure == 0)] = 0    # the necks: to be decided
    grey = np.clip(weight, 0.0, 255.0).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), markers)
    pieces: list[np.ndarray] = []
    for label in range(2, count + 1):
        piece = (markers == label) & (binary != 0)
        if int(piece.sum()) >= int(min_area):
            pieces.append(np.ascontiguousarray(piece))
    return pieces if len(pieces) >= 2 else []


def _dt_point(box: Box, mask: np.ndarray) -> tuple[int, int]:
    """The distance-transform peak of a region, in full-frame coordinates.

    Deep inside the region by construction, which the centroid is not: a merged
    or C-shaped region's centroid regularly sits in the hole it wraps around,
    and that is most of the 47-64 % versus 21-41 % gap the probe measured.
    """
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    index = int(np.argmax(dist))
    y, x = divmod(index, mask.shape[1])
    return (box[0] + int(x), box[1] + int(y))


def _prior_fit(frac: float, band: Optional[tuple[float, float]]) -> float:
    """How well an area fraction agrees with a class's band; 1.0 inside it.

    Decays as the square of the *log* ratio to the nearest bound, so being a
    factor of e out halves the score and a factor of e**2 out divides it by
    five. The bands in ``configs/area_priors.yaml`` are p05-p95 and deliberately
    generous, and they were measured against a nominal ROI rather than this
    frame's, so anything sharper than this would be reading noise.
    """
    if band is None:
        return 1.0
    try:
        low, high = float(band[0]), float(band[1])
    except (TypeError, ValueError, IndexError):
        return 1.0
    if not (low > 0.0 and high > 0.0) or frac <= 0.0:
        return 1.0
    if low > high:
        low, high = high, low
    if low <= frac <= high:
        return 1.0
    ratio = frac / low if frac < low else frac / high
    return 1.0 / (1.0 + math.log(ratio) ** 2)


def _appearance(grad_prev: np.ndarray, grad_cur: np.ndarray, box: Box,
                mask: np.ndarray, origin: tuple[int, int]) -> float:
    """``(prev - cur) / (prev + cur)`` of mean edge energy inside the region.

    Positive means the region carries more detail in the frame being annotated
    than in the next one -- a part that was lifted out. Negative means the
    opposite -- a socket or a screw hole that the removal *revealed*. The two
    are the pair this whole module exists to tell apart.
    """
    ox, oy = origin
    x0, y0, x1, y1 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
    h, w = grad_prev.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
        return 0.0
    sub_prev = grad_prev[y0:y1, x0:x1][mask]
    sub_cur = grad_cur[y0:y1, x0:x1][mask]
    if sub_prev.size == 0:
        return 0.0
    a, b = float(sub_prev.mean()), float(sub_cur.mean())
    total = a + b
    return 0.0 if total <= 1e-6 else float((a - b) / total)


def _gradient(image: np.ndarray, roi: Box) -> np.ndarray:
    """Sobel magnitude of the ROI window, as float32 -- the texture measure."""
    x0, y0, x1, y1 = roi
    grey = cv2.cvtColor(np.ascontiguousarray(image[y0:y1, x0:x1]),
                        cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _mask_iou(a: tuple[Box, np.ndarray], b: tuple[Box, np.ndarray]) -> float:
    """IoU of two box-local masks."""
    (ax0, ay0, ax1, ay1), am = a
    (bx0, by0, bx1, by1), bm = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    area_a, area_b = int(am.sum()), int(bm.sum())
    if ix1 <= ix0 or iy1 <= iy0 or area_a == 0 or area_b == 0:
        return 0.0
    inter = int((am[iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
                 & bm[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]).sum())
    union = area_a + area_b - inter
    return float(inter) / float(union) if union else 0.0


def _covered_by(box: Box, mask: np.ndarray,
                known: Sequence[np.ndarray]) -> float:
    """Largest fraction of a region already covered by one drawn shape."""
    area = int(mask.sum())
    if area == 0:
        return 0.0
    x0, y0, x1, y1 = box
    best = 0.0
    for other in known:
        arr = np.asarray(other)
        if arr.ndim != 2:
            continue
        sub = arr[y0:y1, x0:x1]
        if sub.shape != mask.shape:
            continue
        best = max(best, float(int((sub.astype(bool) & mask).sum())) / area)
    return best


# ---------------------------------------------------------------------------
# the proposal
# ---------------------------------------------------------------------------
def propose_parts(
    prev_rgb: np.ndarray,
    cur_rgb: np.ndarray,
    roi: Optional[Box] = None,
    *,
    expect_area: Optional[tuple[float, float]] = None,
    known_masks: Sequence[np.ndarray] = (),
    max_proposals: int = 3,
    min_area: int = 80,
    delta_e: Optional[np.ndarray] = None,
    max_side: Optional[int] = None,
) -> list[PartProposal]:
    """Rank the parts a change between two frames could be about.

    Args:
        prev_rgb: the frame **being annotated** -- the part is still present.
        cur_rgb: the next step, where it is gone. Same shape as ``prev_rgb``.
        roi: the pose segment's ROI, ``(x0, y0, x1, y1)``. ``None`` uses the
            whole frame, which is only sensible on a synthetic pair: on a real
            one the table around the machine dominates.
        expect_area: ``(min_frac, max_frac)`` of the ROI area for the class the
            card is asking for -- ``configs/area_priors.yaml``. The single most
            useful cue, because a screw and the region its removal reveals differ
            by two orders of magnitude in area and by nothing else. ``None``
            ranks on shape and texture alone.
        known_masks: shapes already drawn on this frame; a candidate more than
            :data:`KNOWN_COVER` covered by one of them is not proposed again.
        max_proposals: how many to return, best first (``Shift+C`` cycles them).
        min_area: pixels below which a piece is noise. Scale it with the view's
            resolution, as the probe harness does.
        delta_e: an already-computed :func:`~tda.core.diffmap.diff_delta_e` map
            for this pair and ROI; the UI has one and must not pay twice.
        max_side: passed to :func:`~tda.core.diffmap.diff_delta_e` when the map
            has to be computed here.

    Returns:
        Up to ``max_proposals`` :class:`PartProposal`, best first. **Empty** when
        nothing changed, when everything changed (a lighting jump: a full-ROI
        box narrows nothing, so it is not offered as one), or when every
        candidate is already drawn.

    Raises:
        ValueError: a frame is not ``HxWx3`` uint8 RGB, or the two differ in
            shape.
    """
    prev = _as_rgb_uint8(prev_rgb, "prev_rgb")
    cur = _as_rgb_uint8(cur_rgb, "cur_rgb")
    if prev.shape != cur.shape:
        raise ValueError(f"frames differ in shape: {prev.shape} vs {cur.shape}")

    h, w = prev.shape[:2]
    window = _clip_roi(roi, h, w)
    roi_area = float(max(1, _box_area(window)))
    if roi_area <= 1.0:
        return []

    delta = (np.asarray(delta_e, dtype=np.float32) if delta_e is not None
             else diff_delta_e(prev, cur, roi=roi, max_side=max_side))
    blobs = diff_blobs(delta, min_area=min_area, max_blobs=MAX_PARENTS)
    if not blobs:
        return []

    candidates = _tightest(_candidates(delta, blobs, int(min_area), roi_area))
    if not candidates:
        return []

    grad_prev = _gradient(prev, window)
    grad_cur = _gradient(cur, window)
    origin = (window[0], window[1])
    known = [np.asarray(m) for m in known_masks if np.asarray(m).ndim == 2]

    scored: list[tuple[float, Box, np.ndarray, int]] = []
    for box, mask, parent in candidates:
        area = int(mask.sum())
        if area < int(min_area):
            continue
        if known and _covered_by(box, mask, known) >= KNOWN_COVER:
            continue
        strength = float(delta[box[1]:box[3], box[0]:box[2]][mask].mean())
        base = strength * math.sqrt(area)
        prior = _prior_fit(area / roi_area, expect_area) ** PRIOR_WEIGHT
        appear = _appearance(grad_prev, grad_cur, box, mask, origin)
        solidity = (area / float(max(1, _box_area(box)))) ** SOLIDITY_WEIGHT
        score = base * prior * (1.0 + APPEAR_WEIGHT * appear) * solidity
        scored.append((float(score), box, mask, parent))

    # Sort by score, then by geometry so that equal scores never depend on the
    # order the candidate generators happened to run in.
    scored.sort(key=lambda item: (-item[0], item[1]))

    return [
        PartProposal(box=box, point=_dt_point(box, mask), score=score,
                     blob_index=parent, area=int(mask.sum()),
                     mask=np.ascontiguousarray(mask))
        for score, box, mask, parent in scored[: max(0, int(max_proposals))]
    ]


def _tightest(candidates: list[tuple[Box, np.ndarray, int]]
              ) -> list[tuple[Box, np.ndarray, int]]:
    """Of every group of near-identical regions, keep the one with the tightest box.

    Done **before** ranking and not as a tie-break after it, because the score
    cannot see what is wrong here. A merged blob and the one child that
    dominates it are the same region to within a few per cent of area, and the
    parent regularly scores a hair higher (it swallows the brightest pixels of
    its other child) while being the strictly worse *prompt* -- its box reaches
    across everything that was merged in. The box is what SAM is given, so among
    equals the smallest box wins.
    """
    kept: list[tuple[Box, np.ndarray, int]] = []
    for item in sorted(candidates, key=lambda c: (_box_area(c[0]), c[0])):
        if any(_mask_iou((item[0], item[1]), (other[0], other[1])) >= DEDUP_IOU
               for other in kept):
            continue
        kept.append(item)
    return kept


def _candidates(delta: np.ndarray, blobs: list[DiffBlob], min_area: int,
                roi_area: float) -> list[tuple[Box, np.ndarray, int]]:
    """Every piece a blob could be about: the blob, its cores, its lobes.

    Three generators, all restricted to the blob's own region so a piece can
    never straddle two unrelated changes:

    * the **blob itself** -- when a part really is one clean region, splitting
      it would only fragment the answer;
    * components at :data:`HIGH_DELTA_E` -- the merged region's cores, without
      the lukewarm bridge that joined them;
    * a **watershed** cut of the ``BLOB_DELTA_E`` components at their
      distance-transform necks, which separates two lobes that are genuinely
      connected at full strength (a part still touching the socket it came out
      of).

    Anything covering more than :data:`MAX_ROI_FRAC` of the ROI is dropped here
    rather than ranked: it is not a prompt at any score.
    """
    limit = MAX_ROI_FRAC * roi_area
    out: list[tuple[Box, np.ndarray, int]] = []

    def offer(box: Box, mask: np.ndarray, parent: int) -> None:
        if int(mask.sum()) < min_area or _box_area(box) > limit:
            return
        out.append((box, np.ascontiguousarray(mask), parent))

    for index, blob in enumerate(blobs):
        bx0, by0, bx1, by1 = blob.box
        region = (blob.mask if blob.mask is not None
                  else np.ones((by1 - by0, bx1 - bx0), dtype=bool))
        sub = delta[by0:by1, bx0:bx1]
        offer(tuple(int(v) for v in blob.box), region, index)  # type: ignore[arg-type]

        for box, mask in _components((sub >= HIGH_DELTA_E) & region, min_area):
            offer((bx0 + box[0], by0 + box[1], bx0 + box[2], by0 + box[3]),
                  mask, index)

        for box, mask in _components((sub >= BLOB_DELTA_E) & region, min_area):
            full = (bx0 + box[0], by0 + box[1], bx0 + box[2], by0 + box[3])
            offer(full, mask, index)
            for piece in _watershed_split(mask, sub[box[1]:box[3],
                                                    box[0]:box[2]], min_area):
                ys, xs = np.nonzero(piece)
                if xs.size == 0:
                    continue
                px0, py0 = int(xs.min()), int(ys.min())
                px1, py1 = int(xs.max()) + 1, int(ys.max()) + 1
                offer((full[0] + px0, full[1] + py0,
                       full[0] + px1, full[1] + py1),
                      piece[py0:py1, px0:px1], index)
    return out
