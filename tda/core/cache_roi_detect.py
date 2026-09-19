"""Where the chassis is in a scanner frame: two strategies and one referee.

:func:`tda.core.cache.suggest_roi` is the entry point; this module holds the
measuring.  A scanner frame is a white board framed by orange tape with one
desktop on it, photographed from above, and the chassis is found twice:

* :func:`scan_chassis_candidates` -- the largest **dark** objects inside the
  tape square.  That is what a black or dark-grey machine is, and it is what the
  thresholds were calibrated on.
* :func:`scan_bed_candidates` -- the largest regions that are **not the scan
  bed**, measured against a background model taken from the frame's border ring.
  That is what a light or silver machine is: on D64 the dark stage returns the
  motherboard's PCB at 13 % of the frame, with every sampled step agreeing, so
  no median over steps can catch it either.

Both answer with *candidates* rather than with one box, because the largest
region is not always the right one: a hand or a tool lying against the chassis
merges into it, and the second- or third-largest region is then the machine.
:func:`box_plausibility` is the referee -- area, aspect and how solidly the
component fills its own bounding box -- and it judges both strategies on the
same scale, so "which of these is a chassis" is one question asked once.

Every box is the component's **axis-aligned** bounding box, padded by
:data:`ROI_PAD_FRAC`.  A ``minAreaRect`` is tempting and wrong here: the ROI
crops an upright rectangle, and the axis-aligned hull of a rotated rect around
an irregular blob is far larger than the blob (on the real D64 it reached 74 %
of the frame and included bare scan bed, against 45 % for the plain bbox).

Images are OpenCV BGR arrays and all coordinates are in *original* image pixels.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

__all__ = [
    "BED_MIN_AREA_FRAC",
    "BED_MIN_DIST",
    "BED_OPEN_FRAC",
    "BED_RING_FRAC",
    "Candidate",
    "DARK_MAX",
    "DARK_MIN_AREA_FRAC",
    "ROI_MAX_AREA_FRAC",
    "ROI_MAX_ASPECT",
    "ROI_MIN_AREA_FRAC",
    "ROI_MIN_ASPECT",
    "BED_MIN_RECTANGULARITY",
    "DARK_MIN_RECTANGULARITY",
    "ROI_MIN_RECTANGULARITY",
    "ROI_PAD_FRAC",
    "TOP_COMPONENTS",
    "YELLOW_HI",
    "YELLOW_LO",
    "best_candidate",
    "board_mask",
    "box_plausibility",
    "pad_box",
    "scan_bed_box",
    "scan_bed_candidates",
    "scan_chassis_box",
    "scan_chassis_candidates",
]

# --- the tape square -----------------------------------------------------
YELLOW_LO = (10, 60, 60)  # HSV bounds of the yellow/orange tape square
YELLOW_HI = (40, 255, 255)
TAPE_MIN_AREA_FRAC = 0.005  # ignore yellow specks: no tape found -> fallback
TAPE_MIN_SPAN_FRAC = 0.5  # ... and a tape square frames the board, so it has to
#                           span at least half the frame; a smaller yellow blob is
#                           something else (a label, a cable) -> fallback

# --- the dark-object strategy --------------------------------------------
DARK_MAX = 90  # gray below this counts as "dark object" (chassis)
DARK_MIN_AREA_FRAC = 0.005
ROI_PAD_FRAC = 0.03  # pad the chassis box by 3% of its own size

# --- the scan-bed strategy -----------------------------------------------
# The bed is a bright, low-saturation board, so a chassis of *any* colour is
# what differs from it.
BED_RING_FRAC = 0.03  # width of the border ring the background is modelled from
BED_MIN_DIST = 18.0  # CIE-Lab distance at which a pixel stops being "the bed"
BED_MIN_AREA_FRAC = 0.02  # a region smaller than this is not a chassis
BED_OPEN_FRAC = 0.012  # opening radius: breaks a hand or cable off the chassis

# --- the referee ---------------------------------------------------------
#: How many of the largest components each strategy offers for judging. Three
#: is enough for the case it exists for -- an arm or a tool merged into the
#: chassis leaves the machine as the second region, not the tenth -- and small
#: enough that "the biggest plausible thing" cannot drift into a speck.
TOP_COMPONENTS = 3
#: A box outside these bounds is not a chassis seen from above. The area bounds
#: are the ones :mod:`tda.core.cache_thumbs` already applied after the fact.
ROI_MIN_AREA_FRAC = 0.20
ROI_MAX_AREA_FRAC = 0.95
#: Aspect is width/height: the 66 real scanner frames run 0.75-1.55, and a box
#: thinner than 1:2 either way is a panel, a cable run or a strip of tape.
ROI_MIN_ASPECT = 0.5
ROI_MAX_ASPECT = 2.0
#: How solidly the component fills its own (unpadded, axis-aligned) bounding
#: box -- and it is per strategy, because the two measure different things.
#:
#: The **bed** strategy's component is the whole machine, silhouette and all, so
#: it fills its box: over the first frame of all 66 real scanner desktops the
#: bed components score 0.48-0.98. What a low bed fill means is exactly the
#: failure this guard is for -- a hand, an arm or a tool merged into the chassis,
#: or a region grown out into the bench -- so the floor sits just under the
#: measured range.
#:
#: The **dark** strategy's component is only the *dark parts* of a machine that
#: is open on the scanner: the frame, the shadows and the PSU, with a bright
#: motherboard, drive cages and cables in between. Its fill is structurally
#: lower and runs 0.20-0.74 over the same 66 frames, every one of them a box
#: that is right. A floor near the bed's would throw seventeen correct answers
#: away, so this one only has to reject a spray of specks whose bounding box
#: means nothing.
BED_MIN_RECTANGULARITY = 0.45
DARK_MIN_RECTANGULARITY = 0.18
#: The default for a caller that does not say which strategy it is judging.
ROI_MIN_RECTANGULARITY = DARK_MIN_RECTANGULARITY

#: One strategy's answer: the padded box and how solidly the component filled
#: its own bounding box (see :data:`ROI_MIN_RECTANGULARITY`).
Candidate = tuple[tuple[int, int, int, int], float]


def _odd(value: float, minimum: int = 3) -> int:
    """Nearest odd kernel size >= ``minimum``."""
    return max(minimum, int(round(value)) | 1)


def pad_box(box: tuple[int, int, int, int], width: int, height: int,
            frac: float = ROI_PAD_FRAC) -> tuple[int, int, int, int]:
    """Grow ``box`` by ``frac`` of its own size, clipped to the image."""
    x0, y0, x1, y1 = box
    px, py = int(round((x1 - x0) * frac)), int(round((y1 - y0) * frac))
    return (max(0, x0 - px), max(0, y0 - py), min(width, x1 + px), min(height, y1 + py))


def box_plausibility(box: tuple[int, int, int, int], rectangularity: float,
                     width: int, height: int,
                     min_fill: float = ROI_MIN_RECTANGULARITY) -> Optional[float]:
    """Is this box a chassis at all, and how convincingly? ``None`` when it is not.

    Three questions, all about shape rather than about colour: does the box
    cover a plausible fraction of the frame, is it roughly as wide as it is
    tall, and does the component actually *fill* it. The score is the
    rectangularity, which is what tells a chassis from a sprawl that happens to
    span the same corners -- and ``min_fill`` is per strategy, see
    :data:`BED_MIN_RECTANGULARITY`.
    """
    x0, y0, x1, y1 = box
    area = float((x1 - x0) * (y1 - y0))
    if area <= 0:
        return None
    frac = area / float(max(width * height, 1))
    if not ROI_MIN_AREA_FRAC <= frac <= ROI_MAX_AREA_FRAC:
        return None
    aspect = (x1 - x0) / float(max(y1 - y0, 1))
    if not ROI_MIN_ASPECT <= aspect <= ROI_MAX_ASPECT:
        return None
    return rectangularity if rectangularity >= min_fill else None


def best_candidate(candidates: list[Candidate], width: int, height: int,
                   min_fill: float = ROI_MIN_RECTANGULARITY) -> Optional[Candidate]:
    """The most convincing plausible candidate, or ``None`` when none is.

    "Most convincing", not "biggest": a hand or a tool lying against the chassis
    merges into one region whose box is bigger and emptier than the machine's,
    and the machine is then the second candidate.
    """
    scored = [
        (score, candidate)
        for candidate in candidates
        for score in (box_plausibility(*candidate, width, height, min_fill),)
        if score is not None
    ]
    return max(scored, key=lambda pair: pair[0])[1] if scored else None


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #
def _largest_component(mask: np.ndarray) -> tuple[Optional[int], np.ndarray, np.ndarray]:
    """Label of the biggest non-background blob in ``mask`` plus the label image."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None, labels, stats
    return 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), labels, stats


def _candidates(mask: np.ndarray, min_area: float, width: int, height: int,
                limit: int = TOP_COMPONENTS) -> list[Candidate]:
    """The ``limit`` largest components of ``mask`` as boxes, biggest first.

    Each carries its own fill ratio, so a component merged with a hand can be
    told from the chassis alone without looking at the pixels again.
    """
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    order = sorted(range(1, count), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    out: list[Candidate] = []
    for label in order[:limit]:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            break  # sorted by area: everything after this one is smaller still
        x, y, w, h = (int(v) for v in stats[label, :4])
        if w <= 0 or h <= 0:
            continue
        out.append((pad_box((x, y, x + w, y + h), width, height), area / float(w * h)))
    return out


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #
def board_mask(bgr: np.ndarray) -> Optional[np.ndarray]:
    """The scan bed inside the yellow tape square, or None when there is none.

    The board is framed by a yellow tape square. Small gaps in the tape (and the
    loose corner scraps) are bridged by a dilation, the filled convex hull of
    the bridged square gives the board region, and the tape band itself is then
    removed from it. Nine of the 66 machines cover enough of the tape that no
    square is found at all; for them this is ``None`` and every stage that needs
    a board has to say so rather than guess one.
    """
    height, width = bgr.shape[:2]
    small_k = _odd(min(height, width) * 0.005)
    bridge_k = _odd(min(height, width) * 0.02)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((small_k, small_k), np.uint8))
    bridged = cv2.dilate(yellow, np.ones((bridge_k, bridge_k), np.uint8))

    label, labels, stats = _largest_component(bridged)
    if label is None or stats[label, cv2.CC_STAT_AREA] < TAPE_MIN_AREA_FRAC * height * width:
        return None
    if (stats[label, cv2.CC_STAT_WIDTH] < TAPE_MIN_SPAN_FRAC * width
            or stats[label, cv2.CC_STAT_HEIGHT] < TAPE_MIN_SPAN_FRAC * height):
        return None  # not a square framing the board

    contours, _ = cv2.findContours((labels == label).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    board = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(board, cv2.convexHull(max(contours, key=cv2.contourArea)), 255)
    board = cv2.erode(board, np.ones((bridge_k, bridge_k), np.uint8))  # undo the bridging
    board[yellow > 0] = 0  # the tape band is not part of the inner region
    return board


# --------------------------------------------------------------------------- #
# strategy 1: the darkest thing on the board
# --------------------------------------------------------------------------- #
def scan_chassis_candidates(bgr: np.ndarray) -> list[Candidate]:
    """The largest dark objects on the board, biggest first; empty without a board."""
    height, width = bgr.shape[:2]
    board = board_mask(bgr)
    if board is None:
        return []
    small_k = _odd(min(height, width) * 0.005)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = ((gray < DARK_MAX) & (board > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((small_k, small_k), np.uint8))
    return _candidates(dark, DARK_MIN_AREA_FRAC * height * width, width, height)


# --------------------------------------------------------------------------- #
# strategy 2: whatever is not the scan bed
# --------------------------------------------------------------------------- #
def scan_bed_candidates(bgr: np.ndarray) -> list[Candidate]:
    """The largest regions that differ from the scan bed, biggest first.

    The background is modelled from the frame's border ring, which is bed on
    every one of the 66 real frames whatever the machine is made of, and the
    distance is measured in CIE-Lab so a silver chassis is as visible as a black
    one. The orange tape is removed -- it differs from the bed too and would weld
    the chassis to the frame's edge -- and the search is confined to the board
    where there is one: without that the region grows through the tape and the
    edge shadow into the whole scan.

    An **opening** runs before the labelling. A hand, an arm or a tool lying
    against the machine is not the bed either, and touches it along a thin
    bridge; the opening breaks that bridge, and the top-:data:`TOP_COMPONENTS`
    candidates cover the case where it does not.
    """
    height, width = bgr.shape[:2]
    ring = max(4, int(round(min(height, width) * BED_RING_FRAC)))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    border = np.zeros((height, width), bool)
    border[:ring, :] = border[-ring:, :] = True
    border[:, :ring] = border[:, -ring:] = True
    background = np.median(lab[border], axis=0)  # robust: the ring is mostly bed

    distance = np.linalg.norm(lab - background, axis=2)
    foreground = (distance >= BED_MIN_DIST).astype(np.uint8)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    foreground[cv2.inRange(hsv, YELLOW_LO, YELLOW_HI) > 0] = 0  # the tape is not a part
    board = board_mask(bgr)
    if board is not None:
        foreground[board == 0] = 0

    small_k = _odd(min(height, width) * 0.005)
    open_k = _odd(min(height, width) * BED_OPEN_FRAC)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((small_k, small_k), np.uint8)
    )
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)
    )
    return _candidates(foreground, BED_MIN_AREA_FRAC * height * width, width, height)


# --------------------------------------------------------------------------- #
# the two, as one answer each
# --------------------------------------------------------------------------- #
def scan_chassis_box(bgr: np.ndarray) -> Optional[Candidate]:
    """The most convincing dark-object candidate, or ``None``."""
    height, width = bgr.shape[:2]
    return best_candidate(scan_chassis_candidates(bgr), width, height,
                          DARK_MIN_RECTANGULARITY)


def scan_bed_box(bgr: np.ndarray) -> Optional[Candidate]:
    """The most convincing not-the-bed candidate, or ``None``."""
    height, width = bgr.shape[:2]
    return best_candidate(scan_bed_candidates(bgr), width, height,
                          BED_MIN_RECTANGULARITY)
