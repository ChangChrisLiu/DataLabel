"""Where the chassis is in a frame: the strategies and the referee.

:func:`tda.core.cache.suggest_roi` is the entry point; this module holds the
measuring.  There are two kinds of picture and they are measured differently.

A **scanner** frame is a white board framed by orange tape with one desktop on
it, photographed from above, and the chassis is found twice:

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

An **OAK** frame (4032x3040, both cameras) is a different picture of the same
workbench: the machine sits inside a yellow tape square that covers roughly a
quarter to a half of the frame, and around it are the bench, the lab floor, the
operator in dark clothes and the robot rig.  :func:`oak_chassis_box` is that
frame's answer -- the tape square at OAK spans (:data:`OAK_TAPE_MIN_SPAN_FRAC`),
then whatever on the board is **not the board**, measured on a downscale
(:data:`OAK_WORK_SIDE`) because a 12 MP colour conversion costs more than the
box is worth.  Both dark and light machines are found by one stage, because
"differs from a white bench" does not care which; the operator is excluded not
by being bright but by standing **off the board**, which is the only gate that
holds when they are wearing black.

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
    "OAK_BOARD_GROW",
    "OAK_MAX_AREA_FRAC",
    "OAK_MAX_ASPECT",
    "OAK_MIN_AREA_FRAC",
    "OAK_MIN_ASPECT",
    "OAK_MIN_DIST",
    "OAK_MIN_RECTANGULARITY",
    "OAK_OPEN_FRAC",
    "OAK_TAPE_MIN_SPAN_FRAC",
    "OAK_WORK_SIDE",
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
    "oak_board_mask",
    "oak_chassis_box",
    "oak_chassis_candidates",
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

# --- the OAK bench -------------------------------------------------------
#: The tape square frames the *machine* in an OAK frame rather than the frame,
#: so it spans far less of the picture than on the scanner. Measured over the
#: 24 frames of ``experiments/roi_oak_eval.py``: 0.33-0.72 of the width on oak1
#: and 0.30-0.50 on oak2, so a quarter is the floor that still rejects a yellow
#: label, a roll of tape on the bench or the wooden block in the oak2 frames.
OAK_TAPE_MIN_SPAN_FRAC = 0.25
#: The square is the region the detector looks in, with no margin around it.
#:
#: The machine does overhang the square, mostly at the edge nearest the camera,
#: and the box therefore clips 30-50 px off the chassis there on D24, D29 and
#: D34. Letting a region be followed off the square was tried at 1.04, 1.07 and
#: 1.12 -- searching the margin outright, and following only a region that was
#: already mostly on the square -- and both are worse over the 50 real frames:
#: the margin reaches the aluminium extrusion at the bench edge and the operator
#: beyond it, the machine's region merges with them, and the box grows from 46 %
#: of the frame to 70 % (D29 oak1) or is refused for being too big (D24 oak1
#: step 1, D36 oak1 steps 1 and 18, all of which have a good tight answer at
#: 1.0). Proposals over the 50 frames: **25 at 1.0**, 23 at 1.04, 24 at 1.07,
#: and the ones at 1.0 are the tighter boxes. A few clipped pixels the annotator
#: drags out beat a box that has swallowed the bench.
OAK_BOARD_GROW = 1.0
#: CIE-Lab distance at which a pixel stops being bench. Higher than the
#: scanner's: the OAK frames carry glare, shadow gradients and the pencil marks
#: on the board, none of which is a machine.
OAK_MIN_DIST = 26.0
#: Opening radius as a fraction of the short side: it breaks the thin bridge a
#: cable, a hand or a strip of shadow makes between the machine and something
#: else on the board.
OAK_OPEN_FRAC = 0.016
#: The machine covers this much of an OAK frame, or the proposal is refused.
#: The plan asked for 8-70 %; 8 % turned out to be exactly the oak2 figure for
#: a small-form-factor Dell (D13 step 1 measures 8.3 %), so the floor sits
#: below it -- a band whose edge is a real answer rejects that answer on the
#: next machine. The ceiling is what stops "the whole bench" and "the whole
#: frame" (D63, D64) being offered as a crop.
OAK_MIN_AREA_FRAC = 0.05
OAK_MAX_AREA_FRAC = 0.70
#: Aspect is width/height. An OAK camera looks at the bench from the side, so a
#: machine is more foreshortened than on the scanner and the band is wider.
OAK_MIN_ASPECT = 0.4
OAK_MAX_ASPECT = 2.6
#: How solidly the component fills its own box. A machine seen from the side is
#: a filled quadrilateral; this rejects an L of shadow or a run of cable.
OAK_MIN_RECTANGULARITY = 0.45
#: Long side the detector measures on. A 12 MP BGR->Lab conversion alone costs
#: ~0.2 s and the box is confirmed by a human anyway, so the whole measurement
#: is made here and scaled back up -- 4 px of quantisation on the original.
OAK_WORK_SIDE = 1024

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
                     min_fill: float = ROI_MIN_RECTANGULARITY,
                     area_band: tuple[float, float] = (ROI_MIN_AREA_FRAC,
                                                       ROI_MAX_AREA_FRAC),
                     aspect_band: tuple[float, float] = (ROI_MIN_ASPECT,
                                                         ROI_MAX_ASPECT),
                     ) -> Optional[float]:
    """Is this box a chassis at all, and how convincingly? ``None`` when it is not.

    Three questions, all about shape rather than about colour: does the box
    cover a plausible fraction of the frame, is it roughly as wide as it is
    tall, and does the component actually *fill* it. The score is the
    rectangularity, which is what tells a chassis from a sprawl that happens to
    span the same corners -- and ``min_fill`` is per strategy, see
    :data:`BED_MIN_RECTANGULARITY`.

    ``area_band`` and ``aspect_band`` default to the scanner's; an OAK frame
    looks at the bench from the side, so it brings its own (:data:`OAK_MIN_AREA_FRAC`).
    """
    x0, y0, x1, y1 = box
    area = float((x1 - x0) * (y1 - y0))
    if area <= 0:
        return None
    frac = area / float(max(width * height, 1))
    if not area_band[0] <= frac <= area_band[1]:
        return None
    aspect = (x1 - x0) / float(max(y1 - y0, 1))
    if not aspect_band[0] <= aspect <= aspect_band[1]:
        return None
    return rectangularity if rectangularity >= min_fill else None


def best_candidate(candidates: list[Candidate], width: int, height: int,
                   min_fill: float = ROI_MIN_RECTANGULARITY,
                   **bands) -> Optional[Candidate]:
    """The most convincing plausible candidate, or ``None`` when none is.

    "Most convincing", not "biggest": a hand or a tool lying against the chassis
    merges into one region whose box is bigger and emptier than the machine's,
    and the machine is then the second candidate.
    """
    scored = [
        (score, candidate)
        for candidate in candidates
        for score in (box_plausibility(*candidate, width, height, min_fill,
                                       **bands),)
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
def board_mask(bgr: np.ndarray,
               min_span_frac: float = TAPE_MIN_SPAN_FRAC) -> Optional[np.ndarray]:
    """The board inside the yellow tape square, or None when there is none.

    The board is framed by a yellow tape square. Small gaps in the tape (and the
    loose corner scraps) are bridged by a dilation, the filled convex hull of
    the bridged square gives the board region, and the tape band itself is then
    removed from it. Nine of the 66 machines cover enough of the tape that no
    square is found at all; for them this is ``None`` and every stage that needs
    a board has to say so rather than guess one.

    ``min_span_frac`` is how much of the frame the square has to cover before it
    counts as one. On the scanner the tape frames the *picture*, so half the
    frame is the right floor; an OAK camera stands back far enough that the same
    square covers a third of the picture, which is what
    :data:`OAK_TAPE_MIN_SPAN_FRAC` is for.
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
    if (stats[label, cv2.CC_STAT_WIDTH] < min_span_frac * width
            or stats[label, cv2.CC_STAT_HEIGHT] < min_span_frac * height):
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


# --------------------------------------------------------------------------- #
# the OAK bench
# --------------------------------------------------------------------------- #
def _downscaled(bgr: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    """``(image, scale)`` where ``scale`` takes a measurement back to full size."""
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= side:
        return bgr, 1.0
    factor = side / float(longest)
    small = cv2.resize(bgr, (max(1, int(round(width * factor))),
                             max(1, int(round(height * factor)))),
                       interpolation=cv2.INTER_AREA)
    return small, longest / float(max(small.shape[:2]))


def oak_board_mask(bgr: np.ndarray, grow: float = 1.0) -> Optional[np.ndarray]:
    """The bench inside the tape square of an OAK frame, or ``None``.

    Two differences from the scanner's :func:`board_mask`, both because the
    machine is *inside* the square here rather than the square being the frame:

    * the square only has to span :data:`OAK_TAPE_MIN_SPAN_FRAC` of the picture;
    * the region is the **rotated bounding rectangle** of the visible tape, not
      its convex hull. A machine standing on the square hides one or two of its
      corners, and the convex hull of the remaining L is a *triangle* -- on the
      real D29, D34 and D64 that triangle cut the board diagonally and left
      half the machine outside the only region the detector may look in. Two
      full sides of a rectangle determine it, so the fitted rectangle recovers
      the whole square from the same L.

    The rectangle is the tape's own, so it leans with the bench; the tape band
    itself is removed from it, as on the scanner. ``grow`` scales it about its
    own centre, which is how :func:`oak_chassis_candidates` asks for the margin
    a machine may overhang into (:data:`OAK_BOARD_GROW`).
    """
    height, width = bgr.shape[:2]
    small_k = _odd(min(height, width) * 0.005)
    bridge_k = _odd(min(height, width) * 0.02)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE,
                              np.ones((small_k, small_k), np.uint8))
    bridged = cv2.dilate(yellow, np.ones((bridge_k, bridge_k), np.uint8))

    label, labels, stats = _largest_component(bridged)
    if label is None or stats[label, cv2.CC_STAT_AREA] < TAPE_MIN_AREA_FRAC * height * width:
        return None
    if (stats[label, cv2.CC_STAT_WIDTH] < OAK_TAPE_MIN_SPAN_FRAC * width
            or stats[label, cv2.CC_STAT_HEIGHT] < OAK_TAPE_MIN_SPAN_FRAC * height):
        return None  # a label, a roll of tape, one strip: not a square

    contours, _ = cv2.findContours((labels == label).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    centre, size, angle = cv2.minAreaRect(np.vstack(contours))
    scaled = (centre, (size[0] * float(grow), size[1] * float(grow)), angle)
    board = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(board, np.int32(np.round(cv2.boxPoints(scaled))), 255)
    board = cv2.erode(board, np.ones((bridge_k, bridge_k), np.uint8))  # undo the bridging
    board[yellow > 0] = 0  # the tape band is not part of the inner region
    return board


def oak_chassis_candidates(bgr: np.ndarray) -> list[Candidate]:
    """The largest things on the OAK bench that are not the bench, biggest first.

    "Not the bench" rather than "dark": the machines in this dataset run from a
    black Dell small-form-factor to a bare silver Apple G4, and half of every
    open chassis is bright drive cage and PSU label anyway, so a luminance
    threshold returns a piece of the machine instead of the machine. The bench
    is a white board with a known colour, taken as the **median** of the board
    region (robust: the machine is well under half of it), and the distance is
    measured in CIE-Lab so a silver chassis is as visible as a black one.

    Everything that makes this safe is the board. The lab floor, the operator --
    who wears black and can be the largest dark region in an oak2 frame -- the
    robot rig and the parts trolley are all *outside* the tape square, so they
    are not candidates at any threshold. Without a square there is no board and
    therefore no candidate at all, which is the honest answer for the frames
    where the machine covers the tape (D63, D64).

    An opening breaks the thin bridge a cable, a shadow or a hand makes between
    the machine and something else on the board, and the top
    :data:`TOP_COMPONENTS` are offered because it does not always.

    Nothing off the square is looked at, not even by following a region that
    starts on it: see :data:`OAK_BOARD_GROW` for what that costs and what it
    saves.
    """
    height, width = bgr.shape[:2]
    board = oak_board_mask(bgr, OAK_BOARD_GROW)
    if board is None:
        return []
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bench = np.median(lab[board > 0], axis=0)
    foreground = (np.linalg.norm(lab - bench, axis=2) >= OAK_MIN_DIST).astype(np.uint8)
    foreground[board == 0] = 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    foreground[cv2.inRange(hsv, YELLOW_LO, YELLOW_HI) > 0] = 0  # the tape is not a part

    small_k = _odd(min(height, width) * 0.005)
    open_k = _odd(min(height, width) * OAK_OPEN_FRAC)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((small_k, small_k), np.uint8)
    )
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)
    )
    return _candidates(foreground, OAK_MIN_AREA_FRAC * height * width, width, height)


def oak_chassis_box(bgr: np.ndarray) -> Optional[Candidate]:
    """The machine on an OAK bench, in **original** pixels, or ``None``.

    Measured on a downscale (:data:`OAK_WORK_SIDE`) and scaled back: a 12 MP
    colour conversion costs about a fifth of a second and the ROI is confirmed
    by a human, so 4 px of quantisation is not worth 10x the time.
    """
    small, scale = _downscaled(bgr, OAK_WORK_SIDE)
    height, width = small.shape[:2]
    found = best_candidate(
        oak_chassis_candidates(small), width, height, OAK_MIN_RECTANGULARITY,
        area_band=(OAK_MIN_AREA_FRAC, OAK_MAX_AREA_FRAC),
        aspect_band=(OAK_MIN_ASPECT, OAK_MAX_ASPECT),
    )
    if found is None:
        return None
    (x0, y0, x1, y1), fill = found
    full_h, full_w = bgr.shape[:2]
    box = (max(0, int(round(x0 * scale))), max(0, int(round(y0 * scale))),
           min(full_w, int(round(x1 * scale))), min(full_h, int(round(y1 * scale))))
    return (box, fill)
