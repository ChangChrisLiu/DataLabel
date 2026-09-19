"""Table-fixed landmark extraction.

The work table is a white board carrying yellow tape rectangles, black marks and
screw heads.  Everything that moves during a teardown -- the chassis, hands,
tools, removed parts -- is *not* white table and *not* yellow tape, so a mask of
"white or yellow, holes filled, then shrunk away from anything else" keeps only
pixels that belong to the table.  Features are detected inside that mask, which
is what makes the estimate table-fixed rather than chassis-fixed.
"""
from __future__ import annotations

import cv2
import numpy as np

# Per-view HSV gates.  The scanner's tape reads tan rather than saturated yellow
# and its table is blown out, so it gets its own thresholds.
GATES = {
    # view: (yellow_lo, yellow_hi, white_smax, white_vmin)
    "oak1": ((16, 60, 90), (40, 255, 255), 70, 150),
    "oak2": ((16, 60, 90), (40, 255, 255), 70, 150),
    "rs":   ((16, 60, 90), (40, 255, 255), 70, 150),
    "scan": ((10, 35, 90), (40, 255, 255), 45, 140),
}


def _fill_small_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    """Fill background holes (black dots, screws, scratches) inside the table."""
    h, w = mask.shape
    inv = (mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    out = mask.copy()
    lim = max_frac * h * w
    # Components touching the border are outside-the-table background, not holes.
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area > lim:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        out[lab == i] = 255
    return out


def table_mask(bgr: np.ndarray, view: str):
    """Return (table_safe, yellow) uint8 masks for one frame."""
    ylo, yhi, smax, vmin = GATES.get(view, GATES["oak1"])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, ylo, yhi)
    white = cv2.inRange(hsv, (0, 0, vmin), (180, smax, 255))
    core = cv2.bitwise_or(yellow, white)
    k3 = np.ones((3, 3), np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, k3)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    core = _fill_small_holes(core, 0.01)

    # Keep only large components that reach the image border.  The table always
    # runs off the edge of the frame in all four views, whereas a bright patch
    # on the hardware itself -- a PSU label, bare optical-drive metal -- is an
    # island enclosed by the chassis.  Without this test those islands join the
    # mask and their keypoints move with the chassis.
    h, w = core.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    keep = np.zeros_like(core)
    lim = 0.015 * h * w
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < lim:
            continue
        if not (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
            continue
        keep[lab == i] = 255

    # Pull back from everything that is not table, so that the chassis outline,
    # hands and removed parts contribute no keypoints.
    nottable = cv2.bitwise_not(keep)
    nottable = cv2.dilate(nottable, np.ones((9, 9), np.uint8), iterations=1)
    safe = cv2.bitwise_and(keep, cv2.bitwise_not(nottable))
    # Give the tape edges back: tape is table, so re-add yellow that survived.
    safe = cv2.bitwise_or(safe, cv2.bitwise_and(keep, yellow))
    return safe, yellow


def detect(bgr: np.ndarray, view: str, nfeat: int = 1200):
    """ORB keypoints/descriptors restricted to table-fixed pixels."""
    safe, yellow = table_mask(bgr, view)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    orb = cv2.ORB_create(nfeatures=nfeat, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=15, fastThreshold=7)
    kp, des = orb.detectAndCompute(gray, safe)
    return kp, des, safe, yellow
