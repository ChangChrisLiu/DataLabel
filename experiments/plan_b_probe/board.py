"""Q3 (time-boxed feasibility): can the fiducial board be detected at all?

The board is a printed 8x5 = 40 tag grid lying in a plastic sleeve on the table.
This sweeps the predefined OpenCV dictionaries over a sample of captures and
reports tags found / 40 per camera, on both the 12 MP jpg and the 1280x800
aligned png.  Nothing downstream depends on the outcome.
"""
from __future__ import annotations

import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OAK = r"F:\PHD Data Backup\Desktop Dataset\OAKD Capture\Desktop_Datacollection\DesktopData"
RS_CAL = r"F:\PHD Data Backup\Desktop Dataset\Realsense Capture\Dataset Information\Exp"
RS_ALL = r"F:\PHD Data Backup\Desktop Dataset\Realsense Capture\Dataset Information\CalibrationALL\Aruco"

DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


def make_params(aggressive: bool):
    p = cv2.aruco.DetectorParameters()
    if aggressive:
        # Small, soft, glare-hit tags: widen the thresholding sweep and relax
        # the border/error gates.
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 43
        p.adaptiveThreshWinSizeStep = 4
        p.minMarkerPerimeterRate = 0.01
        p.maxMarkerPerimeterRate = 4.0
        p.polygonalApproxAccuracyRate = 0.05
        p.minCornerDistanceRate = 0.03
        p.perspectiveRemovePixelPerCell = 8
        p.maxErroneousBitsInBorderRate = 0.5
        p.errorCorrectionRate = 0.8
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def detect_all(img, aggressive=True):
    """Return {dict_name: (n_tags, ids, corners)} for every dictionary."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    out = {}
    params = make_params(aggressive)
    for name in DICTS:
        did = getattr(cv2.aruco, name, None)
        if did is None:
            continue
        d = cv2.aruco.getPredefinedDictionary(did)
        det = cv2.aruco.ArucoDetector(d, params)
        corners, ids, _rej = det.detectMarkers(gray)
        n = 0 if ids is None else len(ids)
        out[name] = (n, None if ids is None else ids.ravel().tolist(), corners)
    return out


def oak_calib_capture(desktop: int, cam: int):
    """(jpg, aligned) paths of the calibration capture, or (None, None).

    Two naming conventions are in use on F:, ``Camera_1``/``Camera_2`` and the
    shorter ``C1``/``C2``.
    """
    base = os.path.join(OAK, "Desktop %d" % desktop, "Calibration")
    root = None
    for cand in ("Camera_%d" % cam, "C%d" % cam):
        p = os.path.join(base, cand)
        if os.path.isdir(p):
            root = p
            break
    if root is None:
        return None, None
    subs = sorted(os.listdir(root))
    if not subs:
        return None, None
    p = os.path.join(root, subs[0])
    jpg = aligned = None
    for f in os.listdir(p):
        if f.endswith("_rgb_12mp.jpg"):
            jpg = os.path.join(p, f)
        elif f.endswith("_rgb_aligned.png"):
            aligned = os.path.join(p, f)
    return jpg, aligned


def oak_calib_desktops():
    out = []
    for name in os.listdir(OAK):
        m = re.fullmatch(r"Desktop (\d+)", name)
        if m and os.path.isdir(os.path.join(OAK, name, "Calibration")):
            out.append(int(m.group(1)))
    return sorted(out)
