"""Objective per-event check, independent of the keypoint matcher.

A flag says "the table landmarks moved by S".  That is testable directly on the
pixels: warp frame A's tape mask by S and see whether it lines up with frame B's
tape better than it did before.

  iou_before  -- tape overlap with no alignment
  iou_after   -- tape overlap after applying the estimated transform

A real move has iou_before clearly below iou_after.  A spurious flag has both
high and nearly equal: there was nothing to correct.  This uses the tape shape
only, so it shares no failure mode with the ORB matching that raised the flag.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import tablemask as T     # noqa: E402
import estimate as E      # noqa: E402
from extract import Store, WORK, ORIG   # noqa: E402

MIN_TAPE_PX = 400


def _tape(path, view):
    img = C.imread(path, WORK)
    if img is None:
        return None, None
    _safe, y = T.table_mask(img, view)
    return y, img


def _iou(a, b):
    u = np.logical_or(a > 0, b > 0).sum()
    return float(np.logical_and(a > 0, b > 0).sum()) / float(u) if u else float("nan")


def check(path_a, path_b, view, S_work):
    """S_work: 2x3 similarity at the WORK scale mapping A onto B."""
    ya, _ia = _tape(path_a, view)
    yb, _ib = _tape(path_b, view)
    out = {"iou_before": None, "iou_after": None, "tape_px_a": 0, "tape_px_b": 0,
           "usable": False}
    if ya is None or yb is None:
        return out
    out["tape_px_a"] = int((ya > 0).sum())
    out["tape_px_b"] = int((yb > 0).sum())
    if out["tape_px_a"] < MIN_TAPE_PX or out["tape_px_b"] < MIN_TAPE_PX:
        return out
    h, w = ya.shape
    warped = cv2.warpAffine(ya, np.asarray(S_work, np.float32), (w, h),
                            flags=cv2.INTER_NEAREST)
    # Only judge where the warp has support, so the frame edge does not count.
    sup = cv2.warpAffine(np.full_like(ya, 255), np.asarray(S_work, np.float32),
                         (w, h), flags=cv2.INTER_NEAREST)
    m = sup > 0
    out["iou_before"] = _iou(ya[m], yb[m])
    out["iou_after"] = _iou(warped[m], yb[m])
    out["usable"] = True
    return out


def transform_for(view, desktop, step_a, step_b, desktop_b=None):
    """Re-derive the fitted similarity (at WORK scale) for a pair of frames.

    `desktop_b` lets the pair straddle two desktops, for the Q1 boundaries.
    """
    st = Store(view, desktop)
    stb = st if desktop_b in (None, desktop) else Store(view, desktop_b)
    ka, da = st.by_step(step_a)
    kb, db = stb.by_step(step_b)
    if ka is None or kb is None:
        return None
    good = E.match(da, db)
    if len(good) < E.MIN_MATCHES:
        return None
    pa = np.float32([ka[m.queryIdx] for m in good])
    pb = np.float32([kb[m.trainIdx] for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                         ransacReprojThreshold=E.RANSAC_PX,
                                         maxIters=4000, confidence=0.995)
    if M is None or inl is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < E.MIN_INLIERS:
        return None
    return E.umeyama(pa[inl], pb[inl])
