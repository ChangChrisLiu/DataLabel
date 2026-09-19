"""Fallback estimator for the cases ORB cannot resolve.

A camera move large enough to change the framing also changes scale and
appearance, which is exactly where binary ORB descriptors give up -- so the
failures are biased towards the events that matter most.  SIFT on the same
table-fixed mask recovers them.  This runs from the images rather than the
feature cache, and only on the handful of undetermined pairs.
"""
from __future__ import annotations

import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import tablemask as T     # noqa: E402
import estimate as E      # noqa: E402

WORK = 900
MIN_INLIERS = 15


def _sift_feats(path, view):
    img = C.imread(path, WORK)
    if img is None:
        return None, None, None
    safe, _y = T.table_mask(img, view)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=3000)
    kp, des = sift.detectAndCompute(gray, safe)
    if des is None or len(kp) < 8:
        return None, None, img
    return np.float32([k.pt for k in kp]), des, img


def estimate_big(path_a, path_b, view, orig_w):
    """Similarity between two frames using SIFT on table-fixed pixels."""
    out = {"ok": False, "px": None, "rot_deg": None, "scale": None,
           "n_inlier": 0, "n_match": 0, "reason": ""}
    ka, da, ia = _sift_feats(path_a, view)
    kb, db, ib = _sift_feats(path_b, view)
    if ka is None or kb is None:
        out["reason"] = "no_table_features"
        return out
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m[0] for m in bf.knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    out["n_match"] = len(good)
    if len(good) < 12:
        out["reason"] = "few_matches"
        return out
    pa = np.float32([ka[m.queryIdx] for m in good])
    pb = np.float32([kb[m.trainIdx] for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                         ransacReprojThreshold=4.0,
                                         maxIters=8000, confidence=0.999)
    if M is None or inl is None:
        out["reason"] = "no_model"
        return out
    inl = inl.ravel().astype(bool)
    out["n_inlier"] = int(inl.sum())
    if out["n_inlier"] < MIN_INLIERS:
        out["reason"] = "few_inliers"
        return out
    ia_, ib_ = pa[inl], pb[inl]
    if float(np.sqrt(ia_[:, 0].var() + ia_[:, 1].var())) < 50.0:
        out["reason"] = "clustered"
        return out
    S = E.umeyama(ia_, ib_)
    up = orig_w / float(WORK)
    h, w = ia.shape[:2]
    sc = float(np.hypot(S[0, 0], S[1, 0]))
    # A similarity that collapses or explodes the frame is a degenerate fit on
    # near-collinear points, not a camera move.
    if not (0.5 <= sc <= 2.0):
        out["reason"] = "degenerate_scale"
        return out
    out["px"] = E.grid_displacement(S, w, h) * up
    out["rot_deg"] = float(np.degrees(np.arctan2(S[1, 0], S[0, 0])))
    out["scale"] = sc
    out["ok"] = True
    return out
