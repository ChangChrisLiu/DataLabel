"""Robust table-to-table transform between two frames of the same view."""
from __future__ import annotations

import cv2
import numpy as np

MIN_MATCHES = 20
RANSAC_PX = 3.0          # at the 640-px working scale
MIN_INLIERS = 20
MIN_RATIO = 0.40         # inliers / matches; a low ratio means RANSAC guessed
MIN_SPREAD = 40.0        # px std of inlier positions; clustered points are not a rig


def match(des_a, des_b, ratio: float = 0.78):
    if des_a is None or des_b is None or len(des_a) < 8 or len(des_b) < 8:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des_a, des_b, k=2)
    good = []
    for m in knn:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])
    return good


def estimate(kp_a, des_a, kp_b, des_b, upscale: float = 1.0):
    """Map frame A onto frame B using table-fixed points.

    Returns a dict with the median inlier displacement in ORIGINAL-image pixels
    (`px`), the similarity rotation/scale, and the inlier bookkeeping.  `ok` is
    False when there is not enough table evidence -- callers must report those
    as undetermined rather than as a number.
    """
    out = {"ok": False, "n_match": 0, "n_inlier": 0, "px": None,
           "rot_deg": None, "scale": None, "spread": None, "reason": ""}
    good = match(des_a, des_b)
    out["n_match"] = len(good)
    if len(good) < MIN_MATCHES:
        out["reason"] = "few_matches"
        return out
    pa = np.float32([kp_a[m.queryIdx] for m in good])
    pb = np.float32([kp_b[m.trainIdx] for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_PX,
                                         maxIters=4000, confidence=0.995)
    if M is None or inl is None:
        out["reason"] = "no_model"
        return out
    inl = inl.ravel().astype(bool)
    out["n_inlier"] = int(inl.sum())
    if out["n_inlier"] < MIN_INLIERS:
        out["reason"] = "few_inliers"
        return out
    if out["n_inlier"] < MIN_RATIO * len(good):
        out["reason"] = "low_inlier_ratio"
        return out
    ia, ib = pa[inl], pb[inl]
    spread = float(np.sqrt(ia[:, 0].var() + ia[:, 1].var()))
    out["spread"] = spread * upscale
    if spread < MIN_SPREAD:
        out["reason"] = "clustered"
        return out
    # Model-agnostic magnitude: how far the table landmarks actually moved.
    d = np.linalg.norm(ib - ia, axis=1)
    out["px"] = float(np.median(d)) * upscale
    out["rot_deg"] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    out["scale"] = float(np.hypot(M[0, 0], M[1, 0]))
    out["ok"] = True
    return out
