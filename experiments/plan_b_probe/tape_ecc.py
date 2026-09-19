"""Last-resort registration: align the yellow tape pattern itself.

The white table carries almost no texture, and the tape is made of long straight
strips, which is the worst case for a keypoint descriptor: nothing distinctive
to latch onto.  When both ORB and SIFT run out of matches the tape can still be
registered as a *shape*, by maximising the correlation of the blurred tape masks
(ECC) from a coarse translation/scale start.

The accept test is geometric, not photometric: the warped tape mask has to
actually overlap the other frame's tape (IoU), so a confident-looking but wrong
alignment is rejected.
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

WORK = 480
MIN_CC = 0.45
MIN_IOU = 0.45
MIN_TAPE_FRAC = 0.010     # tape must cover at least 1 % of the frame


def tape(path, view):
    img = C.imread(path, WORK)
    if img is None:
        return None, None
    _safe, y = T.table_mask(img, view)
    if y.mean() / 255.0 < MIN_TAPE_FRAC:
        return None, img
    return y, img


def _iou(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter) / float(union) if union else 0.0


def estimate_tape(path_a, path_b, view, orig_w):
    out = {"ok": False, "px": None, "rot_deg": None, "scale": None,
           "cc": None, "iou": None, "iou0": None, "reason": ""}
    ya, ia = tape(path_a, view)
    yb, ib = tape(path_b, view)
    if ya is None or yb is None:
        out["reason"] = "no_tape"
        return out
    fa = cv2.GaussianBlur(ya.astype(np.float32) / 255.0, (0, 0), 5)
    fb = cv2.GaussianBlur(yb.astype(np.float32) / 255.0, (0, 0), 5)
    out["iou0"] = _iou(ya, yb)

    h, w = fa.shape
    best = None
    (dx, dy), _resp = cv2.phaseCorrelate(fa.astype(np.float64), fb.astype(np.float64))
    starts = [(1.0, dx, dy), (1.0, 0.0, 0.0)]
    for s in (0.85, 0.92, 1.08, 1.18):
        starts.append((s, dx, dy))
    for s, tx, ty in starts:
        W = np.array([[s, 0, tx], [0, s, ty]], np.float32)
        try:
            cc, W = cv2.findTransformECC(
                fa, fb, W, cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                None, 5)
        except cv2.error:
            continue
        warped = cv2.warpAffine(ya, W, (w, h), flags=cv2.INTER_NEAREST)
        iou = _iou(warped, yb)
        if best is None or iou > best[1]:
            best = (W, iou, float(cc))
    if best is None:
        out["reason"] = "ecc_failed"
        return out
    W, iou, cc = best
    out["cc"], out["iou"] = cc, iou
    if cc < MIN_CC or iou < MIN_IOU:
        out["reason"] = "poor_alignment"
        return out
    sc = float(np.hypot(W[0, 0], W[1, 0]))
    if not (0.5 <= sc <= 2.0):
        out["reason"] = "degenerate_scale"
        return out
    up = orig_w / float(WORK)
    out["px"] = E.grid_displacement(np.asarray(W, np.float64), w, h) * up
    out["rot_deg"] = float(np.degrees(np.arctan2(W[1, 0], W[0, 0])))
    out["scale"] = sc
    out["ok"] = True
    return out
