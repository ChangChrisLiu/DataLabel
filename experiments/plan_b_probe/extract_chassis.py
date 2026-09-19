"""Stage 1b: cache the chassis blob and its own ORB features per frame.

Q2 needs to tell "the camera moved" from "the chassis was repositioned".  The
table pass (extract.py) deliberately throws the chassis away, so this second
pass keeps the complementary signal: the central non-table blob's centroid,
area and second moments, plus ORB features taken *inside* it.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import tablemask as T       # noqa: E402
from extract import WORK, ORIG   # noqa: E402

NFEAT = 500


def cache_path(view: str, desktop: int) -> str:
    return os.path.join(C.TMP, "chas_%s_%02d.npz" % (view, desktop))


def chassis_blob(bgr, view):
    """Largest non-table component overlapping the middle of the frame."""
    safe, _y = T.table_mask(bgr, view)
    h, w = safe.shape
    # Everything that is not table: chassis, hands, tools, removed parts.
    nt = cv2.bitwise_not(safe)
    nt = cv2.morphologyEx(nt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(nt, 8)
    cx0, cy0 = w * 0.5, h * 0.5
    best, best_area = 0, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.01 * h * w:
            continue
        # must straddle the middle of the frame, where the chassis sits
        if not (x <= cx0 <= x + bw and y <= cy0 <= y + bh):
            continue
        if area > best_area:
            best, best_area = i, area
    if best == 0:
        return None, None
    mask = (lab == best).astype(np.uint8) * 255
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return None, None
    feat = np.array([
        m["m10"] / m["m00"], m["m01"] / m["m00"], m["m00"] / 255.0,
        m["mu20"] / m["m00"], m["mu11"] / m["m00"], m["mu02"] / m["m00"],
    ], np.float64)
    return mask, feat


def _one(args):
    view, desktop = args
    out = cache_path(view, desktop)
    if os.path.exists(out):
        return view, desktop, -1
    frames = C.load_frames(view, [desktop])
    steps, offs, nkp, kps, dess, blobs = [], [], [], [], [], []
    off = 0
    orb = cv2.ORB_create(nfeatures=NFEAT, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    for f in frames:
        steps.append(f.step)
        blob = np.full(6, np.nan)
        n = 0
        if not f.missing and f.path:
            img = C.imread(f.image_path(), WORK)
            if img is not None:
                mask, feat = chassis_blob(img, view)
                if feat is not None:
                    blob = feat
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    er = cv2.erode(mask, np.ones((7, 7), np.uint8))
                    kp, des = orb.detectAndCompute(gray, er)
                    if des is not None and len(kp):
                        n = len(kp)
                        kps.append(np.float32([k.pt for k in kp]))
                        dess.append(des.astype(np.uint8))
        offs.append(off)
        nkp.append(n)
        off += n
        blobs.append(blob)
    np.savez_compressed(
        out,
        steps=np.asarray(steps, np.int32),
        offs=np.asarray(offs, np.int64),
        nkp=np.asarray(nkp, np.int32),
        blob=np.asarray(blobs, np.float64),
        kp=np.concatenate(kps) if kps else np.zeros((0, 2), np.float32),
        des=np.concatenate(dess) if dess else np.zeros((0, 32), np.uint8),
    )
    return view, desktop, int(sum(nkp))


class ChStore:
    def __init__(self, view, desktop):
        z = np.load(cache_path(view, desktop))
        self.steps = z["steps"]
        self.offs = z["offs"]
        self.nkp = z["nkp"]
        self.blob = z["blob"]
        self.kp = z["kp"]
        self.des = z["des"]
        self.upscale = ORIG[view] / float(WORK)

    def at(self, i):
        n = int(self.nkp[i])
        if n == 0:
            return None, None
        o = int(self.offs[i])
        return self.kp[o:o + n], self.des[o:o + n]


def main():
    views = sys.argv[1:] or list(C.VIEWS)
    os.makedirs(C.TMP, exist_ok=True)
    jobs = [(v, d) for v in views for d in range(1, 67)]
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        for view, desktop, n in ex.map(_one, jobs):
            done += 1
            if done % 20 == 0:
                print("[%5.1fs] %3d/%3d %s d%02d kp=%s" %
                      (time.time() - t0, done, len(jobs), view, desktop, n), flush=True)
    print("done in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
