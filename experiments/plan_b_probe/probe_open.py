"""How big must the bridge-breaking opening be to drop hardware islands?"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import tablemask as T     # noqa: E402
import validate as V      # noqa: E402

if __name__ == "__main__":
    view, d, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    f = V.frames(view)[(d, s)]
    img = C.imread(f.image_path(), 640)
    ylo, yhi, smax, vmin = T.GATES[view]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yraw = cv2.morphologyEx(cv2.inRange(hsv, ylo, yhi), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))
    tape = T.tape_strips(yraw)
    white = cv2.inRange(hsv, (0, 0, vmin), (180, smax, 255))
    base = cv2.bitwise_or(tape, white)
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    base = T._fill_small_holes(base, 0.01)
    h, w = base.shape
    for k in (11, 15, 19, 23, 27):
        core = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
        keep = np.zeros_like(core)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.015 * h * w:
                continue
            if not (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
                continue
            keep[lab == i] = 255
        tape_kept = cv2.bitwise_and(keep, tape).mean() / 255
        print("k=%2d  kept=%.3f  tape_kept=%.4f (tape total %.4f)"
              % (k, keep.mean() / 255, tape_kept, tape.mean() / 255))
        vis = img.copy()
        vis[keep > 0] = (0.55 * vis[keep > 0] +
                         0.45 * np.array([0, 255, 0])).astype(np.uint8)
        C.save_preview(vis, "open_%s_d%02d_s%02d_k%02d.jpg" % (view, d, s, k), 620)
