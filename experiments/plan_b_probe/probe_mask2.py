"""Check the tightened mask on the frames that produced false positives."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import tablemask as T     # noqa: E402
import validate as V      # noqa: E402

CASES = [("scan", 36, 9), ("scan", 36, 10), ("oak1", 24, 8), ("oak1", 24, 9),
         ("scan", 55, 21), ("scan", 55, 22), ("oak1", 2, 5), ("oak1", 2, 6),
         ("rs", 24, 8), ("oak2", 30, 19)]

if __name__ == "__main__":
    for view, d, s in CASES:
        fs = V.frames(view)
        f = fs.get((d, s))
        if not f:
            print(view, d, s, "missing")
            continue
        img = C.imread(f.image_path(), 640)
        safe, tape = T.table_mask(img, view)
        print("%-5s d%02d s%02d  safe=%.3f tape=%.4f" %
              (view, d, s, safe.mean() / 255, tape.mean() / 255))
        vis = img.copy()
        vis[safe > 0] = (0.55 * vis[safe > 0] +
                         0.45 * np.array([0, 255, 0])).astype(np.uint8)
        vis[tape > 0] = (0.4 * vis[tape > 0] +
                         0.6 * np.array([255, 0, 255])).astype(np.uint8)
        C.save_preview(vis, "mask3_%s_d%02d_s%02d.jpg" % (view, d, s), 700)
