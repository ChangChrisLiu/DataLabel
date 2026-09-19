"""Q3 feasibility run: dictionary sweep on a sample of calibration captures."""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B  # noqa: E402

if __name__ == "__main__":
    ds = B.oak_calib_desktops()
    sample = ds[::max(1, len(ds) // 10)][:10]
    print("calibration desktops:", len(ds), "sample:", sample)
    print()
    # Step 1: which dictionary, on the easiest possible image (12 MP, cam 1).
    jpg, aligned = B.oak_calib_capture(sample[0], 1)
    img = cv2.imread(jpg)
    print("dictionary sweep on 12MP cam1 d%d %s" % (sample[0], img.shape))
    for name, (n, ids, _c) in sorted(B.detect_all(img).items(), key=lambda kv: -kv[1][0]):
        if n:
            print("   %-22s %3d tags  ids=%s" % (name, n, sorted(ids)[:12]))
    print()
    img2 = cv2.imread(aligned)
    print("dictionary sweep on aligned cam1 d%d %s" % (sample[0], img2.shape))
    for name, (n, ids, _c) in sorted(B.detect_all(img2).items(), key=lambda kv: -kv[1][0]):
        if n:
            print("   %-22s %3d tags  ids=%s" % (name, n, sorted(ids)[:12]))
