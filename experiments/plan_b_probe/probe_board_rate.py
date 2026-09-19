"""Q3 feasibility: AprilTag 36h11 detection rate per capture, both cameras."""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B  # noqa: E402

DICT = "DICT_APRILTAG_36h11"


def n_tags(path, aggressive=True):
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
    det = cv2.aruco.ArucoDetector(d, B.make_params(aggressive))
    corners, ids, _ = det.detectMarkers(img)
    return 0 if ids is None else len(ids)


if __name__ == "__main__":
    ds = B.oak_calib_desktops()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sample = ds[:: max(1, len(ds) // n)][:n] if n < len(ds) else ds
    print("desktop  cam1_12mp cam1_align  cam2_12mp cam2_align   (of 40)")
    tot = {"c1j": [], "c1a": [], "c2j": [], "c2a": []}
    for d in sample:
        j1, a1 = B.oak_calib_capture(d, 1)
        j2, a2 = B.oak_calib_capture(d, 2)
        r = [n_tags(j1), n_tags(a1), n_tags(j2), n_tags(a2)]
        for k, v in zip(("c1j", "c1a", "c2j", "c2a"), r):
            if v is not None:
                tot[k].append(v)
        print("  %3d      %s" % (d, "   ".join("%9s" % x for x in r)))
    print()
    for k in ("c1j", "c1a", "c2j", "c2a"):
        v = tot[k]
        if v:
            print("%s: n=%d mean=%.1f/40 min=%d max=%d  >=30 in %d/%d" %
                  (k, len(v), sum(v) / len(v), min(v), max(v),
                   sum(1 for x in v if x >= 30), len(v)))
