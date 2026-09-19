"""Q3: is the OAK point cloud organized (one point per aligned-rgb pixel)?

If it is, a tag corner pixel maps straight to a 3D point and the cam1->cam2
rigid transform can be solved by Kabsch without knowing the intrinsics.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B  # noqa: E402


def ply_header(path, nbytes=600):
    with open(path, "rb") as f:
        head = f.read(nbytes)
    return head.split(b"end_header")[0].decode("ascii", "replace")


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for cam in (1, 2):
        jpg, aligned = B.oak_calib_capture(d, cam)
        if not jpg:
            print("cam%d: no capture" % cam)
            continue
        folder = os.path.dirname(jpg)
        ply = depth = None
        for f in os.listdir(folder):
            if f.endswith("_pointcloud.ply"):
                ply = os.path.join(folder, f)
            if f.endswith("_depth_raw.npy"):
                depth = os.path.join(folder, f)
        print("=== desktop %d camera %d" % (d, cam))
        if depth:
            a = np.load(depth)
            print("  depth npy:", a.shape, a.dtype, "min", a.min(), "max", a.max(),
                  "nonzero %.3f" % (np.count_nonzero(a) / a.size))
        if ply:
            print("  ply size %.1f MB" % (os.path.getsize(ply) / 1e6))
            print("  header:")
            for line in ply_header(ply).splitlines():
                print("     ", line)
            print("  1280*800 =", 1280 * 800)
