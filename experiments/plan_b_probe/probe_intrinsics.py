"""Verify the ply<->depth pixel mapping and recover the OAK intrinsics."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B     # noqa: E402
import ply3d as P     # noqa: E402

if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for cam in (1, 2):
        jpg, _al = B.oak_calib_capture(d, cam)
        if not jpg:
            continue
        files = P.capture_files(os.path.dirname(jpg), cam)
        depth = np.load(files["depth"])
        ply = P.read_ply(files["ply"])
        xyz = P.pixel_to_xyz(depth, ply)
        print("=== d%d cam%d  depth%s  ply n=%d  mapping=%s" % (
            d, cam, depth.shape, len(ply), "OK" if xyz is not None else "MISMATCH"))
        if xyz is None:
            continue
        # Colour check: the ply colour at a pixel must equal the aligned png there.
        rgb = cv2.cvtColor(cv2.imread(files["aligned"]), cv2.COLOR_BGR2RGB)
        mask = depth.reshape(-1) != 0
        px = rgb.reshape(-1, 3)[mask]
        pc = np.stack([ply["r"], ply["g"], ply["b"]], 1)
        agree = float((np.abs(px.astype(int) - pc.astype(int)).max(1) <= 2).mean())
        print("   ply colour == aligned png colour at mapped pixel: %.4f" % agree)
        # Depth-scale check and intrinsics fit.
        z = xyz[:, :, 2]
        dz = depth.astype(np.float64)
        ok = np.isfinite(z) & (dz > 0)
        ratio = np.median(dz[ok] / z[ok])
        print("   depth_npy / ply_z median ratio = %.4f (=> ply z in %s)" %
              (ratio, "metres" if 900 < ratio < 1100 else "?"))
        K = P.fit_intrinsics(xyz)
        print("   intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f  residual %.3f/%.3f px  n=%d"
              % (K["fx"], K["fy"], K["cx"], K["cy"], K["res_u"], K["res_v"], K["n"]))
