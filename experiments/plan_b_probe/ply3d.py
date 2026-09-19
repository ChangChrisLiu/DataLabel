"""Recover a pixel -> 3D map for the OAK captures, without given intrinsics.

The .ply is not organized, but its vertex count equals the number of non-zero
depth pixels: Open3D dropped the invalid ones and kept row-major order.  That
makes the i-th vertex the i-th non-zero pixel of the 800x1280 depth map, which
is enough to (a) look up 3D for any pixel and (b) fit the pinhole intrinsics and
check that they explain the cloud.
"""
from __future__ import annotations

import os

import numpy as np

PLY_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                      ("r", "u1"), ("g", "u1"), ("b", "u1")])


def read_ply(path):
    with open(path, "rb") as f:
        buf = f.read()
    i = buf.find(b"end_header")
    j = buf.find(b"\n", i) + 1
    head = buf[:i].decode("ascii", "replace")
    n = None
    for line in head.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
    arr = np.frombuffer(buf[j:j + n * PLY_DTYPE.itemsize], dtype=PLY_DTYPE, count=n)
    return arr


def capture_files(folder, cam):
    out = {}
    for f in os.listdir(folder):
        if f.endswith("_pointcloud.ply"):
            out["ply"] = os.path.join(folder, f)
        elif f.endswith("_depth_raw.npy"):
            out["depth"] = os.path.join(folder, f)
        elif f.endswith("_rgb_aligned.png"):
            out["aligned"] = os.path.join(folder, f)
        elif f.endswith("_rgb_12mp.jpg"):
            out["jpg"] = os.path.join(folder, f)
    return out


def pixel_to_xyz(depth: np.ndarray, ply: np.ndarray):
    """Dense HxWx3 array of cloud points, NaN where depth is invalid."""
    h, w = depth.shape
    mask = depth.reshape(-1) != 0
    if int(mask.sum()) != len(ply):
        return None
    out = np.full((h * w, 3), np.nan, np.float64)
    out[mask, 0] = ply["x"]
    out[mask, 1] = ply["y"]
    out[mask, 2] = ply["z"]
    return out.reshape(h, w, 3)


def fit_intrinsics(xyz: np.ndarray):
    """Least-squares pinhole fit: u = fx*x/z + cx, v = fy*y/z + cy."""
    h, w = xyz.shape[:2]
    vv, uu = np.mgrid[0:h, 0:w]
    ok = np.isfinite(xyz[:, :, 2]) & (np.abs(xyz[:, :, 2]) > 1e-9)
    x, y, z = xyz[:, :, 0][ok], xyz[:, :, 1][ok], xyz[:, :, 2][ok]
    u, v = uu[ok].astype(np.float64), vv[ok].astype(np.float64)
    a = np.stack([x / z, np.ones_like(z)], 1)
    fx, cx = np.linalg.lstsq(a, u, rcond=None)[0]
    b = np.stack([y / z, np.ones_like(z)], 1)
    fy, cy = np.linalg.lstsq(b, v, rcond=None)[0]
    ru = np.abs(fx * x / z + cx - u)
    rv = np.abs(fy * y / z + cy - v)
    return dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                res_u=float(np.median(ru)), res_v=float(np.median(rv)),
                n=int(ok.sum()))


def backproject(u, v, z, K):
    """Sub-pixel back-projection using the fitted intrinsics."""
    x = (u - K["cx"]) * z / K["fx"]
    y = (v - K["cy"]) * z / K["fy"]
    return np.stack([x, y, z], -1)


def kabsch(p, q):
    """Rigid transform mapping p onto q (both N x 3).  Returns R, t, rmse."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = qc - r @ pc
    rms = float(np.sqrt((((p @ r.T + t) - q) ** 2).sum(1).mean()))
    return r, t, rms
