"""Q3: cam1 <-> cam2 relation from the AprilTag board, per calibration capture.

Two independent estimates per desktop:

* a plane-induced homography cam1 -> cam2 from the 2D tag corners.  For a planar
  scene H depends on the relative camera pose and on the board plane, not on
  where the board was put down, so a rigid rig gives a constant H.
* a full rigid transform, by giving every corner a 3D position.  Depth is noisy
  per pixel, so instead of sampling it at the corner the board plane is fitted
  to the cloud once and each corner's ray is intersected with that plane.

Intrinsics are not supplied with the dataset; they are recovered from the
depth/ply pair (see ply3d.py), which fits a pinhole model to 0.000 px.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B     # noqa: E402
import ply3d as P     # noqa: E402

DICT = "DICT_APRILTAG_36h11"
PLANE_RESID_MAX_MM = 10.0


def detect_corners(gray):
    """{tag_id: 4x2 subpixel corners} using AprilTag 36h11."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
    det = cv2.aruco.ArucoDetector(d, B.make_params(True))
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64)
            for i, c in zip(ids.ravel(), corners)}


def fit_plane(pts):
    """Least-squares plane through Nx3 points -> (unit normal, d) with n.x + d = 0."""
    c = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, float(-n @ c)


def board_plane(xyz, corners):
    """Fit the board plane to cloud points inside the tag hull."""
    if not corners:
        return None
    allc = np.concatenate(list(corners.values()))
    hull = cv2.convexHull(allc.astype(np.float32)).reshape(-1, 2)
    h, w = xyz.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, hull.astype(np.int32), 255)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    sel = (m > 0) & np.isfinite(xyz[:, :, 2])
    pts = xyz[sel]
    if len(pts) < 500:
        return None
    n, d = fit_plane(pts)
    # one robust pass: drop points far from the first fit
    r = np.abs(pts @ n + d)
    keep = r < max(3 * np.median(r), 1e-4)
    if keep.sum() > 500:
        n, d = fit_plane(pts[keep])
    resid = float(np.median(np.abs(pts[keep] @ n + d)))
    return n, d, resid, int(keep.sum())


def rays_to_plane(uv, K, n, d):
    """Intersect the pixel rays with the board plane -> Nx3 points."""
    # Direction in camera coords consistent with the fitted (possibly negative fx).
    dirs = np.stack([(uv[:, 0] - K["cx"]) / K["fx"],
                     (uv[:, 1] - K["cy"]) / K["fy"],
                     np.ones(len(uv))], 1)
    denom = dirs @ n
    bad = np.abs(denom) < 1e-12
    t = np.where(bad, np.nan, -d / np.where(bad, 1.0, denom))
    return dirs * t[:, None]


def capture_data(desktop, cam):
    jpg, aligned = B.oak_calib_capture(desktop, cam)
    if not jpg:
        return None
    files = P.capture_files(os.path.dirname(jpg), cam)
    if not {"depth", "ply", "aligned"} <= set(files):
        return None
    gray = cv2.imread(files["aligned"], cv2.IMREAD_GRAYSCALE)
    corners = detect_corners(gray)
    depth = np.load(files["depth"])
    ply = P.read_ply(files["ply"])
    xyz = P.pixel_to_xyz(depth, ply)
    if xyz is None:
        return None
    K = P.fit_intrinsics(xyz)
    return {"corners": corners, "xyz": xyz, "K": K, "files": files,
            "n_tags_aligned": len(corners),
            "n_tags_12mp": len(detect_corners(cv2.imread(files["jpg"],
                                                         cv2.IMREAD_GRAYSCALE)))}


def pair(desktop):
    """Everything for one desktop's cam1/cam2 calibration capture."""
    a = capture_data(desktop, 1)
    b = capture_data(desktop, 2)
    if a is None or b is None:
        return None
    common = sorted(set(a["corners"]) & set(b["corners"]))
    out = {"desktop": desktop,
           "n_tags_c1": a["n_tags_aligned"], "n_tags_c2": b["n_tags_aligned"],
           "n_tags_c1_12mp": a["n_tags_12mp"], "n_tags_c2_12mp": b["n_tags_12mp"],
           "n_common": len(common), "H": None, "R": None, "t": None,
           "kabsch_rmse_mm": None, "plane_resid_mm_c1": None,
           "plane_resid_mm_c2": None, "H_reproj_px": None,
           "plane_rejected": False}
    if len(common) < 6:
        return out
    ua = np.concatenate([a["corners"][i] for i in common])
    ub = np.concatenate([b["corners"][i] for i in common])
    H, inl = cv2.findHomography(ua, ub, cv2.RANSAC, 3.0)
    if H is not None:
        proj = cv2.perspectiveTransform(ua.reshape(-1, 1, 2), H).reshape(-1, 2)
        out["H"] = H
        out["H_reproj_px"] = float(np.median(np.linalg.norm(proj - ub, axis=1)))
    pa = board_plane(a["xyz"], {i: a["corners"][i] for i in common})
    pb = board_plane(b["xyz"], {i: b["corners"][i] for i in common})
    if pa and pb:
        out["plane_resid_mm_c1"] = pa[2] * 1000.0
        out["plane_resid_mm_c2"] = pb[2] * 1000.0
        # The board sits in a glossy plastic sleeve; where glare kills the depth
        # the plane fit is meaningless and so is anything built on it.
        if max(pa[2], pb[2]) * 1000.0 <= PLANE_RESID_MAX_MM:
            qa = rays_to_plane(ua, a["K"], pa[0], pa[1])
            qb = rays_to_plane(ub, b["K"], pb[0], pb[1])
            ok = np.isfinite(qa).all(1) & np.isfinite(qb).all(1)
            if ok.sum() >= 6:
                R, t, rms = P.kabsch(qa[ok], qb[ok])
                out["R"], out["t"] = R, t
                out["kabsch_rmse_mm"] = rms * 1000.0
        else:
            out["plane_rejected"] = True
    return out


def h_disagreement(H1, H2, w=1280, h=800):
    """Median pixel disagreement of two homographies over the image grid."""
    if H1 is None or H2 is None:
        return None
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 20), np.linspace(0, h - 1, 14))
    pts = np.stack([gx.ravel(), gy.ravel()], 1).reshape(-1, 1, 2).astype(np.float64)
    a = cv2.perspectiveTransform(pts, H1).reshape(-1, 2)
    b = cv2.perspectiveTransform(pts, H2).reshape(-1, 2)
    return float(np.median(np.linalg.norm(a - b, axis=1)))


def rt_disagreement(R1, t1, R2, t2):
    if R1 is None or R2 is None:
        return None, None
    dR = R1 @ R2.T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    dt = float(np.linalg.norm(np.asarray(t1) - np.asarray(t2)) * 1000.0)
    return ang, dt
