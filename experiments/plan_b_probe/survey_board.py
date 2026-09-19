"""Q3 survey: where does a fiducial-board capture exist on F:?  (read-only)"""
import os
import re
import sys

OAK = r"F:\PHD Data Backup\Desktop Dataset\OAKD Capture\Desktop_Datacollection\DesktopData"
RS = r"F:\PHD Data Backup\Desktop Dataset\Realsense Capture\Dataset Information"
SCAN = r"F:\PHD Data Backup\Desktop Dataset\UGA DATA"
KEYS = ("calib", "tag", "board", "aruco", "april", "chess", "charuco", "marker")


def oak_calib():
    rows = []
    for name in sorted(os.listdir(OAK)):
        m = re.fullmatch(r"Desktop (\d+)", name)
        if not m:
            continue
        d = int(m.group(1))
        cal = os.path.join(OAK, name, "Calibration")
        if not os.path.isdir(cal):
            rows.append((d, 0, 0))
            continue
        c = []
        for cam in ("Camera_1", "Camera_2"):
            p = os.path.join(cal, cam)
            c.append(len(os.listdir(p)) if os.path.isdir(p) else 0)
        rows.append((d, c[0], c[1]))
    return rows


def walk_keys(root, maxdepth=4):
    """Directory names anywhere under root that look like calibration data."""
    hits = []
    root = os.path.abspath(root)
    base_depth = root.rstrip("\\").count("\\")
    for cur, dirs, files in os.walk(root):
        if cur.count("\\") - base_depth > maxdepth:
            dirs[:] = []
            continue
        for d in list(dirs):
            if any(k in d.lower() for k in KEYS):
                hits.append(os.path.join(cur, d))
        for f in files:
            if any(k in f.lower() for k in KEYS):
                hits.append(os.path.join(cur, f))
    return hits


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "oak"
    if what == "oak":
        rows = oak_calib()
        have = [r for r in rows if r[1] or r[2]]
        print("OAK desktops with Calibration:", len(have), "/", len(rows))
        print("missing:", [r[0] for r in rows if not (r[1] or r[2])])
        print("captures per desktop (d, n_cam1, n_cam2):")
        for r in have:
            print("  ", r)
    elif what == "rs":
        for h in walk_keys(RS, 3):
            print(h)
    elif what == "scan":
        for h in walk_keys(SCAN, 3):
            print(h)
