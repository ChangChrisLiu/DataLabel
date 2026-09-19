"""Shared helpers for the plan_b_probe camera-move experiment (E1).

Read-only: never opens annotations/tda.sqlite, only the read-only copy under
.cache/tmp/exp_copy.sqlite, and never writes to F:.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

import cv2
import numpy as np

DB_URI = "file:D:/DataSet/.cache/tmp/exp_copy.sqlite?mode=ro"
TMP = r"D:\DataSet\.cache\tmp\e1"
OUT = r"D:\DataSet\experiments_out\plan_b_probe\camera_moves"
OVERLAYS = os.path.join(OUT, "overlays")

VIEWS = ("oak1", "oak2", "rs", "scan")


def connect():
    return sqlite3.connect(DB_URI, uri=True)


@dataclass
class Frame:
    desktop: int
    step: int
    view: str
    path: str
    aux: dict
    ts: str
    missing: int

    def image_path(self) -> str:
        """Prefer the 1280x800 aligned png for OAK (the 12 MP jpg is overkill)."""
        al = self.aux.get("aligned")
        if al:
            return al
        return self.path


def load_frames(view: str, desktops=None) -> list[Frame]:
    con = connect()
    q = "select desktop, step, view, path, aux_json, ts, missing from frame where view=?"
    args = [view]
    if desktops:
        q += " and desktop in (%s)" % ",".join("?" * len(desktops))
        args += list(desktops)
    q += " order by desktop, step"
    rows = con.execute(q, args).fetchall()
    con.close()
    out = []
    for d, s, v, p, aj, ts, miss in rows:
        aux = {}
        if aj:
            try:
                aux = json.loads(aj)
            except Exception:
                aux = {}
        out.append(Frame(d, s, v, p or "", aux, ts or "", miss or 0))
    return out


def imread(path: str, max_side: int = 640):
    """Read an image from F: and downscale so the long side is <= max_side."""
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    sc = max_side / float(max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, (int(round(w * sc)), int(round(h * sc))),
                         interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------- yellow tape

def yellow_mask(bgr) -> np.ndarray:
    """Binary mask of saturated yellow tape on the white work table."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # OpenCV hue is 0..179; yellow sits around 20-35.
    m = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def save_preview(img, name: str, max_side: int = 900) -> str:
    os.makedirs(TMP, exist_ok=True)
    h, w = img.shape[:2]
    sc = max_side / float(max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    p = os.path.join(TMP, name)
    cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return p
