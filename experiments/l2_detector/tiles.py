"""Tiling: how a frame becomes detector input, for training and inference alike.

A frame is cropped to its view's ROI (the drafts' union box padded 10 %, as in
L1), brought to the view's **work scale** and cut into 640 x 640 tiles with a
stride of 512 (128 px overlap, more than the largest small part), the last tile
in each direction flush with the crop's edge; a crop smaller than a tile is
padded with grey (114) rather than resized, so nothing is ever scaled by the
detector's own letterbox.

Work scale: 1.0 (native) for ``scan``, ``oak1`` and ``oak2`` -- a 15-20 px
scanner screw stays 15-20 px, an OAK1 screw stays ~38 px.  ``rs`` is 2.0: its
screws are 6-10 px at native resolution, below what a stride-8 head can place;
at 2x they are 12-20 px like the scanner's.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TILE = 640
STRIDE = 512
PAD_VALUE = 114
WORK_SCALE = {"scan": 1.0, "oak1": 1.0, "oak2": 1.0, "rs": 2.0}


@dataclass
class Tile:
    """One tile: its pixels and where it sits in the full frame."""

    rgb: np.ndarray            # TILE x TILE x 3
    ox: int                    # tile origin in work-scale crop coordinates
    oy: int
    w: int                     # valid (unpadded) width / height
    h: int


def _starts(length: int) -> list[int]:
    if length <= TILE:
        return [0]
    s = list(range(0, length - TILE + 1, STRIDE))
    if s[-1] != length - TILE:
        s.append(length - TILE)
    return s


def work_crop(img: np.ndarray, roi, view: str) -> np.ndarray:
    x0, y0, x1, y1 = roi
    crop = img[y0:y1, x0:x1]
    sc = WORK_SCALE[view]
    if sc != 1.0:
        crop = cv2.resize(crop, None, fx=sc, fy=sc,
                          interpolation=cv2.INTER_LINEAR if sc > 1 else cv2.INTER_AREA)
    return np.ascontiguousarray(crop)


def make_tiles(crop: np.ndarray) -> list[Tile]:
    H, W = crop.shape[:2]
    out = []
    for oy in _starts(H):
        for ox in _starts(W):
            t = crop[oy:oy + TILE, ox:ox + TILE]
            h, w = t.shape[:2]
            if h < TILE or w < TILE:
                pad = np.full((TILE, TILE, 3), PAD_VALUE, np.uint8)
                pad[:h, :w] = t
                t = pad
            out.append(Tile(rgb=np.ascontiguousarray(t), ox=ox, oy=oy, w=w, h=h))
    return out


def to_work(box, roi, view: str) -> tuple[float, float, float, float]:
    """Native frame box -> work-scale crop coordinates."""
    sc = WORK_SCALE[view]
    return ((box[0] - roi[0]) * sc, (box[1] - roi[1]) * sc,
            (box[2] - roi[0]) * sc, (box[3] - roi[1]) * sc)


def to_native(box, roi, view: str) -> tuple[float, float, float, float]:
    """Work-scale crop box -> native frame coordinates."""
    sc = WORK_SCALE[view]
    return (box[0] / sc + roi[0], box[1] / sc + roi[1],
            box[2] / sc + roi[0], box[3] / sc + roi[1])
