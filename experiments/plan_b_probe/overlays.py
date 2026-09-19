"""Validation overlays: make a flagged event checkable by eye.

Each PNG has the two frames side by side on top, and underneath a red/cyan
anaglyph of the same pair.  In the anaglyph anything that held still is grey and
anything that moved shows a coloured fringe, so the two cases are told apart at
a glance:

  * camera moved  -> the tape, the table marks and the table edge all fringe
  * chassis moved -> the tape stays grey, only the hardware fringes
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import tablemask as T     # noqa: E402

PANEL = 460       # width of each top panel
ANA = 940         # width of the anaglyph


def _fit(img, w):
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _label(img, text, y=18, color=(255, 255, 255)):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                cv2.LINE_AA)


def anaglyph(a, b):
    """A in red, B in cyan; identical content stays grey."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    out = np.zeros((*ga.shape, 3), np.uint8)
    out[:, :, 2] = ga          # red   = A
    out[:, :, 1] = gb          # green = B
    out[:, :, 0] = gb          # blue  = B
    return out


def make(path_a, path_b, view, title, sub, out_png, tape_outline=True):
    a = C.imread(path_a, 900)
    b = C.imread(path_b, 900)
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    ana = anaglyph(a, b)
    if tape_outline:
        # Draw the tape found in frame A so the reader knows what to watch.
        _s, ya = T.table_mask(a, view)
        cnts, _ = cv2.findContours(ya, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 150]
        cv2.drawContours(ana, cnts, -1, (0, 255, 255), 1)

    pa, pb, pana = _fit(a, PANEL), _fit(b, PANEL), _fit(ana, ANA)
    _label(pa, "A  " + sub[0])
    _label(pb, "B  " + sub[1])
    _label(pana, "anaglyph  A=red  B=cyan   " + sub[2], 20, (0, 255, 255))

    top = np.hstack([pa, np.full((pa.shape[0], ANA - 2 * PANEL, 3), 30, np.uint8), pb])
    head = np.full((26, ANA, 3), 30, np.uint8)
    _label(head, title, 19)
    canvas = np.vstack([head, top, pana])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return out_png
