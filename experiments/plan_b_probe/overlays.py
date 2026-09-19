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

PANEL = 390       # width of each top panel
ANA = 800         # width of the anaglyph; keeps every PNG under the 1.5 MB cap


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


def zoom_window(yellow, safe, shape, frac=0.115):
    """Crop box over the densest patch of tape edge that is *also* table.

    The box has to sit on table-fixed pixels, otherwise it can land on the
    chassis outline, where doubling proves nothing about the camera.
    """
    h, w = shape
    bw, bh = int(w * frac), int(h * frac)
    edge = cv2.morphologyEx(yellow, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ie = cv2.integral((edge > 0).astype(np.uint8)).astype(np.int64)
    isf = cv2.integral((safe > 0).astype(np.uint8)).astype(np.int64)

    def box(I, x, y):
        return (I[y + bh, x + bw] - I[y, x + bw] - I[y + bh, x] + I[y, x])

    area = float(bw * bh)
    best, bx, by = -1.0, 0, 0
    for y in range(0, max(1, h - bh), max(8, bh // 6)):
        for x in range(0, max(1, w - bw), max(8, bw // 6)):
            frac_safe = box(isf, x, y) / area
            if frac_safe < 0.55:
                continue
            s = box(ie, x, y) * frac_safe
            if s > best:
                best, bx, by = s, x, y
    if best < 0:            # no table-dominated window: fall back to tape only
        for y in range(0, max(1, h - bh), max(8, bh // 6)):
            for x in range(0, max(1, w - bw), max(8, bw // 6)):
                s = box(ie, x, y)
                if s > best:
                    best, bx, by = s, x, y
    return bx, by, bw, bh, best


def make(path_a, path_b, view, title, sub, out_png, tape_outline=True):
    a = C.imread(path_a, 900)
    b = C.imread(path_b, 900)
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    ana = anaglyph(a, b)
    safe_a, ya = T.table_mask(a, view)
    safe_b, yb = T.table_mask(b, view)
    safe_both = cv2.bitwise_and(safe_a, safe_b)
    if tape_outline:
        # Draw the tape found in frame A so the reader knows what to watch.
        cnts, _ = cv2.findContours(ya, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 150]
        cv2.drawContours(ana, cnts, -1, (0, 255, 255), 1)

    # Magnified tape detail: a few-pixel shift is invisible in a downscaled
    # full frame, so the verdict is actually read off this strip.  The same
    # fixed cross is drawn on all three crops, so the question becomes "does
    # the tape edge sit in the same place relative to the cross?".
    zx, zy, zw, zh, score = zoom_window(cv2.bitwise_or(ya, yb), safe_both,
                                        a.shape[:2])
    ca = a[zy:zy + zh, zx:zx + zw].copy()
    cb = b[zy:zy + zh, zx:zx + zw].copy()
    cana = anaglyph(ca, cb)
    cw = ANA // 3
    zf = cw / float(max(1, zw))
    strip = []
    for crop, tag in ((ca, "A"), (cb, "B"), (cana, "A=red B=cyan")):
        z = cv2.resize(crop, (cw, int(zh * zf)), interpolation=cv2.INTER_NEAREST)
        cv2.line(z, (cw // 2, 0), (cw // 2, z.shape[0]), (0, 0, 255), 1)
        cv2.line(z, (0, z.shape[0] // 2), (z.shape[1], z.shape[0] // 2), (0, 0, 255), 1)
        _label(z, tag, 16)
        strip.append(z)
    hmin = min(s.shape[0] for s in strip)
    za = np.hstack([s[:hmin] for s in strip])
    if za.shape[1] < ANA:
        za = np.hstack([za, np.full((za.shape[0], ANA - za.shape[1], 3), 30, np.uint8)])
    hdr = np.full((22, ANA, 3), 30, np.uint8)
    _label(hdr, "ZOOM x%.1f of the red box on the TABLE -- same cross on each crop"
           % zf, 16, (0, 255, 255))
    za = np.vstack([hdr, za])
    cv2.rectangle(ana, (zx, zy), (zx + zw, zy + zh), (0, 0, 255), 2)

    pa, pb, pana = _fit(a, PANEL), _fit(b, PANEL), _fit(ana, ANA)
    _label(pa, "A  " + sub[0])
    _label(pb, "B  " + sub[1])
    _label(pana, "anaglyph  A=red  B=cyan   " + sub[2], 20, (0, 255, 255))

    top = np.hstack([pa, np.full((pa.shape[0], ANA - 2 * PANEL, 3), 30, np.uint8), pb])
    head = np.full((26, ANA, 3), 30, np.uint8)
    _label(head, title, 19)
    canvas = np.vstack([head, top, pana, za])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return out_png
