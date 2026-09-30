"""How the 64 instance colours are chosen (task U3, minor c): as far apart as possible.

The palette used to be 64 golden-ratio hues on a 2 x 2 grid of saturation and
value.  Once five overlay hues were cut out of the wheel (U2g-U2i) only 190
degrees were left, and two colours of one brightness could land a hue-degree
apart: the closest pair was CIEDE2000 1.58 -- (43, 242, 29) against
(81, 242, 29), one colour to anybody.

:func:`build_palette` chooses them instead, deterministically:

* **the candidates** are every sRGB colour on a grid over the free hue arcs
  (:func:`tda.ui.canvas.overlay._free_arcs`, whole degrees), saturation
  :data:`SAT_LEVELS` and value :data:`VAL_LEVELS` -- lightness and saturation
  vary as much as the arcs need -- whose *rounded* colour still has a free hue
  and lies at least :data:`MIN_DE_TO_RESERVED` CIEDE2000 from the editing
  highlight and from every reserved overlay colour (:data:`MIN_DE_TO_ROI`
  from the ROI's magenta);
* **greedy farthest-point**: the first colour is the candidate farthest from
  those reserved colours, each next one the candidate farthest from every
  colour already chosen (ties to the lowest index -- the grid order);
* **then swaps**: each colour in turn is replaced by the candidate farthest
  from the other 63 whenever that raises the pair distance it takes part in,
  until a pass changes nothing -- the smallest distance in the palette never
  goes down.

It runs offline: :data:`tda.ui.canvas.overlay.PALETTE_64` is its output as a
literal table, and ``tests/test_overlay_style.py`` checks that the two agree.
The palette is display-only: no colour is stored anywhere.
"""
from __future__ import annotations

import colorsys
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["MIN_DE_TO_RESERVED", "MIN_DE_TO_ROI", "SAT_LEVELS", "VAL_LEVELS",
           "build_palette", "candidates", "delta_e_2000", "min_pairwise", "srgb_to_lab"]

#: Saturation grid.  Down to 0.45: at the old floor of 0.60 the best palette
#: found had its closest pair at 8.5; at 0.45 it is above 10.  White (the drag
#: band) is one of the reserved colours, so no pale colour comes near it.
SAT_LEVELS = tuple(round(0.45 + 0.05 * i, 2) for i in range(12))      # 0.45 .. 1.00
#: Value grid: dark enough to separate colours by lightness, bright enough to
#: read as a colour on the dark chassis.
VAL_LEVELS = tuple(round(0.50 + 0.05 * i, 2) for i in range(11))      # 0.50 .. 1.00
#: How far (CIEDE2000) every instance colour stays from the editing yellow and
#: from each overlay colour.
MIN_DE_TO_RESERVED = 10.0
#: ... and from the ROI's magenta, twice that: the one outline drawn round the
#: whole chassis, over every part (U2h widened its hue band for the same
#: reason).  At 10 the search put D13's chassis on hot pink (255, 25, 136),
#: 17 from the ROI and a degree outside its band; the closest pair of the
#: palette stays above 10 either way.
MIN_DE_TO_ROI = 20.0
#: How many colours.
SIZE = 64


def srgb_to_lab(rgb) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB colours given as 0-255 triples, shape ``(..., 3)``."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def delta_e_2000(lab1, lab2) -> np.ndarray:
    """CIEDE2000 between L*a*b* colours; broadcasts like numpy (Sharma et al. 2005)."""
    lab1, lab2 = np.asarray(lab1, np.float64), np.asarray(lab2, np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    cbar = 0.5 * (c1 + c2)
    g = 0.5 * (1.0 - np.sqrt(cbar ** 7 / (cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dL = L2 - L1
    dC = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(c1p * c2p == 0.0, 0.0, dh)
    dH = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2.0)
    Lbar = 0.5 * (L1 + L2)
    cbarp = 0.5 * (c1p + c2p)
    hsum = h1p + h2p
    hbar = np.where(np.abs(h1p - h2p) > 180.0,
                    np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0),
                    hsum / 2.0)
    hbar = np.where(c1p * c2p == 0.0, hsum, hbar)
    t = (1.0 - 0.17 * np.cos(np.radians(hbar - 30.0)) + 0.24 * np.cos(np.radians(2 * hbar))
         + 0.32 * np.cos(np.radians(3 * hbar + 6.0)) - 0.20 * np.cos(np.radians(4 * hbar - 63.0)))
    d_theta = 30.0 * np.exp(-(((hbar - 275.0) / 25.0) ** 2))
    rc = 2.0 * np.sqrt(cbarp ** 7 / (cbarp ** 7 + 25.0 ** 7))
    sl = 1.0 + 0.015 * (Lbar - 50.0) ** 2 / np.sqrt(20.0 + (Lbar - 50.0) ** 2)
    sc = 1.0 + 0.045 * cbarp
    sh = 1.0 + 0.015 * cbarp * t
    rt = -np.sin(np.radians(2.0 * d_theta)) * rc
    return np.sqrt((dL / sl) ** 2 + (dC / sc) ** 2 + (dH / sh) ** 2
                   + rt * (dC / sc) * (dH / sh))


def min_pairwise(colours: Sequence[Sequence[int]]) -> tuple[float, tuple, tuple]:
    """The smallest CIEDE2000 between two of ``colours`` and that pair."""
    lab = srgb_to_lab(np.asarray(colours, np.float64))
    d = delta_e_2000(lab[:, None, :], lab[None, :, :])
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    return float(d[i, j]), tuple(colours[i]), tuple(colours[j])


def _in_arcs(hue: float, arcs: Iterable[tuple[float, float]]) -> bool:
    return any(lo <= hue < hi for lo, hi in arcs)


def candidates(arcs: Sequence[tuple[float, float]], reserved: Sequence[Sequence[int]],
               sats: Sequence[float] = SAT_LEVELS, vals: Sequence[float] = VAL_LEVELS,
               min_de: float = MIN_DE_TO_RESERVED, margin_deg: float = 0.5,
               stricter: Sequence[tuple[Sequence[int], float]] = ()) -> np.ndarray:
    """Every grid colour allowed in the palette, in grid order, as ``(n, 3)`` ints.

    A colour is allowed when its hue *after rounding to 8 bits* lies inside a
    free arc with ``margin_deg`` to spare, it is at least ``min_de`` from every
    ``reserved`` colour, and at least the given distance from each colour in
    ``stricter`` (``(rgb, min_de)`` pairs).
    """
    inner = [(lo + margin_deg, hi - margin_deg) for lo, hi in arcs]
    out: list[tuple[int, int, int]] = []
    seen: set = set()
    for lo, hi in arcs:
        for hue in np.arange(np.ceil(lo), hi, 1.0):
            for sat in sats:
                for val in vals:
                    r, g, b = colorsys.hsv_to_rgb(float(hue) / 360.0, float(sat), float(val))
                    rgb = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
                    if rgb in seen:
                        continue
                    h, s, _v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
                    if s < min(sats) - 1e-9 or not _in_arcs((h * 360.0) % 360.0, inner):
                        continue
                    seen.add(rgb)
                    out.append(rgb)
    cand = np.asarray(out, dtype=np.int64)
    if len(reserved):
        d = delta_e_2000(srgb_to_lab(cand)[:, None, :],
                         srgb_to_lab(np.asarray(reserved, np.float64))[None, :, :])
        cand = cand[d.min(axis=1) >= min_de]
    for rgb, far in stricter:
        d = delta_e_2000(srgb_to_lab(cand), srgb_to_lab(np.asarray(rgb, np.float64)))
        cand = cand[d >= float(far)]
    return cand


def build_palette(arcs: Sequence[tuple[float, float]], reserved: Sequence[Sequence[int]],
                  size: int = SIZE, max_passes: int = 50,
                  pool: Optional[np.ndarray] = None,
                  stricter: Sequence[tuple[Sequence[int], float]] = ()
                  ) -> tuple[tuple[int, int, int], ...]:
    """``size`` colours from :func:`candidates`, as far apart as the search gets them."""
    cand = (candidates(arcs, reserved, stricter=stricter) if pool is None
            else np.asarray(pool, np.int64))
    if len(cand) < size:
        raise ValueError(f"only {len(cand)} candidate colours for a palette of {size}")
    lab = srgb_to_lab(cand)
    res = srgb_to_lab(np.asarray(reserved, np.float64)) if len(reserved) else None
    start = (delta_e_2000(lab[:, None, :], res[None, :, :]).min(axis=1)
             if res is not None else np.zeros(len(cand)))
    chosen = [int(np.argmax(start))]
    nearest = delta_e_2000(lab, lab[chosen[0]])
    nearest[chosen[0]] = -1.0
    while len(chosen) < size:
        pick = int(np.argmax(nearest))
        chosen.append(pick)
        nearest = np.minimum(nearest, delta_e_2000(lab, lab[pick]))
        nearest[chosen] = -1.0
    # distances from every candidate to every chosen colour, kept up to date
    dist = delta_e_2000(lab[:, None, :], lab[chosen][None, :, :])     # (n, size)
    for _ in range(max_passes):
        changed = False
        for slot in range(size):
            others = np.delete(dist, slot, axis=1).min(axis=1)       # to the other 63
            now = float(others[chosen[slot]])
            others[chosen] = -1.0
            best = int(np.argmax(others))
            if others[best] > now + 1e-9:
                chosen[slot] = best
                dist[:, slot] = delta_e_2000(lab, lab[best])
                changed = True
        if not changed:
            break
    return tuple(tuple(int(v) for v in cand[i]) for i in chosen)  # type: ignore[misc]
