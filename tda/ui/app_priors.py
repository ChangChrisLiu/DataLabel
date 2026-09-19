"""Is this mask a plausible size for what it claims to be?

Two rules, both **advisory**.  The annotator is the authority on what a part
looks like; these exist because the rehearsal found a 1,502,386-pixel mask
committed as a ``screw`` and 40 annotations under 50 px (the smallest 8 px, one
pixel wide), and nothing had said a word about either.

* **too small** -- under :data:`MIN_PIXELS`, or a bounding box thinner than
  :data:`MIN_SIDE_PX`.  It applies to every class, including one the priors
  have never seen, because a 9-pixel anything is a slip of the brush.
* **out of prior** -- outside the per-class band in ``configs/area_priors.yaml``,
  which is keyed by class and expressed as a fraction of the **ROI area** so
  that a screw on a 1600x1600 scan and the same screw on a 4032x3040 OAK frame
  are the same number.  The bands are generous on purpose (see the generator,
  ``experiments/area_priors_from_db.py``): a warning that cries wolf teaches the
  annotator to press ``Enter`` twice without reading, which is worse than no
  warning at all.

Nothing here blocks a commit.  The window shows a bar, a second ``Enter``
writes the mask as it is, and the override is logged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["MIN_PIXELS", "MIN_SIDE_PX", "AreaPriors", "area_warning", "load_priors"]

#: A mask smaller than this is a slip, whatever it is of.
MIN_PIXELS = 12
#: ... and so is one whose bounding box is thinner than this on either side.
MIN_SIDE_PX = 2
#: How far *below* its class band a mask may fall before the bar appears.  Only
#: the lower side has slack: tiny-but-valid happens (a screw head half behind a
#: bracket), while "much bigger than this class has ever been" is the mistake
#: the bar exists for.  Multiplying the **upper** bound by this as well put the
#: ceiling for a screw at 1.9 ROIs and for a motherboard at 10 -- a mask of the
#: whole frame is 3.5 ROIs, so the rule could not fire at all and the
#: rehearsal's 1,502,386-pixel "screw" went in in silence.
UNDERSIZE_SLACK = 10.0

_DEFAULT_FILE = Path(__file__).resolve().parents[2] / "configs" / "area_priors.yaml"


class AreaPriors:
    """The per-class bands, loaded once and asked per commit."""

    def __init__(self, classes: Optional[dict] = None) -> None:
        self.classes: dict[str, dict] = dict(classes or {})

    def band(self, cls: str) -> Optional[tuple[float, float]]:
        """``(min_frac, max_frac)`` for a class, or ``None`` when unknown."""
        row = self.classes.get(str(cls))
        if not isinstance(row, dict):
            return None
        try:
            return float(row["min_frac"]), float(row["max_frac"])
        except (KeyError, TypeError, ValueError):
            return None

    def check(self, cls: str, pixels: int, roi_area: float) -> Optional[str]:
        """Why this area is implausible for ``cls``, or ``None``."""
        band = self.band(cls)
        if band is None or roi_area <= 0 or pixels <= 0:
            return None
        low, high = band
        frac = float(pixels) / float(roi_area)
        if frac > high:
            return (f"{cls} 通常占 ROI 的 {low:.4%}–{high:.4%}，这个掩码占 "
                    f"{frac:.4%}（{pixels} px）—— 是不是多选了别的零件？")
        if low > 0 and frac * UNDERSIZE_SLACK < low:
            return (f"{cls} 通常占 ROI 的 {low:.4%}–{high:.4%}，这个掩码只占 "
                    f"{frac:.4%}（{pixels} px）—— 是不是只画到了一角？")
        return None


def load_priors(path: Optional[str] = None) -> AreaPriors:
    """Read ``configs/area_priors.yaml``; a missing or broken file is empty priors.

    Never raises: a typo in a tuning file must not stop anybody annotating.
    """
    target = Path(path) if path else _DEFAULT_FILE
    try:
        import yaml

        with open(target, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        classes = data.get("classes") if isinstance(data, dict) else None
        return AreaPriors(classes if isinstance(classes, dict) else {})
    except Exception:  # noqa: BLE001 - see the docstring
        return AreaPriors({})


def too_small(mask: np.ndarray) -> Optional[str]:
    """Is this mask too small or too thin to be anything?  The reason, or ``None``."""
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        return None      # an empty layer is "nothing to commit", not a warning
    if pixels < MIN_PIXELS:
        return f"掩码过小：只有 {pixels} 像素 / the mask is only {pixels} px"
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    height = int(rows[-1] - rows[0]) + 1 if rows.size else 0
    width = int(cols[-1] - cols[0]) + 1 if cols.size else 0
    if min(height, width) < MIN_SIDE_PX:
        return (f"掩码过小：外框只有 {width}x{height} 像素 / "
                f"the mask is {width}x{height} px")
    return None


def area_warning(mask: np.ndarray, cls: str, roi_area: float,
                 priors: Optional[AreaPriors] = None) -> Optional[str]:
    """One sentence for the warning bar, or ``None`` when the mask looks fine."""
    small = too_small(mask)
    if small is not None:
        return small
    if not cls:
        return None      # unknown class: rule (i) only, by ruling
    table = priors if priors is not None else load_priors()
    return table.check(str(cls), int(np.count_nonzero(mask)), float(roi_area))
