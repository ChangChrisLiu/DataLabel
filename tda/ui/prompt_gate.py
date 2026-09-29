"""No box when unsure: is the difference map's box the size of the part asked for?

Task U2h wires **M1** of the L1 localisation experiment
(``experiments/l1_localise/REPORT.md``, sections 4 and 8).  The difference map
arms its strongest unexplained blob as SAM's box prompt, and on the scanner
that box is wrong more often than it is right for small parts: on D13/scan 37
it was a 6,332-pixel box on a black cable while the part to add back was a
screw of about 300.  A box of the wrong size is worse than none -- sent with
the click, it asks SAM for the cable -- so when the blob's area is outside the
area band of **every** class the frame asks to add back by more than a factor
``k``, the box is withheld and the annotator clicks the part instead.

Where it is on, ``k`` and the bands are configuration:

* ``configs/prompt_gate.yaml`` -- the views (``scan`` only: L1 withheld 36 %
  of the *good* boxes on oak2) and ``k`` (3);
* ``configs/prompt_box_bands.yaml`` -- per view and class, the 5th-95th
  percentile of the old Label Studio drafts' areas in image pixels, generated
  by ``experiments/l1_localise/fit_bands.py``.

A class without a band (too few drafts) never withholds, and a frame whose
card asks for nothing to be added is never gated.  Nothing here raises: a
missing or broken file is a gate that is off, and the box is armed as before.

``TDA_PROMPT_GATE`` names another gate file, or ``off``: the test suite sets
it (``tests/conftest.py``), because its synthetic 64 x 64 scan scene has
parts of no real size, and the tests that are about the gate load the
shipped file themselves.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["DEFAULT_FILE", "DEFAULT_K", "ENV_VAR", "PromptGate", "Withheld",
           "load_prompt_gate", "outside_factor"]

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "configs" / "prompt_gate.yaml"
#: Another gate file for this process, or ``off``.
ENV_VAR = "TDA_PROMPT_GATE"
#: L1's factor: the box goes when its area is more than three times above the
#: band's top or below its bottom.
DEFAULT_K = 3.0

Band = tuple[float, float]


def outside_factor(area: float, band: Optional[Band]) -> float:
    """How far outside ``band`` ``area`` is, as a factor >= 1 (1 = inside).

    L1's ``methods.outside_factor``: ``lo / area`` below the band, ``area / hi``
    above it.  No band, or no area, is "inside".
    """
    if band is None or not area or area <= 0:
        return 1.0
    lo, hi = band
    if area < lo:
        return float(lo) / float(area)
    if area > hi:
        return float(area) / float(hi)
    return 1.0


@dataclass(frozen=True)
class Withheld:
    """Why a box was withheld: its area, the classes, how far outside the nearest band."""

    area: int
    classes: tuple[str, ...]
    factor: float


class PromptGate:
    """The views the gate is on for, its factor and its bands."""

    def __init__(self, views: Iterable[str] = (), k: float = DEFAULT_K,
                 bands: Optional[dict] = None) -> None:
        self.views = frozenset(str(v) for v in views)
        self.k = float(k)
        #: ``{view: {cls: (lo_px, hi_px)}}``
        self.bands: dict[str, dict[str, Band]] = {
            str(view): {str(cls): (float(lo), float(hi))
                        for cls, (lo, hi) in (classes or {}).items()}
            for view, classes in (bands or {}).items()
        }

    def applies(self, view: Optional[str]) -> bool:
        """Is the gate on for ``view``?"""
        return str(view) in self.views

    def band(self, view: str, cls: str) -> Optional[Band]:
        """``(lo_px, hi_px)`` of a class in a view, or ``None`` when it has none."""
        return self.bands.get(str(view), {}).get(str(cls))

    def judge(self, view: Optional[str], classes: Iterable[str],
              area: float) -> Optional[Withheld]:
        """``None`` keeps the box; a :class:`Withheld` says why it goes.

        The box is kept when the gate is off for the view, when no class is
        asked for, when any asked-for class has no band (it could be that
        part), and when the area fits any class's band within ``k``.
        """
        if not self.applies(view):
            return None
        asked = tuple(dict.fromkeys(str(c) for c in classes if c))
        if not asked:
            return None
        factors = []
        for cls in asked:
            band = self.band(str(view), cls)
            if band is None:
                return None
            factor = outside_factor(area, band)
            if factor <= self.k:
                return None
            factors.append(factor)
        return Withheld(int(area), asked, min(factors))


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def load_prompt_gate(path: Optional[str] = None) -> PromptGate:
    """The gate of ``configs/prompt_gate.yaml`` (or ``path``); never raises.

    Without ``path`` the environment's :data:`ENV_VAR` is read first: ``off``
    is a gate that is off, anything else a file to read instead.  A file that
    is missing or does not parse is a gate that is off -- a typo in a tuning
    file must not stop anybody annotating, and "off" is exactly the box the
    annotator had before.
    """
    if path is None:
        chosen = os.environ.get(ENV_VAR, "").strip()
        if chosen.lower() == "off":
            return PromptGate()
        path = chosen or None
    target = Path(path) if path else DEFAULT_FILE
    try:
        policy = _read_yaml(target)
        views = [str(v) for v in (policy.get("views") or [])]
        k = float(policy.get("k", DEFAULT_K))
        bands_file = target.parent / str(policy.get("bands") or "prompt_box_bands.yaml")
        fitted = _read_yaml(bands_file).get("views") or {}
        bands: dict = {}
        for view, entry in fitted.items():
            classes = (entry or {}).get("classes") or {}
            bands[str(view)] = {
                str(cls): (float(row["lo_px"]), float(row["hi_px"]))
                for cls, row in classes.items()
                if isinstance(row, dict) and "lo_px" in row and "hi_px" in row
            }
        return PromptGate(views, k, bands)
    except Exception:  # noqa: BLE001 - see the docstring
        return PromptGate()
