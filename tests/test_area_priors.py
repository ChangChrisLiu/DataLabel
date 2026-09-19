"""The shipped area priors, against the sizes a reviewer called plausible.

The bands in ``configs/area_priors.yaml`` are the ones the window loads, so
these are acceptance tests for the file as well as for the rule: the numbers
below are the reviewer's own list, measured at the ROI the real D13 scan
produces (827 x 876 px).
"""
from __future__ import annotations

import numpy as np
import pytest

from tda.ui.app_priors import area_warning, load_priors

#: The ROI of D13/scan, which is what the rehearsal measured against.
ROI_W, ROI_H = 827, 876
ROI_AREA = float(ROI_W * ROI_H)


@pytest.fixture(scope="module")
def priors():
    table = load_priors()
    assert table.classes, "configs/area_priors.yaml did not load"
    return table


def mask_of(pixels: int) -> np.ndarray:
    """A square-ish mask of ``pixels`` set pixels, big enough to hold them."""
    side = max(2, int(np.ceil(np.sqrt(pixels))))
    mask = np.zeros((side + 2, side + 2), dtype=bool)
    flat = mask.reshape(-1)
    flat[:pixels] = True
    return flat.reshape(mask.shape)


def box_mask(w: int, h: int) -> np.ndarray:
    mask = np.zeros((h + 2, w + 2), dtype=bool)
    mask[1:h + 1, 1:w + 1] = True
    return mask


# --------------------------------------------------------------------------- #
# what must warn
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls,pixels,why", [
    ("screw", int(ROI_AREA), "a whole-ROI screw"),
    ("screw", 1_502_386, "the rehearsal's 1.5 M px screw"),
    ("psu", int(1600 * 1600), "a whole-frame psu"),
    ("ram_latch", int(ROI_AREA), "a whole-ROI latch"),
])
def test_an_implausible_mask_warns(priors, cls, pixels, why):
    warning = area_warning(mask_of(pixels), cls, ROI_AREA, priors)
    assert warning is not None, f"{why} was not warned about"
    assert cls in warning


# --------------------------------------------------------------------------- #
# what must stay silent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls,w,h", [
    ("screw", 14, 14), ("screw", 60, 60),
    ("connector", 25, 20), ("connector", 70, 40),
    ("ram_module", 380, 30), ("ram_module", 430, 45),
    ("psu", 250, 200), ("psu", 420, 300),
    ("motherboard", 500, 400), ("motherboard", 760, 650),
])
def test_a_plausible_mask_is_silent(priors, cls, w, h):
    assert area_warning(box_mask(w, h), cls, ROI_AREA, priors) is None, (
        f"{cls} {w}x{h} ({w * h} px, {w * h / ROI_AREA:.4%} of the ROI) warned"
    )


def test_an_unknown_class_only_gets_the_absolute_rule(priors):
    assert area_warning(box_mask(300, 300), "gremlin", ROI_AREA, priors) is None
    assert area_warning(mask_of(6), "gremlin", ROI_AREA, priors) is not None


def test_the_shipped_bands_are_not_reachable_by_a_whole_frame(priors):
    """Every ceiling has to be inside one frame, or the rule cannot fire.

    The frame is ~3.5 ROIs on D13/scan; a band above that is a band nothing can
    ever exceed, which is how a 1.5 M px screw was committed in silence.
    """
    for cls in priors.classes:
        band = priors.band(cls)
        assert band is not None and band[1] <= 1.2, (cls, band)
