"""The two pieces of SAM plumbing that are not a tool: the crop and the bridge.

:func:`viewport_crop` turns what is on screen into the image SAM is asked
about, and :class:`SamResultBridge` is the only object that touches a result on
the worker thread -- it does nothing but re-emit it through a queued signal, so
the mask is applied on the GUI thread.  Both are used by every SAM tool and
neither knows anything about prompts, which is why they live apart from
:mod:`tda.ui.canvas.sam_tools`.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from tda.ui.canvas.tools import Box, Rect

__all__ = ["MAX_SAM_SIDE", "SamResultBridge", "norm_box", "viewport_crop"]

#: SAM 2.1 resizes its input to 1024 anyway, so a longer crop wastes work
#: and costs boundary precision on the way back (spec 4.6).
MAX_SAM_SIDE = 1024


def norm_box(a: tuple[float, float], b: tuple[float, float]) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def viewport_crop(
    canvas: Any, max_side: int = MAX_SAM_SIDE
) -> Optional[tuple[np.ndarray, Rect, float]]:
    """``(crop, rect, scale)`` for the visible image region, or ``None``.

    ``rect`` is the crop window in image coordinates and ``scale`` the factor
    applied to fit ``max_side`` (1.0 when the viewport is already small enough,
    which is the normal case once the annotator has zoomed in).

    A viewport larger than ``max_side`` is **downscaled rather than tiled**
    (spec 4.6 mentions tiling; deferred to P2).  SAM 2 resizes whatever it gets
    to 1024x1024 internally, so tiling would buy detail only where the
    annotator is already expected to zoom in, and there the crop is native
    resolution.  The one visible consequence: ``SamService`` measures its local
    refinement radius (:data:`~tda.models.sam_service.REFINE_RADIUS_PX`, 48 px)
    in *crop* pixels, so the region a refinement click can change spans
    ``REFINE_RADIUS_PX / scale`` **image** pixels -- a zoomed-out view refines
    coarsely.  Zoom in for a tight correction.
    """
    rgb = canvas.image_rgb()
    if rgb is None:
        return None
    rect = canvas.viewport_image_rect()
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return None
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        crop = cv2.resize(
            crop,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(crop), rect, scale


class SamResultBridge(QObject):
    """Moves a SAM result from the worker thread onto the GUI thread.

    ``SamQueue`` calls its callback on its own thread; touching the overlay or
    the scene from there would be a data race.  :meth:`deliver` is the callback
    and does nothing but emit -- the connection is queued, so the slot runs in
    the thread that owns this object (the GUI thread).
    """

    sigResult = Signal(object)
    #: An inference that raised; the text is for the status bar.
    sigFailed = Signal(str)

    def deliver(self, payload: object) -> None:
        """Callback for ``SamQueue.submit`` -- runs on the worker thread."""
        self.sigResult.emit(payload)

    def deliver_error(self, text: str) -> None:
        """``on_error`` callback for ``SamQueue.submit`` -- worker thread too."""
        self.sigFailed.emit(str(text))
