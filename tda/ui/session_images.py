"""Where a frame's pixels come from, and how many of them stay in memory.

The annotator browses backwards through a hundred frames and flashes between
``k`` and ``k-1`` with ``Tab`` (spec 4.5), so decoded images have to be reused
-- but a scanner frame is 7.7 MB and an OAK frame 37 MB, and a cache bounded by
*count* is not a bound on anything that matters.  :class:`ImageCache` is an LRU
over bytes, with a count cap as a secondary guard and the most recent image
always kept, however large it is.

It also knows the two places a frame's pixels live: the local cache built by
:mod:`tda.core.cache`, and the small offline-built thumbnail the timeline draws
instead of decoding a 12 MP image per row.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional

import cv2
import numpy as np

from tda.core.cache import VIEW_EXT, cache_path
from tda.core.db import Db
from tda.core.model import FrameKey

__all__ = ["IMAGE_BUDGET_BYTES", "IMAGE_CACHE_SIZE", "ImageCache", "thumb_path"]

#: How many decoded frames are kept at most.  The real bound is
#: :data:`IMAGE_BUDGET_BYTES`; this only stops a view of small images from
#: keeping hundreds of them alive.
IMAGE_CACHE_SIZE = 8
#: Memory the decoded images may use in total.
IMAGE_BUDGET_BYTES = 256 * 1024 * 1024


def thumb_path(cache_dir: str, key: FrameKey) -> str:
    """Where the offline-built timeline thumbnail of one frame lives.

    Built by a separate pass over the cache, so drawing the timeline never
    decodes a full frame.  Kept here rather than in :mod:`tda.core.cache` only
    until that pass lands.
    """
    root = str(cache_dir).replace("\\", "/").rstrip("/")
    return f"{root}/thumbs/{key.view}/D{key.desktop:02d}/s{key.step:03d}.jpg"


def _first_readable(candidates: Iterable) -> Optional[str]:
    """The first of these paths that exists and can be opened, or ``None``."""
    for candidate in candidates:
        if not candidate:
            continue
        try:
            with open(str(candidate), "rb"):
                return str(candidate)
        except OSError:
            continue
    return None


class ImageCache:
    """Decoded RGB frames of one open desktop/view, bounded by memory."""

    def __init__(self, db: Db, cache_dir: str) -> None:
        self.db = db
        self.cache_dir = str(cache_dir)
        #: Memory the decoded images may use; a test or a small machine lowers it.
        self.budget_bytes = IMAGE_BUDGET_BYTES
        self._images: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()

    # -- paths --------------------------------------------------------------
    def image_path(self, key: FrameKey) -> Optional[str]:
        """The full-resolution image of one frame, or ``None`` when it has none.

        The cache layout is :func:`tda.core.cache.cache_path`'s; the frame row's
        own path is the fallback for a database whose images were never copied
        into the local cache.
        """
        row = self.db.get_frame(key) or {}
        return _first_readable([
            cache_path(self.cache_dir, key, VIEW_EXT.get(key.view, "png")),
            (row.get("aux") or {}).get("cache_path"),
            row.get("path"),
        ])

    def thumb_path(self, key: FrameKey) -> Optional[str]:
        """The timeline thumbnail, falling back to the full image."""
        small = _first_readable([thumb_path(self.cache_dir, key)])
        return small if small is not None else self.image_path(key)

    # -- pixels -------------------------------------------------------------
    def get(self, key: FrameKey) -> Optional[np.ndarray]:
        """The frame as RGB, decoding and caching it if need be."""
        slot = (key.view, int(key.step))
        hit = self._images.get(slot)
        if hit is not None:
            self._images.move_to_end(slot)
            return hit
        path = self.image_path(key)
        if path is None:
            return None
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._images[slot] = rgb
        self._trim()
        return rgb

    def put(self, key: FrameKey, rgb: np.ndarray) -> None:
        """Adopt a frame somebody else decoded (the background prefetch)."""
        self._images[(key.view, int(key.step))] = rgb
        self._images.move_to_end((key.view, int(key.step)))
        self._trim()

    def peek(self, key: FrameKey) -> Optional[np.ndarray]:
        """The frame if it is already decoded, without reading any file."""
        return self._images.get((key.view, int(key.step)))

    def _trim(self) -> None:
        """Drop the least recently used frames until the cache fits its bounds."""
        used = sum(img.nbytes for img in self._images.values())
        while len(self._images) > 1 and (
            used > self.budget_bytes or len(self._images) > IMAGE_CACHE_SIZE
        ):
            _slot, dropped = self._images.popitem(last=False)
            used -= dropped.nbytes

    def clear(self) -> None:
        self._images.clear()

    def as_dict(self) -> dict:
        """The cache contents (read-only; for tests and memory reporting)."""
        return dict(self._images)

    def __len__(self) -> int:
        return len(self._images)
