"""Offline thumbnail tier of the local image cache (spec 2.4).

The timeline shows one image per step, and the cached full frames are far too
big for that: 1600x1600 PNG for the scanner, up to 4032x3040 JPEG for the OAK.
Decoding and scaling those per visible row is what makes the panel crawl, so the
thumbnails are built *once*, offline, by the same batch job that fills the cache
and are written next to it::

    <cache_dir>/thumbs/<view>/D<nn>/s<kkk>.jpg

That layout is hard-coded in the UI, so :func:`thumb_path` is the contract; the
tier is otherwise disposable - deleting ``<cache_dir>/thumbs`` costs one re-run
and nothing else.

Which steps exist is read from the per-desktop ``manifest.json``
:func:`tda.core.cache.build_cache` writes, never from the source drive: a
thumbnail is a derivative of the *cached* frame, so building it never touches F:.

Cropping matters more than scaling here.  The chassis fills roughly a third of
an OAK frame, and a 192 px picture of the whole workbench tells a human nothing,
so ``roi_lookup`` may hand back the frame's chassis box; the thumbnail is then
cut to it (padded by :data:`ROI_PAD_FRAC`) before being scaled.
:class:`DbRoiLookup` is the lookup that reads those boxes from the annotation
database - strictly read-only, because a batch job has no business migrating,
stamping or even creating the file the annotator is using.

Images are OpenCV BGR arrays; no Qt here.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from typing import Any, Callable, Iterator, Optional
from urllib.parse import quote

import cv2
import numpy as np

from tda.core.cache import MANIFEST_NAME, VIEW_EXT, _load_manifest, cache_path
from tda.core.model import FrameKey

__all__ = [
    "DbRoiLookup", "THUMBS_DIRNAME", "add_thumb_args", "build_thumbs", "run_thumb_cli",
    "thumb_path",
]

THUMBS_DIRNAME = "thumbs"  # one sibling directory, so the tier is easy to drop
DEFAULT_MAX_SIDE = 192  # a timeline row is ~120 px tall on a HiDPI screen
DEFAULT_QUALITY = 85  # JPEG quality; 85 is visually lossless at this size
ROI_PAD_FRAC = 0.04  # grow a chassis box by 4% of its own size before cropping

_DESKTOP_RE = re.compile(r"^D(\d+)$")

#: What a ``roi_lookup`` is: a frame in, its chassis box ``(x0, y0, x1, y1)``
#: in *original image pixels* out, or ``None`` for "use the whole frame".
RoiLookup = Callable[[FrameKey], Optional[tuple[int, int, int, int]]]


def _norm(path) -> str:
    """Forward-slash form of ``path`` (the cache paths all use it)."""
    return str(path).replace("\\", "/")


def thumb_path(cache_dir, key: FrameKey) -> str:
    """Where the thumbnail of ``key`` lives, e.g. ``.../thumbs/scan/D13/s042.jpg``.

    Always a ``.jpg``, whatever the view's full-size format is: the tier exists
    to be small and quick to decode, and it is never the pixels anyone annotates.
    """
    root = _norm(cache_dir).rstrip("/")
    return f"{root}/{THUMBS_DIRNAME}/{key.view}/D{key.desktop:02d}/s{key.step:03d}.jpg"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _pad_and_clip(box, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    """Grow ``box`` by :data:`ROI_PAD_FRAC` of its own size, clipped to the image."""
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    px = int(round((x1 - x0) * ROI_PAD_FRAC))
    py = int(round((y1 - y0) * ROI_PAD_FRAC))
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(width, x1 + px), min(height, y1 + py)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _scaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Size whose longest side is ``max_side``; a smaller image is left alone.

    Upscaling would cost memory and add nothing - every real frame is far bigger
    than a thumbnail - so the scale is capped at 1.
    """
    scale = min(1.0, float(max_side) / max(width, height, 1))
    return (max(1, int(round(width * scale))), max(1, int(round(height * scale))))


def _thumbnail(img: np.ndarray, max_side: int,
               roi: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    """Crop ``img`` to ``roi`` (padded) and area-average it down to ``max_side``."""
    height, width = img.shape[:2]
    if roi is not None:
        padded = _pad_and_clip(roi, width, height)
        if padded is not None:
            x0, y0, x1, y1 = padded
            img = img[y0:y1, x0:x1]
            height, width = img.shape[:2]
    size = _scaled_size(width, height, max_side)
    if size == (width, height):
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _write_jpeg(img: np.ndarray, dest: str, quality: int) -> None:
    """Write ``img`` to ``dest`` atomically: a temp file beside it, then replace.

    A half-written thumbnail would be indistinguishable from a good one on the
    next run (it is newer than its source), so nothing is ever written to
    ``dest`` directly.  The temp file carries the ``.jpg`` suffix because that is
    how OpenCV picks the encoder, and it is removed on every failure path.
    """
    folder = os.path.dirname(dest)
    os.makedirs(folder, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=".tda-thumb-", suffix=".jpg", dir=folder)
    os.close(handle)
    try:
        if not cv2.imwrite(tmp, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
            raise OSError(f"cv2 could not encode a JPEG thumbnail for {dest}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _up_to_date(src: str, dest: str) -> bool:
    """True when the thumbnail is at least as new as the frame it came from."""
    try:
        return os.stat(dest).st_mtime_ns >= os.stat(src).st_mtime_ns
    except OSError:
        return False


# ---------------------------------------------------------------------------
# walking the cache
# ---------------------------------------------------------------------------
def _cached_keys(view_dir: str, view: str, wanted: Optional[set[int]]) -> Iterator[FrameKey]:
    """Every step of ``view`` the manifests know about, desktop by desktop."""
    try:
        entries = sorted(os.listdir(view_dir))
    except OSError:  # the view was never cached
        return
    for entry in entries:
        match = _DESKTOP_RE.match(entry)
        if match is None:
            continue
        desktop = int(match.group(1))
        if wanted is not None and desktop not in wanted:
            continue
        manifest = _load_manifest(f"{view_dir}/{entry}/{MANIFEST_NAME}")
        steps = sorted(s for s in manifest if str(s).lstrip("-").isdigit())
        for step in sorted(steps, key=int):
            yield FrameKey(desktop, int(step), view)


def build_thumbs(cache_dir, views=("scan",), desktops=None, max_side: int = DEFAULT_MAX_SIDE,
                 quality: int = DEFAULT_QUALITY, force: bool = False,
                 roi_lookup: Optional[RoiLookup] = None) -> dict:
    """Write one small JPEG per cached step; return what was done.

    For every desktop of ``views`` (all of them unless ``desktops`` names some)
    the steps listed in ``<cache>/<view>/D<nn>/manifest.json`` are read from the
    cache, cropped to ``roi_lookup(key)`` when it returns a box, scaled so the
    longest side is ``max_side`` (``INTER_AREA``, never upscaled) and written to
    :func:`thumb_path`.

    Re-runs are cheap: a thumbnail at least as new as its source is skipped
    unless ``force`` is set, so rebuilding a handful of cached frames only
    rebuilds those thumbnails.  Nothing outside ``<cache_dir>/thumbs`` is ever
    written, and one unreadable frame cannot stop the run - it is counted and
    described instead.

    Returns ``written``, ``skipped``, ``missing_source`` (a step in the manifest
    whose cached frame is not on disk), ``failed``, the matching ``failures``
    list of ``{"path", "error"}``, ``bytes`` written and ``elapsed_s``.
    """
    started = time.perf_counter()
    stats: dict[str, Any] = {"written": 0, "skipped": 0, "missing_source": 0, "failed": 0,
                             "failures": [], "bytes": 0, "elapsed_s": 0.0}
    wanted = None if desktops is None else {int(d) for d in desktops}
    root = _norm(cache_dir).rstrip("/")

    for view in views:
        ext = VIEW_EXT.get(view, "png")
        for key in _cached_keys(f"{root}/{view}", view, wanted):
            src = cache_path(cache_dir, key, ext)
            dest = thumb_path(cache_dir, key)
            try:
                if not os.path.isfile(src):
                    stats["missing_source"] += 1
                    continue
                if not force and _up_to_date(src, dest):
                    stats["skipped"] += 1
                    continue
                img = cv2.imread(src, cv2.IMREAD_COLOR)
                if img is None:
                    raise OSError(f"cannot decode cached frame: {src}")
                roi = roi_lookup(key) if roi_lookup is not None else None
                _write_jpeg(_thumbnail(img, max_side, roi), dest, quality)
                stats["written"] += 1
                stats["bytes"] += os.path.getsize(dest)
            except Exception as exc:  # one bad frame must not stop the batch
                stats["failed"] += 1
                stats["failures"].append({"path": src, "error": f"{type(exc).__name__}: {exc}"})

    stats["elapsed_s"] = round(time.perf_counter() - started, 2)
    return stats


# ---------------------------------------------------------------------------
# ROI from the annotation database
# ---------------------------------------------------------------------------
def _as_box(raw) -> Optional[tuple[int, int, int, int]]:
    """Parse a stored ROI into ``(x0, y0, x1, y1)``; ``None`` when it is not one.

    Accepts the shapes :func:`tda.core.export.coco.roi_of` accepts:
    ``[x0, y0, x1, y1]`` and a mapping with ``x0/y0/x1/y1`` or ``x/y/w/h``.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
    elif isinstance(raw, dict) and {"x0", "y0", "x1", "y1"} <= set(raw):
        x0, y0, x1, y1 = (int(raw[n]) for n in ("x0", "y0", "x1", "y1"))
    elif isinstance(raw, dict) and {"x", "y", "w", "h"} <= set(raw):
        x0, y0 = int(raw["x"]), int(raw["y"])
        x1, y1 = x0 + int(raw["w"]), y0 + int(raw["h"])
    else:
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


class DbRoiLookup:
    """``roi_lookup`` reading ``pose_segment.roi_json`` from a read-only database.

    A frame's segment is the one whose ``[start_step, end_step]`` contains its
    step for that desktop and view; a frame outside every segment, or in one
    without a ROI, gets ``None`` and therefore a whole-frame thumbnail.

    The file is opened through a ``file:...?mode=ro`` URI and *not* through
    :class:`tda.core.db.Db`: the schema bootstrap would create, migrate and
    re-stamp it, and a database written by a newer build would be refused
    outright - none of which a thumbnail batch may do to the annotator's working
    copy.  Segments are read once per ``(desktop, view)`` and kept, so a whole
    desktop costs one query.

    Use it as a context manager, or call :meth:`close` when done.
    """

    def __init__(self, db_path) -> None:
        self.path = _norm(db_path)
        self.conn = sqlite3.connect(self._read_only_uri(db_path), uri=True)
        self.conn.row_factory = sqlite3.Row
        self._segments: dict[tuple[int, str], list[tuple[int, int, Any]]] = {}

    @staticmethod
    def _read_only_uri(db_path) -> str:
        """``file:`` URI of ``db_path`` that sqlite may only read."""
        absolute = _norm(os.path.abspath(str(db_path)))
        return f"file:{quote(absolute, safe='/:')}?mode=ro"

    def __call__(self, key: FrameKey) -> Optional[tuple[int, int, int, int]]:
        for start, end, raw in self._for(key.desktop, key.view):
            if start <= key.step <= end:
                return _as_box(raw)
        return None

    def _for(self, desktop: int, view: str) -> list[tuple[int, int, Any]]:
        """The ROI-carrying segments of one view, read at most once."""
        cached = self._segments.get((desktop, view))
        if cached is None:
            rows = self.conn.execute(
                "SELECT start_step, end_step, roi_json FROM pose_segment "
                "WHERE desktop=? AND view=? AND roi_json IS NOT NULL "
                "AND start_step IS NOT NULL AND end_step IS NOT NULL ORDER BY seg",
                (desktop, view),
            ).fetchall()
            cached = [(int(r["start_step"]), int(r["end_step"]), r["roi_json"]) for r in rows]
            self._segments[(desktop, view)] = cached
        return cached

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DbRoiLookup":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI (wired into tda.core.cache.main)
# ---------------------------------------------------------------------------
def add_thumb_args(parser) -> None:
    """Add the thumbnail flags to the ``python -m tda.core.cache`` parser."""
    parser.add_argument("--thumbs", action="store_true",
                        help="also build the thumbnail tier after the cache")
    parser.add_argument("--thumbs-only", action="store_true",
                        help="build only the thumbnail tier (the cache itself is left alone)")
    parser.add_argument("--thumb-side", type=int, default=DEFAULT_MAX_SIDE,
                        help=f"longest side of a thumbnail, px (default {DEFAULT_MAX_SIDE})")
    parser.add_argument("--thumb-quality", type=int, default=DEFAULT_QUALITY,
                        help=f"JPEG quality (default {DEFAULT_QUALITY})")
    parser.add_argument("--force-thumbs", action="store_true",
                        help="rewrite every thumbnail, even an up-to-date one")
    parser.add_argument("--db", default=None,
                        help="annotation DB, opened READ-ONLY, whose pose_segment ROIs "
                             "crop the thumbnails; without it nothing is cropped")


def run_thumb_cli(args, cache_dir, views, desktops, emit) -> int:
    """Build the thumbnails of one CLI run; returns the process exit code."""
    lookup = DbRoiLookup(args.db) if args.db else None
    try:
        stats = build_thumbs(cache_dir, views=views, desktops=desktops,
                             max_side=args.thumb_side, quality=args.thumb_quality,
                             force=args.force_thumbs, roi_lookup=lookup)
    finally:
        if lookup is not None:
            lookup.close()
    emit(f"[{time.strftime('%H:%M:%S')}] thumbs written={stats['written']} "
         f"skipped={stats['skipped']} missing_source={stats['missing_source']} "
         f"failed={stats['failed']} MB={stats['bytes'] / 1e6:.1f} side={args.thumb_side} "
         f"roi={'db' if lookup is not None else 'none'} "
         f"elapsed={stats['elapsed_s'] / 60:.1f}min")
    for item in stats["failures"]:
        emit(f"  THUMB FAIL {item['path']}: {item['error']}")
    return 1 if stats["failed"] else 0
