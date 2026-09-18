"""Offline thumbnail tier of the local image cache (spec 2.4).

The timeline shows one image per step, and the cached full frames are far too
big for that: 1600x1600 PNG for the scanner, up to 4032x3040 JPEG for the OAK.
Decoding and scaling those per visible row is what makes the panel crawl, so the
thumbnails are built *once*, offline, by the same batch job that fills the cache
and are written next to it::

    <cache_dir>/thumbs/<view>/D<nn>/s<kkk>.jpg
    <cache_dir>/thumbs/<view>/D<nn>/thumbs.json   <- the ROI plan they were cut with

That layout is hard-coded in the UI, so :func:`thumb_path` is the contract; the
tier is otherwise disposable - deleting ``<cache_dir>/thumbs`` costs one re-run
and nothing else.

Which steps exist is read from the per-desktop ``manifest.json``
:func:`tda.core.cache.build_cache` writes, never from the source drive: a
thumbnail is a derivative of the *cached* frame, so building it never touches F:.

**Cropping matters more than scaling.**  The chassis is about a third of a
scanner frame and less of an OAK one, and a 192 px picture of the whole bench
tells a human nothing.  So every desktop+view gets *one* ROI, and every step of
it is cut to that same box: a per-step crop would make the timeline jitter and
stop the frames being comparable.  The box comes from ``roi_lookup`` (the
annotation database, via :class:`DbRoiLookup`) when it has one, else from
:func:`tda.core.cache.suggest_roi` measured on the desktop's reference frame -
its first step, the fully assembled chassis and so the largest silhouette - with
a median over a few sampled steps as the fallback.  The plan that was used is
recorded in ``thumbs.json`` so a later run with a different ROI source rebuilds
the tier instead of leaving half of it cut the old way.

Images are read and written through ``imdecode``/``imencode`` and plain Python
file IO, because ``cv2.imread``/``cv2.imwrite`` cannot open a non-ASCII path on
Windows.  Images are OpenCV BGR arrays; no Qt here.

This module imports :mod:`tda.core.cache` inside its functions, not at the top:
``cache`` re-exports the tier, and a late import is what keeps the two acyclic.
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

from tda.core.model import FrameKey

__all__ = [
    "DbRoiLookup", "SIDECAR_NAME", "THUMBS_DIRNAME", "add_thumb_args", "build_thumbs",
    "run_thumb_cli", "thumb_path",
]

THUMBS_DIRNAME = "thumbs"  # one sibling directory, so the tier is easy to drop
SIDECAR_NAME = "thumbs.json"  # the ROI plan a desktop's thumbnails were cut with
DEFAULT_MAX_SIDE = 192  # a timeline row is ~120 px tall on a HiDPI screen
DEFAULT_QUALITY = 85  # JPEG quality; 85 is visually lossless at this size
ROI_PAD_FRAC = 0.04  # grow a chassis box by 4% of its own size before cropping
ROI_SAMPLES = 5  # how many steps the fallback ROI is measured over
ROI_MIN_AREA_FRAC = 0.10  # a box smaller than this is not a chassis - distrust it
ROI_MAX_AREA_FRAC = 0.95  # ... and one this big is not a crop worth making

_DESKTOP_RE = re.compile(r"^D(\d+)$")

#: What a ``roi_lookup`` is: a frame in, its chassis box ``(x0, y0, x1, y1)``
#: in *original image pixels* out, or ``None`` for "nothing recorded".
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
# image IO (non-ASCII paths included)
# ---------------------------------------------------------------------------
def _read_image(path: str) -> np.ndarray:
    """Decode ``path`` into a BGR array; raises ``OSError`` when it cannot be read."""
    buffer = np.fromfile(path, dtype=np.uint8)  # not cv2.imread: Windows + non-ASCII
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if img is None:
        raise OSError(f"cannot decode cached frame: {path}")
    return img


def _write_jpeg(img: np.ndarray, dest: str, quality: int) -> None:
    """Write ``img`` to ``dest`` atomically: a temp file beside it, then replace.

    A half-written thumbnail would be indistinguishable from a good one on the
    next run (it is newer than its source), so nothing is ever written to
    ``dest`` directly.  The encode happens first, so a failure there leaves not
    even a temp file behind, and the temp file is removed on every other failure
    path too.
    """
    ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise OSError(f"cv2 could not encode a JPEG thumbnail for {dest}")
    folder = os.path.dirname(dest)
    os.makedirs(folder, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=".tda-thumb-", suffix=".jpg", dir=folder)
    os.close(handle)
    try:
        with open(tmp, "wb") as fh:
            fh.write(buffer.tobytes())
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _pad_and_clip(box, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    """Grow ``box`` by :data:`ROI_PAD_FRAC` of its own size, clipped to the image.

    ``None`` when nothing usable is left - a degenerate, inverted or wholly
    out-of-frame box, which the caller turns into "use the whole frame".
    """
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    px, py = int(round((x1 - x0) * ROI_PAD_FRAC)), int(round((y1 - y0) * ROI_PAD_FRAC))
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


def _thumbnail(img: np.ndarray, max_side: int, box) -> np.ndarray:
    """Crop ``img`` to ``box`` (padded) and area-average it down to ``max_side``."""
    height, width = img.shape[:2]
    padded = _pad_and_clip(box, width, height) if box else None
    if padded is not None:
        x0, y0, x1, y1 = padded
        img = img[y0:y1, x0:x1]
        height, width = img.shape[:2]
    size = _scaled_size(width, height, max_side)
    return img if size == (width, height) else cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _up_to_date(src: str, dest: str) -> bool:
    """True when the thumbnail is at least as new as the frame it came from."""
    try:
        return os.stat(dest).st_mtime_ns >= os.stat(src).st_mtime_ns
    except OSError:
        return False


# ---------------------------------------------------------------------------
# the ROI plan: one box per desktop+view
# ---------------------------------------------------------------------------
def _sample(keys: list[FrameKey], limit: int = ROI_SAMPLES) -> list[FrameKey]:
    """The reference step first, then up to ``limit`` steps spread over the run."""
    if len(keys) <= limit:
        return list(keys)
    last = len(keys) - 1
    picks = sorted({int(round(i * last / (limit - 1))) for i in range(limit)})
    return [keys[i] for i in picks]


def _plausible(box, width: int, height: int) -> bool:
    """Is this box a chassis at all - not a speck, not the whole scan bed?"""
    area = (box[2] - box[0]) * (box[3] - box[1])
    frac = area / float(max(width * height, 1))
    return ROI_MIN_AREA_FRAC <= frac <= ROI_MAX_AREA_FRAC


def _suggested(src: str, view: str) -> Optional[tuple[tuple[int, int, int, int], int, int]]:
    """``suggest_roi`` of one cached frame plus its size; ``None`` if unreadable."""
    from tda.core.cache import suggest_roi  # late: cache re-exports this module

    try:
        img = _read_image(src)
    except Exception:
        return None
    height, width = img.shape[:2]
    return (tuple(suggest_roi(img, view)), width, height)


def _auto_box(cache_dir, keys: list[FrameKey], view: str) -> Optional[list[int]]:
    """One ROI for the whole desktop+view, measured on its own frames.

    The reference frame is the first step - the chassis still assembled, so the
    largest silhouette of the run.  If that frame cannot be read, or its box is
    implausible (a detector that latched onto a cable or onto the whole bed), the
    component-wise median over the sampled steps is used instead; if that is
    implausible too, there is no trustworthy crop and the whole frame wins.
    """
    from tda.core.cache import VIEW_EXT, cache_path  # late: see the module docstring

    ext = VIEW_EXT.get(view, "png")
    sampled = _sample(keys)
    measured = [m for m in (_suggested(cache_path(cache_dir, k, ext), view) for k in sampled)
                if m is not None]
    if not measured:
        return None
    box, width, height = measured[0]
    if _plausible(box, width, height):
        return [int(v) for v in box]
    median = np.median(np.array([m[0] for m in measured], dtype=float), axis=0)
    box = tuple(int(round(v)) for v in median)
    return [int(v) for v in box] if _plausible(box, width, height) else None


def _plan_roi(cache_dir, keys: list[FrameKey], view: str,
              roi_lookup: Optional[RoiLookup], auto_roi: bool) -> dict:
    """Decide the one box this desktop+view is cut with, and where it came from.

    ``roi_lookup`` wins wherever it has an answer: it is a human's ROI, and it is
    consulted on the sampled steps rather than per frame so that the whole run
    keeps one box.  Otherwise :func:`_auto_box` measures one, and if even that
    fails the thumbnails are whole frames (``source="none"``).
    """
    if roi_lookup is not None:
        for key in _sample(keys):
            box = roi_lookup(key)
            if box:
                return {"source": "db", "box": [int(v) for v in box]}
    if auto_roi:
        box = _auto_box(cache_dir, keys, view)
        if box:
            return {"source": "auto", "box": box}
    return {"source": "none", "box": None}


def _load_sidecar(path: str) -> dict:
    """The recorded plan of a desktop+view, tolerating a missing or corrupt file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_sidecar(path: str, record: dict) -> None:
    """Record the plan next to the thumbnails it produced (atomically)."""
    tmp = f"{path}.part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _cached_desktops(view_dir: str, view: str,
                     wanted: Optional[set[int]]) -> Iterator[tuple[int, list[FrameKey]]]:
    """Each cached desktop of ``view`` with the steps its manifest lists."""
    from tda.core.cache import MANIFEST_NAME, _load_manifest  # late: see the module docstring

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
        steps = sorted((int(s) for s in manifest if str(s).lstrip("-").isdigit()))
        if steps:
            yield desktop, [FrameKey(desktop, s, view) for s in steps]


def build_thumbs(cache_dir, views=("scan",), desktops=None, max_side: int = DEFAULT_MAX_SIDE,
                 quality: int = DEFAULT_QUALITY, force: bool = False,
                 roi_lookup: Optional[RoiLookup] = None, auto_roi: bool = True) -> dict:
    """Write one small JPEG per cached step; return what was done.

    For every desktop of ``views`` (all of them unless ``desktops`` names some)
    the steps listed in ``<cache>/<view>/D<nn>/manifest.json`` are read from the
    cache, cropped to the desktop's single ROI (see :func:`_plan_roi`; switch the
    measured one off with ``auto_roi=False``), scaled so the longest side is
    ``max_side`` (``INTER_AREA``, never upscaled) and written to
    :func:`thumb_path`.

    Re-runs are cheap: a thumbnail at least as new as its source is skipped
    unless ``force`` is set *or* the desktop's recorded plan no longer matches
    the one this run computed - a changed ROI, ROI source, size or quality
    rebuilds that desktop, which is how a later ``--db`` run replaces thumbnails
    cut with the automatic box.  Nothing outside ``<cache_dir>/thumbs`` is ever
    written, and one unreadable frame cannot stop the run: it is counted and
    described instead.

    Returns ``written``, ``skipped``, ``missing_source`` (a step in the manifest
    whose cached frame is not on disk), ``failed``, the matching ``failures``
    list of ``{"path", "error"}``, the ``rois`` that were used (one record per
    desktop+view), ``bytes`` written and ``elapsed_s``.
    """
    from tda.core.cache import VIEW_EXT, cache_path  # late: see the module docstring

    started = time.perf_counter()
    stats: dict[str, Any] = {"written": 0, "skipped": 0, "missing_source": 0, "failed": 0,
                             "failures": [], "rois": [], "bytes": 0, "elapsed_s": 0.0}
    wanted = None if desktops is None else {int(d) for d in desktops}
    root = _norm(cache_dir).rstrip("/")

    for view in views:
        ext = VIEW_EXT.get(view, "png")
        for desktop, keys in _cached_desktops(f"{root}/{view}", view, wanted):
            plan = _plan_roi(cache_dir, keys, view, roi_lookup, auto_roi)
            stats["rois"].append({"view": view, "desktop": desktop, **plan})
            record = {**plan, "max_side": int(max_side), "quality": int(quality)}
            sidecar = f"{root}/{THUMBS_DIRNAME}/{view}/D{desktop:02d}/{SIDECAR_NAME}"
            stale = {k: _load_sidecar(sidecar).get(k) for k in record} != record
            written_here = 0

            for key in keys:
                src, dest = cache_path(cache_dir, key, ext), thumb_path(cache_dir, key)
                try:
                    if not os.path.isfile(src):
                        stats["missing_source"] += 1
                        continue
                    if not (force or stale) and _up_to_date(src, dest):
                        stats["skipped"] += 1
                        continue
                    _write_jpeg(_thumbnail(_read_image(src), max_side, plan["box"]), dest, quality)
                    written_here += 1
                    stats["bytes"] += os.path.getsize(dest)
                except Exception as exc:  # one bad frame must not stop the batch
                    stats["failed"] += 1
                    stats["failures"].append(
                        {"path": src, "error": f"{type(exc).__name__}: {exc}"})

            stats["written"] += written_here
            if written_here:  # only a tier that exists gets a plan recorded
                _write_sidecar(sidecar, {**record, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

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

    Which segment a frame is in is decided exactly as
    :meth:`tda.core.db.Db.pose_segment_for` decides it: the frame's own
    ``frame.pose_segment`` wins, and only when that is NULL (or names a segment
    that no longer exists) does the ``[start_step, end_step]`` range decide.  A
    frame outside every segment, or in one without a ROI, gets ``None`` and
    therefore an uncropped thumbnail.

    The file is opened through a ``file:...?mode=ro`` URI and *not* through
    :class:`~tda.core.db.Db`: the schema bootstrap would create, migrate and
    re-stamp it, and a database written by a newer build would be refused
    outright - none of which a thumbnail batch may do to the annotator's working
    copy.  Read-only here means the database itself is untouched (same bytes,
    same ``schema_version``, same tables); sqlite may still create the ``-wal``
    and ``-shm`` side files of a WAL database, which is harmless.  A
    ``<db>.lock`` left by a running annotator is reported through ``notify`` and
    otherwise ignored - reading alongside it is safe.

    Segments are read once per ``(desktop, view)``, so a whole desktop costs two
    queries.  Use it as a context manager, or call :meth:`close` when done.
    """

    def __init__(self, db_path, notify: Callable[[str], None] = print) -> None:
        self.path = _norm(db_path)
        if os.path.exists(f"{self.path}.lock"):
            notify(f"note: {self.path}.lock exists (the annotator may be open); reading anyway")
        self.conn = sqlite3.connect(self._read_only_uri(db_path), uri=True)
        self.conn.row_factory = sqlite3.Row
        self._segments: dict[tuple[int, str], dict[int, tuple[int, int, Any]]] = {}
        self._overrides: dict[tuple[int, str], dict[int, int]] = {}

    @staticmethod
    def _read_only_uri(db_path) -> str:
        """``file:`` URI of ``db_path`` that sqlite may only read."""
        absolute = _norm(os.path.abspath(str(db_path)))
        return f"file:{quote(absolute, safe='/:')}?mode=ro"

    def __call__(self, key: FrameKey) -> Optional[tuple[int, int, int, int]]:
        segments = self._segments_of(key.desktop, key.view)
        seg = self._overrides_of(key.desktop, key.view).get(key.step)
        chosen = segments.get(seg) if seg is not None else None
        if chosen is None:
            chosen = next((s for s in segments.values() if s[0] <= key.step <= s[1]), None)
        return _as_box(chosen[2]) if chosen is not None else None

    def _segments_of(self, desktop: int, view: str) -> dict[int, tuple[int, int, Any]]:
        """``{seg: (start, end, roi_json)}`` of one view, read at most once."""
        cached = self._segments.get((desktop, view))
        if cached is None:
            rows = self.conn.execute(
                "SELECT seg, start_step, end_step, roi_json FROM pose_segment "
                "WHERE desktop=? AND view=? AND start_step IS NOT NULL "
                "AND end_step IS NOT NULL ORDER BY seg", (desktop, view)).fetchall()
            cached = {int(r["seg"]): (int(r["start_step"]), int(r["end_step"]), r["roi_json"])
                      for r in rows}
            self._segments[(desktop, view)] = cached
        return cached

    def _overrides_of(self, desktop: int, view: str) -> dict[int, int]:
        """``{step: seg}`` for the frames that name a segment themselves."""
        cached = self._overrides.get((desktop, view))
        if cached is None:
            try:
                rows = self.conn.execute(
                    "SELECT step, pose_segment FROM frame WHERE desktop=? AND view=? "
                    "AND pose_segment IS NOT NULL", (desktop, view)).fetchall()
            except sqlite3.Error:  # a database without a frame table: ranges only
                rows = []
            cached = {int(r["step"]): int(r["pose_segment"]) for r in rows}
            self._overrides[(desktop, view)] = cached
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
    parser.add_argument("--no-auto-roi", dest="auto_roi", action="store_false",
                        help="do not measure a chassis ROI; keep the whole frame")
    parser.add_argument("--db", default=None,
                        help="annotation DB, opened READ-ONLY, whose pose_segment ROIs "
                             "crop the thumbnails instead of the measured ones")


def run_thumb_cli(args, cache_dir, views, desktops, emit=None) -> int:
    """Run the thumbnail stage of :func:`tda.core.cache.main`; returns its exit code.

    A no-op returning 0 unless ``--thumbs``/``--thumbs-only`` was given, so the
    caller can hand every run to it.  With no ``emit`` - the ``--thumbs-only``
    path, which runs before the cache stage opens its log - it writes its own.
    """
    if not (args.thumbs or args.thumbs_only):
        return 0
    own = open(args.log, "a", encoding="utf-8", buffering=1) if emit is None and args.log else None

    def say(line: str) -> None:
        if emit is not None:
            return emit(line)
        print(line, flush=True)
        if own is not None:
            own.write(line + "\n")

    lookup = DbRoiLookup(args.db, notify=say) if args.db else None
    try:
        stats = build_thumbs(cache_dir, views=views, desktops=desktops,
                             max_side=args.thumb_side, quality=args.thumb_quality,
                             force=args.force_thumbs, roi_lookup=lookup,
                             auto_roi=args.auto_roi)
        sources: dict[str, int] = {}
        for roi in stats["rois"]:
            sources[roi["source"]] = sources.get(roi["source"], 0) + 1
        say(f"[{time.strftime('%H:%M:%S')}] thumbs written={stats['written']} "
            f"skipped={stats['skipped']} missing_source={stats['missing_source']} "
            f"failed={stats['failed']} MB={stats['bytes'] / 1e6:.1f} side={args.thumb_side} "
            f"roi={'/'.join(f'{k}:{v}' for k, v in sorted(sources.items())) or 'none'} "
            f"elapsed={stats['elapsed_s'] / 60:.1f}min")
        for item in stats["failures"]:
            say(f"  THUMB FAIL {item['path']}: {item['error']}")
        return 1 if stats["failed"] else 0
    finally:
        if lookup is not None:
            lookup.close()
        if own is not None:
            own.close()
