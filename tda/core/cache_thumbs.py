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
tells a human nothing.  The box a frame is cut with comes from ``roi_lookup``
(the annotation database, via :class:`tda.core.cache_roi.DbRoiLookup`, which
this module re-exports), and that lookup is asked *per frame*:
a chassis that was re-oriented has one pose segment per orientation and each
segment's own ``roi_json`` is the right crop for its steps.  Where the lookup
has nothing, :func:`tda.core.cache.suggest_roi` measures **one** box for the
whole desktop+view - on the reference frame, its first step, the assembled
chassis and so the largest silhouette, falling back to a median over a few
sampled steps and then to the frame's central box.  (``suggest_roi`` itself
measures the frame twice, as the darkest object on the bed and as whatever is
not the bed, and picks by shape; this tier only sees its answer.)  A measured
box is
deliberately *not* per frame: it would wobble from step to step and make the
timeline jitter.  A frame the lookup has no box for falls back to that measured
box, never to a neighbouring segment's.

The steps are therefore grouped by the box they share, and ``thumbs.json``
records those groups.  Staleness is decided per group, so adjusting one
segment's ROI rebuilds that segment's thumbnails and leaves the rest alone,
while a changed size or quality rebuilds the desktop.

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
import tempfile
import time
from typing import Any, Iterator, Optional

import cv2
import numpy as np

from tda.core.cache_roi import DbRoiLookup, RoiLookup  # re-exported: the ROI source
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
#: A measured box outside these bounds is not a chassis and is thrown away, and
#: the desktop is then not cropped at all.
#: :func:`~tda.core.cache.suggest_roi` applies the same bounds itself -- it has a
#: second strategy for the light chassis this floor used to be a stopgap for
#: (D64) and picks between them by shape -- so this is the guard on the *median
#: over sampled steps* below, which ``suggest_roi`` never sees.  They are
#: deliberately the same numbers: a box this tier would reject is not one to cut
#: a timeline with, whichever stage measured it.  The upper bound is also what
#: turns ``suggest_roi``'s "no crop" answer (the whole frame) into ``None``
#: here.
ROI_MIN_AREA_FRAC = 0.20
ROI_MAX_AREA_FRAC = 0.95  # ... and one this big is not a crop worth making

_DESKTOP_RE = re.compile(r"^D(\d+)$")
_UNSET = object()  # "the automatic box has not been measured yet", which None is not


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
    largest silhouette of the run.  If that frame's box is implausible (a
    detector that latched onto the motherboard of a light-coloured chassis, or
    onto the whole bed), the component-wise median over the sampled steps is
    tried, and if that is implausible too the desktop is **not cropped at all**.

    ``None`` means "no crop", and it means two different things that want the
    same answer: not one sampled frame could be read, or nothing measurable on
    them is chassis-shaped.  Cropping to a central box instead - the old
    behaviour - cut the machine in half on every one of the nine real desktops
    whose chassis covers its own tape square, which are exactly the desktops
    that get here.  :func:`_plan_groups` records the group as ``"none"``.
    """
    from tda.core.cache import VIEW_EXT, cache_path  # late: see the docstring

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


def _plan_groups(cache_dir, keys: list[FrameKey], view: str,
                 roi_lookup: Optional[RoiLookup], auto_roi: bool) -> list[dict]:
    """Group the steps of one desktop+view by the ROI each is cut with.

    ``roi_lookup`` is asked per frame, because a re-oriented chassis has one pose
    segment - and one ROI - per orientation; a lookup that can name the segment
    (:meth:`DbRoiLookup.segment_of`) makes the group survive a later adjustment
    of that segment's box, and a plain callable groups by the box itself.  Every
    frame it has nothing for shares the *one* box :func:`_auto_box` measures for
    the whole desktop+view, which is computed only if some frame needs it.

    Each group is ``{"source", "segment", "steps": [first, last], "box"}`` with
    ``_steps`` carrying the actual step numbers for the caller.  ``source`` is
    ``"db"`` (a pose segment's own ROI), ``"auto"`` (measured here) or
    ``"none"`` -- and ``"none"`` with a ``box`` of ``None`` means the thumbnail
    is cut from the **whole frame**, which is the right answer for a chassis
    that fills it.
    """
    segment_of = getattr(roi_lookup, "segment_of", None)
    groups: dict[tuple, dict] = {}
    auto: Any = _UNSET

    for key in keys:
        box = roi_lookup(key) if roi_lookup is not None else None
        segment = None
        if box is not None:
            source, box = "db", [int(v) for v in box]
            segment = segment_of(key) if segment_of is not None else None
        else:
            if auto is _UNSET:
                auto = _auto_box(cache_dir, keys, view) if auto_roi else None
            source, box = ("auto" if auto else "none"), auto
        identity = segment if segment is not None else (tuple(box) if box else None)
        group = groups.setdefault((source, identity), {"source": source, "segment": segment,
                                                       "box": box, "_steps": []})
        group["_steps"].append(key.step)

    for group in groups.values():
        group["steps"] = [group["_steps"][0], group["_steps"][-1]]
    return list(groups.values())


def _ranges(steps: list[int]) -> list[list[int]]:
    """``[1, 2, 3, 7, 8]`` -> ``[[1, 3], [7, 8]]``: membership, exactly, but short.

    A frame moved between segments by hand leaves a group with a hole in it, and
    a ``[first, last]`` pair would silently claim the steps in between.
    """
    out: list[list[int]] = []
    for step in steps:
        if out and step == out[-1][1] + 1:
            out[-1][1] = step
        else:
            out.append([step, step])
    return out


def _recorded(group: dict) -> dict:
    """The part of a group that goes into ``thumbs.json``."""
    return {"source": group["source"], "segment": group["segment"],
            "steps": _ranges(group["_steps"]), "box": group["box"]}


def _load_sidecar(path: str) -> dict:
    """The recorded plan of a desktop+view, tolerating a missing or corrupt file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _recorded_step_boxes(record: dict) -> Optional[dict[int, Any]]:
    """``{step: the box its thumbnail was built with}``, or ``None`` if unreadable.

    Staleness has to be decided per *step*, not per group: moving a pose-segment
    boundary changes which box a step is cut with while leaving both boxes - and
    both groups - exactly as they were.  ``None`` (a missing record, or one in
    either older format: a single box, or groups with ``[first, last]`` steps)
    means nothing can be trusted and every step is rebuilt.
    """
    groups = record.get("groups")
    if not isinstance(groups, list):
        return None
    out: dict[int, Any] = {}
    for group in groups:
        ranges = group.get("steps") if isinstance(group, dict) else None
        if not isinstance(ranges, list) or not all(
                isinstance(r, list) and len(r) == 2 for r in ranges):
            return None
        for first, last in ranges:
            for step in range(int(first), int(last) + 1):
                out[step] = group.get("box")
    return out


#: ``s042.jpg`` -- the only files in a thumbnail folder this module owns.
_THUMB_RE = re.compile(r"^s(\d+)\.jpg$", re.IGNORECASE)


def _drop_orphans(folder: str, steps: set[int]) -> int:
    """Delete the ``sNNN.jpg`` of every step the manifest no longer lists.

    The manifest is the authority on which steps a desktop has (see the module
    docstring), so a thumbnail outside it is a picture of a step that does not
    exist any more -- and the timeline will happily show it. Only files matching
    :data:`_THUMB_RE` are touched: the sidecar, and anything a human dropped in
    the folder, are not this function's business. A file that cannot be removed
    is left where it is; an orphan is untidy, a crashed batch job is worse.
    """
    removed = 0
    try:
        entries = os.listdir(folder)
    except OSError:
        return 0
    for name in entries:
        match = _THUMB_RE.match(name)
        if match is None or int(match.group(1)) in steps:
            continue
        try:
            os.remove(os.path.join(folder, name))
            removed += 1
        except OSError:
            pass
    return removed


def _write_sidecar(path: str, record: dict) -> None:
    """Record the plan next to the thumbnails it describes (atomically).

    The directory may not exist yet -- a membership change with nothing to
    rebuild still has to be recorded -- but the caller does not ask for a
    desktop that produced no thumbnail at all, because the folder alone would
    read as a built desktop on the next run.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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
    cache, cropped to the box :func:`_plan_groups` gives them (``roi_lookup`` per
    frame, else one measured box per desktop+view, which ``auto_roi=False``
    switches off), scaled so the longest side is ``max_side`` (``INTER_AREA``,
    never upscaled) and written to :func:`thumb_path`.

    Re-runs are cheap: a thumbnail at least as new as its source is skipped
    unless ``force`` is set, the desktop's ``max_side``/``quality`` changed, or
    *its own group's* box changed.  Adjusting one pose segment's ROI therefore
    rebuilds that segment's steps and leaves the rest of the desktop alone.
    Nothing outside ``<cache_dir>/thumbs`` is ever written, and one unreadable
    frame cannot stop the run: it is counted and described instead.

    A thumbnail whose step is no longer in the manifest is **deleted** and
    counted as ``removed``: a step dropped from the cache (a burst re-decided
    away, a desktop re-indexed shorter) otherwise leaves a picture the timeline
    can still open, of a step that does not exist. Only ``sNNN.jpg`` files are
    ever considered; the sidecar and anything else in the folder are left alone.

    Returns ``written``, ``skipped``, ``removed``, ``missing_source`` (a step in
    the manifest whose cached frame is not on disk), ``failed``, the matching
    ``failures`` list of ``{"path", "error"}``, the ``rois`` that were used
    (``{"view", "desktop", "groups"}`` per desktop+view), ``bytes`` written and
    ``elapsed_s``.
    """
    from tda.core.cache import VIEW_EXT, cache_path  # late: see the module docstring

    started = time.perf_counter()
    stats: dict[str, Any] = {"written": 0, "skipped": 0, "removed": 0,
                             "missing_source": 0, "failed": 0, "failures": [],
                             "rois": [], "bytes": 0, "elapsed_s": 0.0}
    wanted = None if desktops is None else {int(d) for d in desktops}
    root = _norm(cache_dir).rstrip("/")

    for view in views:
        ext = VIEW_EXT.get(view, "png")
        for desktop, keys in _cached_desktops(f"{root}/{view}", view, wanted):
            groups = _plan_groups(cache_dir, keys, view, roi_lookup, auto_roi)
            planned = [_recorded(g) for g in groups]
            stats["rois"].append({"view": view, "desktop": desktop, "groups": planned})
            sidecar = f"{root}/{THUMBS_DIRNAME}/{view}/D{desktop:02d}/{SIDECAR_NAME}"
            stored = _load_sidecar(sidecar)
            params = (int(max_side), int(quality))
            rebuild_all = force or (stored.get("max_side"), stored.get("quality")) != params
            was_built_with = _recorded_step_boxes(stored)
            written_here = 0

            for group in groups:
                for step in group["_steps"]:
                    key = FrameKey(desktop, step, view)
                    src, dest = cache_path(cache_dir, key, ext), thumb_path(cache_dir, key)
                    try:
                        if not os.path.isfile(src):
                            stats["missing_source"] += 1
                            continue
                        # _UNSET, not None: "nothing recorded" is not "recorded uncropped"
                        stale = rebuild_all or was_built_with is None or \
                            was_built_with.get(step, _UNSET) != group["box"]
                        if not stale and _up_to_date(src, dest):
                            stats["skipped"] += 1
                            continue
                        img = _thumbnail(_read_image(src), max_side, group["box"])
                        _write_jpeg(img, dest, quality)
                        written_here += 1
                        stats["bytes"] += os.path.getsize(dest)
                    except Exception as exc:  # one bad frame must not stop the batch
                        stats["failed"] += 1
                        stats["failures"].append(
                            {"path": src, "error": f"{type(exc).__name__}: {exc}"})

            stats["written"] += written_here
            folder = os.path.dirname(sidecar)
            if not os.path.isdir(folder):
                # Nothing of this desktop is on disk. Writing the sidecar would
                # create the folder for a plan describing no thumbnail at all,
                # and the next run would read that as a built desktop.
                continue
            stats["removed"] += _drop_orphans(folder, {k.step for k in keys})
            # A membership change on its own leaves every thumbnail correct but the
            # record wrong, so it too has to land; an unchanged plan is left alone,
            # or every run would dirty every sidecar with a new built_at.
            if written_here or rebuild_all or planned != stored.get("groups"):
                _write_sidecar(sidecar, {"groups": planned, "max_side": params[0],
                                         "quality": params[1],
                                         "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    stats["elapsed_s"] = round(time.perf_counter() - started, 2)
    return stats


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
        sources: dict[str, int] = {}  # how many desktop step-ranges each source cut
        for roi in stats["rois"]:
            for group in roi["groups"]:
                sources[group["source"]] = sources.get(group["source"], 0) + 1
        say(f"[{time.strftime('%H:%M:%S')}] thumbs written={stats['written']} "
            f"skipped={stats['skipped']} removed={stats['removed']} "
            f"missing_source={stats['missing_source']} "
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
