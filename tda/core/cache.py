"""Local image cache, scanner burst selection and chassis ROI suggestion (spec 2.4).

The annotator never reads the source drive while a person is waiting: the image
each step needs is copied once into ``<cache>/<view>/D<nn>/s<kkk>.<ext>``.  For
the scanner every step is a *burst* of 5-10 shots of the same scene, so the copy
step also has to decide which shot represents the step:

* the default is ``P_0`` (the first shot of the burst);
* a burst that starts at ``P_2`` (D49 ``RGB261``) has no ``P_0``, so the
  lowest-numbered shot that passes the exposure check is used
  (reason ``"p0_missing"``);
* a ``P_0`` that is unexposed, blown out, or far from the burst's pixel-wise
  median (a hand or tool reaching over the board) is replaced by the sharpest
  remaining shot, and the reason is recorded for human review.

Every decision, together with the per-shot metrics it was made from, is written
to ``<cache>/<view>/D<nn>/manifest.json`` so the choice can be reviewed and
overridden in the UI without re-reading the source drive.

Nothing here ever writes outside ``cache_dir``; the source roots are read-only.
Images are OpenCV BGR arrays and all coordinates are in *original* image pixels.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np

from tda.core.index import DEFAULT_PATHS_PATH, REPO_ROOT, DesktopIndex, FrameFile, load_index
from tda.core.model import FrameKey

__all__ = [
    "build_cache", "burst_metrics", "cache_path", "choose_scan_image", "suggest_roi",
]

# --- burst metrics -------------------------------------------------------
DOWNSCALE = 8  # metrics are computed on a 1/8 copy (1600^2 -> 200^2) for speed
SAT_LEVEL = 250  # a pixel counts as saturated when every channel is >= this

# --- burst selection (thresholds pinned by the user, see module docstring)
# Only a gross failure may override decision C10 ("use P_0"); the thresholds are
# absolute, because within a burst the scanner lamp drifts by a gray level or
# two and the white reference board saturates ~30% of every good frame (median
# 0.29, up to 0.58, measured over 396 real bursts).
MEAN_MIN = 40.0  # mean gray below this -> unexposed (a lamp failure sits at ~28)
SAT_MAX = 0.5  # saturated fraction above this -> blown out
DIST_MAX = 20.0  # mean abs difference to the burst median above this gray levels
#                  -> a gross scene difference, e.g. a hand or tool in the shot

# --- cache layout --------------------------------------------------------
VIEW_EXT = {"scan": "png", "rs": "png", "oak1": "jpg", "oak2": "jpg"}
MANIFEST_NAME = "manifest.json"
MANIFEST_FLUSH_EVERY = 25  # flush mid-desktop so an interrupted run resumes
SINGLE_REASON = "only"  # views without a burst have nothing to choose

# --- ROI suggestion ------------------------------------------------------
YELLOW_LO = (10, 60, 60)  # HSV bounds of the yellow/orange tape square
YELLOW_HI = (40, 255, 255)
TAPE_MIN_AREA_FRAC = 0.005  # ignore yellow specks: no tape found -> fallback
TAPE_MIN_SPAN_FRAC = 0.5  # ... and a tape square frames the board, so it has to
#                           span at least half the frame; a smaller yellow blob is
#                           something else (a label, a cable) -> fallback
DARK_MAX = 90  # gray below this counts as "dark object" (chassis)
DARK_MIN_AREA_FRAC = 0.005
ROI_PAD_FRAC = 0.03  # pad the chassis box by 3% of its own size
CENTRAL_FRAC = 0.70  # fallback / non-scan views: central 70% box

_P_RE = re.compile(r"P_(\d+)", re.IGNORECASE)


def _norm(path: str) -> str:
    """Forward-slash form of ``path`` (the index and manifests use it)."""
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# burst metrics
# ---------------------------------------------------------------------------
def _downscale(img: np.ndarray) -> np.ndarray:
    """Area-average ``img`` down by :data:`DOWNSCALE` (never below 1 px)."""
    h, w = img.shape[:2]
    size = (max(1, w // DOWNSCALE), max(1, h // DOWNSCALE))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _p_index(path: str, fallback: int) -> int:
    """Burst number from a ``P_<k>.png`` name, else the position in the burst."""
    m = _P_RE.search(os.path.basename(str(path)))
    return int(m.group(1)) if m else fallback


def burst_metrics(paths: list[str]) -> list[dict]:
    """Per-image quality metrics for one scanner burst, in the given order.

    Each record holds ``path``, ``p_index`` (the ``k`` of ``P_k.png``), ``mean``
    and ``lap_var`` (mean gray and variance of the Laplacian of the 1/8 gray
    copy), ``sat_frac`` (fraction of pixels saturated in all three channels) and
    ``dist_to_median`` (mean absolute difference to the pixel-wise median of the
    downscaled burst - this is what a hand or tool in one shot shows up as).

    Raises ``ValueError`` on an empty burst and ``OSError`` on an image that
    cannot be decoded.  ``dist_to_median`` is left at 0 if the burst mixes image
    sizes, since no pixel-wise median exists then.
    """
    if not paths:
        raise ValueError("burst_metrics() needs at least one image path")

    grays: list[np.ndarray] = []
    out: list[dict] = []
    for i, path in enumerate(paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"cannot read image: {path}")
        small = _downscale(img)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grays.append(gray)
        out.append({
            "path": _norm(path),
            "p_index": _p_index(path, i),
            "mean": round(float(gray.mean()), 3),
            "sat_frac": round(float(np.all(small >= SAT_LEVEL, axis=2).mean()), 5),
            "lap_var": round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 2),
            "dist_to_median": 0.0,
        })

    if len({g.shape for g in grays}) == 1:
        median = np.median(np.stack(grays), axis=0)
        for rec, gray in zip(out, grays):
            rec["dist_to_median"] = round(float(np.abs(gray - median).mean()), 3)
    return out


# ---------------------------------------------------------------------------
# burst selection
# ---------------------------------------------------------------------------
def _reject_kind(rec: dict) -> Optional[str]:
    """Why this shot is unusable (``dark``/``saturated``/``outlier``), or None."""
    if float(rec.get("mean", 0.0)) < MEAN_MIN:
        return "dark"
    if float(rec.get("sat_frac", 0.0)) > SAT_MAX:
        return "saturated"
    if float(rec.get("dist_to_median", 0.0)) > DIST_MAX:
        return "outlier"
    return None


def choose_scan_image(metrics: list[dict]) -> tuple[int, str]:
    """Pick the shot that represents the step; return its *position* and why.

    ``P_0`` wins unless it fails :func:`_reject_kind`, in which case the sharpest
    (highest ``lap_var``) shot that passes the same checks is used.  Reasons are
    ``"p0"``, ``"p0_missing"`` (no ``P_0`` in the burst - lowest available number
    that passes), and ``"p0_dark"``/``"p0_saturated"``/``"p0_outlier"`` when
    ``P_0`` was replaced.  When *no* shot passes, the checks cannot say which
    shot is better, so the default wins anyway and the reason gets a ``"_kept"``
    suffix (``"p0_missing_fallback"`` for a burst without a ``P_0``); those are
    the steps to review.
    """
    if not metrics:
        raise ValueError("choose_scan_image() needs at least one metrics record")

    passing = [i for i, m in enumerate(metrics) if _reject_kind(m) is None]

    def p_index(i: int) -> int:
        return int(metrics[i].get("p_index", i))

    def sharpest(pool: Iterable[int]) -> int:
        return max(pool, key=lambda i: (float(metrics[i].get("lap_var", 0.0)), -p_index(i)))

    p0 = next((i for i in range(len(metrics)) if p_index(i) == 0), None)
    if p0 is None:  # e.g. D49 RGB261, whose burst on disk starts at P_2
        pool = passing or list(range(len(metrics)))
        lowest = min(pool, key=lambda i: (p_index(i), i))
        return lowest, "p0_missing" if passing else "p0_missing_fallback"

    if p0 in passing:
        return p0, "p0"

    reason = f"p0_{_reject_kind(metrics[p0])}"
    alternatives = [i for i in passing if i != p0]
    if alternatives:
        return sharpest(alternatives), reason
    return p0, f"{reason}_kept"  # nothing passed: the default still wins


# ---------------------------------------------------------------------------
# ROI suggestion
# ---------------------------------------------------------------------------
def _central_box(width: int, height: int) -> tuple[int, int, int, int]:
    """The central :data:`CENTRAL_FRAC` box of a ``width`` x ``height`` image."""
    bw, bh = int(round(width * CENTRAL_FRAC)), int(round(height * CENTRAL_FRAC))
    x0, y0 = (width - bw) // 2, (height - bh) // 2
    return (x0, y0, x0 + bw, y0 + bh)


def _pad_box(box: tuple[int, int, int, int], width: int, height: int,
             frac: float = ROI_PAD_FRAC) -> tuple[int, int, int, int]:
    """Grow ``box`` by ``frac`` of its own size, clipped to the image."""
    x0, y0, x1, y1 = box
    px, py = int(round((x1 - x0) * frac)), int(round((y1 - y0) * frac))
    return (max(0, x0 - px), max(0, y0 - py), min(width, x1 + px), min(height, y1 + py))


def _odd(value: float, minimum: int = 3) -> int:
    """Nearest odd kernel size >= ``minimum``."""
    return max(minimum, int(round(value)) | 1)


def _largest_component(mask: np.ndarray) -> tuple[Optional[int], np.ndarray, np.ndarray]:
    """Label of the biggest non-background blob in ``mask`` plus the label image."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None, labels, stats
    return 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), labels, stats


def _scan_chassis_box(bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Chassis box of a scanner frame, or None when the tape square is absent.

    The board is framed by a yellow tape square; the chassis is the largest dark
    object inside it.  Small gaps in the tape (and the loose corner scraps) are
    bridged by a dilation, the filled convex hull of the bridged square gives the
    board region, and the tape band itself is then removed from it.
    """
    height, width = bgr.shape[:2]
    small_k = _odd(min(height, width) * 0.005)
    bridge_k = _odd(min(height, width) * 0.02)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((small_k, small_k), np.uint8))
    bridged = cv2.dilate(yellow, np.ones((bridge_k, bridge_k), np.uint8))

    label, labels, stats = _largest_component(bridged)
    if label is None or stats[label, cv2.CC_STAT_AREA] < TAPE_MIN_AREA_FRAC * height * width:
        return None
    if (stats[label, cv2.CC_STAT_WIDTH] < TAPE_MIN_SPAN_FRAC * width
            or stats[label, cv2.CC_STAT_HEIGHT] < TAPE_MIN_SPAN_FRAC * height):
        return None  # not a square framing the board

    contours, _ = cv2.findContours((labels == label).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    board = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(board, cv2.convexHull(max(contours, key=cv2.contourArea)), 255)
    board = cv2.erode(board, np.ones((bridge_k, bridge_k), np.uint8))  # undo the bridging
    board[yellow > 0] = 0  # the tape band is not part of the inner region

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = ((gray < DARK_MAX) & (board > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((small_k, small_k), np.uint8))

    label, _, stats = _largest_component(dark)
    if label is None or stats[label, cv2.CC_STAT_AREA] < DARK_MIN_AREA_FRAC * height * width:
        return None
    x, y, w, h = (int(v) for v in stats[label, :4])
    return _pad_box((x, y, x + w, y + h), width, height)


def suggest_roi(img: np.ndarray, view: str) -> tuple[int, int, int, int]:
    """Suggest the chassis ROI ``(x0, y0, x1, y1)`` in original image pixels.

    For ``scan`` this is the largest dark object inside the yellow tape square,
    padded by 3%.  Any other view - and any scanner frame where the tape square
    or a dark object cannot be found - falls back to the central 70% box.  The
    suggestion is always confirmed by a human (spec 2.4); frames where the
    chassis covers the tape are the ones that usually need correcting.
    """
    if img is None or getattr(img, "size", 0) == 0:
        raise ValueError("suggest_roi() needs a non-empty image")
    height, width = img.shape[:2]
    if view == "scan":
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        box = _scan_chassis_box(bgr)
        if box is not None:
            return box
    return _central_box(width, height)


# ---------------------------------------------------------------------------
# the cache itself
# ---------------------------------------------------------------------------
def cache_path(cache_dir, key: FrameKey, ext: str) -> str:
    """Where the image of ``key`` lives in the cache, e.g. ``.../scan/D13/s042.png``."""
    root = _norm(cache_dir).rstrip("/")
    return f"{root}/{key.view}/D{key.desktop:02d}/s{key.step:03d}.{ext}"


def _load_manifest(path: str) -> dict:
    """Read an existing manifest, tolerating a missing or corrupt file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_manifest(path: str, manifest: dict) -> None:
    """Write the manifest atomically, steps in numeric order."""
    ordered = {k: manifest[k] for k in sorted(manifest, key=lambda s: int(s))}
    tmp = f"{path}.part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _select_source(frame: FrameFile, view: str) -> dict:
    """Manifest record for one frame: which source file to copy, and why."""
    if view == "scan":
        burst = [str(p) for p in (frame.aux.get("burst") or [frame.path])]
        metrics = burst_metrics(burst)
        chosen, reason = choose_scan_image(metrics)
        return {"chosen": chosen, "reason": reason, "metrics": metrics,
                "src": metrics[chosen]["path"]}
    return {"chosen": 0, "reason": SINGLE_REASON, "metrics": [], "src": _norm(frame.path)}


def _redecide(record: dict, view: str) -> Optional[dict]:
    """Re-run the choice on the metrics already in ``record``; None if unchanged.

    The metrics do not depend on the selection rule, so a cached step can be
    re-decided without touching the source drive again - that is how a re-run
    picks up a changed rule.
    """
    if view != "scan" or not record.get("metrics"):
        return None
    chosen, reason = choose_scan_image(record["metrics"])
    if chosen == record.get("chosen") and reason == record.get("reason"):
        return None
    updated = dict(record)
    updated["chosen"] = chosen
    updated["reason"] = reason
    updated["src"] = record["metrics"][chosen]["path"]
    return updated


def _up_to_date(src: str, dest: str) -> bool:
    """True when ``dest`` already holds a full copy of ``src`` (same byte size)."""
    try:
        return os.path.getsize(src) == os.path.getsize(dest)
    except OSError:
        return False


def _copy(src: str, dest: str) -> None:
    """Copy ``src`` to ``dest`` atomically (via a ``.part`` file in the cache)."""
    tmp = f"{dest}.part"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def build_cache(index: dict[int, DesktopIndex], cache_dir: str, views=("scan",),
                desktops=None, progress: Optional[Callable[[int, int, str], Any]] = None) -> dict:
    """Copy the image every step needs into the local cache; write the manifests.

    For each desktop in ``desktops`` (default: all of ``index``) and each view in
    ``views``, the chosen source image is copied to
    :func:`cache_path` and the decision is appended to
    ``<cache>/<view>/D<nn>/manifest.json`` as
    ``{step: {"chosen", "reason", "metrics", "src"}}``.

    Re-runs are cheap and idempotent: a step whose manifest record is present and
    whose cached file already has the source's byte size is not read again, only
    *re-decided* from the metrics in its record - so a re-run after a change to
    :func:`choose_scan_image` replaces the cached file (``rechosen``) or just the
    recorded reason (``relabelled``) without touching the source drive.  The
    manifest is flushed every :data:`MANIFEST_FLUSH_EVERY` copies as well as at
    the end of each desktop, so an interrupted run resumes instead of redoing the
    desktop it was in.  A step that cannot be read is recorded in ``failures``
    and does not stop the run.  ``progress(desktop, step, view)`` is called once
    per step, after it is handled; an exception it raises is not caught.

    Returns counters plus the manifests: ``copied``, ``skipped``, ``rechosen``,
    ``relabelled``, ``steps``, ``bytes_copied``, ``failures``, ``non_p0`` (every
    scanner step not represented by ``P_0``, whether written now or already
    cached), ``manifests`` and ``elapsed_s``.
    """
    started = time.perf_counter()
    stats: dict[str, Any] = {
        "copied": 0, "skipped": 0, "rechosen": 0, "relabelled": 0, "steps": 0,
        "bytes_copied": 0, "failures": [], "non_p0": [], "manifests": {},
        "elapsed_s": 0.0,
    }
    wanted = None if desktops is None else set(int(d) for d in desktops)

    for desktop in sorted(index):
        if wanted is not None and desktop not in wanted:
            continue
        frames = index[desktop].frames
        for view in views:
            keys = sorted((k for k in frames if k.view == view), key=lambda k: k.step)
            if not keys:
                continue
            ext = VIEW_EXT.get(view, "png")
            view_dir = os.path.dirname(cache_path(cache_dir, keys[0], ext))
            os.makedirs(view_dir, exist_ok=True)
            manifest_path = f"{view_dir}/{MANIFEST_NAME}"
            manifest = _load_manifest(manifest_path)
            dirty = False

            for key in keys:
                stats["steps"] += 1
                dest = cache_path(cache_dir, key, ext)
                step = str(key.step)
                try:
                    record = manifest.get(step)
                    if record and _up_to_date(record.get("src", ""), dest):
                        again = _redecide(record, view)
                        if again is None:
                            stats["skipped"] += 1
                        elif again["src"] != record["src"]:  # another shot wins now
                            _copy(again["src"], dest)
                            manifest[step] = record = again
                            dirty = True
                            stats["copied"] += 1
                            stats["rechosen"] += 1
                            stats["bytes_copied"] += os.path.getsize(dest)
                        else:  # same image, new reason: only the record changes
                            manifest[step] = record = again
                            dirty = True
                            stats["relabelled"] += 1
                    else:
                        record = _select_source(frames[key], view)
                        _copy(record["src"], dest)
                        manifest[step] = record
                        dirty = True
                        stats["copied"] += 1
                        stats["bytes_copied"] += os.path.getsize(dest)
                    if record["metrics"] and record["reason"] != "p0":
                        stats["non_p0"].append({
                            "desktop": desktop, "step": key.step, "view": view,
                            "chosen": record["chosen"], "reason": record["reason"],
                            "src": record["src"],
                        })
                except Exception as exc:  # a broken source must not stop the run
                    stats["failures"].append({
                        "desktop": desktop, "step": key.step, "view": view,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                if dirty and stats["copied"] % MANIFEST_FLUSH_EVERY == 0:
                    _write_manifest(manifest_path, manifest)  # resume point
                    dirty = False
                if progress is not None:
                    progress(desktop, key.step, view)

            if dirty:
                _write_manifest(manifest_path, manifest)
            stats["manifests"].setdefault(view, {})[desktop] = manifest

    stats["elapsed_s"] = round(time.perf_counter() - started, 2)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_cache_dir(paths_path: str = DEFAULT_PATHS_PATH) -> str:
    """``cache_dir`` from configs/paths.yaml."""
    import yaml

    path = paths_path if (os.path.isabs(paths_path) or os.path.exists(paths_path)) \
        else os.path.join(REPO_ROOT, paths_path)
    with open(path, "r", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("cache_dir", "D:/DataSet/cache")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: build the cache for a range of desktops, logging progress."""
    import argparse

    ap = argparse.ArgumentParser(description="Build the local TDA image cache.")
    ap.add_argument("--cache", default=None, help="cache dir (default: paths.yaml cache_dir)")
    ap.add_argument("--index", default=None, help="index JSON (default: <cache>/index.json)")
    ap.add_argument("--views", default="scan", help="comma separated views")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=66)
    ap.add_argument("--log", default=None, help="progress log file (default: stdout only)")
    args = ap.parse_args(argv)

    cache_dir = args.cache or _default_cache_dir()
    index_path = args.index or f"{_norm(cache_dir).rstrip('/')}/index.json"
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    desktops = [d for d in range(args.first, args.last + 1)]

    index = load_index(index_path)
    total = sum(1 for d in desktops if d in index
                for k in index[d].frames if k.view in views)
    log = open(args.log, "a", encoding="utf-8", buffering=1) if args.log else None

    def emit(line: str) -> None:
        print(line, flush=True)
        if log:
            log.write(line + "\n")

    started = time.time()
    done = 0

    def on_step(desktop: int, step: int, view: str) -> None:
        nonlocal done
        done += 1
        if done % 25 == 0 or done == total:
            elapsed = time.time() - started
            eta = elapsed / done * (total - done) if done else 0.0
            emit(f"[{time.strftime('%H:%M:%S')}] {done}/{total} "
                 f"({100.0 * done / max(total, 1):.1f}%) D{desktop:02d} s{step:03d} {view} "
                 f"elapsed {elapsed / 60:.1f}min eta {eta / 60:.1f}min")

    emit(f"[{time.strftime('%H:%M:%S')}] start views={views} desktops={args.first}-{args.last} "
         f"steps={total} cache={cache_dir} index={index_path}")
    try:
        stats = build_cache(index, cache_dir, views=views, desktops=desktops, progress=on_step)
        emit(f"[{time.strftime('%H:%M:%S')}] done steps={stats['steps']} copied={stats['copied']} "
             f"skipped={stats['skipped']} rechosen={stats['rechosen']} "
             f"relabelled={stats['relabelled']} GB={stats['bytes_copied'] / 1e9:.2f} "
             f"non_p0={len(stats['non_p0'])} failures={len(stats['failures'])} "
             f"elapsed={stats['elapsed_s'] / 60:.1f}min")
        for item in stats["non_p0"]:
            emit(f"  non_p0 D{item['desktop']:02d} s{item['step']:03d} "
                 f"chosen={item['chosen']} reason={item['reason']} src={item['src']}")
        for item in stats["failures"]:
            emit(f"  FAIL D{item['desktop']:02d} s{item['step']:03d} {item['view']}: {item['error']}")
    finally:
        if log:
            log.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
