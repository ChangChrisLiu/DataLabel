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

from tda.core.cache_roi_detect import (  # re-exported: suggest_roi's two strategies
    scan_bed_box,
    scan_bed_candidates,
    scan_chassis_box,
    scan_chassis_candidates,
)
from tda.core.cache_thumbs import (DbRoiLookup, add_thumb_args, build_thumbs,  # re-exported
                                   run_thumb_cli, thumb_path)
from tda.core.index import DEFAULT_PATHS_PATH, REPO_ROOT, DesktopIndex, FrameFile, load_index
from tda.core.model import FrameKey

__all__ = [
    "DbRoiLookup", "build_cache", "build_thumbs", "burst_metrics", "cache_path",
    "choose_scan_image", "full_frame", "scan_bed_candidates", "scan_chassis_candidates",
    "suggest_roi", "thumb_path",
]

# --- burst metrics -------------------------------------------------------
DOWNSCALE = 8  # metrics are computed on a 1/8 copy (1600^2 -> 200^2) for speed
SAT_LEVEL = 250  # a pixel counts as saturated when every channel is >= this

# --- burst selection (thresholds pinned by the user, see module docstring)
# Only a gross failure may override decision C10 ("use P_0"); the thresholds are
# absolute, because within a burst the scanner lamp drifts by a gray level or
# two and the white reference board saturates ~30% of every good frame (median
# 0.29, up to 0.58, measured over 2807 real bursts).
MEAN_MIN = 40.0  # mean gray below this -> unexposed (a lamp failure sits at ~28)
SAT_MAX = 0.75  # saturated fraction above this -> blown out; the board alone
#                 reaches 0.52-0.54 on the six brightest desktops, so a lower
#                 bar swapped good P_0 shots for equally bright siblings
DIST_MAX = 20.0  # mean abs difference to the burst median above this gray levels
#                  -> a gross scene difference, e.g. a hand or tool in the shot

# --- cache layout --------------------------------------------------------
VIEW_EXT = {"scan": "png", "rs": "png", "oak1": "jpg", "oak2": "jpg"}
MANIFEST_NAME = "manifest.json"
MANIFEST_FLUSH_EVERY = 25  # flush mid-desktop so an interrupted run resumes
METRICS_VERSION = 1  # bump whenever burst_metrics changes what it measures (a new
#                      metric, another SAT_LEVEL or DOWNSCALE): records stamped
#                      with an older version are measured again instead of being
#                      re-decided from stale numbers.  A record written before the
#                      stamp existed counts as version 1.
SINGLE_REASON = "only"  # views without a burst have nothing to choose

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

    **Every number is measured on the 1/8 downscale**, never on the full frame,
    and the thresholds in :func:`choose_scan_image` are calibrated against that:
    area-averaging suppresses sensor noise and softens saturated edges, so
    ``lap_var`` and ``sat_frac`` in particular are not comparable to values taken
    at full resolution.  Changing :data:`DOWNSCALE`, :data:`SAT_LEVEL` or the set
    of metrics therefore means bumping :data:`METRICS_VERSION`, which makes cached
    steps measure themselves again.

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
def full_frame(width: int, height: int) -> tuple[int, int, int, int]:
    """The whole image as a box -- what "no crop" means in ROI coordinates."""
    return (0, 0, int(width), int(height))


def suggest_roi(img: np.ndarray, view: str) -> tuple[int, int, int, int]:
    """Suggest the chassis ROI ``(x0, y0, x1, y1)`` in original image pixels.

    A scanner frame is measured twice by :mod:`tda.core.cache_roi_detect` -- as
    the darkest object on the board and as whatever is not the scan bed -- and
    the better answer wins.  The dark-object stage is asked first and kept
    whenever it is plausible: it is the stage the thresholds were calibrated on,
    it is tighter on the 55 machines it gets right, and preferring the
    higher-scoring box instead re-cropped most of the 66 real frames for no
    benefit.  The bed stage is the answer for a light or silver machine, where
    the dark one measures something that is not a chassis at all.

    Any other view, and a scanner frame where neither strategy convinces, gets
    the **whole frame**: not a crop at all.  It used to be the central 70 % box,
    on the theory that some crop beats none -- and on all four tape-less real
    desktops that were looked at (D46, D47, D63, D66) it cut the machine in half,
    because a chassis that covers its own tape square is exactly a chassis that
    fills the frame.  A timeline picture of the whole bench is a nuisance; one
    that hides the part being removed is a wrong answer.  The suggestion is
    always confirmed by a human anyway (spec 2.4).
    """
    if img is None or getattr(img, "size", 0) == 0:
        raise ValueError("suggest_roi() needs a non-empty image")
    height, width = img.shape[:2]
    if view == "scan":
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for found in (scan_chassis_box(bgr), scan_bed_box(bgr)):
            if found is not None:
                return found[0]
    return full_frame(width, height)


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
                "src": metrics[chosen]["path"], "metrics_version": METRICS_VERSION}
    return {"chosen": 0, "reason": SINGLE_REASON, "metrics": [],
            "src": _norm(frame.path), "metrics_version": METRICS_VERSION}


def _metrics_stale(record: Optional[dict], view: str) -> bool:
    """True when ``record`` was measured by an older :func:`burst_metrics`."""
    if not record or view != "scan":
        return False
    return int(record.get("metrics_version", 1)) != METRICS_VERSION


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
                desktops=None, progress: Optional[Callable[[int, int, str], Any]] = None,
                force: bool = False, recompute: bool = False) -> dict:
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
    recorded reason (``relabelled``) without touching the source drive.  That
    shortcut only holds while the stored metrics still mean what they say, so a
    record stamped with an older :data:`METRICS_VERSION` is measured again;
    ``recompute=True`` measures every step again regardless (copying only what
    actually changed) and ``force=True`` additionally re-copies every file.  The
    manifest is flushed every :data:`MANIFEST_FLUSH_EVERY` copies as well as at
    the end of each desktop, so an interrupted run resumes instead of redoing the
    desktop it was in.  A step that cannot be read is recorded in ``failures``
    and does not stop the run.  ``progress(desktop, step, view)`` is called once
    per step, after it is handled; an exception it raises is not caught.

    Returns counters plus the manifests: ``copied``, ``skipped``, ``rechosen``,
    ``relabelled``, ``steps``, ``bytes_copied``, ``failures``, ``non_p0`` (every
    scanner step whose cached image is *not* ``P_0``), ``flagged`` (``P_0`` kept
    although the burst failed a check - review the burst, not the choice),
    ``manifests`` and ``elapsed_s``.  Both lists cover every step seen, whether
    written now or already cached.
    """
    started = time.perf_counter()
    stats: dict[str, Any] = {
        "copied": 0, "skipped": 0, "rechosen": 0, "relabelled": 0, "steps": 0,
        "bytes_copied": 0, "failures": [], "non_p0": [], "flagged": [],
        "manifests": {}, "elapsed_s": 0.0,
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
                    measure = force or recompute or _metrics_stale(record, view)
                    if record and not measure and _up_to_date(record.get("src", ""), dest):
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
                        fresh = _select_source(frames[key], view)
                        if force or not _up_to_date(fresh["src"], dest):
                            _copy(fresh["src"], dest)
                            stats["copied"] += 1
                            stats["bytes_copied"] += os.path.getsize(dest)
                            if record and record.get("src") != fresh["src"]:
                                stats["rechosen"] += 1
                        else:  # measured again, same image: only the record changes
                            stats["relabelled"] += 1
                        manifest[step] = record = fresh
                        dirty = True
                    if record["metrics"] and record["reason"] != "p0":
                        where = ("non_p0"
                                 if record["metrics"][record["chosen"]]["p_index"] != 0
                                 else "flagged")
                        stats[where].append({
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
        found = (yaml.safe_load(fh) or {}).get("cache_dir")
    # No literal fallback: the configuration owns this path, and a default
    # pointing at one machine's real cache is how a test writes to it.
    if not found:
        raise KeyError(f"{path} defines no cache_dir")
    return str(found)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: build the cache for a range of desktops, logging progress.

    Returns 0, or 1 when any step could not be cached (the failures are listed
    and counted in the log, so a shell or scheduler notices a partial run).
    """
    import argparse

    ap = argparse.ArgumentParser(description="Build the local TDA image cache.")
    ap.add_argument("--cache", default=None, help="cache dir (default: paths.yaml cache_dir)")
    ap.add_argument("--index", default=None, help="index JSON (default: <cache>/index.json)")
    ap.add_argument("--views", default="scan", help="comma separated views")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=66)
    ap.add_argument("--log", default=None, help="progress log file (default: stdout only)")
    ap.add_argument("--recompute", action="store_true",
                    help="measure every burst again instead of reusing the stored metrics")
    ap.add_argument("--force", action="store_true",
                    help="rebuild: measure again and re-copy every file")
    add_thumb_args(ap)
    args = ap.parse_args(argv)

    cache_dir = args.cache or _default_cache_dir()
    index_path = args.index or f"{_norm(cache_dir).rstrip('/')}/index.json"
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    desktops = [d for d in range(args.first, args.last + 1)]
    if args.thumbs_only:  # the tier alone needs neither the index nor the source drive
        return run_thumb_cli(args, cache_dir, views, desktops)

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
    code = 0
    try:
        stats = build_cache(index, cache_dir, views=views, desktops=desktops, progress=on_step,
                            force=args.force, recompute=args.recompute)
        emit(f"[{time.strftime('%H:%M:%S')}] done steps={stats['steps']} copied={stats['copied']} "
             f"skipped={stats['skipped']} rechosen={stats['rechosen']} "
             f"relabelled={stats['relabelled']} GB={stats['bytes_copied'] / 1e9:.2f} "
             f"non_p0={len(stats['non_p0'])} flagged={len(stats['flagged'])} "
             f"failures={len(stats['failures'])} elapsed={stats['elapsed_s'] / 60:.1f}min")
        counts: dict[str, int] = {}
        for item in stats["non_p0"] + stats["flagged"]:
            counts[item["reason"]] = counts.get(item["reason"], 0) + 1
        emit(f"  reasons {dict(sorted(counts.items()))}")
        for item in stats["non_p0"]:
            emit(f"  non_p0 D{item['desktop']:02d} s{item['step']:03d} "
                 f"chosen={item['chosen']} reason={item['reason']} src={item['src']}")
        for item in stats["flagged"]:
            emit(f"  flagged D{item['desktop']:02d} s{item['step']:03d} "
                 f"reason={item['reason']} (P_0 kept)")
        for item in stats["failures"]:
            emit(f"  FAIL D{item['desktop']:02d} s{item['step']:03d} {item['view']}: {item['error']}")
        if stats["failures"]:
            code = 1
            emit(f"[{time.strftime('%H:%M:%S')}] {len(stats['failures'])} step(s) could not be "
                 f"cached - see the FAIL lines above")
        code = run_thumb_cli(args, cache_dir, views, desktops, emit) or code  # no-op w/o --thumbs
    finally:
        if log:
            log.close()
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
