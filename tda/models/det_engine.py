"""One frame's detections and candidate scores, from the disk cache or the model.

Qt-free; the worker thread in :mod:`tda.ui.app_detect` owns one
:class:`DetectionEngine` and calls it one frame at a time.

**The cache** (:class:`DetCache`) is one JSON file per desktop and view under
``<cache_root>/<model sha1[:12]>/<desktop>_<view>.json``.  Each step's entry
is keyed by the file the frame's pixels were read from -- its path, size and
modification time -- and by what it was detected on (the ROI); a change to
any of them is a miss.  A different model is a different directory, so it can
never be served another model's detections; a different recipe (tiling,
thresholds) empties the file it would have read.

The candidates' dE is cached in the same entry, one slot per neighbour step
(``changes``), each keyed the same way by the neighbour's file: ``Tab``
compares with the neighbour the task card is written against, and so does
this -- in the reverse walk the step above, browsing forward the step below.

Writes are atomic (a temporary file, then :func:`os.replace`): a window closed
mid-write leaves the previous file, never half of one.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

from tda.models.detector import (
    Det,
    DetectorConfig,
    FrameDets,
    box_change,
    file_sha1,
    make_detector,
    whole,
)

__all__ = ["CACHE_VERSION", "DetCache", "DetectionEngine", "PlanItem", "file_stamp",
           "first_readable"]

log = logging.getLogger(__name__)

#: 2 (round 3): the candidates' dE is kept per neighbour step (``changes``).
CACHE_VERSION = 2
#: Decoded frames the engine keeps: ``j`` and its neighbour, and the one the
#: walk steps onto next is always one of the two.
DECODED_KEPT = 2


def first_readable(candidates: Sequence[Optional[str]]) -> Optional[str]:
    """The first of these paths that can be opened, like the window's image cache."""
    for candidate in candidates:
        if not candidate:
            continue
        try:
            with open(str(candidate), "rb"):
                return str(candidate)
        except OSError:
            continue
    return None


def file_stamp(path: Optional[str]) -> Optional[dict]:
    """``{"path", "size", "mtime_ns"}`` of a file, or ``None`` when it is not there."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"path": str(path).replace("\\", "/"), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns)}


def _same_file(entry: Optional[dict], stamp: Optional[dict]) -> bool:
    return (entry is not None and stamp is not None
            and entry.get("path") == stamp["path"]
            and entry.get("size") == stamp["size"]
            and entry.get("mtime_ns") == stamp["mtime_ns"])


@dataclass
class PlanItem:
    """One frame the worker is asked about.

    ``candidates`` are where its pixels may be (the first readable wins, the
    window's own rule); ``pixels`` are those pixels when the window already
    has them decoded -- the frame on screen -- so the worker need not read the
    file again.  The same pair describes the neighbour.
    """

    step: int
    candidates: tuple
    crop: Optional[tuple]
    neighbour: Optional[int] = None
    neighbour_candidates: tuple = ()
    pixels: Optional[np.ndarray] = None
    neighbour_pixels: Optional[np.ndarray] = None
    priority: bool = False


class DetCache:
    """The detections of one desktop/view for one model, on disk."""

    def __init__(self, root: Path, model_key: str, recipe: dict,
                 model_path: str = "") -> None:
        self.dir = Path(root) / str(model_key)[:12]
        self.model_key = str(model_key)
        self.model_path = str(model_path)
        self.recipe = dict(recipe)
        self.desktop: Optional[int] = None
        self.view: Optional[str] = None
        self.steps: dict[str, dict] = {}
        self.dirty = False
        #: Why the file on disk was not used, or "" (for the log).
        self.discarded = ""

    def path_for(self, desktop: int, view: str) -> Path:
        return self.dir / f"{int(desktop)}_{view}.json"

    def open(self, desktop: int, view: str) -> None:
        """Load the file of one desktop/view; anything unusable starts empty."""
        self.desktop, self.view = int(desktop), str(view)
        self.steps, self.dirty, self.discarded = {}, False, ""
        path = self.path_for(desktop, view)
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.discarded = f"unreadable ({type(exc).__name__})"
            return
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            self.discarded = "another cache version"
            return
        if data.get("model_sha1") != self.model_key:
            self.discarded = "another model"
            return
        if data.get("recipe") != json.loads(json.dumps(self.recipe)):
            self.discarded = "another recipe"
            return
        steps = data.get("steps")
        if isinstance(steps, dict):
            self.steps = {str(k): v for k, v in steps.items() if isinstance(v, dict)}

    def get(self, step: int) -> Optional[dict]:
        return self.steps.get(str(int(step)))

    def put(self, step: int, entry: dict) -> None:
        self.steps[str(int(step))] = entry
        self.dirty = True

    def flush(self) -> Optional[Path]:
        """Write the file if anything changed; the path written, or ``None``."""
        if not self.dirty or self.desktop is None or self.view is None:
            return None
        path = self.path_for(self.desktop, self.view)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"version": CACHE_VERSION, "model_sha1": self.model_key,
                "model": self.model_path, "desktop": self.desktop, "view": self.view,
                "recipe": self.recipe,
                "steps": {k: self.steps[k] for k in sorted(self.steps, key=int)}}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
        self.dirty = False
        return path


class DetectionEngine:
    """Detections and candidate dE for one frame at a time; the worker's half.

    ``factory`` builds the model from the configuration -- the backend's
    (:func:`~tda.models.detector.make_detector`: a ``YoloTileDetector`` or an
    ``RFDetrTileDetector``) unless a test passes a stub.  A model has
    ``identity`` (the cache key), ``names`` and ``detect(img_rgb, crop, view,
    step)``; ``describe()`` is optional.
    """

    def __init__(self, config: DetectorConfig, cache_root: Optional[Path] = None,
                 factory: Optional[Callable[[DetectorConfig], Any]] = None,
                 change_fn: Callable = box_change, identity: Optional[str] = None) -> None:
        self.config = config
        self.cache_root = Path(cache_root) if cache_root else Path(config.cache_root)
        self._real_model = factory is None
        self.factory = factory or make_detector
        self.change_fn = change_fn
        #: The cache key; the model file's SHA-1 unless a stub names one.
        self.identity: Optional[str] = identity
        self.model: Any = None
        self.cache: Optional[DetCache] = None
        self.view: Optional[tuple[int, str]] = None
        #: Decoded frames by ``(path, size, mtime_ns)``: a file rewritten under
        #: the same name is a new picture.
        self._decoded: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

    # -- lifecycle ----------------------------------------------------------
    def prepare(self) -> None:
        """Open the cache -- no model, no torch: its key is the model file's SHA-1.

        Round 3: a warm start is served from the disk cache while the model is
        still loading (seconds), so the first frame gets the detector's box
        within the frame-change budget.  A factory that is not the real model
        (a test's stub) either names its ``identity`` up front or is built here.
        """
        identity = self.identity
        if identity is None:
            if self._real_model:
                identity = file_sha1(self.config.model)
            else:
                self.model = self.factory(self.config)
                identity = str(self.model.identity)
        self.identity = str(identity)
        self.cache = DetCache(self.cache_root, self.identity, self.config.recipe(),
                              str(self.config.model))

    def load_model(self) -> str:
        """Build the model; raises when it cannot run.  Returns what it is."""
        if self.model is None:
            model = self.factory(self.config)
            if self.identity is not None and str(model.identity) != self.identity:
                raise RuntimeError(f"the model changed while it loaded: {model.identity} "
                                   f"is not {self.identity}")
            self.model = model
        describe = getattr(self.model, "describe", None)
        return describe() if callable(describe) else f"model {self.model.identity}"

    def load(self) -> str:
        """:meth:`prepare` and :meth:`load_model` in one go."""
        self.prepare()
        return self.load_model()

    def release(self) -> None:
        """A pass is over: drop the decoded frames and the model's GPU cache."""
        self._decoded.clear()
        release = getattr(self.model, "release", None)
        if callable(release):
            release()

    def open_view(self, desktop: int, view: str) -> None:
        """Serve one desktop/view from here on: its cache file is read now."""
        self.flush()
        assert self.cache is not None, "prepare() first"
        self.cache.open(desktop, view)
        if self.cache.discarded:
            log.info("detector cache %s ignored: %s", self.cache.path_for(desktop, view),
                     self.cache.discarded)
        self.view = (int(desktop), str(view))
        self._decoded.clear()

    def flush(self) -> Optional[Path]:
        if self.cache is None:
            return None
        try:
            return self.cache.flush()
        except OSError as exc:
            log.warning("the detector cache could not be written: %s", exc)
            return None

    # -- pixels -------------------------------------------------------------
    def _read(self, path: Optional[str], given: Optional[np.ndarray] = None
              ) -> Optional[np.ndarray]:
        if given is not None:
            return given
        stamp = file_stamp(path)
        if stamp is None:
            return None
        key = (stamp["path"], stamp["size"], stamp["mtime_ns"])
        hit = self._decoded.get(key)
        if hit is not None:
            self._decoded.move_to_end(key)
            return hit
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._decoded[key] = rgb
        while len(self._decoded) > DECODED_KEPT:
            self._decoded.popitem(last=False)
        return rgb

    def candidates_of(self, dets: Sequence[Det]) -> list[int]:
        """Indices of the detections that may arm a box: class and confidence."""
        wanted = set(self.config.classes)
        return [i for i, d in enumerate(dets)
                if d.cls in wanted and d.conf >= self.config.conf]

    # -- one frame ----------------------------------------------------------
    def answer_from_cache(self, item: PlanItem) -> Optional[FrameDets]:
        """The frame's answer if the disk cache holds all of it, else ``None``.

        No model and no pixels: a stat of the frame's file and of its
        neighbour's, and a lookup.  What the worker serves while the model is
        still loading (round 3).
        """
        assert self.view is not None and self.cache is not None, "open_view() first"
        started = time.perf_counter()
        stamp = file_stamp(first_readable(item.candidates))
        crop = None if item.crop is None else [int(v) for v in item.crop]
        entry = self.cache.get(item.step)
        if not (_same_file(entry, stamp) and entry.get("crop") == crop):  # type: ignore[union-attr]
            return None
        dets = [Det.from_row(r) for r in entry.get("dets") or []]  # type: ignore[union-attr]
        wanted = self.candidates_of(dets)
        change: dict = {}
        if item.neighbour is not None and wanted:
            nstamp = file_stamp(first_readable(item.neighbour_candidates))
            change = self._cached_change(entry, item, wanted, nstamp)
            if change is None:
                return None
        return FrameDets(step=int(item.step), crop=None if crop is None else tuple(crop),
                         dets=tuple(dets), neighbour=item.neighbour, change=change,
                         detected=False, measured=False,
                         seconds=time.perf_counter() - started)

    def process(self, item: PlanItem) -> Optional[FrameDets]:
        """The frame's detections and its candidates' dE; ``None`` without pixels."""
        assert self.view is not None and self.cache is not None, "open_view() first"
        started = time.perf_counter()
        view = self.view[1]
        path = first_readable(item.candidates)
        stamp = file_stamp(path)
        crop = None if item.crop is None else [int(v) for v in item.crop]
        entry = self.cache.get(item.step)
        img: Optional[np.ndarray] = None
        detected = False
        if _same_file(entry, stamp) and entry.get("crop") == crop:  # type: ignore[union-attr]
            dets = [Det.from_row(r) for r in entry.get("dets") or []]  # type: ignore[union-attr]
        else:
            img = self._read(path, item.pixels)
            if img is None:
                return None
            box = whole(img, item.crop)
            found = self.model.detect(img, box, view, item.step)
            dets = [d for d in found if d.conf >= self.config.store_conf]
            detected = True
            entry = None
            if stamp is not None:
                entry = dict(stamp, crop=crop, dets=[d.row() for d in dets])
                self.cache.put(item.step, entry)
            # what is stored is what is served: rounded like the file
            dets = [Det.from_row(d.row()) for d in dets]
        change, measured = self._change(item, entry, dets, img, path)
        return FrameDets(step=int(item.step), crop=None if crop is None else tuple(crop),
                         dets=tuple(dets), neighbour=item.neighbour, change=change,
                         detected=detected, measured=measured,
                         seconds=time.perf_counter() - started)

    def _cached_change(self, entry: Optional[dict], item: PlanItem, wanted: list[int],
                       nstamp: Optional[dict]) -> Optional[dict]:
        """``{candidate index: dE}`` from the cache for this neighbour, or ``None``.

        One slot per neighbour step (round 3): browsing forward compares a
        frame with the step *below* it, and a single slot made the two
        directions overwrite each other on every visit.
        """
        slots = (entry or {}).get("changes")
        cached = slots.get(str(int(item.neighbour))) if isinstance(slots, dict) else None
        if (isinstance(cached, dict) and _same_file(cached, nstamp)
                and cached.get("classes") == list(self.config.classes)
                and cached.get("conf") == float(self.config.conf)
                and all(str(i) in (cached.get("values") or {}) for i in wanted)):
            values = cached["values"]
            return {i: float(values[str(i)]) for i in wanted}
        return None

    def _change(self, item: PlanItem, entry: Optional[dict], dets: list[Det],
                img: Optional[np.ndarray], path: Optional[str]) -> tuple[dict, bool]:
        """``{candidate index: dE}`` against the neighbour, cached like the detections."""
        wanted = self.candidates_of(dets)
        if item.neighbour is None or not wanted:
            return {}, False
        npath = first_readable(item.neighbour_candidates)
        nstamp = file_stamp(npath)
        cached = self._cached_change(entry, item, wanted, nstamp)
        if cached is not None:
            return cached, False
        img = img if img is not None else self._read(path, item.pixels)
        other = self._read(npath, item.neighbour_pixels)
        if img is None or other is None or img.shape != other.shape:
            return {}, False
        change: dict[int, float] = {}
        for i in wanted:
            try:
                change[i] = round(float(self.change_fn(img, other, dets[i].int_box)), 3)
            except Exception as exc:  # noqa: BLE001 - one box is not worth the frame
                log.warning("dE of %s on step %s failed: %s", dets[i].int_box, item.step, exc)
        if entry is not None and nstamp is not None:
            slots = entry.get("changes")
            if not isinstance(slots, dict):
                slots = entry["changes"] = {}
            slots[str(int(item.neighbour))] = dict(
                nstamp, classes=list(self.config.classes), conf=float(self.config.conf),
                values={str(i): v for i, v in change.items()})
            self.cache.put(item.step, entry)  # type: ignore[union-attr]
        return change, True
