"""The small-part detector in the window: a background pass and the rank-1 guess (task U3).

In the reverse walk the annotator adds back, on frame ``j``, the part removed
at step ``j + 1``.  For small parts the difference map's box is right about
8 % of the time; L2's detector finds the removed motherboard screw 86 % of the
time on the scanner and 62 % on OAK camera 1.  This module wires it for the
views and classes :mod:`configs/detector.yaml <tda.models.detector>` names:

1. **A background pass.**  When a view opens, :class:`DetectionWorker` runs the
   model over every frame of the view that has an image, in walk order
   starting from the frame on screen, one frame at a time with a short sleep
   between frames, and caches the answers on disk
   (:class:`tda.models.det_engine.DetCache`).  A frame change never waits for
   it: when the frame on screen is not ready it is asked for first, with the
   pixels the window already decoded.
2. **The guess** (:meth:`DetectMixin.detector_ranking`): the frame's
   detections of the classes its open ✚ rows ask for -- rows that come back
   on their own, not with a parent (:func:`detector_asked`) -- at the
   configured confidence, inside the ROI; minus those already drawn -- a same-class
   instance whose shape applies at this frame, within 2 px of the candidate's
   centre or at IoU >= 0.3 with it; ranked by native dE between the frame and
   its neighbour inside each box.  :mod:`tda.ui.app_assist` arms the first as
   rank 1 and walks the rest first under ``Shift+C``.
3. **A late answer** -- the frame was not ready when the comparison armed its
   box -- replaces that box only while the prompt is untouched: no click, no
   stroke, no refusal, no ``Shift+C``, no commit or ``Esc``, the same frame and
   the same editing instance.  Otherwise it is kept for the next visit and
   nothing moves under the annotator.

Nothing here decides anything when the detector is off, the view is not one of
its views, or no candidate survives: the difference map's path runs exactly as
before.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, Signal

from tda.core.model import FrameKey
from tda.models.det_engine import DetectionEngine, PlanItem
from tda.models.detector import (
    DetectorConfig,
    FrameDets,
    frame_crop,
    load_detector_config,
)
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import session_api as api

__all__ = ["DRAWN_CENTRE_PX", "DRAWN_IOU", "DetCandidate", "DetRanking",
           "DetectMixin", "DetectionWorker", "already_drawn", "detector_asked"]

log = logging.getLogger(__name__)

#: A candidate is "already drawn" when its centre lies within this many pixels
#: of a drawn same-class box (or that box's centre within this of it) ...
DRAWN_CENTRE_PX = 2.0
#: ... or when the two boxes overlap at least this much (L2's suppression).
DRAWN_IOU = 0.3
#: The background pass writes its cache at least every this many frames.
FLUSH_EVERY = 8


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union) if union > 0.0 else 0.0


def _holds(box, x: float, y: float, margin: float = DRAWN_CENTRE_PX) -> bool:
    return (box[0] - margin <= x < box[2] + margin
            and box[1] - margin <= y < box[3] + margin)


def detector_asked(card) -> set:
    """The classes the card's open ✚ rows ask to add back **on their own**.

    A row with a ``parent`` is a part that comes back in *with* that parent
    -- D13's four captive CPU-cooler screws on frame 12, "跟 cpu_cooler.fan.01
    一起装回来的" -- so the part removed at ``j + 1`` was the parent, not
    the screw, and the detector's premise does not hold: it armed a screw on
    the fan where the difference map had boxed the whole fan (U3 round 2).
    Such rows never ask for the detector.
    """
    return {str(r.get("cls") or str(r["instance"]).split(".", 1)[0])
            for r in card if r.get("kind") == api.KIND_ADD_SHAPE
            and not r.get("done") and r.get("instance") and not r.get("parent")}


def already_drawn(box, drawn) -> bool:
    """Is a candidate ``box`` a part the walk has already drawn on this frame?

    L2's ``suppressed``: the candidate's centre within 2 px of a drawn box, or
    IoU >= 0.3 with one -- and the other way round too (the drawn part's
    centre within 2 px of the candidate): for two boxes of one small part the
    readings agree, and for a big drawn box over a small candidate only this
    one catches it.
    """
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    for kb in drawn:
        if _holds(kb, cx, cy):
            return True
        if _holds(box, 0.5 * (kb[0] + kb[2]), 0.5 * (kb[1] + kb[3])):
            return True
        if _iou(box, kb) >= DRAWN_IOU:
            return True
    return False


@dataclass(frozen=True)
class DetCandidate:
    """One ranked detection: the box SAM would get, and why it is where it is."""

    box: tuple[float, float, float, float]
    cls: str
    conf: float
    change: float
    #: No "click here" cross: the box is the part (``Shift+C``'s alternates
    #: from the split difference map have one).
    point: Optional[tuple] = None

    @property
    def area(self) -> int:
        return int(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))

    @property
    def int_box(self) -> tuple[int, int, int, int]:
        return tuple(int(round(v)) for v in self.box)  # type: ignore[return-value]

    def describe(self) -> tuple:
        """``(box, conf, dE)`` for the log line."""
        return (self.int_box, round(self.conf, 3), round(self.change, 2))


@dataclass
class DetRanking:
    """The frame's candidates in rank order, and the ones skipped as drawn."""

    key: Any
    neighbour: Optional[int]
    chosen: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# the worker thread
# --------------------------------------------------------------------------- #
class _Bridge(QObject):
    """Carries worker-thread payloads onto the GUI thread (queued connections)."""

    sigFrame = Signal(object)
    sigState = Signal(object)
    sigIdle = Signal(object)


class DetectionWorker(QObject):
    """One thread, one frame at a time: the frame on screen first, then the walk.

    The model is built on the thread before anything else (seconds: torch,
    ultralytics, the weights, a warm-up); what it cannot do is said once on
    :attr:`sigState` and the thread ends.  Requests are a **plan** -- the
    frames of the open view in walk order -- and at most one **priority**
    frame, which is taken next.  Opening another view (or re-planning this
    one) bumps a generation; an answer from an older one is dropped on
    delivery, like :class:`tda.ui.app_diff.AssistController`'s superseded
    comparisons.

    Signals:
        sigFrame: a :class:`~tda.models.detector.FrameDets` of the current
            generation, on the GUI thread.
        sigState: ``("on", description)`` once the model is loaded, or
            ``("off", reason)``.
        sigIdle: ``{"view", "frames", "detected", "measured", "cached",
            "seconds"}`` when the plan has run out.
    """

    sigFrame = Signal(object)
    sigState = Signal(object)
    sigIdle = Signal(object)

    def __init__(self, engine: DetectionEngine, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._lock = threading.Condition()
        self._gen = 0
        self._view: Optional[tuple[int, str]] = None
        self._plan: "deque[PlanItem]" = deque()
        self._priority: Optional[PlanItem] = None
        self._busy = False
        self._stopped = False
        #: ``None`` while the model loads, then whether it did.
        self._loaded: Optional[bool] = None
        self._stats: dict = {}
        self._bridge = _Bridge(self)
        conn = Qt.ConnectionType.QueuedConnection
        self._bridge.sigFrame.connect(self._deliver_frame, conn)
        self._bridge.sigState.connect(self.sigState, conn)
        self._bridge.sigIdle.connect(self._deliver_idle, conn)
        self._thread = threading.Thread(target=self._run, name="tda-detect", daemon=True)
        self._thread.start()

    # -- requests (GUI thread) ------------------------------------------------
    @property
    def generation(self) -> int:
        return self._gen

    def open_view(self, desktop: int, view: str, items) -> int:
        """Serve this view from now on, in the order of ``items``; the new generation."""
        with self._lock:
            self._gen += 1
            self._view = (int(desktop), str(view))
            self._plan = deque(items)
            self._priority = None
            self._stats = {"view": self._view, "frames": 0, "detected": 0,
                           "measured": 0, "cached": 0, "started": time.perf_counter()}
            self._lock.notify_all()
            return self._gen

    def prioritise(self, item: PlanItem) -> None:
        """Take ``item`` next, and walk on from it: the annotator is there now."""
        with self._lock:
            if self._stopped or self._view is None:
                return
            self._priority = item
            rest = [p for p in self._plan if p.step != item.step]
            below = sorted((p for p in rest if p.step < item.step),
                           key=lambda p: -p.step)
            above = sorted((p for p in rest if p.step > item.step), key=lambda p: p.step)
            self._plan = deque(below + above)
            self._lock.notify_all()

    def pending(self) -> bool:
        """Is anything still to do (or being done, or the model still loading)?"""
        with self._lock:
            return (self._loaded is None or self._busy or self._priority is not None
                    or (bool(self._plan) and self._loaded is True))

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until the model has loaded and the plan has run out; ``True`` if so."""
        deadline = time.perf_counter() + float(timeout)
        with self._lock:
            while (self._loaded is None or self._busy or self._priority is not None
                   or (self._plan and self._loaded is True and not self._stopped)):
                left = deadline - time.perf_counter()
                if left <= 0:
                    return False
                self._lock.wait(left)
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        """Refuse new work, let the frame in hand finish, write the cache, join."""
        with self._lock:
            self._stopped = True
            self._plan.clear()
            self._priority = None
            loading = self._loaded is None
            self._lock.notify_all()
        if loading:
            # Importing torch and ultralytics cannot be interrupted, and there
            # is nothing to write yet: the thread ends on its own once the
            # load returns, and closing the window does not wait seconds for it.
            return
        if self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    # -- delivery (GUI thread) ----------------------------------------------
    def _deliver_frame(self, stamped: object) -> None:
        gen, frame = stamped  # type: ignore[misc]
        if gen != self._gen:
            return      # the view (or its plan) moved on while it was computed
        self.sigFrame.emit(frame)

    def _deliver_idle(self, stamped: object) -> None:
        gen, stats = stamped  # type: ignore[misc]
        if gen == self._gen:
            self.sigIdle.emit(stats)

    # -- the thread -----------------------------------------------------------
    def _emit(self, signal: str, payload: object) -> None:
        """Post to the GUI thread; a window already gone is nobody to tell."""
        try:
            getattr(self._bridge, signal).emit(payload)
        except RuntimeError:        # the Qt object was deleted under us
            pass

    def _run(self) -> None:
        try:
            text = self.engine.load()
        except BaseException as exc:  # noqa: BLE001 - off, never a crash
            with self._lock:
                self._loaded = False
                stopped = self._stopped
                self._lock.notify_all()
            if not stopped:
                self._emit("sigState", ("off", f"{type(exc).__name__}: {exc}"))
            return
        with self._lock:
            self._loaded = True
            stopped = self._stopped
            self._lock.notify_all()
        if stopped:
            return
        self._emit("sigState", ("on", text))
        opened: Optional[tuple[int, str]] = None
        since_flush = 0
        while True:
            with self._lock:
                while not self._stopped and self._priority is None and not self._plan:
                    self._lock.wait()
                if self._stopped:
                    break
                if self._priority is not None:
                    item, self._priority = self._priority, None
                else:
                    item = self._plan.popleft()
                gen, view, stats = self._gen, self._view, self._stats
                # the pass is timed from its first frame, not from the request:
                # the first view waits for the model to load
                stats.setdefault("first", time.perf_counter())
                self._busy = True
            computed = False
            try:
                if view != opened and view is not None:
                    self.engine.open_view(*view)
                    opened = view
                frame = self.engine.process(item)
                if frame is not None:
                    computed = frame.detected or frame.measured
                    stats["frames"] += 1
                    stats["detected"] += int(frame.detected)
                    stats["measured"] += int(frame.measured)
                    stats["cached"] += int(not frame.detected)
                    self._emit("sigFrame", (gen, frame))
            except Exception as exc:  # noqa: BLE001 - one frame is not worth the pass
                log.warning("detector: step %s failed: %s: %s", item.step,
                            type(exc).__name__, exc)
            with self._lock:
                self._busy = False
                idle = self._priority is None and not self._plan
                self._lock.notify_all()
            since_flush += int(computed)
            if since_flush and (idle or since_flush >= FLUSH_EVERY):
                self.engine.flush()
                since_flush = 0
            if idle:
                now = time.perf_counter()
                done = dict(stats, seconds=now - stats.get("first", now))
                done.pop("first", None)
                done.pop("started", None)
                self._emit("sigIdle", (gen, done))
            elif computed and self.engine.config.yield_ms > 0:
                # One frame at a time, and room between frames for a SAM click.
                time.sleep(self.engine.config.yield_ms / 1000.0)
        self.engine.flush()


# --------------------------------------------------------------------------- #
# the window's half
# --------------------------------------------------------------------------- #
class DetectMixin:
    """Plans the background pass, keeps its answers, and ranks the frame's candidates."""

    def _init_detect(self) -> None:
        self.det_config: Optional[DetectorConfig] = None
        self.det_worker: Optional[DetectionWorker] = None
        #: ``"loading"``, ``"on"`` or ``"off"``.
        self.det_state = "off"
        self._det_view: Optional[tuple[int, str]] = None
        #: The answers of the open view: step -> FrameDets.
        self._det_frames: dict[int, FrameDets] = {}
        #: What each step of the open view was planned with (its crop).
        self._det_plan_crops: dict[int, Optional[tuple]] = {}
        self._det_rows: dict[int, dict] = {}
        #: Everything that makes a late answer "too late": presses on the
        #: canvas, Shift+C, a refusal, a commit / Esc / undo.
        self._det_touches = 0
        #: ``(key, editing instance, touches)`` when the frame was arrived on.
        self._det_visit: Optional[tuple] = None
        #: The frame on screen when it was not ready on arrival.
        self._det_late: Optional[Any] = None
        #: The last idle report, for probes and the log.
        self.det_last_pass: Optional[dict] = None
        #: Set while a late answer re-arms rank 1, for the log line.
        self._det_arming_late = False
        self.canvas.sigMousePress.connect(self._det_on_press)
        config = load_detector_config()
        if config is not None:
            self.enable_detector(config)

    def enable_detector(self, config: DetectorConfig,
                        factory: Optional[Callable[[DetectorConfig], Any]] = None,
                        cache_root: Optional[str] = None,
                        change_fn: Optional[Callable] = None) -> None:
        """Start a worker for ``config`` (a test passes a stub model ``factory``)."""
        self.shutdown_detector()
        root = cache_root or (self.paths or {}).get("det_cache_dir") or config.cache_root
        kwargs = {} if change_fn is None else {"change_fn": change_fn}
        engine = DetectionEngine(config, Path(root), factory=factory, **kwargs)
        worker = DetectionWorker(engine, parent=self)
        worker.sigFrame.connect(self._on_det_frame)
        worker.sigState.connect(self._on_det_state)
        worker.sigIdle.connect(self._on_det_idle)
        self.det_config, self.det_worker, self.det_state = config, worker, "loading"
        self._det_view = None
        self._det_frames = {}
        self._det_plan_crops = {}
        if compat.is_open(self.session):
            self.on_frame_changed_detect(self.session.current())

    def shutdown_detector(self) -> None:
        worker, self.det_worker = self.det_worker, None
        if worker is not None:
            worker.shutdown()
        self.det_state = "off"
        self._det_frames = {}

    # -------------------------------------------------------------- signals
    @S.guard
    def _on_det_state(self, state: object) -> None:
        kind, text = state  # type: ignore[misc]
        config = self.det_config
        if kind == "on" and config is not None:
            self.det_state = "on"
            log.info("small-part detector on for %s, classes %s at conf >= %g; %s; "
                     "config %s, cache %s", list(config.views), list(config.classes),
                     config.conf, text, config.source,
                     self.det_worker.engine.cache_root if self.det_worker else "")
            return
        self.det_state = "off"
        self._det_frames = {}
        log.warning("small-part detector OFF, the guess is the difference map's: %s", text)

    @S.guard
    def _on_det_idle(self, stats: object) -> None:
        self.det_last_pass = dict(stats)  # type: ignore[arg-type]
        view = stats.get("view") or ("?", "?")  # type: ignore[union-attr]
        log.info("detector pass D%s/%s: %d frames in %.1f s (%d detected, %d from the "
                 "cache; dE measured on %d)", view[0], view[1], stats["frames"],
                 stats["seconds"], stats["detected"], stats["cached"], stats["measured"])

    def _det_on_press(self, *_args) -> None:
        self._det_touches += 1

    def note_prompt_touched(self) -> None:
        """The annotator did something with the prompt: a late answer must not move it."""
        self._det_touches += 1

    # ------------------------------------------------------------ the plan
    def on_frame_changed_detect(self, key) -> None:
        """Arriving on a frame: plan the view if it is new, ask for this frame if not ready.

        Costs a dictionary lookup and the ROI read when the answer is on hand
        -- which, after the first pass over a view, it always is.
        """
        worker, config = self.det_worker, self.det_config
        if worker is None or config is None or self.det_state == "off":
            return
        if not compat.is_open(self.session):
            return
        if self._det_visit is None or self._det_visit[0] != key:
            self._det_visit = (key, getattr(self.session, "editing_instance", None),
                               self._det_touches)
            self._det_late = None
        desktop, view = int(self.session.desktop), str(self.session.view)
        if not config.applies(view):
            if self._det_view != (desktop, view):
                self._det_view = (desktop, view)
                self._det_frames = {}
                self._det_plan_crops = {}
                worker.open_view(desktop, view, [])
            return
        roi = self.roi()
        crop = frame_crop(config, roi)
        if (self._det_view != (desktop, view)
                or self._det_plan_crops.get(int(key.step), False) != crop):
            self._det_replan(int(key.step))
        if config.roi_crop and roi is None:
            return          # no ROI, no box: nothing to ask for yet
        if self.session.image() is None or self._det_ready(int(key.step), crop):
            return
        self._det_late = key
        worker.prioritise(self._det_item(int(key.step), crop, current=True))

    def _det_replan(self, current: int) -> None:
        """Hand the worker the whole view, in walk order from ``current``."""
        config = self.det_config
        desktop, view = int(self.session.desktop), str(self.session.view)
        rows = {int(r["step"]): r for r in self.db.frames_for(desktop, view)}
        rois = self._det_rois(desktop, view, rows)
        steps = sorted(int(s) for s in self.session.available_steps())
        order = ([s for s in reversed(steps) if s <= current]
                 + [s for s in steps if s > current])
        # Every step with a frame row, not only the ones with an image: a
        # missing frame visited on purpose must not look unplanned.
        crops = {s: frame_crop(config, rois.get(s)) for s in rows}
        if self._det_view != (desktop, view):
            self._det_frames = {}
        self._det_view = (desktop, view)
        self._det_rows = rows
        self._det_plan_crops = crops
        items = [self._det_item(s, crops[s], steps=steps) for s in order
                 if not (config.roi_crop and crops[s] is None)]
        self.det_worker.open_view(desktop, view, items)

    def _det_rois(self, desktop: int, view: str, rows: dict) -> dict:
        """Each step's stored ROI, as :meth:`tda.core.db.Db.pose_segment_for` finds it.

        Two queries for the view instead of two per step: the frame's own
        segment wins, else the first segment whose range holds the step.
        """
        segments = self.db.pose_segments(desktop, view)
        by_seg = {int(s["seg"]): s for s in segments}
        out: dict[int, Optional[tuple]] = {}
        for step, row in rows.items():
            seg = None
            own = row.get("pose_segment")
            if own is not None:
                seg = by_seg.get(int(own))
            if seg is None:
                seg = next((s for s in segments
                            if s.get("start_step") is not None and s.get("end_step") is not None
                            and int(s["start_step"]) <= step <= int(s["end_step"])), None)
            roi = (seg or {}).get("roi")
            out[step] = tuple(int(v) for v in roi) if roi else None
        return out

    def _det_item(self, step: int, crop: Optional[tuple], steps=None,
                  current: bool = False) -> PlanItem:
        """One frame as the worker wants it: where its pixels and its neighbour's are."""
        from tda.ui.session_images import frame_candidates

        desktop, view = int(self.session.desktop), str(self.session.view)
        cache_dir = getattr(self.session, "cache_dir", "")

        def where(s: Optional[int]) -> tuple:
            if s is None:
                return ()
            row = self._det_rows.get(int(s))
            return tuple(frame_candidates(cache_dir, FrameKey(desktop, int(s), view), row))

        if current:
            neighbour = compat.task_neighbour(self.session)
            return PlanItem(step=step, candidates=where(step), crop=crop,
                            neighbour=neighbour, neighbour_candidates=where(neighbour),
                            pixels=self.session.image(),
                            neighbour_pixels=(None if neighbour is None else
                                              compat.peek_image_at(self.session, neighbour)),
                            priority=True)
        later = [s for s in (steps or []) if s > step]
        neighbour = later[0] if later else None
        return PlanItem(step=step, candidates=where(step), crop=crop, neighbour=neighbour,
                        neighbour_candidates=where(neighbour))

    def _det_ready(self, step: int, crop: Optional[tuple]) -> bool:
        """Is the answer for ``step`` on hand, for this crop and this neighbour?"""
        frame = self._det_frames.get(int(step))
        return (frame is not None and frame.crop == crop
                and frame.neighbour == compat.task_neighbour(self.session))

    # --------------------------------------------------------- the answers
    @S.guard
    def _on_det_frame(self, frame: object) -> None:
        """An answer landed; if it is the late one for the frame on screen, maybe arm it."""
        if not isinstance(frame, FrameDets) or self.det_state == "off":
            return
        self._det_frames[int(frame.step)] = frame
        if not compat.is_open(self.session):
            return
        key = self.session.current()
        if int(frame.step) != int(key.step) or self._det_late != key:
            return
        crop = frame_crop(self.det_config, self.roi())
        if not self._det_ready(int(key.step), crop):
            return
        self._det_late = None
        payload = self.assist_result
        if not payload or payload.get("key") != key:
            # The comparison has not landed: its arrival arms rank 1, and will
            # find this answer waiting.
            return
        if not self._det_untouched():
            log.info("late detector answer for step %s kept, not armed: the prompt was "
                     "touched since the frame was reached", key.step)
            return
        ranking = self.detector_ranking()
        if ranking is None or not ranking.chosen:
            return
        self._det_arming_late = True
        try:
            self._arm_from_card()
        finally:
            self._det_arming_late = False

    def _det_untouched(self) -> bool:
        """Nothing has happened to the prompt since the frame was reached."""
        visit = self._det_visit
        if visit is None or not compat.is_open(self.session):
            return False
        key, instance, touches = visit
        if self.session.current() != key:
            return False
        if getattr(self.session, "editing_instance", None) != instance:
            return False
        if touches != self._det_touches or self._prompt_rank != 0:
            return False
        if self._box_refused_here():
            return False
        if self.sam_point.points or self.sam_box.box is not None:
            return False
        return self._candidate_tool() is None

    # ------------------------------------------------------------ ranking
    def detector_ranking(self, rows: Optional[list] = None) -> Optional[DetRanking]:
        """The open frame's detector candidates, best first; ``None`` when it does not apply.

        ``None`` -- the difference map decides, exactly as before -- when the
        detector is off or not for this view, the frame has no ROI or no answer
        yet, or none of the card's open ✚ rows is a class the detector is for
        and comes back on its own (:func:`detector_asked`: a row with a
        ``parent`` does not count).  ``rows`` is the task card the caller
        already read.
        """
        config = self.det_config
        if (self.det_worker is None or config is None or not self._det_frames
                or not compat.is_open(self.session)):
            return None
        key = self.session.current()
        if not config.applies(key.view):
            return None
        roi = self.roi()
        if roi is None:
            return None
        crop = frame_crop(config, roi)
        if not self._det_ready(int(key.step), crop):
            return None
        card = self.session.task_card() if rows is None else rows
        asked = detector_asked(card)
        wanted = [c for c in config.classes if c in asked]
        if not wanted:
            return None
        frame = self._det_frames[int(key.step)]
        drawn = self._det_drawn(set(wanted))
        ranking = DetRanking(key=key, neighbour=frame.neighbour)
        for index, det in enumerate(frame.dets):
            if det.cls not in wanted or det.conf < config.conf or index not in frame.change:
                continue
            cx, cy = det.centre
            if not (roi[0] <= cx < roi[2] and roi[1] <= cy < roi[3]):
                continue
            candidate = DetCandidate(box=tuple(float(v) for v in det.int_box),  # type: ignore[arg-type]
                                     cls=det.cls, conf=float(det.conf),
                                     change=float(frame.change[index]))
            if already_drawn(candidate.box, drawn.get(det.cls, ())):
                ranking.skipped.append(candidate)
            else:
                ranking.chosen.append(candidate)
        # Stable: equal dE keeps the detector's own order, highest confidence first.
        ranking.chosen.sort(key=lambda c: -c.change)
        return ranking

    def _det_drawn(self, classes: set) -> dict:
        """``{class: [box]}`` of the instances of ``classes`` with a shape on this frame."""
        from tda.core.truth_inputs import instances_of

        try:
            records = instances_of(self.db, int(self.session.desktop))
        except Exception:  # noqa: BLE001 - the keys still say the class
            records = {}
        out: dict[str, list] = {}
        for name, inst in self.session.compiled().instances.items():
            rec = records.get(name)
            cls = str(getattr(rec, "cls", "") or str(name).split(".", 1)[0])
            if cls in classes and inst.box is not None:
                out.setdefault(cls, []).append(tuple(float(v) for v in inst.box))
        return out
